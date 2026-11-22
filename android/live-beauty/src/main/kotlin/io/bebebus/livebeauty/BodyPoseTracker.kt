package io.bebebus.livebeauty

import android.content.Context
import android.os.SystemClock
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 美体用的人体姿态识别（MediaPipe Pose Landmarker lite，CPU）；与人脸跟踪各用一个 worker、各自限频，
 * 互不阻塞。onFailure 在 worker 线程回调，只表示美体故障（模型加载或推理失败），不影响美颜与人脸塑形。
 */
class BodyPoseTracker(context: Context, private val onFailure: () -> Unit) {
    private val appContext = context.applicationContext
    val state = BodyTrackingState()
    private val closed = AtomicBoolean(false)
    private val worker = Executors.newSingleThreadExecutor()
    private var detector: PoseLandmarker? = null // 仅 worker 访问。

    /** GL 线程调用；[sample] 的缓冲区在 complete 之前归 worker 使用。 */
    internal fun submit(sample: FaceTextureSampler.Sample, ticket: BodyTrackingState.Ticket) {
        if (closed.get()) { state.complete(ticket, null, SystemClock.uptimeMillis()); return }
        try { worker.execute { run(sample, ticket) } }
        catch (_: RejectedExecutionException) { state.complete(ticket, null, SystemClock.uptimeMillis()) }
    }

    private fun run(sample: FaceTextureSampler.Sample, ticket: BodyTrackingState.Ticket) {
        try {
            if (closed.get()) return
            val task = detector ?: PoseLandmarker.createFromOptions(appContext,
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL).build())
                    .setRunningMode(RunningMode.VIDEO).setNumPoses(1)
                    .setMinPoseDetectionConfidence(0.5f).setMinPosePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f).build()).also { detector = it }
            val image = ByteBufferImageBuilder(sample.pixels, sample.width, sample.height, MPImage.IMAGE_FORMAT_RGBA).build()
            val geometry = try {
                val poses = task.detectForVideo(image, ticket.timestamp).landmarks()
                poses.firstOrNull()?.let { pose ->
                    BodyGeometry.fromPose(pose.map {
                        BodyGeometry.Landmark(it.x(), it.y(), it.visibility().orElse(0f))
                    }, ticket.width.toFloat() / ticket.height)
                }
            } finally { image.close() }
            state.complete(ticket, geometry, SystemClock.uptimeMillis())
        } catch (_: Exception) {
            // 本次预览内不重试，由调用方停用美体。
            state.complete(ticket, null, SystemClock.uptimeMillis())
            state.enable(false)
            if (!closed.get()) onFailure()
        }
    }

    fun close() {
        if (!closed.compareAndSet(false, true)) return
        state.enable(false); state.reset()
        worker.execute { detector?.close(); detector = null }
        worker.shutdown()
    }

    companion object {
        /** 随安卓包分发（android/src/main/assets），SHA256 由 scripts/app-face-model-check.py 校验。 */
        const val MODEL = "pose_landmarker_lite.task"
    }
}
