package io.bebebus.livebeauty

import android.content.Context
import android.os.SystemClock
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 背景虚化用的人像分割（MediaPipe Selfie Segmenter，256×256，CPU）；与人脸、人体各用一个 worker、各自限频，互不阻塞。
 * onFailure 在 worker 线程回调，只表示背景虚化故障（模型加载或推理失败），不影响其他效果。
 */
class PersonSegmenter(context: Context, private val onFailure: () -> Unit) {
    private val appContext = context.applicationContext
    val state = MaskTrackingState()
    private val closed = AtomicBoolean(false)
    private val worker = Executors.newSingleThreadExecutor()
    private var segmenter: ImageSegmenter? = null // 仅 worker 访问。

    /** GL 线程调用；[sample] 的缓冲区在 complete 之前归 worker 使用。 */
    internal fun submit(sample: FaceTextureSampler.Sample, ticket: MaskTrackingState.Ticket) {
        if (closed.get()) { state.complete(ticket, null, SystemClock.uptimeMillis()); return }
        try { worker.execute { run(sample, ticket) } }
        catch (_: RejectedExecutionException) { state.complete(ticket, null, SystemClock.uptimeMillis()) }
    }

    private fun run(sample: FaceTextureSampler.Sample, ticket: MaskTrackingState.Ticket) {
        try {
            if (closed.get()) return
            val task = segmenter ?: ImageSegmenter.createFromOptions(appContext,
                ImageSegmenter.ImageSegmenterOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL).build())
                    .setRunningMode(RunningMode.VIDEO)
                    .setOutputConfidenceMasks(true).setOutputCategoryMask(false).build()).also { segmenter = it }
            val image = ByteBufferImageBuilder(sample.pixels, sample.width, sample.height, MPImage.IMAGE_FORMAT_RGBA).build()
            val mask = try {
                val masks = task.segmentForVideo(image, ticket.timestamp).confidenceMasks().orElse(null)
                try {
                    // 模型只有一个类别 selfie：第 0 张即「是人」的置信度。
                    masks?.firstOrNull()?.let { confidence ->
                        val floats = ByteBufferExtractor.extract(confidence, MPImage.IMAGE_FORMAT_VEC32F1)
                            .order(ByteOrder.nativeOrder()).asFloatBuffer()
                        val values = FloatArray(confidence.width * confidence.height)
                        floats.get(values)
                        PersonMask.fromConfidence(values, confidence.width, confidence.height)
                    }
                } finally { masks?.forEach { it.close() } }
            } finally { image.close() }
            state.complete(ticket, mask, SystemClock.uptimeMillis())
        } catch (_: Exception) {
            // 本次预览内不重试，由调用方停用背景虚化。
            state.complete(ticket, null, SystemClock.uptimeMillis())
            state.enable(false)
            if (!closed.get()) onFailure()
        }
    }

    fun close() {
        if (!closed.compareAndSet(false, true)) return
        state.enable(false); state.reset()
        worker.execute { segmenter?.close(); segmenter = null }
        worker.shutdown()
    }

    companion object {
        /** 随安卓包分发（android/src/main/assets），SHA256 由 scripts/app-face-model-check.py 校验。 */
        const val MODEL = "selfie_segmenter.tflite"
    }
}
