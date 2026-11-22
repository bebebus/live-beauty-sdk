package io.bebebus.livebeauty

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.os.Process
import android.os.SystemClock
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/** CPU推理与GL隔离；最多一个小图任务，不排帧、不在Dart传帧。onFailure 在 worker 线程回调，只表示人脸故障。 */
class FaceLandmarkTracker(context: Context, private val onFailure: () -> Unit) {
    private val appContext=context.applicationContext
    val state=FaceTrackingState()
    private val closed=AtomicBoolean(false)
    private val worker=Executors.newSingleThreadExecutor()
    private var detector: FaceLandmarker?=null // 仅worker访问。
    private val policy=policyFor(appContext)
    /**
     * 当前（或将要）用 GPU 推理：GPU 推理比 CPU 快 2–4 倍，画面对齐只需延后 1–2 帧；建不起来或推理出错就回退 CPU。
     * 何时不再用 GPU 由 [GpuDelegatePolicy] 决定（原生崩溃与 Java 异常分开计数，成功一次清零）。worker 写，GL 线程读来选采样间隔。
     */
    @Volatile var gpu=policy.allowGpu()
        private set
    private var gpuConfirmed=false // 仅 worker：本实例 GPU 已成功推理过一次。
    /**
     * GL线程调用。可能已被另一线程关闭（OpenGlView.stop() 在调用线程 release 滤镜）：丢弃本次样本，
     * 不算人脸故障。[sample] 的缓冲区在 complete 之前归 worker 使用。
     */
    internal fun submit(sample: FaceTextureSampler.Sample, ticket: FaceTrackingState.Ticket) {
        if (closed.get()) { state.complete(ticket,null,SystemClock.uptimeMillis()); return }
        try { worker.execute { run(sample,ticket) } }
        catch (_: RejectedExecutionException) { state.complete(ticket,null,SystemClock.uptimeMillis()) }
    }
    private fun run(sample: FaceTextureSampler.Sample, ticket: FaceTrackingState.Ticket) {
        try {
            if (closed.get()) return
            val image=ByteBufferImageBuilder(sample.pixels,sample.width,sample.height,MPImage.IMAGE_FORMAT_RGBA).build()
            val (geometry, mesh)=try {
                val faces=try { task().detectForVideo(image,ticket.timestamp).faceLandmarks() }
                catch (e: Throwable) {
                    // GPU 推理出错：关掉 GPU 实例，本帧丢弃（不算人脸故障），本进程之后用 CPU；CPU 出错照旧是人脸故障。
                    if (!gpu) throw e
                    disableGpu()
                    try { detector?.close() } catch (_: Throwable) {}
                    detector=null
                    state.complete(ticket,null,SystemClock.uptimeMillis())
                    return
                }
                if (gpu && !gpuConfirmed) {
                    // 第一次 GPU 推理成功才撤掉「正在尝试」标记并清零计数；在这之前原生崩溃，下次启动按一次崩溃计。
                    gpuConfirmed=true; policy.onGpuSuccess()
                }
                // numFaces = 1 时最多一张（跟踪以上一帧 ROI 为准，第二张脸入镜不会抢走主脸）；NUM_FACES > 1（A/B）时取第一张。
                faces.firstOrNull()?.let { face ->
                    val aspect = ticket.width.toFloat() / ticket.height
                    val pts = face.map { it.x() to it.y() }
                    val meshFrame = FaceMeshFrame.fromMediaPipe(pts, aspect)
                    // 0.45 < |r| <= 0.72 侧脸段由 meshFrame 提供兼容几何（仅遮罩），避免大侧脸突然丢脸闪烁。
                    val geo = FaceGeometry.fromMesh(pts, aspect) ?: meshFrame?.toCompatGeometry(aspect)
                    geo to meshFrame
                } ?: (null to null)
            } finally { image.close() }
            state.complete(ticket,geometry,SystemClock.uptimeMillis(),mesh)
        } catch (_: Throwable) {
            // 模型加载或推理失败是人脸故障（不是渲染故障），与代际无关；本次预览内不重试，由调用方关闭跟踪。
            state.complete(ticket,null,SystemClock.uptimeMillis())
            state.enable(false)
            if (!closed.get()) onFailure()
        }
    }
    /** 仅 worker：按需建检测器，GPU 建不起来就回退 CPU（建 CPU 失败才是人脸故障）。 */
    private fun task(): FaceLandmarker {
        detector?.let { return it }
        if (gpu && policy.allowGpu()) {
            // 先同步落盘「正在尝试 GPU」：建实例或首次推理时原生崩溃，下次启动按一次崩溃计。
            policy.beforeGpuAttempt()
            try { return create(Delegate.GPU).also { detector=it } } catch (_: Throwable) { disableGpu() }
        }
        gpu=false
        return create(Delegate.CPU).also { detector=it }
    }
    /** 仅 worker：Java 异常，本进程改用 CPU（以后的进程仍可再试，见 GpuDelegatePolicy）。 */
    private fun disableGpu() { gpu=false; policy.onGpuJavaFailure() }
    /**
     * minTrackingConfidence 用 MediaPipe 默认 0.5（只在检测器运行时作关联阈值）；检测与在场阈值保持 0.6
     * （下调会让被手遮挡时继续出点，口红画到手上）。
     */
    private fun create(delegate: Delegate)=FaceLandmarker.createFromOptions(appContext,
        FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").setDelegate(delegate).build())
            .setRunningMode(RunningMode.VIDEO).setNumFaces(NUM_FACES)
            .setMinFaceDetectionConfidence(0.6f).setMinFacePresenceConfidence(0.6f)
            .setMinTrackingConfidence(0.5f).build())
    companion object {
        /**
         * 同时检测的人脸数。取 1：MediaPipe 只在 num_faces == 1 的视频模式下开启内置关键点平滑（OneEuro，min_cutoff 0.05、
         * beta 80），并且跟踪到的脸数不少于 num_faces 时跳过检测器（numFaces = 2 时单人直播每帧都多跑一次检测，且没有内置平滑）；
         * 跟踪以上一帧 ROI 为准，第二张脸（路人、海报、镜子）入镜不会抢走主脸。代价：快速平移时关键点本身有滞后（仿真 p95
         * 约 6.8 px@720），Java API 不能单独关闭这层平滑。留作常量供测试包 A/B：改成 > 1 时取第一张脸，
         * 此时没有内置平滑，平滑全靠 FaceMeshFilter（它的死区按自适应噪声底标定，不依赖内置平滑）。
         */
        const val NUM_FACES=1
        private const val PREFS="rtmp_face_tracker"
        /** 进程级 GPU 回退策略（构造时做一次启动检查，见 GpuDelegatePolicy）。 */
        @Volatile private var sharedPolicy: GpuDelegatePolicy?=null
        private fun policyFor(context: Context): GpuDelegatePolicy = sharedPolicy ?: synchronized(this) {
            // minSdk 24 起 Process.getStartUptimeMillis 一定可用；标记同时记 pid 与它，防原生崩溃后 pid 复用被误判为同进程。
            sharedPolicy ?: GpuDelegatePolicy(PrefsStore(context.getSharedPreferences(PREFS,Context.MODE_PRIVATE)),
                Build.FINGERPRINT ?: "", Process.myPid(), Process.getStartUptimeMillis()).also { sharedPolicy=it }
        }
    }
    /** GpuDelegatePolicy 的 SharedPreferences 适配器：类型不符的旧值（含旧版 gpu_trying 指纹字符串）按缺省读，不抛异常。 */
    private class PrefsStore(private val prefs: SharedPreferences) : GpuDelegatePolicy.Store {
        override fun getString(key: String): String? =
            try { prefs.getString(key,null) } catch (_: ClassCastException) { null }
        override fun getInt(key: String, default: Int): Int =
            try { prefs.getInt(key,default) } catch (_: ClassCastException) { default }
        override fun put(changes: Map<String, Any?>, sync: Boolean) {
            val editor=prefs.edit()
            for ((key, value) in changes) when (value) {
                null -> editor.remove(key)
                is Int -> editor.putInt(key,value)
                is String -> editor.putString(key,value)
                else -> editor.putString(key,value.toString())
            }
            if (sync) editor.commit() else editor.apply()
        }
    }
    fun close() {
        if (!closed.compareAndSet(false,true)) return
        state.enable(false); state.reset()
        worker.execute { detector?.close(); detector=null }
        worker.shutdown()
    }
}
