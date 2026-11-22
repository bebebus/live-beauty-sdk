package io.bebebus.livebeauty

import android.content.Context
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 直播美颜毛发边界用的头发分割（MediaPipe Hair Segmenter float32/1，CPU，约 10 Hz）。结构同 [PersonSegmenter]：
 * 共享小图第 4 路（[FaceTextureSampler.HAIR]，竖屏 216×384 全帧，不裁切），独立的单线程 worker 与 [MaskTrackingState]
 * （有效期 [HairMaskTuning.FRESH_MS]），故障域、生命周期与人像分割、人脸各自独立。
 * - 结果复用 [PersonMask]：每像素头发置信度 0–255，行 0 是画面顶部，尺寸等于输入小图；
 * - 模型标签必须恰为 [background, hair]，取置信图第 [HAIR_INDEX] 张，尺寸须等于输入（不符都按失败处理）；
 * - [inferenceP50Ms]：最近 [HairMaskTuning.P50_WINDOW] 次 segmentForVideo 墙钟耗时的中位数，供调用方定间隔与健康判定；
 * - 时域 EMA 在 worker 上做（[HairMaskTuning.temporalBlend]），运动门比较调用方在 [submit] 时给的人脸锚点；
 * - 缓冲复用：置信图直接按 Tasks 输出的 FloatBuffer 量化（不再拷成 FloatArray），量化与 EMA 写进从 state 备用池取回的
 *   遮罩（[MaskTrackingState.obtain]，交接规则见 MaskTrackingState），稳态下每次推理不再新建 216×384 的数组；
 * - 任何异常：本票作废、state 停用、在 worker 线程回调 onFailure，本次预览内不重试，调用方回到第一批启发式（A 通道恒 1）；
 *   [close] 之后不回调、不再推理，worker 释放模型后退出。
 * 头发完成时间不并入 FrameDelay 的 trackCompleted（推理慢不拉长画面延后），由调用方保证。
 *
 * 主构造器供 JVM 单测注入推理后端、worker 与时钟；产品走 (context, onFailure)。
 */
internal class HairSegmenter internal constructor(
    private val open: () -> Engine,
    private val onFailure: () -> Unit,
    private val worker: ExecutorService = Executors.newSingleThreadExecutor(),
    private val clock: () -> Long = { SystemClock.uptimeMillis() },
    private val log: (String) -> Unit = { Log.w(TAG, it) },
) {
    constructor(context: Context, onFailure: () -> Unit) : this(opener(context.applicationContext), onFailure)

    val state = MaskTrackingState(HairMaskTuning.FRESH_MS, HairMaskTuning.MASK_SPARES)
    private val closed = AtomicBoolean(false)
    private var engine: Engine? = null // 仅 worker 访问。
    // EMA 的上一帧结果、它的人脸锚点与代际（仅 worker 访问）。
    private var previous: PersonMask? = null
    private var previousAnchor: FloatArray? = null
    private var previousEpoch = Long.MIN_VALUE
    // 推理耗时环形缓冲（ms，仅 worker 写）；中位数算好后发布给 GL 线程。
    private val timings = FloatArray(HairMaskTuning.P50_WINDOW)
    private var timingCount = 0
    private var timingHead = 0
    @Volatile private var p50 = Float.NaN

    /** 一次推理的后端，只在 worker 上使用：产品为 MediaPipe（[TasksEngine]），JVM 单测注入假实现。 */
    internal interface Engine {
        /** 模型内嵌标签（顺序即置信图顺序）；不是 [HairMaskTuning.LABELS] 就按加载失败处理。 */
        val labels: List<String>?

        /**
         * 推理采样时间 [timestampMs] 的 [sample]，按 [hairMask] 的规则把头发置信度量化写进 [out]（尺寸等于输入小图）；
         * 返回 segmentForVideo 本身的墙钟耗时（ms，供 p50）。不符或推理失败抛异常。
         */
        fun segment(sample: FaceTextureSampler.Sample, timestampMs: Long, out: PersonMask): Float

        fun close()
    }

    /**
     * GL 线程调用；[sample] 的缓冲区在 complete 之前归 worker 使用。[anchor] 是推理时刻的人脸锚点 (cx, cy, 半宽)，
     * 遮罩像素坐标（见 [HairMaskTuning.temporalBlend]），没有脸时传 null（不做 EMA）；这里会拷贝一份。
     */
    fun submit(sample: FaceTextureSampler.Sample, ticket: MaskTrackingState.Ticket, anchor: FloatArray?) {
        if (closed.get()) { state.complete(ticket, null, clock()); return }
        val pinned = anchor?.copyOf()
        try { worker.execute { run(sample, ticket, pinned) } }
        catch (_: RejectedExecutionException) { state.complete(ticket, null, clock()) }
    }

    /** 最近 [HairMaskTuning.P50_WINDOW] 次推理耗时的中位数（ms）；不足 [HairMaskTuning.MIN_SAMPLES] 次为 NaN。任意线程。 */
    fun inferenceP50Ms(): Float = p50

    private fun run(sample: FaceTextureSampler.Sample, ticket: MaskTrackingState.Ticket, anchor: FloatArray?) {
        try {
            if (closed.get()) { state.complete(ticket, null, clock()); return }
            val task = engine ?: create().also { engine = it }
            val w = sample.width; val h = sample.height
            // 不取 EMA 上一帧那块（下面要读它）；池里没有可用的才新建。
            val out = state.obtain(w, h, previous) ?: PersonMask(ByteArray(w * h), w, h)
            record(task.segment(sample, ticket.timestamp, out))
            // out 还没交出去：原地混合，不再另建一块。
            val mask = HairMaskTuning.temporalBlend(previous, previousAnchor, out, anchor, previousEpoch == ticket.epoch, out = out)
            previous = mask; previousAnchor = anchor; previousEpoch = ticket.epoch
            state.complete(ticket, mask, clock())
        } catch (e: Exception) {
            // 本次预览内不重试，由调用方停用头发遮罩（不发 effects_failed，只记一次日志）。
            state.complete(ticket, null, clock())
            state.enable(false)
            if (!closed.get()) {
                log("头发分割停用：${e.javaClass.simpleName}")
                onFailure()
            }
        }
    }

    /** 在 worker 上懒建后端；标签不是 [background, hair] 就先关掉它再按加载失败处理。 */
    private fun create(): Engine {
        val created = open()
        if (!HairMaskTuning.labelsValid(created.labels)) {
            created.close()
            error("模型标签不是 background/hair")
        }
        return created
    }

    private fun record(ms: Float) {
        timings[timingHead] = ms
        timingHead = (timingHead + 1) % timings.size
        timingCount = minOf(timingCount + 1, timings.size)
        p50 = HairMaskTuning.p50(timings, timingCount)
    }

    fun close() {
        if (!closed.compareAndSet(false, true)) return
        state.enable(false); state.reset()
        worker.execute { engine?.close(); engine = null; previous = null; previousAnchor = null }
        worker.shutdown()
    }

    /** 一张置信图的只读视图：行优先、行 0 在顶部（worker 上由 MPImage 抽出，JVM 单测直接构造）。 */
    internal class Plane(val values: FloatBuffer, val width: Int, val height: Int)

    /** 产品后端：MediaPipe ImageSegmenter，仅 CPU（BaseOptions 只设模型路径），VIDEO 模式同步推理。[model] 只供 androidTest 换模型。 */
    internal class TasksEngine(context: Context, model: String = MODEL) : Engine {
        private val task = ImageSegmenter.createFromOptions(context,
            ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(model).build())
                .setRunningMode(RunningMode.VIDEO)
                .setOutputConfidenceMasks(true).setOutputCategoryMask(false).build())

        override val labels: List<String>? get() = task.labels

        override fun segment(sample: FaceTextureSampler.Sample, timestampMs: Long, out: PersonMask): Float {
            val image = ByteBufferImageBuilder(sample.pixels, sample.width, sample.height, MPImage.IMAGE_FORMAT_RGBA).build()
            try {
                val start = SystemClock.elapsedRealtimeNanos()
                Trace.beginSection("beauty.hair.infer")
                val result = try { task.segmentForVideo(image, timestampMs) } finally { Trace.endSection() }
                val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000f
                // 同步 VIDEO 模式没有 resultListener，Tasks 已把每张置信图深拷贝进它自己新分配的直接缓冲（本次推理独占，
                // 这两次分配在同步 API 下去不掉）；extract 格式一致时只返回只读视图，这里在 close 之前直接按视图量化。
                val masks = result.confidenceMasks().orElse(null) ?: error("没有置信图")
                try {
                    hairMask(masks, sample.width, sample.height, out) { confidence ->
                        Plane(ByteBufferExtractor.extract(confidence, MPImage.IMAGE_FORMAT_VEC32F1)
                            .order(ByteOrder.nativeOrder()).asFloatBuffer(), confidence.width, confidence.height)
                    }
                } finally { masks.forEach { it.close() } }
                return ms
            } finally { image.close() }
        }

        override fun close() { task.close() }
    }

    companion object {
        /** 随安卓包分发（android/src/main/assets），SHA256 由 scripts/app-face-model-check.py 校验。 */
        const val MODEL = "hair_segmenter.tflite"
        /** 置信图与标签同序：0 background、1 hair。 */
        const val HAIR_INDEX = 1
        private const val TAG = "HairSegmenter"

        /** 产品后端的工厂：只持有 applicationContext，在 worker 上第一次推理时才建模型。 */
        private fun opener(app: Context): () -> Engine = { TasksEngine(app) }

        /**
         * 从按标签排列的置信图里取头发那张并量化：张数须等于标签数，取第 [HAIR_INDEX] 张（只把这一张转成 [Plane]），
         * 尺寸须等于输入小图（Tasks 默认把遮罩缩放回输入尺寸），再按 [PersonMask.fromConfidence] 转 0–255
         * （非有限值按 0、越界钳位、行序不变）。直接读 Plane 的 FloatBuffer，不拷成 FloatArray；[out] 非 null 时写进它并返回它。
         * 不符抛异常，调用方按推理失败处理。
         */
        internal fun <T> hairMask(masks: List<T>, inputWidth: Int, inputHeight: Int, out: PersonMask? = null,
                                  plane: (T) -> Plane): PersonMask {
            check(masks.size == HairMaskTuning.LABELS.size) { "置信图 ${masks.size} 张，应为 ${HairMaskTuning.LABELS.size} 张" }
            val hair = plane(masks[HAIR_INDEX])
            check(hair.width == inputWidth && hair.height == inputHeight) {
                "遮罩 ${hair.width}x${hair.height} 与输入 ${inputWidth}x$inputHeight 不符"
            }
            return PersonMask.fromConfidence(hair.values, hair.width, hair.height, out)
        }
    }
}
