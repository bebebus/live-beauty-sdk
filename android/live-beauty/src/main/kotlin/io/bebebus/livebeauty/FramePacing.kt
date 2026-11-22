package io.bebebus.livebeauty

import kotlin.math.abs

/**
 * 编码时间戳（PTS）的时基换算与去重（仅 Android，纯逻辑，时间全部由参数传入，只在 GL 线程使用）。
 *
 * RootEncoder 2.7.5 把「预览上屏之后的系统时钟」当作编码 PTS，处理耗时的方差原样进入观众端（D4）。
 * 构建期改写后（LiveSdkPtsHookFactory → [EncoderPts]），PTS 改为「被显示帧的相机时间戳 [content] + 固定时基偏移」：
 * - 时基：SurfaceTexture 时间戳声明为 REALTIME（BOOTTIME）时偏移取 0，UNKNOWN（通常是 MONOTONIC）时取
 *   nowBoot − nowMono；用采集到出帧的延迟 P = nowBoot − (newest + 偏移) 是否落在 [0, 500ms] 校验，不符就换另一种解释，
 *   两种都不符时按经验锚定（偏移 = nowBoot − newest − 50ms）；已有偏移（切镜头）且这一帧可能是旧相机的帧（见
 *   [keepsOffset]）时先保留原偏移、下一帧再判，连续 [STALE_FRAME_LIMIT] 帧都这样才经验锚定。
 *   所有 PTS 都落在 elapsedRealtime（BOOTTIME）时基，与 RootEncoder 给音频编码器的 presentTimeUs 同一时基。
 * - 去重：内容没有前进（ForceRenderer 保活重绘、FrameDelay 加一帧）时不送编码器；距上次送帧满 [KEEPALIVE_NS]
 *   才补一帧保活，PTS = 上次 + 经过时长。
 * - 连续性：内容与墙钟的步进相差超过 [DISCONTINUITY_NS]（切镜头换时基、时基跳变）时按墙钟接上，不让 PTS 跳变或倒退；
 *   任何两帧 PTS 至少相差 [MIN_STEP_NS]。
 * VideoEncoder 在 SURFACE 模式下减去自己的首帧时间戳，所以绝对偏移不影响视频轴；音画对齐见 [FramePacing.audioDelayMs]。
 */
internal class PresentationClock {
    /** [encode] 为 false 时本次绘制不送编码器（不 setPresentationTime、不 swap）。 */
    class Decision(val encode: Boolean, val ptsNs: Long, val keepalive: Boolean)

    enum class Domain { UNSET, REALTIME, MONOTONIC, EMPIRICAL }

    var domain = Domain.UNSET
        private set
    /** 当前锚定期内的时基偏移：PTS = 内容时间戳 + offsetNs。 */
    var offsetNs = 0L
        private set
    /** 最近一次送编码帧时的采集到出帧延迟 P（纳秒，诊断用）。 */
    var captureLatencyNs = 0L
        private set
    /** 按连续性重新锚定的次数（诊断用）。 */
    var reanchors = 0
        private set
    var lastPts = UNSET
        private set
    private var declaredRealtime: Boolean? = null
    private var reanchorRequested = false
    private var staleFrames = 0 // 本次重判中「两种解释都不合理、可能是旧相机的帧」而保留原偏移的帧数
    private var staleNewest = 0L // 上一个计入 staleFrames 的最新相机帧时间戳（同一帧的重绘只计一次）
    private var lastContent = 0L
    private var lastContentAt = 0L // 内容上一次前进时的 nowBoot
    private var lastEmitNow = 0L

    /** 声明相机的时间戳来源（切镜头、开预览时）；下一次 [decide] 重新判定时基。 */
    fun declare(realtime: Boolean?) {
        declaredRealtime = realtime
        reanchorRequested = true
        staleFrames = 0
    }

    /**
     * [content] 为本次绘制实际显示的相机帧时间戳（≤ 0 表示未知），[newest] 为最新相机帧时间戳，
     * [nowBoot] / [nowMono] 为 elapsedRealtimeNanos / System.nanoTime。
     */
    fun decide(content: Long, newest: Long, nowBoot: Long, nowMono: Long): Decision {
        // 时间戳未知：墙钟兜底（同一时基），只保证单调。
        if (content <= 0L) return emit(if (lastPts == UNSET) nowBoot else maxOf(nowBoot, lastPts + MIN_STEP_NS), nowBoot, false)
        val resolved = (domain == Domain.UNSET || reanchorRequested) && run {
            reanchorRequested = false
            resolve(content, newest, nowBoot, nowMono)
        }
        if (!resolved) {
            val dc = content - lastContent
            val dn = nowBoot - lastContentAt
            if (abs(dc - dn) > DISCONTINUITY_NS) {
                continueFrom(content, nowBoot) // 切镜头换了时基或时基跳变：按墙钟接上
                // 保留原偏移等待重判时按墙钟接上，但不结束重判：接上的偏移带着这一帧的年龄，切换后新相机第一帧出得慢（采集延迟
                // 偏大）时就此结束会把它留到整场。下一次绘制照常重判：能合理解释就按它锚定，否则一般按名义 50ms 经验锚定（见
                // [keepsOffset]）。代价：两种解释都不合理的设备上，这次重判若落在同一帧重绘或 FrameDelay 加帧上会送出一帧重复
                // 内容，否则名义 50ms 换掉已对齐墙钟的偏移，出现一次约 23ms 的 PTS 间隔。
            } else if (dc <= 0L || (reanchorRequested && content + offsetNs <= lastPts)) {
                // 没有新内容（保活重绘、延后加帧），或等待重判时这一帧按原偏移排在已送的 PTS 之前（送了只能按 1ms 连发）：
                // 不送；太久没送就补一帧，防止推流断画。
                return if (nowBoot - lastEmitNow >= KEEPALIVE_NS) emit(lastPts + (nowBoot - lastEmitNow), nowBoot, true)
                else SKIP
            }
        }
        val pts = if (lastPts == UNSET) content + offsetNs else maxOf(content + offsetNs, lastPts + MIN_STEP_NS)
        lastContent = content
        lastContentAt = nowBoot
        captureLatencyNs = nowBoot - (newest + offsetNs)
        return emit(pts, nowBoot, false)
    }

    /**
     * 重新判定时基；返回 false 表示这一帧判定不了、保留原偏移，下一帧再判（调用方按常规的连续性与去重处理这一帧）。
     * 已有偏移时两种解释都不合理，可能是在旧相机的帧上判定（切镜头时相机重开期间、年龄超过 500ms 的旧帧）：这时经验锚定
     * 会把旧帧的年龄带进新相机的偏移，之后整场音画错位，所以先不改偏移、维持重新判定的请求，等新相机的帧（见 [keepsOffset]）。
     */
    private fun resolve(content: Long, newest: Long, nowBoot: Long, nowMono: Long): Boolean {
        val realtimeFirst = declaredRealtime != false
        val first = if (realtimeFirst) 0L else nowBoot - nowMono
        val second = if (realtimeFirst) nowBoot - nowMono else 0L
        when {
            plausible(nowBoot - (newest + first)) -> {
                offsetNs = first; domain = if (realtimeFirst) Domain.REALTIME else Domain.MONOTONIC
            }
            plausible(nowBoot - (newest + second)) -> {
                offsetNs = second; domain = if (realtimeFirst) Domain.MONOTONIC else Domain.REALTIME
            }
            keepsOffset(newest, nowBoot) -> { reanchorRequested = true; return false }
            else -> { offsetNs = nowBoot - newest - NOMINAL_LATENCY_NS; domain = Domain.EMPIRICAL }
        }
        staleFrames = 0
        if (lastPts != UNSET) {
            val gap = content + offsetNs - lastPts
            if (gap < 0L || gap > DISCONTINUITY_NS) continueFrom(content, nowBoot)
        }
        return true
    }

    /**
     * 两种解释都不合理时是否保留原偏移。只有已有偏移、且这一帧可能是旧相机的帧时才保留：
     * - 按原偏移已超过 500ms：切镜头时相机重开期间留下的旧帧。
     * - 原偏移是经验锚定（旧相机自己两种解释都不合理，偏移按名义 50ms 锚定）且按原偏移落在 [0, 500ms]：旧相机新鲜的在途帧，
     *   按原偏移恰好约 50ms。声明本应在切换完成后才消费（见 [FramePacing.declareTimestampSource]），这里是纵深防御：在这样的帧
     *   上经验锚定会清掉重判请求，新相机之后不再判定，整场留下与两颗相机时间轴之差等量的错位。
     * 其余情况直接经验锚定：原偏移按 REALTIME / MONOTONIC 判定时，旧相机的新鲜帧本身就能被合理解释（在 [resolve] 里直接判定，
     * 到不了这里），按原偏移不到 500ms 的帧一般不是旧相机的帧；按原偏移落在未来的帧也一般不是，而且保留原偏移送出去会让 PTS
     * 超前墙钟，之后 PTS 不能倒退，这段超前会按连续性留到整场。保留有上限：连续 [STALE_FRAME_LIMIT] 帧（最新帧各不相同，
     * 同一帧的重绘不计）都这样，说明是新相机自己的时间戳不合理，改为经验锚定，否则整场留下与偏差等量的音画错位。
     */
    private fun keepsOffset(newest: Long, nowBoot: Long): Boolean {
        if (domain == Domain.UNSET || lastPts == UNSET) return false
        val latency = nowBoot - (newest + offsetNs)
        val maybeOld = latency > MAX_PLAUSIBLE_LATENCY_NS || (domain == Domain.EMPIRICAL && latency >= 0L)
        if (!maybeOld) return false
        if (staleFrames == 0 || newest != staleNewest) { staleFrames++; staleNewest = newest }
        return staleFrames < STALE_FRAME_LIMIT
    }

    /**
     * 本次绘制按外部时间戳 [pts]（原版绘制时钟，约等于当时的 nowBoot）送了编码器（决策出错兜底）：之后的 PTS 不早于它，
     * 偏移不变——之后几帧按 [MIN_STEP_NS] 追上，而不是整体平移音画关系。
     */
    fun adopt(pts: Long) {
        if (lastPts == UNSET || pts > lastPts) lastPts = pts
        lastEmitNow = maxOf(lastEmitNow, pts)
    }

    /** 让 [content] 落在上一帧 PTS 之后「墙钟经过时长」处。 */
    private fun continueFrom(content: Long, nowBoot: Long) {
        offsetNs = lastPts + maxOf(nowBoot - lastEmitNow, MIN_STEP_NS) - content
        reanchors++
    }

    private fun emit(pts: Long, nowBoot: Long, keepalive: Boolean): Decision {
        lastPts = pts
        lastEmitNow = nowBoot
        return Decision(true, pts, keepalive)
    }

    private fun plausible(latency: Long) = latency in 0L..MAX_PLAUSIBLE_LATENCY_NS

    companion object {
        const val UNSET = Long.MIN_VALUE
        const val MIN_STEP_NS = 1_000_000L
        const val KEEPALIVE_NS = 450_000_000L
        const val DISCONTINUITY_NS = 1_000_000_000L
        const val MAX_PLAUSIBLE_LATENCY_NS = 500_000_000L
        const val NOMINAL_LATENCY_NS = 50_000_000L
        /** 切镜头重判时，可能是旧相机的帧（见 keepsOffset）连续出现这么多个（含本帧）就改为经验锚定。 */
        const val STALE_FRAME_LIMIT = 3
        private val SKIP = Decision(false, 0L, false)
    }
}

/**
 * 一路推流的出帧节奏（由 CameraNativeView 持有）。VideoEffectsFilter.draw 在 GL 线程 [publish]「最新相机帧 → 实际显示的相机帧」，
 * 构建期改写后的 RootEncoder 经 [EncoderPts] 在同一次绘制的编码 setPresentationTime/swap 处调用 [decide] / [shouldSwap]。
 *
 * 映射：本帧发布过就用本帧的；某帧没有发布（滤镜提前返回、未挂载）时沿用上一次的映射（同样的延后），绝不回落到最新帧，
 * 否则内容先前跳 D 帧、再有 D 帧被判为重复，推流冻结 66–133ms。从未发布过（没有滤镜）时显示的就是最新帧。
 *
 * 音画对齐：PTS 按内容时间后，视频轴与 FrameDelay 的延后无关，只取决于开播时首帧的位置。开播前 [markStreamStart] 记下
 * 音频编码器的起点 S（elapsedRealtime），并按最近一次发布的画面延后预置估计 D·T + P（预览期编码 Surface 未就绪，
 * 不会调用 [decide]，没有别的估计来源）；首个送编码帧锁定 Δ = clamp(S − 首帧 PTS, 0, [MAX_AUDIO_DELAY_NS])，整场不变。
 * 只录制、未开播时（同样经过 [decide]）取内容的年龄 nowBoot − (content + 偏移) 作估计。
 *
 * 线程：[publish]、[beginDraw]、[decide]、[shouldSwap]、[fail]、[onSwap]、[takeSwapNs] 在 GL 线程；[markStreamStart]、
 * [declareTimestampSource] 在主线程；[audioDelayMs] 任意线程读；[drainStats] 任意线程。
 */
class FramePacing internal constructor(private val clock: PresentationClock) {
    constructor() : this(PresentationClock())

    private val mapping = Any()
    private var hasMapping = false
    private var publishedNewest = 0L
    private var publishedContent = 0L
    private var cached: PresentationClock.Decision? = null // 本次绘制的决策：编码与录制两路共用
    private var lastNewest = 0L // 仅 GL 线程：上一次决策时的最新相机帧时间戳
    @Volatile private var declarePending = false
    @Volatile private var declared: Boolean? = null
    @Volatile private var streamStartNs = 0L
    @Volatile private var streamStartPending = false
    @Volatile private var audioLocked = false
    @Volatile private var swapNs = 0L
    @Volatile private var effects = 0 // 见 noteEffects

    /** 推流声音应延后的毫秒数（AudioDelayLine 读取）。 */
    @Volatile var audioDelayMs = 0L
        private set

    private val stats = PacingStats()

    /** GL 线程：本次绘制显示的是时间戳为 [content] 的相机帧，最新相机帧为 [newest]。 */
    fun publish(newest: Long, content: Long) {
        synchronized(mapping) {
            publishedNewest = newest; publishedContent = content; hasMapping = true
        }
    }

    /** 最新相机帧为 [newest] 时实际显示的相机帧时间戳。 */
    fun contentFor(newest: Long): Long = synchronized(mapping) {
        when {
            !hasMapping || publishedNewest == newest -> if (hasMapping) publishedContent else newest
            // 本帧没有发布：沿用上一次的延后。
            else -> newest - (publishedNewest - publishedContent)
        }
    }

    /** GL 线程：一次 OpenGlView.draw 开始（改写后的 getCurrentTimeNano 处），清掉上一次绘制的决策。 */
    fun beginDraw() { cached = null }

    /** GL 线程：同一次绘制里编码与录制两路共用同一个决策。 */
    internal fun decide(newest: Long, nowBoot: Long, nowMono: Long): PresentationClock.Decision {
        cached?.let { return it }
        // 时间戳来源的声明只在出现新相机帧的绘制上消费：ForceRenderer 保活重绘、同一帧重画仍是旧相机的帧，
        // 在它上面判定时基会把旧帧的解释（或年龄）带给新相机。
        if (declarePending && newest != lastNewest) { declarePending = false; clock.declare(declared) }
        lastNewest = newest
        val content = contentFor(newest)
        val decision = clock.decide(content, newest, nowBoot, nowMono)
        if (decision.encode) {
            if (streamStartPending) {
                streamStartPending = false
                audioLocked = true
                audioDelayMs = (streamStartNs - decision.ptsNs).coerceIn(0L, MAX_AUDIO_DELAY_NS) / 1_000_000L
            } else if (!audioLocked && content > 0L) {
                audioDelayMs = (nowBoot - (content + clock.offsetNs)).coerceIn(0L, MAX_AUDIO_DELAY_NS) / 1_000_000L
            }
        }
        stats.record(decision, clock.captureLatencyNs, clock.reanchors)
        cached = decision
        return decision
    }

    /** 本次绘制是否把画面送进编码器；没有决策（未经过 [decide]）时照常送。 */
    fun shouldSwap(): Boolean = cached?.encode ?: true

    /** 决策出错：本次绘制按原时间戳 [legacyPts] 照常送，并让之后的 PTS 不早于它。 */
    fun fail(legacyPts: Long) {
        cached = PresentationClock.Decision(true, legacyPts, false)
        clock.adopt(legacyPts)
    }

    /** 编码 swap 耗时（eglSwapBuffers 可能被编码器背压阻塞），只做诊断，不计入美颜负载。 */
    fun onSwap(ns: Long) { swapNs += ns }

    /** 读后清零。 */
    fun takeSwapNs(): Long = swapNs.also { swapNs = 0L }

    /**
     * 主线程，紧挨 startStream（音频编码器以此刻为起点）之前调用。预览期编码 Surface 未就绪、从不 [decide]，
     * 麦克风首块又可能先于首帧送编码：这里先按最近一次发布的画面延后预置估计 D·T + P（P 取 NOMINAL_LATENCY_NS），
     * AudioDelayLine 首块直接到位（第一场不再从 0 起滑），首帧锁定 Δ 后只差几十毫秒。
     */
    fun markStreamStart(sBootNs: Long) {
        val lag = synchronized(mapping) { if (hasMapping) publishedNewest - publishedContent else 0L }
        audioDelayMs = (lag + PresentationClock.NOMINAL_LATENCY_NS).coerceIn(0L, MAX_AUDIO_DELAY_NS) / 1_000_000L
        streamStartNs = sBootNs
        streamStartPending = true
        audioLocked = false
    }

    /**
     * 主线程：开预览、切镜头时声明新相机的时间戳来源（REALTIME 为 true，UNKNOWN 为 false，读不到为 null）。
     * 切镜头须在相机切换完成之后调用（旧相机已关、在途帧已画完），声明在下一个新相机帧上才消费（见 [decide]）。
     */
    fun declareTimestampSource(realtime: Boolean?) {
        declared = realtime
        declarePending = true
    }

    /** 自上次调用以来的统计（任意线程）。 */
    fun drainStats(): PacingStats.Window = stats.drain(clock.domain.name)

    /** GL 线程：美颜滤镜报告负载级别、画面延后帧数与回读模式，只用于诊断日志（打包成一个 volatile 值，不分配）。 */
    fun noteEffects(level: Int, delayFrames: Int, deferredReadback: Boolean) {
        effects = 1 or (level.coerceIn(0, 15) shl 1) or (delayFrames.coerceIn(0, 15) shl 5) or
            (if (deferredReadback) 1 shl 9 else 0)
    }

    /** 诊断日志用的美颜状态；滤镜从未报告时为 effects=off。 */
    fun effectsSummary(): String {
        val packed = effects
        if (packed and 1 == 0) return "effects=off"
        return "level=${(packed shr 1) and 15} delay=${(packed shr 5) and 15} readback=" +
            if (packed and (1 shl 9) != 0) "DEFERRED" else "SYNC"
    }

    companion object {
        /** 与 AudioDelayLine.MAX_DELAY_MS 一致。 */
        const val MAX_AUDIO_DELAY_NS = 300_000_000L
    }
}

/** 出帧节奏的窗口统计（诊断日志 RtmpPacing 用），记录在 GL 线程、读取在主线程。 */
class PacingStats {
    /** 一个统计窗口：PTS 间隔与采集延迟为毫秒分位数（没有样本时为 NaN）。 */
    class Window(
        val encoded: Int, val skipped: Int, val keepalive: Int, val reanchors: Int, val domain: String,
        val intervalP5: Double, val intervalP50: Double, val intervalP95: Double,
        val latencyP5: Double, val latencyP50: Double, val latencyP95: Double,
    ) {
        override fun toString(): String = String.format(
            java.util.Locale.ROOT,
            "encoded=%d skipped=%d keepalive=%d reanchor=%d domain=%s pts(ms) p5/p50/p95=%.1f/%.1f/%.1f " +
                "latency(ms)=%.1f/%.1f/%.1f",
            encoded, skipped, keepalive, reanchors, domain,
            intervalP5, intervalP50, intervalP95, latencyP5, latencyP50, latencyP95,
        )
    }

    private val intervals = LongArray(CAPACITY)
    private val latencies = LongArray(CAPACITY)
    private var samples = 0
    private var encoded = 0
    private var skipped = 0
    private var keepalive = 0
    private var reanchorTotal = 0
    private var reanchors = 0
    private var lastPts = PresentationClock.UNSET

    @Synchronized internal fun record(decision: PresentationClock.Decision, latencyNs: Long, totalReanchors: Int) {
        if (totalReanchors > reanchorTotal) reanchors += totalReanchors - reanchorTotal
        reanchorTotal = totalReanchors
        if (!decision.encode) { skipped++; return }
        encoded++
        if (decision.keepalive) keepalive++
        if (lastPts != PresentationClock.UNSET && samples < CAPACITY) {
            intervals[samples] = decision.ptsNs - lastPts
            latencies[samples] = latencyNs
            samples++
        }
        lastPts = decision.ptsNs
    }

    @Synchronized fun drain(domain: String): Window {
        val window = Window(
            encoded, skipped, keepalive, reanchors, domain,
            percentileMs(intervals, samples, 0.05), percentileMs(intervals, samples, 0.5), percentileMs(intervals, samples, 0.95),
            percentileMs(latencies, samples, 0.05), percentileMs(latencies, samples, 0.5), percentileMs(latencies, samples, 0.95),
        )
        samples = 0; encoded = 0; skipped = 0; keepalive = 0; reanchors = 0
        return window
    }

    private fun percentileMs(values: LongArray, count: Int, q: Double): Double {
        if (count == 0) return Double.NaN
        val sorted = values.copyOf(count).also { it.sort() }
        return sorted[((count - 1) * q).toInt()] / 1e6
    }

    companion object {
        /** 2 秒窗口、60 fps 也够用；满了之后本窗口只计数。 */
        const val CAPACITY = 128
    }
}
