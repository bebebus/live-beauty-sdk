package io.bebebus.livebeauty

import java.nio.BufferUnderflowException
import java.nio.FloatBuffer

/**
 * 人像分割结果（仅 Android）：每像素「是人」的置信度 0–255，[pixels] 行 0 是画面顶部（与 FaceTextureSampler 的读回一致）。
 * 不保存图像本身。
 */
class PersonMask(val pixels: ByteArray, val width: Int, val height: Int) {
    init { require(width > 0 && height > 0 && pixels.size == width * height) }

    companion object {
        /** MediaPipe 的 float 置信度（行优先、行 0 在顶部）→ 0–255；非有限值按 0。 */
        fun fromConfidence(values: FloatArray, width: Int, height: Int): PersonMask {
            require(values.size >= width * height)
            val bytes = ByteArray(width * height) { i -> quantize(values[i]) }
            return PersonMask(bytes, width, height)
        }

        /**
         * 同上，但直接读 MediaPipe 置信图的 [values] 视图（从下标 0 起按绝对下标读，不改它的 position），不先拷成 FloatArray；
         * [into] 非 null 时量化写进它（尺寸须为 [width]×[height]）并返回它本身，否则新建。[values] 的 limit 不足
         * [width]×[height] 时抛 BufferUnderflowException（与整块 get 相同，不读越界）。
         */
        fun fromConfidence(values: FloatBuffer, width: Int, height: Int, into: PersonMask? = null): PersonMask {
            val n = width * height
            if (values.limit() < n) throw BufferUnderflowException()
            val out = into ?: PersonMask(ByteArray(n), width, height)
            require(out.width == width && out.height == height)
            val bytes = out.pixels
            for (i in 0 until n) bytes[i] = quantize(values.get(i))
            return out
        }

        /** 置信度 → 0–255：钳到 [0, 1]、×255 后四舍五入；NaN、无穷按 0。 */
        private fun quantize(v: Float): Byte = (if (v.isFinite()) (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt() else 0).toByte()
    }
}

/**
 * 与 [BodyTrackingState] 相同的规则（单任务、限频、代际取消），保留最近 4 帧带时间戳遮罩供 FrameDelay 按显示时间精确匹配。
 * [freshMs] 是遮罩有效期：人像分割用默认的 [FRESH_MS]；头发分割约 10 Hz、推理更慢，用 HairMaskTuning.FRESH_MS。
 *
 * 遮罩复用（[spares] > 0，头发分割用；人像分割默认 0，不复用、行为不变）：移出历史的遮罩（被挤出、清空、代际作废、过期作废）
 * 进备用池，最多留 [spares] 块；worker 用 [obtain] 取回一块同尺寸的改写，稳态下不再每次新建。交接规则：
 * - 历史里的遮罩不出池（GL 线程随时可能 snapshotAt 取到它）；
 * - GL 线程最近 [LENT_CAP] 次 snapshot / snapshotAt 交出去的遮罩不出池：调用方只在取到它之后、对同一状态再调用
 *   [LENT_CAP] 次之前读 pixels（VideoEffectsFilter 每帧 snapshot 一次、snapshotAt 一次，同帧内上传完）；
 * - [obtain] 的 keep（worker 留作 EMA 上一帧的那块）不出池；
 * - 写入与读取的先后由本对象的锁建立：worker 改写完才 [complete]，GL 线程读完之后的下一次 snapshot 才让它可再出池。
 * 停用（enable(false)）时清空备用池，停用期间迟到的在途结果也不进池，不在长时间停用期间占着内存。
 */
class MaskTrackingState(val freshMs: Long = FRESH_MS, private val spares: Int = 0) {
    data class Ticket(val epoch: Long, val timestamp: Long, val width: Int, val height: Int)
    data class Snapshot(val epoch: Long, val sampleTimestamp: Long, val mask: PersonMask?, val completed: Long = Long.MIN_VALUE)
    private var epoch = 0L
    private var busy = false
    private var lastSample = Long.MIN_VALUE
    private var lastSubmitted = -1L
    private var dimensions = 0 to 0
    private var enabled = false
    private var completed = Long.MIN_VALUE
    private val history = ArrayDeque<Pair<Ticket, PersonMask>>(HISTORY_CAP)
    private val pool = ArrayDeque<PersonMask>(maxOf(spares, 0))
    private val lent = arrayOfNulls<PersonMask>(LENT_CAP)
    private var lentHead = 0
    @Synchronized fun enable(value: Boolean) {
        if (enabled != value) { reset(); enabled = value }
        if (!value) pool.clear()
    }
    @Synchronized fun reset() { epoch++; retireHistory(); lastSample = Long.MIN_VALUE; completed = Long.MIN_VALUE }
    /** 延迟回读画小图那一帧用，见 [FaceTrackingState.due]：不看是否忙、不取票，到期返回当前代际。 */
    @Synchronized fun due(now: Long, width: Int, height: Int, intervalMs: Long = 100L): Long? {
        if (dimensions != (width to height)) { reset(); dimensions = width to height }
        if (!enabled || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now - lastSample < intervalMs)) return null
        return epoch
    }
    /** [expectEpoch] 非 null 时代际不同就不取票，见 [FaceTrackingState.begin]。 */
    @Synchronized fun begin(now: Long, width: Int, height: Int, intervalMs: Long = 100L, expectEpoch: Long? = null): Ticket? {
        if (dimensions != (width to height)) { reset(); dimensions = width to height }
        if (expectEpoch != null && expectEpoch != epoch) return null
        if (!enabled || busy || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now - lastSample < intervalMs)) return null
        busy = true; lastSample = now; lastSubmitted = now
        return Ticket(epoch, now, width, height)
    }
    @Synchronized fun complete(ticket: Ticket, mask: PersonMask?, now: Long): Boolean {
        busy = false
        // 作废（代际变了、已停用）：启用时进备用池；停用期间不进，否则停用时清空的备用池又被迟到的在途结果占住一块。
        if (ticket.epoch != epoch || !enabled) { if (enabled) mask?.let { retire(it) }; return false }
        completed = maxOf(completed, ticket.timestamp)
        if (mask != null && now - ticket.timestamp in 0..freshMs) {
            if (history.size >= HISTORY_CAP) retire(history.removeFirst().second)
            history.addLast(ticket to mask)
        } else {
            retireHistory()
            mask?.let { retire(it) }
        }
        return true
    }

    /**
     * worker 调用：从备用池取一块 [width]×[height] 的遮罩来改写（尺寸不符的顺手丢弃）；不在历史里、不是 GL 线程最近交出的、
     * 也不是 [keep]。池里没有可用的返回 null，由调用方新建。复用关闭（[spares] = 0）时恒为 null。
     */
    @Synchronized fun obtain(width: Int, height: Int, keep: PersonMask?): PersonMask? {
        var i = 0
        while (i < pool.size) {
            val mask = pool[i]
            if (mask.width != width || mask.height != height) { pool.removeAt(i); continue }
            if (mask !== keep && !isLent(mask)) { pool.removeAt(i); return mask }
            i++
        }
        return null
    }
    @Synchronized fun snapshot(now: Long): Snapshot = snapshotAt(now, now)

    /** 取有效期内与被显示帧时间戳 [targetTs] 最接近的人像分割遮罩，消除画面延后引起的抠像残影。 */
    @Synchronized fun snapshotAt(targetTs: Long, now: Long): Snapshot {
        if (!enabled) { lend(null); return Snapshot(epoch, Long.MIN_VALUE, null, Long.MIN_VALUE) }
        var best: Pair<Ticket, PersonMask>? = null
        var bestDiff = Long.MAX_VALUE
        for (entry in history) {
            if (now - entry.first.timestamp in 0..freshMs) {
                val diff = kotlin.math.abs(entry.first.timestamp - targetTs)
                if (diff <= bestDiff) {
                    bestDiff = diff
                    best = entry
                }
            }
        }
        lend(best?.second)
        return Snapshot(epoch, best?.first?.timestamp ?: Long.MIN_VALUE, best?.second, completed)
    }

    /** 记下本次交给 GL 线程的遮罩（覆盖最早的一条），复用关闭时不记。 */
    private fun lend(mask: PersonMask?) {
        if (spares <= 0) return
        lent[lentHead] = mask
        lentHead = (lentHead + 1) % LENT_CAP
    }

    private fun isLent(mask: PersonMask): Boolean = lent.any { it === mask }

    /** 清空历史；复用时逐块先移出历史再进备用池。 */
    private fun retireHistory() {
        if (spares <= 0) { history.clear(); return }
        while (history.isNotEmpty()) retire(history.removeFirst().second)
    }

    /** 移出历史的遮罩进备用池（复用关闭、仍在历史里、已在池里的跳过；满了丢最早的一块）。 */
    private fun retire(mask: PersonMask) {
        if (spares <= 0 || history.any { it.second === mask } || pool.any { it === mask }) return
        if (pool.size >= spares) pool.removeFirst()
        pool.addLast(mask)
    }

    companion object {
        /**
         * 人像分割的遮罩有效期（默认值）。分割每帧都采（约 15–30Hz），超过 200ms 的旧遮罩边缘已明显滞后（人像移动时的背景残影），
         * 宁可按 MaskPresence 保持再渐出；画面延后与显示时间的差另由调用方按比例衰减。
         */
        const val FRESH_MS = 200L
        private const val HISTORY_CAP = 4
        /** 复用时 GL 线程最近交出的这么多张遮罩不出池（每帧 snapshot 与 snapshotAt 各一次）。 */
        const val LENT_CAP = 2
    }
}

/**
 * 遮罩的存在度（仅 GL 线程）：有新鲜遮罩时 [fadeInMs] 渐入；没有时保持 [holdMs] 再 [fadeOutMs] 渐出，
 * 偶发一次识别失败不闪烁，持续没有遮罩时也不拖着旧遮罩太久。默认值是背景虚化用的 200 / [HOLD_MS] / [FADE_OUT_MS]；
 * 头发遮罩约 10 Hz，用更长的 300 / 300 / 400。
 */
class MaskPresence(
    private val fadeInMs: Float = FADE_IN_MS,
    private val holdMs: Float = HOLD_MS,
    private val fadeOutMs: Float = FADE_OUT_MS,
) {
    var value = 0f
        private set
    private var missing = 0f
    private var last = Long.MIN_VALUE

    fun reset() { value = 0f; missing = 0f; last = Long.MIN_VALUE }

    fun update(hasMask: Boolean, nowMs: Long): Float {
        val dt = if (last == Long.MIN_VALUE) 0f else (nowMs - last).coerceIn(0L, 100L).toFloat()
        last = nowMs
        if (hasMask) {
            missing = 0f
            value = (value + dt / fadeInMs).coerceAtMost(1f)
        } else {
            missing += dt
            if (missing > holdMs) value = (value - dt / fadeOutMs).coerceAtLeast(0f)
        }
        return value
    }

    companion object {
        const val FADE_IN_MS = 200f
        const val HOLD_MS = 100f
        const val FADE_OUT_MS = 200f
    }
}
