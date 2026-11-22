package io.bebebus.livebeauty

import kotlin.math.floor

/**
 * 美颜负载降级（仅 Android，纯逻辑，只在 GL 线程使用）。
 *
 * 每个新相机帧调用一次 [onFrame]，传入本帧 VideoEffectsFilter.draw 的墙钟耗时 work（含同步回读与后处理，
 * 不含编码 swap：eglSwapBuffers 被编码器背压阻塞时不能算成美颜过载）。按 [WINDOW_MS] 一个窗口统计，窗口内至少
 * [MIN_FRAMES] 个处理帧才计入：
 * - 过载（任一）：work 的 p95 > [OVER_P95_NS]；忙碌比 Σwork / 窗口时长 > [OVER_BUSY]；掉帧率 > [OVER_DROP] 且帧率 < [OVER_FPS]。
 *   掉帧率沿用旧算法：相机按固定周期出帧，渲染跟不上时相邻两次渲染的时间戳间隔是周期的整数倍；周期取间隔的 20 分位，
 *   暗光降帧时周期本身变长，不会被误判。
 * - 平稳（当前级别 L）：p95 + saved[L] < OVER_P95 − [HYSTERESIS_NS]、忙碌比 < [CALM_BUSY]、掉帧率 < [CALM_DROP]。
 *   saved[L] 是降到 L 实测省下的 p95（升级前 3 个过载窗口 p95 的最小值 − 升级后观察期结束再取 3 个窗口 p95 的中位数，
 *   钳到 [0, [SAVE_CAP_NS]]，测出之前按 SAVE_CAP）：预测回到上一级后仍不过载才恢复。
 * 连续 [UP_STRIKES] 个过载窗口升一级；连续 recover 个平稳窗口降一级；每次变化后 [SETTLE_WINDOWS] 个窗口只观察。
 * recover 从 [RECOVER_START] 开始；降级后 [PROBE_FAIL_MS] 内又升级（试探失败）就翻倍，上限 [RECOVER_MAX]；
 * [BACKOFF_RESET_MS] 内没有再升级则复位。美颜不在处理（关闭、对比原画）时重新计窗、保留级别；
 * 连续 [IDLE_RESET_MS] 没有处理帧时归零。时间戳倒退或停顿超过 0.5 s（切后台、相机重启）时重新计窗。
 * 只看 GL 线程上 draw 的墙钟：各 worker 上的推理（人脸 CPU 回退、人像分割、头发分割）占用的 CPU 不计入，也不接热状态。
 * 头发分割的 CPU 护栏由 VideoEffectsFilter 第 1c' 段自己管（HairMaskTuning）：间隔 max(100, ⌈2·p50⌉) ms（开背景虚化时下限
 * 200 ms，worker 占空不超过 1/2），p50 超过 MAX_P50_MS（125 ms）本次会话停用，画面里近期没有脸时也停。
 *
 * 级别（对外 0..3；上报服务端的契约仍是 1..3 的升级），各级效果由 [LoadFades] 以 400ms 渐变：
 * - 1 = 停美体、背景虚化与头发分割（头发遮罩随 fades.body 渐出，回到第一批启发式），人脸采样降到 15Hz；
 * - 2 = 网格微整、实时美妆、局部修饰与人脸遮罩渐出，渐出到 0 后再停人脸跟踪；
 * - 3 = 磨皮与时域降噪渐出。
 */
class EffectsLoadGovernor {
    var level = 0
        private set
    private val works = LongArray(WORK_SAMPLES)
    private var workCount = 0
    private var workSum = 0L
    private val deltas = LongArray(WORK_SAMPLES)
    private var deltaCount = 0
    private var lastStamp = 0L
    private var windowStart = Long.MIN_VALUE
    private var overStreak = 0
    private var overMinP95 = Long.MAX_VALUE
    private var calmStreak = 0
    private var settle = 0
    private var recover = RECOVER_START
    private var lastUpAt = Long.MIN_VALUE
    private var lastDownAt = Long.MIN_VALUE
    private var idleSince = Long.MIN_VALUE
    private val saved = longArrayOf(0L, SAVE_CAP_NS, SAVE_CAP_NS, SAVE_CAP_NS)
    private var pendingLevel = 0 // 刚升到的级别，等观察期后测 saved；0 表示没有待测
    private var pendingPre = 0L
    private val post = LongArray(3)
    private var postCount = 0

    fun reset() {
        level = 0; overStreak = 0; overMinP95 = Long.MAX_VALUE; calmStreak = 0; settle = 0
        recover = RECOVER_START; lastUpAt = Long.MIN_VALUE; lastDownAt = Long.MIN_VALUE; idleSince = Long.MIN_VALUE
        saved.fill(SAVE_CAP_NS); saved[0] = 0L; pendingLevel = 0; postCount = 0
        restartWindow()
    }

    private fun restartWindow() {
        workCount = 0; workSum = 0L; deltaCount = 0; lastStamp = 0L; windowStart = Long.MIN_VALUE
    }

    /**
     * 每个新相机帧（时间戳变化）调用一次。[stampNs] 为 SurfaceTexture 时间戳，[workNs] 为本帧 draw 的墙钟耗时，
     * [processing] 为上一帧是否在做美颜处理。
     * @return 级别有变化（升或降）时返回新级别，否则 null。
     */
    fun onFrame(stampNs: Long, workNs: Long, processing: Boolean, nowMs: Long): Int? {
        if (!processing || stampNs == 0L) {
            overStreak = 0; calmStreak = 0; restartWindow()
            if (idleSince == Long.MIN_VALUE) idleSince = nowMs
            if (level > 0 && nowMs - idleSince >= IDLE_RESET_MS) { reset(); return 0 }
            return null
        }
        idleSince = Long.MIN_VALUE
        if (stampNs == lastStamp) return null
        if (lastStamp != 0L) {
            val delta = stampNs - lastStamp
            // 时间戳倒退或停顿超过 0.5 s（切后台、相机重启）：重新开始统计。
            if (delta <= 0L || delta > 500_000_000L) {
                restartWindow(); lastStamp = stampNs; windowStart = nowMs; return null
            }
            deltas[deltaCount % deltas.size] = delta; deltaCount++
        }
        lastStamp = stampNs
        works[workCount % works.size] = workNs.coerceAtLeast(0L); workCount++
        workSum += workNs.coerceAtLeast(0L)
        if (windowStart == Long.MIN_VALUE) windowStart = nowMs
        val span = nowMs - windowStart
        if (span < WINDOW_MS) return null
        val enough = workCount >= MIN_FRAMES
        val p95 = percentile(works, minOf(workCount, works.size), 0.95)
        val busy = workSum.toDouble() / (span * 1_000_000.0)
        val drop = dropStats()
        workCount = 0; workSum = 0L; deltaCount = 0; windowStart = nowMs
        return if (enough) window(nowMs, p95, busy, drop) else null
    }

    private fun window(now: Long, p95: Long, busy: Double, drop: DropStats): Int? {
        if (settle > 0) { settle--; return null }
        if (pendingLevel > 0) {
            post[postCount++] = p95
            if (postCount == post.size) {
                post.sort()
                saved[pendingLevel] = (pendingPre - post[1]).coerceIn(0L, SAVE_CAP_NS)
                pendingLevel = 0; postCount = 0
            }
        }
        val over = p95 > OVER_P95_NS || busy > OVER_BUSY || (drop.ratio > OVER_DROP && drop.fps < OVER_FPS)
        if (over) { overStreak++; overMinP95 = minOf(overMinP95, p95) } else { overStreak = 0; overMinP95 = Long.MAX_VALUE }
        if (lastUpAt != Long.MIN_VALUE && now - lastUpAt > BACKOFF_RESET_MS) recover = RECOVER_START
        if (overStreak >= UP_STRIKES && level < MAX_LEVEL) {
            // 刚降级就又过载：试探失败，下次多等一倍再试。
            if (lastDownAt != Long.MIN_VALUE && now - lastDownAt < PROBE_FAIL_MS) recover = minOf(recover * 2, RECOVER_MAX)
            level++
            pendingLevel = level; pendingPre = overMinP95; postCount = 0
            overStreak = 0; overMinP95 = Long.MAX_VALUE; calmStreak = 0; settle = SETTLE_WINDOWS; lastUpAt = now
            return level
        }
        if (level > 0) {
            val calm = p95 + saved[level] < OVER_P95_NS - HYSTERESIS_NS && busy < CALM_BUSY && drop.ratio < CALM_DROP
            calmStreak = if (calm) calmStreak + 1 else 0
            if (calmStreak >= recover) {
                level--
                if (pendingLevel > level) { pendingLevel = 0; postCount = 0 }
                calmStreak = 0; overStreak = 0; overMinP95 = Long.MAX_VALUE; settle = SETTLE_WINDOWS; lastDownAt = now
                return level
            }
        }
        return null
    }

    private class DropStats(val ratio: Double, val fps: Double)

    private fun dropStats(): DropStats {
        val n = minOf(deltaCount, deltas.size)
        if (n < 8) return DropStats(0.0, Double.MAX_VALUE)
        val sorted = deltas.copyOf(n).also { it.sort() }
        val period = sorted[(n * 0.2).toInt()].coerceAtLeast(1L)
        var expected = 0
        var sum = 0L
        // 只有接近整数倍（≥ 1.75 个周期）才算丢帧：相机自动曝光变帧率时的 1.2–1.7 倍间隔不算。
        for (i in 0 until n) {
            expected += floor(deltas[i].toDouble() / period + 0.25).toInt().coerceAtLeast(1)
            sum += deltas[i]
        }
        return DropStats((expected - n).toDouble() / expected, n * 1e9 / sum)
    }

    private fun percentile(values: LongArray, count: Int, q: Double): Long {
        if (count == 0) return 0L
        val sorted = values.copyOf(count).also { it.sort() }
        return sorted[((count - 1) * q + 0.5).toInt().coerceAtMost(count - 1)]
    }

    companion object {
        const val WINDOW_MS = 1_000L
        const val MIN_FRAMES = 10
        const val MAX_LEVEL = 3
        const val OVER_P95_NS = 28_000_000L
        const val HYSTERESIS_NS = 4_000_000L
        const val OVER_BUSY = 0.80
        const val CALM_BUSY = 0.60
        const val OVER_DROP = 0.10
        const val OVER_FPS = 27.0
        const val CALM_DROP = 0.03
        const val UP_STRIKES = 3
        const val SETTLE_WINDOWS = 2
        const val RECOVER_START = 10
        const val RECOVER_MAX = 240
        const val PROBE_FAIL_MS = 60_000L
        const val BACKOFF_RESET_MS = 600_000L
        const val SAVE_CAP_NS = 10_000_000L
        const val IDLE_RESET_MS = 30_000L
        private const val WORK_SAMPLES = 64
    }
}

/**
 * 负载降级的渐变系数（仅 GL 线程）：[body]、[face]、[smooth] 以 1/[FADE_MS] 的速率线性趋向目标——
 * 级别 < 1 / < 2 / < 3 时为 1，否则为 0。VideoEffectsFilter 把它们乘到背景强度、美体存在度与头发遮罩强度、人脸存在度、
 * 磨皮与降噪强度上；对应 tracker 只在系数到 0 之后才停。
 */
internal class LoadFades {
    var body = 1f
        private set
    var face = 1f
        private set
    var smooth = 1f
        private set
    private var last = Long.MIN_VALUE

    fun reset() { body = 1f; face = 1f; smooth = 1f; last = Long.MIN_VALUE }

    fun update(level: Int, nowMs: Long) {
        val dt = if (last == Long.MIN_VALUE) 0f else (nowMs - last).coerceIn(0L, MAX_STEP_MS).toFloat()
        last = nowMs
        val step = dt / FADE_MS
        body = toward(body, if (level < 1) 1f else 0f, step)
        face = toward(face, if (level < 2) 1f else 0f, step)
        smooth = toward(smooth, if (level < 3) 1f else 0f, step)
    }

    private fun toward(value: Float, target: Float, step: Float) =
        if (value < target) minOf(value + step, target) else maxOf(value - step, target)

    companion object {
        const val FADE_MS = 400f
        /** 单步最多推进 100ms：GL 线程卡顿或切后台回来时不一下跳变。 */
        const val MAX_STEP_MS = 100L
    }
}
