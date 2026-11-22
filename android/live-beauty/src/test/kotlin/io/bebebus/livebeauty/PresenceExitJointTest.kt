package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import java.util.Random
import kotlin.math.ln

/**
 * 「遮挡退出 × 出帧锁定」联合测试（纯 Kotlin，不经 GL）。按 VideoEffectsFilter.drawFilter 第 1 段（取票、快照、按采样时间
 * 记有脸 / 无脸）与第 1d 段（按人脸完成时间选帧）的数据流，把真实的 FaceTrackingState（单任务在途、限频、250 ms 新鲜度）、
 * FaceTimeline（presence 按被画那一帧的采样时间计算）与 DelaySchedule / PacingLock（p90 锁定）串起来；
 * 推理按 GPU 延迟分布在采样之后完成。统计的是被显示帧（displayTime = 选中帧的采样时间）上的 presence。
 *
 * 遮挡从相机帧 T 起（之后采样到的帧都没有脸）。一次遮挡的退出时间 = 仍画着效果（presence > 0）的最后一个被显示帧
 * 的采样时间 − T：≤ 80 ms 表示内容时间晚于 T + 80 ms 的帧都不再画美妆与形变。
 */
class PresenceExitJointTest {
    /** 一次遮挡的统计。 */
    internal class Exit(
        /** 仍画着效果的最后一个被显示帧的内容滞后（ms），没有为 0。 */
        val lastShownMs: Long,
        /** 内容滞后在 [EXIT_MS, LATE_WINDOW_MS] 内的被显示帧数，以及其中仍画着效果的帧数。 */
        val lateFrames: Int,
        val lateShown: Int,
        /** 遮挡前 300 ms 内被显示帧的最小 presence（脸确实在，不应提前退出）。 */
        val beforeMin: Float,
        val lockedDelay: Int,
        /** 处于延迟回读（DEFERRED）的帧数，以及其间交给 worker 的人脸样本数。 */
        val deferredFrames: Int = 0,
        val deferredSubmits: Int = 0,
    )

    /**
     * VideoEffectsFilter 的人脸跟踪与画面延后（只保留与 presence、选帧有关的 GL 线程状态）。
     * 第 k 帧的渲染时间 = 采样时间 = [frameTime]；推理在交给 worker 后 [latency] ms 完成，结果里有没有脸由 [faceAt]（采样时间）决定。
     * [stallNs] ≥ 0 时每次回读按这个阻塞时长喂给 PacingLock：超过 STALL_DEFERRED_NS 时锁定后改为延迟回读（DEFERRED）：
     * 本帧只判到期（due）并画小图，下一帧开头再以本帧的采样时间取票（begin 带 expectEpoch），取到才读回并交给 worker
     * （VideoEffectsFilter.collectDeferred）。改动 VideoEffectsFilter 第 0'、1、1'、1d 段时请同步修改这里。
     */
    internal class Rig(
        private val rng: Random,
        private val latency: (Random) -> Long,
        private val intervalMs: Long,
        private val streaming: Boolean,
        private val stallNs: Long = -1L,
        private val faceAt: (Long) -> Boolean,
    ) {
        private val state = FaceTrackingState().apply { enable(true) }
        val timeline = FaceTimeline()
        val schedule = DelaySchedule()
        private var pending: FaceTrackingState.Ticket? = null
        private var pendingDone = 0L
        private var pendingFace = false
        private var epoch = Long.MIN_VALUE
        private var lastGeometry: FaceGeometry? = null
        private var lastCompletedSeen = Long.MIN_VALUE
        // 回读模式（上一帧第 1d 段 setDeferred 的结果）与延迟回读还没取票的记录（采样时间、到期时的代际）。
        private var deferred = false
        private var stagedAt = Long.MIN_VALUE
        private var stagedEpoch: Long? = null
        var k = 0
            private set
        var displayTime = 0L
            private set
        var presence = 0f
            private set
        var deferredFrames = 0
            private set
        var deferredSubmits = 0
            private set

        private fun submit(ticket: FaceTrackingState.Ticket, now: Long) {
            pending = ticket; pendingDone = now + latency(rng); pendingFace = faceAt(ticket.timestamp)
        }

        fun frame() {
            val now = frameTime(k)
            // worker：渲染时刻之前已经完成的推理。
            pending?.let { ticket ->
                if (pendingDone <= now) {
                    state.complete(ticket, if (pendingFace) GEOMETRY.copy() else null, pendingDone, if (pendingFace) MESH else null)
                    pending = null
                }
            }
            // 第 0' 段（collectDeferred）：为上一帧画好的小图取票（采样时间仍是上一帧），取到才读回并交给 worker。
            var read = false
            if (deferred) {
                deferredFrames++
                stagedEpoch?.let { e -> state.begin(stagedAt, WIDTH, HEIGHT, intervalMs, expectEpoch = e) }?.let {
                    submit(it, now); read = true; deferredSubmits++
                }
                stagedEpoch = null
            }
            // 第 1 段（与 VideoEffectsFilter 逐行对应）：DEFERRED 只判到期，SYNC 取票。
            val ticket = if (deferred) {
                stagedEpoch = state.due(now, WIDTH, HEIGHT, intervalMs); stagedAt = now; null
            } else {
                state.begin(now, WIDTH, HEIGHT, intervalMs)
            }
            val snapshot = state.snapshot(now)
            if (snapshot.epoch != epoch) {
                epoch = snapshot.epoch
                timeline.reset()
                timeline.aspect = ASPECT
                lastCompletedSeen = Long.MIN_VALUE
            }
            snapshot.geometry?.let {
                if (it !== lastGeometry) {
                    lastGeometry = it
                    timeline.add(snapshot.sampleTimestamp, it.toState(), snapshot.mesh)
                }
            }
            if (snapshot.completed != Long.MIN_VALUE && snapshot.completed > lastCompletedSeen) {
                lastCompletedSeen = snapshot.completed
                if (snapshot.geometry == null || snapshot.sampleTimestamp < snapshot.completed) timeline.addMiss(snapshot.completed)
            }
            // 第 1' 段：SYNC 当帧读回交给 worker；DEFERRED 只画小图（上面已记下采样时间）。
            ticket?.let { submit(it, now); read = true }
            // 第 1d 段：会话首次跟踪按 GPU 先验接入；本帧有回读才有阻塞样本（takeStallNs）；FrameDelay.push 的选帧部分；
            // setDeferred 按锁定结果切换回读模式（切换时丢掉没取票的记录）；再按 displayTime 取几何。
            if (k == 0) schedule.lock.engage(PacingLock.PRIOR_GPU)
            if (read) schedule.lock.onStall(stallNs)
            timeline.update(now)
            schedule.stored(schedule.next(), now, STAMP0_NS + k * FRAME_NS)
            schedule.select(snapshot.completed, streaming, now)
            if (schedule.lock.deferredReadback != deferred) { deferred = schedule.lock.deferredReadback; stagedEpoch = null }
            displayTime = schedule.time
            val geometry = timeline.at(displayTime)
            timeline.meshAt(displayTime)
            presence = if (geometry != null) timeline.presence else 0f
            k++
        }
    }

    /** 一次遮挡：先跑 [WARMUP_FRAMES] 帧让 PacingLock 锁定，再在随机相位遮挡。 */
    internal fun occlusion(seed: Long, streaming: Boolean, intervalMs: Long = FaceTrackingState.FRAME_INTERVAL_MS,
                           stallNs: Long = -1L, latency: (Random) -> Long = ::gpuLatency): Exit {
        val rng = Random(seed)
        val occludeFrame = WARMUP_FRAMES + rng.nextInt(30)
        val occludedAt = frameTime(occludeFrame)
        val rig = Rig(rng, latency, intervalMs, streaming, stallNs) { ts -> ts < occludedAt }
        var lastShown = 0L
        var late = 0
        var lateShown = 0
        var beforeMin = 1f
        while (rig.k < occludeFrame + 30) {
            rig.frame()
            val lag = rig.displayTime - occludedAt
            if (lag >= 0 && rig.presence > 0f) lastShown = maxOf(lastShown, lag)
            if (lag in EXIT_MS..LATE_WINDOW_MS) { late++; if (rig.presence > 0f) lateShown++ }
            if (lag in -300L until 0L) beforeMin = minOf(beforeMin, rig.presence)
        }
        return Exit(lastShown, late, lateShown, beforeMin, rig.schedule.lock.delay, rig.deferredFrames, rig.deferredSubmits)
    }

    /** 稳定有脸时只有一个采样结果没有脸（[missFrame] 之后的第一个采样）：返回渐入完成后被显示帧的最小 presence。 */
    internal fun singleMiss(seed: Long, streaming: Boolean, intervalMs: Long, stallNs: Long = -1L,
                            latency: (Random) -> Long = ::gpuLatency): Float {
        val rng = Random(seed)
        val missFrame = WARMUP_FRAMES + rng.nextInt(30)
        val missFrom = frameTime(missFrame)
        var missAt = Long.MIN_VALUE
        val rig = Rig(rng, latency, intervalMs, streaming, stallNs) { ts ->
            if (missAt == Long.MIN_VALUE && ts >= missFrom) { missAt = ts; false } else true
        }
        var min = 1f
        while (rig.k < missFrame + 60) {
            rig.frame()
            if (rig.k > FADE_IN_FRAMES) min = minOf(min, rig.presence)
        }
        assertNotEquals("应当发生过一次漏检", Long.MIN_VALUE, missAt)
        return min
    }

    private fun summary(exits: List<Exit>): String {
        val lags = exits.map { it.lastShownMs }.sorted()
        val p = { q: Double -> lags[((lags.size - 1) * q).toInt()] }
        return "退出 p50/p90/max = ${p(0.5)}/${p(0.9)}/${lags.last()} ms，锁定延后 ${exits.map { it.lockedDelay }.distinct()}，" +
            "晚于 $EXIT_MS ms 仍画着的帧 ${exits.sumOf { it.lateShown }}/${exits.sumOf { it.lateFrames }}"
    }

    private fun assertExits(exits: List<Exit>) {
        val text = summary(exits)
        // 遮挡之前脸确实在：不提前退出、不闪。
        assertEquals(text, 1f, exits.minOf { it.beforeMin }, 0f)
        val withinShare = exits.count { it.lastShownMs <= EXIT_MS }.toDouble() / exits.size
        assertTrue("≤ $EXIT_MS ms 退出的遮挡占 $withinShare，应 ≥ 0.9；$text", withinShare >= 0.9)
        val lateShare = exits.sumOf { it.lateShown }.toDouble() / exits.sumOf { it.lateFrames }
        assertTrue("内容晚于遮挡 $EXIT_MS ms 仍画着效果的帧占 $lateShare，应 ≤ 0.1；$text", lateShare <= 0.1)
        // 检测器忙的帧最长按 BUSY_HOLD_MAX 保持再渐出，不会无限冻结。
        assertTrue(text, exits.maxOf { it.lastShownMs } < FaceTimeline.BUSY_HOLD_MAX_MS + FaceTimeline.EXIT_RAMP_MS)
    }

    @Test fun occlusionExitsWithin80msAt30HzInPreviewLock() {
        // 预览：p90 锁定到测得的延后（GPU 推理多半下一帧就到 → 1 帧；慢的多时 2 帧），部分帧落在最新完成样本之后。
        val exits = (0 until TRIALS).map { occlusion(seed = 1000L + it, streaming = false) }
        assertTrue(summary(exits), exits.all { it.lockedDelay in 1..2 })
        assertExits(exits)
    }

    @Test fun occlusionExitsWithin80msAt30HzWhileStreaming() {
        // 推流中：延后取 max(先验 2, 测得)。
        val exits = (0 until TRIALS).map { occlusion(seed = 2000L + it, streaming = true) }
        assertTrue(summary(exits), exits.all { it.lockedDelay >= PacingLock.PRIOR_GPU })
        assertExits(exits)
    }

    @Test fun singleMissDoesNotFlickerAt15Hz() {
        // 负载 1 级：人脸按 FACE_THROTTLED_INTERVAL_MS（约 15 Hz）采样；防抖 = 1.2 × 采样间隔，单次漏检 presence 不低于 0.9。
        for (streaming in listOf(false, true)) {
            val worst = (0 until TRIALS).minOf { singleMiss(3000L + it, streaming, THROTTLED_INTERVAL_MS) }
            assertTrue("15 Hz 单次漏检最低 presence $worst（streaming=$streaming）应 ≥ 0.9", worst >= 0.9f)
        }
    }

    @Test fun singleMissDoesNotFlickerAt30HzWhenInferenceKeepsUp() {
        // 推理都在一帧内完成（每帧都采）时，单次漏检的下一个有脸样本在防抖（约 40 ms）之内到达，不下陷。
        // 推理超过一帧、下一帧跳采的情形由 FaceTimeline.redetect 的跳采豁免兜住，见 singleMissDoesNotFlickerAt30HzWithGpuLatency。
        val keepsUp = { rng: Random -> 14L + rng.nextInt(15) }
        for (streaming in listOf(false, true)) {
            val worst = (0 until TRIALS).minOf { singleMiss(4000L + it, streaming, FaceTrackingState.FRAME_INTERVAL_MS, latency = keepsUp) }
            assertTrue("30 Hz 单次漏检最低 presence $worst（streaming=$streaming）应 ≥ 0.9", worst >= 0.9f)
        }
    }

    @Test fun singleMissDoesNotFlickerAt30HzWithGpuLatency() {
        // 真实 GPU 延迟分布（约 6% 超过一帧）：漏检之后下一帧可能因推理在途而跳采，无脸段仍在防抖 + 渐出之内，
        // 由 FaceTimeline.redetect 的跳采豁免改记为有脸，不淡出再淡入。断言下陷（presence < 0.9）试次的占比而不是最小值：
        // 预览下连续两次慢推理时帧已画到渐出，之后从当前水平连续渐入是合理的。
        for (streaming in listOf(false, true)) {
            val mins = (0 until TRIALS).map { singleMiss(4500L + it, streaming, FaceTrackingState.FRAME_INTERVAL_MS) }
            val dipShare = mins.count { it < 0.9f }.toDouble() / mins.size
            assertTrue("30 Hz 单次漏检 presence < 0.9 的试次占 $dipShare（streaming=$streaming），应 ≤ 0.01", dipShare <= 0.01)
        }
    }

    @Test fun throttledOcclusionStillExitsBoundedAt15Hz() {
        // 负载 1 级（R6 的已知代价）：55 ms 节流在 30 fps 下量化为每 2 帧一采（约 67 ms），防抖 1.2 × 67 ≈ 80 ms + 渐出 40 ms，
        // 从第一个无脸样本起约 120 ms；按遮挡起点计 100–134 ms（含至多一帧的采样相位）。上界 150 ms，不冻结。
        val exits = (0 until TRIALS).map { occlusion(seed = 5000L + it, streaming = true, intervalMs = THROTTLED_INTERVAL_MS) }
        val text = summary(exits)
        assertEquals(text, 1f, exits.minOf { it.beforeMin }, 0f)
        assertTrue(text, exits.maxOf { it.lastShownMs } <= THROTTLED_EXIT_MAX_MS)
    }

    @Test fun deferredReadbackStillExitsWithin80msAt30Hz() {
        // 回读阻塞 p90 > 3 ms 的机型锁定后改为延迟回读：画小图那一帧只判到期，下一帧读回时才取票，单任务在途只覆盖推理本身，
        // GPU 人脸仍约 30 Hz 采样（代价只是多一帧画面延后），退出口径与 SYNC 相同。
        val exits = (0 until TRIALS).map { occlusion(seed = 6000L + it, streaming = true, stallNs = 4_000_000L) }
        val text = summary(exits)
        assertTrue(text, exits.all { it.lockedDelay >= PacingLock.PRIOR_GPU && it.deferredFrames > 0 })
        assertExits(exits)
        val rateHz = exits.sumOf { it.deferredSubmits } * 30.0 / exits.sumOf { it.deferredFrames } // 相机 30 fps
        assertTrue("DEFERRED 下人脸提交频率 $rateHz Hz，应 ≥ 25；$text", rateHz >= 25.0)
    }

    internal companion object {
        const val WIDTH = 720
        const val HEIGHT = 1280
        const val ASPECT = WIDTH.toFloat() / HEIGHT
        const val FRAME_NS = 33_333_333L
        const val STAMP0_NS = 1_000_000_000L
        /** 约 4 s：PacingLock 预热 15 + 测量 45 个样本后锁定，采样间隔 EMA 收敛。 */
        const val WARMUP_FRAMES = 120
        const val FADE_IN_FRAMES = 30
        const val TRIALS = 200
        const val EXIT_MS = 80L
        const val LATE_WINDOW_MS = 200L
        /** 与 VideoEffectsFilter.FACE_THROTTLED_INTERVAL_MS 一致（负载 1 级起的人脸采样间隔）。 */
        const val THROTTLED_INTERVAL_MS = VideoEffectsFilter.FACE_THROTTLED_INTERVAL_MS
        /** 负载 1 级遮挡退出的上界：相位 33 + 防抖 1.2 × 66.7 + 渐出 40 ≈ 153，按帧量化后最大 134。 */
        const val THROTTLED_EXIT_MAX_MS = 150L

        /** 30 fps 相机帧的渲染时间（uptime ms，33 / 34 交替）。 */
        fun frameTime(k: Int): Long = k * 100L / 3

        /**
         * GPU 推理（含共享小图回读、提交与线程调度）的延迟分布：16 ms + 指数分布（均值 6 ms），
         * 中位数约 20 ms、p90 约 30 ms、约 6% 超过一帧（33 ms，此时下一帧不采）。
         */
        fun gpuLatency(rng: Random): Long = 16L + (-6.0 * ln(1.0 - rng.nextDouble())).toLong()

        private val POINTS = FaceTestFixtures.render()
        val GEOMETRY: FaceGeometry = checkNotNull(FaceGeometry.fromMesh(FaceTestFixtures.toMediaPipe(POINTS), ASPECT))
        val MESH: FaceMeshFrame = FaceTestFixtures.frame(POINTS)
    }
}
