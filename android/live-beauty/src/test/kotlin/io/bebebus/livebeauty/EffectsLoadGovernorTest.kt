package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class EffectsLoadGovernorTest {
    private val period = 33_333_333L // 30 fps 相机
    private val ms = 1_000_000L
    /** 一个统计窗口（毫秒）：30 fps、按整毫秒计时，每 30 帧恰好 1000ms。 */
    private val window = EffectsLoadGovernor.WINDOW_MS.toDouble()

    /**
     * 模拟 [seconds] 秒：相机每 [cameraPeriod] 出一帧，[rendered] 决定第 i 个相机帧是否被渲染，[work] 为该帧 draw 耗时（纳秒）。
     * @return 期间所有级别变化（新级别, 发生时刻 ms）。
     */
    private fun EffectsLoadGovernor.run(
        seconds: Int,
        start: Long = 0L,
        cameraPeriod: Long = period,
        processing: Boolean = true,
        work: (Int) -> Long = { 18 * ms },
        rendered: (Int) -> Boolean = { true },
    ): List<Pair<Int, Long>> {
        val changes = mutableListOf<Pair<Int, Long>>()
        val frames = (seconds * 1_000_000_000L / cameraPeriod).toInt()
        for (i in 1..frames) {
            if (!rendered(i)) continue
            val stamp = start + i * cameraPeriod
            onFrame(stamp, work(i), processing, stamp / 1_000_000)?.let { changes += it to stamp / 1_000_000 }
        }
        return changes
    }

    private fun List<Pair<Int, Long>>.levels() = map { it.first }

    @Test fun steadyLoadNeverChangesLevel() {
        val g = EffectsLoadGovernor()
        assertEquals(emptyList<Int>(), g.run(120).levels())
        assertEquals(0, g.level)
    }

    @Test fun lowLightCameraAtFifteenFpsIsNotOverload() {
        // 暗光：相机本身降到 15 fps，但每帧都渲染了，耗时正常。
        val g = EffectsLoadGovernor()
        assertEquals(emptyList<Int>(), g.run(120, cameraPeriod = 66_666_666L).levels())
    }

    @Test fun overloadBetweenTwentyFiveAndThirtyFpsIsDetected() {
        // 每 10 帧丢 1 帧（27 fps）、每帧 31ms：旧判据（掉帧 > 20% 且 < 25 fps）会漏掉。
        val g = EffectsLoadGovernor()
        val changes = g.run(4, work = { 31 * ms }) { it % 10 != 0 }
        assertEquals(listOf(1), changes.levels())
        assertTrue(changes[0].second in 2_900L..3_200L)
    }

    @Test fun sustainedOverloadStepsUpWithSettleWindows() {
        val g = EffectsLoadGovernor()
        val changes = g.run(20, work = { 32 * ms })
        assertEquals(listOf(1, 2, 3), changes.levels())
        // 3 个过载窗口升级，之后 2 个观察窗口 + 3 个过载窗口再升一级。
        for (i in 1 until changes.size) {
            assertEquals(5 * window, (changes[i].second - changes[i - 1].second).toDouble(), 50.0)
        }
        // 最高 3 级。
        assertEquals(emptyList<Int>(), g.run(20, start = 20_000_000_000L, work = { 32 * ms }).levels())
        assertEquals(3, g.level)
    }

    @Test fun briefSpikeDoesNotReduce() {
        val g = EffectsLoadGovernor()
        // 过载只持续约 2 个窗口，之后恢复。
        assertEquals(emptyList<Int>(), g.run(30, work = { i -> if (i in 31..90) 40 * ms else 18 * ms }).levels())
        assertEquals(0, g.level)
    }

    @Test fun recoversStepByStepAfterLoadDrops() {
        val g = EffectsLoadGovernor()
        val up = g.run(10, work = { 32 * ms })
        assertEquals(listOf(1, 2), up.levels())
        val down = g.run(40, start = 10_000_000_000L, work = { 10 * ms })
        assertEquals(listOf(1, 0), down.levels())
        // 第一次降级：升到 2 之后 2 个观察窗口 + 10 个平稳窗口。
        assertEquals(up[1].second + 12 * window, down[0].second.toDouble(), 50.0)
        // 再降一级：2 个观察窗口 + 10 个平稳窗口。
        assertEquals(down[0].second + 12 * window, down[1].second.toDouble(), 50.0)
        assertEquals(0, g.level)
    }

    @Test fun failedProbesBackOffExponentially() {
        // 负载正落在「1 级平稳、0 级过载」：每次试探恢复都失败，等待依次翻倍。
        val g = EffectsLoadGovernor()
        val changes = g.run(400, work = { if (g.level == 0) 29 * ms else 13 * ms })
        val downs = changes.filter { it.first == 0 }.map { it.second }
        assertTrue(downs.size >= 4)
        val gaps = (1 until 4).map { (downs[it] - downs[it - 1]) / window }
        // 相邻两次降级之间 = 2 观察 + 3 过载 + 2 观察 + recover（20、40、80 个窗口）。
        assertEquals(27.0, gaps[0], 0.2)
        assertEquals(47.0, gaps[1], 0.2)
        assertEquals(87.0, gaps[2], 0.2)
    }

    @Test fun measuredSavingIsCappedSoRecoveryStillHappens() {
        // 0 级 60ms、1 级 12ms：实测省下 48ms，钳到 10ms 后预测回到 0 级为 22ms（平稳），仍会试探恢复。
        val g = EffectsLoadGovernor()
        val changes = g.run(30, work = { if (g.level == 0) 60 * ms else 12 * ms })
        assertEquals(listOf(1, 0), changes.levels().take(2))
    }

    @Test fun pausedProcessingRestartsWindowsAndIdleResetsLevel() {
        val g = EffectsLoadGovernor()
        assertEquals(listOf(1), g.run(4, work = { 32 * ms }).levels())
        // 美颜关闭期间重新计窗、保留级别。
        assertEquals(emptyList<Int>(), g.run(10, start = 4_000_000_000L, processing = false).levels())
        assertEquals(1, g.level)
        // 两个过载窗口后关闭，再开：之前的过载窗口不累计。
        assertEquals(emptyList<Int>(), g.run(2, start = 14_000_000_000L, work = { 32 * ms }).levels())
        g.run(1, start = 16_000_000_000L, processing = false)
        // 观察期已过：重新开始的两个窗口不够升级。
        assertEquals(emptyList<Int>(), g.run(2, start = 17_000_000_000L, work = { 32 * ms }).levels())
        // 连续 30 s 没有处理帧：归零并返回 0。
        val idle = g.run(31, start = 19_000_000_000L, processing = false)
        assertEquals(listOf(0), idle.levels())
        assertEquals(0, g.level)
    }

    @Test fun timestampJumpRestartsTheWindow() {
        val g = EffectsLoadGovernor()
        g.run(1) { true }
        // 切后台后时间戳跳跃：重新开始，不把空档算成掉帧。
        assertEquals(emptyList<Int>(), g.run(5, start = 60_000_000_000L).levels())
        assertEquals(0, g.level)
    }

    @Test fun autoExposureJitterIsNotCountedAsDrops() {
        // 暗光自动曝光：间隔在 40–66 ms 之间变化（1.2–2 倍周期之间的非整数倍），每帧都渲染了。
        val g = EffectsLoadGovernor()
        var stamp = 0L
        val pattern = longArrayOf(40, 48, 55, 62, 45, 52, 58, 41)
        for (i in 0 until 400) {
            stamp += pattern[i % pattern.size] * 1_000_000L
            assertNull(g.onFrame(stamp, 18 * ms, true, stamp / 1_000_000))
        }
        assertEquals(0, g.level)
    }

    @Test fun levelStaysWithinContractRange() {
        val g = EffectsLoadGovernor()
        val changes = g.run(300, work = { i -> if ((i / 600) % 2 == 0) 40 * ms else 8 * ms })
        assertTrue(changes.isNotEmpty())
        assertTrue(changes.all { it.first in 0..EffectsLoadGovernor.MAX_LEVEL })
        g.reset()
        assertEquals(0, g.level)
        assertNull(g.onFrame(0L, 0L, true, 0L))
    }

    @Test fun fadesMoveLinearlyOverFourHundredMilliseconds() {
        val fades = LoadFades()
        fades.update(0, 0)
        // 0 → 2 级：美体与人脸 400ms 线性渐出，磨皮保持。
        for (t in 1..4) {
            fades.update(2, t * 100L)
            assertEquals(1f - t * 0.25f, fades.body, 1e-6f)
            assertEquals(1f - t * 0.25f, fades.face, 1e-6f)
            assertEquals(1f, fades.smooth, 0f)
        }
        // 恢复到 0 级：线性回到 1。
        fades.update(0, 500)
        assertEquals(0.25f, fades.face, 1e-6f)
        for (t in 6..8) fades.update(0, t * 100L)
        assertEquals(1f, fades.body, 0f)
        assertEquals(1f, fades.face, 0f)
        // 单步最多推进 100ms：GL 线程卡了 1 s 也不一下跳到底。
        fades.update(3, 1_800)
        assertEquals(0.75f, fades.smooth, 1e-6f)
        fades.reset()
        assertEquals(1f, fades.body, 0f)
    }
}
