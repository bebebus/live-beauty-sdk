package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class FrameReuseGateTest {
    private val beauty = VideoEffectsConfig.original.copy(enabled = true, smoothing = 0.5f)

    private fun FrameReuseGate.can(stamp: Long, config: VideoEffectsConfig = beauty, input: Int = 7,
        width: Int = 720, height: Int = 1280, healthy: Boolean = true, face: Boolean = true) =
        canReuse(stamp, config, input, width, height, healthy, face)

    private fun FrameReuseGate.done(stamp: Long, config: VideoEffectsConfig = beauty, input: Int = 7,
        width: Int = 720, height: Int = 1280, healthy: Boolean = true, face: Boolean = true) =
        rendered(stamp, config, input, width, height, healthy, face)

    @Test fun secondDrawOfSameCameraFrameReusesResult() {
        val g = FrameReuseGate()
        assertFalse(g.can(100)) // 预览：首次必须绘制
        g.done(100)
        assertTrue(g.can(100)) // 编码：同帧沿用
        assertFalse(g.can(133)) // 新相机帧
    }

    @Test fun anyInputChangeForcesRender() {
        val g = FrameReuseGate()
        g.done(100)
        assertFalse(g.can(100, config = beauty.copy(smoothing = 0.6f)))
        assertTrue(g.can(100, config = beauty.copy())) // 值相同的新快照
        assertFalse(g.can(100, input = 8))
        assertFalse(g.can(100, width = 1280, height = 720))
        assertFalse(g.can(100, healthy = false))
        assertFalse(g.can(100, face = false))
    }

    @Test fun unknownTimestampNeverReuses() {
        val g = FrameReuseGate()
        g.done(0)
        assertFalse(g.can(0))
        g.done(100)
        g.done(0) // 取不到时间戳：作废之前的记录
        assertFalse(g.can(100))
    }

    @Test fun invalidateDropsRecord() {
        val g = FrameReuseGate()
        g.done(100)
        g.invalidate()
        assertFalse(g.can(100))
    }
}
