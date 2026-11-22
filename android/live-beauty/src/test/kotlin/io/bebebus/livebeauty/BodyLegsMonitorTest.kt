package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class BodyLegsMonitorTest {
    @Test fun hidesAfterOneSecondWithoutKneesAndRecoversWithDebounce() {
        val m = BodyLegsMonitor()
        assertNull(m.update(true, true, legsVisible = false, nowMs = 0))
        assertNull(m.update(true, true, legsVisible = false, nowMs = 900))
        assertEquals(true, m.update(true, true, legsVisible = false, nowMs = 1_000))
        // 站起来拍到腿：同样持续 1 秒才解除。
        assertNull(m.update(true, true, legsVisible = true, nowMs = 1_100))
        assertNull(m.update(true, true, legsVisible = true, nowMs = 1_500))
        assertEquals(false, m.update(true, true, legsVisible = true, nowMs = 2_100))
    }

    @Test fun briefFlickerAndLostBodyDoNotToggle() {
        val m = BodyLegsMonitor()
        m.update(true, true, legsVisible = false, nowMs = 0)
        assertNull(m.update(true, true, legsVisible = true, nowMs = 500)) // 中断
        assertNull(m.update(true, true, legsVisible = false, nowMs = 600))
        assertNull(m.update(true, true, legsVisible = false, nowMs = 1_500))
        // 识别不到身体：维持现状，计时重来。
        assertNull(m.update(true, false, legsVisible = false, nowMs = 1_550))
        assertNull(m.update(true, true, legsVisible = false, nowMs = 1_600))
        assertEquals(true, m.update(true, true, legsVisible = false, nowMs = 2_600))
    }

    @Test fun turningLegsOffClearsImmediately() {
        val m = BodyLegsMonitor()
        m.update(true, true, false, 0); m.update(true, true, false, 1_000)
        assertTrue(m.hidden)
        assertEquals(false, m.update(false, true, false, 1_010))
        assertNull(m.update(false, true, false, 1_020))
    }
}
