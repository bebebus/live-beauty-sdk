package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class TrackingGraceTest {
    @Test fun keepsTrackingForThreeSecondsAfterNoLongerWanted() {
        val grace = TrackingGrace()
        assertFalse(grace.keep(false, 0)) // 从未需要过
        assertTrue(grace.keep(true, 1_000))
        assertTrue(grace.keep(false, 1_000 + 2_999))
        assertFalse(grace.keep(false, 1_000 + 3_000))
    }

    @Test fun wantedAgainWithinGraceRestartsTheTimer() {
        val grace = TrackingGrace()
        grace.keep(true, 0)
        assertTrue(grace.keep(false, 2_000))
        assertTrue(grace.keep(true, 2_500)) // 参数临时拖到 0 又拖回来：不停 tracker
        assertTrue(grace.keep(false, 5_000))
        assertTrue(grace.keep(false, 5_499))
        assertFalse(grace.keep(false, 5_500))
    }

    @Test fun resetStopsImmediately() {
        val grace = TrackingGrace(graceMs = 3_000)
        grace.keep(true, 0)
        grace.reset()
        assertFalse(grace.keep(false, 1))
    }
}
