package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class TemporalDenoisePassTest {
    @Test fun strengthFollowsDarknessAndIsCapped() {
        val max = TemporalDenoisePass.MAX_WEIGHT
        assertEquals(max, TemporalDenoisePass.strengthForLuma(0.05f), 1e-6f)
        assertEquals(max, TemporalDenoisePass.strengthForLuma(0.08f), 1e-6f)
        assertEquals(max / 2, TemporalDenoisePass.strengthForLuma(0.13f), 1e-4f)
        assertEquals(0f, TemporalDenoisePass.strengthForLuma(0.18f), 1e-6f)
        assertEquals(0f, TemporalDenoisePass.strengthForLuma(0.42f), 1e-6f)
        assertEquals(0f, TemporalDenoisePass.strengthForLuma(0.8f), 1e-6f)
        // 尚未测到亮度（NaN）时不降噪。
        assertEquals(0f, TemporalDenoisePass.strengthForLuma(Float.NaN), 0f)
    }
}
