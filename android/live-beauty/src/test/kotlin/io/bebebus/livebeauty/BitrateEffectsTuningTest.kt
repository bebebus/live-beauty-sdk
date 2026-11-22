package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class BitrateEffectsTuningTest {
    @Test fun pressureIsZeroNearCapAndFullNearFloor() {
        assertEquals(0f, BitrateEffectsTuning.pressure(1f), 0f)
        assertEquals(0f, BitrateEffectsTuning.pressure(0.75f), 1e-6f)
        assertEquals(0.5f, BitrateEffectsTuning.pressure(0.5f), 1e-6f)
        assertEquals(1f, BitrateEffectsTuning.pressure(0.25f), 1e-6f)
        // 300k / 2000k 下限。
        assertEquals(1f, BitrateEffectsTuning.pressure(0.15f), 0f)
        assertEquals(0f, BitrateEffectsTuning.pressure(Float.NaN), 0f)
    }

    @Test fun lessSharpeningAndSomeDenoiseUnderPressure() {
        assertEquals(0.5f, BitrateEffectsTuning.sharpness(0.5f, 0f), 0f)
        assertEquals(0.2f, BitrateEffectsTuning.sharpness(0.5f, 1f), 1e-6f)
        assertEquals(0f, BitrateEffectsTuning.sharpness(0f, 1f), 0f)
        assertEquals(0f, BitrateEffectsTuning.denoise(0f), 0f)
        val full = BitrateEffectsTuning.denoise(1f)
        assertTrue(full > 0f && full <= TemporalDenoisePass.MAX_WEIGHT)
        assertEquals(full, BitrateEffectsTuning.denoise(3f), 0f)
    }

    @Test fun textureParamHalvesUnderFullPressure() {
        assertEquals(0.6f, BitrateEffectsTuning.texture(0.6f, 0f), 0f)
        assertEquals(0.45f, BitrateEffectsTuning.texture(0.6f, 0.5f), 1e-6f)
        assertEquals(0.3f, BitrateEffectsTuning.texture(0.6f, 1f), 1e-6f)
        assertEquals(0.3f, BitrateEffectsTuning.texture(0.6f, 3f), 1e-6f)
        assertEquals(0.6f, BitrateEffectsTuning.texture(0.6f, -1f), 0f)
        assertEquals(0f, BitrateEffectsTuning.texture(0f, 1f), 0f)
    }
}
