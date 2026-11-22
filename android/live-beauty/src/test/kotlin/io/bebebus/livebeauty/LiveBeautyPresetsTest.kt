package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveBeautyPresetsTest {
    @Test
    fun builtinPresetsProduceValidConfigsAndRoundTripFromMap() {
        assertFalse(LiveBeautyPresets.original.enabled)
        for (preset in listOf(
            LiveBeautyPresets.naturalStudio,
            LiveBeautyPresets.clearColdWhite,
            LiveBeautyPresets.koreanWaterGlow,
            LiveBeautyPresets.mensClean,
        )) {
            assertTrue(preset.enabled)
            assertTrue(preset.smoothing in 0f..1f)
            assertTrue(preset.whiten in 0f..1f)
            assertTrue(VideoEffectsConfig.isValidLutUrl(preset.lutUrl))
        }
        assertTrue(LiveBeautyPresets.koreanWaterGlow.hasMakeup())
        assertTrue(LiveBeautyPresets.clearColdWhite.hasFaceReshape())
        assertEquals("asset://luts/natural_studio.png", LiveBeautyPresets.mensClean.lutUrl)
    }
}
