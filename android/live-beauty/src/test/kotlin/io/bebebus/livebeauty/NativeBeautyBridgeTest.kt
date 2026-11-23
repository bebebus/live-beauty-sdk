package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeBeautyBridgeTest {
    @Test
    fun packConfigScalarsMapsAll45Fields() {
        val cfg = LiveBeautyPresets.koreanWaterGlow.copy(cranialTop = 0.65f, nasolabial = 0.35f)
        val arr = NativeBeautyBridge.packConfigScalars(cfg)
        assertEquals(45, arr.size)
        assertEquals(cfg.smoothing, arr[0], 1e-6f)
        assertEquals(cfg.cranialTop, arr[26], 1e-6f)
        assertEquals(cfg.nasolabial, arr[30], 1e-6f)
        assertEquals(cfg.lipstick, arr[35], 1e-6f)
        assertEquals(cfg.eyeshadow, arr[38], 1e-6f)
        assertEquals(cfg.catchlight, arr[44], 1e-6f)
    }
}
