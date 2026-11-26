package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeBeautyBridgeTest {
    @Test
    fun packConfigScalarsMapsAll58Fields() {
        val cfg = LiveBeautyPresets.koreanWaterGlow.copy(
            cranialTop = 0.65f,
            nasolabial = 0.35f,
            relight = 0.60f,
            oilControl = 0.45f,
            neckTone = 0.55f,
            ambientMakeupBlend = 0.75f,
            matteTemporal = 0.80f,
            blemishRemove = 0.50f,
            hairDye = 0.68f,
            hairShine = 0.42f,
            bloom = 0.32f,
            filmGrain = 0.14f,
            colorTemp = -0.22f,
            colorTint = 0.11f,
            autoWhiteBalance = 0.78f,
        )
        val arr = NativeBeautyBridge.packConfigScalars(cfg)
        assertEquals(58, arr.size)
        assertEquals(cfg.smoothing, arr[0], 1e-6f)
        assertEquals(cfg.cranialTop, arr[26], 1e-6f)
        assertEquals(cfg.nasolabial, arr[30], 1e-6f)
        assertEquals(cfg.lipstick, arr[35], 1e-6f)
        assertEquals(cfg.eyeshadow, arr[38], 1e-6f)
        assertEquals(cfg.catchlight, arr[44], 1e-6f)
        assertEquals(cfg.relight, arr[45], 1e-6f)
        assertEquals(cfg.oilControl, arr[46], 1e-6f)
        assertEquals(cfg.neckTone, arr[47], 1e-6f)
        assertEquals(cfg.ambientMakeupBlend, arr[48], 1e-6f)
        assertEquals(cfg.matteTemporal, arr[49], 1e-6f)
        assertEquals(cfg.blemishRemove, arr[50], 1e-6f)
        assertEquals(cfg.hairDye, arr[51], 1e-6f)
        assertEquals(cfg.hairShine, arr[52], 1e-6f)
        assertEquals(cfg.bloom, arr[53], 1e-6f)
        assertEquals(cfg.filmGrain, arr[54], 1e-6f)
        assertEquals(cfg.colorTemp, arr[55], 1e-6f)
        assertEquals(cfg.colorTint, arr[56], 1e-6f)
        assertEquals(cfg.autoWhiteBalance, arr[57], 1e-6f)
    }

    @Test
    fun parseAndStatsSupportV3ProFields() {
        val parsed = VideoEffectsConfig.parse(
            mapOf(
                "enabled" to true,
                "smoothing" to 0.5,
                "brightness" to 0.1,
                "filter" to "none",
                "relight" to 0.65,
                "relightStyle" to "studio_softbox",
                "oilControl" to 0.40,
                "neckTone" to 0.55,
                "ambientMakeupBlend" to 0.70,
                "matteTemporal" to 0.75,
                "blemishRemove" to 0.50,
                "performanceTier" to "ultra",
                "hairDye" to 0.72,
                "hairDyeColor" to "rose_gold",
                "hairShine" to 0.55,
            ),
        )
        assertEquals(0.65f, parsed.relight, 1e-6f)
        assertEquals("studio_softbox", parsed.relightStyle)
        assertEquals(0.40f, parsed.oilControl, 1e-6f)
        assertEquals(0.55f, parsed.neckTone, 1e-6f)
        assertEquals(0.70f, parsed.ambientMakeupBlend, 1e-6f)
        assertEquals(0.75f, parsed.matteTemporal, 1e-6f)
        assertEquals(0.50f, parsed.blemishRemove, 1e-6f)
        assertEquals("ultra", parsed.performanceTier)
        assertEquals(0.72f, parsed.hairDye, 1e-6f)
        assertEquals("rose_gold", parsed.hairDyeColor)
        assertEquals(0.55f, parsed.hairShine, 1e-6f)

        val rawStats = floatArrayOf(4.2f, 4.5f, 30f, 0f, 1f, 0.8f, 123f, 42f, 2f, 1000f, 3f)
        val stats = BeautyPerfStats.fromRaw(rawStats)
        assertEquals(3, stats.resolvedTier)
        assertEquals(2, stats.activeFaces)
    }

    @Test
    fun v3612FaceTrackingEliminatesMediaPipeDoubleSmoothingAndBeatStall() {
        assertEquals(2, LiveBeautyCppEngine.resolveFaceDetectorNumFaces(1))
        assertEquals(2, LiveBeautyCppEngine.resolveFaceDetectorNumFaces(2))
        assertEquals(1L, LiveBeautyCppEngine.resolveFaceSampleIntervalMs(gpu = true, deferred = false, loadLevel = 0))
        assertEquals(1L, LiveBeautyCppEngine.resolveFaceSampleIntervalMs(gpu = false, deferred = true, loadLevel = 1))
        assertEquals(1L, LiveBeautyCppEngine.resolveFaceSampleIntervalMs(gpu = false, deferred = false, loadLevel = 0))
        assertEquals(45L, LiveBeautyCppEngine.resolveFaceSampleIntervalMs(gpu = false, deferred = false, loadLevel = 1))
        assertEquals(
            1,
            LiveBeautyCppEngine.selectPrimaryFaceIndex(
                floatArrayOf(0.50f, 0.54f), floatArrayOf(0.46f, 0.47f), floatArrayOf(0.36f, 0.36f),
                2, 0.50f, 0.46f, true,
            ),
        )
        assertEquals(
            1,
            LiveBeautyCppEngine.selectPrimaryFaceIndex(
                floatArrayOf(0.18f, 0.72f), floatArrayOf(0.45f, 0.50f), floatArrayOf(0.28f, 0.30f),
                2, 0.70f, 0.50f, true,
            ),
        )
    }
}
