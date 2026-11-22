package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class VideoEffectsConfigTest {
    private fun values() = mutableMapOf<String, Any>(
        "enabled" to true, "smoothing" to 0.25, "brightness" to 0.15, "filter" to "warm")
    @Test fun acceptsSupportedFiltersAndBoundaries() {
        for (style in listOf("none", "warm", "cool", "monochrome", "sepia", "rose")) {
            val values = values().apply { put("filter", style); put("smoothing", 0); put("brightness", 1) }
            val actual = VideoEffectsConfig.parse(values)
            assertEquals(style, actual.filter)
            assertEquals(0f, actual.smoothing)
            assertEquals(1f, actual.brightness)
        }
    }
    @Test fun rejectsMalformedAndNonFiniteInputs() {
        for ((key, value) in listOf(
            "enabled" to "true", "smoothing" to -0.1, "brightness" to 1.01,
            "smoothing" to Double.NaN, "brightness" to Double.POSITIVE_INFINITY,
            "filter" to "face_swap", "smoothing" to true
        )) {
            try {
                VideoEffectsConfig.parse(values().apply { put(key, value) })
                fail("应拒绝参数 $key=$value")
            } catch (_: IllegalArgumentException) {
            } catch (_: IllegalStateException) {
            }
        }
    }
    @Test fun rejectsMissingFieldsEvenWhenDisabled() {
        val values = values().apply { put("enabled", false); remove("brightness") }
        try { VideoEffectsConfig.parse(values); fail("应拒绝缺少字段") }
        catch (_: IllegalStateException) { }
    }
    @Test fun parsesBodyShapingAndRejectsOutOfRange() {
        val actual = VideoEffectsConfig.parse(values().apply {
            put("waistSlim", 0.4); put("bustEnhance", 0.3); put("legLengthen", 1)
        })
        assertEquals(0.4f, actual.waistSlim)
        assertEquals(0.3f, actual.bustEnhance)
        assertEquals(1f, actual.legLengthen)
        val legacy = VideoEffectsConfig.parse(values())
        assertEquals(0f, legacy.waistSlim)
        assertEquals(0f, legacy.bustEnhance)
        assertEquals(0f, legacy.legLengthen)
        val more = VideoEffectsConfig.parse(values().apply { put("shoulderSlim", 0.5); put("neckLengthen", 0.2) })
        assertEquals(0.5f, more.shoulderSlim)
        assertEquals(0.2f, more.neckLengthen)
        assertEquals(0f, legacy.shoulderSlim)
        assertEquals(0f, legacy.neckLengthen)
        assertEquals(0.7f, VideoEffectsConfig.parse(values().apply { put("backgroundBlur", 0.7) }).backgroundBlur)
        assertEquals(0f, legacy.backgroundBlur)
        val face = VideoEffectsConfig.parse(values().apply { put("chinShape", 0.3); put("foreheadShape", 0.4); put("noseSlim", 0.5) })
        assertEquals(0.3f, face.chinShape); assertEquals(0.4f, face.foreheadShape); assertEquals(0.5f, face.noseSlim)
        assertEquals(0f, legacy.chinShape)
        assertEquals(0.6f, VideoEffectsConfig.parse(values().apply { put("teethWhiten", 0.6) }).teethWhiten)
        assertEquals(0f, legacy.darkCircles); assertEquals(0f, legacy.brightEyes)
        for (key in listOf("waistSlim", "bustEnhance", "legLengthen", "shoulderSlim", "neckLengthen", "backgroundBlur",
                "chinShape", "foreheadShape", "noseSlim", "darkCircles", "brightEyes", "teethWhiten", "nasolabial")) {
            try {
                VideoEffectsConfig.parse(values().apply { put(key, 1.2) })
                fail("应拒绝 $key 越界")
            } catch (_: IllegalArgumentException) {
            }
        }
    }
    @Test fun parsesLutUrlAndRejectsNonHttpsOrOversized() {
        val actual = VideoEffectsConfig.parse(values().apply { put("lutUrl", "https://cdn.example/luts/a.png"); put("lutIntensity", 0.6) })
        assertEquals("https://cdn.example/luts/a.png", actual.lutUrl)
        assertEquals(0.6f, actual.lutIntensity)
        val builtIn = VideoEffectsConfig.parse(values().apply { put("lutUrl", "asset://luts/clear_white.png"); put("lutIntensity", 0.8) })
        assertEquals("asset://luts/clear_white.png", builtIn.lutUrl)
        assertEquals(0.8f, builtIn.lutIntensity)
        val legacy = VideoEffectsConfig.parse(values())
        assertEquals("", legacy.lutUrl)
        assertEquals(1f, legacy.lutIntensity)
        for (bad in listOf<Any>(
            "http://cdn.example/a.png", "/data/a.png", "https://a b", "https://" + "a".repeat(2048), 7,
            "asset://luts/../secret.png", "asset://other/clear_white.png", "asset://luts/Clear.png",
            "asset://luts/a.jpg", "asset://luts/" + "a".repeat(33) + ".png",
        )) {
            try {
                VideoEffectsConfig.parse(values().apply { put("lutUrl", bad) })
                fail("应拒绝 lutUrl=$bad")
            } catch (_: IllegalArgumentException) {
            } catch (_: IllegalStateException) {
            }
        }
    }
    @Test fun parsesProSkinTextureAndRadiance() {
        val actual = VideoEffectsConfig.parse(values().apply { put("skinTexture", 0.55); put("radiance", 0.4) })
        assertEquals(0.55f, actual.skinTexture)
        assertEquals(0.4f, actual.radiance)
        val legacy = VideoEffectsConfig.parse(values())
        assertEquals(0f, legacy.skinTexture)
        assertEquals(0f, legacy.radiance)
        for ((key, bad) in listOf<Pair<String, Any?>>(
            "skinTexture" to -0.01, "skinTexture" to 1.01, "skinTexture" to Double.NaN, "skinTexture" to null,
            "radiance" to -0.01, "radiance" to 1.01, "radiance" to Double.POSITIVE_INFINITY, "radiance" to true,
        )) {
            val input = mutableMapOf<String, Any?>().apply { putAll(values()); put(key, bad) }
            try { VideoEffectsConfig.parse(input); fail("应拒绝字段 $key=$bad") }
            catch (_: IllegalArgumentException) { }
            catch (_: IllegalStateException) { }
        }
    }
    @Test fun parsesAll14MeshReshapeParamsIncludingBidirectional() {
        val actual = VideoEffectsConfig.parse(values().apply {
            put("vFace", 0.5); put("faceNarrow", 0.3); put("faceSmall", 0.4)
            put("cheekboneSlim", 0.25); put("mouthSmile", 0.6); put("cranialTop", 0.55)
            put("chinShape", -0.4); put("foreheadShape", -0.35)
            put("eyeDistance", -0.5); put("eyeCorner", 0.45)
            put("noseLength", -0.2); put("mouthSize", 0.3)
        })
        assertEquals(0.5f, actual.vFace)
        assertEquals(0.55f, actual.cranialTop)
        assertEquals(0.3f, actual.faceNarrow)
        assertEquals(0.4f, actual.faceSmall)
        assertEquals(0.25f, actual.cheekboneSlim)
        assertEquals(0.6f, actual.mouthSmile)
        assertEquals(-0.4f, actual.chinShape)
        assertEquals(-0.35f, actual.foreheadShape)
        assertEquals(-0.5f, actual.eyeDistance)
        assertEquals(0.45f, actual.eyeCorner)
        assertEquals(-0.2f, actual.noseLength)
        assertEquals(0.3f, actual.mouthSize)
        assertTrue(actual.hasFaceReshape())
        assertFalse(VideoEffectsConfig.parse(values()).hasFaceReshape())
        assertTrue(VideoEffectsConfig.parse(values().apply { put("cranialTop", 0.35) }).hasFaceReshape())
        for (key in listOf("vFace", "faceNarrow", "faceSmall", "cheekboneSlim", "mouthSmile", "cranialTop")) {
            for (bad in listOf(-0.01, 1.01, Double.NaN)) {
                try { VideoEffectsConfig.parse(values().apply { put(key, bad) }); fail("应拒绝 $key=$bad") }
                catch (_: IllegalArgumentException) {}
            }
        }
        for (key in listOf("chinShape", "foreheadShape", "eyeDistance", "eyeCorner", "noseLength", "mouthSize")) {
            for (bad in listOf(-1.01, 1.01, Double.POSITIVE_INFINITY)) {
                try { VideoEffectsConfig.parse(values().apply { put(key, bad) }); fail("应拒绝 $key=$bad") }
                catch (_: IllegalArgumentException) {}
            }
        }
    }
    @Test fun parsesAllMakeupParamsAndEnforcesWhitelists() {
        val actual = VideoEffectsConfig.parse(values().apply {
            put("lipstick", 0.65); put("lipstickColor", "cherry_red"); put("lipstickFinish", "glossy")
            put("blush", 0.45); put("blushStyle", "under_eye"); put("blushColor", "berry")
            put("eyebrow", 0.35); put("eyebrowStyle", "arched"); put("eyebrowColor", "grey_black")
            put("eyeshadow", 0.40); put("eyeshadowColor", "rose_wine")
            put("eyeliner", 0.50); put("aegyoSal", 0.30)
            put("contour", 0.25); put("highlight", 0.35)
            put("contactLens", 0.60); put("contactLensStyle", "sakura_pink")
            put("catchlight", 0.55)
        })
        assertTrue(actual.hasMakeup())
        assertEquals(0.65f, actual.lipstick); assertEquals("cherry_red", actual.lipstickColor); assertEquals("glossy", actual.lipstickFinish)
        assertEquals(0.45f, actual.blush); assertEquals("under_eye", actual.blushStyle); assertEquals("berry", actual.blushColor)
        assertEquals(0.35f, actual.eyebrow); assertEquals("arched", actual.eyebrowStyle); assertEquals("grey_black", actual.eyebrowColor)
        assertEquals(0.40f, actual.eyeshadow); assertEquals("rose_wine", actual.eyeshadowColor)
        assertEquals(0.50f, actual.eyeliner); assertEquals(0.30f, actual.aegyoSal)
        assertEquals(0.25f, actual.contour); assertEquals(0.35f, actual.highlight)
        assertEquals(0.60f, actual.contactLens); assertEquals("sakura_pink", actual.contactLensStyle)
        assertEquals(0.55f, actual.catchlight)

        val legacy = VideoEffectsConfig.parse(values())
        assertFalse(legacy.hasMakeup())
        assertEquals("rose_bean", legacy.lipstickColor)
        assertEquals("velvet", legacy.lipstickFinish)
        assertEquals("apple", legacy.blushStyle)
        assertEquals("peach", legacy.blushColor)
        assertEquals("natural", legacy.eyebrowStyle)
        assertEquals("dark_brown", legacy.eyebrowColor)
        assertEquals("earth", legacy.eyeshadowColor)
        assertEquals("hazel", legacy.contactLensStyle)

        for (key in listOf("lipstick", "blush", "eyebrow", "eyeshadow", "eyeliner", "aegyoSal",
                "contour", "highlight", "contactLens", "catchlight")) {
            for (bad in listOf(-0.01, 1.01, Double.NaN)) {
                try { VideoEffectsConfig.parse(values().apply { put(key, bad) }); fail("应拒绝 $key=$bad") }
                catch (_: IllegalArgumentException) {}
            }
        }
        for ((key, bad) in listOf<Pair<String, Any?>>(
            "lipstickColor" to "neon_green", "lipstickColor" to 1,
            "lipstickFinish" to "glitter", "blushStyle" to "clown",
            "blushColor" to "blue", "eyebrowStyle" to "thick",
            "eyebrowColor" to "blonde", "eyeshadowColor" to "purple",
            "contactLensStyle" to "cat_eye", "contactLensStyle" to null,
        )) {
            val input = mutableMapOf<String, Any?>().apply { putAll(values()); put(key, bad) }
            try { VideoEffectsConfig.parse(input); fail("应拒绝枚举 $key=$bad") }
            catch (_: IllegalArgumentException) {}
            catch (_: IllegalStateException) {}
        }
    }
    @Test fun legacyPayloadKeepsNeutralDefaults() {
        val actual = VideoEffectsConfig.parse(values())
        assertEquals(0f, actual.faceSlim)
        assertEquals(0f, actual.eyeEnlarge)
        assertFalse(actual.skinOnly)
        assertFalse(actual.compareOriginal)
        assertEquals(0f, actual.sharpness)
        assertEquals(1f, actual.saturation)
        assertEquals(1f, actual.filterIntensity)
        assertEquals(0f, actual.whiten)
        assertEquals(0f, actual.rosy)
    }
    @Test fun acceptsWhitenAndRosyWithinUnitRange() {
        for (value in listOf<Number>(0, 0.35, 1)) {
            val actual = VideoEffectsConfig.parse(values().apply { put("whiten", value); put("rosy", value) })
            assertEquals(value.toFloat(), actual.whiten)
            assertEquals(value.toFloat(), actual.rosy)
        }
        // 末尾带默认值的构造参数：既有调用方不受影响。
        assertEquals(0f, VideoEffectsConfig(true, 0f, 0f, "none").whiten)
        assertEquals(0f, VideoEffectsConfig.original.rosy)
    }
    @Test fun rejectsInvalidWhitenAndRosy() {
        for ((key, value) in listOf<Pair<String, Any?>>(
            "whiten" to -0.01, "whiten" to 1.01, "whiten" to Double.NaN, "whiten" to null,
            "whiten" to true, "whiten" to "0.5",
            "rosy" to -0.01, "rosy" to 1.01, "rosy" to Double.POSITIVE_INFINITY, "rosy" to null,
            "rosy" to false, "rosy" to "0.2",
        )) {
            val input = mutableMapOf<String, Any?>().apply { putAll(values()); put(key, value) }
            try { VideoEffectsConfig.parse(input); fail("应拒绝字段 $key=$value") }
            catch (_: IllegalArgumentException) { }
            catch (_: IllegalStateException) { }
        }
    }
    @Test fun acceptsDetailAndComparisonSettings() {
        val actual = VideoEffectsConfig.parse(values().apply {
            put("skinOnly", true); put("compareOriginal", true)
            put("sharpness", 0.4); put("saturation", 2); put("filterIntensity", 0)
        })
        assertTrue(actual.skinOnly)
        assertTrue(actual.compareOriginal)
        assertEquals(0.4f, actual.sharpness)
        assertEquals(2f, actual.saturation)
        assertEquals(0f, actual.filterIntensity)
    }
    @Test fun rejectsInvalidOptionalFieldsInsteadOfSilentlyDefaulting() {
        for ((key, value) in listOf<Pair<String, Any?>>(
            "skinOnly" to 1, "compareOriginal" to "true", "sharpness" to 1.01,
            "saturation" to 2.01, "saturation" to -0.01,
            "filterIntensity" to Double.NaN, "filterIntensity" to null,
            "skinOnly" to null, "compareOriginal" to null,
            "faceSlim" to 1.01, "eyeEnlarge" to -0.01, "faceSlim" to Double.NaN,
            "eyeEnlarge" to true, "faceSlim" to null,
        )) {
            val input = mutableMapOf<String, Any?>().apply { putAll(values()); put(key, value) }
            try { VideoEffectsConfig.parse(input); fail("应拒绝字段 $key") }
            catch (_: IllegalArgumentException) { }
            catch (_: IllegalStateException) { }
        }
    }
}
