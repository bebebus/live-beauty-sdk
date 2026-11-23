package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

/** 验证 v2.1.0 虚拟背景、2D 贴纸、性能遥测与内置 C++ 引擎辅助纯逻辑。 */
class LiveBeautyExtensionsTest {

    @Test
    fun virtualBackgroundWireNamesMatchCppParser() {
        assertEquals("none", VirtualBackgroundMode.NONE.wireName)
        assertEquals("blur", VirtualBackgroundMode.BLUR.wireName)
        assertEquals("virtual_image", VirtualBackgroundMode.VIRTUAL_IMAGE.wireName)
        assertEquals("virtual_blur", VirtualBackgroundMode.VIRTUAL_BLUR.wireName)
    }

    @Test
    fun stickerAnchorIdsMatchCppEnumOrder() {
        assertEquals(0, StickerAnchor.FOREHEAD_CROWN.id)
        assertEquals(1, StickerAnchor.EYES_BRIDGE.id)
        assertEquals(2, StickerAnchor.NOSE_TIP.id)
        assertEquals(3, StickerAnchor.CHEEKS.id)
        assertEquals(4, StickerAnchor.FULL_FACE.id)
    }

    @Test
    fun perfStatsUnpacksRawEightFloatTelemetry() {
        val raw = floatArrayOf(4.25f, 4.10f, 30.0f, 1f, 0.98f, 0.85f, 645f, 120f, 2f, 5000f)
        val st = BeautyPerfStats.fromRaw(raw)
        assertEquals(4.25f, st.lastFrameMs, 1e-5f)
        assertEquals(4.10f, st.emaFrameMs, 1e-5f)
        assertEquals(30.0f, st.fps, 1e-5f)
        assertEquals(1, st.loadLevel)
        assertEquals(0.98f, st.facePresence, 1e-5f)
        assertEquals(0.85f, st.hairShown, 1e-5f)
        assertEquals(645, st.activePassesMask)
        assertEquals(120L, st.frameCount)
        assertEquals(2, st.activeFaces)
        assertEquals(BeautyPerfStats(), BeautyPerfStats.fromRaw(floatArrayOf(1f, 2f)))
    }

    @Test
    fun cppEngineRoiSanitizerAndSafeJvmFallback() {
        assertEquals(FaceRoiSelector.CropRect.FULL, LiveBeautyCppEngine.roiToCropRect(floatArrayOf(0f, 0f, 1f, 1f)))
        assertEquals(FaceRoiSelector.CropRect.FULL, LiveBeautyCppEngine.roiToCropRect(floatArrayOf(Float.NaN, 0f, 0.4f, 0.4f)))
        val sub = LiveBeautyCppEngine.roiToCropRect(floatArrayOf(0.2f, 0.1f, 0.5f, 0.5f))
        assertFalse(sub.isFullFrame)
        assertEquals(0.2f, sub.left, 1e-6f)

        val engine = LiveBeautyCppEngine(clockMs = { 2000L })
        engine.setPipelineOptions(maxFaces = 2, frameDelayFrames = 1)
        engine.setVirtualBackgroundTexture(glTextureId = 7, aspect = 0.5625f, mode = VirtualBackgroundMode.VIRTUAL_IMAGE)
        engine.setSticker(0, LiveBeautySticker(anchor = StickerAnchor.EYES_BRIDGE, glTextureId = 9))
        engine.resetTracking()
        assertEquals(12, engine.process(12, 720, 1280, 1_000_000_000L))
        assertEquals(0, engine.processOes(15, null, 720, 1280, 1_000_000_000L))
        engine.clearStickers()
        engine.onGlReleased(false)
    }
}
