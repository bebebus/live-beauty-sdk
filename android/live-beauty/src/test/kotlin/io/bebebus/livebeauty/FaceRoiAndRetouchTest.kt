package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.hypot

class FaceRoiAndRetouchTest {
    @Test
    fun faceRoiSelectorEntersWithHysteresisMaintainsSquareAspectAndFallsBackOnMiss() {
        val roi = FaceRoiSelector()
        assertFalse(roi.isActive)
        assertTrue(roi.select(720, 1280, 100L).isFullFrame)

        // 近景大脸（faceWidth = 0.44 > 0.30）：保持全画幅，零额外裁剪。
        roi.onFace(0.50f, 0.60f, 0.44f, 100L)
        assertFalse(roi.isActive)
        assertTrue(roi.select(720, 1280, 100L).isFullFrame)

        // 半身远景小脸（faceWidth = 0.18 <= 0.30）：进入各向同性正方形 ROI。
        val far = FaceRoiSelector()
        far.onFace(0.50f, 0.65f, 0.18f, 200L)
        assertTrue(far.isActive)
        val portrait = far.select(720, 1280, 210L)
        assertFalse(portrait.isFullFrame)
        // 验证竖屏 720x1280 下裁剪区域像素宽高严格 1:1 正方形（无几何拉伸）。
        assertEquals(portrait.width * 720f, portrait.height * 1280f, 0.05f)
        // 验证横屏 1280x720 下裁剪区域像素宽高同样 1:1 正方形且不越界。
        val landscape = far.select(1280, 720, 210L)
        assertEquals(landscape.width * 1280f, landscape.height * 720f, 0.05f)
        assertTrue(landscape.left >= 0f && landscape.left + landscape.width <= 1.0001f)
        assertTrue(landscape.top >= 0f && landscape.top + landscape.height <= 1.0001f)

        // 验证反投影精确还原中心点：GL centerY=0.65 对应 MediaPipe mpCy=0.35。
        assertEquals(0.50f, portrait.unprojectX(0.5f), 1e-4f)
        assertEquals(0.35f, portrait.unprojectY(0.5f), 1e-4f)

        // 迟滞保持区（0.30 < 0.33 < 0.36）：不来回跳变。
        far.onFace(0.50f, 0.65f, 0.34f, 240L)
        far.onFace(0.50f, 0.65f, 0.34f, 270L)
        assertTrue(far.isActive)

        // 单次漏检立即退回全画幅检索，杜绝快速走位死锁。
        far.onMiss()
        assertFalse(far.isActive)
        assertTrue(far.select(720, 1280, 280L).isFullFrame)

        // 超时（> STALE_MS）或乱序回退时间戳（nowMs < lastSeenMs）同样自动退回全画幅。
        far.onFace(0.10f, 0.92f, 0.16f, 300L)
        val corner = far.select(720, 1280, 310L)
        assertEquals(0f, corner.left, 1e-6f)
        assertEquals(0f, corner.top, 1e-6f)
        assertTrue(far.select(720, 1280, 290L).isFullFrame)
        assertFalse(far.isActive)
        far.onFace(0.10f, 0.92f, 0.16f, 300L)
        assertTrue(far.select(720, 1280, 300L + FaceRoiSelector.STALE_MS + 10L).isFullFrame)
    }

    @Test
    fun faceTextureSamplerLeasesCroppedRoiToFaceOnlyWithoutAffectingBodyOrMasks() {
        val sampler = FaceTextureSampler(384)
        val mainW = 216
        val mainH = 384
        val mainBuf = ByteBuffer.allocateDirect(mainW * mainH * 4).apply {
            put(0, 11.toByte())
        }
        val side = FaceRoiSelector.ROI_SIDE_PX
        val roiBuf = ByteBuffer.allocateDirect(side * side * 4).apply {
            put(0, 99.toByte())
        }
        val crop = FaceRoiSelector.CropRect(0.25f, 0.15f, 0.50f, 0.28125f)
        sampler.stageForTest(mainBuf, mainW, mainH, roiPixels = roiBuf, roiCrop = crop)

        val faceSample = sampler.lease(FaceTextureSampler.FACE)
        assertEquals(side, faceSample.width)
        assertEquals(side, faceSample.height)
        assertEquals(crop, faceSample.crop)
        assertEquals(99.toByte(), faceSample.pixels.get(0))

        for (other in listOf(FaceTextureSampler.BODY, FaceTextureSampler.SEGMENT, FaceTextureSampler.HAIR)) {
            val s = sampler.lease(other)
            assertEquals(mainW, s.width)
            assertEquals(mainH, s.height)
            assertTrue(s.crop.isFullFrame)
            assertEquals(11.toByte(), s.pixels.get(0))
        }
    }

    @Test
    fun pboSupportDetectsOpenGlEs3AndAboveOnly() {
        assertTrue(PboSupport.isEs3Version("OpenGL ES 3.0 V@415.0"))
        assertTrue(PboSupport.isEs3Version("OpenGL ES 3.2 build 1.15@6052913"))
        assertFalse(PboSupport.isEs3Version("OpenGL ES 2.0 (ANGLE 2.1)"))
        assertFalse(PboSupport.isEs3Version(null))
        assertFalse(PboSupport.isEs3Version(""))
    }

    @Test
    fun anatomicalRetouchWeightsProtectAegyoSalAndInnerMouthHasZeroLipOverlap() {
        // 1. 卧蚕保护点与眼裂点权重必须严格为 0，泪沟核心点权重显著 > 0.5。
        for (v in FaceRetouchMeshPass.AEGYO_SAL_PROTECTED_SET) {
            assertEquals("卧蚕点 $v 必须零权重保护", 0f, FaceRetouchMeshPass.STATIC_TEAR_WEIGHTS[v], 0f)
        }
        for (v in FaceMakeupUv.RIGHT_EYE_SET + FaceMakeupUv.LEFT_EYE_SET) {
            assertEquals(0f, FaceRetouchMeshPass.STATIC_TEAR_WEIGHTS[v], 0f)
        }
        for (tearCore in listOf(229, 230, 449, 450)) {
            assertTrue("泪沟核心点 $tearCore 权重应 > 0.5", FaceRetouchMeshPass.STATIC_TEAR_WEIGHTS[tearCore] > 0.5f)
        }

        // 2. 鼻唇沟弧线关键点权重与单位法向校验。
        for (foldCore in listOf(203, 206, 216, 423, 426, 436)) {
            assertTrue("法令纹关键点 $foldCore 权重应 > 0.5", FaceRetouchMeshPass.STATIC_FOLD_WEIGHTS[foldCore] > 0.5f)
            val nx = FaceRetouchMeshPass.STATIC_FOLD_NORMALS[foldCore * 2]
            val ny = FaceRetouchMeshPass.STATIC_FOLD_NORMALS[foldCore * 2 + 1]
            assertEquals(1f, hypot(nx, ny), 1e-4f)
        }

        // 3. 18 个内唇口腔三角形（54 个索引）必须全部落在 INNER_LIP_SET 内，与外唇红零重叠。
        assertEquals(18 * 3, FaceMakeupUv.INNER_MOUTH_INDICES.size)
        for (idx in FaceMakeupUv.INNER_MOUTH_INDICES) {
            assertTrue(idx.toInt() in FaceMakeupUv.INNER_LIP_SET)
        }
        assertEquals(28 * 3, FaceMakeupUv.EYE_OPENING_INDICES.size)
        assertTrue(FaceRetouchMeshPass.SKIN_RETOUCH_INDICES.size in (60 * 3)..(220 * 3))

        // 4. 动态有效权重选向校验：仅开法令纹 (dc=0, nf=1) 时必须取横向跨沟偏移，仅开黑眼圈 (dc=1, nf=0) 时必须取纵向泪沟偏移。
        val pass = FaceRetouchMeshPass()
        val dummyLm = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT * 2) { 0.5f }
        pass.packVertices(dummyLm, null, 1f, 0f, 0f, 1f, 0.40f, 0.50f, 9f / 16f, 1f, dc = 0f, nf = 1f)
        val foldOffX = pass.packedVerticesForTest[206 * 8 + 6]
        val tearUnusedX = pass.packedVerticesForTest[229 * 8 + 6]
        val tearUnusedY = pass.packedVerticesForTest[229 * 8 + 7]
        assertTrue("仅开法令纹时法令纹核心点 206 应取非零横向跨沟偏移", kotlin.math.abs(foldOffX) > 0.005f)
        assertEquals("仅开法令纹时纯泪沟点 229 偏移必须归零", 0f, hypot(tearUnusedX, tearUnusedY), 1e-6f)

        pass.packVertices(dummyLm, null, 1f, 0f, 0f, 1f, 0.40f, 0.50f, 9f / 16f, 1f, dc = 1f, nf = 0f)
        val tearOffX = pass.packedVerticesForTest[229 * 8 + 6]
        val tearOffY = pass.packedVerticesForTest[229 * 8 + 7]
        val foldUnusedX = pass.packedVerticesForTest[206 * 8 + 6]
        assertEquals(0f, tearOffX, 1e-6f)
        assertTrue("仅开黑眼圈时泪沟点 229 应向下 (-v) 采样苹果肌干净肤色", tearOffY < -0.008f)
        assertEquals("仅开黑眼圈时纯法令纹点 206 偏移必须归零", 0f, foldUnusedX, 1e-6f)

        // 5. 双频奶油肌匀光曲线边界校验。
        assertEquals(0f, BeautyQualityTuning.macroEvenness(0f), 0f)
        assertEquals(0f, BeautyQualityTuning.macroEvenness(-0.2f), 0f)
        assertEquals(0f, BeautyQualityTuning.macroEvenness(Float.NaN), 0f)
        val mid = BeautyQualityTuning.macroEvenness(0.5f)
        val hi = BeautyQualityTuning.macroEvenness(1.0f)
        assertTrue(mid > 0.25f && hi > mid && hi <= 0.45f)
    }
}
