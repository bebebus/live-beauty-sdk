package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class FaceMakeupUvTest {
    @Test fun skinFaceIndicesExcludeEyesAndInnerMouth() {
        assertEquals(852 * 3, FaceMakeupUv.SKIN_FACE_INDICES.size)
        val rightEye = setOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        val leftEye = setOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
        val innerLip = setOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)

        val indices = FaceMakeupUv.SKIN_FACE_INDICES
        for (i in indices.indices step 3) {
            val a = indices[i].toInt()
            val b = indices[i + 1].toInt()
            val c = indices[i + 2].toInt()
            assertTrue("索引应在 0..467 范围内", a in 0..467 && b in 0..467 && c in 0..467)
            assertFalse("面部皮肤子网不应含右眼裂三角形", a in rightEye && b in rightEye && c in rightEye)
            assertFalse("面部皮肤子网不应含左眼裂三角形", a in leftEye && b in leftEye && c in leftEye)
            assertFalse("面部皮肤子网不应含内唇口腔三角形", a in innerLip && b in innerLip && c in innerLip)
        }
        // 852 皮肤 + 14 右眼 + 14 左眼 + 18 口腔 = 898 完整规范面部网格
        assertEquals(898, 852 + FaceMeshTopology.RIGHT_EYE_INDICES.size / 3 + FaceMeshTopology.LEFT_EYE_INDICES.size / 3 + 18)
        assertEquals(120 * 3, FaceMeshTopology.LIP_BAND_INDICES.size)
    }

    @Test fun packVerticesMapsNdcScreenUvAndCanonicalUv() {
        val landmarks = FloatArray(FaceMeshFrame.COORD_COUNT) { idx ->
            val v = idx / 2
            if (idx % 2 == 0) 0.2f + (v % 32) * 0.02f else 0.15f + (v / 32) * 0.04f
        }
        val out = FloatArray(FaceMakeupUv.VERTEX_DATA_SIZE)
        FaceMakeupUv.packVertices(landmarks, out)
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            val k = i * FaceMakeupUv.VERTEX_STRIDE_FLOATS
            val sx = landmarks[i * 2]
            val sy = landmarks[i * 2 + 1]
            assertEquals(sx * 2f - 1f, out[k], 1e-6f)
            assertEquals(sy * 2f - 1f, out[k + 1], 1e-6f)
            assertEquals(sx, out[k + 2], 1e-6f)
            assertEquals(sy, out[k + 3], 1e-6f)
            assertEquals(FaceMeshTopology.CANONICAL_UV[i * 2], out[k + 4], 1e-6f)
            assertEquals(FaceMeshTopology.CANONICAL_UV[i * 2 + 1], out[k + 5], 1e-6f)
        }
    }

    @Test fun computeIrisCircleMeasuresMetricRadiusAndRejectsDegenerate() {
        val landmarks = FloatArray(FaceMeshFrame.COORD_COUNT)
        val aspect = 0.5625f // 9:16 竖屏
        val cx = 0.42f
        val cy = 0.58f
        val rMetric = 0.024f
        landmarks[468 * 2] = cx; landmarks[468 * 2 + 1] = cy
        landmarks[469 * 2] = cx + rMetric; landmarks[469 * 2 + 1] = cy
        landmarks[470 * 2] = cx; landmarks[470 * 2 + 1] = cy + rMetric * aspect
        landmarks[471 * 2] = cx - rMetric; landmarks[471 * 2 + 1] = cy
        landmarks[472 * 2] = cx; landmarks[472 * 2 + 1] = cy - rMetric * aspect

        val out = FloatArray(4)
        assertTrue(FaceMakeupUv.computeIrisCircle(landmarks, 468, aspect, out))
        assertEquals(cx, out[0], 1e-5f)
        assertEquals(cy, out[1], 1e-5f)
        assertEquals(rMetric * 1.08f, out[2], 1e-5f)
        assertEquals(aspect, out[3], 1e-5f)

        // 零半径退化点被拒绝
        assertFalse(FaceMakeupUv.computeIrisCircle(landmarks, 473, aspect, out))
        // 非正宽高比被拒绝
        assertFalse(FaceMakeupUv.computeIrisCircle(landmarks, 468, 0f, out))
        // 过短数组被拒绝
        assertFalse(FaceMakeupUv.computeIrisCircle(FloatArray(10), 468, aspect, out))
    }

    @Test fun curatedPalettesCoverEveryWhitelistedEnumDistinctly() {
        val lipColors = VideoEffectsConfig.LIPSTICK_COLORS.map { FaceMakeupUv.lipstickRgb(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.LIPSTICK_COLORS.size, lipColors.size)
        for (rgb in lipColors) assertTrue(rgb.all { it in 0f..1f })

        val blushColors = VideoEffectsConfig.BLUSH_COLORS.map { FaceMakeupUv.blushRgb(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.BLUSH_COLORS.size, blushColors.size)
        val blushChannels = VideoEffectsConfig.BLUSH_STYLES.map { FaceMakeupUv.blushChannel(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.BLUSH_STYLES.size, blushChannels.size)

        val browColors = VideoEffectsConfig.EYEBROW_COLORS.map { FaceMakeupUv.eyebrowRgb(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.EYEBROW_COLORS.size, browColors.size)
        val browChannels = VideoEffectsConfig.EYEBROW_STYLES.map { FaceMakeupUv.eyebrowChannel(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.EYEBROW_STYLES.size, browChannels.size)

        val shadowMain = VideoEffectsConfig.EYESHADOW_COLORS.map { FaceMakeupUv.eyeshadowMainRgb(it).toList() }.toSet()
        val shadowDeep = VideoEffectsConfig.EYESHADOW_COLORS.map { FaceMakeupUv.eyeshadowDeepRgb(it).toList() }.toSet()
        assertEquals(VideoEffectsConfig.EYESHADOW_COLORS.size, shadowMain.size)
        assertEquals(VideoEffectsConfig.EYESHADOW_COLORS.size, shadowDeep.size)

        val irisTiles = VideoEffectsConfig.CONTACT_LENS_STYLES.map { FaceMakeupUv.irisTile(it) }.toSet()
        assertEquals(VideoEffectsConfig.CONTACT_LENS_STYLES.size, irisTiles.size)
        for ((tx, ty) in irisTiles) {
            assertTrue(tx == 0f || tx == 0.5f)
            assertTrue(ty == 0f || ty == 0.5f)
        }
    }

    @Test fun skinMaskStaticWeightsExcludeFeaturesAndHighlightBonePlanes() {
        val weights = SkinMaskPass.STATIC_WEIGHTS
        assertEquals((468 + 36) * 3, weights.size)
        // 右眼上睑 159、左眼上睑 386、右眉心 52、上唇内缘 13 必须严格排除磨皮/美白 (skin=0, excl=1)
        for (idx in intArrayOf(159, 386, 52, 13)) {
            assertEquals("顶点 $idx skin 应为 0", 0f, weights[idx * 3], 1e-6f)
            assertEquals("顶点 $idx excl 应为 1", 1f, weights[idx * 3 + 1], 1e-6f)
            assertEquals("顶点 $idx hiPlane 应为 0", 0f, weights[idx * 3 + 2], 1e-6f)
        }
        // 鼻梁中段 197 (u≈0.50, v≈0.44) 应为满皮肤权重且具备高 3D 骨相高光权重
        assertEquals(1f, weights[197 * 3], 1e-6f)
        assertEquals(0f, weights[197 * 3 + 1], 1e-6f)
        assertTrue("鼻梁骨相高光权重应 > 0.85", weights[197 * 3 + 2] > 0.85f)
        // 外圈 36 个额头冠/下颌羽化环顶点必须全部归零，确保人脸边缘丝滑过渡至背景
        for (j in 0 until 36) {
            val base = (468 + j) * 3
            assertEquals(0f, weights[base], 1e-6f)
            assertEquals(0f, weights[base + 1], 1e-6f)
            assertEquals(0f, weights[base + 2], 1e-6f)
        }
    }
}
