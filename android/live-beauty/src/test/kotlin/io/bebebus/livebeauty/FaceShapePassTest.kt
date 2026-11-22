package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class FaceShapePassTest {
    private val aspect = 9f / 16f

    /** 正脸：椭圆中心 (0.5, 0.55)（纹理坐标），rx 0.224、ry 0.23（度量），u = (1, 0)；双眼 y 0.6，嘴 y 0.45。 */
    private fun state() = FloatArray(FaceGeometrySmoother.STATE_SIZE).apply {
        this[16] = 0.4f; this[17] = 0.6f; this[19] = 0.6f; this[20] = 0.6f // 双眼中心
        this[22] = 0.5f; this[23] = 0.55f; this[24] = 0.224f; this[25] = 0.23f // 椭圆
        this[26] = 1f; this[27] = 0f // u
        this[28] = 0.5f; this[29] = 0.45f; this[30] = 0.07f; this[31] = 0.03f // 嘴
        for (i in 32 until 40) this[i] = 0.1f * (i - 31) // 眼睛遮罩
        this[40] = 1f
    }

    @Test fun derivesChinForeheadAndNoseFromTheFaceOval() {
        val out = FloatArray(FaceShapePass.VALUES)
        assertTrue(FaceShapePass.pack(state(), aspect, 1f, 1f, 1f, out))
        val halfHeight = 0.23f / 1.15f
        // 下巴在椭圆中心下方 halfHeight（度量），纹理 y 乘 aspect。
        assertEquals(0.5f, out[0], 1e-5f)
        assertEquals(0.55f - halfHeight * aspect, out[1], 1e-5f)
        assertEquals(0.25f, out[3], 1e-6f)
        // 额头在上方 0.9 halfHeight。
        assertEquals(0.55f + 0.9f * halfHeight * aspect, out[5], 1e-5f)
        assertEquals(0.2f, out[7], 1e-6f)
        // 鼻翼：双眼中点 (0.5, 0.6) 到嘴 (0.5, 0.45) 的 62%。
        assertEquals(0.5f, out[8], 1e-5f)
        assertEquals(0.6f + 0.62f * (0.45f - 0.6f), out[9], 1e-5f)
        assertEquals(0.3f, out[11], 1e-6f)
        assertEquals(1f, out[12], 1e-6f); assertEquals(0f, out[13], 1e-6f); assertEquals(aspect, out[14], 1e-6f)
    }

    @Test fun retouchCopiesEyeAndMouthEllipsesAndStrengths() {
        val out = FloatArray(FaceShapePass.VALUES)
        assertTrue(FaceShapePass.pack(state(), aspect, 0f, 0f, 0f, out, darkCircles = 0.5f, teeth = 1.4f, folds = 0.3f))
        assertEquals(FaceShapePass.UNIFORMS.size * 4, FaceShapePass.VALUES)
        // 变形三项为 0。
        assertEquals(0f, out[3], 0f); assertEquals(0f, out[7], 0f); assertEquals(0f, out[11], 0f)
        for (i in 0 until 8) assertEquals(0.1f * (i + 1), out[16 + i], 1e-6f)
        assertEquals(0.5f, out[24], 1e-6f); assertEquals(0.45f, out[25], 1e-6f)
        assertEquals(0.07f, out[26], 1e-6f); assertEquals(0.03f, out[27], 1e-6f)
        assertEquals(0.5f, out[28], 1e-6f); assertEquals(0f, out[29], 0f); assertEquals(1f, out[30], 0f) // 夹到 1
        assertEquals(0.3f, out[31], 1e-6f)
        // 只开法令纹也算生效；鼻翼中心与半径照常写入（着色器用它定位法令纹）。
        assertTrue(FaceShapePass.pack(state(), aspect, 0f, 0f, 0f, out, folds = 0.5f))
        assertEquals(0f, out[11], 0f)
        assertTrue(out[10] > 0f)
    }

    @Test fun nothingToDoOrDegenerateGeometryReturnsFalse() {
        val out = FloatArray(FaceShapePass.VALUES) { 9f }
        assertFalse(FaceShapePass.pack(state(), aspect, 0f, 0f, 0f, out))
        assertTrue(out.all { it == 0f })
        assertFalse(FaceShapePass.pack(state().apply { this[26] = 0f }, aspect, 1f, 0f, 0f, out))
        assertFalse(FaceShapePass.pack(state().apply { this[25] = 0f }, aspect, 1f, 0f, 0f, out))
        assertFalse(FaceShapePass.pack(state(), Float.NaN, 1f, 0f, 0f, out))
        // 只开一项时其余强度为 0。
        assertTrue(FaceShapePass.pack(state(), aspect, 0f, 0f, 0.5f, out))
        assertEquals(0f, out[3], 0f); assertEquals(0f, out[7], 0f); assertEquals(0.15f, out[11], 1e-6f)
    }
}
