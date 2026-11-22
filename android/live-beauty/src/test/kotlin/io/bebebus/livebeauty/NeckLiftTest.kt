package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class NeckLiftTest {
    private val aspect = 9f / 16f
    /** 竖直站立：胯 (0.5, 0.4)，身体方向朝下 (0, -1)，躯干 0.5，半宽 0.12。 */
    private val body = floatArrayOf(0.5f, 0.4f, 0f, -1f, 0.5f, 0.12f, 1f)

    private fun face(x: Float = 0.5f) = FloatArray(FaceGeometrySmoother.STATE_SIZE) { i -> if (i % 2 == 0) x else 0.8f }.apply {
        this[24] = 0.2f; this[25] = 0.2f; this[26] = 1f; this[27] = 0f; this[40] = 1f
    }

    @Test fun headCentredOnTheBodyMovesUpByBandTimesStrength() {
        val out = FloatArray(FaceGeometrySmoother.STATE_SIZE)
        val src = face()
        NeckLift.apply(src, body, 0.3f, aspect, out)
        val lift = 0.3f * 0.5f * 0.3f * aspect // band × neck，纹理 y 乘 aspect
        // 所有中心点的 y（cheeks 8 个、双眼、椭圆、嘴、双眼遮罩）都上移同样的量。
        for (i in intArrayOf(1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 23, 29, 33, 37)) assertEquals(src[i] + lift, out[i], 1e-5f)
        // 眼睛宽度（18、21）不动。
        assertEquals(src[18], out[18], 0f); assertEquals(src[21], out[21], 0f)
        // 半径、方向等非坐标量不动。
        assertEquals(0.2f, out[24], 0f); assertEquals(1f, out[26], 0f); assertEquals(1f, out[40], 0f)
        // x 不变（身体竖直）。
        assertEquals(0.5f, out[22], 1e-6f)
    }

    @Test fun noNeckOrFarFromTheBodyLeavesTheFaceAlone() {
        val out = FloatArray(FaceGeometrySmoother.STATE_SIZE)
        val src = face()
        assertArrayEquals(src, NeckLift.apply(src, body, 0f, aspect, out), 0f)
        // 脸在身体侧面 2.2 半宽之外：着色器在那里已不再上移。
        val side = face(x = 0.5f + 2.3f * 0.12f)
        assertArrayEquals(side, NeckLift.apply(side, body, 0.3f, aspect, out), 0f)
    }
}
