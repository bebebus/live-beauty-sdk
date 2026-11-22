package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

/** 背景虚化细节保留 β（BG_MIX 的 uBackground.w）随「人脸近期可见」渐变（评审复核：第一批 D3 曾随人脸有无单帧硬切）。 */
class BackgroundBlurPassTest {
    @Test fun detailKeepFollowsTheFaceWeightContinuously() {
        // 没有人脸：β = 0，与系数无关。
        for (w in listOf(0f, 0.5f, 1f, 2f, Float.NaN)) assertEquals("w=$w", 0f, BackgroundBlurPass.detailKeep(false, w), 0f)
        // 系数为 1（缺省）：恰为 DETAIL_KEEP，与之前逐字节相同；为 0、负数或 NaN：恰为 0，与没有人脸相同。
        assertEquals(BackgroundBlurPass.DETAIL_KEEP.toRawBits(), BackgroundBlurPass.detailKeep(true, 1f).toRawBits())
        for (w in listOf(0f, -0f, -0.3f, Float.NaN, Float.NEGATIVE_INFINITY)) {
            assertEquals("w=$w", 0f.toRawBits(), BackgroundBlurPass.detailKeep(true, w).toRawBits())
        }
        // 超过 1 按 1。
        assertEquals(BackgroundBlurPass.DETAIL_KEEP, BackgroundBlurPass.detailKeep(true, 1.7f), 0f)
        assertEquals(BackgroundBlurPass.DETAIL_KEEP, BackgroundBlurPass.detailKeep(true, Float.POSITIVE_INFINITY), 0f)
        // 0..1 之间连续、单调：相邻系数的 β 之差不超过 DETAIL_KEEP × 系数之差（没有台阶，不再是有脸 / 无脸的二值）。
        var previous = BackgroundBlurPass.detailKeep(true, 0f)
        for (i in 1..1000) {
            val w = i / 1000f
            val beta = BackgroundBlurPass.detailKeep(true, w)
            assertTrue("w=$w 单调", beta >= previous)
            assertTrue("w=$w 连续：$previous → $beta", beta - previous <= BackgroundBlurPass.DETAIL_KEEP / 1000f + 1e-6f)
            assertEquals("w=$w", BackgroundBlurPass.DETAIL_KEEP * w, beta, 1e-6f)
            previous = beta
        }
        assertTrue(abs(previous - BackgroundBlurPass.DETAIL_KEEP) == 0f)
    }
}
