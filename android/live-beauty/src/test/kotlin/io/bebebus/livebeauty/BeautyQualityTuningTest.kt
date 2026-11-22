package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class BeautyQualityTuningTest {
    @Test fun guidedEpsFollowsSigmaCurveAndClamps() {
        assertEquals(0.01845f, BeautyQualityTuning.guidedEps(0f), 5e-4f)
        assertEquals(0.0443f, BeautyQualityTuning.guidedEps(0.55f), 5e-4f)
        assertEquals(0.0738f, BeautyQualityTuning.guidedEps(1f), 5e-4f)
        var previous = 0f
        for (i in 0..20) {
            val eps = BeautyQualityTuning.guidedEps(i / 20f)
            assertTrue("eps 随磨皮单调上升：${i / 20f} → $eps", eps > previous)
            previous = eps
        }
        // 超出 [0, 1] 钳位，NaN 按 0。
        assertEquals(BeautyQualityTuning.guidedEps(0f), BeautyQualityTuning.guidedEps(-1f), 0f)
        assertEquals(BeautyQualityTuning.guidedEps(1f), BeautyQualityTuning.guidedEps(3f), 0f)
        assertEquals(BeautyQualityTuning.guidedEps(0f), BeautyQualityTuning.guidedEps(Float.NaN), 0f)
    }

    @Test fun downTapScaleMakesFourBilinearTapsA3x3Box() {
        val scale = BeautyQualityTuning.DOWN_TAP_SCALE
        assertTrue("DOWN_TAP_SCALE 应为 2/3：$scale", kotlin.math.abs(scale - 2f / 3f) < 1e-6f)
        // 一维：±scale 两个双线性抽头的平均，中心像素权重 1 − scale，两侧各 scale / 2；三者都为 1/3 才是盒式均值，
        // 二维按行列可分离即 3×3 盒式均值。旧值 1 时中心权重为 0（只采到四个角）。
        assertEquals(1f / 3f, 1f - scale, 1e-6f)
        assertEquals(1f / 3f, scale / 2f, 1e-6f)
    }

    @Test fun coreThresholdRisesInLowLightAndUnderBitratePressure() {
        assertEquals(0.005f, BeautyQualityTuning.coreThreshold(0.5f, 0f), 1e-6f)
        assertEquals(0.005f, BeautyQualityTuning.coreThreshold(0.30f, 0f), 1e-6f)
        assertEquals(0.009f, BeautyQualityTuning.coreThreshold(0.20f, 0f), 1e-6f)
        assertEquals(0.013f, BeautyQualityTuning.coreThreshold(0.08f, 0f), 1e-6f)
        // 还没采到亮度（NaN）视为亮。
        assertEquals(0.005f, BeautyQualityTuning.coreThreshold(Float.NaN, 0f), 1e-6f)
        assertEquals(0.009f, BeautyQualityTuning.coreThreshold(0.5f, 1f), 1e-6f)
        assertEquals(0.009f, BeautyQualityTuning.coreThreshold(0.5f, 3f), 1e-6f)
        assertEquals(0.005f, BeautyQualityTuning.coreThreshold(0.5f, Float.NaN), 1e-6f)
        assertEquals(0.017f, BeautyQualityTuning.coreThreshold(0f, 1f), 1e-6f)
        // 对亮度单调不增、对压力单调不减。
        var previous = Float.MAX_VALUE
        for (i in 0..20) {
            val core = BeautyQualityTuning.coreThreshold(i / 40f, 0.5f)
            assertTrue(core <= previous)
            previous = core
        }
        previous = 0f
        for (i in 0..20) {
            val core = BeautyQualityTuning.coreThreshold(0.2f, i / 20f)
            assertTrue(core >= previous)
            previous = core
        }
    }

    private fun GuidedSpread.feed(px: Float, frames: Int = 120) { repeat(frames) { update(px) } }

    @Test fun spreadConvergesToFaceSizeInSteps() {
        val spread = GuidedSpread()
        assertEquals(GuidedSpread.DEFAULT, spread.value, 0f)
        // 720p 竖屏常见脸宽 324px：目标 1.08，按 0.05 步进停在 1.10。
        var previous = spread.value
        repeat(120) {
            val value = spread.update(324f)
            // 每帧最多走一步。
            assertTrue(kotlin.math.abs(value - previous) <= GuidedSpread.STEP + 1e-6f)
            previous = value
        }
        assertEquals(1.10f, spread.value, 1e-4f)
    }

    @Test fun spreadIgnoresLandmarkJitter() {
        val spread = GuidedSpread()
        spread.feed(324f)
        val settled = spread.value
        for (i in 0 until 300) {
            spread.update(if (i % 2 == 0) 330f else 318f)
            assertEquals(settled, spread.value, 0f)
        }
    }

    @Test fun spreadClampsToRange() {
        val small = GuidedSpread()
        small.feed(100f)
        assertEquals(GuidedSpread.MIN, small.value, 1e-4f)
        val large = GuidedSpread()
        large.feed(1000f)
        assertEquals(GuidedSpread.MAX, large.value, 1e-4f)
    }

    @Test fun spreadHoldsWithoutFaceAndResets() {
        val spread = GuidedSpread()
        spread.feed(1000f)
        val held = spread.value
        for (bad in floatArrayOf(0f, -5f, Float.NaN, Float.POSITIVE_INFINITY)) {
            assertEquals(held, spread.update(bad), 0f)
        }
        spread.reset()
        assertEquals(GuidedSpread.DEFAULT, spread.value, 0f)
        // 复位后平滑状态也回到默认（不残留上一个上下文的大脸）：一帧常见脸宽不足以让半径走一步。
        assertEquals(GuidedSpread.DEFAULT, spread.update(324f), 0f)
    }

    // ---- LUT 换表不断档（LutPass 的绑定语义，纯逻辑）。

    @Test fun lutKeepsPreviousTableWhileNextOneLoads() {
        val lut = LutBinding()
        assertEquals("", lut.select("a"))
        assertEquals("a", lut.pendingUrl)
        lut.loaded("a")
        assertEquals("", lut.pendingUrl)
        assertEquals("a", lut.select("a"))
        // 换到 b：上传完成前继续绑 a，画面不会出现「完全不调色」的帧。
        repeat(3) { assertEquals("a", lut.select("b")) }
        assertEquals("b", lut.pendingUrl)
        assertEquals("a", lut.boundUrl)
        lut.loaded("b")
        assertEquals("b", lut.select("b"))
        assertEquals("", lut.pendingUrl)
    }

    @Test fun lutDoesNotBorrowStaleTableAfterGap() {
        val lut = LutBinding()
        lut.loaded("a")
        assertEquals("a", lut.select("a"))
        // 关掉 LUT（或 LUT 故障、地址判定失败，调用方传空串）：本帧不查表。
        assertEquals("", lut.select(""))
        assertEquals("", lut.boundUrl)
        // 再开另一张：上一帧没有查表，不借用更早留下的 a，等 b 上传完成。
        assertEquals("", lut.select("b"))
        assertEquals("b", lut.pendingUrl)
        lut.loaded("b")
        assertEquals("b", lut.select("b"))
        // 新上下文（reset）忘掉已上传的表，要等重新上传。
        lut.reset()
        assertEquals("", lut.loadedUrl)
        assertEquals("", lut.select("b"))
    }
}
