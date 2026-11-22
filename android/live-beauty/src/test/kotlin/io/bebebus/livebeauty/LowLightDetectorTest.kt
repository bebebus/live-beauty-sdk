package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer

class LowLightDetectorTest {
    @Test fun entersAfterThreeDarkSamplesAndLeavesWithHysteresis() {
        val d = LowLightDetector()
        assertNull(d.onSample(0.1f))
        assertNull(d.onSample(0.1f))
        assertEquals(true, d.onSample(0.1f))
        assertTrue(d.dark)
        // 0.28 介于两个阈值之间：既不算暗也不算亮，保持暗光。
        repeat(5) { assertNull(d.onSample(0.28f)) }
        assertNull(d.onSample(0.4f))
        assertNull(d.onSample(0.4f))
        assertEquals(false, d.onSample(0.4f))
        assertFalse(d.dark)
    }

    @Test fun interruptedStreakStartsOverAndNonFiniteIsIgnored() {
        val d = LowLightDetector()
        d.onSample(0.1f); d.onSample(0.1f)
        assertNull(d.onSample(0.5f)) // 中断
        assertNull(d.onSample(Float.NaN))
        assertNull(d.onSample(0.1f)); assertNull(d.onSample(0.1f))
        assertEquals(true, d.onSample(0.1f))
        d.reset()
        assertFalse(d.dark)
    }

    private fun image(width: Int, height: Int, pixel: (Int, Int) -> Int): ByteBuffer =
        ByteBuffer.allocate(width * height * 4).apply {
            for (y in 0 until height) for (x in 0 until width) {
                val v = pixel(x, y).toByte()
                put(v); put(v); put(v); put(-1)
            }
            position(0)
        }

    @Test fun meanLumaUsesCentreRegionOnly() {
        assertEquals(1f, LowLightDetector.meanLuma(image(40, 80) { _, _ -> 255 }, 40, 80), 1e-3f)
        assertEquals(0f, LowLightDetector.meanLuma(image(40, 80) { _, _ -> 0 }, 40, 80), 1e-3f)
        // 四周一圈亮边（如窗户、补光灯直射边缘）不影响中部的判断。
        val framed = image(40, 80) { x, y -> if (x < 8 || x >= 32 || y < 16 || y >= 64) 255 else 25 }
        assertEquals(25f / 255f, LowLightDetector.meanLuma(framed, 40, 80), 1e-3f)
        assertEquals(0, framed.position())
    }
}
