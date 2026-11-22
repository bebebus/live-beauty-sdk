package io.bebebus.livebeauty

import java.nio.ByteBuffer

/**
 * 暗光检测（仅 Android，纯逻辑，只在 GL 线程使用）。约每秒一个样本：相机原始帧（美颜前）画面中部的平均亮度。
 * 连续 [SAMPLES] 个样本低于 [ENTER] 判为暗光，之后连续 [SAMPLES] 个样本高于 [EXIT] 才解除（回差避免来回闪）。
 * 只看原始帧，所以美颜「提亮」不会解除提示，提高曝光或补光才会。
 */
class LowLightDetector {
    var dark = false
        private set
    private var streak = 0

    fun reset() { dark = false; streak = 0 }

    /** @return 状态变化时返回新状态，否则 null。非有限值忽略。 */
    fun onSample(luma: Float): Boolean? {
        if (!luma.isFinite()) return null
        val toward = if (dark) luma > EXIT else luma < ENTER
        streak = if (toward) streak + 1 else 0
        if (streak < SAMPLES) return null
        streak = 0
        dark = !dark
        return dark
    }

    companion object {
        const val ENTER = 0.25f
        const val EXIT = 0.32f
        const val SAMPLES = 3
        const val INTERVAL_MS = 1_000L

        /**
         * RGBA（每像素 4 字节）画面中部 60%×60% 区域的平均亮度（Rec.601，0–1），每 4 个像素取 1 个。
         * 不改变 [pixels] 的 position。
         */
        fun meanLuma(pixels: ByteBuffer, width: Int, height: Int): Float {
            require(width > 0 && height > 0 && pixels.capacity() >= width * height * 4)
            val x0 = width / 5; val x1 = (width - x0).coerceAtLeast(x0 + 1)
            val y0 = height / 5; val y1 = (height - y0).coerceAtLeast(y0 + 1)
            var sum = 0.0
            var count = 0
            var y = y0
            while (y < y1) {
                var x = x0
                while (x < x1) {
                    val i = (y * width + x) * 4
                    val r = pixels.get(i).toInt() and 0xFF
                    val g = pixels.get(i + 1).toInt() and 0xFF
                    val b = pixels.get(i + 2).toInt() and 0xFF
                    sum += 0.299 * r + 0.587 * g + 0.114 * b
                    count++
                    x += 4
                }
                y += 4
            }
            return (sum / count / 255.0).toFloat()
        }
    }
}
