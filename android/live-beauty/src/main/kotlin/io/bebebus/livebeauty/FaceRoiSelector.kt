package io.bebebus.livebeauty

/**
 * 远景/半身场景自适应人脸 ROI 裁剪选择器（纯逻辑，零 Android/GL 依赖，JVM 可测）。
 *
 * 痛点：半身、站立才艺或舞蹈直播时，主播距离镜头 1.2–2.2 m，人脸宽度仅占画面宽 `12%–28%`；
 * 在共享 `216×384` 小图中人脸仅有 `26–60 px`，导致 MediaPipe 关键点量化抖动放大且侧头易断检。
 *
 * 策略：
 * - 迟滞门限：当 `faceWidth <= ENTER_FACE_WIDTH`（默认 `0.30`，且 `>= MIN_FACE_WIDTH`）时激活各向同性正方形裁剪，
 *   当 `faceWidth >= EXIT_FACE_WIDTH`（`0.36`）自动退回全幅采样，避免临界距离来回跳变；
 * - 各向同性无拉伸：裁剪框满足 `cropH = cropW * aspect`（当宽屏 `aspect > 1` 时满足 `cropW = cropH / aspect`），
 *   渲染到正方形 `256×256` FBO 时像素纵横比严格为 `1:1`，不引入任何几何压扁；
 * - 零死锁安全兜底：一旦某次 ROI 推理未检出人脸（[onMiss]）或距上次有效人脸超过 [STALE_MS]，下一帧立即退回全画幅检索。
 */
internal class FaceRoiSelector {
    /** 归一化顶向下（MediaPipe 图像系：`x` 左→右，`y` 上→下）裁剪矩形。 */
    data class CropRect(
        val left: Float,
        val top: Float,
        val width: Float,
        val height: Float,
    ) {
        val isFullFrame: Boolean
            get() = left == 0f && top == 0f && width == 1f && height == 1f

        /** 将 ROI 内 MediaPipe 归一化坐标 `(rx, ry)` 反投影回全画幅 MediaPipe 归一化坐标。 */
        fun unprojectX(rx: Float): Float = left + rx * width
        fun unprojectY(ry: Float): Float = top + ry * height

        companion object {
            val FULL = CropRect(0f, 0f, 1f, 1f)
        }
    }

    private var active = false
    private var smoothCx = 0.5f
    private var smoothCy = 0.5f
    private var smoothFaceWidth = 0f
    private var lastSampleMs = Long.MIN_VALUE
    private var lastReceivedMs = Long.MIN_VALUE

    val isActive: Boolean get() = active

    fun reset() {
        active = false
        smoothCx = 0.5f
        smoothCy = 0.5f
        smoothFaceWidth = 0f
        lastSampleMs = Long.MIN_VALUE
        lastReceivedMs = Long.MIN_VALUE
    }

    /**
     * 更新最新检出的人脸位置（[glCenterX]、[glCenterY] 为 GL 纹理坐标，`y` 向上；[faceWidth] 为度量脸宽）。
     *
     * @param sampleMs 该帧人脸样本的采集时刻
     * @param receivedMs 该次推理结果到达渲染线程的时刻（慢机 CPU 推理延迟 120–200 ms 时，以到达时刻判 [STALE_MS] 避免 ROI 与全幅来回抖动，
     *                   同时以 [MAX_SAMPLE_AGE_MS] 限制采样绝对年龄）
     */
    fun onFace(
        glCenterX: Float,
        glCenterY: Float,
        faceWidth: Float,
        sampleMs: Long,
        receivedMs: Long = sampleMs,
    ) {
        if (!glCenterX.isFinite() || !glCenterY.isFinite() || !faceWidth.isFinite() || faceWidth < MIN_FACE_WIDTH) {
            return
        }
        if (receivedMs < sampleMs || receivedMs - sampleMs > MAX_SAMPLE_AGE_MS) {
            return
        }
        if (lastSampleMs != Long.MIN_VALUE && sampleMs < lastSampleMs) {
            return
        }
        val mpCx = glCenterX.coerceIn(0f, 1f)
        val mpCy = (1f - glCenterY).coerceIn(0f, 1f)
        if (lastReceivedMs == Long.MIN_VALUE || receivedMs < lastReceivedMs || receivedMs - lastReceivedMs > STALE_MS) {
            smoothCx = mpCx
            smoothCy = mpCy
            smoothFaceWidth = faceWidth
        } else {
            smoothCx += (mpCx - smoothCx) * EMA_ALPHA
            smoothCy += (mpCy - smoothCy) * EMA_ALPHA
            smoothFaceWidth += (faceWidth - smoothFaceWidth) * EMA_ALPHA
        }
        lastSampleMs = sampleMs
        lastReceivedMs = receivedMs
        active = if (active) {
            smoothFaceWidth < EXIT_FACE_WIDTH
        } else {
            smoothFaceWidth <= ENTER_FACE_WIDTH
        }
    }

    /** ROI 裁剪帧未检出人脸时立即退出 ROI，保证下一帧全画幅兜底重捕。 */
    fun onMiss() {
        active = false
        lastSampleMs = Long.MIN_VALUE
        lastReceivedMs = Long.MIN_VALUE
    }

    /** 计算本帧采样使用的人脸裁剪框；近景或状态过期时返回 [CropRect.FULL]。 */
    fun select(width: Int, height: Int, nowMs: Long): CropRect {
        if (!active || width <= 0 || height <= 0 ||
            lastReceivedMs == Long.MIN_VALUE || nowMs < lastReceivedMs || nowMs - lastReceivedMs > STALE_MS ||
            lastSampleMs == Long.MIN_VALUE || nowMs < lastSampleMs || nowMs - lastSampleMs > MAX_SAMPLE_AGE_MS
        ) {
            active = false
            return CropRect.FULL
        }
        val aspect = width.toFloat() / height.toFloat()
        if (!aspect.isFinite() || aspect <= 0f) return CropRect.FULL

        // 目标裁剪边长覆盖约 2.8 倍脸宽（预留快速转头/走位余量），并保证各向同性正方形不越界。
        val desiredW = (smoothFaceWidth * EXPAND_FACTOR).coerceIn(MIN_CROP_SPAN, MAX_CROP_SPAN)
        val cropW: Float
        val cropH: Float
        if (aspect <= 1f) {
            // 竖屏（常见 9:16）：cropH = cropW * aspect <= cropW，两者必 <= 1。
            cropW = desiredW.coerceAtMost(1f)
            cropH = (cropW * aspect).coerceIn(MIN_CROP_SPAN * aspect, 1f)
        } else {
            // 横屏（16:9）：高度是短边，先钳 cropH 再反推 cropW = cropH / aspect。
            cropH = (desiredW * aspect).coerceIn(MIN_CROP_SPAN, MAX_CROP_SPAN)
            cropW = (cropH / aspect).coerceAtMost(1f)
        }
        if (cropW >= 0.96f || cropH >= 0.96f) return CropRect.FULL

        val left = (smoothCx - 0.5f * cropW).coerceIn(0f, 1f - cropW)
        val top = (smoothCy - 0.5f * cropH).coerceIn(0f, 1f - cropH)
        return CropRect(left, top, cropW, cropH)
    }

    companion object {
        const val MIN_FACE_WIDTH = 0.045f
        const val ENTER_FACE_WIDTH = 0.30f
        const val EXIT_FACE_WIDTH = 0.36f
        const val EXPAND_FACTOR = 2.80f
        const val MIN_CROP_SPAN = 0.28f
        const val MAX_CROP_SPAN = 0.85f
        const val ROI_SIDE_PX = 256
        const val STALE_MS = 350L
        const val MAX_SAMPLE_AGE_MS = 650L
        private const val EMA_ALPHA = 0.45f
    }
}

/** OpenGL ES 3.0 PBO 能力解析辅助（纯函数，JVM 可测）。 */
internal object PboSupport {
    /** 解析 `glGetString(GL_VERSION)`（如 `"OpenGL ES 3.2 V@0502.0"`），主版本号 >= 3 时支持 `GL_PIXEL_PACK_BUFFER`。 */
    fun isEs3Version(glVersion: String?): Boolean {
        if (glVersion.isNullOrBlank()) return false
        val idx = glVersion.indexOf("OpenGL ES ")
        val tail = if (idx >= 0) glVersion.substring(idx + "OpenGL ES ".length).trimStart() else glVersion.trimStart()
        val majorEnd = tail.indexOfFirst { !it.isDigit() }.let { if (it < 0) tail.length else it }
        if (majorEnd == 0) return false
        val major = tail.substring(0, majorEnd).toIntOrNull() ?: return false
        return major >= 3
    }
}
