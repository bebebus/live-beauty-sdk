package io.bebebus.livebeauty

import kotlin.math.sqrt

/**
 * 画质调参（仅 Android，纯函数，JVM 可测）：导向滤波 eps 曲线与质感 / 锐化频带的 coring 门限。
 *
 * 能量刻度：GUIDED_DIFF_FRAGMENT 把 1/3 分辨率的 |I − mean|（RGB 欧氏距离）乘 [ENERGY_SCALE] 存进 alpha，
 * 二次模糊后得 s；GUIDED_COEFF_FRAGMENT 以 r = s² / eps 求系数 a = r^1.5 / (1 + r^1.5)。
 * 三通道等幅、标准差 σ0（0–255）的纹理对应 s ≈ [ENERGY_SCALE]·√3·σ0 / 255，所以 eps 取它的平方：
 * 局部起伏小于 σ0 的皮肤被平滑（a → 0），明显大于 σ0 的发丝、睫毛、硬边保留（a → 1）。
 * σ0 随磨皮强度从 [SIGMA0_LO] 线性升到 [SIGMA0_HI]（/255）。门禁 glsl_gate.py 用正则读本文件的常量，
 * 每个常量单独一行、写成 `const val NAME = 数字f`。
 */
internal object BeautyQualityTuning {
    /** GUIDED_DIFF_FRAGMENT 的能量系数，须与着色器里的 `* 2.5` 一致（门禁核对）。 */
    const val ENERGY_SCALE = 2.5f
    /**
     * 导向滤波 1/3 降采样（P1 DOWN）的抽头间距，单位为原始像素：GuidedFilterPrePass 传 uTexel = DOWN_TAP_SCALE / (w, h)。
     * 2/3 时每个双线性抽头按 2:1 混合相邻两个像素，4 个抽头恰好合成 3×3 盒式均值；取 1 则只采到 3×3 的四个角，
     * 1px 发丝周围的晕随相位闪烁。门禁按本常量驱动导向滤波探针，并核对 GuidedFilterPrePass 确实用了它。
     */
    const val DOWN_TAP_SCALE = 0.6666667f
    /** 磨皮 0 时视为「皮肤起伏」的标准差上限（/255）。 */
    const val SIGMA0_LO = 8.0f
    /** 磨皮 1 时视为「皮肤起伏」的标准差上限（/255）。 */
    const val SIGMA0_HI = 16.0f
    /** 质感回填与锐化频带的基础 coring 门限（室内光下噪声带通响应的上沿）。 */
    const val CORE_BASE = 0.005f
    /** 暗光时额外抬高的门限（暗光噪声约是室内的 2–3 倍）。 */
    const val CORE_DARK = 0.008f
    /** 码率压力满时额外抬高的门限：少往编码器里送高频。 */
    const val CORE_PRESSURE = 0.004f

    /** 画面中部亮度高于它不算暗；低于 [DARK_LUMA] 视为全暗，二者之间线性。 */
    private const val BRIGHT_LUMA = 0.30f
    private const val DARK_LUMA = 0.10f

    /**
     * 导向滤波 eps：(ENERGY_SCALE·√3·σ0 / 255)²，σ0 = LO + (HI − LO)·clamp(smoothing)。
     * 磨皮 0 → 0.01845，0.55 → 0.0443，1 → 0.0738；NaN 按 0 处理。
     */
    fun guidedEps(smoothing: Float): Float {
        val t = if (smoothing.isNaN()) 0f else smoothing.coerceIn(0f, 1f)
        val sigma = ENERGY_SCALE * sqrt(3f) * (SIGMA0_LO + (SIGMA0_HI - SIGMA0_LO) * t) / 255f
        return sigma * sigma
    }

    /**
     * 质感回填与锐化的 coring 门限（uGrade.y）：亮处 [CORE_BASE]，画面中部亮度从 0.30 降到 0.10 时线性加
     * [CORE_DARK]，码率压力 0..1 再线性加 [CORE_PRESSURE]。[lightLuma] 为 NaN（还没采到）视为亮，
     * [pressure] 非有限视为 0。
     */
    fun coreThreshold(lightLuma: Float, pressure: Float): Float {
        val dark = if (lightLuma.isNaN()) 0f else ((BRIGHT_LUMA - lightLuma) / (BRIGHT_LUMA - DARK_LUMA)).coerceIn(0f, 1f)
        val load = if (pressure.isNaN()) 0f else pressure.coerceIn(0f, 1f)
        return CORE_BASE + CORE_DARK * dark + CORE_PRESSURE * load
    }
}

/**
 * 导向滤波采样间距 SPREAD 随人脸大小自适应（仅 GL 线程）。1/3 分辨率上的 5 抽头模糊（σ = 2 texel）在全分辨率
 * 约为 [PX_PER_SPREAD]·SPREAD 像素，目标取脸宽的 [FACE_FRACTION]：720p 竖屏常见脸宽 324px 时为 1.08，与旧的
 * 固定值 1.25 接近；钳在 [MIN]..[MAX]，上限避免稀疏抽头混叠。
 *
 * 关键点逐帧抖动不应让半径跟着抖：目标先做 [ALPHA] 的指数平滑，平滑值与当前值相差满 [STEP] 才按 [STEP]
 * 走一步（迟滞）。输入非有限或 ≤ 0（丢脸、网格无效）时保持原值；新上下文 [reset] 回 [DEFAULT]。
 */
internal class GuidedSpread {
    /** 当前采样间距（1/3 分辨率 texel），GuidedFilterPrePass.render 的 spread 参数。 */
    var value = DEFAULT
        private set
    private var smoothed = DEFAULT

    fun reset() {
        value = DEFAULT
        smoothed = DEFAULT
    }

    /** [faceWidthPx]：全分辨率下的脸宽像素（mesh.faceWidth × 帧宽）。@return 更新后的 [value]。 */
    fun update(faceWidthPx: Float): Float {
        if (!faceWidthPx.isFinite() || faceWidthPx <= 0f) return value
        val target = (faceWidthPx * FACE_FRACTION / PX_PER_SPREAD).coerceIn(MIN, MAX)
        smoothed += (target - smoothed) * ALPHA
        // 平滑值逼近钳位端点时只能无限接近，留一点余量，免得最后一步永远走不到。
        val gap = smoothed - value
        if (gap >= STEP - SLACK) value = (value + STEP).coerceAtMost(MAX)
        else if (gap <= -(STEP - SLACK)) value = (value - STEP).coerceAtLeast(MIN)
        return value
    }

    companion object {
        const val DEFAULT = 1.25f
        const val MIN = 0.9f
        const val MAX = 1.6f
        const val FACE_FRACTION = 0.02f
        const val ALPHA = 0.15f
        const val STEP = 0.05f
        /** 1/3 分辨率 σ = 2 texel 的模糊在全分辨率约 6 像素。 */
        const val PX_PER_SPREAD = 6f
        private const val SLACK = 0.001f
    }
}
