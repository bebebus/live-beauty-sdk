package io.bebebus.livebeauty

/**
 * 码率联动（仅 Android，纯函数）：网络差、自适应码率被压低时调整美颜，让有限的码率花在画面内容上。
 * - 降低锐化：低码率下编码器本就丢掉细节，额外锐化只会增加高频、更快出现块状与振铃。
 * - 减少质感回填：毛孔层同样是高频。压力满时质感参数（uParams.w）减半，但着色器的回填系数 mix(0.35, 0.85, w)
 *   有 0.35 的底，回填系数只降 0–29%（质感 0.5 时约 21%，质感 0 时系数不变）。实际回填量是 coring 后的带通乘以系数，
 *   压力满时 BeautyQualityTuning.coreThreshold 把 coring 门限抬高 0.004（压掉小幅带通），所以即使质感为 0 实际回填量
 *   也会下降。低码率下的高频收敛主要靠这次门限上调和上一条的锐化下调。
 * - 加一点时域降噪：传感器噪点最耗码率，降噪后同样码率画面更干净。磨皮强度不变。
 * 压力 = 0（码率 ≥ 上限 75%）到 1（≤ 25%），线性。
 */
internal object BitrateEffectsTuning {
    fun pressure(ratio: Float): Float =
        if (!ratio.isFinite()) 0f else ((0.75f - ratio) / 0.5f).coerceIn(0f, 1f)

    /** 压力满时锐化降到 40%。 */
    fun sharpness(base: Float, pressure: Float): Float = base * (1f - 0.6f * pressure.coerceIn(0f, 1f))

    /** 质感参数（uParams.w）：压力满时减半。是参数减半，不是回填量减半（见类注释）。 */
    fun texture(base: Float, pressure: Float): Float = base * (1f - 0.5f * pressure.coerceIn(0f, 1f))

    /** 压力满时的降噪权重，不超过 TemporalDenoisePass.MAX_WEIGHT（避免低码率下产生明显拖影）。 */
    fun denoise(pressure: Float): Float = 0.16f * pressure.coerceIn(0f, 1f)
}
