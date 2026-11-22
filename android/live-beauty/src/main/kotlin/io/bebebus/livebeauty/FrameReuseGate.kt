package io.bebebus.livebeauty

/**
 * VideoEffectsFilter 的同帧复用判定（规格 §5.4），仅 GL 线程。
 *
 * RootEncoder 2.7.5 的 OpenGlView.draw(Z) 在同一次绘制里先 updateTexImage + drawSource 各一次，再调用
 * MainRender.drawFilters(true)（预览）与 drawFilters(false)（编码/拍照），两次之间只有预览上屏。单滤镜时
 * reOrderFilters 两次都把相机纹理接成 previousTexId，滤镜 FBO 由 MainRender.initGl 按编码尺寸一次性建立，
 * 中间没有任何写入本 FBO 的操作。所以输入（相机帧时间戳、参数、输入纹理、尺寸、健康状态）完全相同时，
 * FBO 里已经是本次要画的结果，可以直接沿用。ForceRenderer 在相机没有新帧时强制重画同一帧，同样适用。
 *
 * 时间戳为 0（取不到或相机尚未出帧）时一律不复用。只在一次绘制正常结束后 [rendered]，异常中断的不记。
 */
internal class FrameReuseGate {
    private var valid = false
    private var stamp = 0L
    private var config: VideoEffectsConfig? = null
    private var input = 0
    private var width = 0
    private var height = 0
    private var healthy = false
    private var faceHealthy = false

    fun canReuse(
        stamp: Long, config: VideoEffectsConfig, input: Int, width: Int, height: Int,
        healthy: Boolean, faceHealthy: Boolean,
    ): Boolean = valid && stamp != 0L && stamp == this.stamp && config == this.config &&
        input == this.input && width == this.width && height == this.height &&
        healthy == this.healthy && faceHealthy == this.faceHealthy

    fun rendered(
        stamp: Long, config: VideoEffectsConfig, input: Int, width: Int, height: Int,
        healthy: Boolean, faceHealthy: Boolean,
    ) {
        if (stamp == 0L) { invalidate(); return }
        this.stamp = stamp; this.config = config; this.input = input
        this.width = width; this.height = height
        this.healthy = healthy; this.faceHealthy = faceHealthy
        valid = true
    }

    fun invalidate() {
        valid = false
        config = null
    }
}
