package io.bebebus.livebeauty

import android.content.Context

/**
 * Bebebus Live Beauty SDK 面向原生 Android 音视频管线（WebRTC、Agora、Zego、TRTC、CameraX、MediaCodec 等）的高层统一入口。
 *
 * 线程约定：
 * - [updateConfig] / [resetTracking] 可在任意线程调用（内部无锁原子快照切换）；
 * - [onGlCreated] / [configure] / [process] / [onGlReleased] 在宿主 OpenGL ES 渲染线程调用。
 */
class LiveBeautyProcessor(
    private val appContext: Context? = null,
    onFailure: () -> Unit = {},
    onFaceFailure: () -> Unit = {},
    onLoadReduced: (Int) -> Unit = {},
    onLegsHidden: (Boolean) -> Unit = {},
    onLowLight: (Boolean) -> Unit = {},
    streaming: () -> Boolean = { false },
    bitrateRatio: () -> Float = { 1f },
    useNativeCppCore: Boolean = true,
) : BeautyEngine {

    private var activeCppEngine: LiveBeautyCppEngine? = null

    val filter = VideoEffectsFilter(
        onFailure = onFailure,
        onFaceFailure = onFaceFailure,
        onLoadReduced = onLoadReduced,
        onLegsHidden = onLegsHidden,
        lutCacheDir = appContext?.cacheDir,
        lutAssets = appContext?.assets,
        onLowLight = onLowLight,
        bitrateRatio = bitrateRatio,
        streaming = streaming,
        engineFactory = if (useNativeCppCore && NativeBeautyBridge.isAvailable) {
            {
                LiveBeautyCppEngine(
                    bitrateRatio = bitrateRatio,
                    lutCacheDir = appContext?.cacheDir,
                    lutAssets = appContext?.assets,
                    onFaceFailure = onFaceFailure,
                    onLowLight = onLowLight,
                ).also { activeCppEngine = it }
            }
        } else null,
    )

    override val latencyFrames: Int
        get() = ((filter.videoDelayMs + 16L) / 33L).toInt().coerceIn(0, EngineStamps.MAX_LATENCY_FRAMES)

    /** 任意线程安全更新美颜参数快照。 */
    fun updateConfig(config: VideoEffectsConfig) {
        filter.config = config
    }

    /** 配置无绿幕虚拟背景替换纹理（GL_TEXTURE_2D，需启用 C++17 原生内核）。 */
    fun setVirtualBackgroundTexture(
        glTextureId: Int,
        aspect: Float = 0f,
        mode: VirtualBackgroundMode = VirtualBackgroundMode.VIRTUAL_IMAGE,
        strength: Float = 1.0f,
    ) {
        activeCppEngine?.setVirtualBackgroundTexture(glTextureId, aspect, mode, strength)
    }

    /** 配置人脸锚定 2D AR 挂件贴纸层（slot 0..3，传 null 清除该层）。 */
    fun setSticker(slot: Int, sticker: LiveBeautySticker?) {
        activeCppEngine?.setSticker(slot, sticker)
    }

    /** 清除全部 2D AR 贴纸层。 */
    fun clearStickers() {
        activeCppEngine?.clearStickers()
    }

    /** 配置双人同屏人数上限（1..2）与 C++ 零滞后帧对齐环帧数（0..2）。 */
    fun setPipelineOptions(maxFaces: Int = 1, frameDelayFrames: Int = 0) {
        activeCppEngine?.setPipelineOptions(maxFaces, frameDelayFrames)
    }

    /** 读取 C++17 内核实时性能遥测快照。 */
    fun perfStats(): BeautyPerfStats = activeCppEngine?.perfStats() ?: BeautyPerfStats()

    /**
     * CameraX / Camera2 / WebRTC `GL_TEXTURE_EXTERNAL_OES` 一步直入处理（GL 线程）。
     * 若当前处于纯 Kotlin 回退模式则返回 0，请使用 `process(glTexture2D, ...)`。
     */
    fun processOes(
        oesTexture: Int,
        stMatrix16: FloatArray?,
        width: Int,
        height: Int,
        timestampNs: Long,
    ): Int = activeCppEngine?.processOes(oesTexture, stMatrix16, width, height, timestampNs) ?: 0

    /** 切换前后置摄像头或重开预览时重置人脸/人像/美发跟踪代际。 */
    override fun resetTracking() {
        filter.resetFaceTracking()
    }

    override fun onGlCreated(context: Context) {
        filter.initGlFilter(context.applicationContext ?: context)
    }

    override fun configure(config: VideoEffectsConfig) {
        filter.config = config
    }

    override fun process(inputTexture: Int, width: Int, height: Int, timestampNs: Long): Int {
        return filter.processTexture(inputTexture, width, height, timestampNs)
    }

    override fun onGlReleased(contextAlive: Boolean) {
        filter.release()
        if (contextAlive) {
            filter.releaseRenderTarget()
        }
    }
}

/** 开箱即用的秀场级预设参数模板。 */
object LiveBeautyPresets {
    /** 官方在线体验与实机测试站点地址。 */
    const val DEMO_SITE_URL: String = "https://dh.video"

    /** 原画直通（零效果）。 */
    val original: VideoEffectsConfig = VideoEffectsConfig.original

    /** 自然原生清透档（日常连麦/才艺开播推荐默认）。 */
    val naturalStudio: VideoEffectsConfig = VideoEffectsConfig(
        enabled = true,
        smoothing = 0.52f,
        brightness = 0.10f,
        filter = "none",
        skinOnly = true,
        skinTexture = 0.45f,
        whiten = 0.32f,
        rosy = 0.16f,
        radiance = 0.25f,
        sharpness = 0.20f,
        faceSlim = 0.35f,
        vFace = 0.25f,
        eyeEnlarge = 0.28f,
        noseSlim = 0.20f,
        darkCircles = 0.35f,
        brightEyes = 0.30f,
        teethWhiten = 0.25f,
        lutUrl = "asset://luts/natural_studio.png",
        lutIntensity = 0.45f,
    )

    /** 冷白奶油瓷肌档（女团/带货高显白）。 */
    val clearColdWhite: VideoEffectsConfig = naturalStudio.copy(
        smoothing = 0.68f,
        skinTexture = 0.38f,
        whiten = 0.58f,
        rosy = 0.22f,
        radiance = 0.38f,
        faceSlim = 0.52f,
        vFace = 0.42f,
        faceSmall = 0.20f,
        eyeEnlarge = 0.42f,
        lipstick = 0.45f,
        lipstickColor = "rose_bean",
        lipstickFinish = "glossy",
        blush = 0.30f,
        blushStyle = "apple",
        blushColor = "peach",
        catchlight = 0.35f,
        lutUrl = "asset://luts/clear_white.png",
        lutIntensity = 0.55f,
    )

    /** 韩系水光蜜桃妆（全套微整 + 水光唇釉 + 卧蚕眼神光）。 */
    val koreanWaterGlow: VideoEffectsConfig = clearColdWhite.copy(
        rosy = 0.30f,
        radiance = 0.52f,
        aegyoSal = 0.40f,
        eyeshadow = 0.32f,
        eyeshadowColor = "peach_pink",
        eyeliner = 0.28f,
        contactLens = 0.38f,
        contactLensStyle = "hazel",
        catchlight = 0.48f,
        lutUrl = "asset://luts/korean_glow.png",
        lutIntensity = 0.50f,
    )

    /** 男士清爽质感档（保留胡须毛流与面部骨相，仅做干净匀肤与消暗沉）。 */
    val mensClean: VideoEffectsConfig = VideoEffectsConfig(
        enabled = true,
        smoothing = 0.38f,
        brightness = 0.08f,
        filter = "none",
        skinOnly = true,
        skinTexture = 0.65f,
        whiten = 0.18f,
        sharpness = 0.25f,
        faceSlim = 0.20f,
        cheekboneSlim = 0.15f,
        noseSlim = 0.15f,
        darkCircles = 0.42f,
        brightEyes = 0.25f,
        lutUrl = "asset://luts/natural_studio.png",
        lutIntensity = 0.35f,
    )
}
