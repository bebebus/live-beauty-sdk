package io.bebebus.livebeauty

/**
 * JNI 桥接门面：将 Android OpenGL ES 渲染管线与数学核心绑定至统一 C++17 内核 (`liblivebeauty_jni.so`)。
 *
 * 在未加载原生库（如纯 JVM 单元测试环境）时 [isAvailable] 返回 false，调用方自动安全回退纯 Kotlin 路径。
 */
object NativeBeautyBridge {
    val isAvailable: Boolean = try {
        System.loadLibrary("livebeauty_jni")
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    } catch (_: SecurityException) {
        false
    }

    @JvmStatic external fun nativeCreate(): Long
    @JvmStatic external fun nativeDestroy(handle: Long, contextAlive: Boolean)
    @JvmStatic external fun nativeResetTracking(handle: Long)
    @JvmStatic external fun nativeReleaseGpu(handle: Long, contextAlive: Boolean)
    @JvmStatic external fun nativeSetAssets(
        handle: Long,
        lutGlTex: Int,
        blushGlTex: Int,
        eyebrowGlTex: Int,
        eyeMakeupGlTex: Int,
        contourHighlightGlTex: Int,
        irisGlTex: Int,
    )
    /** 配置无绿幕虚拟背景替换（mode: "none" / "blur" / "virtual_image" / "virtual_blur"）。 */
    @JvmStatic external fun nativeSetVirtualBackground(
        handle: Long,
        bgGlTex: Int,
        bgAspect: Float,
        mode: String,
        strength: Float,
    )
    /** 配置 2D 人脸锚定 AR 挂件贴纸层（slot 0..3；anchor: 0=头饰 1=眼镜 2=鼻贴 3=颊贴 4=全脸）。 */
    @JvmStatic external fun nativeSetSticker(
        handle: Long,
        slot: Int,
        enabled: Boolean,
        anchor: Int,
        texGlId: Int,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        rotationRad: Float,
        opacity: Float,
        textureAspect: Float,
    )
    @JvmStatic external fun nativeClearStickers(handle: Long)
    /** 配置高级管线选项：maxFaces (1..2 双人同屏) 与 frameDelayFrames (0..2 零滞后帧对齐环)。 */
    @JvmStatic external fun nativeSetPipelineOptions(handle: Long, maxFaces: Int, frameDelayFrames: Int)
    /** 批量提交 1..2 张人脸原始 468/478 点坐标（flatMpXy = faceCount * pointsPerFace * 2）。 */
    @JvmStatic external fun nativeSubmitMultiFaceLandmarks(
        handle: Long,
        sampleTsMs: Long,
        flatMpXy: FloatArray,
        faceCount: Int,
        pointsPerFace: Int,
        width: Int,
        height: Int,
        receivedMs: Long,
    ): Int
    @JvmStatic external fun nativeSetConfig(
        handle: Long,
        enabled: Boolean,
        skinOnly: Boolean,
        compareOriginal: Boolean,
        scalars: FloatArray,
        filter: String,
        lutUrl: String,
        lipstickColor: String,
        lipstickFinish: String,
        blushStyle: String,
        blushColor: String,
        eyebrowStyle: String,
        eyebrowColor: String,
        eyeshadowColor: String,
        contactLensStyle: String,
    )
    @JvmStatic external fun nativeSubmitFaceLandmarks(
        handle: Long,
        sampleTsMs: Long,
        mpXy: FloatArray,
        pointCount: Int,
        width: Int,
        height: Int,
        receivedMs: Long,
    ): Boolean
    @JvmStatic external fun nativeSubmitFaceMiss(handle: Long, sampleTsMs: Long)
    @JvmStatic external fun nativeSubmitBodyPose(
        handle: Long,
        sampleTsMs: Long,
        poseXyv: FloatArray,
        pointCount: Int,
        width: Int,
        height: Int,
    ): Boolean
    @JvmStatic external fun nativeSelectFaceRoi(
        handle: Long,
        width: Int,
        height: Int,
        nowMs: Long,
        outRect4: FloatArray,
    )
    @JvmStatic external fun nativeProcessFrame(
        handle: Long,
        inputGlTex: Int,
        width: Int,
        height: Int,
        frameTsMs: Long,
        nowMs: Long,
        personMaskGlTex: Int,
        lightLuma: Float,
        bitrateRatio: Float,
    ): Int
    /** 扩展版处理入口：同时接入头发语义分割遮罩（防磨皮保护 + 背景飞发合并）。 */
    @JvmStatic external fun nativeProcessFrameEx(
        handle: Long,
        inputGlTex: Int,
        width: Int,
        height: Int,
        frameTsMs: Long,
        nowMs: Long,
        personMaskGlTex: Int,
        hairMaskGlTex: Int,
        hairMaskStampMs: Long,
        lightLuma: Float,
        bitrateRatio: Float,
    ): Int
    /** Android CameraX/Camera2/WebRTC SurfaceTexture GL_TEXTURE_EXTERNAL_OES 一步直入入口（内部自动 OES->2D）。 */
    @JvmStatic external fun nativeProcessOesFrame(
        handle: Long,
        oesGlTex: Int,
        stMatrix16: FloatArray?,
        width: Int,
        height: Int,
        frameTsMs: Long,
        nowMs: Long,
        personMaskGlTex: Int,
        hairMaskGlTex: Int,
        hairMaskStampMs: Long,
        lightLuma: Float,
        bitrateRatio: Float,
    ): Int
    /** 读取 C++17 内核实时性能遥测：[lastMs, emaMs, fps, loadLevel, facePresence, hairShown, activePassesMask, frameCount]。 */
    @JvmStatic external fun nativeGetStats(handle: Long, outStats8: FloatArray)

    /** 将 [VideoEffectsConfig] 的 45 个标量打包为扁平数组，供 [nativeSetConfig] 单次 JNI 传参。 */
    fun packConfigScalars(cfg: VideoEffectsConfig, out: FloatArray = FloatArray(45)): FloatArray {
        out[0] = cfg.smoothing
        out[1] = cfg.brightness
        out[2] = cfg.sharpness
        out[3] = cfg.saturation
        out[4] = cfg.filterIntensity
        out[5] = cfg.faceSlim
        out[6] = cfg.eyeEnlarge
        out[7] = cfg.whiten
        out[8] = cfg.rosy
        out[9] = cfg.waistSlim
        out[10] = cfg.bustEnhance
        out[11] = cfg.legLengthen
        out[12] = cfg.shoulderSlim
        out[13] = cfg.neckLengthen
        out[14] = cfg.chinShape
        out[15] = cfg.foreheadShape
        out[16] = cfg.noseSlim
        out[17] = cfg.vFace
        out[18] = cfg.faceNarrow
        out[19] = cfg.faceSmall
        out[20] = cfg.cheekboneSlim
        out[21] = cfg.eyeDistance
        out[22] = cfg.eyeCorner
        out[23] = cfg.noseLength
        out[24] = cfg.mouthSize
        out[25] = cfg.mouthSmile
        out[26] = cfg.cranialTop
        out[27] = cfg.darkCircles
        out[28] = cfg.brightEyes
        out[29] = cfg.teethWhiten
        out[30] = cfg.nasolabial
        out[31] = cfg.backgroundBlur
        out[32] = cfg.lutIntensity
        out[33] = cfg.skinTexture
        out[34] = cfg.radiance
        out[35] = cfg.lipstick
        out[36] = cfg.blush
        out[37] = cfg.eyebrow
        out[38] = cfg.eyeshadow
        out[39] = cfg.eyeliner
        out[40] = cfg.aegyoSal
        out[41] = cfg.contour
        out[42] = cfg.highlight
        out[43] = cfg.contactLens
        out[44] = cfg.catchlight
        return out
    }
}
