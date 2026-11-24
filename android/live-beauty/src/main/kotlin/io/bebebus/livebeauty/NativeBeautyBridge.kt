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
    /** 配置高级管线选项：maxFaces (1..2 双人同屏) 与 frameDelayFrames (0..4 零滞后帧对齐环)。 */
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
    /** 配置 P1/P4 专业选项：3D 打光风格与芯片分级档位。 */
    @JvmStatic external fun nativeSetProOptions(handle: Long, relightStyle: String, performanceTier: String)
    /** v3.4: 配置实时沙龙染发、天使光圈发质高光与半精度中间管线开关。 */
    @JvmStatic external fun nativeSetHairStyle(
        handle: Long,
        hairDye: Float,
        hairDyeColor: String,
        hairShine: Float,
        preferHalfFloat: Boolean = true,
    )
    /** 上报宿主机 GPU 渲染器名、CPU 核心数与总内存（MB），供 DeviceTierPolicy 自动分级。 */
    @JvmStatic external fun nativeSetHardwareProfile(handle: Long, gpuRenderer: String, cpuCores: Int, totalRamMb: Int)
    /** 上报当前环境光照 RGB 白平衡增益（0.65..1.35），供美妆自然融入环境色温。 */
    @JvmStatic external fun nativeSetAmbientRgb(handle: Long, r: Float, g: Float, b: Float)
    /** 绑定可选 R8 人脸解析/手部遮挡掩码纹理（255=可见肤区，0=遮挡区）。 */
    @JvmStatic external fun nativeSetOcclusionMask(handle: Long, occlusionMaskGlTex: Int)
    /** 绑定可选人脸 ROI 神经网络残差纹理（RGBA8，0.5=零残差，A=置信度）。 */
    @JvmStatic external fun nativeSetAiResidual(
        handle: Long,
        residualGlTex: Int,
        roiLeft: Float,
        roiTop: Float,
        roiWidth: Float,
        roiHeight: Float,
        strength: Float,
    )
    /** 配置 3D 头部遮挡 AR 挂件层（slot 0..3，支持弹簧摆物理与头部深度遮挡）。 */
    @JvmStatic external fun nativeSetAr3dAccessory(
        handle: Long,
        slot: Int,
        enabled: Boolean,
        anchor: Int,
        texGlId: Int,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        offsetZ: Float,
        yawRad: Float,
        pitchRad: Float,
        rollRad: Float,
        opacity: Float,
        textureAspect: Float,
        springStiffness: Float,
        springDamping: Float,
        headOcclusion: Boolean,
    )
    @JvmStatic external fun nativeSetAr3dAccessoryEx(
        handle: Long,
        slot: Int,
        enabled: Boolean,
        anchor: Int,
        meshPreset: Int,
        triggerRule: Int,
        texGlId: Int,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        offsetZ: Float,
        yawRad: Float,
        pitchRad: Float,
        rollRad: Float,
        opacity: Float,
        textureAspect: Float,
        springStiffness: Float,
        springDamping: Float,
        headOcclusion: Boolean,
    )
    @JvmStatic external fun nativeClearAr3dAccessories(handle: Long)
    /** v3.2: 提交 MediaPipe 52 维蒸馏表情系数（微笑/张嘴/挑眉/眨眼/鼓腮/吐舌），由 C++ 表情总线驱动挂件脉冲。 */
    @JvmStatic external fun nativeSubmitFaceBlendshapes(
        handle: Long,
        smile: Float,
        jawOpen: Float,
        browUp: Float,
        blink: Float,
        cheekPuff: Float,
        tongueOut: Float,
    )
    @JvmStatic external fun nativeSetExpressionReactive(handle: Long, enabled: Boolean)
    /** v3.2: 提交 top-down RGBA DirectByteBuffer，在 C++17 + ARM NEON 内核完成前景异物遮挡与 ROI 皮肤残差计算并自动上屏。 */
    @JvmStatic external fun nativeSubmitAuxRgbaSample(
        handle: Long,
        rgbaDirectBuffer: java.nio.ByteBuffer,
        srcWidth: Int,
        srcHeight: Int,
        retouchStrength: Float,
    ): Boolean
    /** v3.3: 提交 19 类 CelebAMask-HQ / BiSeNetV2 / SegFormer 人脸语义分割类别图（0..19），自动驱动防眼镜/刘海/露齿遮挡与头发保护。 */
    @JvmStatic external fun nativeSubmitFaceParsingClassMap(
        handle: Long,
        classIds: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        sampleTsMs: Long,
    ): Boolean
    /** v3.3: 提交外部 ONNX/TFLite 微型 UNet 皮肤精修残差浮点张量（3xHxW 或 4xHxW NCHW）。 */
    @JvmStatic external fun nativeSubmitNeuralResidualTensor(
        handle: Long,
        tensorNchw: FloatArray,
        channels: Int,
        srcWidth: Int,
        srcHeight: Int,
        roiLeft: Float,
        roiTop: Float,
        roiWidth: Float,
        roiHeight: Float,
        strength: Float,
    ): Boolean
    /** v3.5: 解析并应用声明式 `.lbfx` 特效包 JSON 清单（美颜滤镜、染发、2D/3D 挂件及表情触发规则）。 */
    @JvmStatic external fun nativeApplyEffectBundleJson(handle: Long, manifestJson: String): Boolean
    /** v3.5.1: 解析并应用带 GL 纹理资产映射表的 `.lbfx` 特效包 JSON 清单。 */
    @JvmStatic external fun nativeApplyEffectBundleJsonWithAssets(
        handle: Long,
        manifestJson: String,
        assetPaths: Array<String>,
        texGlIds: IntArray,
    ): Boolean
    /** 读取 C++17 内核实时性能遥测：[lastMs, emaMs, fps, loadLevel, facePresence, hairShown, activePassesMask, frameCount, activeFaces, displayTs, resolvedTier]。 */
    @JvmStatic external fun nativeGetStats(handle: Long, outStats: FloatArray)
    /** v3.6: 配置演播室绿幕/蓝幕 YCbCr 色度抠像（keyColor: "green" / "blue"）与边缘抑色（despill）。 */
    @JvmStatic external fun nativeSetChromaKey(
        handle: Long,
        keyColor: String,
        similarity: Float,
        smoothness: Float,
        despill: Float,
    )
    /** v3.6: 提交 MediaPipe selfie_multiclass 6 分类 NHWC float logits（bg/hair/body_skin/face_skin/clothes/accessories）。 */
    @JvmStatic external fun nativeSubmitMulticlassLogits(
        handle: Long,
        logitsNhwc6: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        sampleTsMs: Long,
    ): Boolean
    /** v3.6: 触发交互式 2D SDF 手势粒子爆发（gestureId: 1=heart, 2=thumbs_up, 3=victory, 4=open_palm）。 */
    @JvmStatic external fun nativeTriggerGesture(
        handle: Long,
        gestureId: Int,
        normX: Float,
        normY: Float,
        nowMs: Long,
    )
    /** v3.6: 更新实时双掌归一化中心（top-down [0..1]，无效传负值），防止瘦腰拉伸手掌与手指。 */
    @JvmStatic external fun nativeSetHandShield(
        handle: Long,
        hand0X: Float,
        hand0Y: Float,
        hand1X: Float,
        hand1Y: Float,
    )
    /** v3.6.2: 上报宿主陀螺仪/加速度计快速晃动状态，瞬时清空时域帧/人像遮罩历史并抑制过期异步掩码拖影。 */
    @JvmStatic external fun nativeSetCameraShaking(handle: Long, shaking: Boolean)

    /** 将 [VideoEffectsConfig] 的 51 个标量打包为扁平数组，供 [nativeSetConfig] 单次 JNI 传参。 */
    fun packConfigScalars(cfg: VideoEffectsConfig, out: FloatArray = FloatArray(51)): FloatArray {
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
        if (out.size >= 51) {
            out[45] = cfg.relight
            out[46] = cfg.oilControl
            out[47] = cfg.neckTone
            out[48] = cfg.ambientMakeupBlend
            out[49] = cfg.matteTemporal
            out[50] = cfg.blemishRemove
        }
        return out
    }
}
