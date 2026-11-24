package io.bebebus.livebeauty

/** 无绿幕人像背景模式。 */
enum class VirtualBackgroundMode(val wireName: String) {
    /** 跟随 `VideoEffectsConfig.backgroundBlur`（>0 为人像虚化，0 为关闭）。 */
    NONE("none"),
    /** 人像背景虚化（Bokeh）。 */
    BLUR("blur"),
    /** 虚拟背景图片替换（自动居中 Aspect-Fill，不拉伸变形）。 */
    VIRTUAL_IMAGE("virtual_image"),
    /** 虚拟背景图片 + 柔和虚化。 */
    VIRTUAL_BLUR("virtual_blur"),
}

/** 2D AR 人脸挂件贴纸解剖锚点（与 C++17 `livebeauty::math::StickerAnchor` 一一对应）。 */
enum class StickerAnchor(val id: Int) {
    /** 皇冠 / 兔耳 / 头饰（锚定额顶上方）。 */
    FOREHEAD_CROWN(0),
    /** 眼镜 / 墨镜（锚定双瞳连线中点，自动随头旋转缩放）。 */
    EYES_BRIDGE(1),
    /** 鼻贴 / 动物鼻（锚定鼻尖点 1）。 */
    NOSE_TIP(2),
    /** 面颊贴花（锚定面中颧骨区）。 */
    CHEEKS(3),
    /** 整脸面具（锚定脸椭圆中心）。 */
    FULL_FACE(4),
}

/** 单个 2D AR 人脸贴纸层配置（最多同屏 4 层，slot 0..3）。 */
data class LiveBeautySticker(
    val enabled: Boolean = true,
    val anchor: StickerAnchor = StickerAnchor.FOREHEAD_CROWN,
    val glTextureId: Int = 0,
    val scale: Float = 1.0f,
    val offsetX: Float = 0.0f,
    val offsetY: Float = 0.0f,
    val rotationRad: Float = 0.0f,
    val opacity: Float = 1.0f,
    val textureAspect: Float = 1.0f,
)

/** 3D 头部遮挡 AR 挂件锚点（与 C++17 `livebeauty::math::Ar3dAnchor` 一一对应）。 */
enum class Ar3dAnchor(val id: Int) {
    HEAD_CROWN(0),
    GLASSES_BRIDGE(1),
    LEFT_EAR(2),
    RIGHT_EAR(3),
    NOSE_TIP(4),
    CHIN(5),
}

/** 3D AR 曲面网格预设（与 C++17 `livebeauty::math::Ar3dMeshPreset` 一一对应）。 */
enum class Ar3dMeshPreset(val id: Int) {
    BILLBOARD_QUAD(0),
    CURVED_GLASSES(1),
    CURVED_TIARA(2),
}

/** 单个 3D AR 挂件层配置（支持 3D 透视旋转、曲面网格、头部深度遮挡与二阶弹簧摆物理）。 */
data class LiveBeautyAr3dAccessory @JvmOverloads constructor(
    val enabled: Boolean = true,
    val anchor: Ar3dAnchor = Ar3dAnchor.HEAD_CROWN,
    val glTextureId: Int = 0,
    val scale: Float = 1.0f,
    val offsetX: Float = 0.0f,
    val offsetY: Float = 0.0f,
    val offsetZ: Float = 0.0f,
    val yawRad: Float = 0.0f,
    val pitchRad: Float = 0.0f,
    val rollRad: Float = 0.0f,
    val opacity: Float = 1.0f,
    val textureAspect: Float = 1.0f,
    val springStiffness: Float = 0.0f,
    val springDamping: Float = 4.5f,
    val headOcclusion: Boolean = true,
    val meshPreset: Ar3dMeshPreset = Ar3dMeshPreset.BILLBOARD_QUAD,
    val triggerRule: Int = 0,
)

/** C++17 渲染内核实时性能遥测快照。 */
data class BeautyPerfStats(
    val lastFrameMs: Float = 0f,
    val emaFrameMs: Float = 0f,
    val fps: Float = 0f,
    val loadLevel: Int = 0,
    val facePresence: Float = 0f,
    val hairShown: Float = 0f,
    val activePassesMask: Int = 0,
    val frameCount: Long = 0L,
    val activeFaces: Int = 0,
    val resolvedTier: Int = 2,
) {
    companion object {
        const val RAW_SIZE = 11

        fun fromRaw(raw: FloatArray): BeautyPerfStats {
            if (raw.size < 8) return BeautyPerfStats()
            return BeautyPerfStats(
                lastFrameMs = raw[0],
                emaFrameMs = raw[1],
                fps = raw[2],
                loadLevel = raw[3].toInt(),
                facePresence = raw[4],
                hairShown = raw[5],
                activePassesMask = raw[6].toInt(),
                frameCount = raw[7].toLong(),
                activeFaces = if (raw.size >= 9) raw[8].toInt() else 0,
                resolvedTier = if (raw.size >= 11) raw[10].toInt() else 2,
            )
        }
    }
}
