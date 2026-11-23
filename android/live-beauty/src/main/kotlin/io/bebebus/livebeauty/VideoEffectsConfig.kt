package io.bebebus.livebeauty

/** 不依赖 Android，可在 JVM 验证渠道输入边界。 */
data class VideoEffectsConfig @JvmOverloads constructor(
    val enabled: Boolean = true,
    val smoothing: Float = 0f,
    val brightness: Float = 0f,
    val filter: String = "none",
    val skinOnly: Boolean = false,
    val sharpness: Float = 0f,
    val saturation: Float = 1f,
    val filterIntensity: Float = 1f,
    val compareOriginal: Boolean = false,
    val faceSlim: Float = 0f,
    val eyeEnlarge: Float = 0f,
    val whiten: Float = 0f,
    val rosy: Float = 0f,
    /** 美体（仅 Android，需识别到身体）：细腰、丰胸、长腿，0–1。 */
    val waistSlim: Float = 0f,
    val bustEnhance: Float = 0f,
    val legLengthen: Float = 0f,
    /** 美体补充（仅 Android）：瘦肩、天鹅颈，0–1。 */
    val shoulderSlim: Float = 0f,
    val neckLengthen: Float = 0f,
    /** 更多脸型与 478 点网格微整（仅 Android，需人脸）：下巴/额头/眼距/眼角/长鼻/嘴型为双向 -1..1，其余 0..1。 */
    val chinShape: Float = 0f,
    val foreheadShape: Float = 0f,
    val noseSlim: Float = 0f,
    val vFace: Float = 0f,
    val faceNarrow: Float = 0f,
    val faceSmall: Float = 0f,
    val cheekboneSlim: Float = 0f,
    val eyeDistance: Float = 0f,
    val eyeCorner: Float = 0f,
    val noseLength: Float = 0f,
    val mouthSize: Float = 0f,
    val mouthSmile: Float = 0f,
    /** 局部修饰（仅 Android，需人脸）：去黑眼圈、亮眼、白牙，0–1。 */
    val darkCircles: Float = 0f,
    val brightEyes: Float = 0f,
    val teethWhiten: Float = 0f,
    /** 淡化法令纹（仅 Android，需人脸）：0–1。 */
    val nasolabial: Float = 0f,
    /** 背景虚化（仅 Android，人像分割）：0–1。 */
    val backgroundBlur: Float = 0f,
    /** LUT 调色（仅 Android）：512×512 PNG 的 https 地址或内置 asset://luts/<id>.png，空串表示不用；强度 0–1。 */
    val lutUrl: String = "",
    val lutIntensity: Float = 1f,
    /** 质感磨皮微纹理保留（仅 Android，0–1）与柔光立体打光（仅 Android，0–1），0 为中性。 */
    val skinTexture: Float = 0f,
    val radiance: Float = 0f,
    /** 实时美妆（仅 Android，需 478 点人脸网格）：全部强度 0–1，0 为不绘制。 */
    val lipstick: Float = 0f,
    val lipstickColor: String = "rose_bean",
    val lipstickFinish: String = "velvet",
    val blush: Float = 0f,
    val blushStyle: String = "apple",
    val blushColor: String = "peach",
    val eyebrow: Float = 0f,
    val eyebrowStyle: String = "natural",
    val eyebrowColor: String = "dark_brown",
    val eyeshadow: Float = 0f,
    val eyeshadowColor: String = "earth",
    val eyeliner: Float = 0f,
    val aegyoSal: Float = 0f,
    val contour: Float = 0f,
    val highlight: Float = 0f,
    val contactLens: Float = 0f,
    val contactLensStyle: String = "hazel",
    val catchlight: Float = 0f,
    /** 高颅顶 / 蓬松头包脸（仅 Android，0..1，置于末尾保持位置构造兼容）：外蓬颅顶与上颞发冠且不拉长裸额头。 */
    val cranialTop: Float = 0f,
    /** P1: 3D 真实法线演播室重打光强度（0..1）与风格（natural/studio_softbox/butterfly/rembrandt）。 */
    val relight: Float = 0f,
    val relightStyle: String = "natural",
    /** P1: T 区去油光/抑高光过曝（0..1）。 */
    val oilControl: Float = 0f,
    /** P1: 面颈一体化自然肤色衔接（0..1，严格肤色门控不染衣领）。 */
    val neckTone: Float = 0f,
    /** P1: 美妆环境色温白平衡自适应混合系数（0..1）。 */
    val ambientMakeupBlend: Float = 0.5f,
    /** P2: 实时人像抠像时域防闪烁与 RGB 引导边缘锐化（0..1）。 */
    val matteTemporal: Float = 0.6f,
    /** P3: 多尺度孤立痘印/红斑/色斑自动祛除（0..1，保留毛孔与五官）。 */
    val blemishRemove: Float = 0f,
    /** P4: 芯片分级调度档位（auto/lite/standard/ultra）。 */
    val performanceTier: String = "auto",
) {
    /** 任意一项面部与颅顶网格微整处于非零状态。 */
    fun hasFaceReshape(): Boolean =
        faceSlim > 0f || eyeEnlarge > 0f || vFace > 0f || faceNarrow > 0f || faceSmall > 0f ||
            cheekboneSlim > 0f || chinShape != 0f || foreheadShape != 0f || cranialTop > 0f ||
            eyeDistance != 0f || eyeCorner != 0f || noseSlim > 0f ||
            noseLength != 0f || mouthSize != 0f || mouthSmile > 0f

    /** 任意一项实时美妆处于非零状态。 */
    fun hasMakeup(): Boolean =
        lipstick > 0f || blush > 0f || eyebrow > 0f || eyeshadow > 0f || eyeliner > 0f ||
            aegyoSal > 0f || contour > 0f || highlight > 0f || contactLens > 0f || catchlight > 0f

    companion object {
        val ASSET_LUT_PATTERN = Regex("^asset://luts/[a-z0-9_]{1,32}\\.png$")
        val LIPSTICK_COLORS = setOf("rose_bean", "coral_peach", "cherry_red", "milk_tea", "berry_plum", "orange_soda")
        val LIPSTICK_FINISHES = setOf("velvet", "glossy")
        val BLUSH_STYLES = setOf("apple", "drunk", "under_eye")
        val BLUSH_COLORS = setOf("peach", "rose", "apricot", "berry")
        val EYEBROW_STYLES = setOf("natural", "arched")
        val EYEBROW_COLORS = setOf("dark_brown", "soft_brown", "grey_black")
        val EYESHADOW_COLORS = setOf("earth", "peach_pink", "rose_wine", "champagne")
        val CONTACT_LENS_STYLES = setOf("hazel", "chocolate", "mist_grey", "sakura_pink")
        val RELIGHT_STYLES = setOf("natural", "studio_softbox", "butterfly", "rembrandt")
        val PERFORMANCE_TIERS = setOf("auto", "lite", "standard", "ultra")
        val original = VideoEffectsConfig(false, 0f, 0f, "none")

        fun isValidLutUrl(url: String): Boolean {
            if (url.isEmpty()) return true
            if (url.startsWith("https://")) {
                return url.length <= 2048 && url.none { it.isWhitespace() || it.isISOControl() }
            }
            return ASSET_LUT_PATTERN.matches(url)
        }

        fun parse(values: Map<*, *>): VideoEffectsConfig {
            val enabled = values["enabled"] as? Boolean ?: error("invalid_effects")
            fun range(key: String, default: Float? = null, minimum: Double = 0.0, maximum: Double = 1.0): Float {
                if (!values.containsKey(key) && default != null) return default
                val value = (values[key] as? Number)?.toDouble() ?: error("invalid_effects")
                require(value.isFinite() && value in minimum..maximum)
                return value.toFloat()
            }
            fun unit(key: String, default: Float? = null, maximum: Double = 1.0): Float =
                range(key, default, 0.0, maximum)
            fun signed(key: String, default: Float = 0f): Float =
                range(key, default, -1.0, 1.0)
            fun flag(key: String): Boolean = if (values.containsKey(key))
                values[key] as? Boolean ?: error("invalid_effects") else false
            fun enumStr(key: String, default: String, allowed: Set<String>): String {
                if (!values.containsKey(key)) return default
                val value = values[key] as? String ?: error("invalid_effects")
                require(value in allowed)
                return value
            }
            val lutUrl = if (values.containsKey("lutUrl")) values["lutUrl"] as? String ?: error("invalid_effects") else ""
            require(isValidLutUrl(lutUrl))
            val filter = values["filter"] as? String ?: error("invalid_effects")
            require(filter in setOf("none", "warm", "cool", "monochrome", "sepia", "rose"))
            return VideoEffectsConfig(enabled, unit("smoothing"), unit("brightness"), filter,
                flag("skinOnly"), unit("sharpness", 0f), unit("saturation", 1f, 2.0),
                unit("filterIntensity", 1f), flag("compareOriginal"),
                unit("faceSlim", 0f), unit("eyeEnlarge", 0f),
                unit("whiten", 0f), unit("rosy", 0f),
                unit("waistSlim", 0f), unit("bustEnhance", 0f), unit("legLengthen", 0f),
                shoulderSlim = unit("shoulderSlim", 0f), neckLengthen = unit("neckLengthen", 0f),
                backgroundBlur = unit("backgroundBlur", 0f),
                chinShape = signed("chinShape"), foreheadShape = signed("foreheadShape"),
                cranialTop = unit("cranialTop", 0f), noseSlim = unit("noseSlim", 0f),
                vFace = unit("vFace", 0f), faceNarrow = unit("faceNarrow", 0f), faceSmall = unit("faceSmall", 0f),
                cheekboneSlim = unit("cheekboneSlim", 0f), eyeDistance = signed("eyeDistance"), eyeCorner = signed("eyeCorner"),
                noseLength = signed("noseLength"), mouthSize = signed("mouthSize"), mouthSmile = unit("mouthSmile", 0f),
                darkCircles = unit("darkCircles", 0f), brightEyes = unit("brightEyes", 0f), teethWhiten = unit("teethWhiten", 0f),
                nasolabial = unit("nasolabial", 0f),
                lutUrl = lutUrl, lutIntensity = unit("lutIntensity", 1f),
                skinTexture = unit("skinTexture", 0f), radiance = unit("radiance", 0f),
                lipstick = unit("lipstick", 0f),
                lipstickColor = enumStr("lipstickColor", "rose_bean", LIPSTICK_COLORS),
                lipstickFinish = enumStr("lipstickFinish", "velvet", LIPSTICK_FINISHES),
                blush = unit("blush", 0f),
                blushStyle = enumStr("blushStyle", "apple", BLUSH_STYLES),
                blushColor = enumStr("blushColor", "peach", BLUSH_COLORS),
                eyebrow = unit("eyebrow", 0f),
                eyebrowStyle = enumStr("eyebrowStyle", "natural", EYEBROW_STYLES),
                eyebrowColor = enumStr("eyebrowColor", "dark_brown", EYEBROW_COLORS),
                eyeshadow = unit("eyeshadow", 0f),
                eyeshadowColor = enumStr("eyeshadowColor", "earth", EYESHADOW_COLORS),
                eyeliner = unit("eyeliner", 0f),
                aegyoSal = unit("aegyoSal", 0f),
                contour = unit("contour", 0f),
                highlight = unit("highlight", 0f),
                contactLens = unit("contactLens", 0f),
                contactLensStyle = enumStr("contactLensStyle", "hazel", CONTACT_LENS_STYLES),
                catchlight = unit("catchlight", 0f),
                relight = unit("relight", 0f),
                relightStyle = enumStr("relightStyle", "natural", RELIGHT_STYLES),
                oilControl = unit("oilControl", 0f),
                neckTone = unit("neckTone", 0f),
                ambientMakeupBlend = unit("ambientMakeupBlend", 0.5f),
                matteTemporal = unit("matteTemporal", 0.6f),
                blemishRemove = unit("blemishRemove", 0f),
                performanceTier = enumStr("performanceTier", "auto", PERFORMANCE_TIERS))
        }
    }
}
