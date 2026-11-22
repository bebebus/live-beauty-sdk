package io.bebebus.livebeauty

import kotlin.math.hypot

/**
 * 实时美妆子网格拓扑、规范 UV 顶点打包与秀场色号/质地调色表（仅 Android，纯 Kotlin 无 GL 依赖）。
 *
 * - [SKIN_FACE_INDICES]：从 898 个规范三角形中剔除 28 个眼裂内部三角形与 18 个内唇口腔三角形后剩余的 852 个面部皮肤三角形，
 *   确保腮红/修容/眼影/眼线/眉妆绝不误涂眼球或张嘴露出的牙齿，同时节省 ~70% 的 1080p 片元填充率；
 * - [FaceMeshTopology.LIP_BAND_INDICES]：120 个纯唇带三角形（口腔零三角形，物理防涂牙）；
 * - [FaceMeshTopology.RIGHT_EYE_INDICES] / [FaceMeshTopology.LEFT_EYE_INDICES]：各 14 个眼裂内部三角形（美瞳物理眼皮裁切）。
 */
internal object FaceMakeupUv {
    const val VERTEX_STRIDE_FLOATS = 6 // dstNdcX, dstNdcY, screenU, screenV, makeupU, makeupV
    const val VERTEX_DATA_SIZE = FaceMeshTopology.FACE_VERTEX_COUNT * VERTEX_STRIDE_FLOATS

    internal val RIGHT_EYE_SET = setOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
    internal val LEFT_EYE_SET = setOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
    internal val INNER_LIP_SET = setOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)

    /** 852 个面部皮肤三角形索引（排除眼裂内部与内唇孔洞），供腮红/修容/高光/眉妆/眼妆渲染。 */
    val SKIN_FACE_INDICES: ShortArray = buildSkinFaceIndices()

    /** 18 个纯内唇口腔三角形索引（三顶点均在内唇环内，唇红覆盖率为物理 0，供 468 点白牙子网格防染唇渲染）。 */
    val INNER_MOUTH_INDICES: ShortArray = buildInnerMouthIndices()

    /** 28 个双眼眼裂内部三角形索引（供 468 点亮眼子网格精准提亮巩膜与虹膜对比度）。 */
    val EYE_OPENING_INDICES: ShortArray = ShortArray(FaceMeshTopology.RIGHT_EYE_INDICES.size + FaceMeshTopology.LEFT_EYE_INDICES.size).apply {
        FaceMeshTopology.RIGHT_EYE_INDICES.copyInto(this, 0)
        FaceMeshTopology.LEFT_EYE_INDICES.copyInto(this, FaceMeshTopology.RIGHT_EYE_INDICES.size)
    }

    private fun buildInnerMouthIndices(): ShortArray {
        val src = FaceMeshTopology.CANONICAL_INDICES
        val kept = ArrayList<Short>(18 * 3)
        for (i in src.indices step 3) {
            val a = src[i].toInt()
            val b = src[i + 1].toInt()
            val c = src[i + 2].toInt()
            if (a in INNER_LIP_SET && b in INNER_LIP_SET && c in INNER_LIP_SET) {
                kept.add(src[i]); kept.add(src[i + 1]); kept.add(src[i + 2])
            }
        }
        return ShortArray(kept.size) { kept[it] }
    }

    private fun buildSkinFaceIndices(): ShortArray {
        val src = FaceMeshTopology.CANONICAL_INDICES
        val kept = ArrayList<Short>(852 * 3)
        for (i in src.indices step 3) {
            val a = src[i].toInt()
            val b = src[i + 1].toInt()
            val c = src[i + 2].toInt()
            val inRightEye = a in RIGHT_EYE_SET && b in RIGHT_EYE_SET && c in RIGHT_EYE_SET
            val inLeftEye = a in LEFT_EYE_SET && b in LEFT_EYE_SET && c in LEFT_EYE_SET
            val inMouth = a in INNER_LIP_SET && b in INNER_LIP_SET && c in INNER_LIP_SET
            if (!inRightEye && !inLeftEye && !inMouth) {
                kept.add(src[i]); kept.add(src[i + 1]); kept.add(src[i + 2])
            }
        }
        return ShortArray(kept.size) { kept[it] }
    }

    /** 将变形并经天鹅颈平移后的 468 点纹理坐标与规范 UV 打包进 [out]（长度 [VERTEX_DATA_SIZE]）。 */
    fun packVertices(landmarks: FloatArray, out: FloatArray) {
        require(landmarks.size >= FaceMeshTopology.FACE_VERTEX_COUNT * 2 && out.size >= VERTEX_DATA_SIZE)
        val uv = FaceMeshTopology.CANONICAL_UV
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            val sx = landmarks[i * 2]
            val sy = landmarks[i * 2 + 1]
            val k = i * VERTEX_STRIDE_FLOATS
            out[k] = sx * 2f - 1f
            out[k + 1] = sy * 2f - 1f
            out[k + 2] = sx
            out[k + 3] = sy
            out[k + 4] = uv[i * 2]
            out[k + 5] = uv[i * 2 + 1]
        }
    }

    /**
     * 从 468..472（右眼）或 473..477（左眼）计算虹膜中心（纹理坐标）与度量半径（轻微放瞳 1.08× 显灵动）。
     * 结果写入 [out] 的 4 个槽位 `(cx, cy, radiusMetric, aspect)`；退化时返回 false。
     */
    fun computeIrisCircle(landmarks: FloatArray, baseIdx: Int, aspect: Float, out: FloatArray): Boolean {
        if (landmarks.size < FaceMeshFrame.COORD_COUNT || out.size < 4 || !(aspect > 0f)) return false
        val cx = landmarks[baseIdx * 2]
        val cy = landmarks[baseIdx * 2 + 1]
        var sumR = 0f
        for (k in 1..4) {
            val px = landmarks[(baseIdx + k) * 2]
            val py = landmarks[(baseIdx + k) * 2 + 1]
            sumR += hypot(px - cx, (py - cy) / aspect)
        }
        val r = (sumR * 0.25f) * 1.08f
        if (!cx.isFinite() || !cy.isFinite() || !r.isFinite() || r < 0.002f || r > 0.25f) return false
        out[0] = cx; out[1] = cy; out[2] = r; out[3] = aspect
        return true
    }

    // ---- 秀场精选调色板（RGB 0..1）----

    fun lipstickRgb(color: String): FloatArray = when (color) {
        "coral_peach" -> floatArrayOf(0.86f, 0.36f, 0.27f)
        "cherry_red" -> floatArrayOf(0.70f, 0.09f, 0.11f)
        "milk_tea" -> floatArrayOf(0.64f, 0.39f, 0.27f)
        "berry_plum" -> floatArrayOf(0.54f, 0.12f, 0.26f)
        "orange_soda" -> floatArrayOf(0.88f, 0.30f, 0.10f)
        else -> floatArrayOf(0.73f, 0.32f, 0.33f) // rose_bean 温柔豆沙
    }

    fun blushRgb(color: String): FloatArray = when (color) {
        "rose" -> floatArrayOf(0.88f, 0.44f, 0.54f)
        "apricot" -> floatArrayOf(0.95f, 0.56f, 0.40f)
        "berry" -> floatArrayOf(0.80f, 0.36f, 0.52f)
        else -> floatArrayOf(0.96f, 0.54f, 0.58f) // peach 蜜桃粉
    }

    fun blushChannel(style: String): FloatArray = when (style) {
        "drunk" -> floatArrayOf(0f, 1f, 0f, 0f)
        "under_eye" -> floatArrayOf(0f, 0f, 1f, 0f)
        else -> floatArrayOf(1f, 0f, 0f, 0f) // apple
    }

    fun eyebrowRgb(color: String): FloatArray = when (color) {
        "soft_brown" -> floatArrayOf(0.42f, 0.30f, 0.22f)
        "grey_black" -> floatArrayOf(0.24f, 0.24f, 0.26f)
        else -> floatArrayOf(0.30f, 0.22f, 0.18f) // dark_brown
    }

    fun eyebrowChannel(style: String): FloatArray = when (style) {
        "arched" -> floatArrayOf(0f, 1f, 0f, 0f)
        else -> floatArrayOf(1f, 0f, 0f, 0f) // natural
    }

    fun eyeshadowMainRgb(color: String): FloatArray = when (color) {
        "peach_pink" -> floatArrayOf(0.86f, 0.58f, 0.56f)
        "rose_wine" -> floatArrayOf(0.74f, 0.42f, 0.48f)
        "champagne" -> floatArrayOf(0.82f, 0.66f, 0.52f)
        else -> floatArrayOf(0.68f, 0.52f, 0.42f) // earth 大地棕
    }

    fun eyeshadowDeepRgb(color: String): FloatArray = when (color) {
        "peach_pink" -> floatArrayOf(0.66f, 0.36f, 0.38f)
        "rose_wine" -> floatArrayOf(0.50f, 0.22f, 0.28f)
        "champagne" -> floatArrayOf(0.58f, 0.42f, 0.30f)
        else -> floatArrayOf(0.44f, 0.32f, 0.25f) // earth
    }

    /** 2×2 iris_atlas.png 格起点 (tileX, tileY) ∈ {0.0, 0.5}。 */
    fun irisTile(style: String): Pair<Float, Float> = when (style) {
        "chocolate" -> 0.5f to 0f
        "mist_grey" -> 0f to 0.5f
        "sakura_pink" -> 0.5f to 0.5f
        else -> 0f to 0f // hazel
    }
}
