package io.bebebus.livebeauty

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot

/**
 * 头发遮罩（HairSegmenter，仅 Android）的调度、健康、新鲜度、运动补偿与时域常量和纯函数，JVM 可测。
 * 取值见 docs/plans/2026-11-21-live-beauty-hair-edge-plan.md §5；数值都是推断初值，待真机 Trace 与录屏校准。
 *
 * 时间口径：p50 是最近 [P50_WINDOW] 次 segmentForVideo 墙钟耗时的中位数；lag 是遮罩采样时间与被显示帧 displayTime 之差的绝对值。
 * 间隔取 2 × p50（worker 墙钟占空 ≤ 50%），健康上限下的最坏 lag ≈ p50 + 间隔 + 1 帧（125 + 250 + 33 = 408 ms），
 * 新鲜度窗口从 [FRESH_FULL_MS] 起衰减，正常运行时恒为 1，只在推理停摆时渐出（防锯齿不变量，见 HairMaskTuningTest）。
 */
internal object HairMaskTuning {
    /** 基础推理间隔（10 Hz）。 */
    const val INTERVAL_MS = 100L
    /**
     * 自适应间隔 = max([INTERVAL_MS], ⌈p50 × 该系数⌉)：worker 墙钟占空 p50 / 间隔 ≤ 1/2，机型越慢频率越低，
     * 不会在 p50 较大时一直停在高占空（1.3 时恒为约 77%）。
     */
    const val INTERVAL_P50_SCALE = 2f
    /** 开着背景虚化时的间隔下限：人像分割与头发分割的 XNNPACK 线程池会争核（H13），给人像分割让出时间。 */
    const val BACKGROUND_INTERVAL_MS = 200L
    /** 推理耗时统计窗口（最近 N 次）。 */
    const val P50_WINDOW = 10
    /** 攒满这么多次才判 p50（首帧含 XNNPACK 预热，样本太少不可信）。 */
    const val MIN_SAMPLES = 8
    /**
     * p50 超过它就本次会话停用头发遮罩，回到第一批启发式。由防锯齿不变量反推：最坏 lag = p50 + ⌈2·p50⌉ + 33 ≤ [FRESH_FULL_MS]
     * 即 p50 ≤ 129；取 125（最坏 408 ms，留 12 ms 余量），不为更慢的机型把新鲜期拉长。
     */
    const val MAX_P50_MS = 125f
    /** 遮罩有效期（MaskTrackingState.freshMs）：lag 超过它的结果直接作废。 */
    const val FRESH_MS = 650L
    /** 新鲜度从这个 lag 起由 1 衰减，到 [FRESH_MS] 为 0。 */
    const val FRESH_FULL_MS = 420f
    /** 存在度（MaskPresence）：渐入、保持、渐出。 */
    const val PRESENCE_FADE_IN_MS = 300f
    const val PRESENCE_HOLD_MS = 300f
    const val PRESENCE_FADE_OUT_MS = 400f
    /** worker 侧 EMA 的新帧权重；设为 1 即关闭 EMA。 */
    const val EMA_ALPHA = 0.5f
    /** EMA 运动门：与上次推理的人脸锚点相比，中心位移超过遮罩宽的这一比例就直接用新帧。 */
    const val EMA_MAX_SHIFT = 0.02f
    /** EMA 运动门：人脸半宽相对变化超过这一比例就直接用新帧。 */
    const val EMA_MAX_SCALE = 0.03f
    /** 运动补偿退回单位阵：两个时刻的脸中心相距超过画面宽的这一比例。 */
    const val AFFINE_MAX_SHIFT = 0.10f
    /** 运动补偿退回单位阵：遮罩时刻与显示时刻的脸半宽之比超出 [AFFINE_MIN_SCALE, AFFINE_MAX_SCALE]。 */
    const val AFFINE_MIN_SCALE = 0.8f
    const val AFFINE_MAX_SCALE = 1.25f
    /** [affine] 的输出长度：两行 vec4（uHairX0、uHairX1），w 恒 0。 */
    const val AFFINE_SIZE = 8
    /** 人脸锚点长度：(cx, cy, 半宽)，遮罩像素坐标。 */
    const val ANCHOR_SIZE = 3
    /**
     * HairSegmenter 的遮罩备用池容量（MaskTrackingState.spares）：历史 4 张 + GL 最近交出的 2 张 + EMA 上一帧之外，
     * 稳态每次推理挤出一张、取回一张，留 3 块足够，不再每次新建 216×384 的 ByteArray。
     */
    const val MASK_SPARES = 3
    /**
     * 「人脸近期可见」系数（[faceKeep]）：确认丢脸（最近一次完成的人脸检测里没有脸）之后，最新有脸样本过去不超过这么久
     * （渲染时间）仍为满，单次漏检不下陷；之后才开始渐出。
     */
    const val FACE_HOLD_MS = 200L
    /** 确认丢脸后渐出的最慢速度（满到 0 的时长）；确认得晚时加快，仍在 [FACE_ZERO_MS] 到 0。 */
    const val FACE_FADE_OUT_MS = 250f
    /**
     * 最新有脸样本过去这么久（渲染时间）必须已为 0：比 FaceTimeline.CLEAR_AFTER_MS（500）早 50 ms（20 fps 也至少一帧），
     * A 通道、合并道与背景虚化的细节保留不会因几何清空整帧硬切。= [FACE_HOLD_MS] + [FACE_FADE_OUT_MS]。
     */
    const val FACE_ZERO_MS = 450L
    /**
     * 检测器停摆（最新有脸样本之后没有任何结果，且没有人脸任务在途，分不清有没有脸）时的兜底：[FACE_ZERO_MS] 之前这么久才开始降，
     * 线性到 0。只在不在途时生效：人脸走 CPU、推理偏慢（SYNC ≥ 180 ms、DEFERRED ≥ 160 ms，或 120 ms 带 +80 ms 长尾）时，
     * 两次并入之间最新样本的年龄约 2 × 推理耗时 + 2–3 帧，会超过 350 ms，而这段时间里下一次推理一直在途（评审复核，见
     * HairFallbackJointTest）。
     */
    const val FACE_STALL_FADE_MS = 100f
    /** 首次检出或重新检出人脸后渐入到满的时长。 */
    const val FACE_FADE_IN_MS = 300f
    /** 模型内嵌标签（顺序即置信图顺序）。 */
    val LABELS = listOf("background", "hair")

    /**
     * 下一次推理的间隔：基础 [INTERVAL_MS]，按 p50 自适应为 max(100, ⌈2 × p50⌉)（向上取整，占空不超过 1/2）；
     * 开着背景虚化时下限 [BACKGROUND_INTERVAL_MS]。p50 未知（NaN，样本不足）时按下限。
     */
    fun interval(p50Ms: Float, backgroundActive: Boolean): Long {
        val floor = if (backgroundActive) BACKGROUND_INTERVAL_MS else INTERVAL_MS
        if (!p50Ms.isFinite() || p50Ms <= 0f) return floor
        return maxOf(floor, ceil(p50Ms * INTERVAL_P50_SCALE).toLong())
    }

    /** [samples] 前 [count] 个（环形缓冲里的最近 [P50_WINDOW] 次耗时，顺序无关）的中位数；不足 [MIN_SAMPLES] 个返回 NaN。 */
    fun p50(samples: FloatArray, count: Int): Float {
        val n = minOf(count, samples.size)
        if (n < MIN_SAMPLES) return Float.NaN
        val sorted = samples.copyOf(n).also { it.sort() }
        return sorted[n / 2]
    }

    /** p50 是否在健康范围内；NaN（样本不足）不判，算健康。 */
    fun healthy(p50Ms: Float): Boolean = !(p50Ms > MAX_P50_MS)

    /**
     * 新鲜度 = 1 − smoothstep([fullMs], [zeroMs], |lag|)。默认窗口 420–650 ms：放在健康上限下最坏 lag（408 ms）之上，
     * 正常运行时恒为 1，不随推理相位一阵有一阵无（锯齿）。窗口参数只供测试对照旧窗口用。
     */
    fun fresh(lagMs: Float, fullMs: Float = FRESH_FULL_MS, zeroMs: Float = FRESH_MS.toFloat()): Float {
        if (lagMs.isNaN()) return 0f
        return 1f - smoothstep(fullMs, zeroMs, abs(lagMs))
    }

    /**
     * 每帧头发遮罩强度 = presence × fresh × fades.body × (1 − [neckWeight])；走外部美颜引擎时为 0。
     * [neckWeight] 是天鹅颈的生效程度 0..1（头部被平移，A 通道不跟这段形变；调用方按人体存在度渐变给出，
     * 例如 neckLifted ? bodyPresence : 0），为 1（或 NaN）时为 0，人体出现、消失时按比例过渡，不再单帧硬切。
     * 「人脸近期可见」（[faceKeep]）由调用方乘进 [presence] 或 [bodyFade]。
     */
    fun shown(presence: Float, fresh: Float, bodyFade: Float, neckWeight: Float, engine: Boolean): Float {
        if (engine || !(neckWeight < 1f)) return 0f
        return (presence * fresh * bodyFade * (1f - neckWeight.coerceAtLeast(0f))).coerceIn(0f, 1f)
    }

    /**
     * 「人脸近期可见」系数（GL 线程每帧推进一次，0..1）：A 通道、背景合并道与背景虚化的细节保留（第一批 D3）都要用人脸几何，
     * FaceTimeline 在最新有脸样本过去 CLEAR_AFTER_MS（500 ms，渲染时间）后清空几何；强度若只看「几何非空」，丢脸 500 ms 时
     * 整帧硬切到 0，重新检出的第一帧又直接跳回满强度。age = [nowMs] − [latestFaceMs]（FaceTimeline.latest，最新有脸样本的采样时间）。
     * - 只在确认丢脸时渐出：[missSinceMs]（FaceTimeline.missSince）晚于 [latestFaceMs]，即最近一次完成的检测里确实没有脸，
     *   且 age > [FACE_HOLD_MS]。只计 [FACE_HOLD_MS] 之后的时间，每帧至少按 1 / [FACE_FADE_OUT_MS] 降，并按剩余时间线性赶在
     *   age = [FACE_ZERO_MS] 时到 0（人脸推理慢、确认得晚时渐出相应加快）。人脸每帧采样时与「保持 200 ms 再 250 ms 渐出」相同。
     *   只看 age 不看检测结果时，人脸走 CPU、推理 100–120 ms 的机型上画面里一直有脸，强度也会随采样节奏锯齿（评审复核）。
     * - 兜底（检测器停摆：最新有脸样本之后没有任何结果，且 [faceInFlight] 为 false，即确实没有人脸任务在途）：上限
     *   clamp(([FACE_ZERO_MS] − age) / [FACE_STALL_FADE_MS], 0, 1)，age ≥ [FACE_ZERO_MS] 时必为 0，几何清空之前一定到 0。
     *   有任务在途（[faceInFlight]，FaceTrackingState.Snapshot.inFlight）时不压上限：人脸推理慢、画面里一直有脸时，两次并入
     *   之间的 age 也会超过 350 ms，兜底若照样生效，头发强度与背景虚化的细节保留 β 会随采样节奏锯齿（评审复核）。确认丢脸的渐出
     *   不受影响（自己在 [FACE_ZERO_MS] 到 0）；任务卡死（一直在途、不交回结果）时不渐出，由 FaceTimeline 在 500 ms 清空几何
     *   那一帧硬切（已知代价）。缺省 false（不知道是否在途时按停摆兜底，几何清空前一定到 0）。
     * - 其余时候上升按 [FACE_FADE_IN_MS] 限速：min(上限, [previous] + dt / FACE_FADE_IN_MS)；dt = [dtMs] 钳在 [0, 100]（同 MaskPresence）。
     * - [latestFaceMs] 为 Long.MIN_VALUE（没有脸）时为 0；晚于 [nowMs] 时按 age = 0 算；[previous] 为 NaN 按 0。
     * 调用方保存上一帧的返回值与渲染时间，把结果乘进强度（见 [shown]）。
     */
    fun faceKeep(previous: Float, latestFaceMs: Long, missSinceMs: Long, nowMs: Long, dtMs: Long,
                 faceInFlight: Boolean = false): Float {
        if (latestFaceMs == Long.MIN_VALUE) return 0f
        val prev = if (previous.isNaN()) 0f else previous.coerceIn(0f, 1f)
        val dt = dtMs.coerceIn(0L, 100L)
        val age = (nowMs - latestFaceMs).coerceAtLeast(0L)
        val cap = if (faceInFlight) 1f else ((FACE_ZERO_MS - age) / FACE_STALL_FADE_MS).coerceIn(0f, 1f)
        val lost = missSinceMs != Long.MIN_VALUE && missSinceMs > latestFaceMs
        if (!lost || age <= FACE_HOLD_MS) return minOf(cap, prev + dt / FACE_FADE_IN_MS)
        // 本帧这一步的渐出区间 (from, age]：上一帧之后、且在保持期之后的那段；按剩余时间 left 线性降到 0。
        val from = maxOf(age - dt, FACE_HOLD_MS)
        val step = (age - from).toFloat()
        val left = (FACE_ZERO_MS - from).toFloat()
        val next = if (step >= left) 0f else prev - step * maxOf(1f / FACE_FADE_OUT_MS, prev / left)
        return minOf(cap, next.coerceAtLeast(0f))
    }

    /**
     * 是否需要头发遮罩（再经 TrackingGrace 3 s 判定启停）：头发分割与整体都健康、美颜开着、[faceWanted]
     * （运动补偿与脸内阈值都要用脸；画面里有没有脸由调用方决定是否并进来），并且 A 通道作用得到的任一项开着：
     * 磨皮、美白、红润、提亮、立体（radiance）、锐化、背景虚化。
     */
    fun wants(hairHealthy: Boolean, healthy: Boolean, faceWanted: Boolean, value: VideoEffectsConfig): Boolean =
        hairHealthy && healthy && value.enabled && faceWanted &&
            (value.smoothing > 0f || value.whiten > 0f || value.rosy > 0f || value.brightness > 0f ||
                value.radiance > 0f || value.sharpness > 0f || value.backgroundBlur > 0f)

    /** 模型标签必须恰为 [background, hair]（取置信图第 1 张的前提）。 */
    fun labelsValid(labels: List<String>?): Boolean =
        labels != null && labels.size == LABELS.size && labels.indices.all { labels[it].trim() == LABELS[it] }

    /** 写入单位阵（[AFFINE_SIZE] 个数：(1, 0, 0, 0, 0, 1, 0, 0)）。 */
    fun identity(out: FloatArray) {
        require(out.size >= AFFINE_SIZE)
        out.fill(0f, 0, AFFINE_SIZE)
        out[0] = 1f; out[5] = 1f
    }

    /**
     * 头部刚体运动补偿：由遮罩时刻与显示时刻的脸椭圆位姿（FaceTimeline.poseAt 的 (cx, cy, rx, ry, axisX, axisY)）
     * 求「显示帧纹理坐标 → 遮罩时刻纹理坐标」的 2×3 仿射，写进 [out]：
     * uv'.x = out[0]·u + out[1]·v + out[2]，uv'.y = out[4]·u + out[5]·v + out[6]（out[3]、out[7] 恒 0，按两个 vec4 上传）。
     * 在度量空间（y 除以 [aspect]，单位为画面宽）里做平移、等比缩放（rx 之比）与绕轴旋转（显示时刻轴向转到遮罩时刻轴向），
     * 脸上同一点在两个时刻的脸坐标系里位置相同。
     * 任一位姿缺失或退化、两个中心相距超过 [AFFINE_MAX_SHIFT] 画面宽、缩放超出 [AFFINE_MIN_SCALE]..[AFFINE_MAX_SCALE] 时
     * 写单位阵并返回 false（不补偿，好过补偿错）。
     */
    fun affine(poseMask: FloatArray?, poseDisplay: FloatArray?, aspect: Float, out: FloatArray): Boolean {
        identity(out)
        if (poseMask == null || poseDisplay == null || !(aspect > 0f)) return false
        require(poseMask.size >= FaceTimeline.POSE_SIZE && poseDisplay.size >= FaceTimeline.POSE_SIZE)
        val mx = poseMask[0]; val my = poseMask[1]; val mr = poseMask[2]
        val dx = poseDisplay[0]; val dy = poseDisplay[1]; val dr = poseDisplay[2]
        val nm = hypot(poseMask[4], poseMask[5]); val nd = hypot(poseDisplay[4], poseDisplay[5])
        if (!(mr > 0f && dr > 0f && nm > 1e-6f && nd > 1e-6f)) return false
        val shift = hypot(mx - dx, (my - dy) / aspect)
        val scale = mr / dr
        if (!(shift <= AFFINE_MAX_SHIFT) || !(scale >= AFFINE_MIN_SCALE && scale <= AFFINE_MAX_SCALE)) return false
        // 旋转 R 把显示时刻的轴向 u 转到遮罩时刻的轴向 w：cos = u·w，sin = u×w。
        val ux = poseDisplay[4] / nd; val uy = poseDisplay[5] / nd
        val wx = poseMask[4] / nm; val wy = poseMask[5] / nm
        val c = scale * (ux * wx + uy * wy)
        val s = scale * (ux * wy - uy * wx)
        // 度量空间 m' = cMask + S·R·(m − cDisplay)，m = (u, v / aspect)；换回纹理坐标 v' = aspect · m'.y。
        out[0] = c; out[1] = -s / aspect; out[2] = mx - c * dx + s * dy / aspect
        out[4] = s * aspect; out[5] = c; out[6] = my - s * aspect * dx - c * dy
        return true
    }

    /**
     * worker 上的时域 EMA（新帧权重 [alpha]，默认 [EMA_ALPHA]）：单帧独立推理的边界会抖，静止时与上一帧结果混合。
     * 以下情况直接用新帧（返回 [next] 本身）：[alpha] ≥ 1、没有上一帧、代际变了（[sameEpoch] 为 false）、尺寸不同、
     * 任一锚点缺失，或锚点 (cx, cy, 半宽)（遮罩像素坐标）的中心位移超过 [EMA_MAX_SHIFT] × 遮罩宽、半宽变化超过 [EMA_MAX_SCALE]
     * （摆头时不拖影）。混合时写进 [out] 并返回它：[out] 为 null 时新建，不改 [prev] 与 [next]（它们可能已在
     * MaskTrackingState 的历史里）；HairSegmenter 传 out = next 原地混合（next 是刚从备用池取来、还没交出去的那块），
     * 逐字节与新建时相同。[out] 不能是 [prev]。
     */
    fun temporalBlend(prev: PersonMask?, prevAnchor: FloatArray?, next: PersonMask, anchor: FloatArray?,
                      sameEpoch: Boolean, alpha: Float = EMA_ALPHA, out: PersonMask? = null): PersonMask {
        if (!(alpha < 1f) || prev == null || !sameEpoch || prev.width != next.width || prev.height != next.height) return next
        if (moved(prevAnchor, anchor, next.width)) return next
        val target = out ?: PersonMask(ByteArray(next.pixels.size), next.width, next.height)
        require(target.width == next.width && target.height == next.height && target.pixels !== prev.pixels)
        val a = alpha.coerceAtLeast(0f)
        val old = prev.pixels; val new = next.pixels; val dst = target.pixels
        for (i in dst.indices) {
            dst[i] = ((new[i].toInt() and 0xFF) * a + (old[i].toInt() and 0xFF) * (1f - a) + 0.5f).toInt().coerceIn(0, 255).toByte()
        }
        return target
    }

    /** EMA 运动门：任一锚点缺失或退化、中心位移或尺度变化超限时为 true。 */
    fun moved(prevAnchor: FloatArray?, anchor: FloatArray?, maskWidth: Int): Boolean {
        if (prevAnchor == null || anchor == null || prevAnchor.size < ANCHOR_SIZE || anchor.size < ANCHOR_SIZE) return true
        val shift = hypot(anchor[0] - prevAnchor[0], anchor[1] - prevAnchor[1])
        if (!(shift <= EMA_MAX_SHIFT * maskWidth)) return true
        val r0 = prevAnchor[2]; val r1 = anchor[2]
        if (!(r0 > 0f && r1 > 0f)) return true
        return !(abs(r1 / r0 - 1f) <= EMA_MAX_SCALE)
    }

    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }
}
