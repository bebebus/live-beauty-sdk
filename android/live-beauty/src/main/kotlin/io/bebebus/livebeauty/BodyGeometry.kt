package io.bebebus.livebeauty

import kotlin.math.hypot

/**
 * 美体用的人体几何（仅 Android）。坐标与 [FaceGeometry] 相同：纹理坐标、y 向上；长度为度量值
 * （p' = (x, y/aspect) 的 x 单位）。不保存图像或身份。
 * - [hipX]/[hipY]：两胯中点；
 * - [axisX]/[axisY]：度量空间单位向量，肩中点 → 胯中点（沿身体向下）；
 * - [torso]：肩中点到胯中点的度量距离；[half]：肩宽与胯宽较大者的一半；
 * - [legs]：两膝都可见且在胯下方（画面里拍到腿），1 或 0；长腿只在此时生效；
 * - [visibility]：双肩、双胯 4 个点可见度的最小值，[BodyTrackingState] 据此做 0.6 / 0.4 迟滞（不进 [toState]）。
 */
data class BodyGeometry(
    val hipX: Float,
    val hipY: Float,
    val axisX: Float,
    val axisY: Float,
    val torso: Float,
    val half: Float,
    val legs: Float,
    val visibility: Float = 1f,
) {
    fun toState(): FloatArray = floatArrayOf(hipX, hipY, axisX, axisY, torso, half, legs)

    /** MediaPipe Pose 的一个关键点：归一化坐标（y 向下）与可见度。 */
    data class Landmark(val x: Float, val y: Float, val visibility: Float)

    companion object {
        const val STATE_SIZE = 7
        private const val L_SHOULDER = 11
        private const val R_SHOULDER = 12
        private const val L_HIP = 23
        private const val R_HIP = 24
        private const val L_KNEE = 25
        private const val R_KNEE = 26
        /** 膝盖算「拍到」的可见度（长腿开关，二值）。 */
        private const val MIN_VISIBILITY = 0.5f
        /** 迟滞：没在跟踪时肩胯可见度都要到 0.6 才开始，跟踪中降到 0.4 以下才算丢。 */
        const val VISIBILITY_ACQUIRE = 0.6f
        const val VISIBILITY_KEEP = 0.4f

        /**
         * 全有或全无：肩、胯任一点可见度低于 [VISIBILITY_KEEP]、非有限或明显出画、躯干尺寸退化都返回 null。
         * 是否达到开始跟踪的 [VISIBILITY_ACQUIRE] 由 [BodyTrackingState] 按 [visibility] 判断。[aspect] = 宽 / 高。
         */
        fun fromPose(points: List<Landmark>, aspect: Float): BodyGeometry? {
            if (points.size < 33 || !aspect.isFinite() || aspect <= 0f) return null
            fun visible(i: Int, margin: Float, minVisibility: Float) = points[i].let {
                it.x.isFinite() && it.y.isFinite() && it.visibility >= minVisibility &&
                    it.x in -margin..1f + margin && it.y in -margin..1f + margin
            }
            fun usable(i: Int, margin: Float) = visible(i, margin, VISIBILITY_KEEP)
            if (!usable(L_SHOULDER, 0.1f) || !usable(R_SHOULDER, 0.1f) ||
                !usable(L_HIP, 0.1f) || !usable(R_HIP, 0.1f)) return null
            val ratio = aspect.toDouble()
            // 纹理坐标（y 向上）。
            fun p(i: Int) = points[i].let { doubleArrayOf(it.x.toDouble(), 1.0 - it.y) }
            // 度量坐标。
            fun m(v: DoubleArray) = doubleArrayOf(v[0], v[1] / ratio)
            val ls = p(L_SHOULDER); val rs = p(R_SHOULDER)
            val lh = p(L_HIP); val rh = p(R_HIP)
            val shoulder = doubleArrayOf((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            val hip = doubleArrayOf((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            val ms = m(shoulder); val mh = m(hip)
            val dx = mh[0] - ms[0]; val dy = mh[1] - ms[1]
            val torso = hypot(dx, dy)
            if (torso !in 0.05..1.5) return null
            val shoulderSpan = hypot(m(ls)[0] - m(rs)[0], m(ls)[1] - m(rs)[1])
            val hipSpan = hypot(m(lh)[0] - m(rh)[0], m(lh)[1] - m(rh)[1])
            val half = maxOf(shoulderSpan, hipSpan) / 2
            if (half !in 0.02..0.8) return null
            val ax = dx / torso; val ay = dy / torso
            // 身体基本竖直（倒置或横躺时不处理）：肩到胯的方向与画面向下的夹角不超过约 50°。
            if (-ay < 0.64) return null
            fun belowHip(i: Int): Boolean {
                if (!visible(i, 0.05f, MIN_VISIBILITY)) return false
                val k = m(p(i))
                return (k[0] - mh[0]) * ax + (k[1] - mh[1]) * ay > 0.3 * torso
            }
            val legs = if (belowHip(L_KNEE) && belowHip(R_KNEE)) 1f else 0f
            val visibility = minOf(points[L_SHOULDER].visibility, points[R_SHOULDER].visibility,
                minOf(points[L_HIP].visibility, points[R_HIP].visibility))
            return BodyGeometry(hip[0].toFloat(), hip[1].toFloat(), ax.toFloat(), ay.toFloat(),
                torso.toFloat(), half.toFloat(), legs, visibility)
        }
    }
}

/**
 * 与 [FaceTrackingState] 相同的规则（单任务、限频、250 ms 有效期、代际取消），保留最近 4 帧供 FrameDelay 按显示时间匹配。
 * 可见度迟滞：没在跟踪时肩胯可见度要到 [BodyGeometry.VISIBILITY_ACQUIRE] 才收；单次推理失败（无人体或不新鲜）不清历史，
 * 连续 [MISSES_TO_CLEAR] 次才清（以前一次失败就清，美体形变随之闪一下再渐入）。
 */
class BodyTrackingState {
    data class Ticket(val epoch: Long, val timestamp: Long, val width: Int, val height: Int)
    data class Snapshot(val epoch: Long, val sampleTimestamp: Long, val geometry: BodyGeometry?, val completed: Long = Long.MIN_VALUE)
    private var epoch = 0L
    private var busy = false
    private var lastSample = Long.MIN_VALUE
    private var lastSubmitted = -1L
    private var dimensions = 0 to 0
    private var enabled = false
    private var completed = Long.MIN_VALUE
    private var misses = 0
    private val history = ArrayDeque<Pair<Ticket, BodyGeometry>>(4)
    @Synchronized fun enable(value: Boolean) {
        if (enabled != value) { reset(); enabled = value }
    }
    @Synchronized fun reset() { epoch++; history.clear(); misses = 0; lastSample = Long.MIN_VALUE; completed = Long.MIN_VALUE }
    /** 延迟回读画小图那一帧用，见 [FaceTrackingState.due]：不看是否忙、不取票，到期返回当前代际。 */
    @Synchronized fun due(now: Long, width: Int, height: Int, intervalMs: Long = 100L): Long? {
        if (dimensions != (width to height)) { reset(); dimensions = width to height }
        if (!enabled || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now - lastSample < intervalMs)) return null
        return epoch
    }
    /** [expectEpoch] 非 null 时代际不同就不取票，见 [FaceTrackingState.begin]。 */
    @Synchronized fun begin(now: Long, width: Int, height: Int, intervalMs: Long = 100L, expectEpoch: Long? = null): Ticket? {
        if (dimensions != (width to height)) { reset(); dimensions = width to height }
        if (expectEpoch != null && expectEpoch != epoch) return null
        if (!enabled || busy || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now - lastSample < intervalMs)) return null
        busy = true; lastSample = now; lastSubmitted = now
        return Ticket(epoch, now, width, height)
    }
    @Synchronized fun complete(ticket: Ticket, geometry: BodyGeometry?, now: Long): Boolean {
        busy = false
        if (ticket.epoch != epoch || !enabled) return false
        completed = maxOf(completed, ticket.timestamp)
        val tracking = history.isNotEmpty()
        val threshold = if (tracking) BodyGeometry.VISIBILITY_KEEP else BodyGeometry.VISIBILITY_ACQUIRE
        if (geometry != null && now - ticket.timestamp in 0..250 && geometry.visibility >= threshold) {
            if (history.size >= 4) history.removeFirst()
            history.addLast(ticket to geometry)
            misses = 0
        } else if (++misses >= MISSES_TO_CLEAR) {
            history.clear()
        }
        return true
    }
    @Synchronized fun snapshot(now: Long): Snapshot = snapshotAt(now, now)

    @Synchronized fun snapshotAt(targetTs: Long, now: Long): Snapshot {
        if (!enabled) return Snapshot(epoch, Long.MIN_VALUE, null, Long.MIN_VALUE)
        var best: Pair<Ticket, BodyGeometry>? = null
        var bestDiff = Long.MAX_VALUE
        for (entry in history) {
            if (now - entry.first.timestamp in 0..250) {
                val diff = kotlin.math.abs(entry.first.timestamp - targetTs)
                if (diff <= bestDiff) {
                    bestDiff = diff
                    best = entry
                }
            }
        }
        return Snapshot(epoch, best?.first?.timestamp ?: Long.MIN_VALUE, best?.second, completed)
    }

    private companion object {
        /** 连续这么多次推理失败才清历史。 */
        const val MISSES_TO_CLEAR = 2
    }
}

/**
 * 人体几何时间平滑；纯逻辑，只在 GL 线程使用。新样本到达时按 [ALPHA] 指数平滑（10 Hz 采样下约 0.2 s 跟上），
 * 每帧调用任意次。presence：有目标时 200 ms 渐入；目标消失后保持 [HOLD_MS]，再 [FADE_OUT_MS] 渐出，到 0 清空
 * （以前 200 + 300 ms，形变场最多残留约 0.5 s；单次推理失败由 [BodyTrackingState] 兜住，不再靠长保持防闪）。
 */
class BodyGeometrySmoother {
    var presence = 0f
        private set
    private var state: FloatArray? = null
    private var ts = Long.MIN_VALUE
    private var missing = 0f
    private var last = Long.MIN_VALUE

    fun reset() {
        state = null; ts = Long.MIN_VALUE; presence = 0f; missing = 0f; last = Long.MIN_VALUE
    }

    /** @return 当前平滑后的状态（调用方只读），无几何时为 null。 */
    fun update(target: FloatArray?, sampleTimestampMs: Long, nowMs: Long): FloatArray? {
        val dt = if (last == Long.MIN_VALUE) 0f else (nowMs - last).coerceIn(0L, 100L).toFloat()
        last = nowMs
        if (target != null) {
            require(target.size == BodyGeometry.STATE_SIZE)
            missing = 0f
            val current = state
            if (current == null) {
                state = target.copyOf()
            } else if (sampleTimestampMs != ts) {
                for (i in current.indices) current[i] += (target[i] - current[i]) * ALPHA
                // 方向重新归一化，避免平滑后长度漂移（着色器用长度携带长腿强度）。
                val length = hypot(current[2], current[3])
                if (length > 1e-4f) { current[2] /= length; current[3] /= length }
            }
            ts = sampleTimestampMs
            presence = (presence + dt / FADE_IN_MS).coerceAtMost(1f)
        } else if (state != null) {
            missing += dt
            if (missing > HOLD_MS) presence = (presence - dt / FADE_OUT_MS).coerceAtLeast(0f)
            if (presence <= 0f) reset()
        }
        return state
    }

    companion object {
        private const val ALPHA = 0.45f
        private const val FADE_IN_MS = 200f
        /** 目标消失后保持满强度的时间（约 2 帧）。 */
        const val HOLD_MS = 67f
        /** 保持之后渐出到 0 的时间。 */
        const val FADE_OUT_MS = 100f
    }
}

/**
 * 天鹅颈（BODY_FRAGMENT）把肩上 0.3 躯干长的一段拉长、头部整体上移 band × neck（度量），左右 1.2–2.2 半宽渐隐。
 * 人脸几何按原帧算，合成程序在美体之后，所以合成用的人脸几何要跟着平移，否则大眼、瘦脸、肤色遮罩会落在脖子上。纯逻辑。
 */
object NeckLift {
    /** 人脸状态（41 维，见 FaceGeometrySmoother）里所有纹理坐标中心点 (x, y) 的下标。 */
    private val CENTRES = intArrayOf(0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 28, 32, 36)

    /**
     * 把 [face] 的中心点按天鹅颈的上移量平移后写进 [out] 并返回它；不需要平移时原样复制。
     * [body] 为平滑后的人体状态（见 BodyGeometry.toState），[neck] 为着色器里的天鹅颈强度（uBodyC.y），[aspect] = 宽 / 高。
     */
    fun apply(face: FloatArray, body: FloatArray, neck: Float, aspect: Float, out: FloatArray): FloatArray {
        require(face.size >= FaceGeometrySmoother.STATE_SIZE && out.size >= face.size && body.size >= BodyGeometry.STATE_SIZE)
        face.copyInto(out)
        if (!(neck > 0f) || !(aspect > 0f)) return out
        val ax = body[2]; val ay = body[3]
        val length = kotlin.math.hypot(ax, ay)
        if (length < 1e-4f) return out
        val axisX = ax / length; val axisY = ay / length
        val band = 0.3f * body[4]
        val width = body[5]
        // 按人脸椭圆中心离身体中线的侧向距离取渐隐（与着色器相同）。
        val sideX = -axisY; val sideY = axisX
        val s = (face[22] - body[0]) * sideX + (face[23] / aspect - body[1] / aspect) * sideY
        val t = ((kotlin.math.abs(s) - 1.2f * width) / width).coerceIn(0f, 1f)
        val keep = 1f - t * t * (3f - 2f * t)
        val lift = band * neck * keep
        // 身体方向朝下；上移即沿 -axis，纹理 y 乘 aspect。
        val dx = -axisX * lift; val dy = -axisY * lift * aspect
        for (i in CENTRES) { out[i] += dx; out[i + 1] += dy }
        return out
    }

    /**
     * 将 478 点全人脸关键点（GL 纹理坐标，长度 [FaceMeshFrame.COORD_COUNT]）按天鹅颈上移量同步平移写入 [out] 并返回。
     * 供 M3 局部修饰与 M4 实时美妆在 [BodyWarpPass] 之后依然精准贴合面部与虹膜。
     */
    fun applyLandmarks(
        landmarks: FloatArray,
        faceCenterX: Float,
        faceCenterY: Float,
        body: FloatArray,
        neck: Float,
        aspect: Float,
        out: FloatArray,
    ): FloatArray {
        require(landmarks.size >= FaceMeshFrame.COORD_COUNT && out.size >= FaceMeshFrame.COORD_COUNT && body.size >= BodyGeometry.STATE_SIZE)
        landmarks.copyInto(out, 0, 0, FaceMeshFrame.COORD_COUNT)
        if (!(neck > 0f) || !(aspect > 0f)) return out
        val ax = body[2]; val ay = body[3]
        val length = kotlin.math.hypot(ax, ay)
        if (length < 1e-4f) return out
        val axisX = ax / length; val axisY = ay / length
        val band = 0.3f * body[4]
        val width = body[5]
        val sideX = -axisY; val sideY = axisX
        val s = (faceCenterX - body[0]) * sideX + (faceCenterY / aspect - body[1] / aspect) * sideY
        val t = ((kotlin.math.abs(s) - 1.2f * width) / width).coerceIn(0f, 1f)
        val keep = 1f - t * t * (3f - 2f * t)
        val lift = band * neck * keep
        val dx = -axisX * lift; val dy = -axisY * lift * aspect
        for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) {
            out[i * 2] += dx
            out[i * 2 + 1] += dy
        }
        return out
    }
}
