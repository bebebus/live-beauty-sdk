package io.bebebus.livebeauty

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 画面对齐的人脸几何时间线（仅 Android，取代 FaceGeometrySmoother 在安卓上的用法；共享向量与 iOS 不变）。
 * 纯逻辑，只在 GL 线程使用。
 *
 * 以前几何按「现在」平滑插值后画在当前帧上，比画面落后 100–170 ms，镜头一动美颜就跟不上脸（五官残影）。
 * 现在画面由 FrameDelay 延后到人脸结果已出来的那一帧，这里按「被画那一帧的采样时间」取几何：
 * - [add]：每个新样本做一次平滑后存进时间线。41 维状态用 One-Euro（静止截止频率低、一动立刻升高）；
 *   478 点网格交给 [FaceMeshFilter]（刚体姿态快层 + 规范残差慢层，死区相对自适应噪声底），元数据由滤波后网格重算；
 * - [at] / [meshAt]：在前后两个样本之间按时间线性插值；要画的帧比最新样本还新（推理太慢、延后不够）时只按刚体姿态外推，
 *   上限随采样间隔变化（[horizonMs]），外推量记为 [offsetX]/[offsetY]，并由 [dilate] 让脸、眼、唇椭圆盖住保持位置到
 *   外推位置的整段、由 [steadyEye]/[steadyFace] 渐出对应尺度的形变与局部修饰；
 * - presence 按被画那一帧的采样时间 t 计算（[add] 记有脸、[addMiss] 记无脸）：新一轮从第一个样本起 150 ms 渐入
 *   （更早的延后帧为 0，不把「未来」的网格画上去）；第一个无脸样本之后防抖 [missDebounceMs]（30 Hz 约 40 ms）再 40 ms 渐出，
 *   30 Hz 下最迟约 80 ms 退出；检测器忙（最新样本之后没有任何结果）时有上限地保持再渐出；
 *   漏检后检测器跳采（那次推理超过一帧）、这段无脸不超过一个退出窗口且还没有被画帧进入渐出时，
 *   重新检出不开新一轮（观众没看到过渐出，不下陷）；已有帧画到渐出才回来的，从退出到的水平连续渐入；
 * - [denoiseKeep]：脸在动时暂停时域降噪（它没有运动补偿，会把上一帧混进来）；
 * - [poseAt]：只读的脸椭圆位姿插值（头发遮罩运动补偿用），不碰上面这些状态；
 * - [missSince]：最近一次完成的检测是否没有脸（只读），头发遮罩的「人脸近期可见」据此区分丢脸与检测器还在算。
 */
internal class FaceTimeline {
    /** 宽 / 高；纹理 y 换算到度量空间时除以它。只在 reset 前后修改。 */
    var aspect = 1f
    private val times = LongArray(HISTORY)
    private val states = Array(HISTORY) { FloatArray(FaceGeometrySmoother.STATE_SIZE) }
    // 478 点全网格预分配循环缓冲（16 × 956 floats ≈ 61 KB），与 41 维状态共享同槽位，零运行期 GC。
    private val meshCoords = Array(HISTORY) { FloatArray(FaceMeshFrame.COORD_COUNT) }
    private val meshMeta = Array(HISTORY) { FloatArray(FaceMeshFrame.META_SIZE) }
    // 滤波后的弱透视姿态（见 FaceMeshFilter.POSE_SIZE），外推用。
    private val meshPose = Array(HISTORY) { FloatArray(FaceMeshFilter.POSE_SIZE) }
    private val meshPresent = BooleanArray(HISTORY)
    private val meshHasIris = BooleanArray(HISTORY)
    private val meshOutCoords = FloatArray(FaceMeshFrame.COORD_COUNT)
    private val meshOutMeta = FloatArray(FaceMeshFrame.META_SIZE)
    private val meshFilter = FaceMeshFilter()
    private var head = 0 // 下一个写入位置
    private var count = 0

    /** [meshAt] 返回的只读视图（对象复用，下次 [meshAt] 前有效）。 */
    class SmoothedMesh(
        val landmarks: FloatArray,
        val meta: FloatArray,
    ) {
        var hasIris: Boolean = false
            internal set
        val centerX: Float get() = meta[FaceMeshFrame.META_CENTER_X]
        val centerY: Float get() = meta[FaceMeshFrame.META_CENTER_Y]
        val axisX: Float get() = meta[FaceMeshFrame.META_AXIS_X]
        val axisY: Float get() = meta[FaceMeshFrame.META_AXIS_Y]
        val faceWidth: Float get() = meta[FaceMeshFrame.META_FACE_WIDTH]
        val faceHeight: Float get() = meta[FaceMeshFrame.META_FACE_HEIGHT]
        val signedYaw: Float get() = meta[FaceMeshFrame.META_SIGNED_YAW]
        val leftWeight: Float get() = meta[FaceMeshFrame.META_LEFT_WEIGHT]
        val rightWeight: Float get() = meta[FaceMeshFrame.META_RIGHT_WEIGHT]
        val yawWeight: Float get() = meta[FaceMeshFrame.META_YAW_WEIGHT]
        /** 滤波后网格的睁眼比（竖直睑裂 / 眼宽）。 */
        val rightEyeOpen: Float get() = meta[FaceMeshFrame.META_RIGHT_EYE_OPEN]
        val leftEyeOpen: Float get() = meta[FaceMeshFrame.META_LEFT_EYE_OPEN]
        val mouthOpen: Float get() = meta[FaceMeshFrame.META_MOUTH_OPEN]
        /** 眨眼门控（0 = 闭眼收起美瞳，1 = 睁眼），供 MakeupPass。 */
        val rightIrisGate: Float get() = meta[FaceMeshFrame.META_RIGHT_IRIS_GATE]
        val leftIrisGate: Float get() = meta[FaceMeshFrame.META_LEFT_IRIS_GATE]
    }
    private val meshView = SmoothedMesh(meshOutCoords, meshOutMeta)
    // 原始（未滤波）脸中心，与 states 同槽位：One-Euro 的速度按固定时间基线从原始样本求，不随采样频率放大抖动。
    private val rawX = FloatArray(HISTORY)
    private val rawY = FloatArray(HISTORY)
    private var edxX = 0f // One-Euro 的速度估计（度量单位 / 秒，按轴低通后再取大小）
    private var edxY = 0f
    // 外推量（dilate 与 steady 用）：变大立刻跟上、变小约 HOLD_DECAY_MS 回落，逐帧插值 / 外推切换时保护区不闪。
    private var holdX = 0f
    private var holdY = 0f
    private val output = FloatArray(FaceGeometrySmoother.STATE_SIZE)
    private var velocityX = 0f // 无网格时的脸中心速度，度量单位 / ms（由滤波后的样本按时间基线求）
    private var velocityY = 0f
    // 最新样本的姿态速度（每 ms，相邻两个滤波后姿态之差）与是否超过姿态死区（静止时不外推）。
    private val poseVel = FloatArray(FaceMeshFilter.POSE_SIZE)
    private var poseVelValid = false
    private var poseMoving = false
    private val shift = FloatArray(2)
    private var last = Long.MIN_VALUE
    // 最新样本的去抖动速度（画面宽 / 秒）：脸中心与含原地转头、说话的整体运动；过期（MAX_GAP_MS）后在 update 里清零。
    private var sampleSpeed = 0f
    private var sampleMotion = 0f
    // update 记下本次绘制的 dt，at 算出外推量后一起推进 steady 的渐变。
    private var pendingDt = 0f
    private var steadyEyeTarget = 1f
    private var steadyFaceTarget = 1f
    private var eyeMakeupTarget = 1f

    // 有脸 / 无脸事件环（采样时间严格递增）：presence 按被画那一帧的采样时间判断它之前最近的事件。
    // 被 redetect 豁免的跳采漏检会改记为有脸，读这个环时不能把 eventFace 当成检测器的原始结果。
    private val eventTimes = LongArray(EVENTS)
    private val eventFace = BooleanArray(EVENTS)
    private var eventHead = 0
    private var eventCount = 0
    private var intervalKnown = false
    // 当前一轮从 runFrom（第一个样本）开始，渐入按 runStart 计（重新检出时由退出到一半的水平接着渐入，runStart 早于 runFrom）；
    // runFrom 之前的帧按上一轮 prevRunStart 计（没有上一轮为 MIN_VALUE，presence 为 0）。
    private var runFrom = Long.MIN_VALUE
    private var runStart = Long.MIN_VALUE
    private var prevRunStart = Long.MIN_VALUE
    // 已交给 at / meshAt 的最大被画帧采样时间：重新检出时判断这段无脸是否已被观众看到（进入渐出）。
    private var shownMax = Long.MIN_VALUE

    /** 最近一次 [at] / [meshAt] 那一帧的 presence（0..1）。 */
    var presence = 0f
        private set
    var offsetX = 0f
        private set
    var offsetY = 0f
        private set
    /** 脸中心速度，画面宽 / 秒（已去抖动）。 */
    var speed = 0f
        private set
    var steadyEye = 1f
        private set
    var steadyFace = 1f
        private set
    /** 精细眼妆（眼影/眼线/卧蚕）防残影保持系数：综合人脸速度与紧致外推衰减，快退（35 ms）慢回（160 ms）。 */
    var eyeMakeupKeep = 1f
        private set
    var denoiseKeep = 1f
        private set
    /** 新一轮（首个样本、断档、换人或丢脸后重新检出）时加一；形变器据此重置时间状态。 */
    var runId = 0
        private set
    /** 检测器结果（有脸或无脸）的间隔 EMA（τ = 500 ms，夹到 16..200 ms），驱动外推上限与丢脸防抖。 */
    var sampleIntervalMs = DEFAULT_INTERVAL_MS
        private set
    /** 最新样本的采样时间，没有为 MIN_VALUE。 */
    val latest: Long get() = if (count == 0) Long.MIN_VALUE else times[(head - 1 + HISTORY) % HISTORY]
    /**
     * 最近一次完成的检测里没有脸（最新事件是 [addMiss]）时，这一段连续无脸的第一个样本的采样时间；最近一次完成的检测有脸、
     * 被跳采豁免改记为有脸、或还没有任何结果时为 MIN_VALUE。与 [latest] 一起区分「检测器确实报告丢脸」与「检测器还在算」：
     * 人脸走 CPU、推理偏慢时，[latest] 的年龄在两次并入之间可达约 2 × 推理耗时 + 2 帧，并不表示丢脸（HairMaskTuning.faceKeep）。
     */
    val missSince: Long get() {
        if (eventCount == 0) return Long.MIN_VALUE
        return if (eventFace[(eventHead - 1 + EVENTS) % EVENTS]) Long.MIN_VALUE else missRunStart(0)
    }
    /** 外推上限：1.25 个采样间隔，夹到 40..100 ms（兜住延后不够、约 10% 需要外推的帧）。 */
    val horizonMs: Float get() = (EXTRAPOLATE_INTERVALS * sampleIntervalMs).coerceIn(EXTRAPOLATE_MIN_MS, EXTRAPOLATE_MAX_MS)
    /**
     * 第一个无脸样本之后保持满强度的时间：max(33, 1.2 × 采样间隔)；30 Hz 约 40 ms，15 Hz 约 80 ms（单次漏检不闪）。
     * 漏检那次推理超过一帧、下一帧跳采时，下一个有脸样本会晚于防抖才到，由 [redetect] 的跳采豁免兜住。
     */
    val missDebounceMs: Float get() = maxOf(MISS_DEBOUNCE_MIN_MS, MISS_DEBOUNCE_INTERVALS * sampleIntervalMs)

    /** 硬重置（换镜头、尺寸变化、代际变化）：清空全部历史与两眼睁眼比基线。 */
    fun reset() {
        clearHistory()
        meshFilter.reset(keepBaseline = false)
        sampleIntervalMs = DEFAULT_INTERVAL_MS; intervalKnown = false
        last = Long.MIN_VALUE; pendingDt = 0f; steadyEyeTarget = 1f; steadyFaceTarget = 1f; eyeMakeupTarget = 1f
        presence = 0f; offsetX = 0f; offsetY = 0f; speed = 0f; steadyEye = 1f; steadyFace = 1f; eyeMakeupKeep = 1f; denoiseKeep = 1f
        holdX = 0f; holdY = 0f
        runId++
    }

    /** 丢脸太久的清理：清空样本、事件与这一轮，保留两眼睁眼比基线与采样间隔。 */
    private fun clearHistory() {
        count = 0; head = 0; edxX = 0f; edxY = 0f; velocityX = 0f; velocityY = 0f
        sampleSpeed = 0f; sampleMotion = 0f; poseVelValid = false; poseMoving = false
        meshPresent.fill(false); meshHasIris.fill(false)
        eventHead = 0; eventCount = 0
        runFrom = Long.MIN_VALUE; runStart = Long.MIN_VALUE; prevRunStart = Long.MIN_VALUE
        shownMax = Long.MIN_VALUE
        meshFilter.reset(keepBaseline = true)
    }

    /** 新样本（41 维原始状态 + 可选 478 点全网格，采样时间 [sampleTs] 严格递增；不递增的忽略）。 */
    fun add(sampleTs: Long, target: FloatArray, mesh: FaceMeshFrame? = null) {
        require(target.size == FaceGeometrySmoother.STATE_SIZE)
        if (count > 0 && sampleTs <= latest) return
        // 断档（丢脸、卡顿）：从头开始，不跨空档插值。
        if (count > 0 && sampleTs - latest > MAX_GAP_MS) dropSamples()
        val slot = head
        if (mesh != null) {
            if (count == 0) meshFilter.reset(keepBaseline = true)
            meshFilter.filter(sampleTs, mesh.landmarks, aspect, meshCoords[slot], meshPose[slot])
            // 换人（中心或尺度突变）：旧样本不再插值，新的一轮从 0 渐入。
            if (meshFilter.jumped && count > 0) dropSamples()
            val meta = meshMeta[slot]
            if (!FaceMeshFrame.fillMeta(meshCoords[slot], aspect, meta)) mesh.writeMeta(meta)
            meta[FaceMeshFrame.META_RIGHT_IRIS_GATE] = meshFilter.rightGate
            meta[FaceMeshFrame.META_LEFT_IRIS_GATE] = meshFilter.leftGate
            meshPresent[slot] = true
            meshHasIris[slot] = mesh.hasIris
        } else {
            meshPresent[slot] = false
            meshHasIris[slot] = false
        }
        val x = target[OVAL_INDEX]; val y = target[OVAL_INDEX + 1] / aspect
        val prevSlot = (head - 1 + HISTORY) % HISTORY
        if (count == 0) {
            edxX = 0f; edxY = 0f
            target.copyInto(states[slot])
        } else {
            val previous = states[prevSlot]
            val te = (sampleTs - latest).toFloat()
            // One-Euro：原始脸中心在至少 SPEED_BASELINE_MS 的跨度上求速度（每帧采样时抖动不被 1/te 放大），
            // 按轴带符号低通后再取大小、以脸半径为单位，再定位置的截止频率。
            var base = prevSlot
            for (k in 1 until count) {
                val i = (head - 1 - k + HISTORY) % HISTORY
                if (sampleTs - times[i] > SPEED_BASELINE_MS) break
                base = i
            }
            val span = (sampleTs - times[base]).toFloat()
            val d = alpha(D_CUTOFF_HZ, te)
            edxX += d * ((x - rawX[base]) / span * 1000f - edxX)
            edxY += d * ((y - rawY[base]) / span * 1000f - edxY)
            val rx = maxOf(previous[RX_INDEX], RADIUS_MIN)
            val centerExcess = maxOf(0f, hypot(edxX, edxY) / rx - SPEED_DEAD)
            val a = alpha(FC_MIN_HZ + BETA * centerExcess, te)
            val out = states[slot]
            for (i in 0 until FaceGeometrySmoother.STATE_SIZE) out[i] = previous[i] + a * (target[i] - previous[i])
        }
        // 外推用的姿态速度：同一轮里相邻两个都有网格的样本。
        poseVelValid = mesh != null && count > 0 && meshPresent[prevSlot]
        if (poseVelValid) {
            val dt = (sampleTs - times[prevSlot]).toFloat()
            for (k in 0 until FaceMeshFilter.POSE_SIZE) poseVel[k] = (meshPose[slot][k] - meshPose[prevSlot][k]) / dt
        }
        poseMoving = mesh != null && meshFilter.poseSpeed > meshFilter.poseDead
        val restart = count == 0
        rawX[slot] = x; rawY[slot] = y
        normalizeAxis(states[slot])
        times[slot] = sampleTs
        head = (head + 1) % HISTORY; count = minOf(count + 1, HISTORY)
        // 这一轮：断档 / 换人从 0 渐入；丢脸超过防抖后重新检出，从退出到的水平接着渐入（不跳变）。
        if (restart) startRun(sampleTs, 0f, hard = true) else redetect(sampleTs)
        pushEvent(sampleTs, true)
        updateVelocity(sampleTs, slot)
        if (mesh != null) {
            val s = meshFilter.poseScale
            sampleSpeed = s * maxOf(0f, meshFilter.poseSpeed - meshFilter.poseDead)
            sampleMotion = meshFilter.motionSpeed
        } else {
            sampleSpeed = hypot(velocityX, velocityY) * 1000f
            sampleMotion = 0f
        }
    }

    /**
     * 检测器在采样时间 [sampleTs] 的结果里没有脸（或被遮挡、跟丢）。presence 从这一段无脸的第一个样本起算防抖与渐出。
     * 不递增（不晚于上一个有脸 / 无脸样本）的忽略。
     */
    fun addMiss(sampleTs: Long) {
        pushEvent(sampleTs, false)
    }

    /**
     * 每次绘制调用一次（[nowMs] 为渲染时间）：推进 steady 与降噪的渐变；最新样本已过去 [CLEAR_AFTER_MS] 就清空历史
     * （保留两眼睁眼比基线）。presence 由 [at] / [meshAt] 按被画那一帧的采样时间计算。
     */
    fun update(nowMs: Long) {
        val dt = if (last == Long.MIN_VALUE) 0f else (nowMs - last).coerceIn(0L, DT_CLAMP_MS).toFloat()
        last = nowMs
        if (count > 0 && nowMs - latest > CLEAR_AFTER_MS) clearHistory()
        if (count > 0 && nowMs - latest > MAX_GAP_MS) { velocityX = 0f; velocityY = 0f; sampleSpeed = 0f; sampleMotion = 0f; poseMoving = false }
        if (count == 0) { sampleSpeed = 0f; sampleMotion = 0f }
        speed = sampleSpeed
        denoiseKeep = approach(denoiseKeep, 1f - smoothstep(DENOISE_FADE[0], DENOISE_FADE[1], maxOf(speed, sampleMotion)), dt)
        steadyEyeTarget = 1f; steadyFaceTarget = 1f; eyeMakeupTarget = 1f
        pendingDt = dt
    }

    /**
     * 采样时间 [t]（被画那一帧）的几何；没有几何时为 null。返回的数组下次调用前有效，调用方只读。
     * 同时更新 [presence]、[offsetX]/[offsetY]、[steadyEye]/[steadyFace]（每次绘制在 [update] 之后调用一次）。
     */
    fun at(t: Long): FloatArray? {
        presence = presenceAt(t)
        if (count == 0) { offsetX = 0f; offsetY = 0f; settle(); return null }
        val newest = (head - 1 + HISTORY) % HISTORY
        val oldest = (head - count + HISTORY) % HISTORY
        offsetX = 0f; offsetY = 0f
        when {
            t >= times[newest] -> {
                // 比最新样本还新：平移外推（有网格时取刚体姿态外推的脸中心位移），外推量交给 dilate / steady 兜底。
                states[newest].copyInto(output)
                val horizon = minOf((t - times[newest]).toFloat(), horizonMs)
                var mx = 0f; var my = 0f
                if (meshPresent[newest]) {
                    if (poseVelValid && poseMoving && horizon > 0f) {
                        FaceMeshFilter.centerShift(meshCoords[newest], meshPose[newest], poseVel, horizon * EXTRAPOLATE_DAMP, aspect, shift)
                        mx = shift[0]; my = shift[1]
                    }
                } else {
                    mx = velocityX * horizon; my = velocityY * horizon
                }
                if (mx != 0f || my != 0f) {
                    for (i in X_POSITIONS) output[i] += mx
                    for (i in Y_POSITIONS) output[i] += my * aspect
                    // 真实位置在保持位置与外推位置之间：offset 指回保持位置。
                    offsetX = -mx; offsetY = -my
                }
            }
            t <= times[oldest] -> states[oldest].copyInto(output)
            else -> {
                var k = newest
                while (times[(k - 1 + HISTORY) % HISTORY] > t) k = (k - 1 + HISTORY) % HISTORY
                val a = states[(k - 1 + HISTORY) % HISTORY]; val b = states[k]
                val ta = times[(k - 1 + HISTORY) % HISTORY]; val tb = times[k]
                val g = (t - ta).toFloat() / (tb - ta)
                for (i in 0 until FaceGeometrySmoother.STATE_SIZE) output[i] = a[i] + g * (b[i] - a[i])
                normalizeAxis(output)
            }
        }
        // 外推量：变大立刻跟上，变小按 HOLD_DECAY_MS 指数回落。
        if (hypot(offsetX, offsetY) >= hypot(holdX, holdY)) { holdX = offsetX; holdY = offsetY }
        else {
            val k = exp(-pendingDt / HOLD_DECAY_MS)
            holdX = offsetX + (holdX - offsetX) * k; holdY = offsetY + (holdY - offsetY) * k
        }
        val distance = hypot(holdX, holdY)
        val eye = maxOf(output[EYE_RADIUS_INDEX], output[EYE_RADIUS_INDEX + 3], RADIUS_FLOOR)
        val face = maxOf(output[RX_INDEX], RADIUS_FLOOR)
        steadyEyeTarget = 1f - smoothstep(EYE_FADE[0], EYE_FADE[1], distance / eye)
        steadyFaceTarget = 1f - smoothstep(FACE_FADE[0], FACE_FADE[1], distance / face)
        val motion = maxOf(speed, sampleMotion)
        val motionKeep = 1f - 0.85f * smoothstep(EYE_MAKEUP_MOTION[0], EYE_MAKEUP_MOTION[1], motion)
        val extrapKeep = 1f - smoothstep(EYE_MAKEUP_EXTRAP[0], EYE_MAKEUP_EXTRAP[1], distance / eye)
        eyeMakeupTarget = motionKeep * extrapKeep
        settle()
        return output
    }

    /**
     * 采样时间 [t] 的脸椭圆位姿，写入 [out] = (cx, cy, rx, ry, axisX, axisY)：与合成的 uOval、轴向同一坐标系
     * （中心为纹理坐标，半径与轴向在度量空间，y 已除以 [aspect]）。只在历史样本区间 [最早样本, 最新样本] 内线性插值、
     * 轴向重新归一化，不外推；[t] 不在区间内（含没有样本）返回 false，不写 [out]。
     * 无副作用：不改 presence、外推量与 steady，同一帧可在 [at] 前后为多个时刻调用（头发遮罩要取遮罩时刻与显示时刻两个位姿）。
     */
    fun poseAt(t: Long, out: FloatArray): Boolean {
        require(out.size >= POSE_SIZE)
        if (count == 0) return false
        val newest = (head - 1 + HISTORY) % HISTORY
        val oldest = (head - count + HISTORY) % HISTORY
        if (t < times[oldest] || t > times[newest]) return false
        if (t == times[oldest]) {
            states[oldest].copyInto(out, 0, OVAL_INDEX, OVAL_INDEX + POSE_SIZE)
            return true
        }
        // 与 at() 相同的找区间：times[k − 1] < t ≤ times[k]。
        var k = newest
        while (times[(k - 1 + HISTORY) % HISTORY] >= t) k = (k - 1 + HISTORY) % HISTORY
        val ia = (k - 1 + HISTORY) % HISTORY
        val a = states[ia]; val b = states[k]
        val g = (t - times[ia]).toFloat() / (times[k] - times[ia])
        for (i in 0 until POSE_SIZE) out[i] = a[OVAL_INDEX + i] + g * (b[OVAL_INDEX + i] - a[OVAL_INDEX + i])
        val n = hypot(out[4], out[5])
        if (n > 1e-6f) { out[4] /= n; out[5] /= n }
        return true
    }

    /**
     * 采样时间 [t]（被画那一帧）的 478 点全人脸网格与元数据；无网格样本时返回 null。
     * 与 [at] 使用一致的插值比例；比最新样本还新时按刚体姿态外推（与 [at] 同一上限与阻尼），返回复用的 [SmoothedMesh] 视图（只读）。
     */
    fun meshAt(t: Long): SmoothedMesh? {
        presence = presenceAt(t)
        if (count == 0) return null
        val newest = (head - 1 + HISTORY) % HISTORY
        val oldest = (head - count + HISTORY) % HISTORY
        when {
            t >= times[newest] -> {
                if (!meshPresent[newest]) return null
                meshMeta[newest].copyInto(meshOutMeta)
                val horizon = minOf((t - times[newest]).toFloat(), horizonMs) * EXTRAPOLATE_DAMP
                if (poseVelValid && poseMoving && horizon > 0f) {
                    // 只推刚体姿态，不推表情残差：急停时五官不过冲。
                    FaceMeshFilter.extrapolate(meshCoords[newest], meshPose[newest], poseVel, horizon, aspect, meshOutCoords)
                    FaceMeshFilter.centerShift(meshCoords[newest], meshPose[newest], poseVel, horizon, aspect, shift)
                    meshOutMeta[FaceMeshFrame.META_CENTER_X] += shift[0]
                    meshOutMeta[FaceMeshFrame.META_CENTER_Y] += shift[1] * aspect
                } else {
                    meshCoords[newest].copyInto(meshOutCoords)
                }
                meshView.hasIris = meshHasIris[newest]
            }
            t <= times[oldest] -> {
                if (!meshPresent[oldest]) return null
                meshCoords[oldest].copyInto(meshOutCoords)
                meshMeta[oldest].copyInto(meshOutMeta)
                meshView.hasIris = meshHasIris[oldest]
            }
            else -> {
                var k = newest
                while (times[(k - 1 + HISTORY) % HISTORY] > t) k = (k - 1 + HISTORY) % HISTORY
                val iA = (k - 1 + HISTORY) % HISTORY
                val iB = k
                val hasA = meshPresent[iA]
                val hasB = meshPresent[iB]
                if (!hasA && !hasB) return null
                if (hasA && hasB) {
                    val g = (t - times[iA]).toFloat() / (times[iB] - times[iA])
                    val aC = meshCoords[iA]; val bC = meshCoords[iB]
                    val aM = meshMeta[iA]; val bM = meshMeta[iB]
                    for (i in 0 until FaceMeshFrame.COORD_COUNT) meshOutCoords[i] = aC[i] + g * (bC[i] - aC[i])
                    for (i in 0 until FaceMeshFrame.META_SIZE) meshOutMeta[i] = aM[i] + g * (bM[i] - aM[i])
                    normalizeMetaAxis(meshOutMeta)
                    meshView.hasIris = meshHasIris[iA] && meshHasIris[iB]
                } else {
                    val src = if (hasB) iB else iA
                    meshCoords[src].copyInto(meshOutCoords)
                    meshMeta[src].copyInto(meshOutMeta)
                    meshView.hasIris = meshHasIris[src]
                }
            }
        }
        return meshView
    }

    /**
     * 采样时间 [t] 的 presence = 渐入 × 退出。退出看 t 之前最近的事件：
     * - 有脸样本，且之后还有事件：1；
     * - 有脸样本，之后没有任何结果（检测器忙）：保持 [busyHoldMs] 后 [EXIT_RAMP_MS] 线性降到 0；
     * - 无脸样本：从这一段无脸的第一个样本起，防抖 [missDebounceMs] 后 [EXIT_RAMP_MS] 线性降到 0。
     */
    private fun presenceAt(t: Long): Float {
        if (t > shownMax) shownMax = t
        if (count == 0 || runFrom == Long.MIN_VALUE) return 0f
        val start = if (t >= runFrom) runStart else prevRunStart
        if (start == Long.MIN_VALUE || t <= start) return 0f
        val fadeIn = smoothstep(0f, 1f, (t - start).toFloat() / T_IN_MS)
        if (fadeIn <= 0f) return 0f
        var e = -1
        for (k in 0 until eventCount) {
            val i = (eventHead - 1 - k + EVENTS) % EVENTS
            if (eventTimes[i] <= t) { e = k; break }
        }
        if (e < 0) return fadeIn
        val idx = (eventHead - 1 - e + EVENTS) % EVENTS
        val exit = if (eventFace[idx]) {
            if (e > 0) 1f else 1f - ((t - eventTimes[idx]).toFloat() - busyHoldMs).coerceIn(0f, EXIT_RAMP_MS) / EXIT_RAMP_MS
        } else {
            val missStart = missRunStart(e)
            1f - ((t - missStart).toFloat() - missDebounceMs).coerceIn(0f, EXIT_RAMP_MS) / EXIT_RAMP_MS
        }
        return fadeIn * exit
    }

    /** 检测器忙（最新样本之后没有结果）时的保持时间：clamp(2 × 采样间隔 + 50, 100, 250) ms。 */
    private val busyHoldMs: Float get() = (2f * sampleIntervalMs + 50f).coerceIn(BUSY_HOLD_MIN_MS, BUSY_HOLD_MAX_MS)

    /** 从倒数第 [fromNewest] 个事件（无脸）往前，这一段连续无脸的第一个样本时间。 */
    private fun missRunStart(fromNewest: Int): Long {
        var start = eventTimes[(eventHead - 1 - fromNewest + EVENTS) % EVENTS]
        for (k in fromNewest + 1 until eventCount) {
            val i = (eventHead - 1 - k + EVENTS) % EVENTS
            if (eventFace[i]) break
            start = eventTimes[i]
        }
        return start
    }

    private fun startRun(sampleTs: Long, level: Float, hard: Boolean) {
        prevRunStart = if (hard) Long.MIN_VALUE else runStart
        runFrom = sampleTs
        // 从 level 接着渐入：smoothstep 的反函数 x = 1/2 − sin(asin(1 − 2y) / 3)。
        val x = if (level <= 0f) 0f else (0.5 - sin(asin(1.0 - 2.0 * level.coerceIn(0f, 1f)) / 3.0)).toFloat()
        runStart = sampleTs - (x * T_IN_MS).toLong()
        runId++
    }

    /**
     * 有脸样本到达前最近的事件是无脸、且这一段无脸已超过防抖（presence 已开始退出）：开始新的一轮。
     * 例外是观众没看到的跳采漏检（见下），把这段无脸改记为有脸后直接返回。
     */
    private fun redetect(sampleTs: Long) {
        if (eventCount == 0) return
        val newest = (eventHead - 1 + EVENTS) % EVENTS
        if (eventFace[newest]) return
        val missStart = missRunStart(0)
        // 漏检后检测器跳采（那次推理超过一帧，下一帧不采）：这段无脸不超过一个退出窗口（防抖 + 渐出，30 Hz 约 80 ms），
        // 且还没有任何被画帧进入渐出时，视为观众没看到的单次漏检：改记为有脸，不开新一轮（否则从约 0.33 重新渐入，下陷约 100 ms）。
        // 两个条件保证：不会让内容晚于第一个无脸样本一个退出窗口的帧继续画效果，也不会在已画出的渐出之后跳回满强度。
        if (sampleTs - missStart <= missDebounceMs + EXIT_RAMP_MS && shownMax <= missStart + missDebounceMs.toLong()) {
            for (k in 0 until eventCount) {
                val i = (eventHead - 1 - k + EVENTS) % EVENTS
                if (eventFace[i]) break
                eventFace[i] = true
            }
            return
        }
        val missed = (sampleTs - missStart).toFloat() - missDebounceMs
        if (missed <= 0f) return
        val level = 1f - missed.coerceIn(0f, EXIT_RAMP_MS) / EXIT_RAMP_MS
        startRun(sampleTs, level, hard = false)
    }

    /** 记一个事件并更新检测器结果间隔的 EMA（首个间隔直接取值）。 */
    private fun pushEvent(ts: Long, face: Boolean) {
        if (eventCount > 0) {
            val lastTs = eventTimes[(eventHead - 1 + EVENTS) % EVENTS]
            if (ts <= lastTs) return
            val dt = (ts - lastTs).toFloat()
            if (dt <= MAX_GAP_MS) {
                sampleIntervalMs = if (!intervalKnown) dt
                    else sampleIntervalMs + (1f - exp(-dt / INTERVAL_TAU_MS)) * (dt - sampleIntervalMs)
                sampleIntervalMs = sampleIntervalMs.coerceIn(INTERVAL_MIN_MS, INTERVAL_MAX_MS)
                intervalKnown = true
            }
        }
        eventTimes[eventHead] = ts; eventFace[eventHead] = face
        eventHead = (eventHead + 1) % EVENTS; eventCount = minOf(eventCount + 1, EVENTS)
    }

    /** 丢掉已有样本（断档、换人；写入位置不变），事件记录保留，当前样本按硬重开处理。 */
    private fun dropSamples() {
        count = 0; edxX = 0f; edxY = 0f; velocityX = 0f; velocityY = 0f
        poseVelValid = false; poseMoving = false
        meshPresent.fill(false); meshHasIris.fill(false)
    }

    /** 无网格时的脸中心速度：最近 BASELINE_MS 内最早的滤波样本到最新样本，去掉抖动死区（静止抖动不算移动）。 */
    private fun updateVelocity(sampleTs: Long, slot: Int) {
        var oldest = -1
        for (k in 1 until count) {
            val i = (head - 1 - k + HISTORY) % HISTORY
            if (sampleTs - times[i] > BASELINE_MS) break
            oldest = i
        }
        if (oldest < 0) { velocityX = 0f; velocityY = 0f; return }
        val span = (sampleTs - times[oldest]).toFloat()
        val newest = states[slot]; val base = states[oldest]
        var mx = newest[OVAL_INDEX] - base[OVAL_INDEX]
        var my = (newest[OVAL_INDEX + 1] - base[OVAL_INDEX + 1]) / aspect
        val length = hypot(mx, my)
        val jitter = STEP_JITTER * maxOf(newest[RX_INDEX], RADIUS_MIN)
        val keep = if (length <= jitter) 0f else (length - jitter) / length
        mx *= keep; my *= keep
        velocityX = mx / span; velocityY = my / span
    }

    private fun normalizeMetaAxis(meta: FloatArray) {
        val n = hypot(meta[FaceMeshFrame.META_AXIS_X], meta[FaceMeshFrame.META_AXIS_Y])
        if (n > 1e-6f) {
            meta[FaceMeshFrame.META_AXIS_X] /= n
            meta[FaceMeshFrame.META_AXIS_Y] /= n
        }
    }

    private fun settle() {
        steadyEye = approach(steadyEye, steadyEyeTarget, pendingDt)
        steadyFace = approach(steadyFace, steadyFaceTarget, pendingDt)
        eyeMakeupKeep = approach(eyeMakeupKeep, eyeMakeupTarget, pendingDt, EYE_MAKEUP_OUT_MS, EYE_MAKEUP_IN_MS)
        pendingDt = 0f
    }

    /**
     * 脸、唇、眼椭圆按（回落平滑后的）外推量的一半平移并放大，使各自的完全生效核心区同时盖住平移前后两个位置
     * 之间的整段；其余分量原样。位移可忽略时直接返回 [geometry]，否则写入并返回 [out]。
     */
    fun dilate(geometry: FloatArray, out: FloatArray): FloatArray {
        val ex = holdX / 2; val ey = holdY / 2
        if (hypot(ex, ey) < MIN_SHIFT) return geometry
        val n = hypot(geometry[FaceGeometrySmoother.AXIS_INDEX], geometry[FaceGeometrySmoother.AXIS_INDEX + 1])
        if (n < 1e-6f) return geometry
        geometry.copyInto(out, 0, 0, FaceGeometrySmoother.STATE_SIZE)
        val ux = geometry[FaceGeometrySmoother.AXIS_INDEX] / n; val uy = geometry[FaceGeometrySmoother.AXIS_INDEX + 1] / n
        val along = ex * ux + ey * uy // 着色器 ellipseE 的 u（横半径方向）
        val across = -ex * uy + ey * ux // v = (−u.y, u.x)（竖半径方向）
        // 眼、唇椭圆放大后不超过脸椭圆的大半径，只防退化；外推量已限在 horizonMs 内。
        val limit = maxOf(geometry[OVAL_INDEX + 2], geometry[OVAL_INDEX + 3])
        for (i in ELLIPSES.indices) {
            val base = ELLIPSES[i]
            val core = sqrt(CORES[i])
            val rx = maxOf(geometry[base + 2], RADIUS_MIN); val ry = maxOf(geometry[base + 3], RADIUS_MIN)
            // 以核心区为单位圆：要盖住圆心在 ±(x, y) 的两个单位圆。先按各轴位移拉长 (1 + x, 1 + y)，
            // 再取覆盖所需的统一倍数 s（逐角度取最大，留 1% 余量）。
            val x = abs(along / rx) / core; val y = abs(across / ry) / core
            val px = 1f + x; val py = 1f + y
            var worst = 0f
            for (j in COS.indices) {
                val a = (COS[j] + x) / px; val b = (SIN[j] + y) / py
                worst = maxOf(worst, a * a + b * b)
            }
            val s = sqrt(worst) * 1.01f
            out[base] = geometry[base] + ex
            out[base + 1] = geometry[base + 1] + ey * aspect
            val cap = if (base == OVAL_INDEX) Float.MAX_VALUE else maxOf(limit, geometry[base + 2], geometry[base + 3])
            out[base + 2] = minOf(geometry[base + 2] * px * s, cap)
            out[base + 3] = minOf(geometry[base + 3] * py * s, cap)
        }
        return out
    }

    private fun normalizeAxis(state: FloatArray) {
        val n = hypot(state[FaceGeometrySmoother.AXIS_INDEX], state[FaceGeometrySmoother.AXIS_INDEX + 1])
        if (n > 1e-6f) { state[FaceGeometrySmoother.AXIS_INDEX] /= n; state[FaceGeometrySmoother.AXIS_INDEX + 1] /= n }
    }

    /** 渐出快、渐入慢。 */
    private fun approach(current: Float, target: Float, dt: Float, outMs: Float = FADE_OUT_MS, inMs: Float = FADE_IN_MS): Float =
        if (target < current) maxOf(target, current - dt / outMs)
        else minOf(target, current + dt / inMs)

    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3 - 2 * t)
    }

    companion object {
        /** [poseAt] 的输出长度：(cx, cy, rx, ry, axisX, axisY)，即状态向量的 oval 与 axis 两段（下标 22..27）。 */
        const val POSE_SIZE = 6
        private const val OVAL_INDEX = 22
        private const val RX_INDEX = FaceGeometrySmoother.RX_INDEX
        private const val EYE_RADIUS_INDEX = 18 // eyes[16..21] = 两眼 (x, y, 变形半径)
        /** 位置类分量（纹理坐标 x / y）：cheeks 8 个点、两眼中心、oval、mouth、两个 eyeMask 的中心。 */
        private val X_POSITIONS = intArrayOf(0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 28, 32, 36)
        private val Y_POSITIONS = intArrayOf(1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 23, 29, 33, 37)
        /** 椭圆 (cx, cy, rx, ry) 在状态向量里的起点：oval、mouth、eyeMasks[0]、eyeMasks[1]。 */
        private val ELLIPSES = intArrayOf(OVAL_INDEX, 28, 32, 36)
        /** 合成着色器 MASKS 里各椭圆完全生效的核心（E 小于该值）：脸 0.50、唇 0.65、眼 0.55。 */
        val CORES = floatArrayOf(0.50f, 0.65f, 0.55f, 0.55f)
        private val COS = FloatArray(32) { cos(2 * PI * it / 32).toFloat() }
        private val SIN = FloatArray(32) { sin(2 * PI * it / 32).toFloat() }
        private const val HISTORY = 16
        private const val EVENTS = 16
        private const val RADIUS_MIN = 1e-3f
        private const val RADIUS_FLOOR = 0.01f
        private const val MIN_SHIFT = 1e-3f
        /**
         * 41 维 One-Euro：静止截止 1 Hz；速度（脸半径 / 秒，先按 4 Hz 低通）超过死区 0.25 后每 1 个半径 / 秒加 10 Hz。
         * 0.3 画面宽 / 秒的平移（脸半径约 0.2）截止约 13 Hz，每帧滞后不到 5 ms。478 点网格见 FaceMeshFilter。
         */
        const val FC_MIN_HZ = 1f
        const val BETA = 10f
        const val SPEED_DEAD = 0.25f
        const val D_CUTOFF_HZ = 4f
        /** One-Euro 求速度的最短时间基线。 */
        const val SPEED_BASELINE_MS = 100L
        /** 外推量变小时的回落时间常数。 */
        const val HOLD_DECAY_MS = 200f
        /** 两个样本相隔超过这么久（丢脸、卡顿）就从头开始，不跨空档插值。 */
        const val MAX_GAP_MS = 300L
        /** 外推上限 = 1.25 个采样间隔，夹到 40..100 ms（41 维与网格共用）；网格与有网格时的 41 维再乘阻尼。 */
        const val EXTRAPOLATE_INTERVALS = 1.25f
        const val EXTRAPOLATE_MIN_MS = 40f
        const val EXTRAPOLATE_MAX_MS = 100f
        const val EXTRAPOLATE_DAMP = 0.85f
        /** 无网格时求速度的时间基线与静止抖动死区（脸椭圆横半径的比例；216×384 小图上约 0.8 采样像素）。 */
        const val BASELINE_MS = 200L
        const val STEP_JITTER = 0.015f
        /** 新一轮渐入。 */
        const val T_IN_MS = 150f
        /** 丢脸防抖：max(33, 1.2 × 采样间隔)；之后 40 ms 渐出。 */
        const val MISS_DEBOUNCE_MIN_MS = 33f
        const val MISS_DEBOUNCE_INTERVALS = 1.2f
        const val EXIT_RAMP_MS = 40f
        /** 检测器忙时的保持上下限。 */
        const val BUSY_HOLD_MIN_MS = 100f
        const val BUSY_HOLD_MAX_MS = 250f
        /** 最新样本过去这么久（渲染时间）就清空历史。 */
        const val CLEAR_AFTER_MS = 500L
        /** 采样间隔 EMA 的时间常数、上下限与未知时的缺省值（30 Hz）。 */
        const val INTERVAL_TAU_MS = 500f
        const val INTERVAL_MIN_MS = 16f
        const val INTERVAL_MAX_MS = 200f
        const val DEFAULT_INTERVAL_MS = 1000f / 30f
        private const val DT_CLAMP_MS = 100L
        /** 外推量 / 眼睛变形半径：0.35 起渐出，0.8 完全渐出。 */
        private val EYE_FADE = floatArrayOf(0.35f, 0.8f)
        /** 外推量 / 脸椭圆横半径：0.15 起渐出，0.35 完全渐出。 */
        private val FACE_FADE = floatArrayOf(0.15f, 0.35f)
        /** 精细眼妆（眼影/眼线/卧蚕）速度渐出区间（画面宽 / 秒）：静止与轻点头满强度，快速转头/平移软隐消残影。 */
        private val EYE_MAKEUP_MOTION = floatArrayOf(0.14f, 0.42f)
        /** 精细眼妆外推渐出区间（外推量 / 眼半径）：约 5 px 即起衰、15 px 归零。 */
        private val EYE_MAKEUP_EXTRAP = floatArrayOf(0.12f, 0.35f)
        private const val EYE_MAKEUP_OUT_MS = 35f
        private const val EYE_MAKEUP_IN_MS = 160f
        /** 脸部运动速度（画面宽 / 秒，已去抖动）：0.05 起渐出降噪，0.12 完全停用。 */
        private val DENOISE_FADE = floatArrayOf(0.05f, 0.12f)
        private const val FADE_OUT_MS = 80f
        private const val FADE_IN_MS = 300f

        private fun alpha(hz: Float, te: Float): Float = (1.0 - exp(-2.0 * PI * hz * te / 1000.0)).toFloat()
    }
}
