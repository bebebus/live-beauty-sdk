package io.bebebus.livebeauty

import kotlin.math.abs
import kotlin.math.hypot

/**
 * 原始纹理坐标，y向上；不保存图像或身份。规格 v1 第 2 节，参考 reference/beauty_ref.py 的 geometry_with_reason()。
 * 中心（cheeks、eyes 的 xy，oval/mouth/eyeMasks 的 cx, cy）是纹理坐标；半径（eyes.z、rx、ry）是度量值
 * （p' = (x, y/aspect) 的 x 单位）；axis 是度量空间单位向量 u，v = (−u.y, u.x) 指向额头。
 */
data class FaceGeometry(
    val cheeks: FloatArray,
    val eyes: FloatArray,
    val oval: FloatArray,
    val axis: FloatArray,
    val mouth: FloatArray,
    val eyeMasks: FloatArray,
    /** 1 − smoothstep(0.24, 0.30, 侧脸比值)，只乘在塑形强度上。 */
    val warpWeight: Float,
) {
    /** 平滑器状态向量（41 维，布局见 FaceGeometrySmoother）。 */
    fun toState(): FloatArray = cheeks + eyes + oval + axis + mouth + eyeMasks + floatArrayOf(warpWeight)

    companion object {
        private val warpPoints = intArrayOf(132, 361, 172, 397, 33, 133, 362, 263, 1)
        private val maskPoints = intArrayOf(234, 454, 10, 152, 61, 291, 0, 17)

        private fun smoothstep(e0: Double, e1: Double, x: Double): Double {
            val t = ((x - e0) / (e1 - e0)).coerceIn(0.0, 1.0)
            return t * t * (3 - 2 * t)
        }

        /** 全有或全无：任一点非有限或越界、侧脸过大、尺寸退化都返回 null。内部用 Double 计算。 */
        fun fromMesh(points: List<Pair<Float, Float>>, aspect: Float): FaceGeometry? {
            if (points.size < 468 || !aspect.isFinite() || aspect <= 0) return null
            fun valid(i: Int, min: Float, max: Float) = points[i].let { (x, y) ->
                x.isFinite() && y.isFinite() && x in min..max && y in min..max }
            if (warpPoints.any { !valid(it, 0f, 1f) }) return null
            // 遮罩点允许略出画面（额头、下巴常被裁切）。
            if (maskPoints.any { !valid(it, -0.25f, 1.25f) }) return null
            val ratio = aspect.toDouble()
            fun p(i: Int) = points[i].let { doubleArrayOf(it.first.toDouble(), 1.0 - it.second) }
            fun distance(a: DoubleArray, b: DoubleArray) = hypot(a[0]-b[0], (a[1]-b[1])/ratio)
            val left = p(132); val right = p(361)
            val span = distance(left, right)
            if (span !in 0.1..0.85) return null
            // 侧脸比值 r = 既有侧脸量 ÷ span；r > 0.45（约 28°）停用，0.24..0.30 之间塑形渐出。
            val nose = p(1)
            val centreX = (left[0]+right[0])/2; val centreY = (left[1]+right[1])/2
            val yaw = abs((nose[0]-centreX)*(right[0]-left[0]) +
                (nose[1]-centreY)*(right[1]-left[1])/(ratio*ratio))/span
            val r = yaw/span
            if (r > 0.45) return null
            val warpWeight = 1.0 - smoothstep(0.24, 0.30, r)
            val cheeks = FloatArray(16)
            for ((pairIndex, pair) in listOf(132 to 361, 172 to 397).withIndex()) {
                val a = p(pair.first); val b = p(pair.second)
                val midX = (a[0]+b[0])/2; val midY = (a[1]+b[1])/2
                for ((j, point) in listOf(a, b).withIndex()) {
                    val k = (pairIndex*2+j)*4
                    cheeks[k] = point[0].toFloat(); cheeks[k+1] = point[1].toFloat()
                    cheeks[k+2] = midX.toFloat(); cheeks[k+3] = midY.toFloat()
                }
            }
            val eyes = FloatArray(6)
            for ((i, pair) in listOf(33 to 133, 362 to 263).withIndex()) {
                val a = p(pair.first); val b = p(pair.second); val width = distance(a, b)
                if (width < 0.015 || width > span*0.5) return null
                eyes[i*3] = ((a[0]+b[0])/2).toFloat(); eyes[i*3+1] = ((a[1]+b[1])/2).toFloat()
                eyes[i*3+2] = (width*0.9).toFloat()
            }
            // 以下在度量空间计算，中心转回纹理坐标时 y 乘 aspect。
            fun m(i: Int) = p(i).let { doubleArrayOf(it[0], it[1]/ratio) }
            fun dot(a: DoubleArray, b: DoubleArray) = a[0]*b[0] + a[1]*b[1]
            fun sub(a: DoubleArray, b: DoubleArray) = doubleArrayOf(a[0]-b[0], a[1]-b[1])
            fun mid(a: DoubleArray, b: DoubleArray) = doubleArrayOf((a[0]+b[0])/2, (a[1]+b[1])/2)
            val p234 = m(234); val p454 = m(454); val p10 = m(10); val p152 = m(152)
            val across = sub(p454, p234)
            val faceWidth = hypot(across[0], across[1])
            if (faceWidth < 0.05) return null
            var u = doubleArrayOf(across[0]/faceWidth, across[1]/faceWidth)
            if (dot(doubleArrayOf(-u[1], u[0]), sub(p10, p152)) < 0) u = doubleArrayOf(-u[0], -u[1])
            if (!u[0].isFinite() || !u[1].isFinite() || abs(hypot(u[0], u[1]) - 1.0) >= 1e-6) return null
            val v = doubleArrayOf(-u[1], u[0])
            // u 分量取侧向点对中点与中线点对中点的平均，v 分量只取中线点对中点。
            fun centre(sideA: DoubleArray, sideB: DoubleArray, midA: DoubleArray, midB: DoubleArray): FloatArray {
                val side = mid(sideA, sideB); val line = mid(midA, midB)
                val cu = 0.5*(dot(side, u) + dot(line, u)); val cv = dot(line, v)
                return floatArrayOf((cu*u[0] + cv*v[0]).toFloat(), ((cu*u[1] + cv*v[1])*ratio).toFloat())
            }
            val ovalCentre = centre(p234, p454, p10, p152)
            val ovalRx = 0.5*faceWidth*1.12
            val ovalRy = 0.5*abs(dot(sub(p10, p152), v))*1.15
            if (ovalRx < 0.03 || ovalRy < 0.03) return null
            val p61 = m(61); val p291 = m(291); val p0 = m(0); val p17 = m(17)
            val mouthCentre = centre(p61, p291, p0, p17)
            val mouthRx = 0.5*abs(dot(sub(p61, p291), u))*1.20
            if (mouthRx < 0.005) return null
            val mouthRy = maxOf(0.5*abs(dot(sub(p0, p17), v))*1.25, 0.25*mouthRx)
            val eyeMasks = FloatArray(8)
            for ((i, pair) in listOf(33 to 133, 362 to 263).withIndex()) {
                val a = m(pair.first); val b = m(pair.second)
                val width = hypot(a[0]-b[0], a[1]-b[1])
                val c = mid(a, b)
                eyeMasks[i*4] = (c[0] + 0.04*width*v[0]).toFloat()
                eyeMasks[i*4+1] = ((c[1] + 0.04*width*v[1])*ratio).toFloat()
                eyeMasks[i*4+2] = (0.72*width).toFloat(); eyeMasks[i*4+3] = (0.36*width).toFloat()
            }
            return FaceGeometry(cheeks, eyes,
                floatArrayOf(ovalCentre[0], ovalCentre[1], ovalRx.toFloat(), ovalRy.toFloat()),
                floatArrayOf(u[0].toFloat(), u[1].toFloat()),
                floatArrayOf(mouthCentre[0], mouthCentre[1], mouthRx.toFloat(), mouthRy.toFloat()),
                eyeMasks, warpWeight.toFloat())
        }
    }
}

/** 单任务、限频、采样时钟有效期和代际取消；worker不能复活旧镜头结果。 */
class FaceTrackingState {
    data class Ticket(val epoch: Long, val timestamp: Long, val width: Int, val height: Int)
    /**
     * 一次加锁取出：代际变化时 GL 线程的平滑器自行 reset；新样本以 sampleTimestamp 变化识别。
     * [completed] 为本代际最近一次完成（不论有没有脸）的采样时间，画面对齐用它判断哪些帧的人脸结果已经出来；没有为 MIN_VALUE。
     * [inFlight] 为有一张票已取出、worker 还没 [complete]（单任务在途；与 [completed]、几何同一把锁取出，刚完成的结果与
     * 「已不在途」同帧可见）。DEFERRED 只 [due]、还没取票的那一帧不算；[reset] 之后旧代际的票交回之前仍算（它的结果会被丢弃，
     * 时间线此时已随代际清空）。GL 线程据此区分「检测器还在算」与停摆（HairMaskTuning.faceKeep 的停摆兜底只在不在途时生效）。
     */
    data class Snapshot(val epoch: Long, val sampleTimestamp: Long, val geometry: FaceGeometry?,
                        val completed: Long=Long.MIN_VALUE, val mesh: FaceMeshFrame?=null, val inFlight: Boolean=false)
    private var epoch=0L
    private var busy=false
    private var lastSample=Long.MIN_VALUE
    private var lastSubmitted=-1L
    private var dimensions=0 to 0
    private var enabled=false
    private var latest: Triple<Ticket,FaceGeometry,FaceMeshFrame?>?=null
    private var completed=Long.MIN_VALUE
    @Synchronized fun enable(value: Boolean) {
        if (enabled != value) { reset(); enabled=value }
    }
    @Synchronized fun reset() { epoch++; latest=null; lastSample=Long.MIN_VALUE; completed=Long.MIN_VALUE }
    /**
     * 延迟回读（DEFERRED）画小图那一帧用：与 [begin] 相同的尺寸检查（变化时 reset）、启用、单调与限频判断，但不看是否忙、
     * 不取票、不动采样记录。到期返回当前代际，否则 null；读回那一帧再以同一 [now] 调 [begin]（带 expectEpoch）取票，
     * 单任务在途因此只覆盖推理本身，不再多占「等下一帧读回」的一帧。
     */
    @Synchronized fun due(now: Long, width: Int, height: Int, interval: Long=SAMPLE_INTERVAL_MS): Long? {
        if (dimensions != (width to height)) { reset(); dimensions=width to height }
        if (!enabled || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now-lastSample < interval)) return null
        return epoch
    }
    /**
     * [interval] 为相邻采样的最小间隔（ms）；画面对齐时每帧都采（见 VideoEffectsFilter 与 FrameDelay）。
     * [expectEpoch] 非 null 时代际不同就不取票（与取票同一把锁：[due] 之后切镜头、reset 的样本作废，不置 busy）。
     */
    @Synchronized fun begin(now: Long, width: Int, height: Int, interval: Long=SAMPLE_INTERVAL_MS, expectEpoch: Long?=null): Ticket? {
        if (dimensions != (width to height)) { reset(); dimensions=width to height }
        if (expectEpoch != null && expectEpoch != epoch) return null
        if (!enabled || busy || now <= lastSubmitted || (lastSample != Long.MIN_VALUE && now-lastSample < interval)) return null
        busy=true; lastSample=now; lastSubmitted=now
        return Ticket(epoch,now,width,height)
    }
    @Synchronized fun complete(ticket: Ticket, geometry: FaceGeometry?, now: Long, mesh: FaceMeshFrame?=null): Boolean {
        busy=false
        if (ticket.epoch != epoch || !enabled) return false
        completed=maxOf(completed,ticket.timestamp)
        latest=if (geometry != null && now-ticket.timestamp in 0..250) Triple(ticket, geometry, mesh) else null
        return true
    }
    @Synchronized fun current(now: Long): FaceGeometry? = fresh(now)?.second
    @Synchronized fun snapshot(now: Long): Snapshot = fresh(now).let {
        Snapshot(epoch, it?.first?.timestamp ?: Long.MIN_VALUE, it?.second, completed, it?.third, busy)
    }
    private fun fresh(now: Long) = latest?.takeIf { enabled && now-it.first.timestamp in 0..250 }

    companion object {
        /** 规格 v1 的 10 Hz。 */
        const val SAMPLE_INTERVAL_MS=100L
        /** 画面对齐、GPU 推理：每帧都采（单任务在途限制实际频率）。 */
        const val FRAME_INTERVAL_MS=1L
        /** 画面对齐、CPU 推理：约隔帧一采，留一个核给编码与渲染。 */
        const val CPU_INTERVAL_MS=45L
    }
}
