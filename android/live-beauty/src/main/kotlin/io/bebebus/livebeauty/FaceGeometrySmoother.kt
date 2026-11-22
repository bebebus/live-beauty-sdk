package io.bebebus.livebeauty

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot

/**
 * 人脸几何时间平滑（规格 v1 第 3 节），逐行移植 reference/beauty_ref.py 的 FinalSmoother；纯逻辑，只在 GL 线程使用。
 * 状态向量 41 维：cheeks[0..15]、eyes[16..21]、oval[22..25]（rx 在 24）、axis[26..27]、mouth[28..31]、
 * eyeMasks[32..39]、warpWeight[40]（不参与偏差）。
 * - 自适应低通只在新样本（采样时间戳变化）到达时执行一次，te 为相邻样本时间戳间隔；
 * - 每帧在 prev → f 之间按实测样本间隔线性插值，消除 10 Hz 台阶；dt == 0 原样返回，可每帧调用任意次；
 * - presence：有目标时 150 ms 渐入；目标消失后先保持 150 ms，再 250 ms 渐出，到 0 清空几何。
 * 其他线程不得直接 reset，只能 reset 已加锁的 FaceTrackingState（代际 +1），由调用方比较代际后 reset。
 */
class FaceGeometrySmoother(aspect: Float) {
    /** 纹理 y 类分量的偏差按 ÷aspect 折算；只在 reset 前后修改。 */
    var aspect = aspect
    var presence = 0f
        private set
    private var f: FloatArray? = null       // 滤波后的样本
    private var prev: FloatArray? = null    // 上个样本到达时的输出
    private var x: FloatArray? = null       // 上次输出
    private var ts = 0L                     // 上个已消费样本的采样时间戳
    private var tArrive = 0L                // 该样本到达时的渲染时间
    private var interval = INTERP_INIT_MS
    private var missing = 0f
    private var last = Long.MIN_VALUE

    fun reset() {
        f = null; prev = null; x = null
        ts = 0L; tArrive = 0L; interval = INTERP_INIT_MS
        presence = 0f; missing = 0f; last = Long.MIN_VALUE
    }

    /**
     * @param target 41 维目标，null 表示本帧没有人脸；[sampleTimestampMs] 只在有目标时使用。
     * @return 当前输出（调用方只读，下次 update/reset 前有效），无几何时为 null；presence 见 [presence]。
     */
    fun update(target: FloatArray?, sampleTimestampMs: Long, nowMs: Long): FloatArray? {
        val dt: Float
        if (last == Long.MIN_VALUE) dt = 0f
        else {
            dt = (nowMs - last).coerceIn(0L, DT_CLAMP_MS).toFloat()
            if (dt == 0f) return x
        }
        last = nowMs
        if (target != null) {
            require(target.size == STATE_SIZE)
            missing = 0f
            val filtered = f
            if (filtered == null) {
                f = target.copyOf(); prev = target.copyOf(); x = target.copyOf()
                ts = sampleTimestampMs; tArrive = nowMs; interval = INTERP_INIT_MS
            } else if (sampleTimestampMs != ts) {
                val te = (sampleTimestampMs - ts).coerceIn(TE_MIN_MS, TE_MAX_MS).toFloat()
                x!!.copyInto(prev!!)
                val rx = maxOf(filtered[RX_INDEX], RX_MIN)
                var worst = 0f
                for (i in 0 until STATE_SIZE) {
                    val weight = when (KINDS[i]) { 'x' -> 1f; 'y' -> 1f / aspect; 'a' -> rx; else -> 0f }
                    worst = maxOf(worst, abs(target[i] - filtered[i]) * weight)
                }
                val s = worst / rx
                val fc = FC_MIN + K * maxOf(0f, s - DEAD)
                val alpha = (1.0 - exp(-2.0 * PI * fc * te / 1000.0)).toFloat()
                for (i in 0 until STATE_SIZE) filtered[i] += alpha * (target[i] - filtered[i])
                interval = (0.5f * interval + 0.5f * te).coerceIn(INTERP_MIN_MS, INTERP_MAX_MS)
                ts = sampleTimestampMs; tArrive = nowMs
            }
            presence = minOf(1f, presence + dt / T_IN_MS)
        } else {
            missing += dt
            if (missing > HOLD_MS) presence = maxOf(0f, presence - dt / T_OUT_MS)
            if (presence <= 0f) {
                f = null; prev = null; x = null
                ts = 0L; missing = 0f
            }
        }
        val filtered = f ?: return null
        val from = prev!!; val out = x!!
        val g = ((nowMs - tArrive + dt) / interval).coerceIn(0f, 1f)
        for (i in 0 until STATE_SIZE) out[i] = from[i] + g * (filtered[i] - from[i])
        val n = hypot(out[AXIS_INDEX], out[AXIS_INDEX + 1])
        if (n > 1e-6f) { out[AXIS_INDEX] /= n; out[AXIS_INDEX + 1] /= n }
        return out
    }

    companion object {
        const val STATE_SIZE = 41
        const val RX_INDEX = 24
        const val AXIS_INDEX = 26
        const val WARP_INDEX = 40
        /** 各分量的偏差类别：x ×1、y ÷aspect、a（axis）× oval.rx、n 不参与。 */
        private const val KINDS = "xyxyxyxyxyxyxyxy" + "xyxxyx" + "xyxx" + "aa" + "xyxx" + "xyxxxyxx" + "n"
        private const val FC_MIN = 0.6f
        private const val K = 60f
        private const val DEAD = 0.012f
        private const val T_IN_MS = 150f
        private const val HOLD_MS = 150f
        private const val T_OUT_MS = 250f
        private const val INTERP_MIN_MS = 66f
        private const val INTERP_MAX_MS = 150f
        private const val INTERP_INIT_MS = 100f
        private const val DT_CLAMP_MS = 100L
        private const val TE_MIN_MS = 33L
        private const val TE_MAX_MS = 500L
        private const val RX_MIN = 1e-3f
    }
}
