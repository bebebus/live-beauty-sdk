package io.bebebus.livebeauty

import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 人脸网格测试共用夹具（与设计仿真 facesim.py 同构）：
 * - 真值：MediaPipe 规范 3D 脸（[FaceMeshTopology.CANONICAL_XYZ]，另按眼心 ±0.21 眼宽生成 10 个虹膜点）加头部姿态、
 *   张嘴与眨眼，绕 (0, 0, −9) 旋转，相机距离 2.6 倍规范脸宽，真透视投影；默认脸宽占画面宽 0.45；
 *   眨眼（与仿真不同，仿真的眨眼只把睁眼比降到 0.69 倍）：上眼睑环落到同一横坐标的下眼睑上，上方眼皮按距离渐弱跟随，
 *   blink = 1 时睁眼比为 0；
 * - 噪声：在 FaceTextureSampler(384) 的小图上每个坐标独立高斯（σ 单位为采样像素）：9:16 为 216×384，4:3 为 288×384；
 * - 度量：每坐标 RMS，度量空间 (x, y / aspect) × 720，即 px@720（1 采样像素 = 3.33 px@720，9:16）。
 */
internal object FaceTestFixtures {
    const val PORTRAIT = 720f / 1280f
    const val PORTRAIT_4_3 = 720f / 960f
    const val LANDMARKS = FaceMeshFrame.LANDMARK_COUNT

    private val X0 = DoubleArray(LANDMARKS * 3)
    private val JAW = DoubleArray(LANDMARKS)
    /** blink = 1 时各点的竖直位移（规范单位，向下为负）。 */
    private val BLINK = DoubleArray(LANDMARKS)
    private val UPPER_LIDS = arrayOf(intArrayOf(33, 246, 161, 160, 159, 158, 157, 173, 133), intArrayOf(263, 466, 388, 387, 386, 385, 384, 398, 362))
    private val LOWER_LIDS = arrayOf(intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133), intArrayOf(263, 249, 390, 373, 374, 380, 381, 382, 362))
    private val FW_CANON: Double

    /** 嘴唇（120 个唇带三角形的顶点）、双眼眼裂环 32 点、脸廓 36 点、全部 468 点、两个虹膜中心。 */
    val LIPS: IntArray = FaceMeshTopology.LIP_BAND_INDICES.map { it.toInt() }.distinct().sorted().toIntArray()
    val EYES = intArrayOf(
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    )
    val CONTOUR: IntArray = FaceMeshTopology.FACE_OVAL
    val ALL = IntArray(FaceMeshFrame.FACE_COUNT) { it }
    val IRIS_CENTERS = intArrayOf(468, 473)

    init {
        val xyz = FaceMeshTopology.CANONICAL_XYZ
        for (i in 0 until 468 * 3) X0[i] = xyz[i].toDouble()
        var k = 468
        for (eye in arrayOf(intArrayOf(33, 133, 159, 145), intArrayOf(362, 263, 386, 374))) {
            val c = DoubleArray(3)
            for (i in eye) for (d in 0 until 3) c[d] += X0[i * 3 + d] / 4
            c[2] += 0.25
            val a = eye[0]; val b = eye[1]
            val r = 0.21 * sqrt((0 until 3).sumOf { (X0[a * 3 + it] - X0[b * 3 + it]).let { v -> v * v } })
            for ((dx, dy) in listOf(0.0 to 0.0, r to 0.0, 0.0 to r, -r to 0.0, 0.0 to -r)) {
                X0[k * 3] = c[0] + dx; X0[k * 3 + 1] = c[1] + dy; X0[k * 3 + 2] = c[2]; k++
            }
        }
        FW_CANON = sqrt((0 until 3).sumOf { (X0[454 * 3 + it] - X0[234 * 3 + it]).let { v -> v * v } })
        for (i in 0 until 468) {
            val x = X0[i * 3]; val y = X0[i * 3 + 1]
            // 下唇与下巴随张嘴下移。
            JAW[i] = ((-4.1 - y) / 1.2).coerceIn(0.0, 1.0) * (1 - ((abs(x) - 4.0) / 3.0).coerceIn(0.0, 1.0))
        }
        // 眨眼：眼角之间、上眼睑及以上的点下移「上睑 − 下睑」的距离，上眼睑环正好落到下眼睑上，上方 0.9 规范单位内渐弱。
        for (e in 0 until 2) {
            val upper = UPPER_LIDS[e]; val lower = LOWER_LIDS[e]
            val inner = setOf(*upper.toTypedArray())
            fun lidY(ring: IntArray, x: Double): Double {
                val pts = ring.map { X0[it * 3] to X0[it * 3 + 1] }.sortedBy { it.first }
                if (x <= pts.first().first) return pts.first().second
                for (k in 1 until pts.size) if (x <= pts[k].first) {
                    val (x0, y0) = pts[k - 1]; val (x1, y1) = pts[k]
                    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                }
                return pts.last().second
            }
            val xs = upper.map { X0[it * 3] }
            val xMin = xs.minOrNull()!!; val xMax = xs.maxOrNull()!!
            for (i in 0 until 468) {
                val x = X0[i * 3]; val y = X0[i * 3 + 1]
                if (x <= xMin || x >= xMax) continue
                val yUp = lidY(upper, x); val yLow = lidY(lower, x)
                val gap = yUp - yLow
                if (gap <= 0) continue
                val w = when {
                    i in inner -> 1.0
                    y > yUp -> (1 - (y - yUp) / 0.9).coerceIn(0.0, 1.0)
                    else -> 0.0
                }
                BLINK[i] = -gap * w
            }
        }
    }

    /**
     * 渲染 478 点 GL 纹理坐标（y 向上）。[yawRad]/[pitchRad]/[rollRad] 为头部转角，[tx]/[ty] 为平移（画面宽 / 高的比例），
     * [mouth]、[blink] 为 0..1 的张嘴与闭眼程度，[aspect] 为画面宽 / 高（宽固定为 720）。
     */
    fun render(
        yawRad: Double = 0.0,
        pitchRad: Double = 0.0,
        rollRad: Double = 0.0,
        tx: Double = 0.0,
        ty: Double = 0.0,
        mouth: Double = 0.0,
        blink: Double = 0.0,
        faceWidthFrac: Double = 0.45,
        aspect: Float = PORTRAIT,
    ): FloatArray {
        val w = 720.0; val h = w / aspect
        val d = 2.6 * FW_CANON
        val f = faceWidthFrac * w * (d + 2.44) / FW_CANON
        val cy = cos(yawRad); val sy = sin(yawRad)
        val cp = cos(pitchRad); val sp = sin(pitchRad)
        val cr = cos(rollRad); val sr = sin(rollRad)
        // R = Rz · Rx · Ry（与仿真一致），绕 (0, 0, −9) 旋转。
        val ry = arrayOf(doubleArrayOf(cy, 0.0, sy), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(-sy, 0.0, cy))
        val rx = arrayOf(doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, cp, -sp), doubleArrayOf(0.0, sp, cp))
        val rz = arrayOf(doubleArrayOf(cr, -sr, 0.0), doubleArrayOf(sr, cr, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
        val r = mul(rz, mul(rx, ry))
        val out = FloatArray(LANDMARKS * 2)
        for (i in 0 until LANDMARKS) {
            val px = X0[i * 3]
            val py = X0[i * 3 + 1] - mouth * 1.6 * JAW[i] + blink * BLINK[i]
            val pz = X0[i * 3 + 2] + 9.0
            val qx = r[0][0] * px + r[0][1] * py + r[0][2] * pz
            val qy = r[1][0] * px + r[1][1] * py + r[1][2] * pz
            val qz = r[2][0] * px + r[2][1] * py + r[2][2] * pz - 9.0
            val depth = d - qz
            val xp = w / 2 + tx * w + f * qx / depth
            val yp = h / 2 + ty * h + f * qy / depth
            out[i * 2] = (xp / w).toFloat()
            out[i * 2 + 1] = (yp / h).toFloat()
        }
        return out
    }

    private fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>) =
        Array(3) { i -> DoubleArray(3) { j -> (0 until 3).sumOf { a[i][it] * b[it][j] } } }

    /** 在 FaceTextureSampler(384) 小图上加每坐标独立高斯噪声（σ 为采样像素），返回新数组。 */
    fun noisy(points: FloatArray, sigmaSamplePx: Double, rng: Random, aspect: Float = PORTRAIT): FloatArray {
        val sampleH = 384.0; val sampleW = sampleH * aspect
        return FloatArray(points.size) { i ->
            val scale = if (i % 2 == 0) sigmaSamplePx / sampleW else sigmaSamplePx / sampleH
            (points[i] + rng.nextGaussian() * scale).toFloat()
        }
    }

    /** 每坐标 RMS 误差（度量空间 × 720 = px@720），只计 [idx] 里的点。 */
    fun rmsPx720(a: FloatArray, b: FloatArray, idx: IntArray, aspect: Float = PORTRAIT): Double {
        var sum = 0.0
        for (i in idx) {
            val dx = (a[i * 2] - b[i * 2]).toDouble() * 720
            val dy = (a[i * 2 + 1] - b[i * 2 + 1]).toDouble() / aspect * 720
            sum += dx * dx + dy * dy
        }
        return sqrt(sum / (2 * idx.size))
    }

    /** [idx] 里各点误差长度（px@720）的第 [q] 分位，[samples] 为每帧的 (输出, 真值)。 */
    fun percentilePx720(samples: List<Pair<FloatArray, FloatArray>>, idx: IntArray, q: Double, aspect: Float = PORTRAIT): Double {
        val all = ArrayList<Double>(samples.size * idx.size)
        for ((a, b) in samples) for (i in idx) {
            all += hypot((a[i * 2] - b[i * 2]).toDouble() * 720, (a[i * 2 + 1] - b[i * 2 + 1]).toDouble() / aspect * 720)
        }
        all.sort()
        return all[((all.size - 1) * q).toInt()]
    }

    /** 多帧的每坐标 RMS（px@720）。 */
    fun rmsPx720(samples: List<Pair<FloatArray, FloatArray>>, idx: IntArray, aspect: Float = PORTRAIT): Double {
        var sum = 0.0
        for ((a, b) in samples) sum += rmsPx720(a, b, idx, aspect).let { it * it }
        return sqrt(sum / samples.size)
    }

    /** MediaPipe 输出格式（图像归一化坐标，y 向下）。 */
    fun toMediaPipe(points: FloatArray): List<Pair<Float, Float>> =
        List(points.size / 2) { points[it * 2] to 1f - points[it * 2 + 1] }

    /** 由纹理坐标构造 FaceMeshFrame（经 fromMediaPipe，与生产一致）。 */
    fun frame(points: FloatArray, aspect: Float = PORTRAIT): FaceMeshFrame =
        checkNotNull(FaceMeshFrame.fromMediaPipe(toMediaPipe(points), aspect)) { "fixture_mesh_rejected" }

    /** 41 维状态（经 FaceGeometry.fromMesh，侧脸时用兼容几何）。 */
    fun state(points: FloatArray, aspect: Float = PORTRAIT): FloatArray {
        val pts = toMediaPipe(points)
        val geometry = FaceGeometry.fromMesh(pts, aspect) ?: frame(points, aspect).toCompatGeometry(aspect)
        return checkNotNull(geometry).toState()
    }
}
