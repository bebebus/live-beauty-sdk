package io.bebebus.livebeauty

import io.bebebus.livebeauty.FaceTestFixtures.ALL
import io.bebebus.livebeauty.FaceTestFixtures.CONTOUR
import io.bebebus.livebeauty.FaceTestFixtures.EYES
import io.bebebus.livebeauty.FaceTestFixtures.LIPS
import io.bebebus.livebeauty.FaceTestFixtures.PORTRAIT
import io.bebebus.livebeauty.FaceTestFixtures.PORTRAIT_4_3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * 网格时间平滑（FaceMeshFilter）的验收：阈值取自设计仿真（facesim9 / run11，σ = 0.5 采样像素）并留出余量，
 * 单位一律为 px@720（每坐标 RMS）。
 */
class FaceMeshFilterTest {
    private class Pose(
        val yaw: Double = 0.0, val pitch: Double = 0.0, val roll: Double = 0.0,
        val tx: Double = 0.0, val ty: Double = 0.0, val mouth: Double = 0.0, val blink: Double = 0.0,
    )

    /** 按 [fps] / [every] 采样喂给滤波器，返回预热 [warmSec] 之后每个样本的 (输出, 真值)。 */
    private fun run(
        scenario: (Double) -> Pose,
        sigma: Double,
        seconds: Double = 10.0,
        every: Int = 1,
        warmSec: Double = 1.5,
        aspect: Float = PORTRAIT,
        seed: Long = 3,
        filter: FaceMeshFilter = FaceMeshFilter(),
        onSample: (FaceMeshFilter) -> Unit = {},
    ): List<Pair<FloatArray, FloatArray>> {
        val rng = Random(seed)
        val out = ArrayList<Pair<FloatArray, FloatArray>>()
        val frames = (seconds * 30).toInt()
        val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
        for (k in 0 until frames step every) {
            val t = k / 30.0
            val p = scenario(t)
            val truth = FaceTestFixtures.render(p.yaw, p.pitch, p.roll, p.tx, p.ty, p.mouth, p.blink, aspect = aspect)
            val input = if (sigma > 0) FaceTestFixtures.noisy(truth, sigma, rng, aspect) else truth
            val result = FloatArray(FaceMeshFrame.COORD_COUNT)
            filter.filter(Math.round(k * 1000.0 / 30.0), input, aspect, result, pose)
            onSample(filter)
            if (t >= warmSec) out += result to truth
        }
        return out
    }

    private val still = { _: Double -> Pose() }

    /** ±20° 三角波，30°/s。 */
    private val yaw30 = { t: Double ->
        val period = 80.0 / 30.0
        val ph = (t % period) / period
        Pose(yaw = Math.toRadians(-20 + 40 * (if (ph < 0.5) 2 * ph else 2 - 2 * ph)))
    }

    /** 0.5 画面宽 / 秒的三角波平移（±0.15）。 */
    private val pan = { t: Double ->
        val period = 1.2
        val ph = (t % period) / period
        Pose(tx = -0.15 + 0.3 * (if (ph < 0.5) 2 * ph else 2 - 2 * ph))
    }

    /** 3.5 Hz 说话（张嘴幅度 0.5，0.7 Hz 起伏）。 */
    private val talk = { t: Double ->
        Pose(mouth = 0.5 * (1 - cos(2 * PI * 3.5 * t)) / 2 * (1 + 0.5 * sin(2 * PI * 0.7 * t)))
    }

    @Test fun poseFitRecoversWeakPerspectiveExactly() {
        val xyz = FaceMeshTopology.CANONICAL_XYZ
        val fw = FaceMeshFilter.CANONICAL_FACE_WIDTH
        val rng = Random(11)
        val mx = FloatArray(FaceMeshFrame.FACE_COUNT)
        val my = FloatArray(FaceMeshFrame.FACE_COUNT)
        val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
        repeat(20) { trial ->
            // 随机旋转（偏航 / 俯仰 / 翻滚各 ±35°）、尺度与平移的正交投影 m = t + s·(R·X / fw)[0..1]。
            val yaw = (rng.nextDouble() - 0.5) * 1.2; val pitch = (rng.nextDouble() - 0.5) * 1.2; val roll = (rng.nextDouble() - 0.5) * 1.2
            val s = 0.2 + 0.4 * rng.nextDouble()
            val tx = rng.nextDouble(); val ty = rng.nextDouble() * 1.7
            val r = rotation(yaw, pitch, roll)
            val mirror = if (trial % 2 == 0) 1.0 else -1.0
            for (i in 0 until FaceMeshFrame.FACE_COUNT) {
                val x = xyz[i * 3] / fw; val y = xyz[i * 3 + 1] / fw; val z = xyz[i * 3 + 2] / fw
                mx[i] = (tx + mirror * s * (r[0][0] * x + r[0][1] * y + r[0][2] * z)).toFloat()
                my[i] = (ty + s * (r[1][0] * x + r[1][1] * y + r[1][2] * z)).toFloat()
            }
            FaceMeshFilter.fitPose(mx, my, pose)
            val expected = doubleArrayOf(mirror * s * r[0][0], mirror * s * r[0][1], s * r[1][0], s * r[1][1], mirror * s * r[0][2], s * r[1][2])
            val got = floatArrayOf(pose[2], pose[3], pose[4], pose[5], pose[6], pose[7])
            for (k in expected.indices) assertEquals("trial $trial k=$k", expected[k], got[k].toDouble(), 1e-4)
            // 投影尺度与转角无关；镜像点集 det A < 0。
            assertEquals(s * s, FaceMeshFilter.viewScale2(pose).toDouble(), 1e-4)
            assertEquals(mirror < 0, FaceMeshFilter.detOf(pose) < 0f)
        }
        // 静止无噪声：两层滤波都是恒等（输出逐点等于输入）。
        val truth = FaceTestFixtures.render(yawRad = 0.3, pitchRad = 0.1)
        val filter = FaceMeshFilter()
        val out = FloatArray(FaceMeshFrame.COORD_COUNT)
        for (k in 0 until 10) filter.filter(k * 33L, truth, PORTRAIT, out, pose)
        for (i in 0 until FaceMeshFrame.FACE_COUNT * 2) assertEquals(truth[i], out[i], 1e-5f)
    }

    @Test fun stillMeshJitterBelow0p8PxAt720p_30Hz() {
        val samples = run(still, sigma = 0.5)
        val all = FaceTestFixtures.rmsPx720(samples, ALL)
        val lips = FaceTestFixtures.rmsPx720(samples, LIPS)
        val eyes = FaceTestFixtures.rmsPx720(samples, EYES)
        // 仿真约 0.63 / 0.70 / 0.70；原始输入约 1.67。
        assertTrue("静止残留 all=$all lips=$lips eyes=$eyes 应 ≤ 0.8 px@720", all <= 0.8 && lips <= 0.8 && eyes <= 0.8)
    }

    @Test fun stillMeshJitterAt15HzAnd10Hz() {
        val at15 = FaceTestFixtures.rmsPx720(run(still, sigma = 0.5, every = 2), ALL)
        val at10 = FaceTestFixtures.rmsPx720(run(still, sigma = 0.5, every = 3), ALL)
        val noise = FaceTestFixtures.rmsPx720(noisyStill(), ALL)
        assertTrue("15 Hz 静止残留 $at15 应 ≤ 0.85", at15 <= 0.85)
        assertTrue("10 Hz 静止残留 $at10 应 ≤ 1.0", at10 <= 1.0)
        assertTrue("相对原始输入 $noise 至少降低 40%：15 Hz $at15、10 Hz $at10", at15 <= 0.6 * noise && at10 <= 0.6 * noise)
    }

    private fun noisyStill(): List<Pair<FloatArray, FloatArray>> {
        val rng = Random(3)
        val truth = FaceTestFixtures.render()
        return List(100) { FaceTestFixtures.noisy(truth, 0.5, rng) to truth }
    }

    @Test fun stillMeshJitterOn4by3Sampler() {
        // 4:3 机型保留传感器比例（720×960 推流，小图 288×384）：同一口径下静止残留同样 ≤ 0.8 px@720。
        val samples = run(still, sigma = 0.5, aspect = PORTRAIT_4_3)
        val all = FaceTestFixtures.rmsPx720(samples, ALL, PORTRAIT_4_3)
        val lips = FaceTestFixtures.rmsPx720(samples, LIPS, PORTRAIT_4_3)
        val eyes = FaceTestFixtures.rmsPx720(samples, EYES, PORTRAIT_4_3)
        assertTrue("4:3 静止残留 all=$all lips=$lips eyes=$eyes 应 ≤ 0.8", all <= 0.8 && lips <= 0.8 && eyes <= 0.8)
    }

    @Test fun yaw30DegPerSecondLagBounded() {
        val samples = run(yaw30, sigma = 0.0)
        val all = FaceTestFixtures.rmsPx720(samples, ALL)
        val lips = FaceTestFixtures.percentilePx720(samples, LIPS, 0.95)
        val eyes = FaceTestFixtures.percentilePx720(samples, EYES, 0.95)
        val contour = FaceTestFixtures.percentilePx720(samples, CONTOUR, 0.95)
        // 仿真：RMS 0.29、p95 0.83，轮廓 p95 2.0。
        assertTrue("30°/s 转头滞后 RMS $all ≤ 0.5", all <= 0.5)
        assertTrue("唇 p95 $lips、眼 p95 $eyes ≤ 1.0", lips <= 1.0 && eyes <= 1.0)
        assertTrue("轮廓 p95 $contour ≤ 2.5", contour <= 2.5)
    }

    @Test fun panHasNoLag() {
        val all = FaceTestFixtures.rmsPx720(run(pan, sigma = 0.0), ALL)
        assertTrue("0.5 画面宽 / 秒平移滞后 $all ≤ 0.2", all <= 0.2)
    }

    @Test fun talkingMouthFollows() {
        val lips = FaceTestFixtures.percentilePx720(run(talk, sigma = 0.0), LIPS, 0.95)
        assertTrue("3.5 Hz 说话唇 p95 $lips ≤ 2.5", lips <= 2.5)
    }

    @Test fun noiseFloorUnaffectedByRigidMotion() {
        var stillFloor = 0.0; var n = 0
        run(still, sigma = 0.5, onSample = { if (it.noiseFloor > 0f) { stillFloor += it.noiseFloor; n++ } })
        stillFloor /= n
        var panFloor = 0.0; var m = 0
        run(pan, sigma = 0.5, onSample = { if (it.noiseFloor > 0f) { panFloor += it.noiseFloor; m++ } })
        panFloor /= m
        assertTrue("平移时噪声底 $panFloor 不应超过静止 $stillFloor 的 1.5 倍", panFloor <= 1.5 * stillFloor)
    }

    @Test fun irisStableInEyeFrameAndSaccadeTracked() {
        val filter = FaceMeshFilter()
        val samples = run(still, sigma = 0.5, filter = filter)
        // 圆心相对眼角中点的抖动、半径抖动（px@720）。
        var centerSq = 0.0; var radii = ArrayList<Double>()
        for ((out, truth) in samples) {
            for ((base, a, b) in listOf(Triple(468, 33, 133), Triple(473, 362, 263))) {
                fun rel(p: FloatArray, axis: Int) = (p[base * 2 + axis] - 0.5f * (p[a * 2 + axis] + p[b * 2 + axis])).toDouble() /
                    (if (axis == 0) 1.0 else PORTRAIT.toDouble()) * 720
                val dx = rel(out, 0) - rel(truth, 0); val dy = rel(out, 1) - rel(truth, 1)
                centerSq += dx * dx + dy * dy
                radii += hypot((out[(base + 1) * 2] - out[base * 2]).toDouble(), (out[(base + 1) * 2 + 1] - out[base * 2 + 1]).toDouble() / PORTRAIT) * 720
            }
        }
        val center = kotlin.math.sqrt(centerSq / (2 * samples.size * 2))
        val mean = radii.average()
        val radiusJitter = kotlin.math.sqrt(radii.sumOf { (it - mean) * (it - mean) } / radii.size)
        assertTrue("虹膜圆心相对眼眶抖动 $center ≤ 0.4 px@720", center <= 0.4)
        assertTrue("虹膜半径抖动 $radiusJitter ≤ 0.2 px@720", radiusJitter <= 0.2)
        // 半径比例落在 [0.16, 0.28] 眼宽。
        val out = samples.last().first
        val ew = hypot((out[133 * 2] - out[33 * 2]).toDouble(), (out[133 * 2 + 1] - out[33 * 2 + 1]).toDouble() / PORTRAIT)
        val r = hypot((out[469 * 2] - out[468 * 2]).toDouble(), (out[469 * 2 + 1] - out[468 * 2 + 1]).toDouble() / PORTRAIT)
        assertTrue("半径比例 ${r / ew}", r / ew in 0.159..0.281)

        // 扫视：虹膜沿眼轴突然移动 0.15 眼宽，一个样本内跟上 ≥ 80%（无噪声）。
        val saccade = FaceMeshFilter()
        val base = FaceTestFixtures.render()
        val res = FloatArray(FaceMeshFrame.COORD_COUNT)
        val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
        for (k in 0 until 30) saccade.filter(k * 33L, base, PORTRAIT, res, pose)
        val before = res[468 * 2]
        val shifted = base.copyOf()
        val ewTex = base[133 * 2] - base[33 * 2]
        for (i in 468 until 478) shifted[i * 2] += 0.15f * ewTex
        saccade.filter(30 * 33L, shifted, PORTRAIT, res, pose)
        val followed = (res[468 * 2] - before) / (0.15f * ewTex)
        assertTrue("扫视一个样本内跟上 $followed ≥ 0.8", followed >= 0.8f)
    }

    /** 让睁眼比降到正常值 [ratio] 倍的眨眼程度（无噪声渲染上二分）。 */
    private fun blinkFor(ratio: Double): Double {
        fun open(b: Double): Double {
            val p = FaceTestFixtures.render(blink = b)
            fun d(i: Int, j: Int) = hypot((p[i * 2] - p[j * 2]).toDouble(), (p[i * 2 + 1] - p[j * 2 + 1]).toDouble() / PORTRAIT)
            return d(159, 145) / d(33, 133)
        }
        val target = open(0.0) * ratio
        var lo = 0.0; var hi = 1.0
        repeat(40) { val mid = (lo + hi) / 2; if (open(mid) > target) lo = mid else hi = mid }
        return (lo + hi) / 2
    }

    @Test fun blinkGateHysteresisNoFlicker() {
        // 先正常睁眼 3 秒学到基线，再像低头一样睁眼比降到 0.6 倍并保持；σ = 1.0 下门控 0 次翻转、一直开着。
        val narrow = blinkFor(0.6)
        var flips = 0; var last = -1f; var minGate = 1f
        var k = 0
        run({ t -> Pose(blink = if (t < 3) 0.0 else narrow) }, sigma = 1.0, seconds = 23.0, warmSec = 0.0, onSample = { f ->
            if (k++ >= 90) {
                if (last >= 0f && (f.rightGate < 0.5f) != (last < 0.5f)) flips++
                last = f.rightGate
                minGate = minOf(minGate, f.rightGate, f.leftGate)
            }
        })
        assertEquals("门控翻转次数", 0, flips)
        assertEquals(1f, minGate, 0f)
    }

    @Test fun blinkGateClosesOnFullClosureAndRampsOpen() {
        // 眨眼：1.0 s 起 70 ms 闭合、保持 60 ms、120 ms 睁开（30 Hz，σ = 0.5）。
        fun blink(t: Double): Double {
            val p = t - 1.0
            return when {
                p < 0 -> 0.0
                p < 0.07 -> p / 0.07
                p < 0.13 -> 1.0
                p < 0.25 -> 1 - (p - 0.13) / 0.12
                else -> 0.0
            }
        }
        val gates = ArrayList<Pair<Double, Float>>()
        var k = 0
        run({ t -> Pose(blink = blink(t)) }, sigma = 0.5, seconds = 2.0, warmSec = 0.0, onSample = { f ->
            gates += (k++ / 30.0) to minOf(f.rightGate, f.leftGate)
        })
        val closedAt = gates.first { it.first >= 1.07 - 1e-9 }
        assertEquals("眼睑完全闭合那个样本门控应为 0（t=${closedAt.first}）", 0f, closedAt.second, 0f)
        assertTrue("闭眼前一直开着", gates.filter { it.first < 1.0 }.all { it.second == 1f })
        val reopened = gates.firstOrNull { it.first > 1.25 && it.second >= 1f }
        assertTrue("完全睁开后约 120 ms 回到 1：${reopened?.first}", reopened != null && reopened.first - 1.25 <= 0.25)
    }

    @Test fun identityJumpStartsNewRun() {
        val filter = FaceMeshFilter()
        val res = FloatArray(FaceMeshFrame.COORD_COUNT)
        val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
        val a = FaceTestFixtures.render()
        assertEquals(FaceMeshFilter.NEW_RUN, filter.filter(0, a, PORTRAIT, res, pose))
        for (k in 1 until 10) assertEquals(FaceMeshFilter.CONTINUE, filter.filter(k * 33L, a, PORTRAIT, res, pose))
        assertFalse(filter.jumped)
        // 中心跳变 0.8 倍脸宽（0.45 × 0.8 画面宽）：另一张脸。
        val b = FaceTestFixtures.render(tx = 0.36)
        assertEquals(FaceMeshFilter.NEW_RUN, filter.filter(330, b, PORTRAIT, res, pose))
        assertTrue(filter.jumped)
        for (i in 0 until FaceMeshFrame.FACE_COUNT * 2) assertEquals(b[i], res[i], 1e-6f)
        // 尺度突变（远处另一张小脸）同样重新开始；超过 MAX_GAP_MS 的间隔也是。
        val c = FaceTestFixtures.render(tx = 0.36, faceWidthFrac = 0.28)
        assertEquals(FaceMeshFilter.NEW_RUN, filter.filter(363, c, PORTRAIT, res, pose))
        assertEquals(FaceMeshFilter.NEW_RUN, filter.filter(363 + FaceMeshFilter.MAX_GAP_MS + 1, c, PORTRAIT, res, pose))
        assertFalse(filter.jumped)
    }

    @Test fun yawWeightsDoNotFlickerWhenStill() {
        // 静止、偏航比约 0.15（远侧半脸权重正处在渐出斜坡上）：由滤波后网格重算的左右权重逐帧跳动有上限。
        val yaw = 0.14
        val meta = FloatArray(FaceMeshFrame.META_SIZE)
        assertTrue(FaceMeshFrame.fillMeta(FaceTestFixtures.render(yawRad = yaw), PORTRAIT, meta))
        val signedYaw = meta[FaceMeshFrame.META_SIGNED_YAW]
        assertTrue("夹具偏航比 $signedYaw 应落在 0.12..0.44 斜坡上", abs(signedYaw) in 0.12f..0.30f)
        val filtered = ArrayList<Float>(); val raw = ArrayList<Float>()
        val rng = Random(5)
        val truth = FaceTestFixtures.render(yawRad = yaw)
        val filter = FaceMeshFilter()
        val res = FloatArray(FaceMeshFrame.COORD_COUNT)
        val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
        for (k in 0 until 300) {
            val input = FaceTestFixtures.noisy(truth, 0.5, rng)
            filter.filter(k * 33L, input, PORTRAIT, res, pose)
            if (k < 45) continue
            assertTrue(FaceMeshFrame.fillMeta(res, PORTRAIT, meta))
            filtered += minOf(meta[FaceMeshFrame.META_RIGHT_WEIGHT], meta[FaceMeshFrame.META_LEFT_WEIGHT])
            assertTrue(FaceMeshFrame.fillMeta(input, PORTRAIT, meta))
            raw += minOf(meta[FaceMeshFrame.META_RIGHT_WEIGHT], meta[FaceMeshFrame.META_LEFT_WEIGHT])
        }
        fun maxStep(v: List<Float>) = v.zipWithNext { a, b -> abs(a - b) }.maxOrNull() ?: 0f
        val f = maxStep(filtered); val r = maxStep(raw)
        assertTrue("滤波后远侧权重逐帧最大跳动 $f 应 ≤ 0.02，且不超过原始 $r 的 1/3", f <= 0.02f && f <= r / 3)
    }

    private fun rotation(yaw: Double, pitch: Double, roll: Double): Array<DoubleArray> {
        val cy = cos(yaw); val sy = sin(yaw); val cp = cos(pitch); val sp = sin(pitch); val cr = cos(roll); val sr = sin(roll)
        val ry = arrayOf(doubleArrayOf(cy, 0.0, sy), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(-sy, 0.0, cy))
        val rx = arrayOf(doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, cp, -sp), doubleArrayOf(0.0, sp, cp))
        val rz = arrayOf(doubleArrayOf(cr, -sr, 0.0), doubleArrayOf(sr, cr, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
        fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>) =
            Array(3) { i -> DoubleArray(3) { j -> (0 until 3).sumOf { a[i][it] * b[it][j] } } }
        return mul(rz, mul(rx, ry))
    }
}
