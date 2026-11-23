package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

class FaceTimelineTest {
    /** 41 维状态：脸中心在 (cx, cy)，其余部件随之平移。 */
    private fun state(cx: Float, cy: Float, axisX: Float = 1f, axisY: Float = 0f) = FloatArray(FaceGeometrySmoother.STATE_SIZE).also {
        for (i in 0 until 8) { it[i * 2] = cx + (i % 4 - 1.5f) * 0.06f; it[i * 2 + 1] = cy + (i / 4 - 0.5f) * 0.1f }
        it[16] = cx - 0.08f; it[17] = cy + 0.05f; it[18] = 0.06f
        it[19] = cx + 0.08f; it[20] = cy + 0.05f; it[21] = 0.06f
        it[22] = cx; it[23] = cy; it[24] = 0.22f; it[25] = 0.3f
        it[26] = axisX; it[27] = axisY
        it[28] = cx; it[29] = cy - 0.12f; it[30] = 0.07f; it[31] = 0.03f
        it[32] = cx - 0.08f; it[33] = cy + 0.05f; it[34] = 0.05f; it[35] = 0.025f
        it[36] = cx + 0.08f; it[37] = cy + 0.05f; it[38] = 0.05f; it[39] = 0.025f
        it[40] = 1f
    }

    /** 合成着色器 ellipseE：中心为纹理坐标，半径为度量值。 */
    private fun ellipseE(x: Float, y: Float, geometry: FloatArray, base: Int, aspect: Float): Float {
        val dx = x - geometry[base]; val dy = (y - geometry[base + 1]) / aspect
        val ux = geometry[26]; val uy = geometry[27]
        val a = (dx * ux + dy * uy) / geometry[base + 2]
        val b = (-dx * uy + dy * ux) / geometry[base + 3]
        return a * a + b * b
    }

    /**
     * 每 33 ms 一帧、每 [every] 帧一个样本（推理 [latency] ms 后才可用），画面延后 [delay] 帧；
     * 返回每帧的 (真实脸中心 x, 几何脸中心 x)。
     */
    private fun run(timeline: FaceTimeline, position: (Long) -> Float, frames: Int, latency: Long, delay: Int,
                    noise: (Int) -> Float = { 0f }, every: Int = 1): List<Pair<Float, Float>> {
        val result = mutableListOf<Pair<Float, Float>>()
        for (k in 0 until frames) {
            val now = k * 33L
            // 采样于第 j 帧、在 now 之前已完成的样本。
            for (j in 0..k step every) {
                val ts = j * 33L
                if (ts + latency <= now && ts + latency > now - 33) timeline.add(ts, state(position(ts) + noise(j), 0.5f))
            }
            timeline.update(now)
            val shown = maxOf(0, k - delay) * 33L
            val geometry = timeline.at(shown)
            if (geometry != null) result += position(shown) to geometry[22]
        }
        return result
    }

    @Test fun alignedPanHasNoVisibleLag() {
        val timeline = FaceTimeline()
        // 0.5 画面宽 / 秒横移；GPU 推理 15 ms，延后 1 帧即可对齐。
        val frames = run(timeline, { t -> 0.2f + 0.0005f * t }, 60, latency = 15, delay = 1)
        val worst = frames.drop(20).maxOf { (truth, drawn) -> abs(truth - drawn) }
        assertTrue("对齐后误差 $worst 应远小于 1 px（720p 下 0.0014）", worst < 0.0014f)
        assertEquals(0f, timeline.offsetX, 0f) // 不需要外推
        assertEquals(1f, timeline.steadyEye, 0f)
        assertEquals(0f, timeline.denoiseKeep, 0.01f) // 在动：暂停时域降噪
    }

    @Test fun stillFaceJitterIsSmoothedAndNotTreatedAsMotion() {
        val timeline = FaceTimeline()
        var seed = 7
        val noise = { _: Int -> seed = seed * 1103515245 + 12345; ((seed ushr 8) % 2001 - 1000) / 1000f * 0.003f }
        val frames = run(timeline, { 0.5f }, 150, latency = 15, delay = 1, noise = noise)
        val worst = frames.drop(60).maxOf { (truth, drawn) -> abs(truth - drawn) }
        assertTrue("静止抖动 ±0.003 经低通后 $worst 应明显变小", worst < 0.0015f)
        assertEquals(0f, timeline.speed, 0f)
        assertEquals(1f, timeline.denoiseKeep, 0f)
        assertEquals(1f, timeline.steadyEye, 0f); assertEquals(1f, timeline.steadyFace, 0f)
    }

    @Test fun tooLittleDelayExtrapolatesAndFadesEyeScaleEffects() {
        val timeline = FaceTimeline()
        // 1 画面宽 / 秒，10 Hz 采样、推理 120 ms 却不延后：只能外推（上限 1.25 × 100 ms 夹到 100 ms），外推量交给 dilate / steady。
        run(timeline, { t -> 0.1f + 0.001f * t }, 30, latency = 120, delay = 0, every = 3)
        assertEquals(100f, timeline.horizonMs, 0.5f)
        assertTrue(timeline.offsetX < -0.05f)
        assertEquals(0f, timeline.steadyEye, 0f)
        assertTrue(timeline.steadyFace < 1f)
    }

    /** 30 Hz 有脸样本到 [lastFace]，之后（[misses] 为 true 时）每 33 ms 一个无脸样本到 [until]。 */
    private fun faceThenMisses(timeline: FaceTimeline, lastFace: Long, until: Long, misses: Boolean = true) {
        for (ts in 0..lastFace step 33) timeline.add(ts, state(0.5f, 0.5f))
        if (misses) for (ts in lastFace + 33..until step 33) timeline.addMiss(ts)
    }

    @Test fun missingFaceExitsWithin80msOfFirstMiss() {
        val timeline = FaceTimeline()
        faceThenMisses(timeline, lastFace = 300, until = 600)
        timeline.update(600)
        // 第一个无脸样本在 333：防抖 max(33, 1.2 × 33) ≈ 40 ms 内满强度，之后 40 ms 渐出，80 ms 内退出。
        assertEquals(33f, timeline.sampleIntervalMs, 0.5f)
        timeline.at(300); assertEquals(1f, timeline.presence, 0f)
        timeline.at(333 + 33); assertEquals(1f, timeline.presence, 0f)
        timeline.at(333 + 60); assertTrue(timeline.presence in 0.1f..0.9f)
        timeline.at(333 + 80); assertEquals(0f, timeline.presence, 0f)
        // 丢脸之前的帧照常画（那时脸确实在），不会冻结。
        timeline.at(250); assertEquals(1f, timeline.presence, 0f)
        // 渲染时刻超过 500 ms 后清空历史。
        timeline.update(300 + FaceTimeline.CLEAR_AFTER_MS + 1)
        assertNull(timeline.at(800))
        assertEquals(0f, timeline.presence, 0f)
        assertEquals(Long.MIN_VALUE, timeline.latest)
    }

    @Test fun isolatedSingleMissDoesNotDipAt30Hz() {
        val timeline = FaceTimeline()
        for (ts in 0..300L step 33) timeline.add(ts, state(0.5f, 0.5f))
        timeline.addMiss(333)
        // 下一个结果还没回来（检测器忙）时也保持；回来后照常。
        for (t in 300L..366L) { timeline.at(t); assertEquals("t=$t", 1f, timeline.presence, 0f) }
        timeline.add(366, state(0.5f, 0.5f))
        for (ts in 399..600L step 33) timeline.add(ts, state(0.5f, 0.5f))
        for (t in 300L..600L) { timeline.at(t); assertEquals("t=$t", 1f, timeline.presence, 0f) }
    }

    @Test fun singleMissWithSkippedNextSampleDoesNotDipAt30Hz() {
        // 漏检那次推理超过一帧（单任务在途，下一帧不采）：330 漏检、363 不采、396 有脸，与第一个无脸样本相隔 66 ms，超过防抖（约 40 ms）。
        // 每帧先 add / addMiss 再按延后 1 或 2 帧的采样时间 at（同 VideoEffectsFilter）：那时还没有任何被画帧进入渐出，
        // 观众没看到过退出，不开新一轮、presence 恒为 1（以前从约 0.34 重新渐入，下陷约 100 ms）。
        for (delay in 1..2) {
            val timeline = FaceTimeline()
            var run = 0
            for (k in 0..20) {
                val ts = k * 33L
                when (k) {
                    10 -> timeline.addMiss(ts)
                    11 -> Unit
                    else -> timeline.add(ts, state(0.5f, 0.5f))
                }
                timeline.update(ts)
                val shown = ts - delay * 33L
                timeline.at(shown)
                if (k == 9) run = timeline.runId
                if (shown >= 297) assertEquals("delay=$delay t=$shown", 1f, timeline.presence, 0f)
            }
            assertEquals("delay=$delay 跳采漏检不应开新一轮", run, timeline.runId)
        }
    }

    @Test fun redetectionAfterDrawnFadeOutResumesWithoutJumpingBack() {
        // 已有被画帧进入渐出后才回来的有脸样本：照常开新一轮、从退出到的水平连续渐入，不跳回满强度（跳采豁免只给观众没看到的漏检）。
        val timeline = FaceTimeline()
        faceThenMisses(timeline, lastFace = 300, until = 333)
        timeline.at(333 + 60)
        val faded = timeline.presence
        assertTrue("已画到渐出：$faded", faded in 0.1f..0.9f)
        val run = timeline.runId
        timeline.add(399, state(0.5f, 0.5f))
        assertNotEquals("已画出渐出，重新检出开始新的一轮", run, timeline.runId)
        var previous = faded
        for (ts in 432L..800L step 33) {
            timeline.add(ts, state(0.5f, 0.5f))
            timeline.at(ts - 33)
            assertTrue("t=${ts - 33} presence ${timeline.presence} 单帧上跳不应超过 0.34（上一帧 $previous）",
                timeline.presence - previous <= 0.34f)
            previous = timeline.presence
        }
        assertEquals(1f, previous, 0f)
    }

    @Test fun singleMissAt15HzDoesNotFlicker() {
        // 负载降级 1 级（人脸 15 Hz）：防抖 = 1.2 × 66.7 ≈ 80 ms，单次漏检不下陷（已知代价：退出约 120 ms）。
        val timeline = FaceTimeline()
        for (k in 0..9) { val ts = k * 67L; if (k != 5) timeline.add(ts, state(0.5f, 0.5f)) else timeline.addMiss(ts) }
        assertEquals(67f, timeline.sampleIntervalMs, 2f)
        assertEquals(80f, timeline.missDebounceMs, 3f)
        for (t in 0L..603L) {
            timeline.at(t)
            if (t >= FaceTimeline.T_IN_MS.toLong()) assertEquals("t=$t", 1f, timeline.presence, 0f)
        }
        // 持续丢脸：第一个无脸样本后约 120 ms 退出。
        for (k in 10..15) timeline.addMiss(k * 67L)
        timeline.at(670 + 80); assertEquals(1f, timeline.presence, 0.01f)
        timeline.at(670 + 121); assertEquals(0f, timeline.presence, 0f)
    }

    @Test fun redetectionDoesNotDrawFutureMeshOnDelayedFrames() {
        // 有脸 0..297，无脸 330..561，594 重新检出（与上一个有脸样本相隔不超过 MAX_GAP_MS，旧样本保留）。
        val timeline = FaceTimeline()
        for (k in 0..9) timeline.add(k * 33L, state(0.5f, 0.5f))
        for (k in 10..17) timeline.addMiss(k * 33L)
        timeline.update(590)
        val run = timeline.runId
        timeline.add(594, state(0.8f, 0.5f))
        assertNotEquals("丢脸后重新检出开始新的一轮", run, timeline.runId)
        // 丢脸之前的旧帧仍按上一轮画。
        timeline.at(200); assertEquals(1f, timeline.presence, 0f)
        // 延后显示的帧（采样时间早于新一轮第一个样本）不画脸，不把 594 的网格画到更早的帧上。
        timeline.at(561); assertEquals(0f, timeline.presence, 0f)
        timeline.at(593); assertEquals(0f, timeline.presence, 0f)
        // 从新一轮第一个样本起按 150 ms 渐入。
        for (k in 19..27) timeline.add(k * 33L, state(0.8f, 0.5f))
        timeline.at(594); assertEquals(0f, timeline.presence, 0f)
        timeline.at(594 + 75); assertEquals(0.5f, timeline.presence, 0.01f)
        timeline.at(594 + 150); assertEquals(1f, timeline.presence, 0f)
        // 空档超过 MAX_GAP_MS 时旧样本丢弃，新一轮之前的帧一律不画。
        val long = FaceTimeline()
        faceThenMisses(long, lastFace = 300, until = 667)
        long.add(700, state(0.8f, 0.5f))
        long.at(200); assertEquals(0f, long.presence, 0f)
        long.at(699); assertEquals(0f, long.presence, 0f)
    }

    @Test fun shortOcclusionResumesFromCurrentLevelWithoutJump() {
        // 无脸 60 ms（越过防抖、渐出到一半）后重新检出：从退出到的水平接着渐入，presence 连续。
        // 先画一帧渐出（否则这段无脸没被观众看到，按跳采漏检豁免，见 singleMissWithSkippedNextSampleDoesNotDipAt30Hz）。
        val timeline = FaceTimeline()
        faceThenMisses(timeline, lastFace = 300, until = 366)
        timeline.at(333 + 60)
        timeline.add(399, state(0.5f, 0.5f))
        for (ts in 432L..800L step 33) timeline.add(ts, state(0.5f, 0.5f))
        var previous = 1f
        for (t in 300L..700L) {
            timeline.at(t)
            assertTrue("t=$t presence ${timeline.presence} 不应跳变（上一帧 $previous）", abs(timeline.presence - previous) < 0.05f)
            previous = timeline.presence
        }
        assertEquals(1f, previous, 0f)
    }

    @Test fun busyDetectorHoldsBoundedThenFades() {
        // 检测器忙：最新样本之后没有任何结果。保持 clamp(2 × 33 + 50, 100, 250) ≈ 116 ms，再 40 ms 渐出。
        val timeline = FaceTimeline()
        faceThenMisses(timeline, lastFace = 300, until = 0, misses = false)
        timeline.at(300 + 110); assertEquals(1f, timeline.presence, 0f)
        timeline.at(300 + 136); assertTrue(timeline.presence in 0.1f..0.9f)
        timeline.at(300 + 160); assertEquals(0f, timeline.presence, 0f)
        // 10 Hz 时保持上限 250 ms。
        val slow = FaceTimeline()
        for (ts in 0..1000L step 100) slow.add(ts, state(0.5f, 0.5f))
        slow.at(1000 + 245); assertEquals(1f, slow.presence, 0f)
        slow.at(1000 + 291); assertEquals(0f, slow.presence, 0f)
    }

    @Test fun missSinceMarksOnlyARunOfCompletedMissesAfterTheNewestFace() {
        // 头发遮罩的「人脸近期可见」据此区分「检测器报告丢脸」与「检测器还在算」（HairMaskTuning.faceKeep）。
        val timeline = FaceTimeline()
        assertEquals(Long.MIN_VALUE, timeline.missSince)
        timeline.addMiss(900) // 还没检出过脸
        assertEquals(900L, timeline.missSince)
        timeline.add(1000, state(0.5f, 0.5f))
        assertEquals("最近一次完成的检测有脸", Long.MIN_VALUE, timeline.missSince)
        // 检测器忙（最新样本之后没有任何结果）：不算丢脸，最新样本的年龄再大也一样。
        timeline.update(1000 + 400)
        assertEquals(1000L, timeline.latest)
        assertEquals(Long.MIN_VALUE, timeline.missSince)
        // 连续无脸：这一段的第一个样本；不递增的结果忽略。
        timeline.addMiss(1150); timeline.addMiss(1300)
        assertEquals(1150L, timeline.missSince)
        timeline.addMiss(1300); timeline.addMiss(1200)
        assertEquals(1150L, timeline.missSince)
        // 几何清空（最新样本过去 CLEAR_AFTER_MS）时事件一并清空。
        timeline.update(1000 + FaceTimeline.CLEAR_AFTER_MS + 1)
        assertEquals(Long.MIN_VALUE, timeline.latest)
        assertEquals(Long.MIN_VALUE, timeline.missSince)
        // 重新检出、再丢：从新的一段算起；硬重置清空。
        timeline.add(2000, state(0.5f, 0.5f)); timeline.addMiss(2400)
        assertEquals(2400L, timeline.missSince)
        timeline.add(2450, state(0.5f, 0.5f))
        assertEquals(Long.MIN_VALUE, timeline.missSince)
        timeline.addMiss(2500)
        timeline.reset()
        assertEquals(Long.MIN_VALUE, timeline.missSince)
    }

    @Test fun meshExtrapolationHorizonFollowsSampleInterval() {
        for ((interval, horizon) in listOf(33L to 41.25f, 66L to 82.5f, 100L to 100f)) {
            val timeline = FaceTimeline().apply { aspect = FaceTestFixtures.PORTRAIT }
            // 0.5 画面宽 / 秒平移（有网格）。
            for (ts in 0..1000L step interval) {
                val p = FaceTestFixtures.render(tx = -0.25 + 0.0005 * ts)
                timeline.add(ts, FaceTestFixtures.state(p), FaceTestFixtures.frame(p))
            }
            val newest = timeline.latest
            assertEquals(horizon, timeline.horizonMs, 1f)
            val base = timeline.meshAt(newest)!!.landmarks[1 * 2]
            val far = timeline.meshAt(newest + 300)!!.landmarks[1 * 2]
            // 只外推到上限 × 阻尼 0.85。
            val expected = 0.0005f * horizon * FaceTimeline.EXTRAPOLATE_DAMP
            assertEquals("间隔 $interval", expected, far - base, expected * 0.1f)
        }
    }

    @Test fun steadyUsesMeshExtrapolation() {
        // 有网格、推理慢且不延后：41 维几何的外推量取刚体姿态外推的脸中心位移，steady 据此渐出。
        val timeline = FaceTimeline().apply { aspect = FaceTestFixtures.PORTRAIT }
        for (ts in 0..600L step 100) {
            val p = FaceTestFixtures.render(tx = -0.3 + 0.001 * ts)
            timeline.add(ts, FaceTestFixtures.state(p), FaceTestFixtures.frame(p))
        }
        for (t in 600L..800L step 10) { timeline.update(t); timeline.at(t) }
        // 外推 100 ms × 0.85 × 1 画面宽 / 秒 ≈ 0.085；外推量远大于眼睛变形半径，眼睛尺度的形变渐出。
        assertEquals(-0.085f, timeline.offsetX, 0.01f)
        assertEquals(0f, timeline.steadyEye, 0f)
        assertTrue(timeline.steadyFace < 1f)
        val mesh = timeline.meshAt(800)!!
        assertEquals(0.085f, mesh.centerX - FaceTestFixtures.frame(FaceTestFixtures.render(tx = 0.3)).centerX, 0.01f)
        // 静止时不外推：offset 为 0。
        val still = FaceTimeline().apply { aspect = FaceTestFixtures.PORTRAIT }
        val p = FaceTestFixtures.render()
        for (ts in 0..600L step 100) still.add(ts, FaceTestFixtures.state(p), FaceTestFixtures.frame(p))
        still.update(700); still.at(700)
        assertEquals(0f, still.offsetX, 0f)
    }

    @Test fun stillMeshJitterRegressionThroughTimeline() {
        // 走 add + meshAt 的集成路径（30 Hz、σ = 0.5、延后 1 帧显示）：静止残留 ≤ 0.8 px@720。
        val timeline = FaceTimeline().apply { aspect = FaceTestFixtures.PORTRAIT }
        val truth = FaceTestFixtures.render()
        val rng = Random(3)
        val samples = ArrayList<Pair<FloatArray, FloatArray>>()
        for (k in 0 until 300) {
            val ts = k * 33L
            val p = FaceTestFixtures.noisy(truth, 0.5, rng)
            timeline.add(ts, FaceTestFixtures.state(p), FaceTestFixtures.frame(p))
            timeline.update(ts + 15)
            if (k > 45) samples += timeline.meshAt(ts - 33)!!.landmarks.copyOf() to truth
        }
        val rms = FaceTestFixtures.rmsPx720(samples, FaceTestFixtures.ALL)
        assertTrue("经时间线的静止残留 $rms ≤ 0.8 px@720", rms <= 0.8)
    }

    @Test fun secondFaceDoesNotChangeMainFaceEffects() {
        // numFaces = 1 时跟踪不被第二张脸抢走（需真机验证）；这里只喂主脸样本时 presence 一直满、runId 不变，
        // 一旦换成另一张脸（中心跳变 0.8 脸宽），开始新的一轮、从 0 渐入，旧脸的样本不再插值。
        val timeline = FaceTimeline().apply { aspect = FaceTestFixtures.PORTRAIT }
        val main = FaceTestFixtures.render(tx = -0.18)
        for (ts in 0..600L step 33) timeline.add(ts, FaceTestFixtures.state(main), FaceTestFixtures.frame(main))
        val run = timeline.runId
        for (t in 200L..600L step 10) { timeline.at(t); assertEquals(1f, timeline.presence, 0f) }
        assertEquals(run, timeline.runId)
        val other = FaceTestFixtures.render(tx = 0.18)
        timeline.add(633, FaceTestFixtures.state(other), FaceTestFixtures.frame(other))
        assertNotEquals(run, timeline.runId)
        timeline.at(633); assertEquals(0f, timeline.presence, 0f)
        timeline.at(600); assertEquals(0f, timeline.presence, 0f)
        timeline.at(633 + 75); assertEquals(0.5f, timeline.presence, 0.01f)
        assertEquals(other[1 * 2], timeline.meshAt(600)!!.landmarks[1 * 2], 1e-6f)
    }

    @Test fun gapRestartsWithoutInterpolatingAcrossIt() {
        val timeline = FaceTimeline()
        timeline.add(0, state(0.2f, 0.5f)); timeline.add(33, state(0.2f, 0.5f))
        timeline.add(1000, state(0.8f, 0.5f))
        timeline.update(1000)
        assertEquals(0.8f, timeline.at(1000)!![22], 1e-6f)
        assertEquals(0.8f, timeline.at(500)!![22], 1e-6f) // 空档前的旧样本已丢弃
        timeline.add(900, state(0.1f, 0.5f)) // 不递增：忽略
        assertEquals(1000L, timeline.latest)
    }

    @Test fun dilatedCoreCoversHeldAndExtrapolatedEllipses() {
        val aspect = 720f / 1280f
        val speeds = listOf(0.6f to 0f, 0f to 0.6f, 0.5f to -0.4f, 1.5f to 1.2f)
        val axes = listOf(1f to 0f, 0.96f to 0.28f)
        val ellipses = intArrayOf(22, 28, 32, 36)
        for ((vx, vy) in speeds) for ((ax, ay) in axes) {
            val timeline = FaceTimeline(); timeline.aspect = aspect
            // 匀速样本到 200 ms，然后在 300 ms 取几何：外推 100 ms。
            for (ts in 0..200L step 33) timeline.add(ts, state(0.3f + vx * ts / 1000f, 0.4f + vy * ts / 1000f * aspect, ax, ay))
            timeline.update(300)
            val drawn = timeline.at(300)!!.copyOf()
            val held = drawn.copyOf().also { g ->
                for (i in intArrayOf(22, 28, 32, 36)) { g[i] += timeline.offsetX; g[i + 1] += timeline.offsetY * aspect }
            }
            val out = timeline.dilate(drawn, FloatArray(FaceGeometrySmoother.STATE_SIZE))
            assertNotSame(drawn, out)
            for ((index, base) in ellipses.withIndex()) {
                val core = FaceTimeline.CORES[index]
                for (end in listOf(drawn, held)) for (j in 0 until 90) {
                    val angle = 2 * PI * j / 90
                    val k = kotlin.math.sqrt(core)
                    val a = (k * end[base + 2] * cos(angle)).toFloat(); val b = (k * end[base + 3] * sin(angle)).toFloat()
                    val x = end[base] + a * ax - b * ay
                    val y = end[base + 1] + (a * ay + b * ax) * aspect
                    assertTrue("v=($vx,$vy) axis=($ax,$ay) base=$base j=$j",
                        ellipseE(x, y, out, base, aspect) <= core + 1e-4f)
                }
            }
            for (i in (0 until 22) + (26 until 28)) assertEquals(drawn[i], out[i], 0f)
        }
    }

    /**
     * 头发遮罩的运动补偿要取遮罩时刻与显示时刻两个位姿，调用在 at() 前后都有：poseAt 不得改动 at() 的几何、
     * presence、外推量与 steady。两条时间线喂同样的样本，一条在每帧 at() 前后插入 poseAt，结果须逐值相同。
     */
    @Test fun poseAtLeavesAtResultsUntouched() {
        val aspect = 720f / 1280f
        val plain = FaceTimeline().also { it.aspect = aspect }
        val probed = FaceTimeline().also { it.aspect = aspect }
        val pose = FloatArray(FaceTimeline.POSE_SIZE)
        val dilatedA = FloatArray(FaceGeometrySmoother.STATE_SIZE)
        val dilatedB = FloatArray(FaceGeometrySmoother.STATE_SIZE)
        var extrapolated = 0
        // 1 画面宽 / 秒横移，10 Hz 采样、推理 120 ms、不延后：多数帧要外推，外推量、steady 与降噪都在变。
        for (k in 0 until 90) {
            val now = k * 33L
            for (j in 0..k step 3) {
                val ts = j * 33L
                if (ts + 120 <= now && ts + 120 > now - 33) {
                    val moving = state(0.1f + 0.001f * minOf(ts, 1500L), 0.5f)
                    plain.add(ts, moving); probed.add(ts, moving.copyOf())
                }
            }
            plain.update(now); probed.update(now)
            probed.poseAt(now - 200, pose); probed.poseAt(now, pose); probed.poseAt(probed.latest, pose)
            val a = plain.at(now)?.copyOf()
            val b = probed.at(now)?.copyOf()
            probed.poseAt(now - 120, pose)
            assertArrayEquals("k=$k", a, b, 0f)
            assertEquals("k=$k", plain.presence, probed.presence, 0f)
            assertEquals("k=$k", plain.offsetX, probed.offsetX, 0f)
            assertEquals("k=$k", plain.offsetY, probed.offsetY, 0f)
            assertEquals("k=$k", plain.steadyEye, probed.steadyEye, 0f)
            assertEquals("k=$k", plain.steadyFace, probed.steadyFace, 0f)
            assertEquals("k=$k", plain.denoiseKeep, probed.denoiseKeep, 0f)
            assertEquals("k=$k", plain.runId, probed.runId)
            if (a != null && b != null) assertArrayEquals("k=$k", plain.dilate(a, dilatedA), probed.dilate(b, dilatedB), 0f)
            if (plain.offsetX != 0f) extrapolated++
        }
        assertTrue("外推帧 $extrapolated", extrapolated > 20)
    }

    @Test fun poseAtInterpolatesInsideHistoryAndNeverExtrapolates() {
        val aspect = 720f / 1280f
        val timeline = FaceTimeline().also { it.aspect = aspect }
        val pose = FloatArray(FaceTimeline.POSE_SIZE)
        assertFalse("没有样本", timeline.poseAt(0, pose))
        // 中心、半径、轴向都在变，100 ms 一个样本。
        for (ts in 0..300L step 100) {
            val turn = ts / 1000f
            timeline.add(ts, state(0.3f + ts / 1000f, 0.45f, cos(turn), sin(turn)).also {
                it[24] = 0.20f + ts / 5000f; it[25] = 0.28f + ts / 5000f
            })
        }
        timeline.update(300)
        // 区间内与 at() 的插值逐分量一致（oval 与轴向就是合成用的 uOval、axis），轴向单位长。
        for (t in 0..300L step 10) {
            assertTrue("t=$t", timeline.poseAt(t, pose))
            val geometry = timeline.at(t)!!
            for (i in 0 until FaceTimeline.POSE_SIZE) assertEquals("t=$t i=$i", geometry[22 + i], pose[i], 1e-6f)
            assertEquals(1f, hypot(pose[4], pose[5]), 1e-6f)
        }
        // 线性：两个样本正中间等于两端平均（中心与半径）。
        val p100 = FloatArray(FaceTimeline.POSE_SIZE).also { assertTrue(timeline.poseAt(100, it)) }
        val p200 = FloatArray(FaceTimeline.POSE_SIZE).also { assertTrue(timeline.poseAt(200, it)) }
        assertTrue(timeline.poseAt(150, pose))
        for (i in 0 until 4) assertEquals("i=$i", (p100[i] + p200[i]) / 2, pose[i], 1e-6f)
        assertTrue(pose[0] > p100[0] && pose[0] < p200[0])
        // 区间外：返回 false，不写输出；at() 会外推的时刻 poseAt 也不外推。
        pose.fill(42f)
        assertFalse(timeline.poseAt(-1, pose))
        assertFalse(timeline.poseAt(301, pose))
        assertNotNull(timeline.at(340))
        assertFalse(timeline.poseAt(340, pose))
        for (v in pose) assertEquals(42f, v, 0f)
        // 历史清空后没有位姿。
        timeline.update(300 + FaceTimeline.CLEAR_AFTER_MS + 1)
        assertFalse(timeline.poseAt(300, pose))
    }

    @Test fun eyeMakeupKeepStaysFullWhenStillAndFadesFastOnPan() {
        val timeline = FaceTimeline()
        // 1) 静止 20 帧：eyeMakeupKeep 恒为 1，不影响正常静帧显色。
        for (k in 0 until 20) {
            val ts = k * 33L
            timeline.add(ts, state(0.5f, 0.5f))
            timeline.update(ts)
            timeline.at(maxOf(0L, ts - 33L))
        }
        assertEquals("静止时 eyeMakeupKeep 应保持 1", 1f, timeline.eyeMakeupKeep, 0.01f)
        // 2) 突然以 0.5 画面宽/秒横向快速移动（即使 FrameDelay 已对齐、steadyEye 仍为 1）：
        //    eyeMakeupKeep 必须在 2 帧内急速压低到 0.35 以下，防止 MediaPipe 内部 OneEuro 滞后拖出眼影残影。
        var panMin = 1f
        for (k in 20 until 35) {
            val ts = k * 33L
            val x = 0.5f + 0.0005f * (ts - 19 * 33L)
            timeline.add(ts, state(x, 0.5f))
            timeline.update(ts)
            timeline.at(ts - 33L)
            if (k >= 23) panMin = minOf(panMin, timeline.eyeMakeupKeep)
        }
        assertEquals("对齐横移时原 steadyEye 契约保持不变", 1f, timeline.steadyEye, 0f)
        assertTrue("快速横移时 eyeMakeupKeep 应迅速淡出至 < 0.30（实际 $panMin）", panMin < 0.30f)
        // 3) 停稳后 15 帧内柔和渐入回到接近 1。
        val stoppedX = 0.5f + 0.0005f * (34 - 19) * 33L
        for (k in 35 until 55) {
            val ts = k * 33L
            timeline.add(ts, state(stoppedX, 0.5f))
            timeline.update(ts)
            timeline.at(ts - 33L)
        }
        assertTrue("停稳后 eyeMakeupKeep 应平滑回升至 > 0.90（实际 ${timeline.eyeMakeupKeep}）",
            timeline.eyeMakeupKeep > 0.90f)
    }
}
