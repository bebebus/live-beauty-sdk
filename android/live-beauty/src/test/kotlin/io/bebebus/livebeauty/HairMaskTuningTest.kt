package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.cos
import kotlin.math.roundToLong
import kotlin.math.sin

/** 头发遮罩的调度、健康、新鲜度、运动补偿、强度、启用与时域 EMA（docs/plans/2026-11-21-live-beauty-hair-edge-plan.md H13–H17、H20）。 */
class HairMaskTuningTest {
    @Test fun intervalAdaptsToInferenceTimeAndBackgroundBlur() {
        assertEquals(100L, HairMaskTuning.interval(Float.NaN, false)) // 样本不足：基础 10 Hz
        assertEquals(100L, HairMaskTuning.interval(50f, false))
        assertEquals(116L, HairMaskTuning.interval(58f, false)) // Pixel 6 CPU 量级：2 × p50
        assertEquals(101L, HairMaskTuning.interval(50.2f, false)) // 向上取整，占空不超过 1/2
        assertEquals(240L, HairMaskTuning.interval(120f, false))
        assertEquals(250L, HairMaskTuning.interval(125f, false))
        // 开着背景虚化：下限 200 ms，给人像分割让出 CPU。
        for (p50 in listOf(Float.NaN, 0f, 58f, 100f)) assertEquals("p50=$p50", 200L, HairMaskTuning.interval(p50, true))
        assertEquals(240L, HairMaskTuning.interval(120f, true))
        assertEquals(250L, HairMaskTuning.interval(125f, true))
        for (p50 in 0..400) {
            assertTrue(HairMaskTuning.interval(p50.toFloat(), true) >= 200L)
            assertTrue(HairMaskTuning.interval(p50.toFloat(), false) >= 100L)
        }
    }

    @Test fun workerDutyStaysAtMostHalfForEveryInferenceTime() {
        // worker 墙钟占空 = p50 / 间隔：机型越慢频率越低，任何 p50 下都不超过 1/2（旧系数 1.3 时 p50≥77 ms 恒为约 77%）。
        for (background in listOf(false, true)) {
            var p50 = 0.1f
            while (p50 <= 400f) {
                val duty = p50 / HairMaskTuning.interval(p50, background)
                assertTrue("p50=$p50 背景=$background 占空 $duty", duty <= 0.5f)
                p50 += 0.1f
            }
        }
        // 对照：旧系数 1.3 在 p50 = 120 ms 时占空约 77%。
        assertTrue(120f / maxOf(100L, (120f * 1.3f).roundToLong()) > 0.75f)
    }

    @Test fun healthNeedsEightSamplesThenComparesMedianWith125ms() {
        // 样本少于 8 个不判（NaN 算健康），哪怕都很慢。
        val slow = FloatArray(HairMaskTuning.P50_WINDOW) { 900f }
        for (n in 0 until HairMaskTuning.MIN_SAMPLES) {
            assertTrue(HairMaskTuning.p50(slow, n).isNaN())
            assertTrue(HairMaskTuning.healthy(HairMaskTuning.p50(slow, n)))
        }
        assertFalse(HairMaskTuning.healthy(HairMaskTuning.p50(slow, HairMaskTuning.MIN_SAMPLES)))
        assertFalse(HairMaskTuning.healthy(125.5f))
        assertFalse(HairMaskTuning.healthy(160f)) // 旧上限 160 不再健康
        assertTrue(HairMaskTuning.healthy(125f))
        assertTrue(HairMaskTuning.healthy(Float.NaN))
        // 中位数：少数几次离群（首帧预热、GC）不改变判定；顺序无关。
        val mixed = floatArrayOf(60f, 2000f, 58f, 61f, 1500f, 59f, 62f, 57f, 63f, 60f)
        assertEquals(61f, HairMaskTuning.p50(mixed, 10), 0f)
        assertTrue(HairMaskTuning.healthy(HairMaskTuning.p50(mixed, 10)))
        val borderline = FloatArray(10) { if (it < 5) 120f else 126f }
        assertEquals(126f, HairMaskTuning.p50(borderline, 10), 0f)
        assertFalse(HairMaskTuning.healthy(HairMaskTuning.p50(borderline, 10)))
    }

    @Test fun freshnessIsFullUntil420msAndZeroAt650ms() {
        assertEquals(1f, HairMaskTuning.fresh(0f), 0f)
        assertEquals(1f, HairMaskTuning.fresh(420f), 0f)
        assertEquals(0f, HairMaskTuning.fresh(650f), 0f)
        assertEquals(0f, HairMaskTuning.fresh(900f), 0f)
        assertEquals(1f, HairMaskTuning.fresh(-300f), 0f) // 按 |lag|
        assertEquals(0.5f, HairMaskTuning.fresh(535f), 1e-6f)
        var previous = 1f
        for (lag in 420..650) {
            val f = HairMaskTuning.fresh(lag.toFloat())
            assertTrue("lag=$lag", f <= previous && f in 0f..1f)
            previous = f
        }
        assertEquals(HairMaskTuning.FRESH_MS.toFloat(), 650f, 0f)
    }

    /**
     * 防锯齿不变量（H15）：健康上限 [maxP50] 内最坏 lag = p50 + 间隔 + 1 帧，新鲜度在这里必须仍为 1。
     * 默认用产品的上限、间隔与新鲜度，参数只供对照旧取值。
     */
    private fun sawtoothFree(
        maxP50: Float = HairMaskTuning.MAX_P50_MS,
        interval: (Float, Boolean) -> Long = HairMaskTuning::interval,
        fresh: (Float) -> Float = { HairMaskTuning.fresh(it) },
    ): Boolean {
        for (background in listOf(false, true)) {
            var p50 = 0f
            while (p50 <= maxP50) {
                val worstLag = p50 + interval(p50, background) + FRAME_MS
                if (fresh(worstLag) != 1f) return false
                p50 += 0.5f
            }
        }
        return true
    }

    @Test fun freshnessStaysFullForEveryHealthyCadence() {
        assertTrue(sawtoothFree())
        // 健康上限处最坏 lag 125 + 250 + 33 = 408 ms，仍在新鲜度窗口起点 420 之下。
        assertEquals(408f, HairMaskTuning.MAX_P50_MS + HairMaskTuning.interval(HairMaskTuning.MAX_P50_MS, false) + FRAME_MS, 0f)
        // 对照一：间隔改成 2 × p50 之后若仍沿用旧上限 160 ms，最坏 lag 160 + 320 + 33 = 513 ms，落进渐出段（锯齿）。
        assertFalse(sawtoothFree(maxP50 = 160f))
        assertTrue(HairMaskTuning.fresh(160f + HairMaskTuning.interval(160f, false) + FRAME_MS) < 0.9f)
        // 对照二：管线组的旧窗口 150–400 ms 在 p50≈120 ms 的低端机上会在 1 与约 0–0.2 之间锯齿摆动。
        assertFalse(sawtoothFree { HairMaskTuning.fresh(it, 150f, 400f) })
        val lowEnd = 120f + HairMaskTuning.interval(120f, false) + FRAME_MS
        assertTrue(HairMaskTuning.fresh(lowEnd, 150f, 400f) < 0.5f)
        assertEquals(1f, HairMaskTuning.fresh(lowEnd), 0f)
        // 上限再高就要把新鲜期拉长：p50 ≤ (420 − 33 − 1) / 3 ≈ 128.7 才成立。
        assertTrue(sawtoothFree(maxP50 = 128.5f))
        assertFalse(sawtoothFree(maxP50 = 130f))
    }

    private fun pose(cx: Float, cy: Float, r: Float, angle: Float) =
        floatArrayOf(cx, cy, r, r * 1.3f, cos(angle), sin(angle))

    private fun apply(m: FloatArray, u: Float, v: Float) = floatArrayOf(m[0] * u + m[1] * v + m[2], m[4] * u + m[5] * v + m[6])

    /** 点在脸坐标系里的 (沿轴, 垂直轴) 分量，以 rx 为单位（度量空间）。 */
    private fun faceCoords(p: FloatArray, u: Float, v: Float, aspect: Float): FloatArray {
        val dx = u - p[0]; val dy = (v - p[1]) / aspect
        return floatArrayOf((dx * p[4] + dy * p[5]) / p[2], (-dx * p[5] + dy * p[4]) / p[2])
    }

    @Test fun affineMapsDisplayPointsToTheSameFacePointAtMaskTime() {
        val out = FloatArray(HairMaskTuning.AFFINE_SIZE)
        // 纯平移：显示时刻脸在 (0.52, 0.40)，遮罩时刻在 (0.50, 0.43)。
        assertTrue(HairMaskTuning.affine(pose(0.50f, 0.43f, 0.2f, 0f), pose(0.52f, 0.40f, 0.2f, 0f), 1f, out))
        apply(out, 0.52f, 0.40f).let { assertEquals(0.50f, it[0], 1e-4f); assertEquals(0.43f, it[1], 1e-4f) }
        apply(out, 0.10f, 0.90f).let { assertEquals(0.08f, it[0], 1e-4f); assertEquals(0.93f, it[1], 1e-4f) }
        assertEquals(0f, out[3], 0f); assertEquals(0f, out[7], 0f)
        // 等比缩放：遮罩时刻脸更大（离镜头近）1.1 倍，以脸中心为不动点。
        assertTrue(HairMaskTuning.affine(pose(0.5f, 0.5f, 0.22f, 0f), pose(0.5f, 0.5f, 0.2f, 0f), 1f, out))
        apply(out, 0.6f, 0.4f).let { assertEquals(0.61f, it[0], 1e-4f); assertEquals(0.39f, it[1], 1e-4f) }
        // 绕轴旋转：显示时刻轴向 0°，遮罩时刻 10°；方阵上点 (中心 + 0.1 沿轴) 转到 10°。
        val ten = Math.toRadians(10.0).toFloat()
        assertTrue(HairMaskTuning.affine(pose(0.5f, 0.5f, 0.2f, ten), pose(0.5f, 0.5f, 0.2f, 0f), 1f, out))
        apply(out, 0.6f, 0.5f).let { assertEquals(0.5f + 0.1f * cos(ten), it[0], 1e-4f); assertEquals(0.5f + 0.1f * sin(ten), it[1], 1e-4f) }
        // 竖屏 aspect（720×1280）下三者叠加：任一显示点映射后，在遮罩时刻脸坐标系里的位置与它在显示时刻脸坐标系里的位置相同。
        val aspect = 720f / 1280f
        val display = pose(0.48f, 0.42f, 0.21f, Math.toRadians(-6.0).toFloat())
        val mask = pose(0.53f, 0.46f, 0.23f, Math.toRadians(8.0).toFloat())
        assertTrue(HairMaskTuning.affine(mask, display, aspect, out))
        for ((u, v) in listOf(0.48f to 0.42f, 0.3f to 0.2f, 0.7f to 0.35f, 0.5f to 0.8f, 0.1f to 0.6f)) {
            val mapped = apply(out, u, v)
            val before = faceCoords(display, u, v, aspect)
            val after = faceCoords(mask, mapped[0], mapped[1], aspect)
            assertEquals("($u,$v) 沿轴", before[0], after[0], 1e-4f)
            assertEquals("($u,$v) 垂直轴", before[1], after[1], 1e-4f)
        }
        apply(out, 0.48f, 0.42f).let { assertEquals(0.53f, it[0], 1e-4f); assertEquals(0.46f, it[1], 1e-4f) }
    }

    @Test fun affineFallsBackToIdentityWhenPoseMissingOrMotionTooLarge() {
        val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f)
        val out = FloatArray(HairMaskTuning.AFFINE_SIZE)
        val here = pose(0.5f, 0.5f, 0.2f, 0f)
        fun rejected(mask: FloatArray?, display: FloatArray?, aspect: Float = 1f) {
            out.fill(7f)
            assertFalse(HairMaskTuning.affine(mask, display, aspect, out))
            assertArrayEquals(identity, out, 0f)
        }
        rejected(null, here)
        rejected(here, null)
        rejected(here, here, 0f)
        rejected(pose(0.5f, 0.5f, 0f, 0f), here) // 退化半径
        rejected(floatArrayOf(0.5f, 0.5f, 0.2f, 0.26f, 0f, 0f), here) // 退化轴向
        rejected(pose(0.61f, 0.5f, 0.2f, 0f), here) // 平移 0.11 画面宽 > 10%
        rejected(pose(0.5f, 0.5f + 0.11f * 0.5625f, 0.2f, 0f), here, 0.5625f) // 竖向平移按 aspect 折算成 0.11 画面宽
        rejected(pose(0.5f, 0.5f, 0.159f, 0f), here) // 缩放 0.795 < 0.8
        rejected(pose(0.5f, 0.5f, 0.251f, 0f), here) // 缩放 1.255 > 1.25
        // 边界以内仍补偿。
        assertTrue(HairMaskTuning.affine(pose(0.59f, 0.5f, 0.2f, 0f), here, 1f, out))
        assertTrue(HairMaskTuning.affine(pose(0.5f, 0.5f, 0.161f, 0f), here, 1f, out))
        assertTrue(HairMaskTuning.affine(pose(0.5f, 0.5f, 0.249f, 0f), here, 1f, out))
        // 同一位姿：单位阵。
        assertTrue(HairMaskTuning.affine(here, here, 0.5625f, out))
        assertArrayEquals(identity, out, 1e-6f)
    }

    @Test fun shownFollowsPresenceFreshBodyFadeAndNeckWeightAndStopsForEngine() {
        assertEquals(0.5f * 0.8f * 0.25f, HairMaskTuning.shown(0.5f, 0.8f, 0.25f, 0f, false), 1e-6f)
        for (k in 0..10) {
            val fade = k / 10f
            assertEquals(fade, HairMaskTuning.shown(1f, 1f, fade, 0f, false), 1e-6f) // 负载 1 级随 fades.body 线性渐出
            // 天鹅颈随人体存在度渐变：强度乘 (1 − neckWeight)，人体出现、消失时不再单帧硬切。
            assertEquals(1f - fade, HairMaskTuning.shown(1f, 1f, 1f, fade, false), 1e-6f)
        }
        assertEquals(0.6f * 0.25f, HairMaskTuning.shown(0.6f, 1f, 1f, 0.75f, false), 1e-6f)
        assertEquals(0f, HairMaskTuning.shown(1f, 1f, 1f, 1f, false), 0f) // 天鹅颈满强度
        assertEquals(0f, HairMaskTuning.shown(1f, 1f, 1f, Float.NaN, false), 0f) // 不确定时按满强度（回到第一批）
        assertEquals(0f, HairMaskTuning.shown(1f, 1f, 1f, 1.5f, false), 0f)
        assertEquals(1f, HairMaskTuning.shown(1f, 1f, 1f, -0.5f, false), 0f)
        assertEquals(0f, HairMaskTuning.shown(1f, 1f, 1f, 0f, true), 0f) // 外部引擎
        assertEquals(1f, HairMaskTuning.shown(1f, 1f, 1f, 0f, false), 0f)
    }

    /**
     * 30 fps 逐帧推进 [faceKeep]：[latestAt] 给出第 k 帧时最新有脸样本的时间（没有为 MIN_VALUE），[missAt] 给出 FaceTimeline.missSince
     * （最近一次完成的检测里没有脸时这一段无脸的第一个样本，否则 MIN_VALUE），[inFlight] 给出第 k 帧是否有人脸任务在途，返回每帧的值。
     */
    private fun faceKeepTrack(frames: Int, missAt: (Int, Long) -> Long = { _, _ -> Long.MIN_VALUE },
                              inFlight: (Int) -> Boolean = { false }, latestAt: (Int, Long) -> Long): List<Float> {
        var keep = 0f
        var last = Long.MIN_VALUE
        return List(frames) { k ->
            val now = 1_000L + k * 100L / 3
            val dt = if (last == Long.MIN_VALUE) 0L else now - last
            last = now
            keep = HairMaskTuning.faceKeep(keep, latestAt(k, now), missAt(k, now), now, dt, inFlight(k))
            keep
        }
    }

    @Test fun faceKeepFadesInAndReachesZeroBeforeTheTimelineClearsGeometry() {
        val frameMs = 100L / 3
        // 第 0–59 帧有脸（每帧一个样本、晚一帧到达）；之后丢脸（无脸结果同样晚一帧到达），第 120 帧起重新检出。
        var latest = Long.MIN_VALUE
        var miss = Long.MIN_VALUE
        val times = mutableListOf<Long>()
        val keep = faceKeepTrack(200, missAt = { _, _ -> miss }) { k, now ->
            times += now
            if (k < 60 || k >= 120) { latest = now - frameMs; miss = Long.MIN_VALUE }
            else if (miss == Long.MIN_VALUE) miss = now - frameMs
            latest
        }
        // 渐入 300 ms：约 9 帧到满，单调、每帧涨幅 ≤ 100 / 300。
        for (k in 1 until 60) assertTrue("k=$k", keep[k] >= keep[k - 1] && keep[k] - keep[k - 1] <= 34f / 300f + 1e-6f)
        assertTrue(keep[8] < 1f); assertEquals(1f, keep[10], 0f)
        assertEquals(1f, keep[59], 0f)
        val lostAt = times[59] - frameMs
        for (k in 60 until 120) {
            val age = times[k] - lostAt
            assertTrue("渐出每帧不超过 34/250：k=$k", keep[k - 1] - keep[k] <= 34f / HairMaskTuning.FACE_FADE_OUT_MS + 1e-6f)
            if (age <= HairMaskTuning.FACE_HOLD_MS) assertEquals("保持 k=$k", 1f, keep[k], 0f)
            // 人脸每帧采样：与「保持 200 ms 再 250 ms 线性渐出」逐帧相同。
            val linear = (1f - (age - HairMaskTuning.FACE_HOLD_MS) / HairMaskTuning.FACE_FADE_OUT_MS).coerceIn(0f, 1f)
            assertEquals("线性渐出 k=$k", linear, keep[k], 1e-5f)
            if (age >= HairMaskTuning.FACE_ZERO_MS) assertEquals("k=$k", 0f, keep[k], 0f)
            // 几何清空（FaceTimeline.update：now − latest > CLEAR_AFTER_MS）的那一帧之前一帧就已经是 0：没有整帧硬切。
            if (age + frameMs > FaceTimeline.CLEAR_AFTER_MS) assertEquals("清空前已到 0：k=$k", 0f, keep[k], 0f)
        }
        // 重新检出：从 0 起渐入，第一帧不跳回满强度。
        assertTrue(keep[120] <= 34f / 300f + 1e-6f)
        assertEquals(1f, keep[135], 0f)
        // 从来没有脸：恒为 0。
        assertTrue(faceKeepTrack(30) { _, _ -> Long.MIN_VALUE }.all { it == 0f })
    }

    @Test fun faceKeepHoldsWhileTheDetectorIsStillWorkingAndFadesOnlyAfterAConfirmedMiss() {
        // 检测器还在算（最新有脸样本之后没有结果，missSince = MIN）：age 到 350 ms 都保持满，人脸走 CPU、推理 100–120 ms 时
        // 两次并入之间的 age（约 2 × 推理 + 2–3 帧）不会让强度锯齿。
        for (age in 0L..350L) {
            assertEquals("age=$age", 1f, HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1000 + age, 33), 0f)
            // 有脸的结果比无脸的新（missSince 早于最新有脸样本）：同样不算丢脸。
            assertEquals("age=$age", 1f, HairMaskTuning.faceKeep(1f, 1000, 900, 1000 + age, 33), 0f)
        }
        // 兜底（停摆）：350–450 ms 线性降到 0，450 ms 起必为 0，与帧间隔无关。
        assertEquals(0.5f, HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1400, 33), 1e-6f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1450, 33), 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1450, 300), 0f)
        // 确认丢脸：保持期内（age ≤ 200）不降，单次漏检不下陷；还在渐入（刚重新检出又漏检）时照常按 300 ms 渐入，不跳。
        assertEquals(1f, HairMaskTuning.faceKeep(1f, 1000, 1033, 1200, 33), 0f)
        assertEquals(0.3f + 33f / 300f, HairMaskTuning.faceKeep(0.3f, 1000, 1033, 1100, 33), 1e-6f)
        // 保持期之后只计 200 ms 以后的那段：GL 卡顿 300 ms（dt 钳到 100）也与「200 + 250 线性」相同，不会停在旧值。
        assertEquals(0.6f, HairMaskTuning.faceKeep(1f, 1000, 1033, 1300, 300), 1e-6f)
        assertEquals(0.936f, HairMaskTuning.faceKeep(1f, 1000, 1033, 1216, 33), 1e-6f)
        // 确认得晚（人脸推理慢，age 已 300 ms 才拿到无脸结果）：从当前值按剩余时间线性赶在 450 ms 到 0，不跳变。
        val late = HairMaskTuning.faceKeep(1f, 1000, 1150, 1300, 33)
        assertEquals(1f - 33f / 183f, late, 1e-5f)
        var keep = late
        var now = 1300L
        while (now < 1450L) {
            val next = HairMaskTuning.faceKeep(keep, 1000, 1150, now + 33, 33)
            assertTrue("单调渐出：$keep → $next", next <= keep)
            assertTrue("每帧最多降 ${33f / 150f}", keep - next <= 33f / 150f + 1e-5f)
            keep = next; now += 33
        }
        assertEquals("450 ms 前到 0", 0f, keep, 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, 1000, 1033, 1450, 300), 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(0.3f, 1000, 1033, 1460, 0), 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, Long.MIN_VALUE, 1033, 1450, 33), 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, Long.MIN_VALUE, Long.MIN_VALUE, 1000, 33), 0f)
        // 上升限速：dt 钳到 100 ms；样本时间晚于渲染时间按 age = 0；previous 为 NaN 按 0。
        assertEquals(100f / 300f, HairMaskTuning.faceKeep(0f, 1000, Long.MIN_VALUE, 1000, 5_000), 1e-6f)
        assertEquals(0f, HairMaskTuning.faceKeep(0f, 1000, Long.MIN_VALUE, 1000, -50), 0f)
        assertEquals(0.5f, HairMaskTuning.faceKeep(0.5f, 1100, Long.MIN_VALUE, 1000, 0), 0f)
        assertEquals(10f / 300f, HairMaskTuning.faceKeep(Float.NaN, 1000, Long.MIN_VALUE, 1000, 10), 1e-6f)
    }

    @Test fun faceKeepAppliesTheStallCapOnlyWhenNoFaceTaskIsInFlight() {
        // 有人脸任务在途（下一次推理还在算）：最新有脸样本再旧也不压上限，直到时间线清空几何（500 ms）都保持满。人脸走 CPU、推理慢
        // （SYNC ≥ 180、DEFERRED ≥ 160 ms，或 120 ms 带 +80 ms 长尾）时两次并入之间的 age 会超过 350 ms，兜底若照样生效，强度与
        // 背景虚化的细节保留 β 随采样节奏锯齿（评审复核；联合仿真见 HairFallbackJointTest）。
        for (age in 0L until FaceTimeline.CLEAR_AFTER_MS) {
            assertEquals("在途 age=$age", 1f, HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1000 + age, 33, faceInFlight = true), 0f)
            assertEquals("在途 age=$age", 1f, HairMaskTuning.faceKeep(1f, 1000, 900, 1000 + age, 33, faceInFlight = true), 0f)
        }
        // 在途时渐入照常按 300 ms 限速，不因 age 大而被压。
        assertEquals(0.3f + 33f / 300f, HairMaskTuning.faceKeep(0.3f, 1000, Long.MIN_VALUE, 1420, 33, faceInFlight = true), 1e-6f)
        // 不在途（停摆）：与缺省相同，350 ms 起线性压到 0，450 ms 起必为 0。
        for (age in 0L..600L) {
            val stalled = HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1000 + age, 33, faceInFlight = false)
            assertEquals("停摆 age=$age", HairMaskTuning.faceKeep(1f, 1000, Long.MIN_VALUE, 1000 + age, 33), stalled, 0f)
            assertEquals("停摆 age=$age", ((450f - age) / 100f).coerceIn(0f, 1f), stalled, 1e-6f)
        }
        // 最后一次推理交回（有脸）之后没有任务在途、也没有新结果：逐帧推进，几何清空（age > 500）之前一帧已为 0，每帧最多降 34 / 100。
        val stalled = faceKeepTrack(30, inFlight = { false }) { _, _ -> 1_000L }
        for (k in 1 until stalled.size) {
            val age = k * 100L / 3
            assertTrue("停摆 k=$k", stalled[k] <= stalled[k - 1] || stalled[k] - stalled[k - 1] <= 34f / 300f + 1e-6f)
            if (age > HairMaskTuning.FACE_ZERO_MS - HairMaskTuning.FACE_STALL_FADE_MS) {
                assertTrue("停摆 k=$k 每帧最多降 34/100", stalled[k - 1] - stalled[k] <= 34f / 100f + 1e-6f)
            }
            if (age + FRAME_MS > FaceTimeline.CLEAR_AFTER_MS) assertEquals("停摆 清空前已到 0：k=$k", 0f, stalled[k], 0f)
        }
        // 同一序列一直在途（任务卡死，已知代价）：保持满到几何清空那一帧。
        assertTrue(faceKeepTrack(16, inFlight = { true }) { _, _ -> 1_000L }.drop(10).all { it == 1f })
        // 确认丢脸：在途与否都按「保持 200 + 渐出、450 ms 到 0」渐出。人脸每帧采样（在途）时与不在途逐帧相同。
        val frameMs = 100L / 3
        var latest = Long.MIN_VALUE
        var miss = Long.MIN_VALUE
        fun scene(k: Int, now: Long): Long {
            if (k < 60) { latest = now - frameMs; miss = Long.MIN_VALUE } else if (miss == Long.MIN_VALUE) miss = now - frameMs
            return latest
        }
        val busy = faceKeepTrack(120, missAt = { _, _ -> miss }, inFlight = { true }) { k, now -> scene(k, now) }
        latest = Long.MIN_VALUE; miss = Long.MIN_VALUE
        val idle = faceKeepTrack(120, missAt = { _, _ -> miss }, inFlight = { false }) { k, now -> scene(k, now) }
        assertEquals(idle, busy)
        assertEquals(0f, busy.last(), 0f)
        // 确认得晚（age 420 ms 才拿到无脸结果）：在途时只按剩余时间赶在 450 ms 到 0，不叠加兜底；不在途时兜底更低。
        assertEquals(1f - 33f / 63f, HairMaskTuning.faceKeep(1f, 1000, 1400, 1420, 33, faceInFlight = true), 1e-5f)
        assertEquals(0.3f, HairMaskTuning.faceKeep(1f, 1000, 1400, 1420, 33, faceInFlight = false), 1e-5f)
        var keep = 1f
        var now = 1387L
        while (now < 1450L) {
            val next = HairMaskTuning.faceKeep(keep, 1000, 1400, now + 33, 33, faceInFlight = true)
            assertTrue("在途确认得晚：单调渐出 $keep → $next", next <= keep)
            keep = next; now += 33
        }
        assertEquals("在途确认得晚：450 ms 前到 0", 0f, keep, 0f)
        assertEquals(0f, HairMaskTuning.faceKeep(1f, 1000, 1033, 1450, 33, faceInFlight = true), 0f)
        // 没有脸：在途与否都为 0。
        assertEquals(0f, HairMaskTuning.faceKeep(1f, Long.MIN_VALUE, Long.MIN_VALUE, 1000, 33, faceInFlight = true), 0f)
    }

    @Test fun wantsTruthTable() {
        val off = VideoEffectsConfig(enabled = true, smoothing = 0f, brightness = 0f, filter = "none", skinOnly = true)
        val single = listOf(
            off.copy(smoothing = 0.5f), off.copy(whiten = 0.3f), off.copy(rosy = 0.2f), off.copy(brightness = 0.1f),
            off.copy(radiance = 0.4f), off.copy(sharpness = 0.2f), off.copy(backgroundBlur = 0.6f),
        )
        for (value in single) assertTrue(value.toString(), HairMaskTuning.wants(true, true, true, value))
        assertFalse("作用项全为 0", HairMaskTuning.wants(true, true, true, off))
        // 只有不受 A 通道影响的项（瘦脸、美妆、美体）开着：不跑头发分割。
        assertFalse(HairMaskTuning.wants(true, true, true, off.copy(faceSlim = 0.3f, lipstick = 0.5f, waistSlim = 0.4f)))
        val on = off.copy(smoothing = 0.5f)
        assertFalse("头发分割故障", HairMaskTuning.wants(false, true, true, on))
        assertFalse("整体故障", HairMaskTuning.wants(true, false, true, on))
        assertFalse("不跟人脸", HairMaskTuning.wants(true, true, false, on))
        assertFalse("美颜关闭", HairMaskTuning.wants(true, true, true, on.copy(enabled = false)))
    }

    private fun mask(value: Int, w: Int = 4, h: Int = 2) = PersonMask(ByteArray(w * h) { value.toByte() }, w, h)

    @Test fun temporalBlendMixesWhenStillAndUsesNewFrameOnMotion() {
        val anchor = floatArrayOf(108f, 120f, 40f)
        val prev = mask(100); val next = mask(201)
        // 静止：新帧权重 0.5，四舍五入；不改动两张输入。
        val blended = HairMaskTuning.temporalBlend(prev, anchor, next, anchor.copyOf(), sameEpoch = true)
        assertNotSame(next, blended)
        assertEquals(List(8) { 151 }, blended.pixels.map { it.toInt() and 0xFF })
        assertEquals(100, prev.pixels[0].toInt() and 0xFF); assertEquals(201, next.pixels[0].toInt() and 0xFF)
        // 小于运动门的微动仍混合：位移 1.9% 宽、半宽变化 2.9%。
        val width = next.width.toFloat()
        val small = floatArrayOf(108f + 0.019f * width, 120f, 40f * 1.029f)
        assertNotSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, small, true))
        // 直接用新帧：位移 > 2% 遮罩宽、半宽变化 > 3%、代际变化、没有上一帧、锚点缺失、尺寸不同、EMA 关闭。
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, floatArrayOf(108f + 0.021f * width, 120f, 40f), true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, floatArrayOf(108f, 120f + 0.021f * width, 40f), true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, floatArrayOf(108f, 120f, 40f * 1.031f), true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, floatArrayOf(108f, 120f, 40f * 0.969f), true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, anchor, sameEpoch = false))
        assertSame(next, HairMaskTuning.temporalBlend(null, null, next, anchor, true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, null, next, anchor, true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, null, true))
        assertSame(next, HairMaskTuning.temporalBlend(mask(100, 2, 4), anchor, next, anchor, true))
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, anchor, true, alpha = 1f))
    }

    @Test fun temporalBlendInPlaceMatchesTheAllocatingPathByteForByte() {
        // HairSegmenter 把刚量化好的那块原地混合（out = next）：结果与新建一块逐字节相同，返回的就是那块。
        val anchor = floatArrayOf(108f, 120f, 40f)
        var seed = 5
        fun noisy() = PersonMask(ByteArray(64) { seed = seed * 1103515245 + 12345; (seed ushr 16).toByte() }, 8, 8)
        repeat(20) {
            val prev = noisy(); val next = noisy()
            val expected = HairMaskTuning.temporalBlend(prev, anchor, next, anchor, true).pixels
            val prevCopy = prev.pixels.copyOf()
            val inPlace = HairMaskTuning.temporalBlend(prev, anchor, next, anchor, true, out = next)
            assertSame(next, inPlace)
            assertArrayEquals(expected, inPlace.pixels)
            assertArrayEquals(prevCopy, prev.pixels)
        }
        // 不混合时照旧返回 next 本身，不碰 out；out 不能是上一帧那块。
        val prev = mask(100); val next = mask(201); val spare = mask(7)
        assertSame(next, HairMaskTuning.temporalBlend(prev, anchor, next, null, true, out = spare))
        assertEquals(7, spare.pixels[0].toInt())
        try { HairMaskTuning.temporalBlend(prev, anchor, next, anchor, true, out = prev); fail() } catch (_: IllegalArgumentException) {}
    }

    @Test fun temporalBlendHalvesIndependentNoiseOnStillHead() {
        // 静止头部、每帧独立噪声 ±40：EMA 后的帧间跳变明显小于原始。
        val anchor = floatArrayOf(108f, 120f, 40f)
        var seed = 11
        fun noisy() = PersonMask(ByteArray(64) { seed = seed * 1103515245 + 12345; (128 + ((seed ushr 8) % 81 - 40)).toByte() }, 8, 8)
        var raw = noisy(); var ema = raw
        var rawJump = 0.0; var emaJump = 0.0
        for (k in 0 until 200) {
            val next = noisy()
            val blended = HairMaskTuning.temporalBlend(ema, anchor, next, anchor, true)
            for (i in 0 until 64) {
                rawJump += kotlin.math.abs((next.pixels[i].toInt() and 0xFF) - (raw.pixels[i].toInt() and 0xFF))
                emaJump += kotlin.math.abs((blended.pixels[i].toInt() and 0xFF) - (ema.pixels[i].toInt() and 0xFF))
            }
            raw = next; ema = blended
        }
        assertTrue("EMA 帧间跳变 $emaJump / 原始 $rawJump", emaJump < 0.65 * rawJump)
    }

    @Test fun constantsMatchThePlan() {
        assertEquals(100L, HairMaskTuning.INTERVAL_MS)
        assertEquals(2f, HairMaskTuning.INTERVAL_P50_SCALE, 0f)
        assertEquals(200L, HairMaskTuning.BACKGROUND_INTERVAL_MS)
        assertEquals(125f, HairMaskTuning.MAX_P50_MS, 0f)
        assertEquals(3, HairMaskTuning.MASK_SPARES)
        // 「人脸近期可见」：确认丢脸后保持 200 + 渐出 250，到 0（450 ms）之后至少还有 1 帧（20 fps）才清空几何（500 ms）；
        // 停摆兜底从 350 ms 起；渐入 300 ms。
        assertEquals(200L, HairMaskTuning.FACE_HOLD_MS)
        assertEquals(250f, HairMaskTuning.FACE_FADE_OUT_MS, 0f)
        assertEquals(450L, HairMaskTuning.FACE_ZERO_MS)
        assertEquals(100f, HairMaskTuning.FACE_STALL_FADE_MS, 0f)
        assertEquals(300f, HairMaskTuning.FACE_FADE_IN_MS, 0f)
        assertEquals(HairMaskTuning.FACE_ZERO_MS.toFloat(), HairMaskTuning.FACE_HOLD_MS + HairMaskTuning.FACE_FADE_OUT_MS, 0f)
        assertTrue(HairMaskTuning.FACE_ZERO_MS + 50L <= FaceTimeline.CLEAR_AFTER_MS)
        assertTrue(HairMaskTuning.FACE_ZERO_MS + FRAME_MS <= FaceTimeline.CLEAR_AFTER_MS)
        assertEquals(10, HairMaskTuning.P50_WINDOW)
        assertEquals(8, HairMaskTuning.MIN_SAMPLES)
        assertEquals(650L, HairMaskTuning.FRESH_MS)
        assertEquals(0.5f, HairMaskTuning.EMA_ALPHA, 0f)
        // 存在度 300 / 300 / 400 ms：渐入 300 ms 到满，缺失保持 300 ms 后 400 ms 渐出。
        val presence = MaskPresence(HairMaskTuning.PRESENCE_FADE_IN_MS, HairMaskTuning.PRESENCE_HOLD_MS, HairMaskTuning.PRESENCE_FADE_OUT_MS)
        for (t in 0L..300L step 50) presence.update(true, t)
        assertEquals(1f, presence.value, 1e-6f)
        for (t in 350L..600L step 50) assertEquals(1f, presence.update(false, t), 1e-6f)
        for (t in 650L..1000L step 50) presence.update(false, t)
        assertEquals(0f, presence.value, 0f)
        assertEquals(HairMaskTuning.FRESH_MS, MaskTrackingState(HairMaskTuning.FRESH_MS).freshMs)
    }

    private companion object {
        /** 一帧（30 fps）。 */
        const val FRAME_MS = 33f
    }
}
