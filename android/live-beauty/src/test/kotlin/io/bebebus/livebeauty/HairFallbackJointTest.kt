package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sin

/** 摆头仿真里每帧求仿射的做法：(时间线, 已上传遮罩的采样时间, at(displayTime) 的几何, displayTime, 输出) → 是否补偿。 */
internal typealias SwingMethod = (FaceTimeline, Long, FloatArray?, Long, FloatArray) -> Boolean

/**
 * 「头发遮罩回落」联合测试（纯 Kotlin，不经 GL）。按 VideoEffectsFilter.drawFilter 第 1 段（人脸结果进 FaceTimeline，没有脸记
 * addMiss）、第 1c' 段（HairMaskGate.enabled：健康、宽限、负载、画面里近期有脸；p50 健康、取票、按代际复位）、第 1' 段（交给
 * worker）、第 1d 段之后（timeline.update、snapshotAt(displayTime)、HairMaskGate.update / uploaded、at(displayTime)、天鹅颈、
 * shown；A 通道要几何非空、合并道要变形前几何非空）与 drawEngine（停用头发分割后直接返回）的数据流，把真实的 MaskTrackingState
 * （新鲜期 HairMaskTuning.FRESH_MS）、HairMaskGate（MaskPresence 300 / 300 / 400 ms、「人脸近期可见」faceKeep）、HairMaskTuning、
 * TrackingGrace、LoadFades、FaceTimeline 与 BodyGeometrySmoother 串起来；推理按固定耗时在采样之后完成，出错时按 HairSegmenter
 * 的故障路径（本票作废、state 停用、onFailure → failHair）处理。
 *
 * 断言的是计划 2026-11-21 §3 的回落语义：推理失败下一帧强度就为 0（不等存在度渐出），负载 1 级随 fades.body 渐出到 0 后停用推理，
 * 外部引擎期间不进头发各段、回到自研管线后从 0 重新渐入；丢脸后强度在几何清空之前渐出到 0、无脸时停止推理、重新检出后从 0 渐入；
 * 天鹅颈按人体存在度渐变。强度为 0 时不画 A 通道、不跑背景合并道，画面与第一批逐字节相同。
 * 另有运动补偿（HairMaskGate.compensate）的用例：显示时刻越过最新人脸样本时用 at() 的外推，不退回单位阵、相邻帧不跳；
 * 以及人脸走 CPU、推理偏慢时「人脸近期可见」不随采样节奏锯齿的用例（CpuFaceRig：真实 FaceTrackingState 的 SYNC / DEFERRED 取票；
 * 推理更慢或带长尾时停摆兜底只在没有人脸任务在途时生效，背景虚化的细节保留 β 同样恒为满；检测器停摆时仍在几何清空前到 0）。
 */
class HairFallbackJointTest {
    /**
     * VideoEffectsFilter 的头发遮罩（只保留与强度、取票有关的 GL 线程状态）。第 k 帧的渲染时间 = [frameTime]，被显示帧比它早
     * [DISPLAY_DELAY_MS]（FrameDelay 约 2 帧）；人脸每帧采样，下一帧开头可用（GPU 推理约 25 ms），[faceAt] 给出该采样里有没有脸；
     * 头发推理在交给 worker 后 [latencyMs] ms 完成，[failAt] 返回 true 的采样推理时抛错。[bgActive] 为开着背景虚化（间隔下限
     * 200 ms），[neckLengthen] 为天鹅颈强度（frame(body = true) 时识别到人体）。
     * 改动 VideoEffectsFilter 第 1、1c'、1'、1d 段的头发部分或 drawEngine 时请同步修改这里。
     */
    internal class Rig(
        private val latencyMs: Long = 60L,
        private val failAt: (Long) -> Boolean = { false },
        private val faceAt: (Long) -> Boolean = { true },
        private val bgActive: Boolean = false,
        private val neckLengthen: Float = 0f,
    ) {
        val state = MaskTrackingState(HairMaskTuning.FRESH_MS)
        val gate = HairMaskGate()
        val fades = LoadFades()
        val timeline = FaceTimeline().also { it.aspect = ASPECT }
        private val bodySmoother = BodyGeometrySmoother()
        private val grace = TrackingGrace()
        private val timings = FloatArray(HairMaskTuning.P50_WINDOW)
        private var timingCount = 0
        private var pending: MaskTrackingState.Ticket? = null
        private var pendingDone = 0L
        private var epoch = Long.MIN_VALUE
        private var faceSample = Long.MIN_VALUE // 上一帧交给人脸 worker 的采样，本帧开头已完成。
        private var faceEpochChanged = false // 人脸 tracker 的代际变了（外部引擎停用过、切镜头）：下一帧时间线从头开始。
        var hairHealthy = true
            private set
        var k = 0
            private set
        /** 本帧 A 通道的实际强度（几何为空、不画时为 0）。 */
        var shown = 0f
            private set
        /** 本帧背景合并道的实际强度（开着虚化时；变形前几何为空时为 0）。 */
        var bg = 0f
            private set
        /** 本帧人体存在度（× fades.body）。 */
        var bodyPresence = 0f
            private set
        /** 本帧 at(displayTime) 是否有几何。 */
        var geometry = false
            private set
        /** 本帧交给 worker 的票数（0 / 1）。 */
        var submitted = 0
            private set
        /** worker 抛错、回调 onFailure 的时间；没有为 MIN_VALUE。 */
        var failedAt = Long.MIN_VALUE
            private set

        private fun failHair() {
            hairHealthy = false
            state.enable(false)
        }

        /** [tier] 为本帧负载级别；[engine] 为 true 时走 drawEngine（外部美颜引擎）；[body] 为本帧是否识别到人体。 */
        fun frame(tier: Int = 0, engine: Boolean = false, body: Boolean = false) {
            val now = frameTime(k)
            submitted = 0
            // worker：渲染时刻之前已经完成的推理（HairSegmenter.run：成功则记耗时、交结果；出错则作废、停用、onFailure）。
            pending?.let { ticket ->
                if (pendingDone <= now) {
                    pending = null
                    if (failAt(ticket.timestamp)) {
                        state.complete(ticket, null, pendingDone)
                        state.enable(false)
                        failedAt = pendingDone
                        failHair()
                    } else {
                        timings[timingCount % timings.size] = latencyMs.toFloat()
                        timingCount++
                        state.complete(ticket, MASK, pendingDone)
                    }
                }
            }
            fades.update(tier, now)
            if (engine) {
                // drawEngine：tracker 与头发分割都停用（不关闭）后直接返回，不进第 1c'、1d 段：gate 不推进，本帧不画 A 通道、
                // 不跑合并道。头发分割的停用（hairSegmenter?.state?.enable(false)）在这里是手抄的，生产那一行由门禁
                // check_hair_wiring 静态核对。
                state.enable(false)
                faceSample = Long.MIN_VALUE; faceEpochChanged = true
                shown = 0f; bg = 0f
                k++
                return
            }
            // 第 1 段：上一帧的人脸采样已完成，有脸进时间线、没有脸记丢脸；tracker 代际变了（引擎期间停用过、切镜头），时间线从头开始。
            if (faceEpochChanged) { timeline.reset(); timeline.aspect = ASPECT; faceEpochChanged = false }
            if (faceSample != Long.MIN_VALUE) {
                if (faceAt(faceSample)) timeline.add(faceSample, faceState(0.5f, 0.45f)) else timeline.addMiss(faceSample)
            }
            faceSample = now
            // 第 1c' 段（读的是上一帧 timeline.update 之后的人脸历史）。
            var ticket: MaskTrackingState.Ticket? = null
            if (hairHealthy) {
                val enabled = HairMaskGate.enabled(hairHealthy, grace.keep(HairMaskTuning.wants(hairHealthy, true, true, CONFIG), now),
                    tier, fades.body, timeline.latest, now)
                val p50 = HairMaskTuning.p50(timings, timingCount)
                if (!HairMaskTuning.healthy(p50)) failHair()
                state.enable(enabled && hairHealthy)
                ticket = state.begin(now, WIDTH, HEIGHT, HairMaskTuning.interval(p50, bgActive))
                val snapshot = state.snapshot(now)
                if (snapshot.epoch != epoch) { epoch = snapshot.epoch; gate.reset() }
            }
            if (!hairHealthy) ticket = null
            // 第 1' 段：SYNC 当帧读回交给 worker（单任务在途由 MaskTrackingState 保证）。
            ticket?.let { pending = it; pendingDone = now + latencyMs; submitted = 1 }
            // 第 1d 段之后：按被显示帧取遮罩，算强度，换了遮罩就上传（这里上传总成功）。
            timeline.update(now)
            val displayTime = now - DISPLAY_DELAY_MS
            var mask: PersonMask? = null
            var maskStamp = Long.MIN_VALUE
            if (hairHealthy) state.snapshotAt(displayTime, now).let { mask = it.mask; maskStamp = it.sampleTimestamp }
            // 本帧的人脸采样刚交给 worker（GPU 推理约 25 ms，下一帧开头才完成）：快照时人脸任务在途。
            val upload = gate.update(hairHealthy, mask, maskStamp, displayTime, fades.body, timeline.latest, timeline.missSince,
                faceSample != Long.MIN_VALUE, now)
            if (upload != null) gate.uploaded(maskStamp)
            // 美体（天鹅颈 neck = neckLengthen × 0.3 × bodyPresence，画了颈部拉伸才算 neckLifted）。
            val smoothed = bodySmoother.update(if (body) BODY else null, now, now)
            bodyPresence = if (smoothed != null) bodySmoother.presence * fades.body else 0f
            val neckLifted = smoothed != null && neckLengthen * 0.3f * bodyPresence > 0f
            val rawGeometry = timeline.at(displayTime)
            geometry = rawGeometry != null
            val level = if (gate.stamp != Long.MIN_VALUE) gate.level else 0f
            bg = if (level > 0f && rawGeometry != null) level else 0f
            val hairShown = if (level > 0f) gate.shown(HairMaskGate.neckWeight(neckLifted, bodyPresence), engine = false) else 0f
            shown = if (hairShown > 0f && rawGeometry != null) hairShown else 0f
            k++
        }

        /**
         * VideoEffectsFilter.resetFaceTracking（切镜头，主线程）：人脸 tracker 的状态复位（代际变了，在途的人脸结果作废，下一帧时间线
         * 从头开始），头发分割的历史一并作废（hairSegmenter?.state?.reset()，生产那一行由门禁静态核对）。
         */
        fun resetFaceTracking() {
            faceSample = Long.MIN_VALUE; faceEpochChanged = true
            state.reset()
        }

        /** 先跑 [WARMUP_FRAMES] 帧（推理攒满 p50、存在度渐入完）：返回这期间渐入完成后的最小强度。 */
        fun warmUp(): Float {
            var min = 1f
            while (k < WARMUP_FRAMES) {
                frame()
                if (k > FADE_IN_FRAMES) min = minOf(min, shown)
            }
            return min
        }
    }

    @Test fun steadyMasksKeepFullStrengthEvenNearTheHealthLimit() {
        // 正常推理与健康上限附近（p50 = 120 ms，间隔 240 ms；开虚化时下限 200 ms）：最坏 lag 也在新鲜度窗口 420 ms 之下，
        // 强度恒为 1，不随推理相位锯齿。
        for (bgActive in listOf(false, true)) for (latency in listOf(60L, 120L)) {
            val rig = Rig(latencyMs = latency, bgActive = bgActive)
            assertEquals("latency=$latency bg=$bgActive", 1f, rig.warmUp(), 0f)
            assertEquals(1f, rig.bg, 0f)
            assertTrue(rig.hairHealthy)
        }
    }

    @Test fun inferenceFailureDropsStrengthToZeroOnTheNextFrame() {
        // 推理在 failFrom 之后的第一次采样上抛错：worker 作废本票、停用 state、回调 onFailure → failHair。
        val failFrom = frameTime(WARMUP_FRAMES + 7)
        val rig = Rig(failAt = { it >= failFrom })
        assertEquals(1f, rig.warmUp(), 0f)
        while (rig.failedAt == Long.MIN_VALUE) {
            rig.frame()
            assertTrue("故障之前强度保持 1：${rig.shown}", rig.failedAt != Long.MIN_VALUE || rig.shown == 1f)
            assertTrue(rig.k < WARMUP_FRAMES + 60)
        }
        // 故障被处理的那一帧起强度就为 0：不等存在度保持 300 ms 再渐出（那样会把故障前最后一张遮罩继续画 0.7 s）。
        assertFalse(rig.hairHealthy)
        assertEquals(0f, rig.shown, 0f)
        assertEquals(0f, rig.bg, 0f)
        repeat(60) {
            rig.frame()
            assertEquals("本次会话内不再恢复（第 ${rig.k} 帧）", 0f, rig.shown, 0f)
            assertEquals("不再取票", 0, rig.submitted)
        }
    }

    @Test fun slowInferenceDisablesOnceEightSamplesExceed125ms() {
        // p50 > MAX_P50_MS（125 ms）：攒满 8 次才判，判定后下一帧强度为 0、不再推理（回到第一批）。
        val rig = Rig(latencyMs = 130L)
        var samples = 0
        while (rig.hairHealthy) {
            rig.frame()
            samples += rig.submitted
            assertTrue(rig.k < 400)
        }
        assertTrue("至少攒满 ${HairMaskTuning.MIN_SAMPLES} 次才判：$samples", samples >= HairMaskTuning.MIN_SAMPLES)
        assertEquals(0f, rig.shown, 0f)
        repeat(30) { rig.frame(); assertEquals(0f, rig.shown, 0f); assertEquals(0, rig.submitted) }
    }

    @Test fun loadTierOneFadesWithBodyThenStopsInference() {
        val rig = Rig()
        assertEquals(1f, rig.warmUp(), 0f)
        var previous = rig.shown
        var submittedWhileFading = 0
        var zeroAt = -1
        // 负载 1 级：强度 = 存在度 × 新鲜度 × fades.body，随 fades.body 单调渐出，400 ms（约 12 帧）内到 0。
        repeat(60) { i ->
            rig.frame(tier = 1)
            assertEquals("强度跟着 fades.body 走（第 $i 帧）", rig.fades.body, rig.shown, 1e-6f)
            assertTrue("单调渐出：$previous → ${rig.shown}", rig.shown <= previous)
            previous = rig.shown
            if (rig.fades.body > 0f) submittedWhileFading += rig.submitted
            if (rig.shown == 0f && zeroAt < 0) zeroAt = i
            if (zeroAt >= 0) {
                assertEquals("渐出到 0 后不再取票（第 $i 帧）", 0, rig.submitted)
                assertEquals(0f, rig.shown, 0f)
            }
        }
        assertTrue("渐出期间推理照常（不是一升级就停）：$submittedWhileFading", submittedWhileFading > 0)
        assertTrue("400 ms 内渐出到 0：第 $zeroAt 帧", zeroAt in 0..(LoadFades.FADE_MS / FRAME_MS).toInt() + 1)
        // 渐出到 0 后头发分割已停用：历史清空，取不到遮罩。
        assertNull(rig.state.snapshotAt(frameTime(rig.k), frameTime(rig.k)).mask)
        assertTrue("负载降级不是故障", rig.hairHealthy)
        // 回到 0 级：重新启用、从 0 渐入。
        repeat(WARMUP_FRAMES) { rig.frame(tier = 0) }
        assertEquals(1f, rig.shown, 0f)
    }

    @Test fun externalEngineSkipsHairStagesAndRestartsFromZeroAfterwards() {
        val rig = Rig()
        assertEquals(1f, rig.warmUp(), 0f)
        val epochBefore = rig.state.snapshot(frameTime(rig.k)).epoch
        val stamp = rig.gate.stamp
        val level = rig.gate.level
        // 外部引擎接管 200 ms：drawEngine 停用头发分割后直接返回，不进第 1c'、1d 段——不取票，gate 不推进（强度为 0 只是因为
        // 本帧根本不画 A 通道、不跑合并道，不是 gate 算出来的）。
        repeat(6) {
            rig.frame(engine = true)
            assertEquals(0, rig.submitted)
            assertEquals("gate 不推进", stamp, rig.gate.stamp)
            assertEquals("gate 不推进", level, rig.gate.level, 0f)
        }
        // 头发分割已停用：历史清空、代际变了，在途结果按代际作废。
        assertNull(rig.state.snapshotAt(frameTime(rig.k), frameTime(rig.k)).mask)
        assertNotEquals(epochBefore, rig.state.snapshot(frameTime(rig.k)).epoch)
        // 引擎故障、回到自研管线：不沿用引擎之前的旧遮罩（它的 lag 还在新鲜期内），新遮罩到了才从 0 按 300 ms 渐入。
        // 这一条有区分力的前提是 drawEngine 停用了头发分割（代际变化 → gate.reset），生产那一行由门禁静态核对。
        // 回来后时间线也从头开始（引擎期间 tracker 停用过），「人脸近期可见」同样从 0 渐入。
        val engineEnd = frameTime(rig.k)
        var firstNonZero = -1
        var previous = 0f
        repeat(WARMUP_FRAMES) { i ->
            rig.frame()
            if (rig.shown > 0f && firstNonZero < 0) {
                firstNonZero = i
                assertTrue("第一次画出来的是回到自研之后采样的新遮罩（采样 ${rig.gate.stamp}，回来 $engineEnd）", rig.gate.stamp >= engineEnd)
            }
            // 渐入 300 ms：出现后的前 6 帧（200 ms）仍低于 1 且单调上升。
            if (firstNonZero >= 0 && i - firstNonZero < 6) {
                assertTrue("从 0 渐入（第 $i 帧 ${rig.shown}）", rig.shown < 1f && rig.shown >= previous)
            }
            previous = rig.shown
        }
        assertTrue("新遮罩至少一次推理之后才出现：第 $firstNonZero 帧", firstNonZero >= 1)
        assertEquals(1f, rig.shown, 0f)
        assertTrue(rig.hairHealthy)
    }

    @Test fun cameraSwitchNeverDrawsTheOldCamerasMask() {
        // 切镜头（NOTICE 隐私段：旧结果失效）：旧镜头的头发遮罩还在新鲜期内（650 ms），也不能再画到新镜头上；新遮罩到了才从 0 渐入。
        val rig = Rig()
        assertEquals(1f, rig.warmUp(), 0f)
        val switchAt = frameTime(rig.k)
        rig.resetFaceTracking()
        var firstNonZero = -1
        repeat(60) { i ->
            rig.frame()
            if (rig.shown > 0f || rig.bg > 0f) {
                assertTrue("切镜头后只画新镜头的遮罩（第 $i 帧：采样 ${rig.gate.stamp}，切换 $switchAt）", rig.gate.stamp >= switchAt)
                if (firstNonZero < 0) firstNonZero = i
            }
        }
        assertTrue("新遮罩至少一次推理之后才出现：第 $firstNonZero 帧", firstNonZero >= 1)
        assertEquals(1f, rig.shown, 0f)
    }

    @Test fun faceLossFadesOutBeforeGeometryClearsStopsInferenceAndRampsBackIn() {
        // 第 WARMUP+10 帧起画面里没有脸，2.5 s 后重新检出（短于 TrackingGrace 的 3 s：无脸停推理不能靠宽限到期）。
        val lostFrom = frameTime(WARMUP_FRAMES + 10)
        val backFrom = lostFrom + 2_500L
        val rig = Rig(faceAt = { it < lostFrom || it >= backFrom })
        assertEquals(1f, rig.warmUp(), 0f)
        var previousShown = rig.shown
        var previousBg = rig.bg
        var lastFace = Long.MIN_VALUE // 最后一个有脸样本的采样时间（= 渲染时间）。
        var lastTicket = Long.MIN_VALUE // 丢脸期间最后一次取票的渲染时间。
        var geometryGoneAt = Long.MIN_VALUE
        var resumedAt = Long.MIN_VALUE
        var fullAgainAt = Long.MIN_VALUE
        var maxStep = 0f
        while (frameTime(rig.k) < backFrom + 1_500L) {
            val now = frameTime(rig.k)
            if (now < lostFrom) lastFace = now
            rig.frame()
            val step = maxOf(abs(rig.shown - previousShown), abs(rig.bg - previousBg))
            maxStep = maxOf(maxStep, step)
            assertTrue("相邻帧强度差 ≤ $MAX_FACE_STEP（t=${now - lostFrom} ms：A $previousShown → ${rig.shown}，" +
                "合并道 $previousBg → ${rig.bg}）", step <= MAX_FACE_STEP)
            previousShown = rig.shown; previousBg = rig.bg
            if (now in lostFrom until backFrom) {
                if (rig.submitted > 0) lastTicket = now
                if (now - lastFace > FaceTimeline.CLEAR_AFTER_MS + 2 * FRAME_MS) {
                    assertNull("无脸时头发分割已停用、历史清空（t=${now - lastFace} ms）", rig.state.snapshot(now).mask)
                }
                if (!rig.geometry && geometryGoneAt == Long.MIN_VALUE) {
                    geometryGoneAt = now
                    assertEquals("几何清空之前已渐出到 0", 0f, previousShown, 0f)
                }
            } else if (now >= backFrom) {
                if (rig.submitted > 0 && resumedAt == Long.MIN_VALUE) resumedAt = now
                if (rig.shown == 1f && fullAgainAt == Long.MIN_VALUE) fullAgainAt = now
            }
        }
        assertTrue("时间线清空了几何", geometryGoneAt != Long.MIN_VALUE)
        // 无脸停推理：时间线清空（500 ms）后的下一帧起不再取票，不靠 3 s 宽限到期；上限 FACE_RECENT_MS。
        val stoppedAfter = lastTicket - lastFace
        assertTrue("丢脸后 ${stoppedAfter} ms 仍在取票", stoppedAfter <= FaceTimeline.CLEAR_AFTER_MS + 2 * FRAME_MS)
        assertTrue(stoppedAfter <= HairMaskGate.FACE_RECENT_MS)
        // 重新检出：人脸样本晚一帧进时间线，随即恢复取票；强度从 0 渐入（相邻帧差已在上面断言），1 s 内回到满强度。
        assertTrue("重新检出后 ${resumedAt - backFrom} ms 才恢复取票", resumedAt != Long.MIN_VALUE && resumedAt - backFrom <= 3 * FRAME_MS)
        assertTrue("重新检出后 ${fullAgainAt - backFrom} ms 回到满强度", fullAgainAt != Long.MIN_VALUE && fullAgainAt - backFrom <= 1_000L)
        println("丢脸：最大相邻帧强度差 %.3f，停止取票 %d ms，几何清空 %d ms，恢复取票 %d ms，回到满强度 %d ms".format(
            maxStep, stoppedAfter, geometryGoneAt - lastFace, resumedAt - backFrom, fullAgainAt - backFrom))
    }

    /**
     * 人脸走 CPU 的驱动（真实的 FaceTrackingState：CPU_INTERVAL_MS 一采、单任务在途），按 VideoEffectsFilter 第 0'、1、1'、1d 段：
     * SYNC 本帧 begin 取票、本帧交给 worker；DEFERRED 本帧 due（不看忙）画小图，下一帧开头 begin(上一帧采样时间, expectEpoch)
     * 再交给 worker（忙则作废）。worker 交出后 [latency] ms 完成（按采样给出，[faceAt] 为该采样里有没有脸）；第 1 段 snapshot
     * 有新几何就 add，本代际新完成的结果没有脸（或有脸的那个已不新鲜）记 addMiss；之后 timeline.update、HairMaskGate.update
     * （在途信号取本帧快照的 inFlight），再按 at(displayTime) 求背景虚化的细节保留 [beta]（BackgroundBlurPass.detailKeep(有脸,
     * hairGate.faceKeep)）。头发遮罩恒为新鲜（采样时间 = displayTime），level 只随 faceKeep 变。第 k 帧渲染时间 = frameTime(k) ±
     * [jitterMs]。渲染时间到 [stallFrom] 起不再取票、不再画小图（相机不再出新帧，GL 照常重绘：newFrame 为 false），已在途的推理照常交回。
     */
    private class CpuFaceRig(
        private val deferred: Boolean,
        private val latency: (Long) -> Long,
        private val faceAt: (Long) -> Boolean,
        private val jitterMs: Int,
        seed: Int,
        private val stallFrom: Long = Long.MAX_VALUE,
    ) {
        val state = FaceTrackingState().also { it.enable(true) }
        val timeline = FaceTimeline().also { it.aspect = ASPECT }
        val gate = HairMaskGate()
        private val random = java.util.Random(seed.toLong())
        private var pending: FaceTrackingState.Ticket? = null
        private var pendingDone = Long.MIN_VALUE
        private var staged: Long? = null
        private var stagedAt = Long.MIN_VALUE
        private var epoch = Long.MIN_VALUE
        private var lastGeometry: FaceGeometry? = null
        private var lastCompletedSeen = Long.MIN_VALUE
        var k = 0
            private set
        var now = Long.MIN_VALUE
            private set
        /** 本帧快照时人脸任务是否在途。 */
        var inFlight = false
            private set
        /** 本帧背景虚化的细节保留 β（at(displayTime) 没有几何时为 0）。 */
        var beta = 0f
            private set

        fun frame() {
            now = frameTime(k) + if (jitterMs > 0) random.nextInt(2 * jitterMs + 1) - jitterMs else 0
            val newFrame = now < stallFrom
            // worker：渲染时刻之前已经完成的推理。
            pending?.let { ticket ->
                if (pendingDone <= now) {
                    pending = null
                    state.complete(ticket, if (faceAt(ticket.timestamp)) GEOMETRY.copy() else null, pendingDone) // 每次结果一个新对象
                }
            }
            // 0'. DEFERRED：上一帧画好的小图在本帧开头取票（采样时间仍是上一帧），取到才交给 worker。
            if (deferred && newFrame) {
                val due = staged
                staged = null
                due?.let { e -> state.begin(stagedAt, WIDTH, HEIGHT, FaceTrackingState.CPU_INTERVAL_MS, expectEpoch = e) }
                    ?.let { pending = it; pendingDone = now + latency(it.timestamp) }
            }
            // 1. 人脸跟踪：DEFERRED 只判到期并画小图，SYNC 当帧取票、第 1' 段交给 worker（同一相机帧的重绘都不做）。
            if (newFrame && deferred) {
                staged = state.due(now, WIDTH, HEIGHT, FaceTrackingState.CPU_INTERVAL_MS)
                stagedAt = now
            } else if (newFrame) {
                state.begin(now, WIDTH, HEIGHT, FaceTrackingState.CPU_INTERVAL_MS)
                    ?.let { pending = it; pendingDone = now + latency(it.timestamp) }
            }
            val snapshot = state.snapshot(now)
            if (snapshot.epoch != epoch) {
                epoch = snapshot.epoch; timeline.reset(); timeline.aspect = ASPECT; lastCompletedSeen = Long.MIN_VALUE
            }
            snapshot.geometry?.let {
                if (it !== lastGeometry) { lastGeometry = it; timeline.add(snapshot.sampleTimestamp, it.toState(), snapshot.mesh) }
            }
            if (snapshot.completed != Long.MIN_VALUE && snapshot.completed > lastCompletedSeen) {
                lastCompletedSeen = snapshot.completed
                if (snapshot.geometry == null || snapshot.sampleTimestamp < snapshot.completed) timeline.addMiss(snapshot.completed)
            }
            inFlight = snapshot.inFlight
            // 1d 段之后。
            timeline.update(now)
            val displayTime = now - DISPLAY_DELAY_MS
            if (gate.update(true, MASK, displayTime, displayTime, 1f, timeline.latest, timeline.missSince, inFlight, now) != null) {
                gate.uploaded(displayTime)
            }
            // 2a'：背景虚化的人脸椭圆只在 at(displayTime) 有几何时传，β 再乘「人脸近期可见」。
            beta = BackgroundBlurPass.detailKeep(timeline.at(displayTime) != null, gate.faceKeep)
            k++
        }

        companion object {
            /** 与 faceState(0.5, 0.45) 同一状态的人脸几何（toState() 逐值相同）。 */
            val GEOMETRY = faceState(0.5f, 0.45f).let { s ->
                FaceGeometry(s.copyOfRange(0, 16), s.copyOfRange(16, 22), s.copyOfRange(22, 26), s.copyOfRange(26, 28),
                    s.copyOfRange(28, 32), s.copyOfRange(32, 40), s[40])
            }
        }
    }

    @Test fun slowCpuFaceKeepsFullStrengthWhileTheFaceStaysAndStillFadesBeforeGeometryClears() {
        // 人脸走 CPU、推理 100–120 ms（SYNC 与 DEFERRED，帧时间 ±4 ms 抖动）：画面里一直有脸时，两次并入之间最新样本的年龄可达约
        // 2 × 推理 + 2–3 帧，「人脸近期可见」不能随这个节奏锯齿——稳态强度恒为 1（修复前只看年龄：SYNC 最低 0.83–0.85、DEFERRED 0.70，评审复核）。
        // 丢脸后检测器给出无脸结果才开始渐出，仍在时间线清空几何（500 ms）之前到 0，渐出单调、没有整帧硬切。
        val lostFrom = frameTime(400)
        val latencies: List<Pair<String, (Long) -> Long>> = listOf(
            "100" to { _ -> 100L }, "110" to { _ -> 110L }, "120" to { _ -> 120L },
            "100–120" to { ts -> 100L + Math.floorMod(ts * 7919L, 21L) })
        var worstAge = 0L
        var worstStep = 0f
        for (deferred in listOf(false, true)) for ((label, latency) in latencies) for (seed in 1..3) {
            val name = "${if (deferred) "DEFERRED" else "SYNC"} L=$label seed=$seed"
            val rig = CpuFaceRig(deferred, latency, faceAt = { it < lostFrom }, jitterMs = 4, seed = seed)
            var maxAge = 0L
            while (rig.k < 400) {
                rig.frame()
                if (rig.k > 60) {
                    maxAge = maxOf(maxAge, rig.now - rig.timeline.latest)
                    assertEquals("$name 第 ${rig.k} 帧（age ${rig.now - rig.timeline.latest} ms）稳态 level 恒为 1", 1f, rig.gate.level, 0f)
                }
            }
            worstAge = maxOf(worstAge, maxAge)
            assertTrue("$name 最大 age $maxAge ms 留在兜底起点 350 ms 之前",
                maxAge < HairMaskTuning.FACE_ZERO_MS - HairMaskTuning.FACE_STALL_FADE_MS.toLong())
            // 丢脸：几何清空那一帧之前已到 0，单调渐出。
            var previous = rig.gate.level
            var clearedAt = Long.MIN_VALUE
            var lastFace = rig.timeline.latest
            while (clearedAt == Long.MIN_VALUE) {
                rig.frame()
                assertTrue("$name 等不到几何清空", rig.k < 600)
                if (rig.timeline.latest == Long.MIN_VALUE) {
                    clearedAt = rig.now
                    assertEquals("$name 几何清空之前已渐出到 0（清空在最新有脸样本之后 ${clearedAt - lastFace} ms）", 0f, previous, 0f)
                } else {
                    lastFace = rig.timeline.latest
                }
                assertTrue("$name 单调渐出：$previous → ${rig.gate.level}", rig.gate.level <= previous)
                // 确认得晚（约 2 × 推理之后才拿到无脸结果），渐出按剩余时间加快，相邻帧降幅仍不超过人脸每帧采样时的上限。
                assertTrue("$name 相邻帧降幅 ≤ $MAX_FACE_STEP：$previous → ${rig.gate.level}", previous - rig.gate.level <= MAX_FACE_STEP)
                worstStep = maxOf(worstStep, previous - rig.gate.level)
                previous = rig.gate.level
            }
            assertEquals(0f, rig.gate.level, 0f)
        }
        println("CPU 人脸 100–120 ms：稳态最大 age %d ms（level 恒为 1），丢脸渐出相邻帧最大降幅 %.3f".format(worstAge, worstStep))
    }

    @Test fun slowOrLongTailCpuFaceKeepsFaceKeepAndDetailKeepFullWhileTheNextInferenceIsInFlight() {
        // 评审复核：人脸走 CPU、推理更慢（SYNC 180 / 200 ms、DEFERRED 160 / 180 ms）或 120 ms 带长尾（20% 的推理 +80 ms）时，画面里
        // 一直有脸，两次并入之间最新样本的年龄也会超过停摆兜底的起点 350 ms；这段时间下一次推理一直在途，兜底不能生效——「人脸近期
        // 可见」与背景虚化的细节保留 β 都恒为满（兜底不看在途时：SYNC 180 最低 0.76、200 最低 0.16，DEFERRED 160 最低 0.78、180 最低
        // 0.42，SYNC 120 带长尾最低 0.15）。丢脸后仍在几何清空之前到 0（推理越慢确认越晚，渐出越陡：已知代价）。
        val lostFrom = frameTime(400)
        val stall = HairMaskTuning.FACE_ZERO_MS - HairMaskTuning.FACE_STALL_FADE_MS.toLong()
        fun fixed(ms: Long): (Int) -> (Long) -> Long = { _ -> { _ -> ms } }
        fun tail(base: Long): (Int) -> (Long) -> Long = { seed ->
            val random = java.util.Random(seed * 31L + 7)
            val pick: (Long) -> Long = { _ -> base + if (random.nextDouble() < 0.2) 80L else 0L }
            pick
        }
        val strict: List<Triple<String, Boolean, (Int) -> (Long) -> Long>> = listOf(
            Triple("SYNC L=180", false, fixed(180)), Triple("SYNC L=200", false, fixed(200)),
            Triple("DEFERRED L=160", true, fixed(160)), Triple("DEFERRED L=180", true, fixed(180)),
            Triple("SYNC L=120 + 20% +80", false, tail(120)))
        var worstAge = 0L
        var worstStep = 0f
        for ((label, deferred, latency) in strict) for (seed in 1..3) {
            val name = "$label seed=$seed"
            val rig = CpuFaceRig(deferred, latency(seed), faceAt = { it < lostFrom }, jitterMs = 4, seed = seed)
            var maxAge = 0L
            while (rig.k < 400) {
                rig.frame()
                if (rig.k > 60) {
                    val age = rig.now - rig.timeline.latest
                    assertTrue("$name 第 ${rig.k} 帧几何不该清空", rig.timeline.latest != Long.MIN_VALUE)
                    maxAge = maxOf(maxAge, age)
                    assertEquals("$name 第 ${rig.k} 帧（age $age ms，在途 ${rig.inFlight}）faceKeep 恒为 1", 1f, rig.gate.faceKeep, 0f)
                    assertEquals("$name 第 ${rig.k} 帧（age $age ms）β 恒为 DETAIL_KEEP", BackgroundBlurPass.DETAIL_KEEP, rig.beta, 0f)
                    assertEquals("$name 第 ${rig.k} 帧 level 恒为 1", 1f, rig.gate.level, 0f)
                }
            }
            // 用例确实走到了兜底会生效的年龄（> 350 ms），而且这些帧上都有任务在途。
            assertTrue("$name 最大 age $maxAge ms 应超过兜底起点 $stall ms", maxAge > stall)
            worstAge = maxOf(worstAge, maxAge)
            // 丢脸：单调渐出，几何清空那一帧之前已到 0（β 同步）。
            var previous = rig.gate.faceKeep
            var previousBeta = rig.beta
            var clearedAt = Long.MIN_VALUE
            while (clearedAt == Long.MIN_VALUE) {
                rig.frame()
                assertTrue("$name 等不到几何清空", rig.k < 600)
                if (rig.timeline.latest == Long.MIN_VALUE) {
                    clearedAt = rig.now
                    assertEquals("$name 几何清空之前已渐出到 0", 0f, previous, 0f)
                    assertEquals("$name 几何清空之前 β 已到 0", 0f, previousBeta, 0f)
                }
                assertTrue("$name 单调渐出：$previous → ${rig.gate.faceKeep}", rig.gate.faceKeep <= previous)
                worstStep = maxOf(worstStep, previous - rig.gate.faceKeep)
                previous = rig.gate.faceKeep; previousBeta = rig.beta
            }
        }
        // 时间线自己会清空几何的情形（DEFERRED 200 ms，或 DEFERRED 120 ms 带长尾：两次并入之间超过 500 ms、或结果超过 250 ms 作废，
        // 修复前后都一样）：faceKeep 只在确认丢脸或几何清空时下降，从不因停摆兜底下降。
        val loose: List<Triple<String, Boolean, (Int) -> (Long) -> Long>> = listOf(
            Triple("DEFERRED L=200", true, fixed(200)), Triple("DEFERRED L=120 + 20% +80", true, tail(120)))
        for ((label, deferred, latency) in loose) for (seed in 1..3) {
            val name = "$label seed=$seed"
            val rig = CpuFaceRig(deferred, latency(seed), faceAt = { true }, jitterMs = 4, seed = seed)
            var previous = 0f
            while (rig.k < 400) {
                rig.frame()
                val lost = rig.timeline.latest == Long.MIN_VALUE ||
                    (rig.timeline.missSince != Long.MIN_VALUE && rig.timeline.missSince > rig.timeline.latest)
                if (rig.k > 60 && rig.gate.faceKeep < previous) {
                    assertTrue("$name 第 ${rig.k} 帧（age ${rig.now - rig.timeline.latest} ms）没有确认丢脸也没有清空几何，" +
                        "faceKeep 却下降：$previous → ${rig.gate.faceKeep}", lost)
                }
                previous = rig.gate.faceKeep
            }
        }
        println("CPU 人脸 SYNC 180 / 200、DEFERRED 160 / 180、SYNC 120 带长尾：稳态最大 age %d ms（faceKeep 与 β 恒为满），丢脸渐出相邻帧最大降幅 %.3f".format(
            worstAge, worstStep))
    }

    @Test fun stalledFaceDetectorStillFadesToZeroBeforeGeometryClearsWhenNoTaskIsInFlight() {
        // 停摆：相机不再出新帧（GL 照常重绘），最后一次推理交回之后没有人脸任务在途、也没有新结果。兜底照常生效：最新有脸样本
        // 350 ms 起线性降，几何清空（500 ms）之前一帧「人脸近期可见」与 β 都已为 0，没有整帧硬切。
        val stallFrom = frameTime(300)
        var worstStep = 0f
        for (deferred in listOf(false, true)) for (latency in listOf(30L, 120L, 160L)) for (seed in 1..2) {
            val name = "${if (deferred) "DEFERRED" else "SYNC"} L=$latency seed=$seed"
            val rig = CpuFaceRig(deferred, { _ -> latency }, faceAt = { true }, jitterMs = 4, seed = seed, stallFrom = stallFrom)
            while (rig.k < 300) rig.frame()
            assertEquals("$name 停摆前满强度", 1f, rig.gate.faceKeep, 0f)
            var previous = rig.gate.faceKeep
            var previousBeta = rig.beta
            var clearedAt = Long.MIN_VALUE
            while (clearedAt == Long.MIN_VALUE) {
                rig.frame()
                assertTrue("$name 等不到几何清空", rig.k < 360)
                if (rig.timeline.latest == Long.MIN_VALUE) {
                    clearedAt = rig.now
                    assertEquals("$name 几何清空之前已渐出到 0", 0f, previous, 0f)
                    assertEquals("$name 几何清空之前 β 已到 0", 0f, previousBeta, 0f)
                    break
                }
                val age = rig.now - rig.timeline.latest
                if (rig.gate.faceKeep < 1f) {
                    assertFalse("$name 降的时候没有任务在途（age $age ms）", rig.inFlight)
                    assertTrue("$name age $age ms 才开始降", age > HairMaskTuning.FACE_ZERO_MS - HairMaskTuning.FACE_STALL_FADE_MS.toLong())
                }
                assertTrue("$name 单调：$previous → ${rig.gate.faceKeep}", rig.gate.faceKeep <= previous)
                // 每帧最多降 (帧间隔 + 抖动) / FACE_STALL_FADE_MS。
                assertTrue("$name 每帧降幅：$previous → ${rig.gate.faceKeep}",
                    previous - rig.gate.faceKeep <= (FRAME_MS + 8f) / HairMaskTuning.FACE_STALL_FADE_MS + 1e-5f)
                worstStep = maxOf(worstStep, previous - rig.gate.faceKeep)
                previous = rig.gate.faceKeep; previousBeta = rig.beta
            }
        }
        println("人脸检测停摆：几何清空前已到 0，相邻帧最大降幅 %.3f".format(worstStep))
    }

    @Test fun swanNeckScalesAlphaWithBodyPresenceInsteadOfHardSwitching() {
        // 天鹅颈开着，人体在第 WARMUP+10 帧出现、1.5 s 后消失：A 通道强度 = level × (1 − 人体存在度)，30 fps 相邻帧差 ≤ 0.4，
        // 人体存在度为 1 时为 0（稳态与第一批逐字节相同）。
        val bodyFrom = WARMUP_FRAMES + 10
        val bodyUntil = bodyFrom + 45
        val rig = Rig(neckLengthen = 1f)
        assertEquals(1f, rig.warmUp(), 0f)
        var previous = rig.shown
        var maxStep = 0f
        var fullBody = 0
        while (rig.k < bodyUntil + 30) {
            rig.frame(body = rig.k in bodyFrom until bodyUntil)
            maxStep = maxOf(maxStep, abs(rig.shown - previous))
            assertTrue("相邻帧 |ΔhairShown| ≤ 0.4（第 ${rig.k} 帧：$previous → ${rig.shown}）", abs(rig.shown - previous) <= 0.4f)
            if (rig.bodyPresence == 1f) { assertEquals("人体存在度为 1 时不画 A 通道", 0f, rig.shown, 0f); fullBody++ }
            previous = rig.shown
        }
        assertTrue("人体稳定的帧数 $fullBody", fullBody >= 30)
        assertEquals("人体消失后回到满强度", 1f, rig.shown, 0f)
        println("天鹅颈：最大相邻帧强度差 %.3f".format(maxStep))
    }

    @Test fun compensationExtrapolatesTheDisplayPoseBeyondTheNewestFaceSample() {
        // 头部匀速右移（0.4 画面宽 / 秒），每 33 ms 一个人脸样本；被显示帧比最新样本晚 1–2 帧（需要外推）。
        val timeline = FaceTimeline().also { it.aspect = ASPECT }
        val poseMask = FloatArray(FaceTimeline.POSE_SIZE)
        val poseDisplay = FloatArray(FaceTimeline.POSE_SIZE)
        val out = FloatArray(HairMaskTuning.AFFINE_SIZE)
        val reference = FloatArray(FaceTimeline.POSE_SIZE)
        var ts = 0L
        while (ts <= 990L) { timeline.add(ts, faceState(0.3f + 0.0004f * ts, 0.45f)); ts += 33L }
        val newest = timeline.latest
        val latestPose = poseAtOrFail(timeline, newest)
        timeline.update(newest + 40)
        for (ahead in listOf(33L, 66L)) {
            val displayTime = newest + ahead
            assertFalse("poseAt 不外推：旧做法在这一帧退回单位阵", timeline.poseAt(displayTime, reference))
            val raw = timeline.at(displayTime)!!
            assertTrue("外推帧：几何越过最新样本", raw[22] > latestPose[0] + 1e-3f)
            // 遮罩时刻在历史里：补偿生效，显示时刻（外推）的脸中心映回遮罩时刻的脸中心，仿射与外推中心一致。
            val maskStamp = newest - 100
            assertTrue(HairMaskGate.compensate(timeline, maskStamp, raw, displayTime, ASPECT, poseMask, poseDisplay, out))
            val mask = poseAtOrFail(timeline, maskStamp)
            assertMapsTo(out, raw[22], raw[23], mask[0], mask[1])
            // 遮罩比最新人脸样本还新（人脸推理比头发慢）：沿外推轨迹（最新样本 → 显示时刻）按时间取遮罩位姿。
            val newer = newest + 20
            assertTrue(HairMaskGate.compensate(timeline, newer, raw, displayTime, ASPECT, poseMask, poseDisplay, out))
            val g = (newer - newest).toFloat() / ahead
            assertMapsTo(out, raw[22], raw[23], latestPose[0] + g * (raw[22] - latestPose[0]), latestPose[1] + g * (raw[23] - latestPose[1]))
            // 遮罩比显示时刻还新：最多取到显示位姿（单位阵），不再往前外推。
            assertTrue(HairMaskGate.compensate(timeline, displayTime + 30, raw, displayTime, ASPECT, poseMask, poseDisplay, out))
            assertMapsTo(out, raw[22], raw[23], raw[22], raw[23])
        }
        // 历史区间内：显示位姿取 at() 的几何与 poseAt 逐值相同，仿射与「两个 poseAt」一致。
        for (displayTime in listOf(newest - 200, newest - 50, newest)) {
            val raw = timeline.at(displayTime)!!.copyOf()
            assertTrue(HairMaskGate.compensate(timeline, newest - 150, raw, displayTime, ASPECT, poseMask, poseDisplay, out))
            val expected = FloatArray(HairMaskTuning.AFFINE_SIZE)
            assertTrue(HairMaskTuning.affine(poseAtOrFail(timeline, newest - 150), poseAtOrFail(timeline, displayTime), ASPECT, expected))
            assertArrayEquals("displayTime=$displayTime", expected, out, 1e-5f)
        }
        // 显示时刻在历史里、遮罩比最新样本新：按最新样本取遮罩位姿。
        val inside = timeline.at(newest - 50)!!.copyOf()
        assertTrue(HairMaskGate.compensate(timeline, newest + 20, inside, newest - 50, ASPECT, poseMask, poseDisplay, out))
        assertMapsTo(out, inside[22], inside[23], latestPose[0], latestPose[1])
        // 没有几何、没有已上传遮罩、人脸历史为空：单位阵。
        val identity = FloatArray(HairMaskTuning.AFFINE_SIZE).also { HairMaskTuning.identity(it) }
        assertFalse(HairMaskGate.compensate(timeline, newest - 100, null, newest, ASPECT, poseMask, poseDisplay, out))
        assertArrayEquals(identity, out, 0f)
        assertFalse(HairMaskGate.compensate(timeline, Long.MIN_VALUE, inside, newest - 50, ASPECT, poseMask, poseDisplay, out))
        assertArrayEquals(identity, out, 0f)
        timeline.update(newest + FaceTimeline.CLEAR_AFTER_MS + 1)
        assertFalse(HairMaskGate.compensate(timeline, newest, faceState(0.5f, 0.45f), newest, ASPECT, poseMask, poseDisplay, out))
        assertArrayEquals(identity, out, 0f)
    }

    @Test fun compensationDoesNotJumpBetweenAdjacentFramesWhileTheHeadSwings() {
        // 摆头（最大 0.45 画面宽 / 秒），头发 10 Hz、推理 60 ms。逐帧算发缘（随头刚体运动的一点）在显示帧上的错位，统计相邻帧
        // 错位之差（px@720×1280）。两种人脸节奏：
        // - GPU：每帧采样，推理多数 25 ms、每 8 个样本有 1 个 70 ms（单任务在途），画面延后 1 帧，约三成被显示帧越过最新样本；
        // - CPU：45 ms 一采（隔帧），推理 40–85 ms，画面延后 2 帧，遮罩常比最新人脸样本新。
        val poseMask = FloatArray(FaceTimeline.POSE_SIZE)
        val poseDisplay = FloatArray(FaceTimeline.POSE_SIZE)
        val current: SwingMethod = { timeline, maskStamp, raw, displayTime, out ->
            HairMaskGate.compensate(timeline, maskStamp, raw, displayTime, ASPECT, poseMask, poseDisplay, out)
        }
        // 对照（只为确认场景有区分力）：旧做法显示位姿取 poseAt(displayTime)，越过最新样本时整帧退回单位阵；
        // 只把遮罩时刻钳到最新样本、不沿外推轨迹取时，人脸走 CPU 的场景会反向多补。
        val old: SwingMethod = { timeline, maskStamp, _, displayTime, out ->
            HairMaskTuning.affine(if (timeline.poseAt(maskStamp, poseMask)) poseMask else null,
                if (timeline.poseAt(displayTime, poseDisplay)) poseDisplay else null, ASPECT, out)
        }
        val clampOnly: SwingMethod = { timeline, maskStamp, raw, _, out ->
            raw?.copyInto(poseDisplay, 0, HairMaskGate.POSE_INDEX, HairMaskGate.POSE_INDEX + FaceTimeline.POSE_SIZE)
            HairMaskTuning.affine(if (timeline.poseAt(minOf(maskStamp, timeline.latest), poseMask)) poseMask else null,
                if (raw != null) poseDisplay else null, ASPECT, out)
        }
        for (cpu in listOf(false, true)) {
            val now = swing(cpu, current)
            val before = swing(cpu, old)
            val clamped = swing(cpu, clampOnly)
            val limit = if (cpu) MAX_JUMP_P99_CPU_PX else MAX_JUMP_P99_GPU_PX
            println("摆头补偿（%s）：现行 p99 %.1f px、最大 %.1f px、补偿帧 %d/%d、外推帧 %d；旧做法 p99 %.1f px；只钳遮罩时刻 p99 %.1f px".format(
                if (cpu) "CPU" else "GPU", now.p99, now.max, now.compensated, now.frames, now.extrapolated, before.p99, clamped.p99))
            assertTrue("场景里要有足够的外推帧：${now.extrapolated}/${now.frames}", now.extrapolated * 10 >= now.frames)
            assertTrue("旧做法在同一场景超限（场景有区分力）：p99 ${before.p99}", before.p99 > limit)
            if (cpu) assertTrue("只钳遮罩时刻在 CPU 场景超限：p99 ${clamped.p99}", clamped.p99 > limit)
            assertTrue("相邻帧补偿错位跳变 p99 ${now.p99} px ≤ $limit", now.p99 <= limit)
            assertTrue("相邻帧补偿错位跳变最大 ${now.max} px", now.max <= limit + 4.0)
            assertTrue("外推帧也在补偿：${now.compensated}/${now.frames}", now.compensated * 10 >= now.frames * 9)
        }
    }

    /** 摆头仿真结果：相邻帧错位跳变的 p99 / 最大值（px），统计帧数、补偿生效帧数、显示时刻越过最新人脸样本的帧数。 */
    internal class Swing(val p99: Double, val max: Double, val frames: Int, val compensated: Int, val extrapolated: Int)

    /** 摆头仿真（见 compensationDoesNotJumpBetweenAdjacentFramesWhileTheHeadSwings），[method] 求每帧的仿射。 */
    private fun swing(cpu: Boolean, method: SwingMethod): Swing {
        val timeline = FaceTimeline().also { it.aspect = ASPECT }
        val out = FloatArray(HairMaskTuning.AFFINE_SIZE)
        fun centerX(t: Long) = (0.5 + 0.1 * sin(2 * PI * t / 1400.0)).toFloat()
        val hairY = 0.30f // 发缘点（在脸中心上方，随头刚体平移）
        val cpuLatency = longArrayOf(40L, 55L, 70L, 50L, 85L, 45L)
        var facePending = Long.MIN_VALUE
        var faceDone = Long.MIN_VALUE
        var faceLast = Long.MIN_VALUE
        var faceCount = 0
        var hairPending = Long.MIN_VALUE
        var hairDone = Long.MIN_VALUE
        var hairLast = Long.MIN_VALUE
        val masks = ArrayList<Long>()
        val jumps = ArrayList<Double>()
        var previous: DoubleArray? = null
        var frames = 0; var compensated = 0; var extrapolated = 0
        for (k in 0 until 900) {
            val now = frameTime(k)
            // 人脸：单任务在途；GPU 每帧可采，CPU 45 ms 一采。
            if (facePending != Long.MIN_VALUE && faceDone <= now) {
                timeline.add(facePending, faceState(centerX(facePending), 0.45f)); facePending = Long.MIN_VALUE
            }
            if (facePending == Long.MIN_VALUE && (!cpu || faceLast == Long.MIN_VALUE || now - faceLast >= FaceTrackingState.CPU_INTERVAL_MS)) {
                val latency = if (cpu) cpuLatency[faceCount % cpuLatency.size] else if (faceCount % 8 == 7) 70L else 25L
                facePending = now; faceLast = now; faceDone = now + latency; faceCount++
            }
            // 头发：10 Hz、推理 60 ms。
            if (hairPending != Long.MIN_VALUE && hairDone <= now) { masks += hairPending; hairPending = Long.MIN_VALUE }
            if (hairPending == Long.MIN_VALUE && (hairLast == Long.MIN_VALUE || now - hairLast >= HairMaskTuning.INTERVAL_MS)) {
                hairPending = now; hairDone = now + 60L; hairLast = now
            }
            timeline.update(now)
            if (k < 30) continue
            val displayTime = frameTime(k - if (cpu) 2 else 1)
            val raw = timeline.at(displayTime) ?: continue
            // snapshotAt(displayTime)：有效期内与被显示帧最接近的遮罩。
            val maskStamp = masks.filter { now - it in 0..HairMaskTuning.FRESH_MS }.minByOrNull { abs(it - displayTime) } ?: continue
            frames++
            if (displayTime > timeline.latest) extrapolated++
            if (method(timeline, maskStamp, raw, displayTime, out)) compensated++
            // 遮罩里发缘在 P(遮罩时刻)；显示帧上 A 通道把它画在 u：affine(u) = P(遮罩时刻)。错位 = u − P(显示时刻)。
            val x = centerX(maskStamp) - out[2]; val y = hairY - out[6]
            val det = out[0] * out[5] - out[1] * out[4]
            val u = (out[5] * x - out[1] * y) / det
            val v = (-out[4] * x + out[0] * y) / det
            val error = doubleArrayOf((u - centerX(displayTime)) * 720.0, (v - hairY) * 1280.0)
            previous?.let { jumps += hypot(error[0] - it[0], error[1] - it[1]) }
            previous = error
        }
        val sorted = jumps.sorted()
        return Swing(sorted[(sorted.size * 0.99).toInt().coerceAtMost(sorted.size - 1)], sorted.last(), frames, compensated, extrapolated)
    }

    private fun poseAtOrFail(timeline: FaceTimeline, t: Long): FloatArray =
        FloatArray(FaceTimeline.POSE_SIZE).also { assertTrue("poseAt($t)", timeline.poseAt(t, it)) }

    /** 仿射把 (u, v) 映到 (x, y)（容差 1e-4 纹理坐标，720p 下约 0.1 px）。 */
    private fun assertMapsTo(affine: FloatArray, u: Float, v: Float, x: Float, y: Float) {
        assertEquals(x, affine[0] * u + affine[1] * v + affine[2], 1e-4f)
        assertEquals(y, affine[4] * u + affine[5] * v + affine[6], 1e-4f)
    }

    internal companion object {
        const val WIDTH = 216
        const val HEIGHT = 384
        const val ASPECT = 720f / 1280f
        const val FRAME_MS = 100f / 3f
        const val DISPLAY_DELAY_MS = 67L
        /** 约 3 s：推理攒满 p50 窗口，存在度渐入（300 ms）完成。 */
        const val WARMUP_FRAMES = 90
        const val FADE_IN_FRAMES = 30
        /** 丢脸、重新检出时相邻帧强度差上限：faceKeep 渐出 250 ms（每帧约 0.136）、渐入与存在度渐入各 300 ms。 */
        const val MAX_FACE_STEP = 0.25f
        /**
         * 摆头时相邻帧补偿错位跳变 p99 的上限（px@720×1280）。GPU 场景余下的跳变来自 at() 自身的外推误差（与合成的脸椭圆同源），
         * 实测 14.8（旧做法 31.3）；CPU 场景实测 9.2（旧做法 20.3，只钳遮罩时刻 18.5）。
         */
        const val MAX_JUMP_P99_GPU_PX = 16.0
        const val MAX_JUMP_P99_CPU_PX = 12.0
        val CONFIG = VideoEffectsConfig(enabled = true, smoothing = 0.55f, brightness = 0f, filter = "none", skinOnly = true)
        val MASK = PersonMask(ByteArray(64) { 200.toByte() }, 8, 8)
        /** 人体状态（BodyGeometry.toState：髋中心、躯干方向、躯干长、半宽、腿）。 */
        val BODY = floatArrayOf(0.5f, 0.8f, 0f, -1f, 0.3f, 0.1f, 0f)

        /** 30 fps 相机帧的渲染时间（uptime ms，33 / 34 交替）。 */
        fun frameTime(k: Int): Long = 1_000L + k * 100L / 3

        /** 41 维人脸状态（同 FaceTimelineTest）：脸中心在 (cx, cy)，其余部件随之平移，轴向朝右。 */
        fun faceState(cx: Float, cy: Float) = FloatArray(FaceGeometrySmoother.STATE_SIZE).also {
            for (i in 0 until 8) { it[i * 2] = cx + (i % 4 - 1.5f) * 0.06f; it[i * 2 + 1] = cy + (i / 4 - 0.5f) * 0.1f }
            it[16] = cx - 0.08f; it[17] = cy + 0.05f; it[18] = 0.06f
            it[19] = cx + 0.08f; it[20] = cy + 0.05f; it[21] = 0.06f
            it[22] = cx; it[23] = cy; it[24] = 0.22f; it[25] = 0.3f
            it[26] = 1f; it[27] = 0f
            it[28] = cx; it[29] = cy - 0.12f; it[30] = 0.07f; it[31] = 0.03f
            it[32] = cx - 0.08f; it[33] = cy + 0.05f; it[34] = 0.05f; it[35] = 0.025f
            it[36] = cx + 0.08f; it[37] = cy + 0.05f; it[38] = 0.05f; it[39] = 0.025f
            it[40] = 1f
        }
    }
}
