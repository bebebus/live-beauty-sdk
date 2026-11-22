package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer

/**
 * 延迟回读（DEFERRED）的取票契约：画小图那一帧 due 只判到期、不占单任务在途；下一帧读回时 begin(采样时间, expectEpoch)
 * 在同一把锁内核对代际再取票。四个跟踪状态（人脸、美体、人像分割、头发分割）规则相同。
 */
class TrackingStateDeferredTest {
    private val w = 216
    private val h = 384

    @Test fun faceDueNeitherTakesTicketNorTouchesSamplingState() {
        val state = FaceTrackingState().apply { enable(true) }
        val epoch = checkNotNull(state.due(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        // 再判一次结果不变：due 不记采样时间、不置忙。
        assertEquals(epoch, state.due(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        val ticket = checkNotNull(state.begin(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = epoch))
        assertEquals(epoch, ticket.epoch)
        assertEquals(100L, ticket.timestamp)
        // 推理在途：due 不看忙（下一帧照样画小图），begin 仍受单任务在途约束。
        assertEquals(epoch, state.due(133, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        assertNull(state.begin(133, w, h, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = epoch))
        assertTrue(state.complete(ticket, null, 120))
        assertNotNull(state.begin(133, w, h, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = epoch))
        // 已提交过的采样时间及之前、限频间隔内都不到期。
        assertNull(state.due(133, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        assertNull(state.due(150, w, h, FaceTrackingState.CPU_INTERVAL_MS))
        // 停用时不到期。
        state.enable(false)
        assertNull(state.due(200, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
    }

    @Test fun faceSnapshotReportsInFlightOnlyBetweenBeginAndComplete() {
        // 快照的 inFlight（HairMaskTuning.faceKeep 据此只在停摆时用兜底）：取票之后、complete 之前为 true，与结果同一把锁取出。
        val state = FaceTrackingState().apply { enable(true) }
        assertFalse(state.snapshot(100).inFlight)
        // DEFERRED 只画小图（due）、还没取票：不算在途。
        val epoch = checkNotNull(state.due(100, w, h, FaceTrackingState.CPU_INTERVAL_MS))
        assertFalse(state.snapshot(100).inFlight)
        val ticket = checkNotNull(state.begin(100, w, h, FaceTrackingState.CPU_INTERVAL_MS, expectEpoch = epoch))
        assertTrue(state.snapshot(133).inFlight)
        assertTrue(state.snapshot(400).inFlight) // 推理再慢也一直在途，直到交回
        // 交回：同一次快照里既看到结果（completed 前进），也不再在途。
        assertTrue(state.complete(ticket, null, 300))
        state.snapshot(333).let { assertFalse(it.inFlight); assertEquals(100L, it.completed) }
        // 切镜头（reset）之后旧代际的票交回之前仍算在途（它的结果会被丢弃）；交回后不在途。
        val old = checkNotNull(state.begin(400, w, h, FaceTrackingState.CPU_INTERVAL_MS))
        state.reset()
        assertTrue(state.snapshot(410).inFlight)
        assertFalse(state.complete(old, null, 420))
        assertFalse(state.snapshot(420).inFlight)
        // 取票被拒（代际不符、限频）不置在途。
        assertNull(state.begin(430, w, h, FaceTrackingState.CPU_INTERVAL_MS, expectEpoch = old.epoch))
        assertFalse(state.snapshot(430).inFlight)
    }

    @Test fun faceBeginRejectsStaleEpochWithoutBecomingBusy() {
        val state = FaceTrackingState().apply { enable(true) }
        val epoch = checkNotNull(state.due(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        state.reset() // 画小图之后、读回之前切镜头
        assertNull(state.begin(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = epoch))
        // 没有取票就不占单任务在途，也没有记采样时间：新代际同一时刻照常取票。
        val fresh = checkNotNull(state.begin(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        assertEquals(epoch + 1, fresh.epoch)
    }

    @Test fun faceDueKeepsDimensionResetSemantics() {
        val state = FaceTrackingState().apply { enable(true) }
        val ticket = checkNotNull(state.begin(100, w, h, FaceTrackingState.FRAME_INTERVAL_MS))
        // 尺寸变化（切镜头、旋转）在 due 里同样 reset：代际当帧变化，在途的旧尺寸结果不被接受。
        val epoch = checkNotNull(state.due(133, h, w, FaceTrackingState.FRAME_INTERVAL_MS))
        assertNotEquals(ticket.epoch, epoch)
        assertEquals(epoch, state.snapshot(133).epoch)
        assertFalse(state.complete(ticket, null, 140))
        assertNotNull(state.begin(133, h, w, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = epoch))
    }

    @Test fun bodyAndMaskFollowTheSameContract() {
        val body = BodyTrackingState().apply { enable(true) }
        val bodyEpoch = checkNotNull(body.due(100, w, h, VideoEffectsFilter.BODY_INTERVAL_MS))
        val bodyTicket = checkNotNull(body.begin(100, w, h, VideoEffectsFilter.BODY_INTERVAL_MS, expectEpoch = bodyEpoch))
        assertEquals(bodyEpoch, body.due(200, w, h, VideoEffectsFilter.BODY_INTERVAL_MS)) // 在途也到期
        assertNull(body.begin(200, w, h, VideoEffectsFilter.BODY_INTERVAL_MS, expectEpoch = bodyEpoch))
        assertNull(body.due(133, w, h, VideoEffectsFilter.BODY_INTERVAL_MS)) // 55 ms 限频
        body.complete(bodyTicket, null, 130)
        body.reset()
        assertNull(body.begin(200, w, h, VideoEffectsFilter.BODY_INTERVAL_MS, expectEpoch = bodyEpoch))
        assertNotNull(body.begin(200, w, h, VideoEffectsFilter.BODY_INTERVAL_MS))

        val mask = MaskTrackingState().apply { enable(true) }
        val maskEpoch = checkNotNull(mask.due(100, w, h, VideoEffectsFilter.SEG_INTERVAL_MS))
        val maskTicket = checkNotNull(mask.begin(100, w, h, VideoEffectsFilter.SEG_INTERVAL_MS, expectEpoch = maskEpoch))
        assertEquals(maskEpoch, mask.due(133, w, h, VideoEffectsFilter.SEG_INTERVAL_MS))
        assertNull(mask.begin(133, w, h, VideoEffectsFilter.SEG_INTERVAL_MS, expectEpoch = maskEpoch))
        mask.complete(maskTicket, null, 120)
        val resized = checkNotNull(mask.due(133, h, w, VideoEffectsFilter.SEG_INTERVAL_MS))
        assertNotEquals(maskEpoch, resized)
        assertNull(mask.begin(133, h, w, VideoEffectsFilter.SEG_INTERVAL_MS, expectEpoch = maskEpoch))
        assertNotNull(mask.begin(133, h, w, VideoEffectsFilter.SEG_INTERVAL_MS, expectEpoch = resized))
    }

    @Test fun hairMaskFollowsTheSameContractAndDropsStaleEpochs() {
        val hair = MaskTrackingState(HairMaskTuning.FRESH_MS).apply { enable(true) }
        val interval = HairMaskTuning.interval(Float.NaN, backgroundActive = false)
        assertEquals(100L, interval)
        val epoch = checkNotNull(hair.due(100, w, h, interval))
        assertEquals(epoch, hair.due(100, w, h, interval)) // due 不记采样时间、不置忙
        val ticket = checkNotNull(hair.begin(100, w, h, interval, expectEpoch = epoch))
        assertEquals(100L, ticket.timestamp)
        assertNull(hair.due(166, w, h, interval)) // 100 ms 限频
        assertEquals(epoch, hair.due(200, w, h, interval)) // 推理在途也到期（照样画小图）
        assertNull(hair.begin(200, w, h, interval, expectEpoch = epoch)) // 单任务在途
        val mask = PersonMask(ByteArray(w * h), w, h)
        assertTrue(hair.complete(ticket, mask, 160))
        assertSame(mask, hair.snapshotAt(100, 700).mask) // 头发有效期 650 ms
        // 画小图之后、读回之前切镜头：代际不符，不取票；迟到的旧代际结果被丢弃、不进历史。
        val staged = checkNotNull(hair.due(300, w, h, interval))
        val inflight = checkNotNull(hair.begin(300, w, h, interval, expectEpoch = staged))
        val next = checkNotNull(hair.due(400, w, h, interval))
        hair.reset()
        assertNull(hair.begin(400, w, h, interval, expectEpoch = next))
        assertFalse(hair.complete(inflight, mask, 420))
        assertNull(hair.snapshotAt(300, 420).mask)
        // 新代际照常往返。
        val fresh = checkNotNull(hair.due(500, w, h, interval))
        assertEquals(next + 1, fresh)
        assertNotNull(hair.begin(500, w, h, interval, expectEpoch = fresh))
    }

    @Test fun hairConsumerLeasesItsOwnDoubleBufferWithoutDisturbingOthers() {
        val consumers = listOf(FaceTextureSampler.FACE, FaceTextureSampler.BODY, FaceTextureSampler.SEGMENT, FaceTextureSampler.HAIR)
        assertEquals(listOf(0, 1, 2, 3), consumers)
        val sampler = FaceTextureSampler(384)
        val sw = 4; val sh = 2
        val staging = ByteBuffer.allocateDirect(sw * sh * 4)
        for (i in 0 until sw * sh * 4) staging.put(i, i.toByte())
        sampler.stageForTest(staging, sw, sh)
        val face1 = sampler.lease(FaceTextureSampler.FACE)
        val seg1 = sampler.lease(FaceTextureSampler.SEGMENT)
        val hair1 = sampler.lease(FaceTextureSampler.HAIR)
        val hair2 = sampler.lease(FaceTextureSampler.HAIR)
        val body1 = sampler.lease(FaceTextureSampler.BODY)
        val face2 = sampler.lease(FaceTextureSampler.FACE)
        // 每路各一对缓冲：头发的两块与其他各路的缓冲都不是同一块，且都是暂存区的完整拷贝、尺寸相同。
        val all = listOf(face1, face2, seg1, body1, hair1, hair2)
        for (a in all.indices) for (b in a + 1 until all.size) assertNotSame("$a/$b", all[a].pixels, all[b].pixels)
        for (sample in all) {
            assertEquals(sw, sample.width); assertEquals(sh, sample.height)
            assertEquals(0, sample.pixels.position())
            for (i in 0 until sw * sh * 4) assertEquals(i.toByte(), sample.pixels.get(i))
        }
        // 头发连租两次不推进人脸的轮转：人脸第三次租回第一块，头发第三次租回第一块。
        assertSame(face1.pixels, sampler.lease(FaceTextureSampler.FACE).pixels)
        assertSame(hair1.pixels, sampler.lease(FaceTextureSampler.HAIR).pixels)
        assertNotSame(seg1.pixels, sampler.lease(FaceTextureSampler.SEGMENT).pixels)
        assertSame(seg1.pixels, sampler.lease(FaceTextureSampler.SEGMENT).pixels)
        // worker 改写头发那块不影响别的路。
        hair1.pixels.put(0, 99)
        assertEquals(0.toByte(), face1.pixels.get(0))
        assertEquals(0.toByte(), seg1.pixels.get(0))
        assertEquals(0.toByte(), staging.get(0))
    }

    /**
     * 按 VideoEffectsFilter 的 DEFERRED 流程（第 0' 段 begin(采样时间, expectEpoch) 取票、交给 worker；第 1 段 due 记录；
     * 第 1' 段画小图）跑 [frames] 帧，推理 [latencyMs] 后完成；返回取到票（读回并提交）的帧数。
     */
    private fun deferredSubmits(frames: Int, latencyMs: Long, due: (Long) -> Long?, begin: (Long, Long) -> Any?,
                                complete: (Any) -> Unit): Int {
        var inflight: Any? = null
        var done = 0L
        var stagedAt = Long.MIN_VALUE
        var stagedEpoch: Long? = null
        var submits = 0
        for (k in 0 until frames) {
            val now = PresenceExitJointTest.frameTime(k)
            inflight?.let { if (done <= now) { complete(it); inflight = null } }
            stagedEpoch?.let { e -> begin(stagedAt, e) }?.let { inflight = it; done = now + latencyMs; submits++ }
            stagedEpoch = due(now)
            stagedAt = now
        }
        return submits
    }

    @Test fun deferredFlowStillSamplesGpuFaceAndSegmentationEveryFrame() {
        // 推理在一帧内完成：以前画小图那一帧就取票、下一帧才交给 worker，单任务在途多占一帧，人脸与分割都只有约 15 Hz。
        val frames = 300
        val face = FaceTrackingState().apply { enable(true) }
        val faceSubmits = deferredSubmits(frames, 25L,
            { now -> face.due(now, w, h, FaceTrackingState.FRAME_INTERVAL_MS) },
            { at, e -> face.begin(at, w, h, FaceTrackingState.FRAME_INTERVAL_MS, expectEpoch = e) },
            { face.complete(it as FaceTrackingState.Ticket, null, 0) })
        assertTrue("人脸提交 $faceSubmits / ${frames - 1} 帧", faceSubmits >= frames - 2)
        val mask = MaskTrackingState().apply { enable(true) }
        val maskSubmits = deferredSubmits(frames, 25L,
            { now -> mask.due(now, w, h, VideoEffectsFilter.SEG_INTERVAL_MS) },
            { at, e -> mask.begin(at, w, h, VideoEffectsFilter.SEG_INTERVAL_MS, expectEpoch = e) },
            { mask.complete(it as MaskTrackingState.Ticket, null, 0) })
        assertTrue("分割提交 $maskSubmits / ${frames - 1} 帧", maskSubmits >= frames - 2)
        // 美体按 55 ms 限频本来就是隔帧采（约 15 Hz），流程改动不改变它。
        val body = BodyTrackingState().apply { enable(true) }
        val bodySubmits = deferredSubmits(frames, 25L,
            { now -> body.due(now, w, h, VideoEffectsFilter.BODY_INTERVAL_MS) },
            { at, e -> body.begin(at, w, h, VideoEffectsFilter.BODY_INTERVAL_MS, expectEpoch = e) },
            { body.complete(it as BodyTrackingState.Ticket, null, 0) })
        assertEquals(frames / 2.0, bodySubmits.toDouble(), 2.0)
        // 头发分割：推理 58 ms（Pixel 6 CPU 量级）时间隔 2 × 58 = 116 ms（worker 占空 ≤ 1/2），落到 30 fps 帧上每 4 帧一次，约 7.5 Hz；
        // 推理 ≤ 50 ms 的机型仍是 100 ms 基础间隔，每 3 帧一次，约 10 Hz。
        for ((latency, every) in listOf(58L to 4, 45L to 3)) {
            val hair = MaskTrackingState(HairMaskTuning.FRESH_MS).apply { enable(true) }
            val hairInterval = HairMaskTuning.interval(latency.toFloat(), backgroundActive = false)
            val hairSubmits = deferredSubmits(frames, latency,
                { now -> hair.due(now, w, h, hairInterval) },
                { at, e -> hair.begin(at, w, h, hairInterval, expectEpoch = e) },
                { hair.complete(it as MaskTrackingState.Ticket, null, 0) })
            assertEquals("推理 $latency ms", frames / every.toDouble(), hairSubmits.toDouble(), 2.0)
        }
    }
}
