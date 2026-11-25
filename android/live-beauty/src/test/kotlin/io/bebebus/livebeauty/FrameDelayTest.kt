package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class FrameDelayTest {
    /**
     * 每 33 ms 一帧（第 k 帧采样时间 k*33），每 [every] 帧采一次，采样 j 在 j*33 + [latency] 完成。
     * [lost] 为 true 的帧传 MIN_VALUE（还没有任何结果）。
     */
    private class Feed(val schedule: DelaySchedule = DelaySchedule()) {
        var k = 0
        val lock get() = schedule.lock

        fun stamp(frame: Int) = 1_000_000_000L + frame * 33_333_333L

        fun frames(n: Int, latency: Long, streaming: Boolean, every: Int = 1, lost: Boolean = false, stall: Long = -1L) {
            repeat(n) {
                val now = k * 33L
                var completed = Long.MIN_VALUE
                if (!lost) {
                    var j = k
                    while (j >= 0) { if (j % every == 0 && j * 33L + latency <= now) { completed = j * 33L; break }; j-- }
                }
                if (stall >= 0L) lock.onStall(stall)
                schedule.stored(schedule.next(), now, stamp(k))
                schedule.select(completed, streaming, now)
                k++
            }
        }
    }

    /** 预览中按推理延迟测量并锁定的延后。 */
    private fun lockedInPreview(latency: Long, every: Int = 1): Int {
        val feed = Feed()
        feed.lock.engage(PacingLock.PRIOR_GPU)
        feed.frames(120, latency, streaming = false, every = every)
        assertTrue(feed.lock.locked)
        return feed.schedule.delay
    }

    @Test fun delayMatchesInferenceLatency() {
        assertEquals(1, lockedInPreview(latency = 15)) // GPU：结果下一帧就到
        assertEquals(2, lockedInPreview(latency = 50))
        assertEquals(3, lockedInPreview(latency = 50, every = 2)) // CPU 隔帧采样
        assertEquals(DelaySchedule.MAX_DELAY, lockedInPreview(latency = 400)) // 太慢：封顶，靠外推
    }

    @Test fun priorDelayIsUsedBeforeMeasurementAndStampFollowsSelectedFrame() {
        val feed = Feed()
        feed.lock.engage(2)
        // 还没有任何结果：不计入测量，按先验延后；缓冲攒满前取能取到的最老一帧。
        for (n in 0 until 10) {
            feed.frames(1, latency = 0, streaming = false, lost = true)
            assertEquals(minOf(2, n), feed.schedule.delay)
            assertEquals(feed.stamp(feed.k - 1 - feed.schedule.delay), feed.schedule.stampNs)
            assertEquals((feed.k - 1 - feed.schedule.delay) * 33L, feed.schedule.time)
        }
        assertFalse(feed.lock.locked)
    }

    @Test fun previewMeasurementLocksOnceAfterWarmup() {
        val feed = Feed()
        feed.lock.engage(PacingLock.PRIOR_GPU)
        var changes = 0
        var last = feed.lock.delay
        for (n in 0 until 300) {
            feed.frames(1, latency = 15, streaming = false)
            if (feed.lock.delay != last) { changes++; last = feed.lock.delay }
            // 预热 15 + 测量 45 个有效样本之前不锁定（第 0 帧还没有结果）。
            if (n < PacingLock.WARMUP + PacingLock.MEASURE) assertFalse("n=$n", feed.lock.locked)
        }
        assertTrue(feed.lock.locked)
        assertEquals(1, feed.lock.delay)
        assertEquals(1, changes)
    }

    @Test fun streamingLockNeverShrinksBelowPrior() {
        val fast = Feed()
        fast.lock.engage(2)
        fast.frames(120, latency = 15, streaming = true) // 测得 1
        assertEquals(2, fast.lock.delay)
        val slow = Feed()
        slow.lock.engage(2)
        slow.frames(120, latency = 80, streaming = true) // 测得 3
        assertEquals(3, slow.lock.delay)
    }

    @Test fun lostFaceDoesNotChangeTheLockedDelay() {
        val feed = Feed()
        feed.lock.engage(2)
        feed.frames(120, latency = 15, streaming = false)
        assertEquals(1, feed.lock.delay)
        // 开播后 60 s 都没有结果（遮挡、代际变化后还没出结果）：延后不变，不重复也不跳帧。
        feed.frames(1800, latency = 15, streaming = true, lost = true)
        assertEquals(1, feed.lock.delay)
        assertEquals(1, feed.schedule.delay)
    }

    @Test fun singleSlowInferenceDoesNotRepeatAFrame() {
        val feed = Feed()
        feed.lock.engage(2)
        feed.frames(120, latency = 15, streaming = false)
        val schedule = feed.schedule
        // 开播；结果通常下一帧就到，偶尔一次慢了两帧。
        for (n in 0 until 600) {
            val now = feed.k * 33L
            val completed = if (n % 100 == 50) (feed.k - 3) * 33L else (feed.k - 1) * 33L
            schedule.stored(schedule.next(), now, feed.stamp(feed.k)); schedule.select(completed, true, now)
            feed.k++
            assertEquals("n=$n", 1, schedule.delay)
        }
    }

    @Test fun streamingGrowsOneFrameAtATimeWithCooldownAndNeverShrinks() {
        val feed = Feed()
        feed.lock.engage(2)
        feed.frames(120, latency = 15, streaming = true)
        assertEquals(2, feed.lock.delay)
        // 推理突然慢到整个缓冲都等不到（want = SLOTS）：p90 ≥ 延后 + 2 只加 1 帧。
        var grows = 0
        var last = feed.lock.delay
        var firstGrowAt = -1
        var secondGrowAt = -1
        for (n in 0 until 2400) {
            feed.frames(1, latency = 400, streaming = true)
            if (feed.lock.delay != last) {
                assertEquals(last + 1, feed.lock.delay)
                grows++; last = feed.lock.delay
                if (firstGrowAt < 0) firstGrowAt = n else if (secondGrowAt < 0) secondGrowAt = n
            }
        }
        assertEquals(2, grows)
        assertEquals(DelaySchedule.MAX_DELAY, feed.lock.delay)
        // 两次加帧至少间隔 30 s（每帧 33 ms）。
        assertTrue((secondGrowAt - firstGrowAt) * 33L >= PacingLock.GROW_COOLDOWN_MS)
        // 推理恢复：直播中不缩。
        feed.frames(3000, latency = 15, streaming = true)
        assertEquals(DelaySchedule.MAX_DELAY, feed.lock.delay)
        // 停播回到预览：重新测量，可以缩。
        feed.frames(120, latency = 15, streaming = false)
        assertEquals(1, feed.lock.delay)
    }

    @Test fun readbackStallSwitchesToDeferredAndAddsOneFrame() {
        val stalled = Feed()
        stalled.lock.engage(2)
        stalled.frames(120, latency = 15, streaming = false, stall = 4_000_000L)
        assertTrue(stalled.lock.deferredReadback)
        assertEquals(2, stalled.lock.delay)
        // 阻塞不超过 3 ms：保持当帧回读。
        val quick = Feed()
        quick.lock.engage(2)
        quick.frames(120, latency = 15, streaming = false, stall = 2_000_000L)
        assertFalse(quick.lock.deferredReadback)
        assertEquals(1, quick.lock.delay)
        // 会话内不再切回：停播后重新测量仍是延迟回读，且不再额外 +1（测得的 want 已含那一帧）。
        stalled.frames(60, latency = 15, streaming = true, stall = 0L)
        stalled.frames(120, latency = 48, streaming = false, stall = 0L)
        assertTrue(stalled.lock.deferredReadback)
        assertEquals(2, stalled.lock.delay)
    }

    @Test fun wantIgnoresPartialBufferAndReportsBeyondBuffer() {
        val schedule = DelaySchedule()
        assertEquals(-1, schedule.want(Long.MIN_VALUE))
        for (k in 0 until 3) schedule.stored(schedule.next(), 100L + k * 33L, 0L)
        assertEquals(-1, schedule.want(50L)) // 缓冲未满且退不到：不计入测量
        assertEquals(1, schedule.want(133L))
        for (k in 3 until 5) schedule.stored(schedule.next(), 100L + k * 33L, 0L)
        assertEquals(DelaySchedule.SLOTS, schedule.want(50L)) // 缓冲已满仍退不到
        assertEquals(0, schedule.want(Long.MAX_VALUE))
    }

    @Test fun neverSelectsMoreFramesThanStoredAndReportsTheSelectedTime() {
        val schedule = DelaySchedule()
        schedule.lock.engage(DelaySchedule.MAX_DELAY)
        for (k in 0 until 8) {
            schedule.stored(schedule.next(), k * 33L, k.toLong()); schedule.select(-1, false, k * 33L) // 结果远远落后
            assertTrue(schedule.delay <= schedule.count - 1)
            assertEquals((k - schedule.delay) * 33L, schedule.time)
            assertEquals((k - schedule.delay).toLong(), schedule.stampNs)
        }
        assertEquals(DelaySchedule.MAX_DELAY, schedule.delay)
    }

    @Test fun bufferResetKeepsTheLockAndThereIsNoClearEntry() {
        val feed = Feed()
        feed.lock.engage(2)
        feed.frames(120, latency = 50, streaming = true)
        assertTrue(feed.lock.locked)
        val locked = feed.lock.delay
        // 尺寸变化、上下文重建：缓冲从头攒，锁定的延后保留。
        feed.schedule.resetBuffer()
        assertEquals(0, feed.schedule.count)
        assertTrue(feed.lock.locked)
        assertEquals(locked, feed.lock.delay)
        // 代际变化、对比原画、负载降级都不再有清空入口。
        assertTrue(DelaySchedule::class.java.declaredMethods.none { it.name == "clear" })
        assertTrue(PacingLock::class.java.declaredMethods.none { it.name == "clear" })
        // 会话结束才解除。
        feed.lock.disengage()
        assertFalse(feed.lock.engaged)
        assertEquals(0, feed.lock.delay)
    }

    @Test fun disengagedScheduleDoesNotDelay() {
        val schedule = DelaySchedule()
        for (k in 0 until 10) { schedule.stored(schedule.next(), k * 33L, 0L); schedule.select(0L, true, k * 33L) }
        assertEquals(0, schedule.delay)
        assertFalse(schedule.lock.locked)
    }

    @Test fun previewAdaptsUpwardWhenInferenceSlowsDownAfterLock() {
        val feed = Feed()
        feed.lock.engage(PacingLock.PRIOR_GPU)
        // 初始静止预览快速推理锁定 delay=1。
        feed.frames(120, latency = 15, streaming = false)
        assertTrue(feed.lock.locked)
        assertEquals(1, feed.lock.delay)
        // 晃动 / CPU 回退导致推理持续升至 55 ms（want=2）：预览态在 ~30 帧内自动上调到 2 保持零滞后对齐。
        feed.frames(45, latency = 55, streaming = false)
        assertEquals(2, feed.lock.delay)
    }
}
