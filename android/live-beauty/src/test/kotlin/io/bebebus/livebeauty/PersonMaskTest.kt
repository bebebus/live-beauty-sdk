package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import java.nio.BufferUnderflowException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Random

class PersonMaskTest {
    @Test fun confidenceIsQuantizedAndClamped() {
        val mask = PersonMask.fromConfidence(floatArrayOf(0f, 0.5f, 1f, 1.7f, -0.2f, Float.NaN), 3, 2)
        assertEquals(listOf(0, 128, 255, 255, 0, 0), mask.pixels.map { it.toInt() and 0xFF })
        try { PersonMask(ByteArray(5), 3, 2); fail() } catch (_: IllegalArgumentException) {}
    }

    @Test fun trackingStateThrottlesAndKeepsMasksLongerThanFaces() {
        val state = MaskTrackingState()
        assertNull(state.begin(1000, 720, 1280))
        state.enable(true)
        val ticket = state.begin(1000, 720, 1280)!!
        assertNull(state.begin(1050, 720, 1280)) // 单任务
        val mask = PersonMask(ByteArray(4), 2, 2)
        assertTrue(state.complete(ticket, mask, 1080))
        assertSame(mask, state.snapshot(1200).mask) // 200 ms 内有效
        assertNull(state.snapshot(1201).mask)
        // 关掉再开：代际变化，迟到结果作废。
        val late = state.begin(1200, 720, 1280)!!
        state.enable(false); state.enable(true)
        assertFalse(state.complete(late, mask, 1250))
    }

    @Test fun presenceFadesInHoldsThenFadesOut() {
        val p = MaskPresence()
        assertEquals(0f, p.update(true, 0), 0f)
        assertEquals(0.5f, p.update(true, 100), 1e-6f)
        assertEquals(1f, p.update(true, 200), 1e-6f)
        // 没有遮罩：保持 100 ms，再 200 ms 渐出。
        assertEquals(1f, p.update(false, 300), 1e-6f)
        assertEquals(0.5f, p.update(false, 400), 1e-6f)
        assertEquals(0f, p.update(false, 500), 0f)
        // 遮罩回来：照常 200 ms 渐入。
        assertEquals(0.5f, p.update(true, 600), 1e-6f)
    }

    @Test fun hairStateKeepsMasksForItsOwnFreshWindow() {
        // 头发分割约 10 Hz、推理更慢：有效期 650 ms。500 ms 前的结果仍收下、仍可取；700 ms 前的结果作废并清空历史。
        val state = MaskTrackingState(freshMs = 650L)
        assertEquals(650L, state.freshMs)
        state.enable(true)
        val mask = PersonMask(ByteArray(4), 2, 2)
        val t1 = state.begin(1000, 216, 384, 100L)!!
        assertTrue(state.complete(t1, mask, 1500))
        assertSame(mask, state.snapshotAt(1000, 1500).mask)
        assertSame(mask, state.snapshotAt(1600, 1650).mask) // 650 ms 内有效
        assertNull(state.snapshotAt(1600, 1651).mask)
        val t2 = state.begin(1700, 216, 384, 100L)!!
        assertTrue(state.complete(t2, mask, 2400)) // 迟到 700 ms：代际没变，照常结票，但结果作废
        assertNull(state.snapshotAt(1700, 2400).mask)
        assertEquals(1700L, state.snapshotAt(1700, 2400).completed)
    }

    @Test fun defaultTrackingStateKeepsSelfieWindowUnchanged() {
        // 人像分割不传参：仍是 200 ms，完成与取用两处都按它判断（与参数化之前逐值相同）。
        val state = MaskTrackingState()
        assertEquals(MaskTrackingState.FRESH_MS, state.freshMs)
        assertEquals(200L, MaskTrackingState.FRESH_MS)
        state.enable(true)
        val mask = PersonMask(ByteArray(4), 2, 2)
        val t1 = state.begin(1000, 720, 1280)!!
        assertTrue(state.complete(t1, mask, 1200)) // 恰好 200 ms：收下
        assertSame(mask, state.snapshotAt(1000, 1200).mask)
        assertNull(state.snapshotAt(1000, 1201).mask)
        val t2 = state.begin(1300, 720, 1280)!!
        assertTrue(state.complete(t2, mask, 1501)) // 201 ms：作废并清空历史
        assertNull(state.snapshotAt(1300, 1501).mask)
    }

    @Test fun parameterizedPresenceUsesItsOwnFadeHoldAndFadeOut() {
        // 头发遮罩：渐入 300、保持 300、渐出 400 ms。
        val p = MaskPresence(300f, 300f, 400f)
        assertEquals(0f, p.update(true, 0), 0f)
        assertEquals(1f / 3f, p.update(true, 100), 1e-6f)
        assertEquals(2f / 3f, p.update(true, 200), 1e-6f)
        assertEquals(1f, p.update(true, 300), 1e-6f)
        for (t in 400L..600L step 100) assertEquals("t=$t 保持", 1f, p.update(false, t), 1e-6f)
        assertEquals(0.75f, p.update(false, 700), 1e-6f)
        assertEquals(0.5f, p.update(false, 800), 1e-6f)
        assertEquals(0.25f, p.update(false, 900), 1e-6f)
        assertEquals(0f, p.update(false, 1000), 0f)
        assertEquals(1f / 3f, p.update(true, 1100), 1e-6f)
    }

    @Test fun defaultPresenceMatchesSelfieTimeline() {
        // 不传参与显式传入旧常量（200 / 100 / 200）在同一串有 / 无遮罩、不等间隔的更新上逐值相同。
        assertEquals(200f, MaskPresence.FADE_IN_MS, 0f)
        assertEquals(100f, MaskPresence.HOLD_MS, 0f)
        assertEquals(200f, MaskPresence.FADE_OUT_MS, 0f)
        val a = MaskPresence()
        val b = MaskPresence(200f, 100f, 200f)
        var t = 0L
        for (k in 0 until 200) {
            t += 10L + (k * 37) % 50
            val has = (k / 7) % 3 != 2
            assertEquals("k=$k", b.update(has, t), a.update(has, t), 0f)
        }
    }

    @Test fun bufferQuantizationMatchesTheArrayPathByteForByte() {
        // 216×384 随机置信度，混入 NaN、±无穷与越界值：直接读 FloatBuffer（堆上 / 本机字节序的直接缓冲视图）与先拷成
        // FloatArray 再量化逐字节相同；写进给定的旧遮罩时返回它本身、整块覆盖旧内容；不改 position。
        val w = 216; val h = 384
        val random = Random(7)
        val specials = floatArrayOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.3f, 1.4f, 0f, 1f, 0.5f)
        repeat(5) {
            val values = FloatArray(w * h) { if (random.nextInt(9) == 0) specials[random.nextInt(specials.size)] else random.nextFloat() }
            val expected = PersonMask.fromConfidence(values, w, h).pixels
            val heap = FloatBuffer.wrap(values).also { it.position(123) }
            val direct = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(values)
            assertArrayEquals(expected, PersonMask.fromConfidence(heap, w, h).pixels)
            assertEquals(123, heap.position())
            val reused = PersonMask(ByteArray(w * h) { 99 }, w, h)
            assertSame(reused, PersonMask.fromConfidence(direct, w, h, reused))
            assertArrayEquals(expected, reused.pixels)
        }
        // limit 不足时与整块 get 一样抛 BufferUnderflowException，不读越界；尺寸不符的旧遮罩不能写。
        try { PersonMask.fromConfidence(FloatBuffer.allocate(5), 3, 2); fail() } catch (_: BufferUnderflowException) {}
        try { PersonMask.fromConfidence(FloatBuffer.allocate(6), 3, 2, PersonMask(ByteArray(6), 2, 3)); fail() }
        catch (_: IllegalArgumentException) {}
    }

    private fun recycling() = MaskTrackingState(650L, spares = 3).apply { enable(true) }

    private fun MaskTrackingState.push(at: Long, mask: PersonMask) = assertTrue(complete(checkNotNull(begin(at, 216, 384, 0L)), mask, at))

    @Test fun recyclingHandsBackOnlyMasksThatLeftTheHistory() {
        val state = recycling()
        val m = List(6) { PersonMask(ByteArray(4), 2, 2) }
        for (k in 0 until 4) state.push(1000L + 100 * k, m[k])
        assertNull("历史里的遮罩都不出池", state.obtain(2, 2, null))
        state.push(1400, m[4]) // 挤出 m0
        assertSame(m[0], state.obtain(2, 2, null))
        assertNull("同一块只出池一次", state.obtain(2, 2, null))
        // 尺寸不符的顺手丢弃，不交出去。
        state.push(1500, m[5]) // 挤出 m1
        assertNull(state.obtain(4, 4, null))
        assertNull(state.obtain(2, 2, null))
        // 过期作废、停用前的代际作废也进池；keep（EMA 上一帧）不出池。
        val late = PersonMask(ByteArray(4), 2, 2)
        val ticket = checkNotNull(state.begin(1600, 216, 384, 0L))
        assertTrue(state.complete(ticket, late, 1600 + 651)) // 迟到：清空历史 m2..m5，late 也不进历史
        assertNull(state.snapshotAt(1600, 2251).mask)
        // 池最多留 3 块（最早进池的 m2、m3 已丢弃）：m4、m5 可取，late 是 keep 不交出；不再 keep 时才交出。
        assertSame(m[4], state.obtain(2, 2, keep = late))
        assertSame(m[5], state.obtain(2, 2, keep = late))
        assertNull(state.obtain(2, 2, keep = late))
        assertSame(late, state.obtain(2, 2, keep = null))
        assertNull(state.obtain(2, 2, keep = null))
        // 代际作废（推理期间 reset、切镜头）的结果没进历史，同样进池。
        val stale = checkNotNull(state.begin(2300, 216, 384, 0L))
        state.reset()
        val orphan = PersonMask(ByteArray(4), 2, 2)
        assertFalse(state.complete(stale, orphan, 2300))
        assertSame(orphan, state.obtain(2, 2, keep = null))
    }

    @Test fun recyclingNeverHandsOutWhatGlTookInItsLastTwoSnapshots() {
        val state = recycling()
        val m = List(7) { PersonMask(ByteArray(4), 2, 2) }
        for (k in 0 until 4) state.push(1000L + 100 * k, m[k])
        // GL 线程：本帧 snapshot(now) 拿到 m3，按显示时间拿到最旧的 m0，正要上传。
        assertSame(m[3], state.snapshot(1300).mask)
        assertSame(m[0], state.snapshotAt(990, 1300).mask)
        state.push(1400, m[4]) // m0 被挤出历史
        assertNull("GL 手上的 m0 不出池", state.obtain(2, 2, null))
        state.push(1500, m[5]) // m1 被挤出
        assertSame(m[1], state.obtain(2, 2, null))
        // 下一帧 GL 又取了两次（上一帧的上传早已完成）：m0 才可再用。
        state.snapshot(1500); state.snapshotAt(1490, 1500)
        assertSame(m[0], state.obtain(2, 2, null))
        // 取到之后、上传之前又插了一次 snapshot（例如再查一次代际）：手上那张仍不出池。
        assertSame(m[2], state.snapshotAt(1200, 1500).mask)
        assertSame(m[5], state.snapshot(1500).mask)
        state.push(1600, m[0]) // m2 被挤出
        assertNull("隔了一次 snapshot 仍受保护", state.obtain(2, 2, null))
        // 停用时连同备用池一起清掉；没取到遮罩（null）也占一条记录，不影响上面的规则。
        state.push(1700, m[6])
        state.enable(false)
        assertNull(state.snapshotAt(1700, 1700).mask)
        assertNull(state.obtain(2, 2, null))
    }

    @Test fun lateResultAfterDisableDoesNotRefillTheClearedPool() {
        // 停用（enable(false)）清空了备用池；之后才完成的在途推理不能再把遮罩放回池里，在长时间停用期间一直占着（216×384 约 83 KB）。
        val state = recycling()
        val m = List(5) { PersonMask(ByteArray(4), 2, 2) }
        for (k in 0 until 5) state.push(1000L + 100 * k, m[k]) // m0 被挤出，进池
        val ticket = checkNotNull(state.begin(1500, 216, 384, 0L))
        state.enable(false)
        assertNull("停用时清空备用池", state.obtain(2, 2, null))
        val late = PersonMask(ByteArray(4), 2, 2)
        assertFalse(state.complete(ticket, late, 1560))
        assertNull("迟到的在途结果不进池", state.obtain(2, 2, null))
        // 重新启用后同样取不到它（池从空开始）；启用期间的代际作废仍照常进池。
        state.enable(true)
        assertNull(state.obtain(2, 2, null))
        val stale = checkNotNull(state.begin(1700, 216, 384, 0L))
        state.reset()
        val orphan = PersonMask(ByteArray(4), 2, 2)
        assertFalse(state.complete(stale, orphan, 1700))
        assertSame(orphan, state.obtain(2, 2, null))
    }

    @Test fun defaultStateNeverRecycles() {
        // 人像分割（默认 spares = 0）：不留备用池，行为与复用之前相同。
        val state = MaskTrackingState().apply { enable(true) }
        val masks = List(8) { PersonMask(ByteArray(4), 2, 2) }
        masks.forEachIndexed { k, mask ->
            val ticket = checkNotNull(state.begin(1000L + 30 * k, 720, 1280, 0L))
            assertTrue(state.complete(ticket, mask, 1000L + 30 * k))
            assertNull(state.obtain(2, 2, null))
        }
        state.reset()
        assertNull(state.obtain(2, 2, null))
    }

    @Test fun snapshotAtMatchesDelayedDisplayTimestampWithoutLeadingGhost() {
        val state = MaskTrackingState()
        state.enable(true)
        val m1 = PersonMask(byteArrayOf(10, 10, 10, 10), 2, 2)
        val m2 = PersonMask(byteArrayOf(90, 90, 90, 90), 2, 2)
        val t1 = state.begin(1000L, 720, 1280, 50L)!!
        assertTrue(state.complete(t1, m1, 1030L))
        val t2 = state.begin(1060L, 720, 1280, 50L)!!
        assertTrue(state.complete(t2, m2, 1090L))

        // 当前已到 1095ms，但 FrameDelay 正在显示 1005ms 的相机帧：应返回 m1 而非超前的 m2。
        val aligned = state.snapshotAt(1005L, 1095L)
        assertSame(m1, aligned.mask)
        assertEquals(1000L, aligned.sampleTimestamp)
        assertEquals(1060L, aligned.completed)
        // 最新帧 snapshot(now) 则返回 m2。
        assertSame(m2, state.snapshot(1095L).mask)
    }
}
