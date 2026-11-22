package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import java.util.IdentityHashMap
import java.util.concurrent.AbstractExecutorService
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * HairSegmenter：纯函数部分（标签核对、取哪一张置信图、量化与行序），以及注入假后端驱动的 worker 契约（故障、关闭、
 * 遮罩缓冲复用与交接）。模型推理本身见 androidTest 的 HairSegmenterModelTest。
 */
class HairSegmenterTest {
    private fun plane(values: FloatArray, width: Int, height: Int): HairSegmenter.Plane {
        // 与 worker 上相同：直接缓冲、本机字节序的 float 视图。
        val buffer = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(values); buffer.position(0)
        return HairSegmenter.Plane(buffer, width, height)
    }

    private fun constant(value: Float, width: Int, height: Int) = plane(FloatArray(width * height) { value }, width, height)

    private fun bytes(mask: PersonMask) = mask.pixels.map { it.toInt() and 0xFF }

    @Test fun labelsMustBeExactlyBackgroundThenHair() {
        assertTrue(HairMaskTuning.labelsValid(listOf("background", "hair")))
        assertTrue(HairMaskTuning.labelsValid(listOf("background", "hair\n"))) // labels.txt 行尾
        assertFalse(HairMaskTuning.labelsValid(null))
        assertFalse(HairMaskTuning.labelsValid(emptyList()))
        assertFalse(HairMaskTuning.labelsValid(listOf("selfie"))) // 误放了人像分割模型
        assertFalse(HairMaskTuning.labelsValid(listOf("hair", "background"))) // 顺序反了，取第 1 张就成了背景
        assertFalse(HairMaskTuning.labelsValid(listOf("background", "hair", "body-skin", "face-skin", "clothes", "others")))
        assertFalse(HairMaskTuning.labelsValid(listOf("background", "Hair")))
        assertEquals("hair", HairMaskTuning.LABELS[HairSegmenter.HAIR_INDEX])
        assertEquals("hair_segmenter.tflite", HairSegmenter.MODEL)
    }

    @Test fun takesTheHairPlaneNotTheBackgroundPlane() {
        val w = 216; val h = 384
        val converted = mutableListOf<Int>()
        val planes = listOf(constant(0.9f, w, h), constant(0.2f, w, h))
        val mask = HairSegmenter.hairMask(planes.indices.toList(), w, h) { converted += it; planes[it] }
        assertEquals(listOf(1), converted) // 只把头发那张转出来
        assertEquals(w, mask.width); assertEquals(h, mask.height)
        assertTrue(bytes(mask).all { it == 51 }) // 0.2 × 255 → 51，不是背景的 0.9 → 230
    }

    @Test fun clampsNonFiniteAndOutOfRangeAndKeepsRowOrder() {
        // 3×2：行 0（画面顶部）是头发，行 1 不是；混入 NaN、无穷与越界值。
        val hair = floatArrayOf(1f, 0.5f, 1.7f, Float.NaN, -0.2f, Float.NEGATIVE_INFINITY)
        val mask = HairSegmenter.hairMask(listOf(constant(0f, 3, 2), plane(hair, 3, 2)), 3, 2) { it }
        assertEquals(listOf(255, 128, 255, 0, 0, 0), bytes(mask))
        val rows = floatArrayOf(0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.1f)
        val ordered = HairSegmenter.hairMask(listOf(constant(0f, 3, 2), plane(rows, 3, 2)), 3, 2) { it }
        assertEquals(listOf(230, 230, 230, 26, 26, 26), bytes(ordered))
    }

    @Test fun rejectsWrongPlaneCountOrSize() {
        fun fails(block: () -> Unit) = try { block(); fail("应当抛异常") } catch (_: IllegalStateException) {}
        val ok = constant(0.5f, 4, 2)
        fails { HairSegmenter.hairMask(listOf(ok), 4, 2) { it } } // 只有一张（人像分割模型）
        fails { HairSegmenter.hairMask(List(6) { ok }, 4, 2) { it } } // 多类模型
        fails { HairSegmenter.hairMask(listOf(ok, constant(0.5f, 2, 4)), 4, 2) { it } } // 尺寸与输入小图不符
        // 缓冲比宽高声明的短：同样按失败处理（不读越界）。
        val short = HairSegmenter.Plane(FloatBuffer.allocate(3), 4, 2)
        try { HairSegmenter.hairMask(listOf(ok, short), 4, 2) { it }; fail() } catch (_: java.nio.BufferUnderflowException) {}
    }

    @Test fun quantizesStraightIntoTheGivenMaskWithoutMovingTheBuffer() {
        // 写进调用方给的那块（备用池取回的旧遮罩，内容任意）并返回它本身；不改置信图视图的 position。
        val hair = floatArrayOf(1f, 0.5f, 1.7f, Float.NaN, -0.2f, Float.NEGATIVE_INFINITY)
        val view = plane(hair, 3, 2)
        view.values.position(4)
        val reused = PersonMask(ByteArray(6) { 77 }, 3, 2)
        val mask = HairSegmenter.hairMask(listOf(constant(0f, 3, 2), view), 3, 2, reused) { it }
        assertSame(reused, mask)
        assertEquals(listOf(255, 128, 255, 0, 0, 0), bytes(mask))
        assertEquals(4, view.values.position())
        // 尺寸不同的旧遮罩不能拿来写。
        try { HairSegmenter.hairMask(listOf(constant(0f, 3, 2), plane(hair, 3, 2)), 3, 2, PersonMask(ByteArray(6), 2, 3)) { it }; fail() }
        catch (_: IllegalArgumentException) {}
    }

    // ---- worker 契约（注入假后端、同步 worker 与时钟；不经 MediaPipe，JVM 可跑） ----

    /** 同步 worker：在调用线程上执行，shutdown 后拒收（与 Executors 的单线程 worker 拒收语义相同）。 */
    private class Inline : AbstractExecutorService() {
        @Volatile private var down = false
        var executed = 0
            private set
        override fun execute(command: Runnable) {
            if (down) throw RejectedExecutionException()
            executed++
            command.run()
        }
        override fun shutdown() { down = true }
        override fun shutdownNow(): MutableList<Runnable> { down = true; return mutableListOf() }
        override fun isShutdown() = down
        override fun isTerminated() = down
        override fun awaitTermination(timeout: Long, unit: TimeUnit) = down
    }

    /** 假后端：第 k 次推理把整张遮罩写成 k（或交给 [fill]），第 [failAt] 次抛错；记录推理与关闭次数、最近写入的那块。 */
    private class FakeEngine(
        override val labels: List<String>? = HairMaskTuning.LABELS,
        private val failAt: Int = -1,
        private val fill: ((Int, PersonMask) -> Unit)? = null,
    ) : HairSegmenter.Engine {
        @Volatile var calls = 0
        @Volatile var closed = 0
        @Volatile var lastOut: PersonMask? = null
        override fun segment(sample: FaceTextureSampler.Sample, timestampMs: Long, out: PersonMask): Float {
            lastOut = out
            val k = calls++
            if (k == failAt) throw IllegalStateException("推理失败")
            if (fill != null) fill.invoke(k, out) else out.pixels.fill(k.toByte())
            return 50f
        }
        override fun close() { closed++ }
    }

    private class Rig(open: () -> HairSegmenter.Engine, worker: ExecutorService = Inline(),
                      val width: Int = 8, val height: Int = 8) {
        @Volatile var now = 1_000L
        val failures = AtomicInteger()
        val logs: MutableList<String> = Collections.synchronizedList(mutableListOf())
        val segmenter = HairSegmenter(open, { failures.incrementAndGet() }, worker, { now }, { logs += it })
        val state get() = segmenter.state
        val sample = FaceTextureSampler.Sample(ByteBuffer.allocateDirect(width * height * 4), width, height)

        /** 推进 100 ms、取票并提交；返回这张票。 */
        fun step(anchor: FloatArray? = null): MaskTrackingState.Ticket {
            now += 100
            val ticket = checkNotNull(state.begin(now, width, height, 0L)) { "取不到票" }
            segmenter.submit(sample, ticket, anchor)
            return ticket
        }
    }

    @Test fun inferenceFailureVoidsTicketDisablesStateAndReportsOnce() {
        val engine = FakeEngine(failAt = 2)
        val rig = Rig({ engine })
        rig.state.enable(true)
        repeat(2) { rig.step() }
        assertNotNull(rig.state.snapshot(rig.now).mask)
        rig.step() // 第 3 次推理抛错
        assertEquals("onFailure 恰好一次", 1, rig.failures.get())
        assertEquals(listOf("头发分割停用：IllegalStateException"), rig.logs.toList())
        // state 停用：历史清空、不再发票（调用方据此回到第一批）。
        assertNull(rig.state.snapshot(rig.now).mask)
        assertNull(rig.state.begin(rig.now + 100, rig.width, rig.height, 0L))
        // 本票已结、不占单任务在途：调用方重新启用就能马上取票。
        rig.state.enable(true)
        assertNotNull(rig.state.begin(rig.now + 200, rig.width, rig.height, 0L))
        assertEquals("模型由 close 释放，故障时不关", 0, engine.closed)
    }

    @Test fun wrongLabelsOrLoadFailureCloseTheModelAndFailOnce() {
        // 标签不符（误放了人像分割模型）：先关掉刚建的模型再按加载失败处理，不推理。
        val selfie = FakeEngine(labels = listOf("selfie"))
        var opened = 0
        val rig = Rig({ opened++; selfie })
        rig.state.enable(true)
        rig.step()
        assertEquals(1, opened); assertEquals(1, selfie.closed); assertEquals(0, selfie.calls)
        assertEquals(1, rig.failures.get())
        assertNull(rig.state.begin(rig.now + 100, rig.width, rig.height, 0L))
        // 模型加载本身抛错（资产缺失等）：同样作废、停用、回调一次。
        val missing = Rig({ throw IllegalArgumentException("模型缺失") })
        missing.state.enable(true)
        missing.step()
        assertEquals(1, missing.failures.get())
        assertNull(missing.state.snapshot(missing.now).mask)
        assertNull(missing.state.begin(missing.now + 100, missing.width, missing.height, 0L))
    }

    @Test fun submitAfterCloseVoidsTicketWithoutInferring() {
        val engine = FakeEngine()
        val worker = Inline()
        val rig = Rig({ engine }, worker)
        rig.state.enable(true)
        rig.step()
        assertEquals(1, engine.calls)
        rig.now += 100
        val ticket = checkNotNull(rig.state.begin(rig.now, rig.width, rig.height, 0L)) // 关闭之前取的票
        rig.segmenter.close()
        assertEquals("close 在 worker 上释放模型", 1, engine.closed)
        assertTrue(worker.isShutdown)
        val executed = worker.executed
        rig.segmenter.submit(rig.sample, ticket, null)
        assertEquals("不再交给 worker", executed, worker.executed)
        assertEquals(1, engine.calls)
        assertEquals("关闭不是故障", 0, rig.failures.get())
        // 票当场结了，不占单任务在途。
        rig.state.enable(true)
        assertNotNull(rig.state.begin(rig.now + 100, rig.width, rig.height, 0L))
        rig.segmenter.close() // 重复关闭无副作用
        assertEquals(1, engine.closed)
    }

    @Test fun closeDuringInferenceNeitherReportsNorRunsQueuedWorkAndTheWorkerExits() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val engine = FakeEngine(fill = { k, out ->
            if (k == 0) { entered.countDown(); release.await(); throw IllegalStateException("关闭时推理出错") }
            out.pixels.fill(1)
        })
        val worker = Executors.newSingleThreadExecutor()
        val rig = Rig({ engine }, worker)
        rig.state.enable(true)
        val first = rig.step()
        assertTrue(entered.await(5, TimeUnit.SECONDS))
        // 第二张在关闭之前已排进 worker（直接构造票，绕开单任务在途）。
        rig.segmenter.submit(rig.sample, MaskTrackingState.Ticket(first.epoch, rig.now + 100, rig.width, rig.height), null)
        rig.segmenter.close()
        release.countDown()
        assertTrue("worker 释放模型后退出", worker.awaitTermination(5, TimeUnit.SECONDS))
        assertEquals("关闭期间的推理异常不回调 onFailure", 0, rig.failures.get())
        assertTrue(rig.logs.isEmpty())
        assertEquals("关闭之后排到的那张不推理", 1, engine.calls)
        assertEquals(1, engine.closed)
    }

    @Test fun steadyStateReusesMaskBuffersWithoutTouchingWhatGlIsReading() {
        val engine = FakeEngine()
        val rig = Rig({ engine })
        val state = rig.state
        state.enable(true)
        val anchor = floatArrayOf(4f, 4f, 2f) // 静止：每次都走原地 EMA
        val seen = Collections.newSetFromMap(IdentityHashMap<ByteArray, Boolean>())
        var newAfterWarmUp = 0
        var guarded = 0
        for (frame in 0 until 60) {
            // GL 线程：本帧先 snapshot(now)，再按显示时间取一张（偶数帧取历史里最旧的那张，它马上会被挤出历史）。
            state.snapshot(rig.now)
            val target = if (frame % 2 == 0) Long.MIN_VALUE / 2 else rig.now - 67
            val held = state.snapshotAt(target, rig.now).mask
            val before = held?.pixels?.copyOf()
            // 最坏交错：上传之前 worker 又完成了 1–5 次推理，把这张挤出历史、进了备用池（GL 这期间不再取遮罩）。
            repeat(1 + frame % 5) {
                rig.step(anchor)
                val written = checkNotNull(engine.lastOut)
                if (seen.add(written.pixels) && frame >= 10) newAfterWarmUp++
            }
            if (held != null) { assertArrayEquals("第 $frame 帧：GL 手上的遮罩被 worker 改写", before, held.pixels); guarded++ }
            // 上传完之后（下一帧）：交出去的最新遮罩就是 worker 写入的那块（原地 EMA，没有另建）。
            assertSame(engine.lastOut, state.snapshot(rig.now).mask)
        }
        assertEquals(0, rig.failures.get())
        assertTrue("交错场景要真的发生：$guarded", guarded > 40)
        assertEquals("稳态下不再新建遮罩", 0, newAfterWarmUp)
        assertTrue("一共只用了 ${seen.size} 块", seen.size <= 4 + MaskTrackingState.LENT_CAP + 2)
    }

    @Test fun steadyStateInferenceAllocatesNoMaskSizedArrays() {
        // 线程分配字节数（HotSpot 的 com.sun.management.ThreadMXBean，反射取，拿不到就跳过）。
        assumeTrue("当前 JVM 不支持按线程统计分配", allocatedBytes() != null)
        val w = 216; val h = 384
        val planes = listOf(plane(FloatArray(w * h) { 0.1f }, w, h), plane(FloatArray(w * h) { (it % 97) / 96f }, w, h))
        // 假后端走真实的量化入口：直接读置信图视图，写进 worker 给的那块。
        val engine = FakeEngine(fill = { _, out -> HairSegmenter.hairMask(planes, w, h, out) { it } })
        val rig = Rig({ engine }, width = w, height = h)
        rig.state.enable(true)
        val anchor = floatArrayOf(108f, 150f, 40f)
        repeat(30) { rig.step(anchor); rig.state.snapshotAt(rig.now - 67, rig.now) } // 预热：攒满历史与备用池
        val runs = 50
        val start = allocatedBytes()!!
        repeat(runs) { rig.step(anchor); rig.state.snapshotAt(rig.now - 67, rig.now) }
        val perRun = (allocatedBytes()!! - start) / runs
        assertEquals(0, rig.failures.get())
        // 改动前每次约 414–498 KB（FloatArray 331,776 B + 量化 82,944 B + EMA 82,944 B）；现在只剩票据、锚点拷贝等小对象。
        assertTrue("稳态每次推理分配 $perRun B", perRun < 8 * 1024)
    }

    private fun allocatedBytes(): Long? = try {
        val bean = Class.forName("java.lang.management.ManagementFactory").getMethod("getThreadMXBean").invoke(null)
        Class.forName("com.sun.management.ThreadMXBean").getMethod("getCurrentThreadAllocatedBytes").invoke(bean) as Long
    } catch (_: Throwable) { null }
}
