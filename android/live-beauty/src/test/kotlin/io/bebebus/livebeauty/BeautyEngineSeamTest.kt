package io.bebebus.livebeauty

import android.content.Context
import org.junit.Assert.*
import org.junit.Test

/** 外部美颜引擎接缝的 GL 线程侧纯逻辑：参数对账（EngineConfigSync）与输出延迟补偿（EngineStamps）。 */
class BeautyEngineSeamTest {
    private class FakeEngine(override val latencyFrames: Int = 0) : BeautyEngine {
        val configured = ArrayList<VideoEffectsConfig>()
        override fun onGlCreated(context: Context) {}
        override fun configure(config: VideoEffectsConfig) { configured += config }
        override fun process(inputTexture: Int, width: Int, height: Int, timestampNs: Long) = inputTexture
        override fun onGlReleased(contextAlive: Boolean) {}
    }

    @Test fun configureFollowsSnapshotReferenceOnGlThread() {
        val sync = EngineConfigSync()
        val engine = FakeEngine()
        // 新引擎只建、不配；主线程此时换了风格（只写快照）：第一帧按最新快照补配，不会停在建引擎之前的旧参数。
        var config = VideoEffectsConfig.original
        config = config.copy(smoothing = 0.7f)
        assertTrue(sync.sync(engine, config))
        assertSame(config, engine.configured.single())
        // 快照没变：之后每帧都不再调。
        repeat(3) { assertFalse(sync.sync(engine, config)) }
        // 新快照（即使内容相同，Dart 重发也是新对象）：按引用比较，再配一次。
        val resent = config.copy()
        assertTrue(sync.sync(engine, resent))
        assertEquals(2, engine.configured.size)
        assertSame(resent, engine.configured.last())
        // 换引擎、重建上下文：reset 后下一帧给新引擎补配同一快照。
        sync.reset()
        val other = FakeEngine()
        assertTrue(sync.sync(other, resent))
        assertSame(resent, other.configured.single())
    }

    @Test fun latencyFramesShiftPublishedContentByThatManyFrames() {
        val period = 33_333_333L
        val base = 5_000_000_000L
        val stamps = EngineStamps()
        val pacing = FramePacing()
        assertEquals(0L, stamps.content(2)) // 还没有帧：未知
        for (k in 0 until 10) {
            val stamp = base + k * period
            stamps.push(stamp)
            // 与 drawEngine 一致：latencyFrames = 2 时发布的内容时间是 2 帧之前的相机帧（历史不足时取最早的一帧）。
            pacing.publish(stamp, stamps.content(2))
            assertEquals("k=$k", base + maxOf(0, k - 2) * period, pacing.contentFor(stamp))
        }
        // 某帧没有发布：FramePacing 沿用同样的延后。
        assertEquals(base + 8 * period, pacing.contentFor(base + 10 * period))
        assertEquals(base + 9 * period, stamps.content(0))
        // 超过上限按 4 帧计（只保留 5 个时间戳），负值按 0 计。
        assertEquals(base + 5 * period, stamps.content(9))
        assertEquals(base + 5 * period, stamps.content(EngineStamps.MAX_LATENCY_FRAMES))
        assertEquals(base + 9 * period, stamps.content(-1))
        stamps.reset()
        assertEquals(0L, stamps.content(2))
    }
}
