package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class MeshFallbackFadeTest {
    /**
     * 与 VideoEffectsFilter 一致的两条路径强度：网格路径只有网格可用时才画得出来（meshDeformed 要求 mesh != null 且
     * faceMeshHealthy），乘份额；回退路径乘 1 − 份额。
     */
    private fun strengths(fade: MeshFallbackFade, meshAvailable: Boolean, nowMs: Long): Pair<Float, Float> {
        val share = fade.update(meshAvailable, visible = true, nowMs = nowMs)
        return (if (meshAvailable) share else 0f) to 1f - share
    }

    @Test fun meshFailureSwitchesToFallbackAtOnceAndRecoveryCrossFades() {
        val fade = MeshFallbackFade()
        for (t in 0L..300L step 33) assertEquals(1f, fade.update(meshAvailable = true, visible = true, nowMs = t), 0f)
        // 网格通道故障：网格路径当帧就画不出来，份额当帧归 0，回退路径满强度接上（不先跌到约 22% 再爬升）。
        var t = 330L
        while (t <= 330L + 200L) {
            assertEquals("t=$t", 0f, fade.update(meshAvailable = false, visible = true, nowMs = t), 0f)
            t += 33
        }
        // 回到网格路径（GL 上下文重建后）：份额按 FADE_MS 每帧上升 33/150，两条路径同时画、交叉渐变。
        fade.update(meshAvailable = true, visible = true, nowMs = t)
        assertEquals(33f / MeshFallbackFade.FADE_MS, fade.mesh, 1e-6f)
        for (k in 1..5) fade.update(meshAvailable = true, visible = true, nowMs = t + k * 33L)
        assertEquals(1f, fade.mesh, 0f)
    }

    @Test fun pathStrengthsAlwaysSumToOne() {
        // 两个方向、单帧拒收与来回切换：每一帧网格路径强度 + 回退路径强度都恒为 1，不出现下陷。
        val fade = MeshFallbackFade()
        val pattern = listOf(true, true, true, false, true, true, false, false, false, true, true, true, true, true, true, false, true)
        for ((i, available) in pattern.withIndex()) {
            val (mesh, fallback) = strengths(fade, available, i * 33L)
            assertEquals("第 $i 帧 网格可用=$available 网格 $mesh + 回退 $fallback", 1f, mesh + fallback, 1e-6f)
        }
    }

    @Test fun invisibleFaceSnapsToTargetAndStallsAreCapped() {
        val fade = MeshFallbackFade()
        // 脸不可见（新一轮由 presence 从 0 渐入）：直接取目标值。
        assertEquals(0f, fade.update(meshAvailable = false, visible = false, nowMs = 0), 0f)
        assertEquals(1f, fade.update(meshAvailable = true, visible = false, nowMs = 33), 0f)
        assertEquals(0f, fade.update(meshAvailable = false, visible = true, nowMs = 66), 0f)
        // GL 线程卡顿或切后台回来：渐变单步最多推进 MAX_STEP_MS。
        assertEquals(MeshFallbackFade.MAX_STEP_MS / MeshFallbackFade.FADE_MS, fade.update(true, true, 5_000), 1e-6f)
        fade.reset()
        assertEquals(1f, fade.mesh, 0f)
        // reset 后第一帧没有 dt（不按与卡顿前的间隔推进）；同一时刻再调用也不前进。
        assertEquals(0f, fade.update(false, true, 10_000), 0f)
        assertEquals(0f, fade.update(true, true, 10_000), 0f)
    }
}
