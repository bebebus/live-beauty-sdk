package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.hypot

class BodyGeometryTest {
    /** 竖屏 9:16 画面里正面站立的人：肩 y=0.30，胯 y=0.60，膝 y=0.78（MediaPipe 坐标，y 向下）。 */
    private fun pose(
        knees: Boolean = true,
        visibility: Float = 0.9f,
        shoulderY: Float = 0.30f,
        hipY: Float = 0.60f,
    ): MutableList<BodyGeometry.Landmark> {
        val points = MutableList(33) { BodyGeometry.Landmark(0.5f, 0.5f, 0.1f) }
        points[11] = BodyGeometry.Landmark(0.62f, shoulderY, visibility)
        points[12] = BodyGeometry.Landmark(0.38f, shoulderY, visibility)
        points[23] = BodyGeometry.Landmark(0.58f, hipY, visibility)
        points[24] = BodyGeometry.Landmark(0.42f, hipY, visibility)
        val kneeVisibility = if (knees) 0.9f else 0.1f
        points[25] = BodyGeometry.Landmark(0.57f, 0.78f, kneeVisibility)
        points[26] = BodyGeometry.Landmark(0.43f, 0.78f, kneeVisibility)
        return points
    }

    private val aspect = 9f / 16f

    @Test fun uprightBodyGivesDownwardAxisAndWidths() {
        val body = BodyGeometry.fromPose(pose(), aspect)!!
        // 纹理坐标 y 向上：胯中点 y = 1 - 0.60。
        assertEquals(0.5f, body.hipX, 1e-5f)
        assertEquals(0.40f, body.hipY, 1e-5f)
        // 方向为单位向量、指向画面下方。
        assertEquals(1f, hypot(body.axisX, body.axisY), 1e-5f)
        assertTrue(body.axisY < -0.99f)
        // 躯干长 = 0.30 / aspect（度量空间），半宽 = 肩宽 0.24 / 2。
        assertEquals(0.30f / aspect, body.torso, 1e-4f)
        assertEquals(0.12f, body.half, 1e-5f)
        assertEquals(1f, body.legs)
        assertEquals(BodyGeometry.STATE_SIZE, body.toState().size)
    }

    @Test fun legsOnlyWhenBothKneesVisibleBelowHips() {
        assertEquals(0f, BodyGeometry.fromPose(pose(knees = false), aspect)!!.legs)
        val seated = pose().apply {
            // 膝盖与胯几乎同高（坐姿）：不算拍到腿。
            this[25] = BodyGeometry.Landmark(0.57f, 0.62f, 0.9f)
            this[26] = BodyGeometry.Landmark(0.43f, 0.62f, 0.9f)
        }
        assertEquals(0f, BodyGeometry.fromPose(seated, aspect)!!.legs)
    }

    @Test fun rejectsMissingInvisibleOrDegenerateBodies() {
        assertNull(BodyGeometry.fromPose(pose().subList(0, 20), aspect))
        assertNull(BodyGeometry.fromPose(pose(visibility = 0.3f), aspect))
        // 0.4..0.6 之间照样算出几何并带上可见度，是否收由 BodyTrackingState 的迟滞决定。
        assertEquals(0.5f, BodyGeometry.fromPose(pose(visibility = 0.5f), aspect)!!.visibility, 1e-6f)
        assertNull(BodyGeometry.fromPose(pose(), Float.NaN))
        // 肩胯重合（躯干退化）。
        assertNull(BodyGeometry.fromPose(pose(shoulderY = 0.6f, hipY = 0.6f), aspect))
        // 倒置（胯在肩上方）。
        assertNull(BodyGeometry.fromPose(pose(shoulderY = 0.7f, hipY = 0.3f), aspect))
        val outside = pose().apply { this[11] = BodyGeometry.Landmark(1.5f, 0.3f, 0.9f) }
        assertNull(BodyGeometry.fromPose(outside, aspect))
    }

    @Test fun trackingStateThrottlesAndExpires() {
        val state = BodyTrackingState()
        assertNull(state.begin(1000, 720, 1280)) // 未启用
        state.enable(true)
        val ticket = state.begin(1000, 720, 1280)!!
        assertNull(state.begin(1050, 720, 1280)) // 单任务
        val body = BodyGeometry.fromPose(pose(), aspect)!!
        assertTrue(state.complete(ticket, body, 1080))
        assertSame(body, state.snapshot(1100).geometry)
        assertNull(state.snapshot(1300).geometry) // 超过 250 ms 失效
        assertNull(state.begin(1090, 720, 1280)) // 100 ms 限频
        assertNotNull(state.begin(1100, 720, 1280))
    }

    @Test fun smootherFadesInFollowsAndFadesOut() {
        val smoother = BodyGeometrySmoother()
        val a = BodyGeometry.fromPose(pose(), aspect)!!.toState()
        assertNotNull(smoother.update(a, 1000, 1000))
        assertEquals(0f, smoother.presence, 1e-6f)
        smoother.update(a, 1000, 1100)
        assertEquals(0.5f, smoother.presence, 1e-6f)
        smoother.update(a, 1000, 1200)
        assertEquals(1f, smoother.presence, 1e-6f)
        // 新样本：向目标靠近，方向仍为单位向量。
        val b = a.copyOf().also { it[0] = 0.6f }
        val out = smoother.update(b, 1100, 1300)!!
        assertTrue(out[0] > 0.5f && out[0] < 0.6f)
        assertEquals(1f, hypot(out[2], out[3]), 1e-5f)
        // 目标消失：保持 67 ms，再 100 ms 渐出后清空（形变场最多残留约 170 ms，以前约 0.5 s）。
        var t = 1300L
        while (t + 10 < 1300 + BodyGeometrySmoother.HOLD_MS.toLong()) { t += 10; smoother.update(null, 0, t) }
        assertEquals(1f, smoother.presence, 0f)
        while (smoother.presence > 0f && t < 2000) { t += 10; smoother.update(null, 0, t) }
        assertTrue("保持 + 渐出 ${t - 1300} ms", t - 1300 in 160L..180L)
        assertNull(smoother.update(null, 0, t + 10))
    }

    @Test fun trackingStateHysteresisAndSingleFailureKeepsHistory() {
        val state = BodyTrackingState()
        state.enable(true)
        var now = 1000L
        fun step(body: BodyGeometry?): BodyGeometry? {
            val ticket = state.begin(now, 720, 1280)!!
            state.complete(ticket, body, now + 20)
            val snap = state.snapshot(now + 20)
            now += 100
            return snap.geometry
        }
        val weak = BodyGeometry.fromPose(pose(visibility = 0.5f), aspect)!!
        val strong = BodyGeometry.fromPose(pose(visibility = 0.9f), aspect)!!
        // 没在跟踪：可见度 0.5 不到 0.6，不收。
        assertNull(step(weak))
        assertSame(strong, step(strong))
        // 跟踪中：0.5 ≥ 0.4，照收。
        assertSame(weak, step(weak))
        // 单次推理失败不清历史：快照仍给出上一帧（250 ms 内）。
        assertSame(weak, step(null))
        // 连续第二次失败才清。
        assertNull(step(null))
        // 清掉之后重新开始，仍要 0.6。
        assertNull(step(weak))
    }
}
