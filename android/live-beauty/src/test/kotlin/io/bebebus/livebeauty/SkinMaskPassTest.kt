package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test

class SkinMaskPassTest {
    /** uFrame.w 只看本帧多边形遮罩是否渲染成功（计划 H10），不看皮肤遮罩纹理是否非 0。 */
    @Test fun framePresenceFollowsThePolygonMaskFlag() {
        assertEquals(4.6f, SkinMaskPass.framePresence(on = true, polyMask = true, presence = 0.6f), 1e-6f)
        // 无脸时 renderEmpty 也会出一张只承载头发 A 通道的纹理（RGB 为 0）：没有多边形遮罩就必须走椭圆回退，不能读成 face = 0。
        assertEquals(0.6f, SkinMaskPass.framePresence(on = true, polyMask = false, presence = 0.6f), 1e-6f)
        assertEquals(1f, SkinMaskPass.framePresence(on = true, polyMask = false, presence = 1f), 1e-6f)
        assertEquals(5f, SkinMaskPass.framePresence(on = true, polyMask = true, presence = 1f), 1e-6f)
    }

    @Test fun framePresenceIsZeroWhenOffOrAbsent() {
        assertEquals(0f, SkinMaskPass.framePresence(on = false, polyMask = true, presence = 0.6f), 0f)
        assertEquals(0f, SkinMaskPass.framePresence(on = false, polyMask = false, presence = 0.6f), 0f)
        // presence 渐出到 0：即使多边形遮罩还在，也不加 4（着色器按 uFrame.w > 0 才算有脸）。
        assertEquals(0f, SkinMaskPass.framePresence(on = true, polyMask = true, presence = 0f), 0f)
        assertEquals(0f, SkinMaskPass.framePresence(on = true, polyMask = false, presence = 0f), 0f)
    }
}
