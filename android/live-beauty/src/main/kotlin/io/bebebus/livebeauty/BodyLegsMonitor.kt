package io.bebebus.livebeauty

/**
 * 「开了长腿、识别到身体，却没拍到膝盖」（坐着、半身入镜）的提示状态，纯逻辑，只在 GL 线程使用。
 * 条件持续 [HOLD_MS] 才进入或解除，避免走动时来回闪；长腿关闭或暂停时立即解除。
 */
class BodyLegsMonitor {
    var hidden = false
        private set
    private var since = Long.MIN_VALUE

    fun reset() { hidden = false; since = Long.MIN_VALUE }

    /** @return 状态变化时返回新状态，否则 null。 */
    fun update(wantLegs: Boolean, bodyPresent: Boolean, legsVisible: Boolean, nowMs: Long): Boolean? {
        if (!wantLegs) {
            since = Long.MIN_VALUE
            if (!hidden) return null
            hidden = false
            return false
        }
        // 识别不到身体时维持现状：不知道有没有腿。
        if (!bodyPresent) { since = Long.MIN_VALUE; return null }
        val target = !legsVisible
        if (target == hidden) { since = Long.MIN_VALUE; return null }
        if (since == Long.MIN_VALUE) since = nowMs
        if (nowMs - since < HOLD_MS) return null
        since = Long.MIN_VALUE
        hidden = target
        return hidden
    }

    companion object {
        const val HOLD_MS = 1_000L
    }
}
