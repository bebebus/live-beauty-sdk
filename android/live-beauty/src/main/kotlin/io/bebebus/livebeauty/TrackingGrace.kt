package io.bebebus.livebeauty

/**
 * 跟踪启停的宽限（仅 Android，纯逻辑，只在 GL 线程使用）：不再需要跟踪后保留 [graceMs] 再停。
 * 人脸、美体、分割三路 tracker 各用一份：把参数临时拖到 0、按住对比原画时不关 tracker，
 * 代际不变、效果不重新渐入；健康故障由调用方立即停用，不经过宽限。
 */
internal class TrackingGrace(private val graceMs: Long = GRACE_MS) {
    private var lastWanted = Long.MIN_VALUE

    /** [wanted] 为本帧是否需要跟踪；返回是否保持跟踪。 */
    fun keep(wanted: Boolean, nowMs: Long): Boolean {
        if (wanted) { lastWanted = nowMs; return true }
        return lastWanted != Long.MIN_VALUE && nowMs - lastWanted < graceMs
    }

    fun reset() { lastWanted = Long.MIN_VALUE }

    companion object {
        const val GRACE_MS = 3_000L
    }
}
