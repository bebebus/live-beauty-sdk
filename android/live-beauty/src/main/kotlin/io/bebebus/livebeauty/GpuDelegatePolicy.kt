package io.bebebus.livebeauty

/**
 * 人脸关键点 GPU 推理的回退策略（纯逻辑，持久化经 [Store]，生产实现基于 SharedPreferences，见 FaceLandmarkTracker）。
 *
 * 以前只要建 GPU 实例或首次推理抛出 Java 异常，「正在尝试」标记就留在磁盘上，同一指纹的设备从此永远走 CPU（约 15 Hz）。
 * 现在把两类故障分开记：
 * - 原生崩溃（进程直接死掉）：尝试前同步写下本进程的标记（pid 与进程启动时间），首次推理成功才撤掉；
 *   下一个进程启动时看到「别的进程」留下的标记，才算一次原生崩溃。满 [MAX_CRASHES] 次后本指纹不再用 GPU。
 * - Java 异常（可恢复）：本进程立即改用 CPU，计数加一；以后的进程仍可再试，满 [MAX_FAILURES] 次后不再用 GPU。
 * GPU 推理成功一次就把两个计数清零；系统 / 驱动升级（指纹变化）也清零。
 *
 * 兼容旧版本：旧版在同一个文件的 [KEY_TRYING] 里存的是 Build.FINGERPRINT 字符串，这里一律按字符串读，
 * 解析不出 pid 的旧值视为「无标记」并移除（不按崩溃计数，也不会抛 ClassCastException）。
 * 标记同时记 pid 与进程启动时间：原生崩溃后新进程恰好拿到同一个 pid 时，启动时间不同，仍判为别的进程。
 *
 * 一个进程只应有一个实例（构造时做一次启动检查，可能在 GL 线程，所以只异步落盘），方法可在任意线程调用。
 */
internal class GpuDelegatePolicy(
    private val store: Store,
    fingerprint: String,
    private val pid: Int,
    /** 本进程启动时间（Process.getStartUptimeMillis）；拿不到时传 [UNKNOWN_START]，此时只比较 pid。 */
    private val processStart: Long = UNKNOWN_START,
) {
    /** 持久化接口：值为 null 表示删除该键；[sync] 为 true 时须在返回前落盘。 */
    interface Store {
        fun getString(key: String): String?
        fun getInt(key: String, default: Int): Int
        fun put(changes: Map<String, Any?>, sync: Boolean)
    }

    private var crashes: Int
    private var failures: Int
    /** 本进程内 GPU 抛过 Java 异常：之后新建的实例直接用 CPU。 */
    private var disabled = false
    /** 本进程已写下「正在尝试」标记且还没撤。 */
    private var trying = false

    init {
        val storedFp = store.getString(KEY_FINGERPRINT)
        if (storedFp != fingerprint) {
            // 首次运行或系统 / 驱动升级：计数归零，旧指纹下的标记一并作废。
            crashes = 0; failures = 0
            store.put(mapOf(KEY_FINGERPRINT to fingerprint, KEY_CRASHES to 0, KEY_FAILURES to 0, KEY_TRYING to null), false)
        } else {
            crashes = store.getInt(KEY_CRASHES, 0).coerceAtLeast(0)
            failures = store.getInt(KEY_FAILURES, 0).coerceAtLeast(0)
            val raw = store.getString(KEY_TRYING)
            if (raw != null) {
                val marker = Marker.parse(raw)
                // 解析不出的旧格式（指纹字符串）按无标记处理；他进程留下的标记 = 上次 GPU 尝试时原生崩溃。
                if (marker != null && !marker.isProcess(pid, processStart)) crashes++
                store.put(mapOf(KEY_TRYING to null, KEY_CRASHES to crashes), false)
            }
        }
    }

    /** 当前是否允许用 GPU 推理。 */
    @Synchronized fun allowGpu(): Boolean = !disabled && crashes < MAX_CRASHES && failures < MAX_FAILURES

    /** 每次建 GPU 实例前调用：同步写下本进程的「正在尝试」标记（建实例或首次推理时原生崩溃，下次启动据此计数）。 */
    @Synchronized fun beforeGpuAttempt() {
        trying = true
        store.put(mapOf(KEY_TRYING to Marker(pid, processStart).encode()), true)
    }

    /** 该实例 GPU 首次推理成功：撤掉标记、两个计数清零（异步落盘即可）；没有要改的就不写盘。 */
    @Synchronized fun onGpuSuccess() {
        if (!trying && crashes == 0 && failures == 0) return
        trying = false
        crashes = 0; failures = 0
        store.put(mapOf(KEY_TRYING to null, KEY_CRASHES to 0, KEY_FAILURES to 0), false)
    }

    /** 建 GPU 实例或推理抛出 Java 异常：撤掉标记、失败计数加一，本进程改用 CPU。 */
    @Synchronized fun onGpuJavaFailure() {
        disabled = true
        trying = false
        failures++
        store.put(mapOf(KEY_TRYING to null, KEY_FAILURES to failures), true)
    }

    /** 「正在尝试」标记：`pid` 或 `pid@进程启动时间`。 */
    internal data class Marker(val pid: Int, val start: Long) {
        fun encode(): String = if (start == UNKNOWN_START) pid.toString() else "$pid@$start"

        /** 同一进程：pid 相同，且双方都知道启动时间时启动时间也相同。 */
        fun isProcess(pid: Int, start: Long): Boolean =
            this.pid == pid && (this.start == UNKNOWN_START || start == UNKNOWN_START || this.start == start)

        companion object {
            fun parse(raw: String): Marker? {
                val at = raw.indexOf('@')
                val pid = (if (at < 0) raw else raw.substring(0, at)).toIntOrNull() ?: return null
                if (at < 0) return Marker(pid, UNKNOWN_START)
                val start = raw.substring(at + 1).toLongOrNull() ?: return null
                return Marker(pid, start)
            }
        }
    }

    companion object {
        /** 原生崩溃满 2 次后本指纹不再用 GPU。 */
        const val MAX_CRASHES = 2
        /** Java 异常满 3 次后本指纹不再用 GPU。 */
        const val MAX_FAILURES = 3
        const val UNKNOWN_START = Long.MIN_VALUE
        const val KEY_FINGERPRINT = "fp"
        const val KEY_CRASHES = "gpu_crashes"
        const val KEY_FAILURES = "gpu_failures"
        /** 与旧版同名同文件：旧版存 Build.FINGERPRINT，新版存 [Marker]。 */
        const val KEY_TRYING = "gpu_trying"
    }
}
