package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GpuDelegatePolicyTest {
    /** 内存版 SharedPreferences：值的类型按写入原样保存，getInt 遇到字符串时与 SharedPreferences 一样抛 ClassCastException。 */
    private class MemoryStore : GpuDelegatePolicy.Store {
        val values = HashMap<String, Any>()
        var syncWrites = 0
        override fun getString(key: String): String? = values[key] as String?
        override fun getInt(key: String, default: Int): Int {
            return (values[key] ?: return default) as Int
        }
        override fun put(changes: Map<String, Any?>, sync: Boolean) {
            if (sync) syncWrites++
            for ((k, v) in changes) if (v == null) values.remove(k) else values[k] = v
        }
    }

    private val fp = "google/device/1:14/UQ1A/123:user/release-keys"

    @Test fun firstLaunchAllowsGpuAndSameProcessRecreateIsNotACrash() {
        val store = MemoryStore()
        val policy = GpuDelegatePolicy(store, fp, pid = 100, processStart = 5000)
        assertTrue(policy.allowGpu())
        policy.beforeGpuAttempt()
        assertEquals("100@5000", store.values[GpuDelegatePolicy.KEY_TRYING])
        assertTrue("标记要同步落盘", store.syncWrites > 0)
        // 同一进程里关掉再重开跟踪（新 tracker 实例）不算崩溃。
        policy.beforeGpuAttempt()
        assertTrue(policy.allowGpu())
        policy.onGpuSuccess()
        assertNull(store.values[GpuDelegatePolicy.KEY_TRYING])
    }

    @Test fun markerLeftByAnotherProcessCountsAsNativeCrash() {
        val store = MemoryStore()
        GpuDelegatePolicy(store, fp, pid = 100, processStart = 5000).beforeGpuAttempt() // 之后原生崩溃，没撤标记
        val second = GpuDelegatePolicy(store, fp, pid = 200, processStart = 9000)
        assertEquals(1, store.values[GpuDelegatePolicy.KEY_CRASHES])
        assertNull(store.values[GpuDelegatePolicy.KEY_TRYING])
        assertTrue("1 次崩溃仍可再试", second.allowGpu())
        second.beforeGpuAttempt() // 又崩溃
        val third = GpuDelegatePolicy(store, fp, pid = 300, processStart = 12000)
        assertEquals(2, store.values[GpuDelegatePolicy.KEY_CRASHES])
        assertFalse("原生崩溃满 2 次后不再用 GPU", third.allowGpu())
    }

    @Test fun reusedPidWithDifferentStartTimeIsStillAnotherProcess() {
        val store = MemoryStore()
        GpuDelegatePolicy(store, fp, pid = 100, processStart = 5000).beforeGpuAttempt()
        // 原生崩溃后新进程恰好拿到同一个 pid：启动时间不同，仍判为上个进程崩溃。
        GpuDelegatePolicy(store, fp, pid = 100, processStart = 7000)
        assertEquals(1, store.values[GpuDelegatePolicy.KEY_CRASHES])
        // 拿不到启动时间时只比较 pid（同 pid 视为同进程，不计数）。
        GpuDelegatePolicy(store, fp, pid = 300, processStart = GpuDelegatePolicy.UNKNOWN_START).beforeGpuAttempt()
        assertEquals("300", store.values[GpuDelegatePolicy.KEY_TRYING])
        GpuDelegatePolicy(store, fp, pid = 300, processStart = 8000)
        assertEquals(1, store.values[GpuDelegatePolicy.KEY_CRASHES])
    }

    @Test fun javaFailureDisablesThisProcessButLaterProcessesRetryUpToThree() {
        val store = MemoryStore()
        for (attempt in 1..3) {
            val policy = GpuDelegatePolicy(store, fp, pid = attempt, processStart = attempt * 1000L)
            assertTrue("第 $attempt 个进程仍可再试", policy.allowGpu())
            policy.beforeGpuAttempt()
            policy.onGpuJavaFailure()
            assertFalse("本进程立即改用 CPU", policy.allowGpu())
            assertNull("Java 异常撤掉标记，不会被下个进程当成原生崩溃", store.values[GpuDelegatePolicy.KEY_TRYING])
            assertEquals(attempt, store.values[GpuDelegatePolicy.KEY_FAILURES])
        }
        assertFalse("满 3 次后不再用 GPU", GpuDelegatePolicy(store, fp, pid = 9, processStart = 9000).allowGpu())
        assertEquals(0, store.values[GpuDelegatePolicy.KEY_CRASHES] ?: 0)
    }

    @Test fun successResetsCounters() {
        val store = MemoryStore()
        GpuDelegatePolicy(store, fp, 1, 1000).apply { beforeGpuAttempt(); onGpuJavaFailure() }
        GpuDelegatePolicy(store, fp, 2, 2000).beforeGpuAttempt() // 原生崩溃
        val policy = GpuDelegatePolicy(store, fp, 3, 3000)
        assertEquals(1, store.values[GpuDelegatePolicy.KEY_CRASHES]); assertEquals(1, store.values[GpuDelegatePolicy.KEY_FAILURES])
        policy.beforeGpuAttempt()
        policy.onGpuSuccess()
        assertEquals(0, store.values[GpuDelegatePolicy.KEY_CRASHES]); assertEquals(0, store.values[GpuDelegatePolicy.KEY_FAILURES])
        assertNull(store.values[GpuDelegatePolicy.KEY_TRYING])
    }

    @Test fun fingerprintChangeResetsCounters() {
        val store = MemoryStore()
        for (pid in 1..3) GpuDelegatePolicy(store, fp, pid, pid * 1000L).apply { beforeGpuAttempt(); onGpuJavaFailure() }
        assertFalse(GpuDelegatePolicy(store, fp, 4, 4000).allowGpu())
        // 系统 / 驱动升级：计数归零、旧标记作废。
        GpuDelegatePolicy(store, fp, 5, 5000).beforeGpuAttempt()
        val upgraded = GpuDelegatePolicy(store, "$fp-ota2", 6, 6000)
        assertTrue(upgraded.allowGpu())
        assertEquals("$fp-ota2", store.values[GpuDelegatePolicy.KEY_FINGERPRINT])
        assertEquals(0, store.values[GpuDelegatePolicy.KEY_CRASHES])
        assertEquals(0, store.values[GpuDelegatePolicy.KEY_FAILURES])
    }

    @Test fun legacyFingerprintMarkerIsTreatedAsNoMarker() {
        // 旧版在同一文件的 gpu_trying 里存 Build.FINGERPRINT 字符串，且没有 fp 键：按字符串读、解析不出 pid，视为无标记，
        // 不抛 ClassCastException、不永久回退 CPU。
        val store = MemoryStore().apply { values[GpuDelegatePolicy.KEY_TRYING] = fp }
        val first = GpuDelegatePolicy(store, fp, 100, 5000)
        assertTrue(first.allowGpu())
        assertNull(store.values[GpuDelegatePolicy.KEY_TRYING])
        assertEquals(0, store.values[GpuDelegatePolicy.KEY_CRASHES])
        // 已有 fp 键的升级场景：旧格式标记同样只清掉、不计崩溃。
        val upgraded = MemoryStore().apply {
            values[GpuDelegatePolicy.KEY_FINGERPRINT] = fp
            values[GpuDelegatePolicy.KEY_TRYING] = fp
        }
        assertTrue(GpuDelegatePolicy(upgraded, fp, 100, 5000).allowGpu())
        assertNull(upgraded.values[GpuDelegatePolicy.KEY_TRYING])
        assertEquals(0, upgraded.values[GpuDelegatePolicy.KEY_CRASHES] ?: 0)
        assertNull(GpuDelegatePolicy.Marker.parse(fp))
        assertEquals(GpuDelegatePolicy.Marker(42, 77), GpuDelegatePolicy.Marker.parse("42@77"))
        assertEquals(GpuDelegatePolicy.Marker(42, GpuDelegatePolicy.UNKNOWN_START), GpuDelegatePolicy.Marker.parse("42"))
    }

    @Test fun trackerUsesSingleFaceForBuiltInSmoothing() {
        // numFaces = 1：MediaPipe 只在单脸视频模式下开启内置关键点平滑，并跳过每帧检测器（常量留作测试包 A/B）。
        assertEquals(1, FaceLandmarkTracker.NUM_FACES)
    }
}
