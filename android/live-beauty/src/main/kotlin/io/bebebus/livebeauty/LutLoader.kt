package io.bebebus.livebeauty

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.LinkedHashMap
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * 在后台线程准备 LUT，支持两类来源：
 * 1. 内置资产（`asset://luts/<id>.png`）：直接从 [assets] 读取 PNG 字节（并做小容量字节 LRU 缓存，避免切滤镜重复 IO）
 *    并解码为独立 512×512 ARGB_8888 位图，毫秒级完成；
 * 2. 远程地址（`https://...`）：按 URL 缓存在 [cacheDir]/luts/<sha256>.png，缺失时下载（最多 [MAX_BYTES]，
 *    先写临时文件再改名）再解码校验 512×512。
 * GL 线程用 [take] 取走结果上传并负责 recycle，不在 GL 线程做任何 IO；只保留最近一次请求的结果。
 */
internal class LutLoader(
    private val cacheDir: File?,
    private val assets: AssetManager? = null,
) {
    /** [bitmap] 为 null 表示加载、下载、解码失败或尺寸不对。 */
    class Result(val url: String, val bitmap: Bitmap?)

    private val worker = ThreadPoolExecutor(0, 1, 5, TimeUnit.SECONDS, LinkedBlockingQueue())
    @Volatile private var requested = ""
    private val ready = AtomicReference<Result?>(null)
    // 仅 worker 线程访问：缓存最近使用过的内置 LUT PNG 原始字节（每张 ~60-110 KB），每次产出全新 Bitmap 防 recycle 冲突。
    private val assetBytesCache = object : LinkedHashMap<String, ByteArray>(ASSET_CACHE_LIMIT + 1, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, ByteArray>?): Boolean =
            size > ASSET_CACHE_LIMIT
    }

    /** 任何线程：请求准备 [url]；与上次请求相同则忽略。空串只清除请求。 */
    @Synchronized fun request(url: String) {
        if (url == requested) return
        requested = url
        if (url.isEmpty()) return
        try { worker.execute { load(url) } } catch (_: Exception) { publish(Result(url, null)) }
    }

    /**
     * 新的 GL 上下文或失败重试：重新准备（文件已缓存则只解码）。[current] 在锁内读取当前配置，
     * 与配置线程的 [request] 交错时不会把请求改回旧地址。
     */
    @Synchronized fun restart(current: () -> String) { requested = ""; request(current()) }

    private fun load(url: String) {
        if (url != requested) return
        val bitmap = try {
            if (url.startsWith(ASSET_PREFIX)) decodeAsset(url)
            else cached(url)?.let { decodeFile(it) }
        } catch (_: Throwable) { null }
        publish(Result(url, bitmap))
    }

    private fun decodeAsset(url: String): Bitmap? {
        if (!VideoEffectsConfig.ASSET_LUT_PATTERN.matches(url)) return null
        val path = url.removePrefix(ASSET_PREFIX)
        val bytes = assetBytesCache[path] ?: readAssetBytes(path)?.also { assetBytesCache[path] = it } ?: return null
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, decodeOptions())
        return validate(bitmap)
    }

    private fun readAssetBytes(path: String): ByteArray? {
        val manager = assets ?: return null
        manager.open(path).use { input ->
            val out = ByteArrayOutputStream()
            val buffer = ByteArray(16 * 1024)
            var total = 0L
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                total += read
                if (total > MAX_BYTES) return null
                out.write(buffer, 0, read)
            }
            return out.toByteArray()
        }
    }

    private fun cached(url: String): File? {
        val dir = File(cacheDir ?: return null, "luts").apply { mkdirs() }
        val digest = MessageDigest.getInstance("SHA-256").digest(url.toByteArray()).joinToString("") { "%02x".format(it) }
        val file = File(dir, "$digest.png")
        if (file.isFile && file.length() > 0) return file
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = TIMEOUT_MS; connection.readTimeout = TIMEOUT_MS
            // 允许 CDN 跳转（HttpURLConnection 不跟随换协议的跳转），结束后再确认最终仍是 https。
            connection.instanceFollowRedirects = true
            if (connection.responseCode != HttpURLConnection.HTTP_OK || connection.url.protocol != "https") return null
            if (connection.contentLengthLong > MAX_BYTES) return null
            val temp = File(dir, "$digest.part")
            connection.inputStream.use { input ->
                temp.outputStream().use { output ->
                    val buffer = ByteArray(16 * 1024)
                    var total = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        if (total > MAX_BYTES) { output.close(); temp.delete(); return null }
                        output.write(buffer, 0, read)
                    }
                }
            }
            return if (temp.renameTo(file)) file else { temp.delete(); null }
        } finally { connection.disconnect() }
    }

    private fun decodeFile(file: File): Bitmap? {
        val valid = validate(BitmapFactory.decodeFile(file.path, decodeOptions()))
        if (valid == null) file.delete() // 坏文件不留在缓存里
        return valid
    }

    private fun decodeOptions() = BitmapFactory.Options().apply {
        inScaled = false
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }

    private fun validate(bitmap: Bitmap?): Bitmap? {
        val valid = bitmap?.takeIf { it.width == LutPass.SIZE && it.height == LutPass.SIZE && it.config == Bitmap.Config.ARGB_8888 }
        if (valid == null) bitmap?.recycle()
        return valid
    }

    private fun publish(result: Result) {
        ready.getAndSet(result)?.bitmap?.let { if (it !== result.bitmap) it.recycle() }
    }

    /** GL 线程：取走 [url] 的结果（调用方负责回收位图）；尚未就绪返回 null。 */
    fun take(url: String): Result? {
        val result = ready.get() ?: return null
        if (result.url != url || !ready.compareAndSet(result, null)) return null
        return result
    }

    companion object {
        const val ASSET_PREFIX = "asset://"
        const val ASSET_CACHE_LIMIT = 4
        const val MAX_BYTES = 2L * 1024 * 1024
        const val TIMEOUT_MS = 10_000
    }
}
