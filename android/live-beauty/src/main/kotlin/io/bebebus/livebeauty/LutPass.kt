package io.bebebus.livebeauty

import android.graphics.Bitmap
import android.opengl.GLES20.*
import android.opengl.GLUtils

/**
 * LUT 纹理（仅 Android）。查表在主合成 FRAGMENT_PRO 内完成（uLut，纹理单元 3，uGrade.x = 强度，位于美妆之前），
 * 不再单独占一道全分辨率绘制。LUT 纹理由 [upload] 从已校验的 512×512 位图上传，按地址记住，地址不变不重传。
 *
 * 换风格时新 LUT 要先下载 / 解码（LutLoader），上传完成前 [select] 继续给出上一帧绑着的那张表，
 * 画面不会出现「完全不调色」的帧；新表上传后同一纹理原地替换，下一帧起生效。不做两张表的交叉渐变
 * （FRAGMENT_PRO 已用 15/16 个 uniform 向量）。
 * 只在 GL 线程使用；任何 GL 故障都抛异常，由调用方停用 LUT。
 */
internal class LutPass {
    private var lutTexture = 0
    private val binding = LutBinding()
    /** 已上传、可以绑定的 LUT 纹理 id；0 表示没有。 */
    val texture get() = lutTexture
    /** 已上传到 [texture] 的 LUT 地址；空串表示没有。 */
    val loadedUrl get() = binding.loadedUrl
    /** 加载中：上一次 [select] 想要、但还没上传完成的地址；空串表示没有在等。 */
    val pendingUrl get() = binding.pendingUrl
    /** 上一次 [select] 实际绑定的表的地址；空串表示上一帧没有查表。 */
    val boundUrl get() = binding.boundUrl

    /** 上传 LUT（调用方保证 512×512 ARGB_8888），之后 [loadedUrl] = [url]。不回收位图。 */
    fun upload(url: String, bitmap: Bitmap) {
        check(bitmap.width == SIZE && bitmap.height == SIZE)
        if (lutTexture == 0) {
            val ids = IntArray(1); glGenTextures(1, ids, 0); lutTexture = ids[0]
        }
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, lutTexture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GL_TEXTURE_2D, 0, bitmap, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        check(glGetError() == GL_NO_ERROR)
        binding.loaded(url)
    }

    /**
     * 每帧调用一次（语义见 [LutBinding.select]），[wanted] 为本帧想用的 LUT 地址：不查表、LUT 故障或该地址已判定
     * 失败时传空串。@return 本帧要绑到纹理单元 3 的 LUT 纹理，0 表示本帧不查表。
     */
    fun select(wanted: String): Int = if (binding.select(wanted).isNotEmpty()) lutTexture else 0

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        lutTexture = 0
        binding.reset()
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        if (lutTexture != 0) glDeleteTextures(1, intArrayOf(lutTexture), 0)
        release()
    }

    companion object {
        const val SIZE = 512
    }
}

/**
 * LUT 换表不断档的绑定语义（纯逻辑，JVM 可测；只在 GL 线程使用）。只有一张 LUT 纹理：新表上传时原地替换，
 * 上传完成前继续绑上一帧那张表。
 */
internal class LutBinding {
    /** 已上传、可绑定的表的地址；空串表示没有。 */
    var loadedUrl = ""
        private set
    /** 加载中：想要、但还没上传完成的地址；空串表示没有在等。 */
    var pendingUrl = ""
        private set
    /** 上一次 [select] 实际绑定的表的地址；空串表示上一帧没有查表。 */
    var boundUrl = ""
        private set

    /** [url] 已上传到 LUT 纹理。 */
    fun loaded(url: String) {
        loadedUrl = url
        if (pendingUrl == url) pendingUrl = ""
    }

    /**
     * 每帧调用一次。@return 本帧要绑的表的地址，空串表示不查表：
     * - [wanted] 为空 → 不查表；
     * - [wanted] 已上传 → 它；
     * - [wanted] 还在加载、上一帧绑着表 → 继续绑那张（[pendingUrl] = [wanted]），画面不出现「完全不调色」的帧；
     * - 上一帧没有查表（从无 LUT 切到有 LUT）→ 不查表，不借用更早留下的旧表，避免闪出不相干的颜色。
     */
    fun select(wanted: String): String {
        val bound = when {
            wanted.isEmpty() || loadedUrl.isEmpty() -> ""
            wanted == loadedUrl -> loadedUrl
            boundUrl == loadedUrl -> loadedUrl
            else -> ""
        }
        pendingUrl = if (wanted.isNotEmpty() && wanted != loadedUrl) wanted else ""
        boundUrl = bound
        return bound
    }

    fun reset() { loadedUrl = ""; pendingUrl = ""; boundUrl = "" }
}
