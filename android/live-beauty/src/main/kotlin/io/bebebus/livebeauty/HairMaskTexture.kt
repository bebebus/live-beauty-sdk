package io.bebebus.livebeauty

import android.opengl.GLES20.*
import android.os.Trace
import java.nio.ByteBuffer

/**
 * 头发置信度遮罩的 GL 纹理（仅 Android）：HairSegmenter 的结果（PersonMask，0–255，第 0 行是画面顶部，竖屏 216×384）
 * 按 LUMINANCE 上传，SemanticAlphaPass 与 BackgroundBlurPass 的头发合并道共用。尺寸变化时重新分配，否则 glTexSubImage2D
 * 原地更新（216×384 约 83 KB）；只在遮罩换了（stamp 变化）时由调用方上传。只在 GL 线程使用；返回后 GL_TEXTURE0 激活、
 * unit 0 未绑定，GL_UNPACK_ALIGNMENT 恢复为 4。只在（重新）分配那一帧检查 glGetError，GL 故障抛异常。
 */
internal class HairMaskTexture {
    /** 纹理 id；还没上传过时为 0。 */
    var texture = 0
        private set
    private var w = 0
    private var h = 0

    fun upload(mask: PersonMask) {
        check(mask.width > 0 && mask.height > 0 && mask.pixels.size >= mask.width * mask.height)
        Trace.beginSection("beauty.hair.upload")
        try {
            glActiveTexture(GL_TEXTURE0)
            val fresh = texture == 0 || mask.width != w || mask.height != h
            if (texture == 0) {
                val ids = IntArray(1); glGenTextures(1, ids, 0); texture = ids[0]
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            } else {
                glBindTexture(GL_TEXTURE_2D, texture)
            }
            // 216 不是 4 的倍数：按 1 字节对齐读行。
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            if (fresh) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, mask.width, mask.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                    ByteBuffer.wrap(mask.pixels))
                w = mask.width; h = mask.height
            } else {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mask.width, mask.height, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                    ByteBuffer.wrap(mask.pixels))
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
            glBindTexture(GL_TEXTURE_2D, 0)
            if (fresh) check(glGetError() == GL_NO_ERROR)
        } finally {
            Trace.endSection()
        }
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        texture = 0; w = 0; h = 0
    }

    /** GL 线程、创建它的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        if (texture != 0) glDeleteTextures(1, intArrayOf(texture), 0)
        release()
    }
}
