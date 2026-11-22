package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 1/4 分辨率 (mean.rgb, var)，供合成磨皮使用（规格 4.1）。只在 GL 线程使用，只依赖源帧与尺寸、不依赖参数。
 * 固定 5 个通道、4 个程序、3 张 RGBA8 纹理（LINEAR、CLAMP_TO_EDGE、无深度附件）：
 * src →DOWN→ T0 →横向→ T1 →纵向(+T0)→ T2=(mean, v0) →横向 alpha→ T1 →纵向 alpha→ T2=(mean, var)。
 * 返回后：GL_TEXTURE0 激活、unit 0/1 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。
 * 任何 GL 故障都抛异常，由调用方转成 effects_failed。
 */
internal class BeautyPrePass {
    private class Program(vertex: String, fragment: String) {
        val id = GlUtil.createProgram(vertex, fragment).also { check(it > 0) }
        val position = glGetAttribLocation(id, "aPosition")
        val coordinate = glGetAttribLocation(id, "aTextureCoord")
        val sampler = glGetUniformLocation(id, "uSampler")
        val texel = glGetUniformLocation(id, "uTexel")
        val direction = glGetUniformLocation(id, "uDir")
        val original = glGetUniformLocation(id, "uOrig")
    }
    private class Programs {
        val down = Program(VideoEffectsShaders.DOWN_VERTEX, VideoEffectsShaders.DOWN_FRAGMENT)
        val blurRgb = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_RGB_FRAGMENT)
        val blurDiff = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_DIFF_FRAGMENT)
        val blurAlpha = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_ALPHA_FRAGMENT)
        fun delete() { for (program in arrayOf(down, blurRgb, blurDiff, blurAlpha)) glDeleteProgram(program.id) }
    }
    private var programs: Programs? = null
    private val textures = IntArray(3)
    private val frames = IntArray(3)
    private var sw = 0
    private var sh = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    /** @return 存放 (mean.rgb, var) 的纹理 T2。 */
    fun render(source: Int, width: Int, height: Int): Int {
        check(width > 0 && height > 0)
        var fresh = false
        // 取本地引用，整次 render 只用这一组程序。
        val p = programs ?: Programs().also { programs = it; fresh = true }
        val w = (width + 3) / 4; val h = (height + 3) / 4
        if (w != sw || h != sh) {
            deleteBuffers()
            glGenTextures(3, textures, 0); glGenFramebuffers(3, frames, 0)
            sw = w; sh = h
            glActiveTexture(GL_TEXTURE0)
            for (i in 0 until 3) {
                glBindTexture(GL_TEXTURE_2D, textures[i])
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
                glBindFramebuffer(GL_FRAMEBUFFER, frames[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0)
                check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
            }
            glBindTexture(GL_TEXTURE_2D, 0)
            fresh = true
        }
        glViewport(0, 0, w, h)
        // P1 DOWN：src → T0。
        pass(p.down, frames[0], source) { glUniform2f(it.texel, 1f / width, 1f / height) }
        // P2 BLUR_RGB：T0 → T1，横向。
        pass(p.blurRgb, frames[1], textures[0]) { glUniform2f(it.direction, 1f / w, 0f) }
        // P3 BLUR_DIFF：T1 → T2，纵向，T0 在 unit 1。
        pass(p.blurDiff, frames[2], textures[1], textures[0]) {
            glUniform2f(it.direction, 0f, 1f / h); glUniform1i(it.original, 1)
        }
        // P4/P5 BLUR_ALPHA：T2 → T1 横向、T1 → T2 纵向。
        pass(p.blurAlpha, frames[1], textures[2]) { glUniform2f(it.direction, 1f / w, 0f) }
        pass(p.blurAlpha, frames[2], textures[1]) { glUniform2f(it.direction, 0f, 1f / h) }
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        // 发布版 AAR 的 checkGlError 是空操作；只在（重新）分配那一帧查一次。
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[2]
    }

    private inline fun pass(program: Program, frame: Int, first: Int, second: Int = 0, uniforms: (Program) -> Unit) {
        glBindFramebuffer(GL_FRAMEBUFFER, frame)
        glUseProgram(program.id)
        vertices.position(0)
        glVertexAttribPointer(program.position, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(program.position)
        vertices.position(2)
        glVertexAttribPointer(program.coordinate, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(program.coordinate)
        if (second != 0) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, second) }
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, first)
        glUniform1i(program.sampler, 0)
        uniforms(program)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(program.position); glDisableVertexAttribArray(program.coordinate)
    }

    private fun deleteBuffers() {
        if (sw != 0) { glDeleteFramebuffers(3, frames, 0); glDeleteTextures(3, textures, 0) }
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
        programs = null
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        programs?.delete(); deleteBuffers(); release()
    }
}
