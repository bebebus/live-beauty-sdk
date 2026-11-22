package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 美体变形（仅 Android）：源帧 → 同尺寸 RGBA8 中间纹理（LINEAR、CLAMP_TO_EDGE、无深度附件），合成与磨皮预处理
 * 再读这张纹理。只在开启美体且识别到身体时调用，其余时间不占 GPU。只在 GL 线程使用。
 * 返回后：GL_TEXTURE0 激活、unit 0 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。
 * 任何 GL 故障都抛异常，由调用方停用美体。
 */
internal class BodyWarpPass {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private var bodyA = -1
    private var bodyB = -1
    private var bodyC = -1
    private var aspect = -1
    private var texture = 0
    private var frame = 0
    private var w = 0
    private var h = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    /** [values] 前 12 个分量为 uBodyA、uBodyB、uBodyC（布局见 VideoEffectsShaders.BODY_FRAGMENT）。@return 变形后的纹理。 */
    fun render(source: Int, width: Int, height: Int, values: FloatArray): Int {
        check(width > 0 && height > 0 && values.size >= 12)
        var fresh = false
        if (program == 0) {
            program = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.BODY_FRAGMENT).also { check(it > 0) }
            position = glGetAttribLocation(program, "aPosition")
            coordinate = glGetAttribLocation(program, "aTextureCoord")
            sampler = glGetUniformLocation(program, "uSampler")
            bodyA = glGetUniformLocation(program, "uBodyA")
            bodyB = glGetUniformLocation(program, "uBodyB")
            bodyC = glGetUniformLocation(program, "uBodyC")
            aspect = glGetUniformLocation(program, "uAspect")
            fresh = true
        }
        if (w != width || h != height) {
            deleteBuffers()
            val ids = IntArray(1)
            glGenTextures(1, ids, 0); texture = ids[0]
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
            glGenFramebuffers(1, ids, 0); frame = ids[0]
            glBindFramebuffer(GL_FRAMEBUFFER, frame)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
            check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
            glBindTexture(GL_TEXTURE_2D, 0)
            w = width; h = height
            fresh = true
        }
        glBindFramebuffer(GL_FRAMEBUFFER, frame)
        glViewport(0, 0, width, height)
        glUseProgram(program)
        vertices.position(0)
        glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(position)
        vertices.position(2)
        glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(coordinate)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, source)
        glUniform1i(sampler, 0)
        glUniform4fv(bodyA, 1, values, 0)
        glUniform4fv(bodyB, 1, values, 4)
        glUniform4fv(bodyC, 1, values, 8)
        glUniform1f(aspect, width.toFloat() / height)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(position); glDisableVertexAttribArray(coordinate)
        glBindTexture(GL_TEXTURE_2D, 0)
        // 发布版 AAR 的 checkGlError 是空操作；只在（重新）分配那一帧查一次。
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return texture
    }

    private fun deleteBuffers() {
        if (texture != 0) glDeleteTextures(1, intArrayOf(texture), 0)
        if (frame != 0) glDeleteFramebuffers(1, intArrayOf(frame), 0)
        texture = 0; frame = 0; w = 0; h = 0
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() { texture = 0; frame = 0; w = 0; h = 0; program = 0 }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        deleteBuffers(); if (program != 0) glDeleteProgram(program)
        release()
    }
}
