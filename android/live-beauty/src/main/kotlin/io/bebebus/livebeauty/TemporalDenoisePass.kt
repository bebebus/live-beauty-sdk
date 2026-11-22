package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 暗光时域降噪（仅 Android）：源帧 + 上一帧结果 → 同尺寸 RGBA8 纹理（两张轮换，本帧输出即下一帧的历史）。
 * 只在需要降噪时调用；中断（尺寸变化、超过 [MAX_GAP_MS] 没调用）后第一帧权重为 0、严格直通，不混入过期历史。
 * 只在 GL 线程使用。返回后：GL_TEXTURE0 激活、unit 0/1 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。
 * 任何 GL 故障都抛异常，由调用方停用降噪。
 */
internal class TemporalDenoisePass {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private var history = -1
    private var params = -1
    private val textures = IntArray(2)
    private val frames = IntArray(2)
    private var current = 0
    private var w = 0
    private var h = 0
    private var lastRender = Long.MIN_VALUE
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    /** [weight] 为最大历史权重（0–[MAX_WEIGHT]）。@return 降噪后的纹理。 */
    fun render(source: Int, width: Int, height: Int, weight: Float, now: Long): Int {
        check(width > 0 && height > 0)
        var fresh = false
        if (program == 0) {
            program = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.DENOISE_FRAGMENT).also { check(it > 0) }
            position = glGetAttribLocation(program, "aPosition")
            coordinate = glGetAttribLocation(program, "aTextureCoord")
            sampler = glGetUniformLocation(program, "uSampler")
            history = glGetUniformLocation(program, "uHistory")
            params = glGetUniformLocation(program, "uDenoise")
            fresh = true
        }
        if (w != width || h != height) {
            deleteBuffers()
            glGenTextures(2, textures, 0)
            glGenFramebuffers(2, frames, 0)
            for (i in 0..1) {
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textures[i])
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
                glBindFramebuffer(GL_FRAMEBUFFER, frames[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0)
                check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
            }
            glBindTexture(GL_TEXTURE_2D, 0)
            w = width; h = height
            lastRender = Long.MIN_VALUE
            fresh = true
        }
        // 同一相机帧的第二次绘制（时间戳未知、复用判定不命中时）：直接沿用本帧结果，不让历史前进两次。
        if (!fresh && lastRender != Long.MIN_VALUE && now - lastRender in 0 until SAME_FRAME_MS) return textures[current]
        // 历史不连续：本帧不混合，只写入新的历史。
        val continuous = lastRender != Long.MIN_VALUE && now - lastRender in 0..MAX_GAP_MS
        lastRender = now
        val target = current xor 1
        glBindFramebuffer(GL_FRAMEBUFFER, frames[target])
        glViewport(0, 0, width, height)
        glUseProgram(program)
        vertices.position(0)
        glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(position)
        vertices.position(2)
        glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(coordinate)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, textures[current])
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, source)
        glUniform1i(sampler, 0)
        glUniform1i(history, 1)
        glUniform4f(params, if (continuous) weight.coerceIn(0f, MAX_WEIGHT) else 0f, DIFF_LOW, DIFF_HIGH, 0f)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(position); glDisableVertexAttribArray(coordinate)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        current = target
        // 发布版 AAR 的 checkGlError 是空操作；只在（重新）分配那一帧查一次。
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[current]
    }

    /** 下一次 render 从头开始（不混入历史）。 */
    fun interrupt() { lastRender = Long.MIN_VALUE }

    private fun deleteBuffers() {
        if (textures[0] != 0) glDeleteTextures(2, textures, 0)
        if (frames[0] != 0) glDeleteFramebuffers(2, frames, 0)
        textures.fill(0); frames.fill(0); current = 0; w = 0; h = 0; lastRender = Long.MIN_VALUE
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        textures.fill(0); frames.fill(0); current = 0; w = 0; h = 0; lastRender = Long.MIN_VALUE; program = 0
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        deleteBuffers(); if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        /** 仅极暗光静止区轻微融合传感器噪点，严格封顶 0.18 彻底杜绝时域拖尾残影。 */
        const val MAX_WEIGHT = 0.18f
        /** 逐像素最大通道差：高于 ~8/255 即按运动 100% 直通。 */
        const val DIFF_LOW = 0.010f
        const val DIFF_HIGH = 0.032f
        /** 超过这么久没降噪（丢帧、切后台、刚开启）就不再信任历史。 */
        const val MAX_GAP_MS = 150L
        /** 同一相机帧的预览与编码两次绘制间隔远小于一帧（30 fps 为 33 ms）。 */
        const val SAME_FRAME_MS = 4L

        /**
         * 仅在极暗光（亮度 ≤ 0.08 满强度，≥ 0.18 完全关闭）才开启微弱降噪，常规室内与日间零历史帧混合；非有限值关闭。
         */
        fun strengthForLuma(luma: Float): Float {
            if (!luma.isFinite()) return 0f
            return ((0.18f - luma) / 0.10f).coerceIn(0f, 1f) * MAX_WEIGHT
        }
    }
}
