package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 双频快速导向滤波预处理（仅 Android，He-Sun-Tang Fast Guided Filter 1/3 分辨率保边 + 1/9 分辨率奶油肌宏匀光）。
 *
 * - P1 以 3×3 盒式均值降到 1/3 分辨率（720p 竖屏下 240×427）：抽头间距 [BeautyQualityTuning.DOWN_TAP_SCALE] = 2/3 原始像素；
 * - 求局部均值 mean，并把 RGB 偏差线性能量存进 alpha（GUIDED_DIFF_FRAGMENT）；
 * - 同步将局部均值降到 1/9 分辨率（720p 下仅 80×142，耗时 < 0.02 ms）做宽核高斯匀光得 `macroMean`，在 P5 / P5' 按骨相光比与边缘门控
 *   融合进线性重构偏移 `b = (1 - a) * evenMean`，一次性抚平 18–40 px 泪沟暗沉与面中色块，不占主合成任何额外纹理单元或 uniform；
 * - 头发加权路径（传入 `hairWeight`）下局部均值与宏均值均天然排除头发像素污染，发际线与胡须边界零暗带。
 */
internal class GuidedFilterPrePass {
    private class Program(vertex: String, fragment: String) {
        val id = GlUtil.createProgram(vertex, fragment).also { check(it > 0) }
        val position = glGetAttribLocation(id, "aPosition")
        val coordinate = glGetAttribLocation(id, "aTextureCoord")
        val sampler = glGetUniformLocation(id, "uSampler")
        val texel = glGetUniformLocation(id, "uTexel")
        val direction = glGetUniformLocation(id, "uDir")
        val original = glGetUniformLocation(id, "uOrig")
        val eps = glGetUniformLocation(id, "uGuidedEps")
        val weight = glGetUniformLocation(id, "uWeight")
        val macro = glGetUniformLocation(id, "uMacro")
        val evenness = glGetUniformLocation(id, "uEvenness")
    }

    private class Programs {
        val down = Program(VideoEffectsShaders.DOWN_VERTEX, VideoEffectsShaders.DOWN_FRAGMENT)
        val blurRgb = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_RGB_FRAGMENT)
        val guidedDiff = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.GUIDED_DIFF_FRAGMENT)
        val blurAlpha = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_ALPHA_FRAGMENT)
        val guidedCoeff = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.GUIDED_COEFF_FRAGMENT)
        fun delete() {
            for (p in arrayOf(down, blurRgb, guidedDiff, blurAlpha, guidedCoeff)) glDeleteProgram(p.id)
        }
    }

    /** 头发加权路径的 P2' / P3' / P5'，第一次传入头发权重时才建（编译失败只停加权，不停导向滤波）。 */
    private class WeightedPrograms {
        val blur = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.GUIDED_WEIGHT_BLUR_FRAGMENT)
        val diff = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.GUIDED_WEIGHT_DIFF_FRAGMENT)
        val coeff = Program(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.GUIDED_WEIGHT_COEFF_FRAGMENT)
        fun delete() {
            for (p in arrayOf(blur, diff, coeff)) glDeleteProgram(p.id)
        }
    }

    private var programs: Programs? = null
    private var weighted: WeightedPrograms? = null
    /** 头发加权路径是否可用；出过错就本上下文内不再加权（[release] 复位）。 */
    var weightHealthy = true
        private set
    private var weightVerified = false
    private val textures = IntArray(3)
    private val frames = IntArray(3)
    private val macroTextures = IntArray(2)
    private val macroFrames = IntArray(2)
    private var sw = 0
    private var sh = 0
    private var mw = 0
    private var mh = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 0f, 0f, 1f, -1f, 1f, 0f, -1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f))
        position(0)
    }

    fun render(source: Int, width: Int, height: Int, smoothing: Float, spread: Float = GuidedSpread.DEFAULT,
               hairWeight: Int = 0): Int {
        check(width > 0 && height > 0)
        var fresh = false
        val p = programs ?: Programs().also { programs = it; fresh = true }
        val w = ((width + 2) / 3).coerceAtLeast(1)
        val h = ((height + 2) / 3).coerceAtLeast(1)
        if (w != sw || h != sh) {
            deleteBuffers()
            glGenTextures(3, textures, 0)
            glGenFramebuffers(3, frames, 0)
            sw = w; sh = h
            mw = ((w + 2) / 3).coerceAtLeast(1)
            mh = ((h + 2) / 3).coerceAtLeast(1)
            glActiveTexture(GL_TEXTURE0)
            for (i in 0 until 3) {
                initAttachment(textures[i], frames[i], w, h)
            }
            glGenTextures(2, macroTextures, 0)
            glGenFramebuffers(2, macroFrames, 0)
            for (i in 0 until 2) {
                initAttachment(macroTextures[i], macroFrames[i], mw, mh)
            }
            glBindTexture(GL_TEXTURE_2D, 0)
            fresh = true
        }

        val step = if (spread.isFinite()) spread.coerceIn(GuidedSpread.MIN, GuidedSpread.MAX) else GuidedSpread.DEFAULT
        val dx = step / w
        val dy = step / h
        val eps = BeautyQualityTuning.guidedEps(smoothing)
        val even = BeautyQualityTuning.macroEvenness(smoothing)

        glViewport(0, 0, w, h)
        if (hairWeight != 0 && weightHealthy) {
            if (fresh) check(glGetError() == GL_NO_ERROR)
            val weightedResult = renderWeighted(p, source, width, height, w, h, dx, dy, eps, even, hairWeight, fresh)
            if (weightedResult != 0) return weightedResult
            glViewport(0, 0, w, h)
        }
        // P1 DOWN：src → T0。抽头 ±2/3 像素即 3×3 盒式均值；门禁核对这里用的是 DOWN_TAP_SCALE。
        pass(p.down, frames[0], source) {
            glUniform2f(it.texel, BeautyQualityTuning.DOWN_TAP_SCALE / width, BeautyQualityTuning.DOWN_TAP_SCALE / height)
        }
        // P2 H-BLUR：T0 → T1。
        pass(p.blurRgb, frames[1], textures[0]) { glUniform2f(it.direction, dx, 0f) }
        // P3 V-DIFF：T1 + T0 → T2 = (mean.rgb, clamp(|T0 − mean| * 2.5, 0, 1))。
        pass(p.guidedDiff, frames[2], textures[1], textures[0]) {
            glUniform2f(it.direction, 0f, dy)
            glUniform1i(it.original, 1)
        }
        val macroTex = if (even > 0f) renderMacro(p, textures[2], w, h) else macroTextures[0]
        // P4 H-VAR：T2 → T1（模糊 alpha 能量，保留 mean.rgb）。
        pass(p.blurAlpha, frames[1], textures[2]) { glUniform2f(it.direction, dx, 0f) }
        // P5 V-COEFF：T1 + macro → T0 = ((1 - a) * evenMean, a)。
        pass(p.guidedCoeff, frames[0], textures[1], macroTex) {
            glUniform2f(it.direction, 0f, dy)
            glUniform1f(it.eps, eps)
            if (it.macro >= 0) glUniform1i(it.macro, 1)
            if (it.evenness >= 0) glUniform1f(it.evenness, even)
        }
        // P6/P7 COEFF-BLUR：T0 → T1 → T2 = (meanB.rgb, meanA)。
        pass(p.blurRgb, frames[1], textures[0]) { glUniform2f(it.direction, dx, 0f) }
        pass(p.blurRgb, frames[2], textures[1]) { glUniform2f(it.direction, 0f, dy) }

        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[2]
    }

    private fun renderWeighted(
        p: Programs,
        source: Int,
        width: Int,
        height: Int,
        w: Int,
        h: Int,
        dx: Float,
        dy: Float,
        eps: Float,
        even: Float,
        hairWeight: Int,
        fresh: Boolean,
    ): Int {
        try {
            var verify = fresh || !weightVerified
            val q = weighted ?: WeightedPrograms().also { weighted = it; verify = true }
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, hairWeight)
            // P1 DOWN：同原路径。
            pass(p.down, frames[0], source) {
                glUniform2f(it.texel, BeautyQualityTuning.DOWN_TAP_SCALE / width, BeautyQualityTuning.DOWN_TAP_SCALE / height)
            }
            // P2' H-BLUR：T0 × 非头发权重 → T1 = (Σw·I, Σw)。
            pass(q.blur, frames[1], textures[0]) {
                glUniform2f(it.direction, dx, 0f)
                glUniform1i(it.weight, 2)
            }
            // P3' V-DIFF：T1 + T0 → T2 = (非头发均值, clamp(|T0 − 均值| × 2.5))。
            pass(q.diff, frames[2], textures[1], textures[0]) {
                glUniform2f(it.direction, 0f, dy)
                glUniform1i(it.original, 1)
            }
            val macroTex = if (even > 0f) renderMacro(p, textures[2], w, h) else macroTextures[0]
            // P4 H-VAR：同原路径。
            pass(p.blurAlpha, frames[1], textures[2]) { glUniform2f(it.direction, dx, 0f) }
            // P5' V-COEFF：T1 + macro → T0，头发像素取恒等系数。
            pass(q.coeff, frames[0], textures[1], macroTex) {
                glUniform2f(it.direction, 0f, dy)
                glUniform1f(it.eps, eps)
                glUniform1i(it.weight, 2)
                if (it.macro >= 0) glUniform1i(it.macro, 1)
                if (it.evenness >= 0) glUniform1f(it.evenness, even)
            }
            // P6/P7：同原路径。
            pass(p.blurRgb, frames[1], textures[0]) { glUniform2f(it.direction, dx, 0f) }
            pass(p.blurRgb, frames[2], textures[1]) { glUniform2f(it.direction, 0f, dy) }
            unbind()
            if (verify) {
                check(glGetError() == GL_NO_ERROR)
                weightVerified = true
            }
            return textures[2]
        } catch (_: Exception) {
            weightHealthy = false
            unbind()
            for (i in 0 until 8) if (glGetError() == GL_NO_ERROR) break
            return 0
        }
    }

    /** 在 1/9 分辨率求大核肤色匀光参考均值（约 32 px 全屏等效尺度），完成后恢复 1/3 viewport。 */
    private fun renderMacro(p: Programs, fineMeanTex: Int, w: Int, h: Int): Int {
        glViewport(0, 0, mw, mh)
        val macroTap = 0.6666667f
        pass(p.down, macroFrames[0], fineMeanTex) {
            glUniform2f(it.texel, macroTap / w, macroTap / h)
        }
        val mdx = 1.55f / mw
        val mdy = 1.55f / mh
        pass(p.blurRgb, macroFrames[1], macroTextures[0]) { glUniform2f(it.direction, mdx, 0f) }
        pass(p.blurRgb, macroFrames[0], macroTextures[1]) { glUniform2f(it.direction, 0f, mdy) }
        glViewport(0, 0, w, h)
        return macroTextures[0]
    }

    private fun initAttachment(tex: Int, fbo: Int, w: Int, h: Int) {
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    }

    private fun unbind() {
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
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
        if (second != 0) {
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, second)
        }
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, first)
        glUniform1i(program.sampler, 0)
        uniforms(program)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(program.position)
        glDisableVertexAttribArray(program.coordinate)
    }

    private fun deleteBuffers() {
        if (sw != 0) {
            glDeleteFramebuffers(3, frames, 0)
            glDeleteTextures(3, textures, 0)
            glDeleteFramebuffers(2, macroFrames, 0)
            glDeleteTextures(2, macroTextures, 0)
        }
        frames.fill(0); textures.fill(0); macroFrames.fill(0); macroTextures.fill(0)
        sw = 0; sh = 0; mw = 0; mh = 0
    }

    fun release() {
        frames.fill(0); textures.fill(0); macroFrames.fill(0); macroTextures.fill(0)
        sw = 0; sh = 0; mw = 0; mh = 0
        programs = null
        weighted = null; weightHealthy = true; weightVerified = false
    }

    fun delete() {
        deleteBuffers()
        programs?.delete()
        weighted?.delete()
        release()
    }
}
