package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 快速导向滤波预处理（仅 Android，He-Sun-Tang Fast Guided Filter 的 1/3 分辨率近似）。
 *
 * - P1 以 3×3 盒式均值降到 1/3 分辨率（720p 竖屏下 240×427）：抽头间距 [BeautyQualityTuning.DOWN_TAP_SCALE] = 2/3 原始像素，
 *   4 次双线性取样覆盖全部 9 个像素（见 DOWN_VERTEX）；
 * - 求局部均值 mean，并把 RGB 偏差的线性能量 clamp(|I − mean| × 2.5) 存进 alpha
 *   （GUIDED_DIFF_FRAGMENT；线性编码让 RGBA8 在 σ ≈ 1/255 处仍有分辨率）；
 * - 模糊能量得 s，按软映射求系数 `a = r^1.5 / (1 + r^1.5)`、`r = s² / eps`，并以 smoothstep(lo, 0.30, s) 把强边缘锁到 1，
 *   锁下限 lo = clamp(√eps − 0.09, 0.12, 0.18) 随磨皮档抬高（GUIDED_COEFF_FRAGMENT）；eps 随磨皮强度取
 *   [BeautyQualityTuning.guidedEps]；
 * - `(b, a) = ((1 − a)·mean, a)` 打包进 RGBA8，再在 1/3 分辨率经 P6/P7 二次平滑；
 * - 主合成在全分辨率按 `q = a·I + b` 重构：平坦皮肤 a 小、趋向局部均值，发丝、睫毛、唇线等高对比处 a 趋近 1、接近原图。
 *   系数是 1/3 分辨率求得再放大的，硬边两侧仍会有 1–2 像素的过渡，主合成另有 RGB 距离护栏兜底。
 * 采样间距 spread（1/3 分辨率 texel）随人脸大小自适应，见 [GuidedSpread]。
 *
 * 头发加权（毛发边界第二批，评审 2026-11-21 #0）：传入 hairWeight（本帧 SemanticAlphaPass 写过 A 的皮肤遮罩，A = 非头发权重）时，
 * P2/P3 改为按非头发权重预乘、模糊后归一化求局部均值（GUIDED_WEIGHT_BLUR / GUIDED_WEIGHT_DIFF）：发缘皮肤侧的均值不再
 * 混进头发，中等、偏深肤色配棕发、金发时额头一侧的暗带随之消失（门禁 G1 矩阵：棕发 × 中等肤色默认档 5.93 → 0.88/255）。
 * 能量不加权；P5 另让头发像素取恒等系数（GUIDED_WEIGHT_COEFF）。
 * 不传（0）时走原路径，与不接头发逐字节相同。加权路径单成故障域：出错时 [weightHealthy] 置 false（[release] 复位），
 * 本帧起退回原路径，导向滤波本身不停。
 *
 * 只在 GL 线程使用；返回后 GL_TEXTURE0 激活、unit 0–2 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。
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
    // 本上下文内是否已校验过一次加权路径的真实绘制（建程序那一帧之外，首次加权绘制也查一次 glGetError）。
    private var weightVerified = false
    private val textures = IntArray(3)
    private val frames = IntArray(3)
    private var sw = 0
    private var sh = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 0f, 0f, 1f, -1f, 1f, 0f, -1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f))
        position(0)
    }

    /**
     * [smoothing] 决定 eps（[BeautyQualityTuning.guidedEps]），[spread] 为模糊采样间距（[GuidedSpread.value]）。
     * [hairWeight]：本帧 SemanticAlphaPass 成功写过 A 通道的 SkinMaskPass 输出纹理（skinMaskTexture，A = 非头发权重，
     * 与主合成 uSkinMask 同一张），传 0 或加权路径已停用时走原路径（与不接头发逐字节相同）。
     * @return 存放 `(meanB.rgb, meanA)` 的 1/3 分辨率 RGBA8 纹理 T2。
     */
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

        val step = if (spread.isFinite()) spread.coerceIn(GuidedSpread.MIN, GuidedSpread.MAX) else GuidedSpread.DEFAULT
        val dx = step / w
        val dy = step / h
        val eps = BeautyQualityTuning.guidedEps(smoothing)

        glViewport(0, 0, w, h)
        if (hairWeight != 0 && weightHealthy) {
            // 建程序或（重新）分配那一帧先查这些操作本身的错误，免得被加权路径记到自己头上、吞掉。
            if (fresh) check(glGetError() == GL_NO_ERROR)
            val weightedResult = renderWeighted(p, source, width, height, dx, dy, eps, hairWeight, fresh)
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
        // P4 H-VAR：T2 → T1（模糊 alpha 能量，保留 mean.rgb）。
        pass(p.blurAlpha, frames[1], textures[2]) { glUniform2f(it.direction, dx, 0f) }
        // P5 V-COEFF：T1 → T0 = ((1 - a) * mean, a)。
        pass(p.guidedCoeff, frames[0], textures[1]) {
            glUniform2f(it.direction, 0f, dy)
            glUniform1f(it.eps, eps)
        }
        // P6/P7 COEFF-BLUR：T0 → T1 → T2 = (meanB.rgb, meanA)。
        pass(p.blurRgb, frames[1], textures[0]) { glUniform2f(it.direction, dx, 0f) }
        pass(p.blurRgb, frames[2], textures[1]) { glUniform2f(it.direction, 0f, dy) }

        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[2]
    }

    /**
     * 头发加权路径：P1 → P2'（按非头发权重预乘横向模糊）→ P3'（归一化均值与能量）→ P4 → P5'（头发取恒等系数）→ P6/P7。
     * 头发权重绑在纹理单元 2。建加权程序那一帧、本上下文首次加权绘制、以及（重新）分配那一帧（[fresh]）查一次 glGetError。
     * 出错时停用加权（[weightHealthy] = false）、尽力清掉本路径留下的 GL 错误并返回 0，由调用方当帧改走原路径。
     */
    private fun renderWeighted(p: Programs, source: Int, width: Int, height: Int, dx: Float, dy: Float, eps: Float,
                               hairWeight: Int, fresh: Boolean): Int {
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
            // P4 H-VAR：同原路径。
            pass(p.blurAlpha, frames[1], textures[2]) { glUniform2f(it.direction, dx, 0f) }
            // P5' V-COEFF：T1 → T0，头发像素取恒等系数。
            pass(q.coeff, frames[0], textures[1]) {
                glUniform2f(it.direction, 0f, dy)
                glUniform1f(it.eps, eps)
                glUniform1i(it.weight, 2)
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
            // 清掉本路径留下的错误，免得当帧原路径的分配检查把它记到导向滤波头上（有界，丢上下文时不死循环）。
            for (i in 0 until 8) if (glGetError() == GL_NO_ERROR) break
            return 0
        }
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
        }
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
    }

    fun release() {
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
        programs = null
        weighted = null; weightHealthy = true; weightVerified = false
    }

    fun delete() {
        programs?.delete()
        weighted?.delete()
        deleteBuffers()
        release()
    }

}
