package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.hypot

/**
 * 更多脸型与局部修饰（仅 Android）：源帧 → 同尺寸 RGBA8 中间纹理，按 [pack] 打包的参数做下巴、额头、瘦鼻局部变形，
 * 再做去黑眼圈、亮眼、白牙调色。只在至少一项强度 > 0 且有人脸时调用；只在 GL 线程使用。返回后：GL_TEXTURE0 激活、unit 0 未绑定、属性数组已关闭；
 * FBO 与 viewport 由调用方恢复。任何 GL 故障都抛异常，由调用方停用这三项。
 */
internal class FaceShapePass {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private val uniforms = IntArray(UNIFORMS.size)
    private var texture = 0
    private var frame = 0
    private var w = 0
    private var h = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    /** [values] 为 [pack] 的 [VALUES] 个分量。@return 处理后的纹理。 */
    fun render(source: Int, width: Int, height: Int, values: FloatArray): Int {
        check(width > 0 && height > 0 && values.size >= VALUES)
        var fresh = false
        if (program == 0) {
            program = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.FACE_FRAGMENT).also { check(it > 0) }
            position = glGetAttribLocation(program, "aPosition")
            coordinate = glGetAttribLocation(program, "aTextureCoord")
            sampler = glGetUniformLocation(program, "uSampler")
            for ((i, name) in UNIFORMS.withIndex()) uniforms[i] = glGetUniformLocation(program, name)
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
        for (i in UNIFORMS.indices) glUniform4fv(uniforms[i], 1, values, i * 4)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(position); glDisableVertexAttribArray(coordinate)
        glBindTexture(GL_TEXTURE_2D, 0)
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

    companion object {
        /** uniform 顺序即 [pack] 输出的 vec4 顺序。 */
        val UNIFORMS = listOf("uChin", "uForehead", "uNose", "uFaceAxis", "uEye0", "uEye1", "uMouth", "uRetouch")
        const val VALUES = 32

        /**
         * 从平滑后的人脸状态（41 维，布局见 FaceGeometrySmoother）推出各处中心与半径，按 [UNIFORMS] 顺序写进 [out]。
         * 强度 0–1 已乘过存在度（变形另乘侧脸权重）；变形换算成「位移 / 半径」比例：下巴最多 0.25、额头最多 0.2、
         * 鼻翼横向外扩最多 0.3；调色强度原样传入（着色器内另有上限）。@return 是否有任何一项生效。
         */
        fun pack(
            state: FloatArray, aspect: Float, chin: Float, forehead: Float, nose: Float, out: FloatArray,
            darkCircles: Float = 0f, brightEyes: Float = 0f, teeth: Float = 0f, folds: Float = 0f,
        ): Boolean {
            require(state.size >= FaceGeometrySmoother.STATE_SIZE && out.size >= VALUES)
            out.fill(0f)
            if (!(aspect > 0f) || (chin <= 0f && forehead <= 0f && nose <= 0f &&
                    darkCircles <= 0f && brightEyes <= 0f && teeth <= 0f && folds <= 0f)) return false
            val ux = state[26]; val uy = state[27]
            val n = hypot(ux, uy)
            if (n < 1e-4f) return false
            val axisX = ux / n; val axisY = uy / n
            val vx = -axisY; val vy = axisX
            // 椭圆：中心（纹理坐标）与度量半径；ry = 0.5 × |10−152| × 1.15，rx = 0.5 × 脸宽 × 1.12。
            val cx = state[22]; val cy = state[23] / aspect
            val halfHeight = state[25] / 1.15f
            val halfWidth = state[24] / 1.12f
            if (halfHeight < 0.01f || halfWidth < 0.01f) return false
            fun put(slot: Int, mx: Float, my: Float, radius: Float, strength: Float) {
                out[slot * 4] = mx; out[slot * 4 + 1] = my * aspect
                out[slot * 4 + 2] = radius; out[slot * 4 + 3] = strength
            }
            put(0, cx - vx * halfHeight, cy - vy * halfHeight, 0.55f * halfHeight, 0.25f * chin.coerceIn(0f, 1f))
            put(1, cx + vx * 0.9f * halfHeight, cy + vy * 0.9f * halfHeight, 0.75f * halfWidth, 0.2f * forehead.coerceIn(0f, 1f))
            // 鼻翼：双眼中点到嘴中心的 62% 处。
            val eyesX = (state[16] + state[19]) / 2; val eyesY = (state[17] + state[20]) / 2 / aspect
            val mouthX = state[28]; val mouthY = state[29] / aspect
            put(2, eyesX + 0.62f * (mouthX - eyesX), eyesY + 0.62f * (mouthY - eyesY), 0.38f * halfWidth, 0.3f * nose.coerceIn(0f, 1f))
            out[12] = axisX; out[13] = axisY; out[14] = aspect
            // 眼睛：eyeMasks[32..39] 已是 (中心, 横半径, 竖半径)；嘴：mouth[28..31]。
            state.copyInto(out, 16, 32, 40)
            state.copyInto(out, 24, 28, 32)
            out[28] = darkCircles.coerceIn(0f, 1f); out[29] = brightEyes.coerceIn(0f, 1f); out[30] = teeth.coerceIn(0f, 1f)
            out[31] = folds.coerceIn(0f, 1f)
            return out[3] > 0f || out[7] > 0f || out[11] > 0f || (28..31).any { out[it] > 0f }
        }

        /**
         * 基于 M3 网格微整后的真实 478 点关键点打包纯局部修饰参数（去黑眼圈、亮眼、白牙、淡化法令纹）。
         * 几何变形（下巴/额头/瘦鼻）已由 [FaceMeshWarpPass] 承担，此处强度固定置 0；鼻翼、上下睑与嘴角直接取真实关键点，
         * 大眼/瘦鼻/微笑唇后依然精准对齐无错位。
         */
        fun packRetouchFromMesh(
            landmarks: FloatArray,
            axisX: Float,
            axisY: Float,
            aspect: Float,
            darkCircles: Float,
            brightEyes: Float,
            teeth: Float,
            folds: Float,
            out: FloatArray,
        ): Boolean {
            require(landmarks.size >= FaceMeshFrame.FACE_COUNT * 2 && out.size >= VALUES)
            out.fill(0f)
            if (!(aspect > 0f) || (darkCircles <= 0f && brightEyes <= 0f && teeth <= 0f && folds <= 0f)) return false
            val n = hypot(axisX, axisY)
            if (n < 1e-4f) return false
            val ux = axisX / n; val uy = axisY / n
            val vx = -uy; val vy = ux

            fun lx(i: Int) = landmarks[i * 2]
            fun ly(i: Int) = landmarks[i * 2 + 1]
            fun mx(i: Int) = lx(i)
            fun my(i: Int) = ly(i) / aspect

            // 鼻翼真实中点与半宽（129 / 358），供 softenFolds 精准定位法令纹起点。
            val noseX = 0.5f * (lx(129) + lx(358))
            val noseY = 0.5f * (ly(129) + ly(358))
            val noseR = (0.5f * hypot(mx(358) - mx(129), my(358) - my(129)) * 1.10f).coerceAtLeast(0.01f)
            out[8] = noseX; out[9] = noseY; out[10] = noseR; out[11] = 0f
            out[12] = ux; out[13] = uy; out[14] = aspect

            // 真实双眼眼眶中心与宽高（右眼 33/133/159/145，左眼 362/263/386/374）。
            val eyeSpecs = intArrayOf(33, 133, 159, 145, 362, 263, 386, 374)
            for (i in 0 until 2) {
                val o = eyeSpecs[i * 4]; val inn = eyeSpecs[i * 4 + 1]
                val top = eyeSpecs[i * 4 + 2]; val bot = eyeSpecs[i * 4 + 3]
                val ew = hypot(mx(o) - mx(inn), my(o) - my(inn)).coerceAtLeast(0.01f)
                val eh = hypot(mx(top) - mx(bot), my(top) - my(bot))
                val cx = 0.25f * (lx(o) + lx(inn) + lx(top) + lx(bot))
                val cy = 0.25f * (ly(o) + ly(inn) + ly(top) + ly(bot))
                out[16 + i * 4] = cx
                out[17 + i * 4] = cy
                out[18 + i * 4] = 0.65f * ew
                out[19 + i * 4] = maxOf(0.55f * eh, 0.30f * ew)
            }

            // 真实嘴部中心与宽高（61/291/0/17），白牙与法令纹终点紧贴真实嘴角。
            val mouthX = 0.25f * (lx(61) + lx(291) + lx(0) + lx(17))
            val mouthY = 0.25f * (ly(61) + ly(291) + ly(0) + ly(17))
            val mw = hypot(mx(291) - mx(61), my(291) - my(61)).coerceAtLeast(0.01f)
            val mh = kotlin.math.abs((mx(0) - mx(17)) * vx + (my(0) - my(17)) * vy)
            out[24] = mouthX
            out[25] = mouthY
            out[26] = 0.60f * mw
            out[27] = maxOf(0.62f * mh, 0.25f * out[26])

            out[28] = darkCircles.coerceIn(0f, 1f)
            out[29] = brightEyes.coerceIn(0f, 1f)
            out[30] = teeth.coerceIn(0f, 1f)
            out[31] = folds.coerceIn(0f, 1f)
            return true
        }
    }
}
