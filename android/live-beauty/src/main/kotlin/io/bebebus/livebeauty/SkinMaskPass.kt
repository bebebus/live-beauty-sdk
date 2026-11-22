package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max

/**
 * 468 点多边形光栅化精准面部皮肤遮罩通道（仅 Android，v4 商业级管线，取代旧 4 椭圆近似遮罩）。
 *
 * 在 1/4 分辨率（如 180×320）离屏 FBO 直接光栅化变形后的 468 点人脸 + 36 点额头冠/下颌羽化环（共 504 顶点、970 三角形），
 * 再经 1 轮轻量高斯柔化输出三通道语义遮罩：
 * - `R (skin)`：纯面部皮肤（精确挖空双眼 32 点、双眉 20 点、双唇及口腔 80 点，并向上自然延伸额头冠区消除发际线截断）；
 * - `G (excl)`：眼、眉、唇精细保护权重（防止高强度磨皮与冷白提亮冲淡眉色、唇色与睫毛）；
 * - `B (hiPlane)`：鼻梁、额心、苹果肌、下巴 3D 骨相高光平面权重（驱动立体水光打光 `radiance` 塑造骨相立体感）；
 * - `A`：非头发权重，本道恒为 1；头发遮罩生效时由 SemanticAlphaPass 在 [outputFrame] 上只写 A（无脸时先 [renderEmpty]）。
 *   A 道不在本类内调用，故障域与本道分开。
 *
 * 静态语义权重在类加载期一次性预计算，每帧仅更新顶点 `(x, y)` 并发出单次 `glDrawElements` + 轻量羽化，耗时 < 0.1 ms。
 */
internal class SkinMaskPass {
    private var meshProg = 0
    private var meshPos = -1
    private var meshWeights = -1

    private var blurProg = 0
    private var blurPos = -1
    private var blurCoord = -1
    private var blurSampler = -1
    private var blurDir = -1

    private val textures = IntArray(2)
    private val frames = IntArray(2)
    private var sw = 0
    private var sh = 0

    // 504 顶点 × 5 floats = (ndcX, ndcY, skinR, exclG, hiPlaneB)，静态权重预填入槽。
    private val vertexArray = FloatArray(MASK_VERTEX_COUNT * STRIDE_FLOATS).apply {
        for (i in 0 until MASK_VERTEX_COUNT) {
            val k = i * STRIDE_FLOATS
            this[k + 2] = STATIC_WEIGHTS[i * 3]
            this[k + 3] = STATIC_WEIGHTS[i * 3 + 1]
            this[k + 4] = STATIC_WEIGHTS[i * 3 + 2]
        }
    }
    private val vertexBuffer = ByteBuffer
        .allocateDirect(vertexArray.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val indexBuffer = ByteBuffer
        .allocateDirect(MASK_INDEX_COUNT * 2)
        .order(ByteOrder.nativeOrder())
        .asShortBuffer()
        .apply {
            put(FaceMeshTopology.FULL_INDICES, 0, MASK_INDEX_COUNT)
            position(0)
        }
    private val quadVertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 0f, 0f, 1f, -1f, 1f, 0f, -1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f))
        position(0)
    }

    /**
     * @param landmarks 变形并经天鹅颈平移后的 478 点纹理坐标
     * @return 1/4 分辨率三通道语义遮罩纹理 `(R=skin, G=excl, B=hiPlane)`，输入退化时返回 0
     */
    fun render(
        landmarks: FloatArray,
        width: Int,
        height: Int,
        axisX: Float,
        axisY: Float,
    ): Int {
        if (width <= 0 || height <= 0 || landmarks.size < FaceMeshTopology.FACE_VERTEX_COUNT * 2) return 0
        val aspect = width.toFloat() / height
        val axisNorm = hypot(axisX, axisY)
        if (axisNorm < 1e-4f) return 0
        val ux = axisX / axisNorm; val uy = axisY / axisNorm
        val vx = -uy; val vy = ux

        // 计算度量空间面心与脸宽高，生成 36 个额头冠/外轮廓羽化环顶点。
        val m234x = landmarks[234 * 2]; val m234y = landmarks[234 * 2 + 1] / aspect
        val m454x = landmarks[454 * 2]; val m454y = landmarks[454 * 2 + 1] / aspect
        val m10x = landmarks[10 * 2]; val m10y = landmarks[10 * 2 + 1] / aspect
        val m152x = landmarks[152 * 2]; val m152y = landmarks[152 * 2 + 1] / aspect
        val fw = hypot(m454x - m234x, m454y - m234y)
        val fh = hypot(m10x - m152x, m10y - m152y)
        if (fw < 0.03f || fh < 0.03f) return 0
        val cx = 0.25f * (m234x + m454x + m10x + m152x)
        val cy = 0.25f * (m234y + m454y + m10y + m152y)

        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            val k = i * STRIDE_FLOATS
            vertexArray[k] = landmarks[i * 2] * 2f - 1f
            vertexArray[k + 1] = landmarks[i * 2 + 1] * 2f - 1f
        }
        val frontFaceCw = FaceMeshFrame.isFrontFaceCw(landmarks)
        val mirrorSign = if (frontFaceCw) -1f else 1f
        val canonicalUv = FaceMeshTopology.CANONICAL_UV
        val oval = FaceMeshTopology.FACE_OVAL
        for (j in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
            val oi = oval[j]
            val ox = landmarks[oi * 2]
            val oy = landmarks[oi * 2 + 1] / aspect
            val cNu = (canonicalUv[oi * 2] - 0.5f) * 2f * mirrorSign
            val cNv = (0.5f - canonicalUv[oi * 2 + 1]) * 2f
            val cLen = hypot(cNu, cNv).coerceAtLeast(1e-4f)
            val cxDir = (cNu * ux + cNv * vx) / cLen
            val cyDir = (cNu * uy + cNv * vy) / cLen
            val dx = ox - cx; val dy = oy - cy
            val len = hypot(dx, dy)
            val radX = if (len > 1e-4f) dx / len else cxDir
            val radY = if (len > 1e-4f) dy / len else cyDir
            val align = (radX * cxDir + radY * cyDir).coerceIn(0f, 1f)
            val wRad = 0.65f * align
            val mx = cxDir * (1f - wRad) + radX * wRad
            val my = cyDir * (1f - wRad) + radY * wRad
            val mNorm = hypot(mx, my).coerceAtLeast(1e-4f)
            val rx = mx / mNorm; val ry = my / mNorm
            val dv = dx * vx + dy * vy
            // 额头上半弧沿额头方向适度延伸 0.08 * fh 柔化发际线过渡，其余轮廓外扩 0.045 * fw 自然羽化且绝不越过头顶渗入背景墙。
            val crownMix = ((dv / fh - 0.22f) / 0.22f).coerceIn(0f, 1f)
            val pad = (1f - crownMix) * (0.045f * fw) + crownMix * (0.08f * fh)
            val dirX = (1f - crownMix * 0.50f) * rx + (crownMix * 0.50f) * vx
            val dirY = (1f - crownMix * 0.50f) * ry + (crownMix * 0.50f) * vy
            val dirNorm = hypot(dirX, dirY).coerceAtLeast(1e-4f)
            val sx = ox + (dirX / dirNorm) * pad
            val sy = (oy + (dirY / dirNorm) * pad) * aspect
            val k = (FaceMeshTopology.MID_BUFFER_OFFSET + j) * STRIDE_FLOATS
            vertexArray[k] = sx * 2f - 1f
            vertexArray[k + 1] = sy * 2f - 1f
        }

        var fresh = false
        if (meshProg == 0) {
            meshProg = GlUtil.createProgram(VideoEffectsShaders.SKIN_MASK_VERTEX, VideoEffectsShaders.SKIN_MASK_FRAGMENT).also { check(it > 0) }
            meshPos = glGetAttribLocation(meshProg, "aPosition")
            meshWeights = glGetAttribLocation(meshProg, "aMaskWeights")
            blurProg = GlUtil.createProgram(VideoEffectsShaders.BLUR_VERTEX, VideoEffectsShaders.BLUR_RGB_FRAGMENT).also { check(it > 0) }
            blurPos = glGetAttribLocation(blurProg, "aPosition")
            blurCoord = glGetAttribLocation(blurProg, "aTextureCoord")
            blurSampler = glGetUniformLocation(blurProg, "uSampler")
            blurDir = glGetUniformLocation(blurProg, "uDir")
            fresh = true
        }
        if (allocate(width, height)) fresh = true
        val w = sw
        val h = sh

        vertexBuffer.clear()
        vertexBuffer.put(vertexArray)

        glViewport(0, 0, w, h)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_BLEND)
        // 1. 按由远及近画家序光栅化 970 个多边形遮罩三角形 → T0：
        //    先画外围 72 个羽化环（关剔除），再画前景 898 个 3D 脸内三角形（开背面剔除），防止侧脸折向脸颊内侧的羽化环把前景脸颊皮肤遮罩冲淡成暗边残影。
        val innerCount = FaceMeshTopology.CANONICAL_TRIANGLE_COUNT * 3
        val featherCount = MASK_INDEX_COUNT - innerCount
        glBindFramebuffer(GL_FRAMEBUFFER, frames[0])
        glClearColor(0f, 0f, 0f, 1f)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(meshProg)
        val byteStride = STRIDE_FLOATS * 4
        if (meshPos >= 0) {
            vertexBuffer.position(0)
            glVertexAttribPointer(meshPos, 2, GL_FLOAT, false, byteStride, vertexBuffer)
            glEnableVertexAttribArray(meshPos)
        }
        if (meshWeights >= 0) {
            vertexBuffer.position(2)
            glVertexAttribPointer(meshWeights, 3, GL_FLOAT, false, byteStride, vertexBuffer)
            glEnableVertexAttribArray(meshWeights)
        }
        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        indexBuffer.position(innerCount)
        glDrawElements(GL_TRIANGLES, featherCount, GL_UNSIGNED_SHORT, indexBuffer)
        glEnable(GL_CULL_FACE)
        glFrontFace(if (frontFaceCw) GL_CW else GL_CCW)
        glCullFace(GL_BACK)
        indexBuffer.position(0)
        glDrawElements(GL_TRIANGLES, innerCount, GL_UNSIGNED_SHORT, indexBuffer)
        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        indexBuffer.position(0)
        if (meshPos >= 0) glDisableVertexAttribArray(meshPos)
        if (meshWeights >= 0) glDisableVertexAttribArray(meshWeights)

        // 2. 轻量高斯柔化（H + V）：让眼眉唇与侧脸下颌线多边形边缘呈丝滑抗锯齿过渡，消除 1/4 分辨率阶梯毛边。
        glUseProgram(blurProg)
        quadVertices.position(0)
        glVertexAttribPointer(blurPos, 2, GL_FLOAT, false, 16, quadVertices)
        glEnableVertexAttribArray(blurPos)
        quadVertices.position(2)
        glVertexAttribPointer(blurCoord, 2, GL_FLOAT, false, 16, quadVertices)
        glEnableVertexAttribArray(blurCoord)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(blurSampler, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, frames[1])
        glBindTexture(GL_TEXTURE_2D, textures[0])
        glUniform2f(blurDir, 1.10f / w, 0f)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindFramebuffer(GL_FRAMEBUFFER, frames[0])
        glBindTexture(GL_TEXTURE_2D, textures[1])
        glUniform2f(blurDir, 0f, 1.10f / h)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glDisableVertexAttribArray(blurPos)
        glDisableVertexAttribArray(blurCoord)
        glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[0]
    }

    /**
     * 本帧没有多边形遮罩、但要承载头发 A 通道时（无脸或遮罩故障）：只把输出清成 (0, 0, 0, 1)，即 face = excl = hiPlane = 0、
     * 非头发权重 1，随后由 SemanticAlphaPass 写 A。不建程序；FBO 与 viewport 由调用方恢复。
     * @return 与 [render] 同一张输出纹理
     */
    fun renderEmpty(width: Int, height: Int): Int {
        check(width > 0 && height > 0)
        val fresh = allocate(width, height)
        glBindFramebuffer(GL_FRAMEBUFFER, frames[0])
        glViewport(0, 0, sw, sh)
        glDisable(GL_SCISSOR_TEST)
        glClearColor(0f, 0f, 0f, 1f)
        glClear(GL_COLOR_BUFFER_BIT)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return textures[0]
    }

    /** 输出纹理（[render] / [renderEmpty] 的返回值）所挂的 FBO，供 SemanticAlphaPass 只写 A；未分配时为 0。 */
    val outputFrame: Int get() = frames[0]
    /** 输出尺寸（帧的 1/4，向上取整）；未分配时为 0。 */
    val outputWidth: Int get() = sw
    val outputHeight: Int get() = sh

    /** 按帧尺寸（重新）分配两张 1/4 分辨率 RGBA8 纹理与 FBO；尺寸不变时什么都不做。@return 是否刚分配。 */
    private fun allocate(width: Int, height: Int): Boolean {
        val w = ((width + 3) / 4).coerceAtLeast(1)
        val h = ((height + 3) / 4).coerceAtLeast(1)
        if (w == sw && h == sh) return false
        deleteBuffers()
        glGenTextures(2, textures, 0)
        glGenFramebuffers(2, frames, 0)
        sw = w; sh = h
        glActiveTexture(GL_TEXTURE0)
        for (i in 0..1) {
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
        return true
    }

    private fun deleteBuffers() {
        if (sw != 0) {
            glDeleteFramebuffers(2, frames, 0)
            glDeleteTextures(2, textures, 0)
        }
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
    }

    fun release() {
        frames.fill(0); textures.fill(0); sw = 0; sh = 0
        meshProg = 0; blurProg = 0
    }

    fun delete() {
        deleteBuffers()
        if (meshProg != 0) glDeleteProgram(meshProg)
        if (blurProg != 0) glDeleteProgram(blurProg)
        release()
    }

    companion object {
        private const val MASK_VERTEX_COUNT = FaceMeshTopology.FACE_VERTEX_COUNT + FaceMeshTopology.OVAL_VERTEX_COUNT
        private const val MASK_INDEX_COUNT =
            (FaceMeshTopology.CANONICAL_TRIANGLE_COUNT + FaceMeshTopology.INNER_BUFFER_TRIANGLE_COUNT) * 3
        private const val STRIDE_FLOATS = 5

        /**
         * 主合成的 uFrame.w：美颜关闭为 0；本帧多边形遮罩渲染成功（[polyMask]）且 presence > 0 时为 presence + 4（FRAGMENT_PRO
         * 读 uSkinMask 的 RGB），否则为 presence（回退椭圆遮罩）。不能按「皮肤遮罩纹理非 0」判断：无脸时 [renderEmpty] 也会出一张
         * 只承载头发 A 通道的纹理（RGB 为 0），按纹理判断会让椭圆回退路径读到 face = 0，磨皮就没了。
         */
        fun framePresence(on: Boolean, polyMask: Boolean, presence: Float): Float = when {
            !on -> 0f
            polyMask && presence > 0f -> presence + 4f
            else -> presence
        }

        /** 预计算 504 个顶点的静态 `(skinR, exclG, hiPlaneB)` 权重表。 */
        internal val STATIC_WEIGHTS: FloatArray = buildStaticWeights()

        private fun buildStaticWeights(): FloatArray {
            val out = FloatArray(MASK_VERTEX_COUNT * 3)
            val uv = FaceMeshTopology.CANONICAL_UV

            val eyeCore = setOf(
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
            )
            val browCore = setOf(52, 65, 53, 105, 63, 282, 295, 283, 334, 293)
            val browBorder = setOf(46, 55, 70, 66, 107, 276, 285, 300, 296, 336)
            val lipOuter = setOf(61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
            val lipInnerAndMid = HashSet<Int>(64).apply {
                addAll(listOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95))
                for (s in FaceMeshTopology.LIP_BAND_INDICES) {
                    val idx = s.toInt()
                    if (idx !in lipOuter) add(idx)
                }
            }
            val ovalSet = FaceMeshTopology.FACE_OVAL.toSet()

            fun gauss(u: Float, v: Float, cu: Float, cv: Float, su: Float, sv: Float): Float {
                val du = (u - cu) / su
                val dv = (v - cv) / sv
                return exp(-0.5f * (du * du + dv * dv))
            }

            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val u = uv[i * 2]
                val v = uv[i * 2 + 1]
                val (skin, excl) = when {
                    i in eyeCore || i in browCore || i in lipInnerAndMid -> 0f to 1f
                    i in browBorder || i in lipOuter -> 0.22f to 0.78f
                    i in ovalSet -> {
                        if (v < 0.32f) 0.80f to 0f else 0.68f to 0f
                    }
                    else -> 1f to 0f
                }

                // 3D 骨相高光区（鼻梁竖带 + 额心 T 区 + 左右苹果肌 C 区 + 下巴尖）。
                val bridge = gauss(u, v, 0.50f, 0.44f, 0.032f, 0.11f)
                val forehead = gauss(u, v, 0.50f, 0.22f, 0.11f, 0.065f) * 0.85f
                val cheekR = gauss(u, v, 0.36f, 0.48f, 0.065f, 0.052f) * 0.85f
                val cheekL = gauss(u, v, 0.64f, 0.48f, 0.065f, 0.052f) * 0.85f
                val chin = gauss(u, v, 0.50f, 0.82f, 0.055f, 0.042f) * 0.70f
                val hiPlane = max(max(bridge, forehead), max(max(cheekR, cheekL), chin)) * skin

                out[i * 3] = skin
                out[i * 3 + 1] = excl
                out[i * 3 + 2] = hiPlane
            }
            // 外圈 36 个冠/羽化环顶点全部归零（平滑过渡至人脸外背景）。
            for (j in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
                val base = (FaceMeshTopology.MID_BUFFER_OFFSET + j) * 3
                out[base] = 0f; out[base + 1] = 0f; out[base + 2] = 0f
            }
            return out
        }
    }
}
