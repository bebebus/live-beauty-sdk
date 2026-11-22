package io.bebebus.livebeauty

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES20.*
import android.opengl.GLUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors

/**
 * 实时局部三角网格美妆渲染通道（仅 Android，GLES 2.0）。
 *
 * 接在主合成（质感磨皮+通透肤色，3D LUT 已在主合成内查表）之后：
 * - 通过 [target] 接收主合成输出纹理，未开任何美妆时零开销完全跳过；
 * - [render] 先将底图原样拷贝至目标 FBO，再在对应局部子网格上按需叠加美妆层（节省 ~70% 1080p 填充率）：
 *   1. 立体修容 / 高光（852 面部皮肤三角形，Multiply / ScreenLift）
 *   2. 腮红（852 面部皮肤三角形，SoftLight 保留肤质纹理）
 *   3. 眉妆 & 卧蚕微阴影（852 面部皮肤三角形，Multiply）
 *   4. 眼影（主+深晕染）& 贴睑眼线 & 卧蚕提亮（852 面部皮肤三角形）
 *   5. 口红 / 水光唇釉（120 纯唇带三角形，内唇口腔零三角形物理防涂牙，保留原唇明暗纹路）
 *   6. 美瞳 & 灵动眼神光（14+14 眼裂内部三角形物理眼皮裁切 + 眨眼门控绝不穿帮）
 *
 * 5 张 512² 贴图在后台单线程解码（[ensureResources] 首次调用时提交），GL 线程只上传；某张贴图还没就绪的帧跳过用到它的妆容，
 * 不阻塞推流（直播中切到整妆风格时以前会在 GL 线程同步解码、卡一下）。[prepare] 只建程序、取 location，供预热调度。
 *
 * 只在 GL 线程使用；返回后 GL_BLEND 已关闭、GL_TEXTURE0 激活且未绑定、顶点属性数组已关闭。
 */
internal class MakeupPass(private val assets: AssetManager?) {
    private var copyProgram = 0
    private var copyPos = -1
    private var copyTex = -1
    private var copySampler = -1

    private var layerProgram = 0
    private var layerPos = -1
    private var layerScreenUv = -1
    private var layerMakeupUv = -1
    private var layerBase = -1
    private var layerAtlas = -1
    private var layerColor = -1
    private var layerChannel = -1
    private var layerMode = -1

    private var lipProgram = 0
    private var lipPos = -1
    private var lipScreenUv = -1
    private var lipMakeupUv = -1
    private var lipBase = -1
    private var lipAtlas = -1
    private var lipColor = -1
    private var lipStyle = -1

    private var irisProgram = 0
    private var irisPos = -1
    private var irisScreenUv = -1
    private var irisMakeupUv = -1
    private var irisBase = -1
    private var irisAtlas = -1
    private var irisCircle = -1
    private var irisTile = -1
    private var irisStrength = -1

    // 输入缓冲（接收主合成 FRAGMENT_PRO 输出）
    private var baseTexture = 0
    private var baseFbo = 0
    private var w = 0
    private var h = 0

    // 5 张通道打包美妆贴图（后台解码、GL 线程上传；0 表示还没就绪）
    private var blushTex = 0
    private var contourTex = 0
    private var browTex = 0
    private var eyeTex = 0
    private var irisTex = 0
    private val atlases = AtlasLoader()

    private val quadVertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 0f, 0f, 1f, -1f, 1f, 0f, -1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f))
        position(0)
    }
    private val meshVertices = FloatArray(FaceMakeupUv.VERTEX_DATA_SIZE)
    private val vertexBuffer = ByteBuffer
        .allocateDirect(FaceMakeupUv.VERTEX_DATA_SIZE * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val skinIndexBuf = directShortBuffer(FaceMakeupUv.SKIN_FACE_INDICES)
    private val lipIndexBuf = directShortBuffer(FaceMeshTopology.LIP_BAND_INDICES)
    private val rightEyeIndexBuf = directShortBuffer(FaceMeshTopology.RIGHT_EYE_INDICES)
    private val leftEyeIndexBuf = directShortBuffer(FaceMeshTopology.LEFT_EYE_INDICES)
    private val irisScratch = FloatArray(4)

    /** 分配/复用主合成写入的目标 FBO（供 VideoEffectsFilter.drawFilter 指向）。 */
    fun target(width: Int, height: Int): Int {
        check(width > 0 && height > 0)
        if (w != width || h != height || baseFbo == 0) {
            deleteTarget()
            val ids = IntArray(1)
            glGenTextures(1, ids, 0); baseTexture = ids[0]
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, baseTexture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
            glGenFramebuffers(1, ids, 0); baseFbo = ids[0]
            glBindFramebuffer(GL_FRAMEBUFFER, baseFbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, baseTexture, 0)
            check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
            glBindTexture(GL_TEXTURE_2D, 0)
            w = width; h = height
        }
        return baseFbo
    }

    /** 只建程序、取 location，不分配 FBO、不加载贴图（供预热调度在空闲帧提前编译）。 */
    fun prepare() {
        ensurePrograms()
    }

    /**
     * 将 [baseTexture] 加局部美妆后绘制到 [outputFbo]。[rightIrisGate]/[leftIrisGate] 为眨眼门控（0 = 闭眼收起美瞳，
     * 1 = 睁眼，见 FaceMeshFilter），映射到 uIrisTile.z = [IRIS_GATE_LO] + [IRIS_GATE_SPAN] × 门控。
     */
    fun render(
        outputFbo: Int,
        width: Int,
        height: Int,
        landmarks: FloatArray,
        aspect: Float,
        rightIrisGate: Float,
        leftIrisGate: Float,
        presence: Float,
        steadyFace: Float,
        steadyEye: Float,
        config: VideoEffectsConfig,
    ) {
        check(baseTexture != 0 && width > 0 && height > 0)
        val fresh = ensureResources()
        FaceMakeupUv.packVertices(landmarks, meshVertices)
        vertexBuffer.clear()
        vertexBuffer.put(meshVertices)

        val frontFaceCw = FaceMeshFrame.isFrontFaceCw(landmarks)
        glBindFramebuffer(GL_FRAMEBUFFER, outputFbo)
        glViewport(0, 0, width, height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

        // 1. 先原样拷贝磨皮底图到输出 FBO（全屏仅 1 次极轻采样）。
        glUseProgram(copyProgram)
        quadVertices.position(0)
        glVertexAttribPointer(copyPos, 2, GL_FLOAT, false, 16, quadVertices)
        glEnableVertexAttribArray(copyPos)
        quadVertices.position(2)
        glVertexAttribPointer(copyTex, 2, GL_FLOAT, false, 16, quadVertices)
        glEnableVertexAttribArray(copyTex)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, baseTexture)
        glUniform1i(copySampler, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(copyPos)
        glDisableVertexAttribArray(copyTex)

        // 2. 开启保 Alpha 混合与硬件背面剔除（侧脸转头时自动剔除翻折到鼻唇后方的远侧脸颊三角，杜绝修容/腮红印到鼻唇上）。
        //    剔除保留，可见度渐隐替代不了它，原因见 FaceMeshDeformer 的说明。
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)
        glEnable(GL_CULL_FACE)
        glFrontFace(if (frontFaceCw) GL_CW else GL_CCW)
        glCullFace(GL_BACK)

        val faceGain = presence * steadyFace
        val eyeGain = presence * steadyEye
        // 口红跟着脸走（与瘦脸同一尺度的外推渐出），不随眼睛尺度的 steadyEye 闪退。
        val lipGain = presence * steadyFace

        // 2a. 立体修容 / 高光 / 腮红 / 眉妆 / 眼妆（共享 852 三角形面部皮肤子网，避开眼球与牙齿）。
        val hasLayer = config.contour > 0f || config.highlight > 0f || config.blush > 0f ||
            config.eyebrow > 0f || config.eyeshadow > 0f || config.eyeliner > 0f || config.aegyoSal > 0f
        if (hasLayer && (faceGain > 0f || eyeGain > 0f)) {
            glUseProgram(layerProgram)
            bindMakeupAttributes(layerPos, layerScreenUv, layerMakeupUv)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, baseTexture)
            glUniform1i(layerBase, 0)
            glUniform1i(layerAtlas, 1)

            if (config.contour > 0f && faceGain > 0f) {
                drawLayer(contourTex, 0.34f, 0.25f, 0.20f, config.contour * 0.62f * faceGain, 1f, 0f, 0f, 0f, 1f)
            }
            if (config.highlight > 0f && faceGain > 0f) {
                drawLayer(contourTex, 1.0f, 0.96f, 0.92f, config.highlight * 0.68f * faceGain, 0f, 1f, 0f, 0f, 2f)
            }
            if (config.blush > 0f && faceGain > 0f) {
                val rgb = FaceMakeupUv.blushRgb(config.blushColor)
                val ch = FaceMakeupUv.blushChannel(config.blushStyle)
                drawLayer(blushTex, rgb[0], rgb[1], rgb[2], config.blush * 0.78f * faceGain, ch[0], ch[1], ch[2], ch[3], 0f)
            }
            if (config.eyebrow > 0f && eyeGain > 0f) {
                val rgb = FaceMakeupUv.eyebrowRgb(config.eyebrowColor)
                val ch = FaceMakeupUv.eyebrowChannel(config.eyebrowStyle)
                drawLayer(browTex, rgb[0], rgb[1], rgb[2], config.eyebrow * 0.80f * eyeGain, ch[0], ch[1], ch[2], ch[3], 1f, 1f)
            }
            if (config.eyeshadow > 0f && eyeGain > 0f) {
                val mainRgb = FaceMakeupUv.eyeshadowMainRgb(config.eyeshadowColor)
                val deepRgb = FaceMakeupUv.eyeshadowDeepRgb(config.eyeshadowColor)
                drawLayer(eyeTex, mainRgb[0], mainRgb[1], mainRgb[2], config.eyeshadow * 0.72f * eyeGain, 1f, 0f, 0f, 0f, 1f, 1f)
                drawLayer(eyeTex, deepRgb[0], deepRgb[1], deepRgb[2], config.eyeshadow * 0.68f * eyeGain, 0f, 1f, 0f, 0f, 1f, 1f)
            }
            if (config.eyeliner > 0f && eyeGain > 0f) {
                drawLayer(eyeTex, 0.14f, 0.11f, 0.10f, config.eyeliner * 0.90f * eyeGain, 0f, 0f, 1f, 0f, 1f, 1f)
            }
            if (config.aegyoSal > 0f && eyeGain > 0f) {
                // 卧蚕提亮（eyeTex.a, ScreenLift）+ 下缘微阴影（browTex.b, Multiply），均启用局部高清眉眼 UV。
                drawLayer(eyeTex, 1.0f, 0.94f, 0.88f, config.aegyoSal * 0.65f * eyeGain, 0f, 0f, 0f, 1f, 2f, 1f)
                drawLayer(browTex, 0.46f, 0.34f, 0.28f, config.aegyoSal * 0.42f * eyeGain, 0f, 0f, 1f, 0f, 1f, 1f)
            }
            unbindMakeupAttributes(layerPos, layerScreenUv, layerMakeupUv)
        }

        // 2b. 口红 / 唇釉（120 纯唇带三角形，口腔零三角形物理不涂牙；传入像素步长用于分离高频唇纹与去乌底色）。
        if (config.lipstick > 0f && lipGain > 0f && contourTex != 0) {
            val rgb = FaceMakeupUv.lipstickRgb(config.lipstickColor)
            val glossy = if (config.lipstickFinish == "glossy") 1f else 0f
            glUseProgram(lipProgram)
            bindMakeupAttributes(lipPos, lipScreenUv, lipMakeupUv)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, baseTexture)
            glUniform1i(lipBase, 0)
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, contourTex)
            glUniform1i(lipAtlas, 1)
            glUniform4f(lipColor, rgb[0], rgb[1], rgb[2], config.lipstick * 0.92f * lipGain)
            glUniform4f(lipStyle, glossy, 1f / width, 1f / height, 0f)
            lipIndexBuf.position(0)
            glDrawElements(GL_TRIANGLES, FaceMeshTopology.LIP_BAND_INDICES.size, GL_UNSIGNED_SHORT, lipIndexBuf)
            unbindMakeupAttributes(lipPos, lipScreenUv, lipMakeupUv)
        }

        // 2c. 美瞳 + 眼神光（仅在 14+14 眼裂内部三角形内光栅化，物理眼皮裁切 + 眨眼门控）。
        if ((config.contactLens > 0f || config.catchlight > 0f) && eyeGain > 0f && irisTex != 0) {
            val (tx, ty) = FaceMakeupUv.irisTile(config.contactLensStyle)
            glUseProgram(irisProgram)
            bindMakeupAttributes(irisPos, irisScreenUv, irisMakeupUv)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, baseTexture)
            glUniform1i(irisBase, 0)
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, irisTex)
            glUniform1i(irisAtlas, 1)
            glUniform4f(irisStrength, config.contactLens * 0.88f * eyeGain, config.catchlight * 0.95f * eyeGain, 0f, 0f)

            if (FaceMakeupUv.computeIrisCircle(landmarks, 468, aspect, irisScratch)) {
                glUniform4fv(irisCircle, 1, irisScratch, 0)
                glUniform4f(irisTile, tx, ty, IRIS_GATE_LO + IRIS_GATE_SPAN * rightIrisGate.coerceIn(0f, 1f), 0f)
                rightEyeIndexBuf.position(0)
                glDrawElements(GL_TRIANGLES, FaceMeshTopology.RIGHT_EYE_INDICES.size, GL_UNSIGNED_SHORT, rightEyeIndexBuf)
            }
            if (FaceMakeupUv.computeIrisCircle(landmarks, 473, aspect, irisScratch)) {
                glUniform4fv(irisCircle, 1, irisScratch, 0)
                glUniform4f(irisTile, tx, ty, IRIS_GATE_LO + IRIS_GATE_SPAN * leftIrisGate.coerceIn(0f, 1f), 0f)
                leftEyeIndexBuf.position(0)
                glDrawElements(GL_TRIANGLES, FaceMeshTopology.LEFT_EYE_INDICES.size, GL_UNSIGNED_SHORT, leftEyeIndexBuf)
            }
            unbindMakeupAttributes(irisPos, irisScreenUv, irisMakeupUv)
        }

        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
    }

    private fun drawLayer(
        atlasTex: Int,
        r: Float, g: Float, b: Float, strength: Float,
        c0: Float, c1: Float, c2: Float, c3: Float,
        mode: Float,
        eyeLocalUv: Float = 0f,
    ) {
        if (strength <= 1e-4f || atlasTex == 0) return
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, atlasTex)
        glUniform4f(layerColor, r, g, b, strength)
        glUniform4f(layerChannel, c0, c1, c2, c3)
        glUniform4f(layerMode, mode, eyeLocalUv, 0f, 0f)
        skinIndexBuf.position(0)
        glDrawElements(GL_TRIANGLES, FaceMakeupUv.SKIN_FACE_INDICES.size, GL_UNSIGNED_SHORT, skinIndexBuf)
    }

    private fun bindMakeupAttributes(posLoc: Int, screenLoc: Int, makeupLoc: Int) {
        val stride = FaceMakeupUv.VERTEX_STRIDE_FLOATS * 4
        if (posLoc >= 0) {
            vertexBuffer.position(0)
            glVertexAttribPointer(posLoc, 2, GL_FLOAT, false, stride, vertexBuffer)
            glEnableVertexAttribArray(posLoc)
        }
        if (screenLoc >= 0) {
            vertexBuffer.position(2)
            glVertexAttribPointer(screenLoc, 2, GL_FLOAT, false, stride, vertexBuffer)
            glEnableVertexAttribArray(screenLoc)
        }
        if (makeupLoc >= 0) {
            vertexBuffer.position(4)
            glVertexAttribPointer(makeupLoc, 2, GL_FLOAT, false, stride, vertexBuffer)
            glEnableVertexAttribArray(makeupLoc)
        }
    }

    private fun unbindMakeupAttributes(posLoc: Int, screenLoc: Int, makeupLoc: Int) {
        if (posLoc >= 0) glDisableVertexAttribArray(posLoc)
        if (screenLoc >= 0) glDisableVertexAttribArray(screenLoc)
        if (makeupLoc >= 0) glDisableVertexAttribArray(makeupLoc)
    }

    /** 建程序（首次）并上传已解码好的贴图；返回本次是否新建了 GL 对象（调用方据此检查 glGetError）。 */
    private fun ensureResources(): Boolean {
        var fresh = ensurePrograms()
        val mgr = checkNotNull(assets) { "makeup_assets_unavailable" }
        atlases.request(mgr)
        // 后台解码好的贴图在 GL 线程上传；解码失败与同步解码时一样算美妆故障（由调用方停用美妆）。
        while (true) {
            val decoded = atlases.poll() ?: break
            val bitmap = decoded.bitmap ?: error("makeup_decode_failed:${ATLAS_PATHS[decoded.index]}")
            val tex = try { uploadTexture(bitmap) } finally { bitmap.recycle() }
            when (decoded.index) {
                0 -> blushTex = tex
                1 -> contourTex = tex
                2 -> browTex = tex
                3 -> eyeTex = tex
                else -> irisTex = tex
            }
            fresh = true
        }
        return fresh
    }

    private fun ensurePrograms(): Boolean {
        var fresh = false
        if (copyProgram == 0) {
            copyProgram = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.ORIGINAL).also { check(it > 0) }
            copyPos = glGetAttribLocation(copyProgram, "aPosition")
            copyTex = glGetAttribLocation(copyProgram, "aTextureCoord")
            copySampler = glGetUniformLocation(copyProgram, "uSampler")
            fresh = true
        }
        if (layerProgram == 0) {
            layerProgram = GlUtil.createProgram(VideoEffectsShaders.MAKEUP_VERTEX, VideoEffectsShaders.MAKEUP_LAYER_FRAGMENT).also { check(it > 0) }
            layerPos = glGetAttribLocation(layerProgram, "aPosition")
            layerScreenUv = glGetAttribLocation(layerProgram, "aScreenUv")
            layerMakeupUv = glGetAttribLocation(layerProgram, "aMakeupUv")
            layerBase = glGetUniformLocation(layerProgram, "uBase")
            layerAtlas = glGetUniformLocation(layerProgram, "uAtlas")
            layerColor = glGetUniformLocation(layerProgram, "uColor")
            layerChannel = glGetUniformLocation(layerProgram, "uChannel")
            layerMode = glGetUniformLocation(layerProgram, "uMode")
            fresh = true
        }
        if (lipProgram == 0) {
            lipProgram = GlUtil.createProgram(VideoEffectsShaders.MAKEUP_VERTEX, VideoEffectsShaders.MAKEUP_LIP_FRAGMENT).also { check(it > 0) }
            lipPos = glGetAttribLocation(lipProgram, "aPosition")
            lipScreenUv = glGetAttribLocation(lipProgram, "aScreenUv")
            lipMakeupUv = glGetAttribLocation(lipProgram, "aMakeupUv")
            lipBase = glGetUniformLocation(lipProgram, "uBase")
            lipAtlas = glGetUniformLocation(lipProgram, "uAtlas")
            lipColor = glGetUniformLocation(lipProgram, "uLipColor")
            lipStyle = glGetUniformLocation(lipProgram, "uLipStyle")
            fresh = true
        }
        if (irisProgram == 0) {
            irisProgram = GlUtil.createProgram(VideoEffectsShaders.MAKEUP_VERTEX, VideoEffectsShaders.MAKEUP_IRIS_FRAGMENT).also { check(it > 0) }
            irisPos = glGetAttribLocation(irisProgram, "aPosition")
            irisScreenUv = glGetAttribLocation(irisProgram, "aScreenUv")
            irisMakeupUv = glGetAttribLocation(irisProgram, "aMakeupUv")
            irisBase = glGetUniformLocation(irisProgram, "uBase")
            irisAtlas = glGetUniformLocation(irisProgram, "uAtlas")
            irisCircle = glGetUniformLocation(irisProgram, "uIris")
            irisTile = glGetUniformLocation(irisProgram, "uIrisTile")
            irisStrength = glGetUniformLocation(irisProgram, "uIrisStrength")
            fresh = true
        }
        return fresh
    }

    /** 仅 GL 线程：把已解码的位图上传成纹理（不回收位图）。 */
    private fun uploadTexture(bitmap: Bitmap): Int {
        val ids = IntArray(1)
        glGenTextures(1, ids, 0)
        val tex = ids[0]
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GL_TEXTURE_2D, 0, bitmap, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex
    }

    private fun deleteTarget() {
        if (baseTexture != 0) glDeleteTextures(1, intArrayOf(baseTexture), 0)
        if (baseFbo != 0) glDeleteFramebuffers(1, intArrayOf(baseFbo), 0)
        baseTexture = 0; baseFbo = 0; w = 0; h = 0
    }

    private fun deleteAtlases() {
        val ids = intArrayOf(blushTex, contourTex, browTex, eyeTex, irisTex).filter { it != 0 }.toIntArray()
        if (ids.isNotEmpty()) glDeleteTextures(ids.size, ids, 0)
        blushTex = 0; contourTex = 0; browTex = 0; eyeTex = 0; irisTex = 0
    }

    /** 上下文已销毁，遗忘旧句柄；在途的解码结果作废，新上下文重新解码。 */
    fun release() {
        baseTexture = 0; baseFbo = 0; w = 0; h = 0
        blushTex = 0; contourTex = 0; browTex = 0; eyeTex = 0; irisTex = 0
        atlases.cancel()
        copyProgram = 0; layerProgram = 0; lipProgram = 0; irisProgram = 0
    }

    /** 活上下文内同步释放所有显存与程序。 */
    fun delete() {
        deleteTarget()
        deleteAtlases()
        if (copyProgram != 0) glDeleteProgram(copyProgram)
        if (layerProgram != 0) glDeleteProgram(layerProgram)
        if (lipProgram != 0) glDeleteProgram(lipProgram)
        if (irisProgram != 0) glDeleteProgram(irisProgram)
        release()
    }

    /**
     * 5 张贴图的后台解码（所有实例共用一个单线程守护线程）。GL 线程 [request] 提交一次、每帧 [poll] 取已解码的结果；
     * [cancel] 换代后，旧一代在途或未取走的位图直接回收，不会上传到新上下文。
     */
    private class AtlasLoader {
        class Decoded(val index: Int, val bitmap: Bitmap?)

        private val ready = ConcurrentLinkedQueue<Pair<Int, Decoded>>()
        @Volatile private var generation = 0
        private var requested = false // 仅 GL 线程

        fun request(mgr: AssetManager) {
            if (requested) return
            requested = true
            val gen = generation
            DECODER.execute {
                for (index in ATLAS_PATHS.indices) {
                    if (gen != generation) return@execute
                    val bitmap = try { decode(mgr, ATLAS_PATHS[index]) } catch (_: Throwable) { null }
                    if (gen != generation) { bitmap?.recycle(); return@execute }
                    ready.add(gen to Decoded(index, bitmap))
                    if (bitmap == null) return@execute
                }
            }
        }

        fun poll(): Decoded? {
            while (true) {
                val (gen, decoded) = ready.poll() ?: return null
                if (gen == generation) return decoded
                decoded.bitmap?.recycle()
            }
        }

        fun cancel() {
            generation++
            requested = false
            while (true) { val (_, decoded) = ready.poll() ?: break; decoded.bitmap?.recycle() }
        }

        private fun decode(mgr: AssetManager, path: String): Bitmap? {
            val opts = BitmapFactory.Options().apply {
                inPremultiplied = false
                inScaled = false
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            return mgr.open(path).use { BitmapFactory.decodeStream(it, null, opts) }
        }
    }

    private companion object {
        /** uIrisTile.z = IRIS_GATE_LO + IRIS_GATE_SPAN × 门控，与 MAKEUP_IRIS_FRAGMENT 的 smoothstep(0.10, 0.22) 绑定：门控 0 时正好 discard。 */
        const val IRIS_GATE_LO = 0.10f
        const val IRIS_GATE_SPAN = 0.12f
        /** 贴图解码顺序与下标：腮红、修容高光（兼唇纹理）、眉、眼妆、美瞳。 */
        val ATLAS_PATHS = arrayOf(
            "makeup/blush_atlas.png",
            "makeup/contour_highlight.png",
            "makeup/eyebrow_atlas.png",
            "makeup/eye_makeup.png",
            "makeup/iris_atlas.png",
        )
        val DECODER: java.util.concurrent.ExecutorService = Executors.newSingleThreadExecutor { r ->
            Thread(r, "makeup-atlas-decoder").apply { isDaemon = true }
        }

        fun directShortBuffer(values: ShortArray) = ByteBuffer
            .allocateDirect(values.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply { put(values); position(0) }
    }
}
