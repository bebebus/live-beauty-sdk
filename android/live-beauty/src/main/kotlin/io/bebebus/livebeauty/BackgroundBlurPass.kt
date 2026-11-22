package io.bebebus.livebeauty

import android.opengl.GLES20.*
import android.os.Trace
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 头发合并道的输入（计划 2026-11-21 §4.3）：传给 [BackgroundBlurPass.render]，shown > 0 时把头部附近的头发并进人像遮罩。
 * @property texture HairMaskTexture.texture（LUMINANCE，第 0 行是画面顶部，与人像遮罩同尺寸）
 * @property shown 头发遮罩强度 0–1（HairMaskTuning.shown）
 * @property affine HairMaskTuning.affine 的输出（显示帧 → 遮罩时刻，两行 vec4）
 * @property oval 显示帧的脸椭圆，同 SemanticAlphaPass.render 的 oval：(cx, cy, rx, ry, axisX, axisY, aspect, 0)
 */
internal class HairBackgroundInput(var texture: Int, var shown: Float, val affine: FloatArray, val oval: FloatArray)

/**
 * 背景虚化（仅 Android）：源帧 → 1/4 分辨率 → 水平/竖直高斯各两遍 → 按人像遮罩与原帧混合，输出同尺寸 RGBA8 纹理。
 * 遮罩由 [updateMask] 上传（LUMINANCE，行 0 为画面顶部）。只在需要虚化且已有遮罩时调用；只在 GL 线程使用。
 * 传入头发输入且 shown > 0 时，先在遮罩分辨率的独立 RGBA8 FBO 上跑 BG_HAIR 合并道（R = max(人像, 头部先验 · shown · 头发)，
 * 仍是第 0 行在顶部），BG_PREMASK 与 BG_MIX 改绑这张纹理（读 .r）；不传时绑原遮罩，与不接头发逐字节相同。
 * 合并道单独成故障域：出错时 [hairHealthy] 置 false（按 GL 上下文复位），本帧起改绑原遮罩，虚化本身不停。
 * BG_MIX 的细节保留（飞发，[DETAIL_KEEP]）只在传入人脸椭圆时生效，并按头部先验限制在头部附近，强度再乘「人脸近期可见」系数
 * （β = [detailKeep]）；不传人脸或系数为 0 时 β = 0。
 * 返回后：GL_TEXTURE0 激活、unit 0–2 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。除合并道外，任何 GL 故障都抛异常。
 * uniform/attribute 的 location 在建程序时取好；每帧不再按名查询，只在建程序或（重新）分配那一帧检查 glGetError。
 */
internal class BackgroundBlurPass {
    /** 一个程序用到的 location；该程序没有的名字为 −1（glUniform* 对 −1 是空操作）。 */
    private class Locations {
        var position = -1
        var coordinate = -1
        var sampler = -1
        var mask = -1
        var background = -1
        var step = -1
        var blurred = -1
        var person = -1
        var semantic = -1
        var hair = -1
        var hairX0 = -1
        var hairX1 = -1
        var faceOval = -1
        var faceAxis = -1

        fun load(program: Int) {
            position = glGetAttribLocation(program, "aPosition")
            coordinate = glGetAttribLocation(program, "aTextureCoord")
            sampler = glGetUniformLocation(program, "uSampler")
            mask = glGetUniformLocation(program, "uMask")
            background = glGetUniformLocation(program, "uBackground")
            step = glGetUniformLocation(program, "uStep")
            blurred = glGetUniformLocation(program, "uBlurred")
            person = glGetUniformLocation(program, "uPerson")
            semantic = glGetUniformLocation(program, "uSemantic")
            hair = glGetUniformLocation(program, "uHair")
            hairX0 = glGetUniformLocation(program, "uHairX0")
            hairX1 = glGetUniformLocation(program, "uHairX1")
            faceOval = glGetUniformLocation(program, "uFaceOval")
            faceAxis = glGetUniformLocation(program, "uFaceAxis")
        }
    }

    private var copy = 0
    private var blur = 0
    private var mix = 0
    private val copyAt = Locations()
    private val blurAt = Locations()
    private val mixAt = Locations()
    private var merge = 0
    private val mergeAt = Locations()
    private var merged = 0
    private var mergedFrame = 0
    private var mergedW = 0
    private var mergedH = 0
    /** 头发合并道是否可用；出过错就本上下文内不再合并（[release] 复位）。 */
    var hairHealthy = true
        private set
    private var maskTexture = 0
    private var maskW = 0
    private var maskH = 0
    /** 是否已上传过遮罩。 */
    var hasMask = false
        private set
    private val small = IntArray(2)
    private val smallFrames = IntArray(2)
    private var output = 0
    private var outputFrame = 0
    private var w = 0
    private var h = 0
    private var sw = 0
    private var sh = 0
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    fun updateMask(mask: PersonMask) {
        glActiveTexture(GL_TEXTURE0)
        // 尺寸不变时只更新内容，不重新分配；只在分配那一帧检查错误。
        val fresh = maskTexture == 0 || mask.width != maskW || mask.height != maskH
        if (maskTexture == 0) {
            val ids = IntArray(1); glGenTextures(1, ids, 0); maskTexture = ids[0]
            glBindTexture(GL_TEXTURE_2D, maskTexture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        } else {
            glBindTexture(GL_TEXTURE_2D, maskTexture)
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        if (fresh) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, mask.width, mask.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                ByteBuffer.wrap(mask.pixels))
            maskW = mask.width; maskH = mask.height
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mask.width, mask.height, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                ByteBuffer.wrap(mask.pixels))
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        hasMask = true
    }

    /** 预热（GL 线程、未开播时由调用方调度）：只建程序、取 location，不分配缓冲。 */
    fun prepare() {
        if (ensurePrograms()) check(glGetError() == GL_NO_ERROR)
    }

    /** 预热头发合并道的程序（与 [prepare] 分开：合并道编译失败只停合并，不停虚化）。 */
    fun prepareHair() {
        if (ensureMerge()) check(glGetError() == GL_NO_ERROR)
    }

    /** @return 是否刚建了合并道程序。 */
    private fun ensureMerge(): Boolean {
        if (merge != 0) return false
        merge = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.BG_HAIR_FRAGMENT).also { check(it > 0) }
        mergeAt.load(merge)
        return true
    }

    /** @return 是否刚建了程序。 */
    private fun ensurePrograms(): Boolean {
        if (copy != 0) return false
        copy = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.BG_PREMASK_FRAGMENT).also { check(it > 0) }
        blur = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.BG_BLUR_FRAGMENT).also { check(it > 0) }
        mix = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.BG_MIX_FRAGMENT).also { check(it > 0) }
        copyAt.load(copy); blurAt.load(blur); mixAt.load(mix)
        return true
    }

    /**
     * @return 虚化后的纹理。[strength] 0–1；[hair] 为 null（或 shown 为 0、合并道已停用）时绑原遮罩，与不接头发逐字节相同。
     * [face]：原帧（美体与网格变形之前）的脸椭圆，格式同 SemanticAlphaPass.render 的 oval：(cx, cy, rx, ry, axisX, axisY, aspect, 0)，
     * 与主合成 uOval 同一坐标。传入时 BG_MIX 的细节保留生效、只作用在头部附近（头部先验，评审 2026-11-21 #1）；
     * null（没有人脸）时细节保留不生效，与 β = 0 逐字节相同。
     * [faceWeight]：「人脸近期可见」系数 0..1（HairMaskGate.faceKeep），β = [detailKeep]：丢脸后在几何清空之前渐出、重新检出后渐入，
     * 细节保留不随人脸有无单帧硬切（评审复核）。为 1（缺省）时 β = [DETAIL_KEEP]，为 0 时与不传人脸逐字节相同。
     */
    fun render(source: Int, width: Int, height: Int, strength: Float, hair: HairBackgroundInput? = null,
               face: FloatArray? = null, faceWeight: Float = 1f): Int {
        check(width > 0 && height > 0 && hasMask)
        check(face == null || face.size >= SemanticAlphaPass.OVAL_SIZE)
        var fresh = ensurePrograms()
        if (w != width || h != height) { allocate(width, height); fresh = true }
        val mask = if (hair != null && hair.shown > 0f && hairHealthy) mergeHair(hair) else maskTexture
        // 1. 按人像遮罩预乘背景并缩小到 1/4（前景人像像素置零，彻底消除头发/肩膀向背景渗色的黑边光晕）。
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, mask)
        draw(copy, copyAt, smallFrames[0], sw, sh, source) {
            glUniform1i(copyAt.mask, 1)
            glUniform4f(copyAt.background, strength.coerceIn(0f, 1f), MASK_LOW, MASK_HIGH, 0f)
        }
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        // 2. 高斯：水平 → 竖直，两轮，第二轮步长加倍（对预乘 RGBA 同步模糊，保持准确背景权重归一化）。
        for (spread in SPREADS) {
            draw(blur, blurAt, smallFrames[1], sw, sh, small[0]) { glUniform2f(blurAt.step, spread / sw, 0f) }
            draw(blur, blurAt, smallFrames[0], sw, sh, small[1]) { glUniform2f(blurAt.step, 0f, spread / sh) }
        }
        // 3. 按遮罩混合到全分辨率输出；遮罩低置信、但亮度明显异于背景模糊图、又在头部附近的细节（飞发）按 β 保留。
        //    没有人脸时头部先验无处可依，细节保留整体关掉（β = 0）；有人脸时 β 随「人脸近期可见」渐变。
        val detail = detailKeep(face != null, faceWeight)
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, mask)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, small[0])
        draw(mix, mixAt, outputFrame, width, height, source) {
            glUniform1i(mixAt.blurred, 1)
            glUniform1i(mixAt.mask, 2)
            glUniform4f(mixAt.background, strength.coerceIn(0f, 1f), MASK_LOW, MASK_HIGH, detail)
            if (face != null) {
                glUniform4f(mixAt.faceOval, face[0], face[1], face[2], face[3])
                glUniform4f(mixAt.faceAxis, face[4], face[5], face[6], 0f)
            } else {
                glUniform4f(mixAt.faceOval, NO_FACE[0], NO_FACE[1], NO_FACE[2], NO_FACE[3])
                glUniform4f(mixAt.faceAxis, 1f, 0f, 1f, 0f)
            }
        }
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        // 发布版 AAR 的 checkGlError 是空操作；只在（重新）分配那一帧查一次。
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return output
    }

    /**
     * 头发合并道：在遮罩分辨率的独立 RGBA8 FBO 上跑 BG_HAIR，返回合并后的遮罩纹理（R 通道，第 0 行在顶部）。
     * 出错时停用合并道并返回原遮罩，本帧照常虚化。
     */
    private fun mergeHair(hair: HairBackgroundInput): Int {
        Trace.beginSection("beauty.bg.hair")
        try {
            check(hair.texture != 0 && hair.affine.size >= HairMaskTuning.AFFINE_SIZE && hair.oval.size >= SemanticAlphaPass.OVAL_SIZE)
            var fresh = ensureMerge()
            if (mergedW != maskW || mergedH != maskH) {
                deleteMerged()
                merged = texture(maskW, maskH); mergedFrame = framebuffer(merged)
                mergedW = maskW; mergedH = maskH
                fresh = true
            }
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, hair.texture)
            draw(merge, mergeAt, mergedFrame, mergedW, mergedH, maskTexture) {
                glUniform1i(mergeAt.person, 0)
                glUniform1i(mergeAt.semantic, 1)
                glUniform4f(mergeAt.hair, hair.shown.coerceIn(0f, 1f), HAIR_LOW, HAIR_HIGH, 0f)
                glUniform4f(mergeAt.hairX0, hair.affine[0], hair.affine[1], hair.affine[2], 0f)
                glUniform4f(mergeAt.hairX1, hair.affine[4], hair.affine[5], hair.affine[6], 0f)
                glUniform4f(mergeAt.faceOval, hair.oval[0], hair.oval[1], hair.oval[2], hair.oval[3])
                glUniform4f(mergeAt.faceAxis, hair.oval[4], hair.oval[5], hair.oval[6], 0f)
            }
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
            if (fresh) check(glGetError() == GL_NO_ERROR)
            return merged
        } catch (_: Exception) {
            hairHealthy = false
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
            return maskTexture
        } finally {
            Trace.endSection()
        }
    }

    private fun deleteMerged() {
        if (merged != 0) glDeleteTextures(1, intArrayOf(merged), 0)
        if (mergedFrame != 0) glDeleteFramebuffers(1, intArrayOf(mergedFrame), 0)
        merged = 0; mergedFrame = 0; mergedW = 0; mergedH = 0
    }

    private inline fun draw(program: Int, at: Locations, frame: Int, width: Int, height: Int, input: Int, uniforms: () -> Unit) {
        glBindFramebuffer(GL_FRAMEBUFFER, frame)
        glViewport(0, 0, width, height)
        glUseProgram(program)
        vertices.position(0)
        glVertexAttribPointer(at.position, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(at.position)
        vertices.position(2)
        glVertexAttribPointer(at.coordinate, 2, GL_FLOAT, false, 16, vertices)
        glEnableVertexAttribArray(at.coordinate)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, input)
        glUniform1i(at.sampler, 0)
        uniforms()
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(at.position); glDisableVertexAttribArray(at.coordinate)
    }

    private fun texture(width: Int, height: Int): Int {
        val ids = IntArray(1); glGenTextures(1, ids, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, ids[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
        glBindTexture(GL_TEXTURE_2D, 0)
        return ids[0]
    }

    private fun framebuffer(texture: Int): Int {
        val ids = IntArray(1); glGenFramebuffers(1, ids, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, ids[0])
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
        return ids[0]
    }

    private fun allocate(width: Int, height: Int) {
        deleteBuffers()
        sw = (width / 4).coerceAtLeast(1); sh = (height / 4).coerceAtLeast(1)
        for (i in 0..1) { small[i] = texture(sw, sh); smallFrames[i] = framebuffer(small[i]) }
        output = texture(width, height); outputFrame = framebuffer(output)
        w = width; h = height
    }

    private fun deleteBuffers() {
        if (small[0] != 0) glDeleteTextures(2, small, 0)
        if (smallFrames[0] != 0) glDeleteFramebuffers(2, smallFrames, 0)
        if (output != 0) glDeleteTextures(1, intArrayOf(output), 0)
        if (outputFrame != 0) glDeleteFramebuffers(1, intArrayOf(outputFrame), 0)
        small.fill(0); smallFrames.fill(0); output = 0; outputFrame = 0; w = 0; h = 0
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        small.fill(0); smallFrames.fill(0); output = 0; outputFrame = 0; w = 0; h = 0
        copy = 0; blur = 0; mix = 0; maskTexture = 0; maskW = 0; maskH = 0; hasMask = false
        merge = 0; merged = 0; mergedFrame = 0; mergedW = 0; mergedH = 0; hairHealthy = true
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        deleteBuffers()
        deleteMerged()
        if (maskTexture != 0) glDeleteTextures(1, intArrayOf(maskTexture), 0)
        for (program in intArrayOf(copy, blur, mix, merge)) if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        /** 人像置信度低于下限按背景、高于上限按人，收紧过渡带让人物边缘更干净锐利。 */
        const val MASK_LOW = 0.42f
        const val MASK_HIGH = 0.68f
        /**
         * 细节保留 β（BG_MIX 的 uBackground.w；BG_PREMASK 不用，仍传 0）：遮罩置信度偏低、但亮度与干净背景模糊图明显不同、又在
         * 头部附近的像素（1–3px 飞发、碎发）按 β 保留原样，阈值 [MASK_LOW]/[MASK_HIGH] 不必放宽。没有人脸时传 0。门禁 G8 从本文件读取。
         */
        const val DETAIL_KEEP = 1f
        /**
         * BG_MIX 的 β（uBackground.w）：没有人脸（[face] 为 false）为 0；有人脸时 = [DETAIL_KEEP] × clamp([faceWeight], 0, 1)，
         * NaN 按 0。随系数连续变化（Lipschitz 常数 [DETAIL_KEEP]），系数为 1 时恰为 [DETAIL_KEEP]（乘 1f 精确，与之前逐字节相同），
         * 为 0 时恰为 0（与没有人脸相同）。门禁 background_constants 静态核对这一写法，G8 按同一公式扫 β。
         */
        fun detailKeep(face: Boolean, faceWeight: Float): Float =
            if (face && faceWeight > 0f) DETAIL_KEEP * minOf(faceWeight, 1f) else 0f
        /** 没有人脸时 BG_MIX 的 uFaceOval：远在画面外（β 已为 0，只为不留上一帧的值）。 */
        private val NO_FACE = floatArrayOf(-4f, -4f, 0.01f, 0.01f)
        /**
         * 头发合并道的置信度阈值：头发类在发丝区多是软置信，比人像阈值低（计划 H8）。pipeline 初值 0.20/0.50 在头发遮罩软边
         * （σ3.63 遮罩像素，research 实测最软 9.3px）时把发缘外 3px 的背景也并进人像，超出成功标准 4 的低对比 halo ≤2px
         * （评审 2026-11-21 #19）；0.30/0.60 三种软边都是 0px，置信 0.6 的低对比金发飞发照样整根保留（门禁 G8 合并道变体、BH1）。
         * 代价：置信 0.30–0.45 的飞发不再被并进人像（待真机样本校准）。门禁 BH1 从本文件读取。
         */
        const val HAIR_LOW = 0.30f
        const val HAIR_HIGH = 0.60f
        private val SPREADS = floatArrayOf(1.5f, 3f)
    }
}
