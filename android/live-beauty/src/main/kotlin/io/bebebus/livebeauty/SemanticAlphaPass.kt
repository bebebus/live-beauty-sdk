package io.bebebus.livebeauty

import android.opengl.GLES20.*
import android.os.Trace
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 本帧网格微整的顶点与绕序，供 [SemanticAlphaPass.render] 让 A 通道跟着脸的变形走。
 * @property vertices [FaceMeshDeformer.vertexData]（dstNdcX, dstNdcY, srcTexX, srcTexY 交错）
 * @property frontFaceCw [FaceMeshDeformer.frontFaceCw]
 */
internal class MeshInput(var vertices: FloatArray, var frontFaceCw: Boolean)

/**
 * 头发遮罩 → 皮肤遮罩 A 通道（仅 Android，GLES 2.0，计划 2026-11-21 §4.3）：在 SkinMaskPass 的输出（帧的 1/4 分辨率）上
 * 用 glColorMask(F, F, F, T) 只写 A = 1 − shown · smoothstep(阈值, hair')，FRAGMENT_PRO 读成非头发权重 nonHair：头发处不磨皮、
 * 不调色，只按背景级锐化。RGB（face / excl / hiPlane）一个字节都不动。
 *
 * 先画全屏四边形（源 = 目标）；本帧做了网格微整时，再按 FaceMeshDeformer 的网格画一遍（剔除规则同 FaceMeshWarpPass），
 * 头发遮罩在相机系取样、按网格正向变形落点，不需要逆映射。遮罩采样时刻与显示时刻之间的头部运动由仿射补偿
 * （HairMaskTuning.affine）。阈值与脸内偏置见 HAIR_ALPHA_FRAGMENT。
 *
 * 只在 GL 线程使用。返回后（含抛异常）：colorMask 恢复全开、混合与背面剔除关闭、GL_TEXTURE0 激活且 unit 0 未绑定、属性数组
 * 已关闭；FBO 与 viewport 由调用方恢复。colorMask 一旦泄漏，后面各道的 RGB 都写不进去，所以放在 finally 里恢复。
 * glGetError 只在三种帧上查（每个 GL 上下文各一次，[release] 复位）：建程序那一帧、程序建好后的首次真实绘制、首次带网格的
 * 绘制。预热（[prepare]）只建程序、只查建程序本身，所以开播后第一次真正画 A 通道、第一次画网格仍各查一次。其余帧不查，
 * 只对 Kotlin 异常负责（与其余各道一致）。查出错误或抛异常时先尽力把目标的 A 清回 1（本帧回到无头发遮罩，即第一批画面），
 * 再抛给调用方停用本道（不影响 SkinMaskPass）。glGetError 读的是整帧累计的错误标志（每帧开头清过一次），同帧更早的某道
 * 留下的错误也会记到本道头上；停用方向是安全的：A 回到 1，画面逐字节等于第一批。
 */
internal class SemanticAlphaPass {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var semantic = -1
    private var hair = -1
    private var hairX0 = -1
    private var hairX1 = -1
    private var faceOval = -1
    private var faceAxis = -1
    // 本上下文内是否已校验过一次真实绘制 / 带网格的绘制（预热只建程序，不算）。
    private var drawVerified = false
    private var meshVerified = false

    // 预分配全屏四边形（x, y, s, t）与网格直连缓冲，渲染期零分配。
    private val quadBuffer = ByteBuffer
        .allocateDirect(16 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply {
            put(floatArrayOf(-1f, -1f, 0f, 0f, 1f, -1f, 1f, 0f, -1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f))
            position(0)
        }
    private val vertexBuffer = ByteBuffer
        .allocateDirect(FaceMeshTopology.TOTAL_VERTEX_COUNT * 4 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val indexBuffer = ByteBuffer
        .allocateDirect(FaceMeshTopology.FULL_INDICES.size * 2)
        .order(ByteOrder.nativeOrder())
        .asShortBuffer()
        .apply {
            put(FaceMeshTopology.FULL_INDICES)
            position(0)
        }

    /** 预热（GL 线程、未开播时由调用方调度）：只建程序、取 location。 */
    fun prepare() {
        if (ensureProgram()) check(glGetError() == GL_NO_ERROR)
    }

    /** @return 是否刚建了程序。 */
    private fun ensureProgram(): Boolean {
        if (program != 0) return false
        program = GlUtil.createProgram(VideoEffectsShaders.SEMANTIC_VERTEX, VideoEffectsShaders.HAIR_ALPHA_FRAGMENT).also { check(it > 0) }
        position = glGetAttribLocation(program, "aPosition")
        coordinate = glGetAttribLocation(program, "aTextureCoord")
        semantic = glGetUniformLocation(program, "uSemantic")
        hair = glGetUniformLocation(program, "uHair")
        hairX0 = glGetUniformLocation(program, "uHairX0")
        hairX1 = glGetUniformLocation(program, "uHairX1")
        faceOval = glGetUniformLocation(program, "uFaceOval")
        faceAxis = glGetUniformLocation(program, "uFaceAxis")
        return true
    }

    /**
     * @param target 本帧已出图（[SkinMaskPass.render] 或 [SkinMaskPass.renderEmpty]）的皮肤遮罩，在它的 [SkinMaskPass.outputFrame] 上写 A
     * @param hairTexture HairMaskTexture.texture（LUMINANCE，第 0 行是画面顶部）
     * @param shown 头发遮罩强度 0–1（HairMaskTuning.shown）；0 时 A 恒为 1，调用方本应直接不调
     * @param affine HairMaskTuning.affine 的输出（HairMaskTuning.AFFINE_SIZE 个数：uHairX0、uHairX1 两行，w 不用）
     * @param oval 显示帧的脸椭圆，[OVAL_SIZE] 个数 (cx, cy, rx, ry, axisX, axisY, aspect, 0)：与主合成 uOval 同一坐标，
     *   前 6 个即 FaceTimeline.poseAt 的输出，aspect = 帧宽 / 帧高
     * @param mesh 本帧做了网格微整时的网格，否则 null
     */
    fun render(target: SkinMaskPass, hairTexture: Int, shown: Float, affine: FloatArray, oval: FloatArray, mesh: MeshInput?) {
        val width = target.outputWidth
        val height = target.outputHeight
        check(width > 0 && height > 0 && target.outputFrame != 0 && hairTexture != 0)
        check(affine.size >= HairMaskTuning.AFFINE_SIZE && oval.size >= OVAL_SIZE)
        check(mesh == null || mesh.vertices.size >= FaceMeshTopology.TOTAL_VERTEX_COUNT * 4)
        Trace.beginSection("beauty.hair.alpha")
        try {
            val fresh = ensureProgram()
            glBindFramebuffer(GL_FRAMEBUFFER, target.outputFrame)
            glViewport(0, 0, width, height)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_SCISSOR_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_CULL_FACE)
            glColorMask(false, false, false, true)
            glUseProgram(program)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, hairTexture)
            glUniform1i(semantic, 0)
            glUniform4f(hair, shown.coerceIn(0f, 1f), ALPHA_LOW, ALPHA_HIGH, FACE_BIAS)
            glUniform4f(hairX0, affine[0], affine[1], affine[2], 0f)
            glUniform4f(hairX1, affine[4], affine[5], affine[6], 0f)
            glUniform4f(faceOval, oval[0], oval[1], oval[2], oval[3])
            glUniform4f(faceAxis, oval[4], oval[5], oval[6], 0f)

            // 1. 全屏四边形：源 = 目标。没有网格的地方、网格背面被剔除的地方都按原位取样。
            quadBuffer.position(0)
            glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, quadBuffer)
            glEnableVertexAttribArray(position)
            quadBuffer.position(2)
            glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, quadBuffer)
            glEnableVertexAttribArray(coordinate)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            // 2. 网格微整：同 FaceMeshWarpPass 由远及近画家序（先画外圈 2D 缓冲与锚点关背面剔除，再画前景 898 个 3D 脸内三角开背面剔除）。
            if (mesh != null) {
                val innerCount = FaceMeshTopology.CANONICAL_TRIANGLE_COUNT * 3
                val outerCount = FaceMeshTopology.FULL_INDICES.size - innerCount
                vertexBuffer.clear()
                vertexBuffer.put(mesh.vertices, 0, FaceMeshTopology.TOTAL_VERTEX_COUNT * 4)
                vertexBuffer.position(0)
                glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertexBuffer)
                vertexBuffer.position(2)
                glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, vertexBuffer)
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                indexBuffer.position(innerCount)
                glDrawElements(GL_TRIANGLES, outerCount, GL_UNSIGNED_SHORT, indexBuffer)
                glEnable(GL_CULL_FACE)
                glFrontFace(if (mesh.frontFaceCw) GL_CW else GL_CCW)
                glCullFace(GL_BACK)
                indexBuffer.position(0)
                glDrawElements(GL_TRIANGLES, innerCount, GL_UNSIGNED_SHORT, indexBuffer)
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                indexBuffer.position(0)
            }
            // 建程序那一帧、首次真实绘制、首次带网格的绘制各查一次：预热建好程序后 fresh 恒为 false，不能只靠它。
            if (fresh || !drawVerified || (mesh != null && !meshVerified)) {
                check(glGetError() == GL_NO_ERROR)
                drawVerified = true
                if (mesh != null) meshVerified = true
            }
        } catch (e: Exception) {
            // 尽力把 A 清回 1：本帧画面回到无头发遮罩（与第一批逐字节相同），RGB 不动。
            try {
                glBindFramebuffer(GL_FRAMEBUFFER, target.outputFrame)
                glViewport(0, 0, width, height)
                glDisable(GL_SCISSOR_TEST)
                glColorMask(false, false, false, true)
                glClearColor(0f, 0f, 0f, 1f)
                glClear(GL_COLOR_BUFFER_BIT)
            } catch (_: Exception) {
            }
            throw e
        } finally {
            // colorMask 泄漏会让后面各道的 RGB 都写不进去：无论成败都恢复。
            glColorMask(true, true, true, true)
            glDisable(GL_BLEND)
            glDisable(GL_CULL_FACE)
            glFrontFace(GL_CCW)
            if (position >= 0) glDisableVertexAttribArray(position)
            if (coordinate >= 0) glDisableVertexAttribArray(coordinate)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)
            Trace.endSection()
        }
    }

    /** 上下文已销毁（新上下文中遗忘旧句柄）；新上下文的首次绘制与首次网格绘制重新校验。 */
    fun release() {
        program = 0
        drawVerified = false
        meshVerified = false
    }

    /** 活上下文内同步释放程序。 */
    fun delete() {
        if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        /** [render] 的 oval 长度：uFaceOval = (cx, cy, rx, ry)，uFaceAxis = (axisX, axisY, aspect, 0)。 */
        const val OVAL_SIZE = 8
        /** 脸外：头发置信度 0.35 以下按非头发、0.65 以上按头发（pipeline 初值，计划 H18，待真机样本校准）。门禁 HA1 从本文件读取。 */
        const val ALPHA_LOW = 0.35f
        const val ALPHA_HIGH = 0.65f
        /** 脸椭圆内两个阈值都抬高这么多（0.50 / 0.80）：脸上被误检成头发的低置信斑块不至于不磨。 */
        const val FACE_BIAS = 0.15f
    }
}
