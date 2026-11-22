package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 基于 MediaPipe 478 点 2D 三角网格的精细微整渲染通道（仅 Android，GLES 2.0）。
 *
 * 使用 `glDrawElements` 一次渲染覆盖全屏的 1014 个水密三角形（898 脸部 + 72 缓冲过渡 + 44 屏幕外恒等背景）；
 * 片元着色器仅 1 个 sampler uniform 与 1 行采样，无 16-uniform 上限压力、低填充开销、背景门框直线零弯曲。
 * 只在 GL 线程使用；返回后 GL_TEXTURE0 激活、unit 0 未绑定、顶点属性数组已关闭；FBO 与 viewport 由调用方恢复。
 */
internal class FaceMeshWarpPass {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private var texture = 0
    private var frame = 0
    private var w = 0
    private var h = 0

    // 预分配全屏原图兜底四边形与直连缓冲（512 × 4 floats = 8 KB，3042 shorts = 6 KB），渲染期零分配。
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

    /**
     * @param source 输入帧纹理 ID
     * @param vertices [FaceMeshDeformer.vertexData]（长度 2048，排布为 dstNdcX, dstNdcY, srcTexX, srcTexY）
     * @param frontFaceCw 正面绕序是否为顺时针（常规点集 false = CCW，镜像点集 true = CW）
     * @return 变形后的全分辨率 RGBA8 纹理 ID
     */
    fun render(source: Int, width: Int, height: Int, vertices: FloatArray, frontFaceCw: Boolean = false): Int {
        check(width > 0 && height > 0 && vertices.size >= FaceMeshTopology.TOTAL_VERTEX_COUNT * 4)
        var fresh = false
        if (program == 0) {
            program = GlUtil.createProgram(VideoEffectsShaders.MESH_WARP_VERTEX, VideoEffectsShaders.MESH_WARP_FRAGMENT).also { check(it > 0) }
            position = glGetAttribLocation(program, "aPosition")
            coordinate = glGetAttribLocation(program, "aTextureCoord")
            sampler = glGetUniformLocation(program, "uSampler")
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
        vertexBuffer.clear()
        vertexBuffer.put(vertices, 0, FaceMeshTopology.TOTAL_VERTEX_COUNT * 4)

        glBindFramebuffer(GL_FRAMEBUFFER, frame)
        glViewport(0, 0, width, height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glUseProgram(program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source)
        glUniform1i(sampler, 0)

        // 1. 先绘制未变形原图全屏底座（替代黑底 glClear，从根源上杜绝任何被剔除背面三角处露出黑洞）。
        quadBuffer.position(0)
        glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, quadBuffer)
        glEnableVertexAttribArray(position)
        quadBuffer.position(2)
        glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, quadBuffer)
        glEnableVertexAttribArray(coordinate)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        // 2. 按真实 3D 由远及近画家序绘制 1086 个水密变形网格三角形：
        //    - 先画外圈 188 个 2D 双环 C1 缓冲与屏幕锚点三角形（关背面剔除，位于 3D 脸颊折线后方/外侧）；
        //    - 再画前景 898 个 3D 脸内三角形（开硬件背面剔除，覆盖在缓冲环之上，彻底杜绝折向脸后的 FACE_OVAL 缓冲带反盖瘦脸脸颊形成双轮廓残影）。
        val innerCount = FaceMeshTopology.CANONICAL_TRIANGLE_COUNT * 3
        val outerCount = FaceMeshTopology.FULL_INDICES.size - innerCount
        vertexBuffer.position(0)
        glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertexBuffer)
        vertexBuffer.position(2)
        glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, vertexBuffer)
        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        indexBuffer.position(innerCount)
        glDrawElements(GL_TRIANGLES, outerCount, GL_UNSIGNED_SHORT, indexBuffer)
        glEnable(GL_CULL_FACE)
        glFrontFace(if (frontFaceCw) GL_CW else GL_CCW)
        glCullFace(GL_BACK)
        indexBuffer.position(0)
        glDrawElements(GL_TRIANGLES, innerCount, GL_UNSIGNED_SHORT, indexBuffer)
        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        indexBuffer.position(0)
        glDisableVertexAttribArray(position)
        glDisableVertexAttribArray(coordinate)
        glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return texture
    }

    private fun deleteBuffers() {
        if (texture != 0) glDeleteTextures(1, intArrayOf(texture), 0)
        if (frame != 0) glDeleteFramebuffers(1, intArrayOf(frame), 0)
        texture = 0; frame = 0; w = 0; h = 0
    }

    /** 上下文已销毁（新上下文中遗忘旧句柄）。 */
    fun release() {
        texture = 0; frame = 0; w = 0; h = 0; program = 0
    }

    /** 活上下文内同步释放显存与程序。 */
    fun delete() {
        deleteBuffers()
        if (program != 0) glDeleteProgram(program)
        release()
    }
}
