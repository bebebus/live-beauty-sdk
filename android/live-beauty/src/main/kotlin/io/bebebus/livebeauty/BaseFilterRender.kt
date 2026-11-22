package io.bebebus.livebeauty

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLContext
import android.opengl.GLES20.*

/**
 * 独立于具体推流器 / RTC 引擎的 OpenGL ES 2D 纹理滤镜基类：
 * 维护一张输出 RGBA8 FBO 纹理（[renderHandler]），按「绑定输出 FBO → viewport → [drawFilter] → 全屏绘制 → [disableResources]」
 * 标准生命周期驱动子类管线。
 */
abstract class BaseFilterRender {
    class RenderHandler {
        val fboId = IntArray(1)
        val texId = IntArray(1)
    }

    protected val renderHandler = RenderHandler()
    protected var width: Int = 0
    protected var height: Int = 0
    protected var previousTexId: Int = 0

    private var targetWidth: Int = 0
    private var targetHeight: Int = 0
    private var targetContext: EGLContext = EGL14.EGL_NO_CONTEXT

    /** 当前输出 RGBA8 `GL_TEXTURE_2D` 句柄；尚未分配时为 0。 */
    val outputTextureId: Int
        get() = renderHandler.texId[0]

    /** 设置输入 2D 纹理与目标帧宽高，并在必要时（首帧、分辨率变化或 EGLContext 重建）分配输出 FBO。 */
    fun setRenderInput(inputTexId: Int, width: Int, height: Int) {
        previousTexId = inputTexId
        this.width = width
        this.height = height
    }

    internal fun prepareRenderTarget(inputTexId: Int, width: Int, height: Int) {
        setRenderInput(inputTexId, width, height)
        ensureRenderTarget()
    }

    private fun ensureRenderTarget() {
        if (width <= 0 || height <= 0) return
        val current = EGL14.eglGetCurrentContext()
        if (current != targetContext) {
            // 旧上下文已销毁：只丢弃旧句柄，不在新上下文误删他处纹理。
            renderHandler.fboId[0] = 0
            renderHandler.texId[0] = 0
            targetWidth = 0
            targetHeight = 0
            targetContext = current
        }
        if (renderHandler.fboId[0] != 0 && renderHandler.texId[0] != 0 &&
            targetWidth == width && targetHeight == height
        ) {
            return
        }
        deleteRenderTargetInCurrentContext()
        glGenTextures(1, renderHandler.texId, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, renderHandler.texId[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
        glGenFramebuffers(1, renderHandler.fboId, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, renderHandler.fboId[0])
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderHandler.texId[0], 0)
        check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) { "Output FBO incomplete" }
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        targetWidth = width
        targetHeight = height
    }

    /** 在创建输出 FBO 的存活 EGLContext 中释放显存。 */
    fun releaseRenderTarget() {
        val current = EGL14.eglGetCurrentContext()
        if (current != EGL14.EGL_NO_CONTEXT && current == targetContext) {
            deleteRenderTargetInCurrentContext()
        } else {
            renderHandler.fboId[0] = 0
            renderHandler.texId[0] = 0
        }
        targetWidth = 0
        targetHeight = 0
        targetContext = EGL14.EGL_NO_CONTEXT
    }

    private fun deleteRenderTargetInCurrentContext() {
        if (renderHandler.texId[0] != 0) {
            glDeleteTextures(1, renderHandler.texId, 0)
            renderHandler.texId[0] = 0
        }
        if (renderHandler.fboId[0] != 0) {
            glDeleteFramebuffers(1, renderHandler.fboId, 0)
            renderHandler.fboId[0] = 0
        }
    }

    open fun initGlFilter(context: Context) {}

    open fun draw() {
        ensureRenderTarget()
        glBindFramebuffer(GL_FRAMEBUFFER, renderHandler.fboId[0])
        glViewport(0, 0, width, height)
        drawFilter()
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        disableResources()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    }

    protected abstract fun drawFilter()

    protected abstract fun disableResources()

    open fun release() {
        releaseRenderTarget()
    }
}
