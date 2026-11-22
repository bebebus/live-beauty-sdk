package io.bebebus.livebeauty

import android.opengl.GLES20.*
import android.os.Trace
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 共享小图采样器（仅 GL 线程）：变形前同一张 2D 纹理缩到长边 [maxSide]（VideoEffectsFilter 用 384），每个新相机帧至多画一次、回读一次，
 * 人脸、美体、人像分割、头发分割（[lease]）与测光（[staging]，按 [reads] 判断本帧是否已读回）共用，采样落在同一节拍，省掉额外的回读。
 * 采样四边形的 v 已上下翻转，读回的 RGBA 行序自上而下，直接交给 MediaPipe（RGBA ByteBuffer），不建 Bitmap。
 * 每个消费者一对缓冲：FaceTrackingState 等同一时刻最多一个任务，worker 用完一块之前 GL 线程只写另一块。
 *
 * 两种回读：SYNC（默认）在 [capture] 当帧读回；[deferred] 时本帧只画到小图，下一个新帧开头 [collect] 再读
 * （那时一路票都取不到就 [discardPending]，不读回），glReadPixels 等的是一帧前就已执行完的小图 pass，GPU 满载时
 * 不再把 CPU 与 GPU 串行化，代价是多一帧延迟（由 PacingLock 按回读阻塞时长 [lastStallNs] 在锁定时决定）。
 * 不查询 GL 状态：调用方传入要恢复的 FBO，viewport 由调用方随后自行设置；只在建程序或分配缓冲那一帧检查 glGetError。
 */
internal class FaceTextureSampler(private val maxSide: Int = 256) {
    /** [pixels] 为 width×height×4 的 RGBA，行 0 是画面顶部；position 为 0。 */
    class Sample(val pixels: ByteBuffer, val width: Int, val height: Int)
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private val textures = IntArray(2)
    private val frames = IntArray(2)
    private var w = 0
    private var h = 0
    private var next = 0
    private var pendingSlot = -1
    private var stagingBuffer: ByteBuffer? = null
    private val consumers = Array(CONSUMERS) { arrayOfNulls<ByteBuffer>(2) }
    private val turns = IntArray(CONSUMERS)
    private var hasStaging = false
    private var unreadStall = -1L
    /** 累计读回次数：调用方比较前后值判断本帧是否已有新的小图。 */
    var reads = 0L
        private set
    /** 延迟一帧回读（见类注释）；切换时丢弃尚未读回的小图。 */
    var deferred = false
        set(value) { if (field != value) { field = value; pendingSlot = -1 } }
    /** 最近一次 glReadPixels 的阻塞时长（纳秒）；还没有回读时为 −1。 */
    var lastStallNs = -1L
        private set
    // 位置 y=-1（读回第 0 行）取纹理 v=1：读回结果即自上而下。
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,1f, 1f,-1f,1f,1f, -1f,1f,0f,0f, 1f,1f,1f,0f)); position(0)
    }

    /** 预热（GL 线程、未开播时由调用方调度）：只建程序、取 location，不分配缓冲。 */
    fun prepare() {
        if (ensureProgram()) check(glGetError() == GL_NO_ERROR)
    }

    /**
     * 把 [source] 画进小图；SYNC（或 [sync] 为 true，例如测光单独取样）时立即读回暂存区，[deferred] 时留到下一帧 [collect]。
     * 结束时绑回 [restoreFbo]。
     */
    fun capture(source: Int, width: Int, height: Int, restoreFbo: Int, sync: Boolean = false) {
        var created = ensureProgram()
        val scale = maxSide.toFloat() / maxOf(width, height)
        val sw = (width * scale).toInt().coerceAtLeast(1); val sh = (height * scale).toInt().coerceAtLeast(1)
        try {
            if (w != sw || h != sh) { allocate(sw, sh); created = true }
            val slot = next; next = next xor 1
            glBindFramebuffer(GL_FRAMEBUFFER, frames[slot]); glViewport(0, 0, w, h); glUseProgram(program)
            vertices.position(0); glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertices); glEnableVertexAttribArray(position)
            vertices.position(2); glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, vertices); glEnableVertexAttribArray(coordinate)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, source)
            glUniform1i(sampler, 0)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glDisableVertexAttribArray(position); glDisableVertexAttribArray(coordinate)
            glBindTexture(GL_TEXTURE_2D, 0)
            if (deferred && !sync) pendingSlot = slot else read()
            // 发布版 AAR 的 checkGlError 是空操作；只在建程序或（重新）分配那一帧查一次。
            if (created) check(glGetError() == GL_NO_ERROR)
        } finally {
            glBindFramebuffer(GL_FRAMEBUFFER, restoreFbo)
        }
    }

    /** [deferred] 时在新帧开头读回上一帧的小图；没有待读的返回 false。结束时绑回 [restoreFbo]。 */
    fun collect(restoreFbo: Int): Boolean {
        val slot = pendingSlot
        if (slot < 0) return false
        pendingSlot = -1
        try {
            glBindFramebuffer(GL_FRAMEBUFFER, frames[slot])
            read()
        } finally {
            glBindFramebuffer(GL_FRAMEBUFFER, restoreFbo)
        }
        return true
    }

    /** [deferred] 时丢弃尚未读回的小图（下一帧开头没有一路取到票）：不做这次 glReadPixels。 */
    fun discardPending() { pendingSlot = -1 }

    /** 读回当前绑定的小图 FBO 到暂存区。 */
    private fun read() {
        val buffer = checkNotNull(stagingBuffer)
        buffer.position(0)
        Trace.beginSection("beauty.readback")
        try {
            val start = System.nanoTime()
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
            lastStallNs = System.nanoTime() - start
        } finally {
            Trace.endSection()
        }
        buffer.position(0)
        unreadStall = lastStallNs
        hasStaging = true; reads++
    }

    /** 把暂存区拷进 [consumer]（[FACE]、[BODY]、[SEGMENT]、[HAIR]）自己的双缓冲，交给它的 worker。 */
    fun lease(consumer: Int): Sample {
        val source = checkNotNull(stagingBuffer)
        val pair = consumers[consumer]
        val turn = turns[consumer]; turns[consumer] = turn xor 1
        // MediaPipe 要求容量恰为 w*h*4。
        val target = pair[turn]?.takeIf { it.capacity() == source.capacity() }
            ?: ByteBuffer.allocateDirect(source.capacity()).also { pair[turn] = it }
        target.clear(); source.position(0)
        target.put(source)
        source.position(0); target.position(0)
        return Sample(target, w, h)
    }

    /** 暂存区（最近一次读回的小图，GL 线程同步使用，不得交给 worker）；还没有读回时为 null。 */
    val staging: Sample? get() = stagingBuffer?.takeIf { hasStaging }?.let { Sample(it, w, h) }

    /**
     * 仅 JVM 单测用：不经 GL，把 [pixels]（width×height×4）当作刚读回的小图放进暂存区，验证 [lease] 各路双缓冲互不干扰。
     * 产品代码只经 [capture] / [collect] 读回。
     */
    internal fun stageForTest(pixels: ByteBuffer, width: Int, height: Int) {
        require(pixels.capacity() == width * height * 4)
        w = width; h = height
        stagingBuffer = pixels.also { it.position(0) }
        hasStaging = true
    }

    /** 自上次调用以来的回读阻塞时长；没有新的回读时为 −1（供 PacingLock.onStall）。 */
    fun takeStallNs(): Long = unreadStall.also { unreadStall = -1L }

    /** @return 是否刚建了程序。 */
    private fun ensureProgram(): Boolean {
        if (program != 0) return false
        program = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.ORIGINAL).also { check(it > 0) }
        position = glGetAttribLocation(program, "aPosition")
        coordinate = glGetAttribLocation(program, "aTextureCoord")
        sampler = glGetUniformLocation(program, "uSampler")
        return true
    }

    private fun allocate(sw: Int, sh: Int) {
        releaseBuffers(); w = sw; h = sh
        glGenTextures(2, textures, 0)
        glGenFramebuffers(2, frames, 0)
        for (i in 0..1) {
            glActiveTexture(GL_TEXTURE0)
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
        stagingBuffer = ByteBuffer.allocateDirect(w * h * 4)
    }

    private fun releaseBuffers() {
        if (textures[0] != 0) glDeleteTextures(2, textures, 0)
        if (frames[0] != 0) glDeleteFramebuffers(2, frames, 0)
        forget()
    }

    private fun forget() {
        textures.fill(0); frames.fill(0); w = 0; h = 0; next = 0; pendingSlot = -1
        stagingBuffer = null; hasStaging = false; unreadStall = -1L
    }

    /** 上下文已销毁（GL线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        forget(); program = 0; deferred = false; lastStallNs = -1L
        for (pair in consumers) pair.fill(null)
        turns.fill(0)
    }

    /** GL线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        releaseBuffers(); if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        const val FACE = 0
        const val BODY = 1
        const val SEGMENT = 2
        /** 头发分割（HairSegmenter，约 10 Hz）：独立一对缓冲，与人像分割各用各的 worker。 */
        const val HAIR = 3
        private const val CONSUMERS = 4
    }
}
