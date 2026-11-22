package io.bebebus.livebeauty

import android.opengl.GLES20.*
import android.opengl.GLES30
import android.os.Trace
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 共享小图采样器（仅 GL 线程）：变形前同一张 2D 纹理缩到长边 [maxSide]（VideoEffectsFilter 用 384），每个新相机帧至多画一次、回读一次，
 * 人脸、美体、人像分割、头发分割（[lease]）与测光（[staging]，按 [reads] 判断本帧是否已读回）共用，采样落在同一节拍，省掉额外的回读。
 * 采样四边形的 v 已上下翻转，读回的 RGBA 行序自上而下，直接交给 MediaPipe（RGBA ByteBuffer），不建 Bitmap。
 * 每个消费者一对缓冲：FaceTrackingState 等同一时刻最多一个任务，worker 用完一块之前 GL 线程只写另一块。
 *
 * v1.1 升级：
 * 1. **OpenGL ES 3.0 双 PBO（`GL_PIXEL_PACK_BUFFER`）异步 DMA 回读**：运行时检测当前上下文为 ES 3.0+ 且处于 [deferred] 跨帧模式时，
 *    当帧 `glReadPixels(..., 0)` 仅触发零阻塞硬件 DMA 传输，下一帧 [collect] 通过 `glMapBufferRange` 映射读出，彻底消除同步回读停顿；
 *    ES 2.0 上下文或异常机型自动无缝回退标准 `GLES20.glReadPixels`。
 * 2. **远景自适应人脸 ROI 高清采样**：当传入非全幅 `faceCrop`（见 [FaceRoiSelector]）时，额外将源纹理对应各向同性正方形子区渲染到
 *    独立 `256×256` ROI 缓冲供 [FACE] 消费者读取，半身/舞蹈远景关键点有效分辨率提升 `2.5–3.5×`，其余消费者仍读全幅小图。
 */
internal class FaceTextureSampler(private val maxSide: Int = 256) {
    /** [pixels] 为 width×height×4 的 RGBA，行 0 是画面顶部；[crop] 为该样本相对全画幅的顶向下归一化裁剪框。 */
    class Sample(
        val pixels: ByteBuffer,
        val width: Int,
        val height: Int,
        val crop: FaceRoiSelector.CropRect = FaceRoiSelector.CropRect.FULL,
    )

    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private val textures = IntArray(2)
    private val frames = IntArray(2)
    private val pboIds = IntArray(2)
    private var pboChecked = false
    private var pboAvailable = false
    /** 当前上下文是 ES 3.0+（与 PBO 是否可用无关：PBO 被停用后仍要保护 GL_PACK_ROW_LENGTH 等 ES3 状态）。 */
    private var es3Context = false
    private val packSaved = IntArray(5)
    private val packTmp = IntArray(1) // 每帧回读都要查询，复用避免分配。
    private var pboPrimedSlot = -1
    private var w = 0
    private var h = 0
    private var next = 0
    private var pendingSlot = -1
    private var stagingBuffer: ByteBuffer? = null

    // 远景人脸专用 256×256 各向同性 ROI 双缓冲（仅在 faceCrop 非全幅时按需分配与绘制）。
    private val roiTextures = IntArray(2)
    private val roiFrames = IntArray(2)
    private val roiPboIds = IntArray(2)
    private var roiAllocated = false
    private var roiStagingBuffer: ByteBuffer? = null
    private var roiCropStaging: FaceRoiSelector.CropRect = FaceRoiSelector.CropRect.FULL
    private var pendingRoiCrop: FaceRoiSelector.CropRect = FaceRoiSelector.CropRect.FULL
    private var hasRoiStaging = false

    private val consumers = Array(CONSUMERS) { arrayOfNulls<ByteBuffer>(2) }
    private val turns = IntArray(CONSUMERS)
    private var hasStaging = false
    private var unreadStall = -1L
    /** 累计读回次数：调用方比较前后值判断本帧是否已有新的小图。 */
    var reads = 0L
        private set
    /** 延迟一帧回读（见类注释）；切换时丢弃尚未读回的小图。 */
    var deferred = false
        set(value) {
            if (field != value) {
                field = value
                pendingSlot = -1
                pboPrimedSlot = -1
                pendingRoiCrop = FaceRoiSelector.CropRect.FULL
            }
        }
    /** 最近一次回读的阻塞时长（纳秒）；还没有回读时为 −1。 */
    var lastStallNs = -1L
        private set

    // 位置 y=-1（读回第 0 行）取纹理 v=1：读回结果即自上而下。
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 0f, 1f, 1f, -1f, 1f, 1f, -1f, 1f, 0f, 0f, 1f, 1f, 1f, 0f)); position(0)
    }
    private val roiQuadScratch = FloatArray(16)
    private val roiVertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer()

    /** 预热（GL 线程、未开播时由调用方调度）：只建程序、取 location，不分配缓冲。 */
    fun prepare() {
        if (ensureProgram()) check(glGetError() == GL_NO_ERROR)
    }

    /**
     * 把 [source] 画进小图；当 [faceCrop] 非全幅时同步渲染 `256×256` 人脸 ROI 小图。
     * SYNC（或 [sync] 为 true，例如测光单独取样）时立即读回暂存区，[deferred] 时留到下一帧 [collect]（ES 3.0 下当帧异步发起 PBO DMA）。
     * 结束时绑回 [restoreFbo]。
     */
    fun capture(
        source: Int,
        width: Int,
        height: Int,
        restoreFbo: Int,
        sync: Boolean = false,
        faceCrop: FaceRoiSelector.CropRect = FaceRoiSelector.CropRect.FULL,
    ) {
        var created = ensureProgram()
        checkPboSupport()
        val scale = maxSide.toFloat() / maxOf(width, height)
        val sw = (width * scale).toInt().coerceAtLeast(1)
        val sh = (height * scale).toInt().coerceAtLeast(1)
        val wantRoi = !sync && !faceCrop.isFullFrame
        try {
            if (w != sw || h != sh) { allocate(sw, sh); created = true }
            if (wantRoi && !roiAllocated) { allocateRoi(); created = true }
            val slot = next; next = next xor 1
            drawQuad(frames[slot], w, h, source, vertices)

            if (wantRoi) {
                fillRoiVertices(faceCrop)
                drawQuad(roiFrames[slot], FaceRoiSelector.ROI_SIDE_PX, FaceRoiSelector.ROI_SIDE_PX, source, roiVertices)
            }

            // 新建路径的错误检查须在发起 PBO 回读之前：enqueuePboRead 会读走错误标志用于判断 PBO 是否可用。
            if (created) check(glGetError() == GL_NO_ERROR)
            if (deferred && !sync) {
                pendingSlot = slot
                pendingRoiCrop = if (wantRoi) faceCrop else FaceRoiSelector.CropRect.FULL
                if (pboAvailable) {
                    if (!enqueuePboRead(frames[slot], pboIds[slot], w, h) ||
                        (wantRoi && !enqueuePboRead(roiFrames[slot], roiPboIds[slot], FaceRoiSelector.ROI_SIDE_PX, FaceRoiSelector.ROI_SIDE_PX))
                    ) {
                        disablePbo()
                    } else {
                        pboPrimedSlot = slot
                    }
                }
            } else {
                hasRoiStaging = false
                roiCropStaging = FaceRoiSelector.CropRect.FULL
                glBindFramebuffer(GL_FRAMEBUFFER, frames[slot])
                readSync(checkNotNull(stagingBuffer), w, h, statMode = STAT_MAIN)
                if (wantRoi) {
                    glBindFramebuffer(GL_FRAMEBUFFER, roiFrames[slot])
                    readSync(checkNotNull(roiStagingBuffer), FaceRoiSelector.ROI_SIDE_PX, FaceRoiSelector.ROI_SIDE_PX, statMode = STAT_ADD_ROI)
                    roiCropStaging = faceCrop
                    hasRoiStaging = true
                }
            }
            if (created) check(glGetError() == GL_NO_ERROR)
        } finally {
            glBindFramebuffer(GL_FRAMEBUFFER, restoreFbo)
        }
    }

    /** [deferred] 时在新帧开头读回上一帧的小图；没有待读的返回 false。结束时绑回 [restoreFbo]。 */
    fun collect(restoreFbo: Int): Boolean {
        val slot = pendingSlot
        if (slot < 0) return false
        val crop = pendingRoiCrop
        val usePbo = pboAvailable && pboPrimedSlot == slot
        pendingSlot = -1
        pboPrimedSlot = -1
        pendingRoiCrop = FaceRoiSelector.CropRect.FULL
        hasRoiStaging = false
        roiCropStaging = FaceRoiSelector.CropRect.FULL
        try {
            if (usePbo) {
                val mainOk = mapPboInto(pboIds[slot], checkNotNull(stagingBuffer), w * h * 4, statMode = STAT_MAIN)
                val roiOk = if (!crop.isFullFrame && roiAllocated) {
                    val side = FaceRoiSelector.ROI_SIDE_PX
                    mapPboInto(roiPboIds[slot], checkNotNull(roiStagingBuffer), side * side * 4, statMode = STAT_ADD_ROI)
                } else true
                if (mainOk && roiOk) {
                    hasStaging = true
                    reads++
                    if (!crop.isFullFrame && roiAllocated) {
                        roiCropStaging = crop
                        hasRoiStaging = true
                    }
                    return true
                }
                disablePbo()
            }
            glBindFramebuffer(GL_FRAMEBUFFER, frames[slot])
            readSync(checkNotNull(stagingBuffer), w, h, statMode = STAT_MAIN)
            if (!crop.isFullFrame && roiAllocated) {
                val side = FaceRoiSelector.ROI_SIDE_PX
                glBindFramebuffer(GL_FRAMEBUFFER, roiFrames[slot])
                readSync(checkNotNull(roiStagingBuffer), side, side, statMode = STAT_ADD_ROI)
                roiCropStaging = crop
                hasRoiStaging = true
            }
        } finally {
            glBindFramebuffer(GL_FRAMEBUFFER, restoreFbo)
        }
        return true
    }

    /** [deferred] 时丢弃尚未读回的小图（下一帧开头没有一路取到票）：不做这次回读。 */
    fun discardPending() {
        pendingSlot = -1
        pboPrimedSlot = -1
        pendingRoiCrop = FaceRoiSelector.CropRect.FULL
    }

    private fun drawQuad(fbo: Int, vpW: Int, vpH: Int, sourceTex: Int, quad: java.nio.FloatBuffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, vpW, vpH)
        glUseProgram(program)
        quad.position(0)
        glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, quad)
        glEnableVertexAttribArray(position)
        quad.position(2)
        glVertexAttribPointer(coordinate, 2, GL_FLOAT, false, 16, quad)
        glEnableVertexAttribArray(coordinate)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, sourceTex)
        glUniform1i(sampler, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(position)
        glDisableVertexAttribArray(coordinate)
        glBindTexture(GL_TEXTURE_2D, 0)
    }

    private fun fillRoiVertices(crop: FaceRoiSelector.CropRect) {
        val u0 = crop.left
        val u1 = crop.left + crop.width
        val vTop = 1f - crop.top
        val vBot = 1f - (crop.top + crop.height)
        roiQuadScratch[0] = -1f; roiQuadScratch[1] = -1f; roiQuadScratch[2] = u0; roiQuadScratch[3] = vTop
        roiQuadScratch[4] = 1f; roiQuadScratch[5] = -1f; roiQuadScratch[6] = u1; roiQuadScratch[7] = vTop
        roiQuadScratch[8] = -1f; roiQuadScratch[9] = 1f; roiQuadScratch[10] = u0; roiQuadScratch[11] = vBot
        roiQuadScratch[12] = 1f; roiQuadScratch[13] = 1f; roiQuadScratch[14] = u1; roiQuadScratch[15] = vBot
        roiVertices.clear()
        roiVertices.put(roiQuadScratch)
        roiVertices.position(0)
    }

    /**
     * 回读前把 PACK 状态设成紧密排布、回读后原样还原：宿主（RTC / 相机 SDK）共享同一上下文，可能改过
     * GL_PACK_ALIGNMENT 或 ES3 的 GL_PACK_ROW_LENGTH / SKIP_*，非零 ROW_LENGTH 会让 glReadPixels 写穿缓冲区；
     * 但只设不还原又会反过来破坏宿主自己的回读，所以必须成对调用 [beginPack] / [endPack]。
     */
    private fun beginPack() {
        val tmp = packTmp
        glGetIntegerv(GL_PACK_ALIGNMENT, tmp, 0); packSaved[0] = tmp[0]
        if (packSaved[0] != 4) glPixelStorei(GL_PACK_ALIGNMENT, 4)
        if (es3Context) {
            try {
                glGetIntegerv(GLES30.GL_PACK_ROW_LENGTH, tmp, 0); packSaved[1] = tmp[0]
                glGetIntegerv(GLES30.GL_PACK_SKIP_PIXELS, tmp, 0); packSaved[2] = tmp[0]
                glGetIntegerv(GLES30.GL_PACK_SKIP_ROWS, tmp, 0); packSaved[3] = tmp[0]
                if (packSaved[1] != 0) glPixelStorei(GLES30.GL_PACK_ROW_LENGTH, 0)
                if (packSaved[2] != 0) glPixelStorei(GLES30.GL_PACK_SKIP_PIXELS, 0)
                if (packSaved[3] != 0) glPixelStorei(GLES30.GL_PACK_SKIP_ROWS, 0)
                // 宿主可能绑着自己的 PACK PBO：同步 glReadPixels(Buffer) 会被当成偏移而报错，我们解绑后也必须还原。
                glGetIntegerv(GLES30.GL_PIXEL_PACK_BUFFER_BINDING, tmp, 0); packSaved[4] = tmp[0]
                if (packSaved[4] != 0) GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            } catch (_: Throwable) {
                packSaved[1] = 0; packSaved[2] = 0; packSaved[3] = 0; packSaved[4] = 0
            }
        }
    }

    private fun endPack() {
        if (packSaved[0] != 4) glPixelStorei(GL_PACK_ALIGNMENT, packSaved[0])
        if (es3Context) {
            try {
                if (packSaved[1] != 0) glPixelStorei(GLES30.GL_PACK_ROW_LENGTH, packSaved[1])
                if (packSaved[2] != 0) glPixelStorei(GLES30.GL_PACK_SKIP_PIXELS, packSaved[2])
                if (packSaved[3] != 0) glPixelStorei(GLES30.GL_PACK_SKIP_ROWS, packSaved[3])
                if (packSaved[4] != 0) GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, packSaved[4])
            } catch (_: Throwable) {}
        }
    }

    /** 同步读回当前绑定的 FBO 到目标缓冲区。 */
    private fun readSync(buffer: ByteBuffer, rw: Int, rh: Int, statMode: Int) {
        buffer.position(0)
        Trace.beginSection("beauty.readback")
        beginPack()
        try {
            val start = System.nanoTime()
            glReadPixels(0, 0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
            val elapsed = System.nanoTime() - start
            if (statMode == STAT_MAIN) {
                lastStallNs = elapsed
                unreadStall = elapsed
                hasStaging = true
                reads++
            } else if (statMode == STAT_ADD_ROI) {
                lastStallNs = (if (lastStallNs >= 0L) lastStallNs else 0L) + elapsed
                unreadStall = (if (unreadStall >= 0L) unreadStall else 0L) + elapsed
            }
        } finally {
            endPack()
            Trace.endSection()
        }
        buffer.position(0)
    }

    private fun enqueuePboRead(fbo: Int, pbo: Int, rw: Int, rh: Int): Boolean {
        if (pbo == 0) return false
        beginPack()
        return try {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbo)
            GLES30.glReadPixels(0, 0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, 0)
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            glGetError() == GL_NO_ERROR
        } catch (_: Throwable) {
            try { GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0) } catch (_: Throwable) {}
            false
        } finally {
            endPack()
        }
    }

    private fun mapPboInto(pbo: Int, target: ByteBuffer, byteCount: Int, statMode: Int): Boolean {
        if (pbo == 0) return false
        Trace.beginSection("beauty.pboMap")
        beginPack()
        var mappedOk = false
        return try {
            val start = System.nanoTime()
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbo)
            val mapped = GLES30.glMapBufferRange(GLES30.GL_PIXEL_PACK_BUFFER, 0, byteCount, GLES30.GL_MAP_READ_BIT) as? ByteBuffer
            if (mapped == null) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
                return false
            }
            mappedOk = true
            mapped.position(0)
            mapped.limit(byteCount)
            target.clear()
            target.put(mapped)
            target.position(0)
            GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)
            mappedOk = false
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            val elapsed = System.nanoTime() - start
            if (statMode == STAT_MAIN) {
                lastStallNs = elapsed
                unreadStall = elapsed
            } else if (statMode == STAT_ADD_ROI) {
                lastStallNs = (if (lastStallNs >= 0L) lastStallNs else 0L) + elapsed
                unreadStall = (if (unreadStall >= 0L) unreadStall else 0L) + elapsed
            }
            true
        } catch (_: Throwable) {
            if (mappedOk) {
                try { GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER) } catch (_: Throwable) {}
            }
            try { GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0) } catch (_: Throwable) {}
            false
        } finally {
            endPack()
            Trace.endSection()
        }
    }

    private fun checkPboSupport() {
        if (pboChecked) return
        pboChecked = true
        es3Context = try {
            PboSupport.isEs3Version(glGetString(GL_VERSION))
        } catch (_: Throwable) {
            false
        }
        pboAvailable = es3Context
    }

    private fun disablePbo() {
        pboAvailable = false
        pboPrimedSlot = -1
        deletePbos()
    }

    /** 把暂存区拷进 [consumer]（[FACE]、[BODY]、[SEGMENT]、[HAIR]）自己的双缓冲，交给它的 worker。 */
    fun lease(consumer: Int): Sample {
        val useRoi = consumer == FACE && hasRoiStaging && roiStagingBuffer != null
        val source = if (useRoi) checkNotNull(roiStagingBuffer) else checkNotNull(stagingBuffer)
        val outW = if (useRoi) FaceRoiSelector.ROI_SIDE_PX else w
        val outH = if (useRoi) FaceRoiSelector.ROI_SIDE_PX else h
        val crop = if (useRoi) roiCropStaging else FaceRoiSelector.CropRect.FULL
        val pair = consumers[consumer]
        val turn = turns[consumer]; turns[consumer] = turn xor 1
        val target = pair[turn]?.takeIf { it.capacity() == source.capacity() }
            ?: ByteBuffer.allocateDirect(source.capacity()).also { pair[turn] = it }
        target.clear(); source.position(0)
        target.put(source)
        source.position(0); target.position(0)
        return Sample(target, outW, outH, crop)
    }

    /** 暂存区（最近一次读回的全幅小图，GL 线程同步使用，不得交给 worker）；还没有读回时为 null。 */
    val staging: Sample? get() = stagingBuffer?.takeIf { hasStaging }?.let { Sample(it, w, h) }

    /**
     * 仅 JVM 单测用：不经 GL，把 [pixels]（width×height×4）当作刚读回的小图放进暂存区，验证 [lease] 各路双缓冲互不干扰。
     */
    internal fun stageForTest(
        pixels: ByteBuffer,
        width: Int,
        height: Int,
        roiPixels: ByteBuffer? = null,
        roiCrop: FaceRoiSelector.CropRect = FaceRoiSelector.CropRect.FULL,
    ) {
        require(pixels.capacity() == width * height * 4)
        w = width; h = height
        stagingBuffer = pixels.also { it.position(0) }
        hasStaging = true
        if (roiPixels != null && !roiCrop.isFullFrame) {
            val side = FaceRoiSelector.ROI_SIDE_PX
            require(roiPixels.capacity() == side * side * 4)
            roiStagingBuffer = roiPixels.also { it.position(0) }
            roiCropStaging = roiCrop
            hasRoiStaging = true
        } else {
            hasRoiStaging = false
            roiCropStaging = FaceRoiSelector.CropRect.FULL
        }
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
        releaseMainBuffers(); w = sw; h = sh
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
        if (pboAvailable) allocatePbos(pboIds, w * h * 4)
    }

    private fun allocateRoi() {
        releaseRoiBuffers()
        val side = FaceRoiSelector.ROI_SIDE_PX
        glGenTextures(2, roiTextures, 0)
        glGenFramebuffers(2, roiFrames, 0)
        for (i in 0..1) {
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, roiTextures[i])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, side, side, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
            glBindFramebuffer(GL_FRAMEBUFFER, roiFrames[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, roiTextures[i], 0)
            check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
        }
        glBindTexture(GL_TEXTURE_2D, 0)
        roiStagingBuffer = ByteBuffer.allocateDirect(side * side * 4)
        roiAllocated = true
        if (pboAvailable) allocatePbos(roiPboIds, side * side * 4)
    }

    private fun allocatePbos(ids: IntArray, byteCount: Int) {
        // allocate() 只在新建路径调用：先把纹理 / FBO 分配阶段的错误按原语义抛出，不能被这里吞掉，再单独判断 PBO 是否可用。
        check(glGetError() == GL_NO_ERROR)
        try {
            GLES30.glGenBuffers(2, ids, 0)
            for (i in 0..1) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, ids[i])
                GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, byteCount, null, GLES30.GL_DYNAMIC_READ)
            }
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            if (glGetError() != GL_NO_ERROR) disablePbo()
        } catch (_: Throwable) {
            disablePbo()
        }
    }

    private fun deletePbos() {
        try {
            if (pboIds[0] != 0) GLES30.glDeleteBuffers(2, pboIds, 0)
            if (roiPboIds[0] != 0) GLES30.glDeleteBuffers(2, roiPboIds, 0)
        } catch (_: Throwable) {}
        pboIds.fill(0)
        roiPboIds.fill(0)
    }

    private fun releaseMainBuffers() {
        if (textures[0] != 0) glDeleteTextures(2, textures, 0)
        if (frames[0] != 0) glDeleteFramebuffers(2, frames, 0)
        try {
            if (pboIds[0] != 0) GLES30.glDeleteBuffers(2, pboIds, 0)
        } catch (_: Throwable) {}
        pboIds.fill(0)
        textures.fill(0); frames.fill(0); w = 0; h = 0; next = 0; pendingSlot = -1; pboPrimedSlot = -1
        stagingBuffer = null; hasStaging = false; unreadStall = -1L
    }

    private fun releaseRoiBuffers() {
        if (roiTextures[0] != 0) glDeleteTextures(2, roiTextures, 0)
        if (roiFrames[0] != 0) glDeleteFramebuffers(2, roiFrames, 0)
        try {
            if (roiPboIds[0] != 0) GLES30.glDeleteBuffers(2, roiPboIds, 0)
        } catch (_: Throwable) {}
        roiPboIds.fill(0)
        roiTextures.fill(0); roiFrames.fill(0); roiAllocated = false
        roiStagingBuffer = null; hasRoiStaging = false; roiCropStaging = FaceRoiSelector.CropRect.FULL
    }

    private fun forget() {
        textures.fill(0); frames.fill(0); pboIds.fill(0)
        roiTextures.fill(0); roiFrames.fill(0); roiPboIds.fill(0); roiAllocated = false
        w = 0; h = 0; next = 0; pendingSlot = -1; pboPrimedSlot = -1
        stagingBuffer = null; roiStagingBuffer = null
        hasStaging = false; hasRoiStaging = false
        roiCropStaging = FaceRoiSelector.CropRect.FULL
        pendingRoiCrop = FaceRoiSelector.CropRect.FULL
        unreadStall = -1L
    }

    /** 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。 */
    fun release() {
        forget(); program = 0; deferred = false; lastStallNs = -1L
        pboChecked = false; pboAvailable = false; es3Context = false
        for (pair in consumers) pair.fill(null)
        turns.fill(0)
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        releaseMainBuffers()
        releaseRoiBuffers()
        if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        const val FACE = 0
        const val BODY = 1
        const val SEGMENT = 2
        /** 头发分割（HairSegmenter，约 10 Hz）：独立一对缓冲，与人像分割各用各的 worker。 */
        const val HAIR = 3
        private const val CONSUMERS = 4
        private const val STAT_MAIN = 1
        private const val STAT_ADD_ROI = 2
    }
}
