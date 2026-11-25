package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 画面延后（仅 Android）：每个相机帧复制进环形 RGBA8 纹理，美颜画的是「人脸结果已经出来」的那一帧，
 * 美颜几何与五官逐帧对齐（见 FaceTimeline）。延后多少帧由 [PacingLock] 每场锁定（见 [DelaySchedule]）：
 * 编码 PTS 按被显示帧的相机时间戳（FramePacing），加一帧在 PTS 层对观众不可见（只让播放端多缓冲一帧），
 * 缩一帧会跳过一帧内容，所以直播中只增不减、缩帧只在预览发生；丢脸、代际变化、对比原画、负载降级都不清空缓冲。
 * 环形缓冲同时记相机时间戳（[stampNs]），供 FramePacing 发布「实际显示的相机帧」。
 * 只在 GL 线程使用。返回后：GL_TEXTURE0 激活、unit 0 未绑定、属性数组已关闭；FBO 与 viewport 由调用方恢复。
 * 任何 GL 故障都抛异常，由调用方停用画面对齐（回到不延后）。
 */
internal class FrameDelay {
    private var program = 0
    private var position = -1
    private var coordinate = -1
    private var sampler = -1
    private val textures = IntArray(SLOTS)
    private val frames = IntArray(SLOTS)
    private var w = 0
    private var h = 0
    val schedule = DelaySchedule()
    private val vertices = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f)); position(0)
    }

    /**
     * 新相机帧：复制 [source] 进环形缓冲（采样时间 [time]、相机时间戳 [stampNs]），再按最近完成的采样时间 [completed]
     * 与是否在推流 [streaming] 选出要画的帧。
     * @return 要画的帧的纹理；其采样时间与相机时间戳见 [DelaySchedule.time]、[stampNs]。
     */
    fun push(source: Int, width: Int, height: Int, time: Long, stampNs: Long, completed: Long, streaming: Boolean): Int {
        store(source, width, height, time, stampNs)
        return textures[schedule.select(completed, streaming, time)]
    }

    private fun store(source: Int, width: Int, height: Int, time: Long, stampNs: Long) {
        check(width > 0 && height > 0)
        var fresh = ensureProgram()
        if (w != width || h != height) {
            deleteBuffers()
            glGenTextures(SLOTS, textures, 0)
            glGenFramebuffers(SLOTS, frames, 0)
            for (i in 0 until SLOTS) {
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textures[i])
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
                glBindFramebuffer(GL_FRAMEBUFFER, frames[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0)
                check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
            }
            glBindTexture(GL_TEXTURE_2D, 0)
            w = width; h = height
            // 尺寸变化（切镜头换分辨率）：旧画面不再要，缓冲从头攒；锁定的延后保留。
            schedule.resetBuffer()
            fresh = true
        }
        val slot = schedule.next()
        glBindFramebuffer(GL_FRAMEBUFFER, frames[slot])
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
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(position); glDisableVertexAttribArray(coordinate)
        glBindTexture(GL_TEXTURE_2D, 0)
        // 发布版 AAR 的 checkGlError 是空操作；只在（重新）分配那一帧查一次。
        if (fresh) check(glGetError() == GL_NO_ERROR)
        schedule.stored(slot, time, stampNs)
    }

    /** @return 是否刚建了程序。 */
    private fun ensureProgram(): Boolean {
        if (program != 0) return false
        program = GlUtil.createProgram(VideoEffectsShaders.VERTEX, VideoEffectsShaders.ORIGINAL).also { check(it > 0) }
        position = glGetAttribLocation(program, "aPosition")
        coordinate = glGetAttribLocation(program, "aTextureCoord")
        sampler = glGetUniformLocation(program, "uSampler")
        return true
    }

    /** 预热（GL 线程、未开播时由调用方调度）：只建程序、取 location，不分配缓冲。 */
    fun prepare() {
        if (ensureProgram()) check(glGetError() == GL_NO_ERROR)
    }

    /** 当前选中的帧（同一相机帧的重复绘制沿用）；没有时为 0。 */
    val selected: Int get() = if (schedule.count == 0) 0 else textures[schedule.selectedSlot]

    /** 选中帧的相机时间戳（SurfaceTexture，纳秒）；没有时为 0。 */
    val stampNs: Long get() = if (schedule.count == 0) 0L else schedule.stampNs

    /** 开始画面对齐（会话内首次需要跟踪）：先用先验延后 [prior]，测量后锁定。 */
    fun engage(prior: Int) = schedule.lock.engage(prior)

    /** 是否还占着环形缓冲的显存。 */
    val allocated: Boolean get() = w != 0

    /** 会话结束（长时间不需要对齐）：删除环形缓冲（5 张全尺寸 RGBA）并解除锁定，程序保留。GL 线程、本上下文。 */
    fun free() {
        deleteBuffers()
        schedule.lock.disengage()
    }

    private fun deleteBuffers() {
        if (textures[0] != 0) glDeleteTextures(SLOTS, textures, 0)
        if (frames[0] != 0) glDeleteFramebuffers(SLOTS, frames, 0)
        textures.fill(0); frames.fill(0); w = 0; h = 0; schedule.resetBuffer()
    }

    /**
     * 上下文已销毁（GL 线程在新上下文里调用）：只忘记 id，GL 对象随旧上下文销毁。
     * [keepLock] 为 true（推流中重建上下文）时保留锁定的延后，避免直播中重新测量缩帧。
     */
    fun release(keepLock: Boolean = false) {
        textures.fill(0); frames.fill(0); w = 0; h = 0; program = 0; schedule.resetBuffer()
        if (!keepLock) schedule.lock.disengage()
    }

    /** GL 线程、创建这些对象的上下文仍是当前上下文：删除后忘记。 */
    fun delete() {
        deleteBuffers(); if (program != 0) glDeleteProgram(program)
        release()
    }

    companion object {
        const val SLOTS = DelaySchedule.SLOTS
    }
}

/**
 * [FrameDelay] 的选帧逻辑，纯逻辑、可单测。环形缓冲最多 [SLOTS] 帧，按存入顺序记采样时间与相机时间戳。
 * [want]：最少要往回退几帧，才能退到采样结果已出来（采样时间 ≤ completed）的帧；实际延后由 [lock] 每场锁定，
 * 缓冲未攒满时取能取到的最老一帧。
 */
internal class DelaySchedule {
    private val times = LongArray(SLOTS)
    private val stamps = LongArray(SLOTS)
    private var head = 0 // 下一个写入位置
    var count = 0
        private set
    /** 本帧实际延后帧数。 */
    var delay = 0
        private set
    var selectedSlot = 0
        private set
    val lock = PacingLock()

    /** 清空缓冲（尺寸变化、上下文重建），锁定的延后保留。 */
    fun resetBuffer() {
        head = 0; count = 0; delay = 0; selectedSlot = 0
    }

    /** 下一帧写入的槽位。 */
    fun next(): Int = head

    fun stored(slot: Int, time: Long, stampNs: Long) {
        times[slot] = time
        stamps[slot] = stampNs
        head = (slot + 1) % SLOTS; count = minOf(count + 1, SLOTS)
    }

    /**
     * 本帧想要的延后：[completed] 为最近完成的采样时间。还没有任何结果（MIN_VALUE），或缓冲未攒满且退不到时返回 −1（不计入测量）；
     * 缓冲已满仍退不到（结果比整个缓冲还慢）返回 [SLOTS]。
     */
    fun want(completed: Long): Int {
        if (completed == Long.MIN_VALUE) return -1
        for (d in 0 until count) if (times[slot(d)] <= completed) return d
        return if (count < SLOTS) -1 else SLOTS
    }

    /** 选帧；返回槽位。[nowMs] 为本帧采样时间（与 [completed] 同一时基）。 */
    fun select(completed: Long, streaming: Boolean, nowMs: Long): Int {
        lock.update(want(completed), streaming, nowMs)
        delay = minOf(lock.delay, count - 1).coerceAtLeast(0)
        selectedSlot = slot(delay)
        return selectedSlot
    }

    /** 选中帧的采样时间。 */
    val time: Long get() = times[selectedSlot]

    /** 选中帧的相机时间戳。 */
    val stampNs: Long get() = stamps[selectedSlot]

    private fun slot(back: Int) = (head - 1 - back + 2 * SLOTS) % SLOTS

    companion object {
        /** 最多延后 4 帧（30 fps 约 133 ms）；再慢的推理靠 FaceTimeline 外推兜底。 */
        const val MAX_DELAY = 4
        const val SLOTS = MAX_DELAY + 1
    }
}

/**
 * 画面延后的每场锁定（纯逻辑）。会话开始（首次需要跟踪）时 [engage] 先验值：GPU 推理 [PRIOR_GPU] 帧、CPU [PRIOR_CPU] 帧；
 * 跳过 [WARMUP] 个有效样本后测 [MEASURE] 个 want，取 p90 钳到 [1, MAX_DELAY] 锁定：
 * - 未推流（预览）时直接改到测得值，只影响预览一次；推流中只取 max(先验, 测得)。
 * - 锁定时回读阻塞 p90 超过 [STALL_DEFERRED_NS]：改为延迟一帧回读（[deferredReadback]，会话内不再切回），延后同时 +1。
 * - 锁定后只在推流中、近 [GROW_WINDOW] 个 want 的 p90 ≥ 延后 + [GROW_MARGIN]、且距上次加帧 ≥ [GROW_COOLDOWN_MS] 时加 1 帧；
 *   永不缩小。停播回到预览时重新测量（可以缩）。
 * want 为 −1（还没有结果、缓冲未攒满）的帧不计入测量，也不改延后，所以丢脸不会让画面跳帧。
 */
internal class PacingLock {
    var delay = 0
        private set
    var engaged = false
        private set
    var locked = false
        private set
    var deferredReadback = false
        private set
    private var samples = 0
    private val wants = IntArray(GROW_WINDOW)
    private var wantCount = 0
    private val histogram = IntArray(DelaySchedule.SLOTS + 1)
    private val stalls = LongArray(STALL_SAMPLES)
    private var stallCount = 0
    private var lastGrowAt = Long.MIN_VALUE
    private var wasStreaming = false

    fun engage(prior: Int) {
        delay = prior.coerceIn(1, DelaySchedule.MAX_DELAY)
        engaged = true; locked = false; deferredReadback = false
        samples = 0; wantCount = 0; stallCount = 0; lastGrowAt = Long.MIN_VALUE; wasStreaming = false
    }

    fun disengage() {
        delay = 0; engaged = false; locked = false; deferredReadback = false
        samples = 0; wantCount = 0; stallCount = 0; lastGrowAt = Long.MIN_VALUE; wasStreaming = false
    }

    /** 一次同步回读的阻塞时长；负数（本帧没有回读）忽略。 */
    fun onStall(ns: Long) {
        if (!engaged || ns < 0L) return
        stalls[stallCount % stalls.size] = ns; stallCount++
    }

    fun update(want: Int, streaming: Boolean, nowMs: Long) {
        if (!engaged) return
        if (wasStreaming && !streaming) { locked = false; samples = 0; wantCount = 0 } // 停播回到预览：重新测量
        wasStreaming = streaming
        if (want >= 0) {
            wants[wantCount % wants.size] = want; wantCount++; samples++
        }
        if (!locked) {
            if (samples < WARMUP + MEASURE) return
            var target = recentWantPercentile(MEASURE).coerceIn(1, DelaySchedule.MAX_DELAY)
            if (!deferredReadback && stallPercentile() > STALL_DEFERRED_NS) {
                deferredReadback = true
                target = minOf(target + 1, DelaySchedule.MAX_DELAY)
            }
            if (streaming && target > delay) lastGrowAt = nowMs
            delay = if (streaming) maxOf(delay, target) else target
            locked = true
            return
        }
        if (want >= 0 && delay < DelaySchedule.MAX_DELAY) {
            if (streaming) {
                if (wantCount >= GROW_WINDOW &&
                    (lastGrowAt == Long.MIN_VALUE || nowMs - lastGrowAt >= GROW_COOLDOWN_MS) &&
                    recentWantPercentile(GROW_WINDOW) >= delay + GROW_MARGIN
                ) {
                    delay++
                    lastGrowAt = nowMs
                }
            } else if (wantCount >= PREVIEW_GROW_WINDOW &&
                (lastGrowAt == Long.MIN_VALUE || nowMs - lastGrowAt >= PREVIEW_GROW_COOLDOWN_MS) &&
                recentWantPercentile(PREVIEW_GROW_WINDOW) >= delay + 1
            ) {
                // 预览态允许在推理变慢（CPU 回退、热降频、剧烈晃动）时向上补齐延后，避免画面永久落后于关键点。
                delay++
                lastGrowAt = nowMs
            }
        }
    }

    /** 最近 [n] 个 want 的 p90（want ∈ 0..SLOTS，按计数求，不分配）。 */
    private fun recentWantPercentile(n: Int): Int {
        val available = minOf(n, wantCount, wants.size)
        if (available == 0) return 0
        histogram.fill(0)
        for (k in 1..available) histogram[wants[(wantCount - k) % wants.size].coerceIn(0, DelaySchedule.SLOTS)]++
        val rank = kotlin.math.ceil(available * LOCK_PERCENTILE).toInt()
        var seen = 0
        for (value in histogram.indices) {
            seen += histogram[value]
            if (seen >= rank) return value
        }
        return DelaySchedule.SLOTS
    }

    private fun stallPercentile(): Long {
        val n = minOf(stallCount, stalls.size)
        if (n == 0) return 0L
        val sorted = stalls.copyOf(n).also { it.sort() }
        return sorted[(kotlin.math.ceil(n * LOCK_PERCENTILE).toInt() - 1).coerceIn(0, n - 1)]
    }

    companion object {
        const val PRIOR_GPU = 2
        const val PRIOR_CPU = 3
        const val WARMUP = 15
        const val MEASURE = 45
        const val LOCK_PERCENTILE = 0.9
        const val STALL_DEFERRED_NS = 3_000_000L
        const val GROW_WINDOW = 90
        const val GROW_MARGIN = 2
        const val GROW_COOLDOWN_MS = 30_000L
        const val PREVIEW_GROW_WINDOW = 30
        const val PREVIEW_GROW_COOLDOWN_MS = 3_000L
        private const val STALL_SAMPLES = 64
    }
}
