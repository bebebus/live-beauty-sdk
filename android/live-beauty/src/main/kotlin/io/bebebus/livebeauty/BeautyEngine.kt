package io.bebebus.livebeauty

import android.content.Context

/**
 * 外部美颜引擎接缝（仅 Android）：VideoEffectsFilter 的 engineFactory 返回非 null 且引擎健康时，drawFilter 跳过自研的
 * 跟踪、画面延后、美妆与 LUT，把相机纹理交给 [process]，再用原画程序把输出直通合成到滤镜 FBO；负载降到 3 级时整体旁路。
 * 默认 engineFactory = null，即内置自研管线；main 分支不引入任何 SDK 依赖。
 *
 * 去重：VideoEffectsFilter.draw 的 FrameReuseGate 保证每个相机帧至多调用一次 [process]（预览与编码共用一次结果），
 * ForceRenderer 保活重绘同样被复用。生命周期跟随 GL 上下文：每个新上下文 [onGlCreated] 一次，拆除时 [onGlReleased]。
 *
 * 第三方或自研引擎统一实现约定：
 * 1. 输入纹理固定为竖屏正立 GL_TEXTURE_2D（RGBA），无需额外旋转；
 * 2. 所有 GL 操作在渲染线程完成，仅允许变更 FBO 绑定与 viewport；
 * 3. 若引擎内部存在多帧缓冲延迟，通过 [latencyFrames] 声明以便音画时间戳对齐。
 */
interface BeautyEngine {
    /** GL 线程、新上下文。抛异常表示不可用，回退自研管线。 */
    fun onGlCreated(context: Context)

    /**
     * GL 线程、参数快照变化时在 [process] 之前调用，不得阻塞。宿主按快照引用对账（[EngineConfigSync]）：主线程只写
     * VideoEffectsConfig，新上下文（[onGlCreated] 之后）的第一帧必定调用一次，所以不会停在建引擎之前的旧参数。
     */
    fun configure(config: VideoEffectsConfig)

    /**
     * GL 线程。[inputTexture] 为 GL_TEXTURE_2D、RGBA、竖屏正立（与 MediaPipe 采样的是同一张纹理），尺寸等于编码尺寸，
     * [timestampNs] 为 SurfaceTexture 时间戳。返回同尺寸的 2D 纹理；直通时返回 [inputTexture]。
     * 只可改变 FBO 绑定与 viewport（调用方随后恢复），其余 GL 状态须还原；p95 预算 ≤ 15ms。
     */
    fun process(inputTexture: Int, width: Int, height: Int, timestampNs: Long): Int

    /**
     * 引擎自身的输出延迟（帧，0..[EngineStamps.MAX_LATENCY_FRAMES]，超出按上限计）：[process] 返回的是 n 帧之前那一帧的结果时填 n。
     * 宿主据此把发布给 FramePacing 的内容时间取成那一帧的相机时间戳（[EngineStamps]），编码 PTS 与声音延后随之对齐；
     * 没有内部缓冲、同步返回本帧结果的引擎填 0。
     */
    val latencyFrames: Int

    /** 任何线程可调（切镜头、开预览）：重置引擎内部的人脸/美体/分割跟踪状态。 */
    fun resetTracking() {}

    /**
     * [contextAlive] 为 true：GL 线程、创建对象的上下文仍是当前上下文，释放 GL 对象；
     * false：上下文已销毁（OpenGlView.stop 在别的线程拆除），只丢引用。
     */
    fun onGlReleased(contextAlive: Boolean)
}

/**
 * 外部引擎的参数对账（纯逻辑，只在 GL 线程使用）：每帧 [process] 之前按 VideoEffectsConfig 快照的引用比较，变化了才
 * configure。configure 因此只在 GL 线程调用，不会与主线程写参数交错（以前主线程 setter 直接 configure，与建引擎并发时
 * 新引擎可能停在旧参数上）；换引擎、重建上下文后 [reset]，由下一帧补配。
 */
internal class EngineConfigSync {
    private var applied: VideoEffectsConfig? = null

    fun reset() { applied = null }

    /** @return 本帧是否调用了 configure；configure 抛出的异常原样抛给调用方（按引擎故障处理），之后不再重试同一快照。 */
    fun sync(engine: BeautyEngine, config: VideoEffectsConfig): Boolean {
        if (config === applied) return false
        applied = config
        engine.configure(config)
        return true
    }
}

/**
 * 外部引擎输出延迟的补偿（纯逻辑，只在 GL 线程使用）：保留最近 [CAPACITY] 个相机帧时间戳。引擎声明 latencyFrames = n 时
 * 本帧输出的是 n 帧之前的画面，[content] 取那一帧的时间戳作为发布给 FramePacing 的内容时间；历史不足时取最早的一帧。
 */
internal class EngineStamps {
    private val stamps = LongArray(CAPACITY)
    private var head = -1
    private var count = 0

    fun reset() { head = -1; count = 0 }

    /** 每个新相机帧调用一次（同一帧的第二次绘制不调用）。 */
    fun push(stamp: Long) {
        head = (head + 1) % CAPACITY
        stamps[head] = stamp
        if (count < CAPACITY) count++
    }

    /** [frames] 帧之前那一帧的时间戳（按 0..[MAX_LATENCY_FRAMES] 截断）；还没有任何帧时为 0（未知）。 */
    fun content(frames: Int): Long {
        if (count == 0) return 0L
        val n = frames.coerceIn(0, MAX_LATENCY_FRAMES).coerceAtMost(count - 1)
        return stamps[(head - n + CAPACITY) % CAPACITY]
    }

    companion object {
        const val CAPACITY = 5
        const val MAX_LATENCY_FRAMES = CAPACITY - 1
    }
}
