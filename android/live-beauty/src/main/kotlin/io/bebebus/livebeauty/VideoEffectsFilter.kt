package io.bebebus.livebeauty

import android.content.Context
import android.content.res.AssetManager
import android.opengl.EGL14
import android.opengl.EGLContext
import android.opengl.GLES20.*
import android.os.SystemClock
import android.os.Trace
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * GL 线程处理，预览与编码共用输出纹理；不改变朝向/镜像。GLHost 每帧调用两次（预览 + 编码/拍照），
 * 同一相机帧、同一输入的第二次直接沿用 FBO 中的结果（见 draw 与 FrameReuseGate）。
 * BaseFilterRender.draw() = 绑定自身 FBO → viewport → drawFilter() → glDrawArrays → disableResources()，
 * 所以 drawFilter 只负责设好程序、属性、纹理与 uniform，且不得抛出。
 * 渲染故障（着色器、FBO）→ onFailure：回原画并发 effects_failed；人脸故障 → onFaceFailure：只停用塑形与人脸遮罩。
 * 线程：除 config/claimAttach/resetFaceTracking 与拆除路径的 release 外，状态只在 GL 线程读写；
 * GLHost.stop() 不等待正在执行的 draw，拆除路径的 release 因此只置标志（见 release）。
 */
class VideoEffectsFilter(
    private val onFailure: () -> Unit,
    private val onFaceFailure: () -> Unit,
    /** 负载级别变化（GL 线程调用，升降两个方向都回调 0..3）：1 停美体与背景，2 再停人脸效果，3 再停磨皮，见 EffectsLoadGovernor。 */
    private val onLoadReduced: (Int) -> Unit = {},
    /** 「开了长腿但没拍到腿」状态变化（GL 线程调用），见 BodyLegsMonitor。 */
    private val onLegsHidden: (Boolean) -> Unit = {},
    /** LUT 下载缓存目录（通常是 Context.cacheDir）；为 null 时仅支持内置资产 LUT。 */
    lutCacheDir: java.io.File? = null,
    /** 内置 LUT 资产管理器（通常是 Context.assets）；为 null 时仅支持 https 远程 LUT。 */
    lutAssets: AssetManager? = null,
    /** 暗光状态变化（GL 线程调用），见 LowLightDetector。 */
    private val onLowLight: (Boolean) -> Unit = {},
    /** 当前码率 / 上限（任何线程可读，未推流为 1），见 BitrateEffectsTuning。 */
    private val bitrateRatio: () -> Float = { 1f },
    /** 当前相机帧的 SurfaceTexture 时间戳（GL 线程调用）；0 表示未知，此时每次都绘制。 */
    private val frameStamp: () -> Long = { 0L },
    /** 编码出帧节奏：每次绘制发布「最新相机帧 → 实际显示的相机帧」，编码 PTS 据此取内容时间（见 FramePacing）；null 不发布。 */
    private val pacing: FramePacing? = null,
    /** 是否在推流（GL 线程调用）：推流中画面延后只增不减，见 PacingLock。 */
    private val streaming: () -> Boolean = { false },
    /** 外部美颜引擎（SDK 接缝，见 BeautyEngine）；null 走内置自研管线。 */
    private val engineFactory: (() -> BeautyEngine)? = null,
) : BaseFilterRender() {
    @Volatile var config = VideoEffectsConfig.original
        set(value) {
            field = value
            // 跟踪启停只由 drawFilter 按 TrackingGrace 决定：参数临时拖到 0、按住对比原画都不关 tracker（代际不变、不重新渐入）。
            // 外部引擎的 configure 不在这里调：由 GL 线程的 drawEngine 按快照引用对账（EngineConfigSync）。
            lutLoader.request(if (value.enabled) value.lutUrl else "")
        }
    @Volatile var healthy = true
        private set
    @Volatile var faceHealthy = true
        private set
    /** 美体（人体姿态）故障只停用美体，不回原画、不通知 Dart。 */
    @Volatile var bodyHealthy = true
        private set
    private val faceFailureSent = AtomicBoolean(false)
    /** GLHost.stop() 已在别的线程 release：GL 线程停止人脸/预处理、不再上报故障，直到新上下文 initGlFilter。 */
    @Volatile private var released = false
    /**
     * 当前上下文的 GL 线程（initGlFilter 记录）。GLHost.stop() 不等待在途绘制，新上下文的 initGlFilter 可能先于旧线程的
     * 那一帧结束；旧线程上的残余绘制据此视为失效：不上报故障、不建对象、不再推进各道处理的状态。
     */
    @Volatile private var glThread: Thread? = null
    private val stale get() = released || Thread.currentThread() !== glThread
    /** SET 已排队或已挂载；release 清除、initGlFilter 置位。见 claimAttach。 */
    private val attachment = AtomicBoolean(false)
    private var glContext: EGLContext? = null // 仅 GL 线程写；release 用它判断是否在本滤镜的活上下文里。
    private var originalRetried = false
    private var context: Context? = null
    @Volatile private var tracker: FaceLandmarkTracker? = null
    // 共享小图（长边 384，竖屏 216×384）：人脸、美体、分割与测光每帧至多画一次、回读一次。
    private val faceSampler = FaceTextureSampler(384)
    private val guidedFilterPass = GuidedFilterPrePass()
    private var guidedFilterHealthy = true // 1/3 快速导向滤波故障只独立回退旧版 BeautyPrePass。
    private val guidedSpread = GuidedSpread() // 仅 GL 线程：导向滤波采样间距随脸宽自适应。
    private val skinMaskPass = SkinMaskPass()
    private var skinMaskHealthy = true // 468 点多边形皮肤遮罩故障只独立回退椭圆遮罩。
    private val prePass = BeautyPrePass()
    private val timeline = FaceTimeline() // 仅 GL 线程：按被画那一帧的采样时间取人脸几何。
    private val frameDelay = FrameDelay() // 仅 GL 线程：画面延后到人脸结果已出来的那一帧。
    private var alignHealthy = true // 画面对齐故障只停用对齐（回到不延后），每个上下文复位。
    private val alignedFace = FloatArray(FaceGeometrySmoother.STATE_SIZE) // 脸、眼、唇椭圆按外推量扩张后的人脸几何，仅合成用。
    private var drawStamp = 0L // 仅 GL 线程：draw() 读到的相机帧时间戳，drawFilter 据此判断是不是新相机帧。
    private var alignOffSince = Long.MIN_VALUE // 仅 GL 线程：关闭美颜（且未推流）的起点，满 ALIGN_FREE_MS 结束画面对齐会话。
    private var engaged = false // 仅 GL 线程：本会话已开始画面对齐（FrameDelay 每场锁定，之后一直维持）。
    /** 本帧实际显示的相机帧时间戳（GL 线程写）；发布给 FramePacing。 */
    @Volatile private var contentStampNs = 0L
    private var shownStampNs = 0L // 仅 GL 线程：最近一次成功写进输出 FBO 的相机帧时间戳。
    /** 推流画面比最新相机帧晚多少毫秒（画面延后，仅诊断；音画对齐见 FramePacing.audioDelayMs）。 */
    @Volatile var videoDelayMs = 0L
        private set
    private val faceGrace = TrackingGrace() // 仅 GL 线程：三路跟踪不再需要后保留 3 s 再停。
    private val bodyGrace = TrackingGrace()
    private val segGrace = TrackingGrace()
    private val fades = LoadFades() // 仅 GL 线程：负载降级各级效果的 400ms 渐变。
    // 仅 GL 线程：延迟回读（DEFERRED）时上一帧已画进共享小图、还没取票的采样（采样时间、尺寸，各路到期时的代际，没到期为
    // null）。本帧开头 collectDeferred 才按它取票，取到才读回并交给 worker：单任务在途只覆盖推理本身，GPU 人脸与分割仍每帧采。
    private var stagedAt = Long.MIN_VALUE
    private var stagedWidth = 0
    private var stagedHeight = 0
    private var stagedFaceInterval = FaceTrackingState.FRAME_INTERVAL_MS
    private var stagedFace: Long? = null
    private var stagedBody: Long? = null
    private var stagedSeg: Long? = null
    private var frameReads = 0L // 仅 GL 线程：本帧开始时共享小图的累计读回次数，测光据此判断能否复用。
    @Volatile private var engine: BeautyEngine? = null
    private var engineHealthy = true // 外部引擎故障只回退自研管线。
    private val engineSync = EngineConfigSync() // 仅 GL 线程：外部引擎参数按快照引用对账。
    private val engineStamps = EngineStamps() // 仅 GL 线程：最近几个相机帧时间戳，按引擎输出延迟取内容时间。
    private var lastWorkNs = Long.MAX_VALUE // 仅 GL 线程：上一帧 draw 的耗时，预热调度用。
    private var warmupIndex = 0
    private var lastFrameStamp = 0L
    private var lastFrameTime = Long.MIN_VALUE
    private var epoch = Long.MIN_VALUE
    private var lastGeometry: FaceGeometry? = null
    // 仅 GL 线程：已处理过的最近一次完成采样时间（本代际），新完成的结果里没有脸就按它的采样时间记丢脸。
    private var lastCompletedSeen = Long.MIN_VALUE
    // 仅 GL 线程：形变器时间平滑状态对应的时间线轮次，变化（新一轮、换人、重新检出）时重置。
    private var deformerRun = -1
    private val meshFade = MeshFallbackFade() // 仅 GL 线程：网格微整与回退路径之间的交叉渐变（R14）。
    private val eyeValues = FloatArray(8)
    private val reuse = FrameReuseGate() // 仅 GL 线程。
    @Volatile private var bodyTracker: BodyPoseTracker? = null
    private val bodySmoother = BodyGeometrySmoother() // 仅 GL 线程。
    private val bodyPass = BodyWarpPass()
    private var bodyEpoch = Long.MIN_VALUE
    private var lastBody: BodyGeometry? = null
    private var lastBodyState: FloatArray? = null
    private val bodyValues = FloatArray(12)
    private val legsMonitor = BodyLegsMonitor() // 仅 GL 线程。
    @Volatile private var segmenter: PersonSegmenter? = null
    @Volatile var segHealthy = true
        private set
    private val maskPresence = MaskPresence() // 仅 GL 线程。
    private val bgPass = BackgroundBlurPass()
    private var segEpoch = Long.MIN_VALUE
    private var lastMaskStamp = Long.MIN_VALUE // 已上传到 bgPass 的遮罩对应的采样时间。
    // 头发遮罩（毛发边界第二批）：头发分割故障或 p50 超限只停用头发遮罩，本实例内一直生效（同 segHealthy），画面回到第一批启发式。
    @Volatile private var hairSegmenter: HairSegmenter? = null
    @Volatile var hairHealthy = true
        private set
    private val hairGrace = TrackingGrace()
    private val hairGate = HairMaskGate() // 仅 GL 线程：存在度、新鲜度与已上传遮罩的采样时间。
    private var hairEpoch = Long.MIN_VALUE
    private var stagedHair: Long? = null
    private var stagedHairInterval = HairMaskTuning.INTERVAL_MS
    private val hairTexture = HairMaskTexture()
    private val semanticPass = SemanticAlphaPass()
    // GL 道故障（每个上下文复位）：SemanticAlphaPass 停用后 A 恒为 1；遮罩上传失败时它与背景合并道都停；合并道自身的故障见
    // bgPass.hairHealthy。
    private var semanticHealthy = true
    private var bgHairHealthy = true
    private val hairAffine = FloatArray(HairMaskTuning.AFFINE_SIZE) // 显示帧 → 遮罩时刻的运动补偿（两行 vec4）。
    private val hairPoseMask = FloatArray(FaceTimeline.POSE_SIZE)
    private val hairPoseDisplay = FloatArray(FaceTimeline.POSE_SIZE)
    private val hairOval = FloatArray(SemanticAlphaPass.OVAL_SIZE) // A 通道：网格微整后的脸椭圆。
    private val hairBackgroundOval = FloatArray(SemanticAlphaPass.OVAL_SIZE) // 合并道：原帧（变形前）的脸椭圆。
    private val hairBackground = HairBackgroundInput(0, 0f, hairAffine, hairBackgroundOval)
    private val hairMesh = MeshInput(FloatArray(0), false)
    private val hairAnchor = FloatArray(HairMaskTuning.ANCHOR_SIZE)
    private var lastGeometryAt = Long.MIN_VALUE // lastGeometry 的采样时间：等于 lastCompletedSeen 表示最近一次人脸结果有脸。
    private val faceShapePass = FaceShapePass()
    private val faceShapeValues = FloatArray(FaceShapePass.VALUES)
    private var faceShapeHealthy = true // 局部修饰/回退脸型故障只停用该道。
    private val faceRetouchPass = FaceRetouchMeshPass()
    private var faceRetouchHealthy = true // 468 点解剖子网格局部精修故障只回退旧版 FaceShapePass。
    private val faceRoi = FaceRoiSelector()
    private val faceMeshPass = FaceMeshWarpPass()
    private val faceMeshDeformer = FaceMeshDeformer()
    private var faceMeshHealthy = true // 478 点网格微整故障只停用网格微整，回退既有路径。
    private val warpedGeometry = FloatArray(FaceGeometrySmoother.STATE_SIZE) // 网格微整后的人脸遮罩状态，供主合成使用。
    private val liftedLandmarks = FloatArray(FaceMeshFrame.COORD_COUNT) // 天鹅颈上移后的 478 点，供后续通道复用。
    // GL 处理故障（每个上下文复位）与识别模型故障（bodyHealthy / segHealthy，本实例内一直生效）分开。
    private var bodyPassHealthy = true
    private var bgPassHealthy = true
    private val liftedFace = FloatArray(FaceGeometrySmoother.STATE_SIZE) // 天鹅颈上移后的人脸几何，仅合成用。
    private var lutRetryAt = Long.MIN_VALUE // 下载失败的 LUT 再次尝试的时间。
    private val load = EffectsLoadGovernor() // 仅 GL 线程。
    private var lastProcessing = false // 仅 GL 线程：上一帧是否在做美颜处理，供负载统计。
    private val light = LowLightDetector()
    private var lastLightSample = Long.MIN_VALUE
    private var lightHealthy = true // 采样故障只停用暗光检测，不影响画面。
    private var lightLuma = Float.NaN // 仅 GL 线程：平滑后的画面中部亮度，驱动暗光降噪强度。
    private val denoisePass = TemporalDenoisePass()
    private var denoiseHealthy = true // 降噪故障只停用降噪。
    private val lutLoader = LutLoader(lutCacheDir, lutAssets)
    private val lutPass = LutPass()
    private var lutHealthy = true // LUT 故障只停用 LUT。
    private var lutFailed = "" // 下载、解码失败或尺寸不对的 LUT 地址，本上下文内不再尝试。
    private val makeupPass = MakeupPass(lutAssets)
    private var makeupHealthy = true // 实时美妆故障只停用美妆，不影响磨皮/微整/LUT 与推流。
    private var makeupPending = false
    private var makeupConfig = VideoEffectsConfig.original
    private var makeupPresence = 0f
    private var makeupSteadyFace = 1f
    private var makeupSteadyEye = 1f
    private var makeupEyeKeep = 1f
    private var makeupRightIrisGate = 0f
    private var makeupLeftIrisGate = 0f
    private val makeupLandmarks = FloatArray(FaceMeshFrame.COORD_COUNT)
    /**
     * 任何线程可调（切镜头、开预览）；平滑器在 GL 线程比较代际后自行 reset。头发分割的历史一并作废：代际加一，
     * GL 线程第 1c' 段据此复位 hairGate，在途的旧票按代际丢弃，旧镜头的头发遮罩不会在新镜头上再用（NOTICE 隐私段）。
     */
    fun resetFaceTracking() { engine?.resetTracking(); tracker?.state?.reset(); segmenter?.state?.reset(); hairSegmenter?.state?.reset() }
    /**
     * 主线程：滤镜既未挂载、也没有排队中的 SET 时返回 true 并占位，调用方随即 glView.setFilter。
     * GLHost.setFilter 只入队，filtersCount() 看不到排队中的 SET；同一实例 SET 两次会在活上下文里
     * release + initGl，重建全部 GL 对象并重启人脸跟踪。
     */
    fun claimAttach(): Boolean = attachment.compareAndSet(false, true)
    /** 人脸跟踪启用谓词（规格 4.2）；按住对比原画不算关闭（不重置代际），drawFilter 再经 TrackingGrace 决定启停。 */
    private fun wantsFace(value: VideoEffectsConfig) = faceHealthy && healthy && value.enabled &&
        (value.hasFaceReshape() || value.hasMakeup() || value.whiten > 0f ||
            value.darkCircles > 0f || value.brightEyes > 0f || value.teethWhiten > 0f || value.nasolabial > 0f ||
            value.rosy > 0f || value.radiance > 0f || (value.skinOnly && value.smoothing > 0f))
    /** 美体启用谓词，见 wantsFace。 */
    private fun wantsBody(value: VideoEffectsConfig) = bodyHealthy && healthy && value.enabled &&
        (value.waistSlim > 0f || value.bustEnhance > 0f || value.legLengthen > 0f ||
            value.shoulderSlim > 0f || value.neckLengthen > 0f)
    /** 背景虚化启用谓词，见 wantsFace。 */
    private fun wantsBackground(value: VideoEffectsConfig) = segHealthy && healthy && value.enabled &&
        value.backgroundBlur > 0f
    /** worker 或 GL 线程；分割器由 GL 线程关闭。背景虚化故障只停用背景虚化。 */
    private fun failSegmentation() {
        segHealthy = false
        segmenter?.state?.enable(false)
    }
    /**
     * worker 或 GL 线程；分割器由 GL 线程关闭。头发遮罩故障（模型、推理、p50 超限、取样）只停用头发遮罩：下一帧起 A 通道恒为 1、
     * 不跑合并道，画面回到第一批；不回原画、不发 effects_failed（HairSegmenter 只记一次日志）。
     */
    private fun failHair() {
        hairHealthy = false
        hairSegmenter?.state?.enable(false)
    }
    /** worker 或 GL 线程；tracker 由 GL 线程关闭。 */
    private fun failBody() {
        bodyHealthy = false
        bodyTracker?.state?.enable(false)
    }
    private fun failEffects() {
        if (!healthy) return
        healthy = false; config = VideoEffectsConfig.original
        onFailure()
    }
    /** worker 或 GL 线程；至多通知一次，tracker 由 GL 线程关闭。 */
    private fun failFace() {
        faceHealthy = false
        tracker?.state?.enable(false)
        if (faceFailureSent.compareAndSet(false, true)) onFaceFailure()
    }
    /** GLHost 的 RenderErrorCallback（GL 线程）：意外的渲染异常同样回原画；拆除期间的异常不算。 */
    fun failRender() { if (!stale) failEffects() }
    private var composite = 0
    private var original = 0
    private var program = 0
    private var fallbackTexture = 0
    private val vertices = ByteBuffer.allocateDirect(16 * 4).order(ByteOrder.nativeOrder())
        .asFloatBuffer().apply {
            put(floatArrayOf(-1f,-1f,0f,0f, 1f,-1f,1f,0f, -1f,1f,0f,1f, 1f,1f,1f,1f))
            position(0)
        }
    private var position = 0
    private var texture = 0
    private var sampler = 0
    private var blur = 0
    private var skinMask = 0
    private var frame = 0
    private var params = 0
    private var details = 0
    private var tone = 0
    private var cheeks = 0
    private var eyes = 0
    private var oval = 0
    private var mouth = 0
    private var eyeMask = 0
    private var lutSampler = 0
    private var grade = 0
    /**
     * 停止/重开预览后可能再次调用：只重建程序，健康状态保留。总在 GL 线程、新上下文里执行，
     * 所以上一个上下文的 id（已随上下文销毁）在这里忘记，人脸状态也在这里从头开始。
     */
    override fun initGlFilter(context: Context) {
        // 下面会静默复位的状态：之前是开着的就告诉 Dart 解除，否则只在原生重启（Dart 仍在预览）时提示会一直挂着。
        try {
            if (light.dark) onLowLight(false)
            if (legsMonitor.hidden) onLegsHidden(false)
            if (load.level > 0) onLoadReduced(0)
        } catch (_: Exception) {}
        glThread = Thread.currentThread()
        tracker?.close(); tracker = null // 拆除路径已关闭；新上下文按需重建。
        bodyTracker?.close(); bodyTracker = null
        segmenter?.close(); segmenter = null
        hairSegmenter?.close(); hairSegmenter = null
        faceShapePass.release(); faceShapeHealthy = true; bodyPassHealthy = true; bgPassHealthy = true
        faceRetouchPass.release(); faceRetouchHealthy = true
        faceMeshPass.release(); faceMeshHealthy = true
        bgPass.release(); maskPresence.reset(); segEpoch = Long.MIN_VALUE
        lastMaskStamp = Long.MIN_VALUE
        hairTexture.release(); semanticPass.release(); semanticHealthy = true; bgHairHealthy = true
        hairGate.reset(); hairEpoch = Long.MIN_VALUE
        guidedFilterPass.release(); guidedFilterHealthy = true; guidedSpread.reset()
        skinMaskPass.release(); skinMaskHealthy = true
        // 还没取票的延迟回读记录随旧上下文丢弃（小图随 faceSampler.release 一起忘记；记录不占单任务在途，无票可作废）。
        clearStaged()
        prePass.release(); faceSampler.release(); bodyPass.release()
        light.reset(); lastLightSample = Long.MIN_VALUE; lightHealthy = true
        lightLuma = Float.NaN; denoisePass.release(); denoiseHealthy = true
        // 推流中重建上下文（切后台离屏等）：保留锁定的画面延后，直播中不重新测量缩帧。
        val keepLock = engaged && streaming()
        frameDelay.release(keepLock); engaged = keepLock; alignHealthy = true; alignOffSince = Long.MIN_VALUE
        lutPass.release(); lutHealthy = true; lutFailed = ""
        makeupPass.release(); makeupHealthy = true; makeupPending = false
        lutLoader.restart { config.let { if (it.enabled) it.lutUrl else "" } }
        lutRetryAt = Long.MIN_VALUE
        resetFaceState(); resetBodyState()
        reuse.invalidate()
        load.reset(); fades.reset(); lastProcessing = false; legsMonitor.reset()
        faceGrace.reset(); bodyGrace.reset(); segGrace.reset(); hairGrace.reset()
        shownStampNs = 0L; contentStampNs = 0L; lastWorkNs = Long.MAX_VALUE; warmupIndex = 0
        // 上一个引擎的上下文已随旧上下文销毁（拆除路径不在别的线程碰它）：先丢引用再通知，再按需新建。
        val previousEngine = engine
        engine = null; engineHealthy = true; engineSync.reset(); engineStamps.reset()
        previousEngine?.let { try { it.onGlReleased(false) } catch (_: Exception) {} }
        released = false
        attachment.set(true)
        glContext = EGL14.eglGetCurrentContext()
        this.context = context
        fallbackTexture = createFallbackTexture()
        original = build(VideoEffectsShaders.ORIGINAL)
        composite = if (!healthy) 0 else build(VideoEffectsShaders.FRAGMENT_PRO)
        if (composite == 0) failEffects()
        originalRetried = false
        use(if (healthy) composite else original)
        engineFactory?.let { factory ->
            try {
                // 只建引擎并发布引用；参数由第一帧 drawEngine 按快照对账配置（configure 只在 GL 线程调用）。
                val created = factory()
                created.onGlCreated(context)
                engine = created
            } catch (_: Exception) { engineHealthy = false; engine = null }
        }
    }
    private fun createFallbackTexture(): Int = try {
        val ids = IntArray(1)
        glGenTextures(1, ids, 0)
        val tex = ids[0]
        if (tex != 0) {
            val px = ByteBuffer.allocateDirect(4).apply {
                put(0); put(0); put(0); put(255.toByte()); position(0)
            }
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px)
            glBindTexture(GL_TEXTURE_2D, 0)
        }
        tex
    } catch (_: Exception) { 0 }
    private fun build(fragment: String) = try {
        GlUtil.createProgram(VideoEffectsShaders.VERTEX, fragment).also { check(it > 0) }
    } catch (_: Exception) { 0 }
    private fun resetFaceState() {
        timeline.reset(); faceRoi.reset(); epoch = Long.MIN_VALUE; lastGeometry = null; lastGeometryAt = Long.MIN_VALUE
        lastCompletedSeen = Long.MIN_VALUE; meshFade.reset()
        lastFrameStamp = 0L; lastFrameTime = Long.MIN_VALUE; alignOffSince = Long.MIN_VALUE; videoDelayMs = 0L
    }
    private fun resetBodyState() {
        bodySmoother.reset(); bodyEpoch = Long.MIN_VALUE; lastBody = null; lastBodyState = null
    }
    private fun use(id: Int) {
        program = id
        // id 为 0 时不查询：glGet*Location(0) 留下的 GL_INVALID_VALUE 会被之后的 glGetError 检查误认成故障。
        fun attribute(name: String) = if (id == 0) -1 else glGetAttribLocation(id, name)
        fun uniform(name: String) = if (id == 0) -1 else glGetUniformLocation(id, name)
        position = attribute("aPosition")
        texture = attribute("aTextureCoord")
        sampler = uniform("uSampler")
        blur = uniform("uBlur")
        skinMask = uniform("uSkinMask")
        frame = uniform("uFrame")
        params = uniform("uParams")
        details = uniform("uDetails")
        tone = uniform("uTone")
        cheeks = uniform("uCheeks[0]")
        eyes = uniform("uEyes[0]")
        oval = uniform("uOval")
        mouth = uniform("uMouth")
        eyeMask = uniform("uEyeMask[0]")
        lutSampler = uniform("uLut")
        grade = uniform("uGrade")
    }
    /**
     * 同一相机帧、同一参数的第二次调用直接返回，FBO 保留第一次的结果（证据见 FrameReuseGate）。
     * 判定键在绘制前读取：绘制期间参数或健康状态变化只会让下一次判定不命中、多画一次，不会沿用过期结果。
     */
    override fun draw() {
        val stamp = if (currentDirectStamp != Long.MIN_VALUE) currentDirectStamp else try { frameStamp() } catch (_: Exception) { 0L }
        val value = config
        val ok = healthy
        val face = faceHealthy
        val input = previousTexId
        if (stale) {
            reuse.invalidate()
        } else if (reuse.canReuse(stamp, value, input, width, height, ok, face)) {
            // 发布点 1：同一相机帧的第二次绘制，FBO 里仍是上次写入的内容。
            pacing?.publish(stamp, shownStampNs)
            return
        }
        drawStamp = stamp
        val start = System.nanoTime()
        var published = false
        Trace.beginSection("beauty.draw")
        try {
            published = render(stamp, value, input, ok, face, start)
        } finally {
            Trace.endSection()
            // 发布点 4：绘制中途抛出（交给 RenderErrorCallback）：输出未定义，按「没有新内容」发布，不回落到最新帧。
            if (!published) pacing?.publish(stamp, shownStampNs)
        }
    }

    /** draw() 的主体；返回时已发布本帧。 */
    private fun render(stamp: Long, value: VideoEffectsConfig, input: Int, ok: Boolean, face: Boolean, start: Long): Boolean {
        super.draw()
        // 合成之后：实时美妆（若开启）从中间纹理画回本滤镜输出 FBO（基类结束时绑的是 FBO 0，这里同样收尾）。
        // LUT 已在主合成内查表（纹理单元 3），在美妆之前。
        val doMakeup = makeupPending
        makeupPending = false
        var postBroken = false
        if (doMakeup && !stale) {
            try {
                makeupPass.render(
                    outputFbo = renderHandler.fboId[0],
                    width = width,
                    height = height,
                    landmarks = makeupLandmarks,
                    aspect = width.toFloat() / height.coerceAtLeast(1),
                    rightIrisGate = makeupRightIrisGate,
                    leftIrisGate = makeupLeftIrisGate,
                    presence = makeupPresence,
                    steadyFace = makeupSteadyFace,
                    steadyEye = makeupSteadyEye,
                    eyeMakeupKeep = makeupEyeKeep,
                    config = makeupConfig,
                )
            } catch (_: Exception) {
                postBroken = true
                if (!stale) makeupHealthy = false
            } finally {
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                glDisable(GL_BLEND)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
            }
        }
        // 美妆失败的这一帧输出 FBO 没写进新内容：不记为可复用，下一次绘制（故障道已停用）重画。
        if (stale || postBroken) {
            reuse.invalidate()
            // 发布点 2：拆除期的残余绘制直通最新帧。美妆失败多半发生在拷贝底图之前（建程序、上传贴图），合成画在美妆的
            // 中间纹理里，输出 FBO 仍是上一次的内容：按「没有新内容」发布。
            pacing?.publish(stamp, if (postBroken) shownStampNs else contentStampNs)
            return true
        }
        reuse.rendered(stamp, value, input, width, height, ok, face)
        shownStampNs = contentStampNs
        // 发布点 3：正常结束。
        pacing?.publish(stamp, contentStampNs)
        // 同一相机帧的第二次绘制已在上面复用返回，这里每个新相机帧只统计一次。
        val now = SystemClock.uptimeMillis()
        sampleLight(input, now)
        // 负载按 draw 墙钟耗时（含同步回读与后处理）统计；编码 swap 耗时只做诊断（FramePacing.takeSwapNs），不计入。
        val work = System.nanoTime() - start
        lastWorkNs = work
        load.onFrame(stamp, work, lastProcessing, now)?.let { level ->
            try { onLoadReduced(level) } catch (_: Exception) {}
        }
        warmup()
        return true
    }

    /**
     * 预热（pacing.F9）：未开播且上一帧耗时 < WARMUP_WORK_MS 时，每帧至多让一道处理提前建程序、取 location，
     * 免得第一次打开某项效果（多半在直播中）时在推流里卡一下。失败不算故障，真正用到时再按各自的规则处理。
     */
    private fun warmup() {
        if (warmupIndex >= WARMUP_STEPS || stale || !healthy || lastWorkNs >= WARMUP_WORK_MS * 1_000_000L) return
        if (try { streaming() } catch (_: Exception) { true }) return
        try {
            when (warmupIndex) {
                0 -> frameDelay.prepare()
                1 -> faceSampler.prepare()
                2 -> makeupPass.prepare()
                3 -> bgPass.prepare()
                // 头发遮罩的两个程序：A 通道（hairAlpha）与背景合并道（bgHair），各自失败互不影响。
                4 -> { try { semanticPass.prepare() } catch (_: Exception) {}; bgPass.prepareHair() }
                5 -> faceRetouchPass.prepare()
            }
        } catch (_: Exception) {}
        warmupIndex++
    }

    private fun drainGlErrors() { var n = 0; while (glGetError() != GL_NO_ERROR && n++ < 16) Unit }

    /** 每秒取一次相机原始帧的缩小图算亮度：本帧已读回共享小图就直接复用，否则单独同步取一次（不交给 worker）。 */
    private fun sampleLight(input: Int, now: Long) {
        if (!lightHealthy || input == 0 || width <= 0 || height <= 0) return
        if (lastLightSample != Long.MIN_VALUE && now - lastLightSample < LowLightDetector.INTERVAL_MS) return
        lastLightSample = now
        try {
            if (faceSampler.reads == frameReads) {
                faceSampler.capture(input, width, height, 0, sync = true)
                glViewport(0, 0, width, height)
            }
            val sample = checkNotNull(faceSampler.staging)
            val luma = LowLightDetector.meanLuma(sample.pixels, sample.width, sample.height)
            if (luma.isFinite()) lightLuma = if (lightLuma.isFinite()) (lightLuma + luma) / 2f else luma
            light.onSample(luma)?.let { dark ->
                try { onLowLight(dark) } catch (_: Exception) {}
            }
        } catch (_: Exception) {
            if (!stale) lightHealthy = false
        }
    }
    override fun drawFilter() {
        val value = config // 每帧不可变快照，避免参数撕裂。
        val now = SystemClock.uptimeMillis()
        // 拆除路径的 release 已执行（本帧与 GLHost.stop() 并发）：只走直通，不在将死的上下文里建对象，
        // 也不把拆除期的异常当成故障。
        val live = !stale
        // 先清掉之前（上一帧合成、LUT 等）遗留的 GL 错误，后面各道只为自己引起的错误负责。
        if (live) drainGlErrors()
        val active = live && healthy && value.enabled && !value.compareOriginal
        // 负载降级的各级效果按 400ms 渐变（LoadFades）；对应 tracker 只在系数到 0 之后才停。
        fades.update(load.level, now)
        val tier = load.level
        // 同一相机帧的第二次绘制（时间戳未知、复用判定不命中时）不再采样、不推进画面延后。
        val stamp = drawStamp
        val newFrame = if (stamp != 0L) stamp != lastFrameStamp
            else lastFrameTime == Long.MIN_VALUE || now - lastFrameTime >= SAME_FRAME_MS
        if (newFrame) { lastFrameStamp = stamp; lastFrameTime = now }
        contentStampNs = stamp
        frameReads = faceSampler.reads
        // 0. 外部美颜引擎（SDK 接缝）：跳过自研的跟踪、画面延后、美妆与 LUT，用原画程序直通引擎输出。
        val external = engine
        if (live && healthy && engineHealthy && external != null && original != 0) {
            drawEngine(external, value, active, tier, stamp, newFrame)
            return
        }
        // 引擎故障后回到自研合成程序。
        if (healthy && composite != 0 && program == original) use(composite)
        // 0'. 延迟回读：新帧开头为上一帧画好的共享小图取票（票的采样时间仍是上一帧），取到才读回并交给各自的 worker。
        if (live && newFrame && faceSampler.deferred) collectDeferred(now)
        // 1. 人脸跟踪：故障只停用依赖人脸的能力；不再需要后宽限 3 s 才停（TrackingGrace），2 级渐出到 0 后才停。
        var faceTracking = false
        var faceCompleted = Long.MIN_VALUE
        // 本帧快照时是否有人脸任务在途（与并入时间线的结果同一把锁取出）：HairMaskGate.update 据此只在停摆时用兜底。
        var faceInFlight = false
        var faceTicket: FaceTrackingState.Ticket? = null
        if (faceHealthy && live) try {
            val faceEnabled = healthy && faceGrace.keep(wantsFace(value), now) && !(tier >= 2 && fades.face <= 0f)
            if (faceEnabled && tracker == null) {
                val created = FaceLandmarkTracker(context!!) { failFace() }
                tracker = created
                epoch = Long.MIN_VALUE // 新实例的代际从头计数。
                // 与 release 配对：那边先置 released 再读 tracker，这边先写 tracker 再读 released（都是 volatile），
                // 至少一方看到对方，并发 release 时新建的 tracker 不会漏关。
                if (released) created.close()
            }
            tracker?.let { detector ->
                detector.state.enable(faceEnabled)
                // 画面对齐：GPU 推理每帧都采，CPU 约隔帧一采；负载 1 级起降到 15 Hz。
                val interval = when {
                    tier >= 1 -> FACE_THROTTLED_INTERVAL_MS
                    detector.gpu -> FaceTrackingState.FRAME_INTERVAL_MS
                    else -> FaceTrackingState.CPU_INTERVAL_MS
                }
                // DEFERRED 时本帧只判到期（尺寸、代际检查照常在这里做）并在第 1' 段画小图，下一帧开头再取票。
                if (newFrame) {
                    if (faceSampler.deferred) {
                        stagedFace = detector.state.due(now, width, height, interval)
                        stagedFaceInterval = interval
                    } else {
                        faceTicket = detector.state.begin(now, width, height, interval)
                    }
                }
                val snapshot = detector.state.snapshot(now)
                if (snapshot.epoch != epoch) {
                    epoch = snapshot.epoch
                    timeline.reset(); faceRoi.reset()
                    timeline.aspect = width.toFloat() / height.coerceAtLeast(1)
                    // 切镜头、尺寸变化等：人脸几何从头开始；画面延后缓冲不清空（FrameDelay 每场锁定，清空会让推流跳帧）。
                    lastCompletedSeen = Long.MIN_VALUE
                }
                snapshot.geometry?.let {
                    if (it !== lastGeometry) {
                        lastGeometry = it
                        lastGeometryAt = snapshot.sampleTimestamp
                        timeline.add(snapshot.sampleTimestamp, it.toState(), snapshot.mesh)
                        val m = snapshot.mesh
                        if (m != null) faceRoi.onFace(m.centerX, m.centerY, m.faceWidth, snapshot.sampleTimestamp, now)
                        else faceRoi.onFace(it.oval[0], it.oval[1], it.oval[2] * 1.78f, snapshot.sampleTimestamp, now)
                    }
                }
                // 丢脸按采样时间记：本代际新完成的结果里没有脸（或有脸的那个已不新鲜），presence 据此在被画那一帧防抖后退出。
                if (snapshot.completed != Long.MIN_VALUE && snapshot.completed > lastCompletedSeen) {
                    lastCompletedSeen = snapshot.completed
                    if (snapshot.geometry == null || snapshot.sampleTimestamp < snapshot.completed) {
                        timeline.addMiss(snapshot.completed)
                        faceRoi.onMiss()
                    }
                }
                // 丢脸时推理照样完成：按完成时间测延后，不因没有脸而中断。
                faceCompleted = snapshot.completed
                faceInFlight = snapshot.inFlight
                // 时间戳未知时分不清预览与编码两次绘制，不做对齐。
                faceTracking = faceEnabled && stamp != 0L
            }
        } catch (_: Exception) { if (!stale) failFace() }
        if (!faceHealthy) {
            tracker?.close(); tracker = null; faceTracking = false; faceTicket = null; stagedFace = null; timeline.reset()
        }
        // 1b. 美体：人体姿态跟踪，故障只停用美体；先取票并同步代际，稍后统一按 displayTime 取对齐帧。1 级起渐出后停。
        var bodyTracking = false
        var bodyTicket: BodyTrackingState.Ticket? = null
        if (bodyHealthy && live) try {
            val bodyEnabled = healthy && bodyGrace.keep(wantsBody(value), now) && !(tier >= 1 && fades.body <= 0f)
            if (bodyEnabled && bodyTracker == null) {
                val created = BodyPoseTracker(context!!) { failBody() }
                bodyTracker = created
                bodyEpoch = Long.MIN_VALUE
                if (released) created.close()
            }
            bodyTracker?.let { detector ->
                detector.state.enable(bodyEnabled)
                if (newFrame) {
                    if (faceSampler.deferred) stagedBody = detector.state.due(now, width, height, BODY_INTERVAL_MS)
                    else bodyTicket = detector.state.begin(now, width, height, BODY_INTERVAL_MS)
                }
                val snapshot = detector.state.snapshot(now)
                if (snapshot.epoch != bodyEpoch) {
                    bodyEpoch = snapshot.epoch; bodySmoother.reset(); lastBody = null; lastBodyState = null
                }
                bodyTracking = bodyEnabled && stamp != 0L
            }
        } catch (_: Exception) { if (!stale) failBody() }
        if (!bodyHealthy) {
            bodyTracker?.close(); bodyTracker = null; bodyTracking = false; bodyTicket = null; stagedBody = null; bodySmoother.reset()
        }
        // 1c. 背景虚化：人像分割，故障只停用背景虚化；只要不忙就每帧都采；1 级起与美体一起渐出后停。
        var segTracking = false
        var segRunning = false // 人像分割本帧在采（头发分割据此把间隔下限提到 200 ms，见 HairMaskTuning.interval）。
        var segCompleted = Long.MIN_VALUE
        var segTicket: MaskTrackingState.Ticket? = null
        if (segHealthy && live) try {
            val segEnabled = healthy && segGrace.keep(wantsBackground(value), now) && !(tier >= 1 && fades.body <= 0f)
            if (segEnabled && segmenter == null) {
                val created = PersonSegmenter(context!!) { failSegmentation() }
                segmenter = created
                segEpoch = Long.MIN_VALUE
                if (released) created.close()
            }
            segmenter?.let { detector ->
                detector.state.enable(segEnabled)
                segRunning = segEnabled
                if (newFrame) {
                    if (faceSampler.deferred) stagedSeg = detector.state.due(now, width, height, SEG_INTERVAL_MS)
                    else segTicket = detector.state.begin(now, width, height, SEG_INTERVAL_MS)
                }
                val snapshot = detector.state.snapshot(now)
                if (snapshot.epoch != segEpoch) { segEpoch = snapshot.epoch; maskPresence.reset(); lastMaskStamp = Long.MIN_VALUE }
                segCompleted = snapshot.completed
                segTracking = segEnabled && stamp != 0L
            }
        } catch (_: Exception) { if (!stale) failSegmentation() }
        if (!segHealthy) {
            segmenter?.close(); segmenter = null; segTracking = false; segTicket = null; stagedSeg = null; maskPresence.reset()
        }
        // 1c'. 头发遮罩：头发分割（CPU，基础 10 Hz，按推理 p50 自适应、开背景虚化时不快于 200 ms 一次），需要人脸跟踪且 A 通道
        // 作用得到的任一项开着；故障或 p50 超限只停用头发遮罩；1 级起与美体一起渐出后停。完成时间不并入 trackCompleted。
        // 画面里近期没有脸（人脸历史为空，或最新有脸样本已过去 FACE_RECENT_MS）时也停：A 通道与合并道都要用人脸几何，推理结果
        // 没人用。这一条不经 TrackingGrace（否则无脸时还要空跑 3 s）；读的是上一帧 timeline.update 之后的历史，差一帧无碍。
        var hairTicket: MaskTrackingState.Ticket? = null
        if (hairHealthy && live) try {
            val hairEnabled = HairMaskGate.enabled(hairHealthy,
                hairGrace.keep(HairMaskTuning.wants(hairHealthy, healthy, wantsFace(value), value), now), tier, fades.body,
                timeline.latest, now)
            if (hairEnabled && hairSegmenter == null) {
                val created = HairSegmenter(context!!) { failHair() }
                hairSegmenter = created
                hairEpoch = Long.MIN_VALUE
                if (released) created.close()
            }
            hairSegmenter?.let { detector ->
                // 最近 10 次推理的 p50 超过 MAX_P50_MS（攒满 8 次才判）：本次会话停用，回到第一批。
                val p50 = detector.inferenceP50Ms()
                if (!HairMaskTuning.healthy(p50)) failHair()
                detector.state.enable(hairEnabled && hairHealthy)
                val interval = HairMaskTuning.interval(p50, segRunning)
                if (newFrame) {
                    if (faceSampler.deferred) {
                        stagedHair = detector.state.due(now, width, height, interval)
                        stagedHairInterval = interval
                    } else {
                        hairTicket = detector.state.begin(now, width, height, interval)
                    }
                }
                val snapshot = detector.state.snapshot(now)
                if (snapshot.epoch != hairEpoch) { hairEpoch = snapshot.epoch; hairGate.reset() }
            }
        } catch (_: Exception) { if (!stale) failHair() }
        // 故障后不复位 hairGate：它按 hairHealthy 当帧归 0（见 HairMaskGate.update），本实例内不再恢复。
        if (!hairHealthy) {
            hairSegmenter?.close(); hairSegmenter = null; hairTicket = null; stagedHair = null
        }
        // 1'. 共享小图：四路本帧至多画一次、回读一次；SYNC 当帧读回交给 worker，DEFERRED 只画、记下采样时间与尺寸，
        // 下一帧开头 collectDeferred 再取票、读回。
        val staged = newFrame && faceSampler.deferred &&
            (stagedFace != null || stagedBody != null || stagedSeg != null || stagedHair != null)
        if (live && (faceTicket != null || bodyTicket != null || segTicket != null || hairTicket != null || staged)) {
            try {
                val faceCrop = if (faceTicket != null || stagedFace != null) faceRoi.select(width, height, now)
                else FaceRoiSelector.CropRect.FULL
                faceSampler.capture(previousTexId, width, height, renderHandler.fboId[0], faceCrop = faceCrop)
                if (faceSampler.deferred) {
                    stagedAt = now; stagedWidth = width; stagedHeight = height
                } else {
                    submit(faceTicket, bodyTicket, segTicket, hairTicket, now)
                }
            } catch (_: Exception) {
                // 采样故障：本帧的票作废，按各自的故障停用（与旧版各自采样失败时一致）。
                faceTicket?.let { tracker?.state?.complete(it, null, now) }
                bodyTicket?.let { bodyTracker?.state?.complete(it, null, now) }
                segTicket?.let { segmenter?.state?.complete(it, null, now) }
                hairTicket?.let { hairSegmenter?.state?.complete(it, null, now) }
                if (!stale) {
                    if (faceTicket != null || stagedFace != null) failFace()
                    if (bodyTicket != null || stagedBody != null) failBody()
                    if (segTicket != null || stagedSeg != null) failSegmentation()
                    if (hairTicket != null || stagedHair != null) failHair()
                }
                clearStaged()
            }
        }
        frameDelay.schedule.lock.onStall(faceSampler.takeStallNs())
        // 1d. 全链路时空对齐：人脸、微整、美妆、美体与背景虚化都按被显示帧的 displayTime 取样。FrameDelay 每场锁定：本会话
        // 首次需要跟踪时接入并一直维持（丢脸、对比原画、负载降级、代际变化都不清空，避免推流跳帧），关闭美颜且未推流满
        // ALIGN_FREE_MS 才结束。延后按人脸完成时间测量，开着背景虚化时与分割完成时间取较小值（美体靠 snapshotAt 对齐）。
        val tracking = faceTracking || bodyTracking || segTracking
        val trackCompleted = when {
            faceTracking && segTracking -> minOf(faceCompleted, segCompleted)
            faceTracking -> faceCompleted
            segTracking -> segCompleted
            else -> Long.MIN_VALUE
        }
        timeline.update(now)
        var displayed = previousTexId
        var displayTime = now
        if (live && !engaged && tracking && alignHealthy) {
            engaged = true
            frameDelay.engage(if (tracker?.gpu == false) PacingLock.PRIOR_CPU else PacingLock.PRIOR_GPU)
        }
        val onAir = try { streaming() } catch (_: Exception) { false }
        if (live && engaged) {
            if (!value.enabled && !onAir) {
                if (alignOffSince == Long.MIN_VALUE) alignOffSince = now
                else if (now - alignOffSince >= ALIGN_FREE_MS) {
                    // 关了美颜且没在推流：结束本会话的画面对齐，释放环形缓冲的显存。
                    try { frameDelay.free() } catch (_: Exception) {}
                    engaged = false
                    setDeferred(false)
                }
            } else {
                alignOffSince = Long.MIN_VALUE
            }
        }
        if (live && engaged && alignHealthy && stamp != 0L) {
            try {
                val texture = if (newFrame) frameDelay.push(previousTexId, width, height, now, stamp, trackCompleted, onAir)
                    else frameDelay.selected
                if (texture != 0) {
                    displayed = texture; displayTime = frameDelay.schedule.time; contentStampNs = frameDelay.stampNs
                }
                setDeferred(frameDelay.schedule.lock.deferredReadback)
            } catch (_: Exception) {
                if (!stale) alignHealthy = false
                displayed = previousTexId; displayTime = now; contentStampNs = stamp
                setDeferred(false)
            }
        }
        videoDelayMs = ((stamp - contentStampNs) / 1_000_000L).coerceIn(0L, MAX_VIDEO_DELAY_MS)
        pacing?.noteEffects(tier, frameDelay.schedule.delay, faceSampler.deferred)
        // 按最终显示的相机帧采样时间 displayTime 提取对应时刻的人体姿态与人像分割遮罩。
        var bodyTarget: FloatArray? = null
        var bodyTimestamp = 0L
        if (bodyHealthy && live) {
            bodyTracker?.state?.snapshotAt(displayTime, now)?.let { snap ->
                snap.geometry?.let { geom ->
                    if (geom !== lastBody) { lastBody = geom; lastBodyState = geom.toState() }
                    bodyTarget = lastBodyState
                    bodyTimestamp = snap.sampleTimestamp
                }
            }
        }
        var mask: PersonMask? = null
        var maskStamp = Long.MIN_VALUE
        if (segHealthy && live) {
            segmenter?.state?.snapshotAt(displayTime, now)?.let { snap ->
                mask = snap.mask
                maskStamp = snap.sampleTimestamp
            }
        }
        val maskShown = maskPresence.update(mask != null, now)
        // 遮罩与被显示帧的时间差超过 MASK_LAG_FULL_MS 起按比例衰减，到 MASK_LAG_ZERO_MS 为 0：太旧的遮罩宁可不虚化，不拖背景残影。
        val shownMaskStamp = if (mask != null) maskStamp else lastMaskStamp
        val maskLagMs = if (shownMaskStamp == Long.MIN_VALUE) Float.POSITIVE_INFINITY else abs(shownMaskStamp - displayTime).toFloat()
        val maskFresh = if (shownMaskStamp == Long.MIN_VALUE) 0f
            else 1f - smoothstep(MASK_LAG_FULL_MS.toFloat(), MASK_LAG_ZERO_MS.toFloat(), maskLagMs)
        val background = if (active && segHealthy) value.backgroundBlur * maskShown * maskFresh * fades.body else 0f
        // 头发遮罩：同样按 displayTime 取，存在度（渐入 300 / 保持 300 / 渐出 400 ms）× 新鲜度（lag 420–650 ms 渐出）× 负载渐出
        // × 「人脸近期可见」（HairMaskTuning.faceKeep：最近一次完成的人脸检测确实没有脸时才渐出，最新有脸样本之后 450 ms 内到 0，
        // 早于时间线 500 ms 清空几何；检测器只是还在算时不降，停摆兜底只在没有人脸任务在途（faceInFlight）时生效，推理慢时
        // 不随采样节奏锯齿；重新检出按 300 ms 渐入）。
        var hairMask: PersonMask? = null
        var hairMaskStamp = Long.MIN_VALUE
        if (hairHealthy && live) {
            hairSegmenter?.state?.snapshotAt(displayTime, now)?.let { snap ->
                hairMask = snap.mask
                hairMaskStamp = snap.sampleTimestamp
            }
        }
        val hairUpload = hairGate.update(active && hairHealthy, hairMask, hairMaskStamp, displayTime, fades.body,
            timeline.latest, timeline.missSince, faceInFlight, now)
        val body = bodySmoother.update(bodyTarget, bodyTimestamp, now)
        val bodyPresence = if (body != null && active) bodySmoother.presence * fades.body else 0f
        // 强度上限：细腰侧向取样最多外扩 18%，丰胸放大同大眼量级，长腿最多拉长 25%。
        val waist = value.waistSlim * 0.18f * bodyPresence
        val bust = value.bustEnhance * 0.25f * bodyPresence
        val legs = value.legLengthen * 0.25f * bodyPresence * (body?.get(6) ?: 0f)
        // 瘦肩侧向取样最多外扩 15%；天鹅颈把肩上 0.3 躯干长的一段最多拉长 30%。
        val shoulders = value.shoulderSlim * 0.15f * bodyPresence
        val neck = value.neckLengthen * 0.3f * bodyPresence
        val bodyOn = body != null && (waist > 0f || bust > 0f || legs > 0f || shoulders > 0f || neck > 0f)
        // 开了长腿、识别到身体却没拍到膝盖（坐着、半身）：持续 1 秒后提示，见 BodyLegsMonitor。
        val fresh = bodyTarget
        legsMonitor.update(active && fades.body > 0f && bodyHealthy && bodyPassHealthy && value.legLengthen > 0f, fresh != null,
            (fresh?.get(6) ?: 0f) > 0.5f, now)?.let { hidden ->
            try { onLegsHidden(hidden) } catch (_: Exception) {}
        }
        val aspect = width.toFloat() / height.coerceAtLeast(1)
        val rawGeometry = timeline.at(displayTime)
        val mesh = timeline.meshAt(displayTime)
        val presence = if (rawGeometry != null && active) timeline.presence * fades.face else 0f
        val warpWeight = rawGeometry?.get(FaceGeometrySmoother.WARP_INDEX) ?: 0f
        val steadyEye = timeline.steadyEye
        val steadyFace = timeline.steadyFace
        val meshReshapeParams = FaceMeshDeformer.Params(
            faceSlim = value.faceSlim,
            vFace = value.vFace,
            faceNarrow = value.faceNarrow,
            faceSmall = value.faceSmall,
            cheekboneSlim = value.cheekboneSlim,
            chinShape = value.chinShape,
            foreheadShape = value.foreheadShape,
            cranialTop = value.cranialTop,
            eyeEnlarge = value.eyeEnlarge,
            eyeDistance = value.eyeDistance,
            eyeCorner = value.eyeCorner,
            noseSlim = value.noseSlim,
            noseLength = value.noseLength,
            mouthSize = value.mouthSize,
            mouthSmile = value.mouthSmile,
        )
        // 时间线开始新的一轮（首个样本、换人、丢脸后重新检出）：形变器的可见度 / 松弛时间平滑从头开始。
        if (timeline.runId != deformerRun) { deformerRun = timeline.runId; faceMeshDeformer.resetTemporal() }
        // 网格路径与回退路径的份额（MeshFallbackFade）：网格通道故障时回退路径当帧满强度接上，回到网格路径时交叉渐变。
        // 目标只看网格是否可用（不看 active）：按住对比原画时 presence 为 0、份额直接取目标，松手后照常是网格路径，不重新渐变。
        val meshShare = meshFade.update(mesh != null && faceMeshHealthy, presence > 0f, now)
        val fallbackShare = 1f - meshShare
        val meshDeformed = mesh != null && active && faceMeshHealthy && presence > 0f && meshShare > 0f &&
            meshReshapeParams.isActive() && faceMeshDeformer.deform(
            landmarks = mesh.landmarks,
            aspect = aspect,
            axisX = mesh.axisX,
            axisY = mesh.axisY,
            faceWidth = mesh.faceWidth,
            faceHeight = mesh.faceHeight,
            leftWeight = mesh.leftWeight,
            rightWeight = mesh.rightWeight,
            yawWeight = mesh.yawWeight,
            presence = presence * meshShare,
            steadyFace = steadyFace,
            steadyEye = steadyEye,
            params = meshReshapeParams,
            nowMs = now,
        )
        // 回退路径（实际触发是网格通道故障 faceMeshHealthy=false；网格被拒收的大侧脸帧没有几何，按丢脸退出）只有片元大眼
        // （uTone.w）与 FaceShapePass 的下巴/额头/瘦鼻：FRAGMENT_PRO 没有 uCheeks，回退路径没有瘦脸（legacySlim 照旧写进
        // uTone.z，合成不读）。两条路径的强度分别乘 meshShare 与 fallbackShare：网格不可用时 meshShare 当帧为 0，
        // 只有回到网格路径的渐变途中两者同时生效。
        val legacySlim = value.faceSlim * 0.12f * presence * fallbackShare * warpWeight * steadyFace
        val legacyEye = value.eyeEnlarge * 0.25f * presence * fallbackShare * warpWeight * steadyEye
        val shapeWeight = presence * fallbackShare * warpWeight
        val workingLandmarksInitial = when {
            meshDeformed -> faceMeshDeformer.warpedLandmarks
            mesh != null -> mesh.landmarks
            else -> null
        }
        val retouchAny = value.darkCircles > 0f || value.brightEyes > 0f || value.teethWhiten > 0f || value.nasolabial > 0f
        val retouchMeshOn = active && faceRetouchHealthy && faceMeshHealthy && mesh != null && presence > 0f && retouchAny
        val faceShapeOn = active && faceShapeHealthy && presence > 0f && !retouchMeshOn && if (mesh != null && workingLandmarksInitial != null && faceMeshHealthy) {
            FaceShapePass.packRetouchFromMesh(
                landmarks = workingLandmarksInitial,
                axisX = mesh.axisX,
                axisY = mesh.axisY,
                aspect = aspect,
                darkCircles = value.darkCircles * presence * steadyEye,
                brightEyes = value.brightEyes * presence * steadyEye,
                teeth = value.teethWhiten * presence * steadyEye,
                folds = value.nasolabial * presence * steadyEye,
                out = faceShapeValues,
            )
        } else if (rawGeometry != null) {
            FaceShapePass.pack(
                rawGeometry, aspect,
                value.chinShape.coerceAtLeast(0f) * shapeWeight * steadyFace,
                value.foreheadShape.coerceAtLeast(0f) * shapeWeight * steadyFace,
                value.noseSlim * shapeWeight * steadyEye, faceShapeValues,
                value.darkCircles * presence * steadyEye, value.brightEyes * presence * steadyEye,
                value.teethWhiten * presence * steadyEye, value.nasolabial * presence * steadyEye,
            )
        } else false
        val makeupWanted = active && makeupHealthy && mesh != null && presence > 0f && value.hasMakeup()
        val lutWanted = value.lutUrl.isNotEmpty() && value.lutIntensity > 0f
        val processing = active && (bodyOn || lutWanted || makeupWanted || background > 0f || meshDeformed || retouchMeshOn || faceShapeOn ||
            legacySlim > 0f || legacyEye > 0f || value.smoothing > 0f || value.brightness > 0f ||
            value.sharpness > 0f || value.whiten > 0f || value.rosy > 0f || value.radiance > 0f || value.saturation != 1f ||
            value.filter != "none" && value.filterIntensity > 0f)
        // 头发遮罩的 GL 输入（A 通道与背景合并道共用）：换了遮罩才上传；再求「显示帧 → 遮罩时刻」的运动补偿仿射
        // （HairMaskGate.compensate）：显示时刻取本帧 at(displayTime) 的几何（rawGeometry，越过最新样本时是 at() 的外推，与
        // A 通道、合并道的脸椭圆同一个几何），遮罩时刻取已上传遮罩的采样时间（越过最新样本时沿同一条外推轨迹）；平移或缩放超限时
        // 退回单位阵。
        if (hairUpload != null && processing && healthy && (semanticHealthy || bgHairHealthy)) {
            try { hairTexture.upload(hairUpload); hairGate.uploaded(hairMaskStamp) }
            catch (_: Exception) { if (!stale) { semanticHealthy = false; bgHairHealthy = false } }
        }
        val hairLevel = if (processing && healthy && hairTexture.texture != 0 && hairGate.stamp != Long.MIN_VALUE)
            hairGate.level else 0f
        if (hairLevel > 0f) {
            HairMaskGate.compensate(timeline, hairGate.stamp, rawGeometry, displayTime, aspect, hairPoseMask, hairPoseDisplay,
                hairAffine)
        }
        // 2a. 时域降噪（开着磨皮且画面偏暗，或码率被压低）：先对要画的那一帧降噪，之后各道都读它；故障只停用降噪。
        var source = displayed
        // 码率被压低时同样降噪（噪点最耗码率），不要求开着磨皮。
        val pressure = BitrateEffectsTuning.pressure(try { bitrateRatio() } catch (_: Exception) { 1f })
        // 没有运动补偿：人脸在移动（多半是镜头在动）时暂停，免得把上一帧的五官混进来。
        val denoise = if (processing && healthy && denoiseHealthy) maxOf(
            if (value.smoothing > 0f) TemporalDenoisePass.strengthForLuma(lightLuma) else 0f,
            BitrateEffectsTuning.denoise(pressure)) * timeline.denoiseKeep * fades.smooth else 0f
        if (denoise > 0f) {
            try { source = denoisePass.render(displayed, width, height, denoise, now) }
            catch (_: Exception) { if (!stale) denoiseHealthy = false; source = displayed }
        } else {
            denoisePass.interrupt()
        }
        // 2a'. 背景虚化：在降噪后的原帧上按人像遮罩虚化背景（遮罩按原帧算，须在美体变形之前）。
        if (processing && healthy && bgPassHealthy && background > 0f) {
            try {
                val latest = mask
                if (latest != null && maskStamp != lastMaskStamp) { bgPass.updateMask(latest); lastMaskStamp = maskStamp }
                val face = if (rawGeometry != null) packHairOval(rawGeometry, aspect, hairBackgroundOval) else null
                val hair = if (hairLevel > 0f && bgHairHealthy && face != null) hairBackground.also {
                    it.texture = hairTexture.texture; it.shown = hairLevel
                } else null
                if (bgPass.hasMask) source = bgPass.render(source, width, height, background, hair, face, hairGate.faceKeep)
            } catch (_: Exception) { if (!stale) bgPassHealthy = false }
        }
        // 2a''. 478 点 2D 三角网格精细微整（15 项微整，双环 C1 柔化缓冲环；在局部修饰与美体之前执行）。
        val preMeshSource = source
        var meshRendered = false
        if (processing && healthy && meshDeformed) {
            try {
                source = faceMeshPass.render(preMeshSource, width, height, faceMeshDeformer.vertexData, faceMeshDeformer.frontFaceCw)
                meshRendered = true
            } catch (_: Exception) {
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                if (!stale) faceMeshHealthy = false
            }
        }
        // 网格微整完成后，主合成遮罩状态与 478 点坐标同步切到变形后坐标，避免磨皮遮罩或后续美妆错位。
        val geometry = if (meshRendered && mesh != null &&
            FaceMeshFrame.fillGeometryState(faceMeshDeformer.warpedLandmarks, mesh.axisX, mesh.axisY, aspect, 0f, warpedGeometry)
        ) warpedGeometry else rawGeometry
        var workingLandmarks = if (meshRendered) faceMeshDeformer.warpedLandmarks else mesh?.landmarks
        // 2a'''. 468 点解剖子网格局部修饰（去黑眼圈留卧蚕、弧形法令纹、眼裂亮眼、内唇防染唇白牙）：直接原位叠加在 faceMeshPass FBO 上，
        // 省去整张 1080p FBO；异常或无网格时自动回退 FaceShapePass。
        var retouchApplied = false
        if (processing && healthy && retouchMeshOn) {
            val meshFrame = checkNotNull(mesh)
            try {
                if (!meshRendered) {
                    source = faceMeshPass.renderBaseOnly(preMeshSource, width, height)
                }
                val retouchLandmarks = workingLandmarks ?: meshFrame.landmarks
                faceRetouchPass.renderInPlace(
                    targetFbo = faceMeshPass.outputFbo,
                    sourceTexture = preMeshSource,
                    width = width,
                    height = height,
                    landmarks = retouchLandmarks,
                    meshVertices = if (meshRendered) faceMeshDeformer.vertexData else null,
                    axisX = meshFrame.axisX,
                    axisY = meshFrame.axisY,
                    faceWidth = meshFrame.faceWidth,
                    faceHeight = meshFrame.faceHeight,
                    frontFaceCw = if (meshRendered) faceMeshDeformer.frontFaceCw else FaceMeshFrame.isFrontFaceCw(retouchLandmarks),
                    darkCircles = value.darkCircles * presence * steadyEye,
                    brightEyes = value.brightEyes * presence * steadyEye,
                    teethWhiten = value.teethWhiten * presence * steadyEye,
                    nasolabial = value.nasolabial * presence * steadyEye,
                )
                retouchApplied = true
            } catch (_: Exception) {
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                glDisable(GL_BLEND)
                if (!stale) faceRetouchHealthy = false
                if (!meshRendered) {
                    source = preMeshSource
                } else {
                    try {
                        source = faceMeshPass.render(preMeshSource, width, height, faceMeshDeformer.vertexData, faceMeshDeformer.frontFaceCw)
                    } catch (_: Exception) {
                        source = preMeshSource
                    }
                }
            }
        }
        if (!retouchApplied && processing && healthy && faceShapeHealthy && presence > 0f && (faceShapeOn || retouchAny)) {
            val ready = faceShapeOn || (mesh != null && workingLandmarksInitial != null && FaceShapePass.packRetouchFromMesh(
                landmarks = workingLandmarksInitial,
                axisX = mesh.axisX,
                axisY = mesh.axisY,
                aspect = aspect,
                darkCircles = value.darkCircles * presence * steadyEye,
                brightEyes = value.brightEyes * presence * steadyEye,
                teeth = value.teethWhiten * presence * steadyEye,
                folds = value.nasolabial * presence * steadyEye,
                out = faceShapeValues,
            ))
            if (ready) {
                try { source = faceShapePass.render(source, width, height, faceShapeValues) }
                catch (_: Exception) { if (!stale) faceShapeHealthy = false }
            }
        }
        // 2b. 美体：单独一道变形画进中间纹理，磨皮预处理与合成都读它；故障只停用美体，本帧沿用上一步的结果。
        var neckLifted = false
        if (processing && healthy && bodyPassHealthy && bodyOn) {
            try {
                // 方向向量的长度携带长腿强度（1 + legs），见 BODY_FRAGMENT。
                val scale = 1f + legs
                bodyValues[0] = body[0]; bodyValues[1] = body[1]
                bodyValues[2] = body[2] * scale; bodyValues[3] = body[3] * scale
                bodyValues[4] = body[4]; bodyValues[5] = body[5]
                bodyValues[6] = waist; bodyValues[7] = bust
                bodyValues[8] = shoulders; bodyValues[9] = neck; bodyValues[10] = 0f; bodyValues[11] = 0f
                source = bodyPass.render(source, width, height, bodyValues)
                neckLifted = neck > 0f
            } catch (_: Exception) { if (!stale) bodyPassHealthy = false }
        }
        var finalLandmarks = workingLandmarks
        if (neckLifted && body != null && finalLandmarks != null && mesh != null) {
            finalLandmarks = NeckLift.applyLandmarks(finalLandmarks, mesh.centerX, mesh.centerY, body, neck, aspect, liftedLandmarks)
        }
        // 2b'. 468 点多边形光栅化精准面部皮肤遮罩（含额头冠延伸、挖空眼眉唇口腔、3D 骨相高光平面），异常时自动回退椭圆遮罩。
        var skinMaskTexture = 0
        var polyMask = false // 本帧多边形遮罩渲染成功：只有它决定 uFrame.w 取 presence + 4（见 SkinMaskPass.framePresence）。
        if (processing && healthy && skinMaskHealthy && faceMeshHealthy && presence > 0f && mesh != null && finalLandmarks != null) {
            try {
                skinMaskTexture = skinMaskPass.render(finalLandmarks, width, height, mesh.axisX, mesh.axisY)
                polyMask = skinMaskTexture != 0
            } catch (_: Exception) {
                glDisable(GL_CULL_FACE)
                glFrontFace(GL_CCW)
                if (!stale) skinMaskHealthy = false
                skinMaskTexture = 0
            }
        }
        // 2b''. 头发 A 通道（毛发边界第二批）：在皮肤遮罩输出上只写 A = 1 − 强度 · smoothstep(阈值, 头发)，主合成读成非头发权重，
        // 头发处不磨皮、不调色，只留背景级锐化。没有多边形遮罩时先清出一张只承载 A 的空遮罩（RGB 为 0，主合成仍走椭圆回退）；
        // 做了网格微整时 A 跟着网格变形走。天鹅颈平移了头部而 A 通道不跟这段形变：强度乘 1 − 天鹅颈生效程度（随人体存在度渐变，
        // 人体稳定时为 0、不画；出现、消失时按比例过渡，不单帧硬切）。外部引擎在第 0 段已经返回。
        // 本帧 A 道画成功才把这张皮肤遮罩交给导向滤波做头发加权（hairWeight）；没画或抛异常时为 0，导向滤波走原路径。
        val hairShown = if (hairLevel > 0f) hairGate.shown(HairMaskGate.neckWeight(neckLifted, bodyPresence), engine = false) else 0f
        var hairWeight = 0
        if (hairShown > 0f && semanticHealthy && geometry != null) {
            try {
                if (skinMaskTexture == 0) skinMaskTexture = skinMaskPass.renderEmpty(width, height)
                val meshInput = if (meshRendered) hairMesh.also {
                    it.vertices = faceMeshDeformer.vertexData; it.frontFaceCw = faceMeshDeformer.frontFaceCw
                } else null
                semanticPass.render(skinMaskPass, hairTexture.texture, hairShown, hairAffine,
                    packHairOval(geometry, aspect, hairOval), meshInput)
                hairWeight = skinMaskTexture
            } catch (_: Exception) {
                // SemanticAlphaPass 已尽力把 A 清回 1；本上下文内不再画 A 通道，回到第一批。
                if (!stale) semanticHealthy = false
            }
        }
        // 2. 磨皮预处理：优先走 1/3 分辨率快速导向滤波（GuidedFilterPrePass，全分辨率线性保边重构），故障时自动回退旧版 BeautyPrePass。
        var blurTexture = 0
        var prepassMode = 0f // 1 = 绑的是 BeautyPrePass 的 (mean, var)，合成按 uGrade.z 换算成 (b, a)。
        lastProcessing = processing
        // 负载 3 级渐出磨皮：强度与合成参数用衰减后的 smoothing；导向滤波的 eps 仍按未衰减的 value.smoothing，渐出中滤波形态不变。
        val smoothing = value.smoothing * fades.smooth
        if (processing && value.smoothing > 0f && fades.smooth > 0f) {
            // 采样间距随脸宽自适应（EMA + 迟滞），没有网格（丢脸）时保持原值。
            mesh?.let { guidedSpread.update(it.faceWidth * width) }
            if (guidedFilterHealthy) {
                try { blurTexture = guidedFilterPass.render(source, width, height, value.smoothing, guidedSpread.value, hairWeight) }
                catch (_: Exception) { if (!stale) guidedFilterHealthy = false; blurTexture = 0 }
            }
            if (blurTexture == 0) {
                try { blurTexture = prePass.render(source, width, height); prepassMode = 1f }
                catch (_: Exception) { if (!stale) failEffects() }
            }
        }
        // 2c. LUT：解码好的 LUT 上传，查表在主合成内（纹理单元 3）；实时美妆由 draw() 在合成之后收尾。
        val lutUrl = value.lutUrl
        // 下载或解码失败的 LUT 过一段时间再试（网络抖动不应让它在本次预览内永久失效）。
        if (live && lutUrl.isNotEmpty() && lutUrl == lutFailed && now >= lutRetryAt) {
            lutFailed = ""
            lutLoader.restart { config.let { if (it.enabled) it.lutUrl else "" } }
        }
        if (live && lutHealthy && lutUrl.isNotEmpty() && lutUrl != lutPass.loadedUrl && lutUrl != lutFailed) {
            lutLoader.take(lutUrl)?.let { result ->
                val bitmap = result.bitmap
                if (bitmap == null) { lutFailed = lutUrl; lutRetryAt = now + LUT_RETRY_MS }
                else try { lutPass.upload(lutUrl, bitmap) }
                catch (_: Exception) { if (!stale) lutHealthy = false }
                finally { bitmap.recycle() }
            }
        }
        var output = renderHandler.fboId[0]
        makeupPending = false
        // 新 LUT 上传完成前继续绑上一帧那张表，不出现「完全不调色」的帧；失败的地址不再查表，回落旧常量滤镜。
        // select 每帧调用一次（不查表的帧也传空串），否则分不清上一帧是否查过表。
        val lutTexture = if (live) lutPass.select(
            if (processing && healthy && lutHealthy && lutWanted && lutUrl != lutFailed) lutUrl else "") else 0
        val lutActive = lutTexture != 0
        if (processing && healthy && makeupWanted && finalLandmarks != null) {
            try {
                output = makeupPass.target(width, height)
                finalLandmarks.copyInto(makeupLandmarks, 0, 0, FaceMeshFrame.COORD_COUNT)
                makeupConfig = value
                makeupPresence = presence
                makeupSteadyFace = steadyFace
                makeupSteadyEye = steadyEye
                makeupEyeKeep = timeline.eyeMakeupKeep
                makeupRightIrisGate = mesh.rightIrisGate
                makeupLeftIrisGate = mesh.leftIrisGate
                makeupPending = true
            } catch (_: Exception) {
                if (!stale) makeupHealthy = false
                output = renderHandler.fboId[0]
            }
        }
        // 3. 恢复 draw() 设好的 FBO（或美妆中间纹理）与 viewport，二者都已知，无需 glGet。
        glBindFramebuffer(GL_FRAMEBUFFER, output)
        glViewport(0, 0, width, height)
        // 两个程序在 initGlFilter 都没建成（例如当时上下文状态异常）：本上下文内在 GL 线程重试一次原画程序。
        if (program == 0 && live && !originalRetried) {
            originalRetried = true
            original = build(VideoEffectsShaders.ORIGINAL)
            if (original != 0) use(original)
        }
        // 回原画程序；原画程序没建成时保留合成程序，以 processing = 0 直通。
        if (!healthy && original != 0 && program != original) use(original)
        val on = processing && healthy
        // 4. 合成：只设状态，glDrawArrays 由基类发出。先绑 unit 3、2、1 再绑 unit 0，最后 GL_TEXTURE0 处于激活。
        glUseProgram(program)
        if (position >= 0) {
            vertices.position(0)
            glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertices)
            glEnableVertexAttribArray(position)
        }
        if (texture >= 0) {
            vertices.position(2)
            glVertexAttribPointer(texture, 2, GL_FLOAT, false, 16, vertices)
            glEnableVertexAttribArray(texture)
        }
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, if (lutActive) lutTexture else fallbackTexture)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, if (skinMaskTexture != 0) skinMaskTexture else fallbackTexture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, if (blurTexture != 0) blurTexture else fallbackTexture)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source)
        // 没有可用程序时 glUniform* 会报 GL_INVALID_OPERATION；本帧输出未定义，但不污染错误队列。
        if (program == 0) return
        if (lutSampler >= 0) glUniform1i(lutSampler, 3)
        if (skinMask >= 0) glUniform1i(skinMask, 2)
        glUniform1i(blur, 1)
        glUniform1i(sampler, 0)
        if (on && presence > 0f && geometry != null) {
            // 天鹅颈把头部上移了：合成用的人脸几何跟着平移（门禁按变量名 geometry 解析打包，下面同名覆盖）。
            val lifted = if (neckLifted && body != null)
                NeckLift.apply(geometry, body, neck, width.toFloat() / height.coerceAtLeast(1), liftedFace) else geometry
            // 只能外推时：脸、眼、唇椭圆盖住保持位置到外推位置的整段，五官不会滑出保护区被磨皮抹开。
            @Suppress("NAME_SHADOWING")
            val geometry = timeline.dilate(lifted, alignedFace)
            glUniform4fv(cheeks, 4, geometry, 0)
            for (i in 0 until 2) {
                geometry.copyInto(eyeValues, i * 4, 16 + i * 3, 19 + i * 3)
                eyeValues[i * 4 + 3] = geometry[FaceGeometrySmoother.AXIS_INDEX + i] // uEyes[0].w = axis.x，uEyes[1].w = axis.y
            }
            glUniform4fv(eyes, 2, eyeValues, 0)
            glUniform4fv(oval, 1, geometry, 22)
            glUniform4fv(mouth, 1, geometry, 28)
            glUniform4fv(eyeMask, 2, geometry, 32)
        }
        val style = when (value.filter) {
            "warm" -> 1f; "cool" -> 2f; "monochrome" -> 3f
            "sepia" -> 4f; "rose" -> 5f; else -> 0f
        }
        // 本帧有效启用 3D LUT 时自动屏蔽旧常量 RGB 滤镜，避免双重色偏叠加；LUT 未就绪/失败时平滑回退旧滤镜。
        val activeStyle = if (lutActive) 0f else style
        val activeFilterIntensity = if (lutActive) 0f else value.filterIntensity
        val flags = if (!on) -1f else activeStyle + if (value.skinOnly) 16f else 0f
        // 按「本帧多边形遮罩渲染成功」取 presence + 4，不按纹理非 0：只承载头发 A 的空遮罩不能让椭圆回退路径读到 face = 0。
        val framePresence = SkinMaskPass.framePresence(on, polyMask, presence)
        glUniform4f(frame, 1f / width.coerceAtLeast(1), 1f / height.coerceAtLeast(1),
            width.toFloat() / height.coerceAtLeast(1), framePresence)
        glUniform4f(params, flags, if (blurTexture != 0) smoothing else 0f,
            value.brightness, BitrateEffectsTuning.texture(value.skinTexture, pressure))
        glUniform4f(details, value.radiance,
            BitrateEffectsTuning.sharpness(value.sharpness, pressure), value.saturation, activeFilterIntensity)
        glUniform4f(tone, value.whiten, value.rosy, if (on) legacySlim else 0f, if (on) legacyEye else 0f)
        // uGrade = (LUT 强度, 质感/锐化 coring 门限, 回退预处理标记, 未用)。用 glUniform4f：门禁的几何打包解析器要求每个 4fv 都是几何数组。
        glUniform4f(grade, if (lutActive) value.lutIntensity.coerceIn(0f, 1f) else 0f,
            BeautyQualityTuning.coreThreshold(lightLuma, pressure), prepassMode, 0f)
    }
    override fun disableResources() {
        if (position >= 0) glDisableVertexAttribArray(position)
        if (texture >= 0) glDisableVertexAttribArray(texture)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
    }
    /**
     * 两种调用方：
     * - RenderPipeline 的 SET/REMOVE/CLEAR：在 GL 线程、本滤镜的上下文仍是当前上下文，没有并发 → 同步删除 GL 对象并重置。
     * - GLHost.stop()：调用线程（通常是主线程）在拆掉 EGL 之后调用，而 shutdownNow() 不等待，GL 线程可能仍在
     *   本帧 drawFilter 里。此时不碰 GL 线程的状态（预处理、采样器、平滑器），只置 released 并关闭 tracker
     *   （submit 容忍已关闭）；旧 id 由下一次 initGlFilter 在新上下文里忘记，GL 对象随旧上下文销毁。
     */
    override fun release() {
        attachment.set(false)
        val current = EGL14.eglGetCurrentContext()
        released = true
        if (current != EGL14.EGL_NO_CONTEXT && current == glContext) {
            tracker?.close(); tracker = null
            bodyTracker?.close(); bodyTracker = null
            segmenter?.close(); segmenter = null
            hairSegmenter?.close(); hairSegmenter = null
            clearStaged() // 还没取票的延迟回读记录随上下文丢弃
            bgPass.delete(); faceShapePass.delete(); faceRetouchPass.delete(); faceMeshPass.delete()
            hairTexture.delete(); semanticPass.delete()
            guidedFilterPass.delete(); skinMaskPass.delete()
            prePass.delete(); faceSampler.delete(); bodyPass.delete()
            denoisePass.delete(); lutPass.delete(); makeupPass.delete(); frameDelay.delete(); engaged = false
            engine?.let { try { it.onGlReleased(true) } catch (_: Exception) {} }
            engine = null; engineSync.reset(); engineStamps.reset()
            if (fallbackTexture != 0) glDeleteTextures(1, intArrayOf(fallbackTexture), 0)
            if (composite != 0) glDeleteProgram(composite)
            if (original != 0) glDeleteProgram(original)
            fallbackTexture = 0; composite = 0; original = 0; use(0)
            resetFaceState(); resetBodyState()
            reuse.invalidate()
            glContext = null
            return
        }
        tracker?.close()
        bodyTracker?.close()
        segmenter?.close()
        hairSegmenter?.close()
    }

    /** 本帧取到的票交给各自的 worker（共享小图已读回暂存区）；提交失败的票作废并按各自的故障停用。 */
    private fun submit(
        faceTicket: FaceTrackingState.Ticket?, bodyTicket: BodyTrackingState.Ticket?, segTicket: MaskTrackingState.Ticket?,
        hairTicket: MaskTrackingState.Ticket?, now: Long,
    ) {
        if (faceTicket != null) tracker?.let { detector ->
            try { detector.submit(faceSampler.lease(FaceTextureSampler.FACE), faceTicket) }
            catch (_: Exception) { detector.state.complete(faceTicket, null, now); if (!stale) failFace() }
        }
        if (bodyTicket != null) bodyTracker?.let { detector ->
            try { detector.submit(faceSampler.lease(FaceTextureSampler.BODY), bodyTicket) }
            catch (_: Exception) { detector.state.complete(bodyTicket, null, now); if (!stale) failBody() }
        }
        if (segTicket != null) segmenter?.let { detector ->
            try { detector.submit(faceSampler.lease(FaceTextureSampler.SEGMENT), segTicket) }
            catch (_: Exception) { detector.state.complete(segTicket, null, now); if (!stale) failSegmentation() }
        }
        if (hairTicket != null) hairSegmenter?.let { detector ->
            try {
                val sample = faceSampler.lease(FaceTextureSampler.HAIR)
                detector.submit(sample, hairTicket, hairAnchor(sample.width, sample.height))
            } catch (_: Exception) { detector.state.complete(hairTicket, null, now); if (!stale) failHair() }
        }
    }

    /**
     * 头发 EMA 运动门的人脸锚点 (cx, cy, 半宽)，单位是遮罩像素（行 0 在画面顶部）：取最近一次人脸结果的脸椭圆；那次结果没有脸
     * （或还没有结果）时返回 null，HairSegmenter 不做 EMA。返回的数组复用，HairSegmenter.submit 会拷贝一份。
     */
    private fun hairAnchor(width: Int, height: Int): FloatArray? {
        val face = lastGeometry ?: return null
        if (lastGeometryAt != lastCompletedSeen) return null
        hairAnchor[0] = face.oval[0] * width
        hairAnchor[1] = (1f - face.oval[1]) * height
        hairAnchor[2] = face.oval[2] * width
        return hairAnchor
    }

    /** 几何状态里的脸椭圆（uOval 的 22..25）与轴向（AXIS_INDEX 起 2 个）加帧宽高比 → SemanticAlphaPass.OVAL_SIZE 个数。 */
    private fun packHairOval(state: FloatArray, aspect: Float, out: FloatArray): FloatArray {
        state.copyInto(out, 0, 22, 26)
        out[4] = state[FaceGeometrySmoother.AXIS_INDEX]
        out[5] = state[FaceGeometrySmoother.AXIS_INDEX + 1]
        out[6] = aspect
        out[7] = 0f
        return out
    }

    /**
     * DEFERRED：为上一帧画进共享小图的采样取票。票的采样时间与尺寸仍是上一帧的，代际须与画小图时一致（同一把锁内判断，
     * 期间切镜头、reset 就不取）；上一次推理还在途也取不到。取到至少一张才读回并交给各自的 worker，一张都没有就丢弃小图、
     * 不读回。读回失败时已取的票作废（不算故障，下一帧照常采）。
     */
    private fun collectDeferred(now: Long) {
        val at = stagedAt; val w = stagedWidth; val h = stagedHeight; val faceInterval = stagedFaceInterval
        val faceEpoch = stagedFace; val bodyEpoch = stagedBody; val segEpoch = stagedSeg
        val hairEpoch = stagedHair; val hairInterval = stagedHairInterval
        clearStaged()
        val face = faceEpoch?.let { tracker?.state?.begin(at, w, h, faceInterval, expectEpoch = it) }
        val body = bodyEpoch?.let { bodyTracker?.state?.begin(at, w, h, BODY_INTERVAL_MS, expectEpoch = it) }
        val seg = segEpoch?.let { segmenter?.state?.begin(at, w, h, SEG_INTERVAL_MS, expectEpoch = it) }
        val hair = hairEpoch?.let { hairSegmenter?.state?.begin(at, w, h, hairInterval, expectEpoch = it) }
        if (face == null && body == null && seg == null && hair == null) { faceSampler.discardPending(); return }
        val ok = try { faceSampler.collect(renderHandler.fboId[0]) } catch (_: Exception) { false }
        if (ok) { submit(face, body, seg, hair, now); return }
        face?.let { tracker?.state?.complete(it, null, now) }
        body?.let { bodyTracker?.state?.complete(it, null, now) }
        seg?.let { segmenter?.state?.complete(it, null, now) }
        hair?.let { hairSegmenter?.state?.complete(it, null, now) }
    }

    /** 丢掉还没取票的延迟回读记录（记录不占单任务在途，无票可作废）。 */
    private fun clearStaged() {
        stagedAt = Long.MIN_VALUE; stagedFace = null; stagedBody = null; stagedSeg = null; stagedHair = null
    }

    /** 回读模式只在锁定时切换（每场一次）：切换时丢掉还没取票的记录，小图随 deferred 切换一起丢弃。 */
    private fun setDeferred(value: Boolean) {
        if (faceSampler.deferred == value) return
        clearStaged()
        faceSampler.deferred = value
    }

    /**
     * 外部引擎路径：自研跟踪暂停（不关闭）、不推进画面延后、不做美妆与 LUT；负载 3 级时整体旁路。
     * 只设好原画程序、属性与输入纹理，glDrawArrays 由基类发出；configure、process 抛异常只停用引擎，本帧直通。
     * 参数在这里按快照引用对账（[value] 为本帧快照）；引擎声明 latencyFrames = n 时显示的是 n 帧之前的相机帧，
     * 发布给 FramePacing 的内容时间与 videoDelayMs 都按那一帧算（EngineStamps）。
     */
    private fun drawEngine(external: BeautyEngine, value: VideoEffectsConfig, active: Boolean, tier: Int, stamp: Long, newFrame: Boolean) {
        tracker?.state?.enable(false); bodyTracker?.state?.enable(false); segmenter?.state?.enable(false)
        hairSegmenter?.state?.enable(false) // 外部引擎不读皮肤遮罩，不跑头发 A 通道与背景合并道。
        setDeferred(false); clearStaged()
        makeupPending = false
        lastProcessing = active
        val configured = try { engineSync.sync(external, value); true } catch (_: Exception) { engineHealthy = false; false }
        if (newFrame && stamp != 0L) engineStamps.push(stamp)
        var input = previousTexId
        var latency = 0
        if (configured && active && tier < EffectsLoadGovernor.MAX_LEVEL) {
            try {
                input = external.process(previousTexId, width, height, stamp)
                latency = external.latencyFrames.coerceIn(0, EngineStamps.MAX_LATENCY_FRAMES)
            } catch (_: Exception) { engineHealthy = false; input = previousTexId; latency = 0 }
        }
        // 旁路（未处理、3 级、故障）时显示的就是本帧，内容时间即 stamp。
        contentStampNs = if (stamp == 0L) 0L else engineStamps.content(latency)
        videoDelayMs = ((stamp - contentStampNs) / 1_000_000L).coerceIn(0L, MAX_VIDEO_DELAY_MS)
        pacing?.noteEffects(tier, latency, false)
        glBindFramebuffer(GL_FRAMEBUFFER, renderHandler.fboId[0])
        glViewport(0, 0, width, height)
        if (program != original) use(original)
        glUseProgram(program)
        if (position >= 0) {
            vertices.position(0)
            glVertexAttribPointer(position, 2, GL_FLOAT, false, 16, vertices)
            glEnableVertexAttribArray(position)
        }
        if (texture >= 0) {
            vertices.position(2)
            glVertexAttribPointer(texture, 2, GL_FLOAT, false, 16, vertices)
            glEnableVertexAttribArray(texture)
        }
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input)
        glUniform1i(sampler, 0)
    }

    private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }


    /**
     * 推流器/RTC 无关的便捷纹理处理入口（GL 线程调用）：
     * 输入竖屏正立 RGBA8 `GL_TEXTURE_2D` [inputTexture]（尺寸 [width]×[height]，相机采样时间戳 [timestampNs]），
     * 执行完整 v4.3 美颜/478 点网格微整/导向滤波磨皮/多边形皮肤遮罩/头发保护/实时美妆/3D LUT/美体/背景虚化，
     * 返回输出 `GL_TEXTURE_2D` ID（未启用或直通故障时直接返回已渲染原画纹理）。
     */
    fun processTexture(inputTexture: Int, width: Int, height: Int, timestampNs: Long = 0L): Int {
        if (inputTexture == 0 || width <= 0 || height <= 0) return inputTexture
        prepareRenderTarget(inputTexture, width, height)
        val prevStampFn = currentDirectStamp
        currentDirectStamp = timestampNs
        try {
            draw()
        } finally {
            currentDirectStamp = prevStampFn
        }
        return outputTextureId.let { if (it != 0) it else inputTexture }
    }
    private var currentDirectStamp: Long = Long.MIN_VALUE

    internal companion object {
        /** 下载失败的 LUT 重试间隔。 */
        const val LUT_RETRY_MS = 30_000L
        /** 时间戳未知时，同一相机帧的预览与编码两次绘制间隔远小于一帧（同 TemporalDenoisePass）。 */
        const val SAME_FRAME_MS = 4L
        /** 关闭美颜（且未推流）这么久后结束画面对齐会话、释放环形缓冲。 */
        const val ALIGN_FREE_MS = 2000L
        /** 诊断用视频延后的上限（4 帧在 15 fps 时约 267 ms）。 */
        const val MAX_VIDEO_DELAY_MS = 300L
        /** 负载 1 级起人脸采样间隔（约 15 Hz）。 */
        const val FACE_THROTTLED_INTERVAL_MS = 55L
        /** 美体采样间隔（约 15 Hz，与降级后的人脸同一节拍）。 */
        const val BODY_INTERVAL_MS = 55L
        /** 人像分割采样间隔：只要不忙就每帧都采。 */
        const val SEG_INTERVAL_MS = 25L
        /** 遮罩比被显示帧旧（或新）超过这么多毫秒开始衰减，到 MASK_LAG_ZERO_MS 为 0。 */
        const val MASK_LAG_FULL_MS = 70L
        const val MASK_LAG_ZERO_MS = 200L
        /** 上一帧耗时低于这么多毫秒才做预热。 */
        const val WARMUP_WORK_MS = 12L
        /** 预热的处理道数：FrameDelay、共享小图、美妆程序、背景虚化、头发遮罩、468 点解剖精修子网格。 */
        const val WARMUP_STEPS = 6
    }
}

/**
 * 网格微整与回退路径之间的份额（仅 Android，纯逻辑，只在 GL 线程使用）。回退路径只有片元大眼与 FaceShapePass 的下巴/额头/
 * 瘦鼻，没有瘦脸，形状也不同。实际触发是网格通道故障（FaceMeshWarpPass 抛错，faceMeshHealthy=false），GL 上下文重建后
 * 再回到网格路径；大侧脸不会走到这里：几何门限（约 23°）比网格门限（约 35°）严，网格被拒收的那一帧没有人脸样本，按丢脸退出。
 * [mesh] 为网格路径的份额（回退路径为 1 − [mesh]），两条路径的强度之和恒为 1：
 * - 网格不可用：网格路径根本画不出来，没有可渐出的对象，份额当帧归 0，回退路径满强度接上（不先下陷再渐入）；
 * - 网格恢复可用：只有这个方向两条路径能同时画，份额按 [FADE_MS] 升到 1，交叉渐变；
 * - 脸不可见（presence 为 0）时直接取目标值：新一轮本身由 presence 从 0 渐入，不需要再渐变一次。
 */
internal class MeshFallbackFade {
    var mesh = 1f
        private set
    private var last = Long.MIN_VALUE

    fun reset() { mesh = 1f; last = Long.MIN_VALUE }

    /** 每次绘制调用一次；[meshAvailable] 为本帧网格路径是否可用，[visible] 为 presence 是否大于 0。@return [mesh]。 */
    fun update(meshAvailable: Boolean, visible: Boolean, nowMs: Long): Float {
        val target = if (meshAvailable) 1f else 0f
        val dt = if (last == Long.MIN_VALUE) 0f else (nowMs - last).coerceIn(0L, MAX_STEP_MS).toFloat()
        last = nowMs
        val step = dt / FADE_MS
        mesh = when {
            !visible || !meshAvailable -> target
            else -> minOf(target, mesh + step)
        }
        return mesh
    }

    companion object {
        /** 与 FaceTimeline 新一轮渐入同一时长。 */
        const val FADE_MS = 150f
        /** 单步最多推进 100ms：GL 线程卡顿或切后台回来时不一下跳变。 */
        const val MAX_STEP_MS = 100L
    }
}

/**
 * 头发遮罩的每帧强度（仅 Android，纯逻辑，只在 GL 线程使用）：把 MaskTrackingState.snapshotAt 的结果、[MaskPresence]
 * （渐入 / 保持 / 渐出 = HairMaskTuning.PRESENCE_*，300 / 300 / 400 ms）、[HairMaskTuning.fresh] 与「人脸近期可见」
 * （[HairMaskTuning.faceKeep]）串起来。VideoEffectsFilter 第 1c' 段按 [enabled] 启停头发分割，第 1d 段之后每帧调用一次
 * [update]，再用 [compensate] 求运动补偿，A 通道经 [shown] 乘天鹅颈与外部引擎；HairFallbackJointTest 按同样的顺序驱动。
 * - 不生效（头发分割故障或 p50 超限、美颜关闭、对比原画）：当帧为 0，不等存在度渐出——故障后下一帧就回到第一批（A 恒为 1）；
 * - 没有新遮罩（推理停摆、在途）：存在度保持再渐出，期间沿用已上传的那张，新鲜度按它的采样时间算，lag 420–650 ms 渐回第一批；
 * - 丢脸：最近一次完成的人脸检测确实没有脸（FaceTimeline.missSince）时 [faceKeep] 才渐出，最新有脸样本之后 450 ms 内到 0
 *   （早于 FaceTimeline 500 ms 清空几何，A 通道、合并道与背景虚化的细节保留不会因几何消失而整帧硬切）；检测器只是还在算时
 *   不降（人脸走 CPU、推理慢时不随采样节奏锯齿），停摆（没有人脸任务在途、也没有新结果）时由兜底在 450 ms 到 0，任务在途时
 *   不用兜底；画面里近期没有脸时 [enabled] 为 false，头发分割停用；重新检出后 [faceKeep] 与存在度都从 0 渐入；
 * - 负载 1 级起乘 fades.body 渐出，到 0 后 [enabled] 为 false，头发分割随之停用。
 */
internal class HairMaskGate {
    private val presence = MaskPresence(HairMaskTuning.PRESENCE_FADE_IN_MS, HairMaskTuning.PRESENCE_HOLD_MS,
        HairMaskTuning.PRESENCE_FADE_OUT_MS)
    /** 已上传（[uploaded]）的遮罩的采样时间；还没有时为 Long.MIN_VALUE。 */
    var stamp = Long.MIN_VALUE
        private set
    /** 本帧强度 = 存在度 × 新鲜度 × fades.body × [faceKeep]；不含天鹅颈（背景合并道在美体变形之前，按原帧坐标）。 */
    var level = 0f
        private set
    /**
     * 「人脸近期可见」系数（[HairMaskTuning.faceKeep]）。跟的是人脸而不是头发分割的代际，不随 [reset] 复位；不看 [update] 的 on，
     * 头发关闭或故障时照常推进，背景虚化的细节保留（BackgroundBlurPass.render 的 faceWeight）也用它。
     */
    var faceKeep = 0f
        private set
    private var faceAt = Long.MIN_VALUE

    fun reset() { presence.reset(); stamp = Long.MIN_VALUE; level = 0f }

    /**
     * 每帧调用一次。[on] 为头发分割健康且本帧美颜生效；[mask] 是 snapshotAt(displayTime) 取到的遮罩（没有为 null），
     * [maskStamp] 是它的采样时间；[latestFace] 是 FaceTimeline.latest，[missSince] 是 FaceTimeline.missSince（都在本帧
     * update 之后取；没有为 Long.MIN_VALUE）；[faceInFlight] 是本帧人脸快照的 FaceTrackingState.Snapshot.inFlight（有人脸任务
     * 在途时 [faceKeep] 不用停摆兜底；人脸跟踪没在跑时为 false）。
     * @return 要上传的新遮罩（[on] 且与已上传的不是同一张），否则 null；上传成功后调用 [uploaded]。
     */
    fun update(on: Boolean, mask: PersonMask?, maskStamp: Long, displayTime: Long, bodyFade: Float, latestFace: Long,
               missSince: Long, faceInFlight: Boolean, nowMs: Long): PersonMask? {
        val present = presence.update(mask != null, nowMs)
        faceKeep = HairMaskTuning.faceKeep(faceKeep, latestFace, missSince, nowMs,
            if (faceAt == Long.MIN_VALUE) 0L else nowMs - faceAt, faceInFlight)
        faceAt = nowMs
        val shownStamp = if (mask != null) maskStamp else stamp
        val fresh = if (shownStamp == Long.MIN_VALUE) 0f else HairMaskTuning.fresh((shownStamp - displayTime).toFloat())
        level = if (on) HairMaskTuning.shown(present, fresh, bodyFade * faceKeep, 0f, engine = false) else 0f
        return if (on && mask != null && maskStamp != stamp) mask else null
    }

    fun uploaded(maskStamp: Long) { stamp = maskStamp }

    /**
     * A 通道的强度 = [level] × (1 − [neckWeight])；[neckWeight] 为天鹅颈的生效程度 0..1（见 [neckWeight]，头部被平移，
     * A 通道不跟这段形变），为 1 或走外部引擎时为 0。
     */
    fun shown(neckWeight: Float, engine: Boolean): Float = HairMaskTuning.shown(level, 1f, 1f, neckWeight, engine)

    companion object {
        /**
         * 「画面里近期有脸」的时限：人脸历史非空且最新有脸样本过去不超过它（渲染时间）。FaceTimeline 在 500 ms 时已清空历史，
         * 所以实际在丢脸约 500 ms 后停；这里只兜住历史没清空、却长时间没有新样本的情况。
         */
        const val FACE_RECENT_MS = 1000L
        /** 状态向量里与 FaceTimeline.poseAt 同布局的一段起点：uOval (cx, cy, rx, ry) 后接轴向 (axisX, axisY)，共 POSE_SIZE 个。 */
        const val POSE_INDEX = 22

        /**
         * 第 1c' 段：是否启用头发分割（取票、推理）。[kept] 是 TrackingGrace 判定后的需求；负载 1 级起 fades.body 渐出到 0 后停；
         * 画面里近期没有脸（[latestFace] 为 Long.MIN_VALUE，或距 [nowMs] 超过 [FACE_RECENT_MS]）时停，这一条不经 TrackingGrace。
         */
        fun enabled(healthy: Boolean, kept: Boolean, tier: Int, bodyFade: Float, latestFace: Long, nowMs: Long): Boolean =
            healthy && kept && !(tier >= 1 && bodyFade <= 0f) &&
                latestFace != Long.MIN_VALUE && nowMs - latestFace <= FACE_RECENT_MS

        /** 天鹅颈的生效程度：开着天鹅颈（本帧画了颈部拉伸）时等于人体存在度（天鹅颈强度与它成正比），否则为 0。 */
        fun neckWeight(neckLifted: Boolean, bodyPresence: Float): Float = if (neckLifted) bodyPresence else 0f

        /**
         * 头部运动补偿（计划 H17，评审 2026-11-21 #2 / #9 / #18 修订）：求「显示帧纹理坐标 → 遮罩时刻纹理坐标」的仿射写进 [out]，
         * 返回是否补偿（false 时为单位阵），见 [HairMaskTuning.affine]。
         * - 显示时刻 [displayTime]：本帧 FaceTimeline.at(displayTime) 的几何 [displayGeometry]（rawGeometry，变形前）的
         *   [POSE_INDEX] 起一段。历史区间内与 poseAt(displayTime) 逐值相同；越过最新样本时是 at() 的外推（上限 horizonMs），与合成、
         *   A 通道、合并道用的脸椭圆是同一个几何。不再调 poseAt(displayTime)：它不外推，需要外推的帧会整帧退回单位阵、逐帧跳。
         * - 遮罩时刻 [maskStamp]：不晚于最新样本时取 poseAt(maskStamp)（只插值）。比最新样本还新（人脸推理比头发慢）时：
         *   显示时刻也越过了最新样本，就沿 at() 的外推轨迹（最新样本位姿 → 显示位姿）按时间线性取，最多取到显示位姿；
         *   显示时刻还在历史里，就按最新样本取（补偿已知的那段位移）。只钳到最新样本、不沿外推轨迹取时，显示位姿已外推而遮罩位姿
         *   没有，会把「最新样本 → 遮罩时刻」那段反向多补一遍（仿真：人脸走 CPU 时相邻帧跳变 p99 从 9 px 升到 19 px）。
         * 复用 [poseMask]、[poseDisplay] 两块缓冲，不分配；不调用 at()（没有副作用）。
         */
        fun compensate(timeline: FaceTimeline, maskStamp: Long, displayGeometry: FloatArray?, displayTime: Long, aspect: Float,
                       poseMask: FloatArray, poseDisplay: FloatArray, out: FloatArray): Boolean {
            val newest = timeline.latest
            if (displayGeometry == null || maskStamp == Long.MIN_VALUE || newest == Long.MIN_VALUE) {
                return HairMaskTuning.affine(null, null, aspect, out)
            }
            displayGeometry.copyInto(poseDisplay, 0, POSE_INDEX, POSE_INDEX + FaceTimeline.POSE_SIZE)
            val mask = when {
                maskStamp <= newest -> timeline.poseAt(maskStamp, poseMask)
                !timeline.poseAt(newest, poseMask) -> false
                displayTime > newest -> {
                    val g = minOf(1f, (maskStamp - newest).toFloat() / (displayTime - newest))
                    for (i in 0 until FaceTimeline.POSE_SIZE) poseMask[i] += g * (poseDisplay[i] - poseMask[i])
                    true
                }
                else -> true
            }
            return HairMaskTuning.affine(if (mask) poseMask else null, poseDisplay, aspect, out)
        }
    }
}
