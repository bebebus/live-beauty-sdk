package io.bebebus.livebeauty

import android.content.Context
import android.content.SharedPreferences
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES20.*
import android.opengl.GLUtils
import android.os.Build
import android.os.Process
import android.os.SystemClock
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * SDK 内置的统一 C++17 渲染与数学引擎（`liblivebeauty_jni.so`）标准实现。
 *
 * 完整支持：
 * - 15 轴 $C^\infty$ 微分同胚面部与高颅顶微整、468 点解剖子网格精修、双频奶油肌导向滤波、6 层 UV 美妆、`64³` 3D LUT；
 * - 头发语义保护与背景飞发合并（`HairSegmenter` -> `nativeProcessFrameEx`）；
 * - 无绿幕虚拟背景替换（[setVirtualBackgroundTexture]）；
 * - 最多 4 层人脸锚定 2D AR 挂件贴纸（[setSticker] / [clearStickers]）；
 * - CameraX / Camera2 / WebRTC `GL_TEXTURE_EXTERNAL_OES` 一步直入（[processOes]）；
 * - 实时性能遥测（[perfStats]）。
 */
internal class LiveBeautyCppEngine(
    private val bitrateRatio: () -> Float = { 1f },
    private val lutCacheDir: File? = null,
    private val lutAssets: AssetManager? = null,
    private val onFaceFailure: () -> Unit = {},
    private val onLowLight: (Boolean) -> Unit = {},
    private val clockMs: () -> Long = { SystemClock.uptimeMillis() },
) : BeautyEngine {

    override val latencyFrames: Int
        get() = frameDelayFrames.coerceIn(0, 4)

    private var handle: Long = 0L
    private var appContext: Context? = null
    private var assets: AssetManager? = null
    private var currentConfig: VideoEffectsConfig = VideoEffectsConfig.original
    private val scalars = FloatArray(SCALAR_COUNT)
    private val roiScratch = FloatArray(4)
    private val statsScratch = FloatArray(BeautyPerfStats.RAW_SIZE)

    private val resetPending = AtomicBoolean(false)
    private val faceFailureSent = AtomicBoolean(false)
    @Volatile private var faceHealthy = true
    @Volatile private var bodyHealthy = true
    @Volatile private var segHealthy = true
    @Volatile private var hairHealthy = true

    private val faceSampler = FaceTextureSampler(maxSide = 384)
    private var faceWorker: RawFaceWorker? = null
    private var bodyWorker: RawBodyWorker? = null
    private var personSegmenter: PersonSegmenter? = null
    private var hairSegmenter: HairSegmenter? = null

    private var maskTex = 0
    private var maskW = 0
    private var maskH = 0
    private var lastMaskStamp = Long.MIN_VALUE
    private val hairTex = HairMaskTexture()
    private var lastHairStamp = Long.MIN_VALUE

    private val lutLoader = LutLoader(lutCacheDir, lutAssets)
    private val lutPass = LutPass()
    private var lutHealthy = true
    private var lutFailed = ""
    private var lutRetryAt = Long.MIN_VALUE

    private val atlasDecoder = AtlasDecoder()
    private var blushTex = 0
    private var eyebrowTex = 0
    private var eyeMakeupTex = 0
    private var contourTex = 0
    private var irisTex = 0
    private var lastBoundLut = -1
    private var lastBoundBlush = -1
    private var lastBoundEyebrow = -1
    private var lastBoundEye = -1
    private var lastBoundContour = -1
    private var lastBoundIris = -1

    private var virtualBgMode = VirtualBackgroundMode.NONE
    private var virtualBgTex = 0
    private var virtualBgAspect = 0f
    private var virtualBgStrength = 1f
    private var virtualBgDirty = false

    private val stickers = arrayOfNulls<LiveBeautySticker>(MAX_STICKERS)
    private var stickersDirty = false
    @Volatile private var maxFaces = 1
    @Volatile private var frameDelayFrames = 0
    private var pipelineOptionsDirty = false

    private val light = LowLightDetector()
    private var lastLightSampleMs = Long.MIN_VALUE
    private var lightHealthy = true
    private var lightLuma = Float.NaN

    private var lastFrameTsMs = Long.MIN_VALUE
    private var lastFaceSampleMs = Long.MIN_VALUE
    private var lastBodySampleMs = Long.MIN_VALUE

    @Volatile private var cachedStats = BeautyPerfStats()

    override fun onGlCreated(context: Context) {
        if (!NativeBeautyBridge.isAvailable) {
            throw IllegalStateException("livebeauty_jni is unavailable on this ABI")
        }
        val created = NativeBeautyBridge.nativeCreate()
        if (created == 0L) {
            throw IllegalStateException("NativeBeautyBridge.nativeCreate returned 0")
        }
        handle = created
        appContext = context.applicationContext
        assets = lutAssets ?: appContext?.assets
        resetPending.set(false)
        lastFrameTsMs = Long.MIN_VALUE
        lastFaceSampleMs = Long.MIN_VALUE
        lastBodySampleMs = Long.MIN_VALUE
        lastMaskStamp = Long.MIN_VALUE
        lastHairStamp = Long.MIN_VALUE
        hairTex.release()
        lutHealthy = true
        lutFailed = ""
        lutRetryAt = Long.MIN_VALUE
        lutPass.release()
        lutLoader.restart { currentConfig.let { if (it.enabled) it.lutUrl else "" } }
        light.reset()
        lastLightSampleMs = Long.MIN_VALUE
        lightHealthy = true
        lightLuma = Float.NaN
        invalidateBoundAssets()
        virtualBgDirty = (virtualBgMode != VirtualBackgroundMode.NONE || virtualBgTex != 0)
        stickersDirty = stickers.any { it != null }
        pipelineOptionsDirty = (maxFaces != 1 || frameDelayFrames != 0)
    }

    override fun configure(config: VideoEffectsConfig) {
        currentConfig = config
        if (config.enabled && config.lutUrl.isNotEmpty() && config.lutUrl != lutFailed) {
            lutLoader.request(config.lutUrl)
        }
        val h = handle
        if (h == 0L) return
        NativeBeautyBridge.packConfigScalars(config, scalars)
        NativeBeautyBridge.nativeSetConfig(
            handle = h,
            enabled = config.enabled,
            skinOnly = config.skinOnly,
            compareOriginal = config.compareOriginal,
            scalars = scalars,
            filter = config.filter,
            lutUrl = config.lutUrl,
            lipstickColor = config.lipstickColor,
            lipstickFinish = config.lipstickFinish,
            blushStyle = config.blushStyle,
            blushColor = config.blushColor,
            eyebrowStyle = config.eyebrowStyle,
            eyebrowColor = config.eyebrowColor,
            eyeshadowColor = config.eyeshadowColor,
            contactLensStyle = config.contactLensStyle,
        )
    }

    fun setVirtualBackgroundTexture(
        glTextureId: Int,
        aspect: Float = 0f,
        mode: VirtualBackgroundMode = VirtualBackgroundMode.VIRTUAL_IMAGE,
        strength: Float = 1.0f,
    ) {
        virtualBgTex = glTextureId
        virtualBgAspect = aspect
        virtualBgMode = if (glTextureId == 0 && (mode == VirtualBackgroundMode.VIRTUAL_IMAGE || mode == VirtualBackgroundMode.VIRTUAL_BLUR)) {
            VirtualBackgroundMode.NONE
        } else {
            mode
        }
        virtualBgStrength = strength.coerceIn(0f, 1f)
        virtualBgDirty = true
    }

    fun setSticker(slot: Int, sticker: LiveBeautySticker?) {
        if (slot !in 0 until MAX_STICKERS) return
        stickers[slot] = sticker
        stickersDirty = true
    }

    fun clearStickers() {
        stickers.fill(null)
        stickersDirty = true
    }

    /** 配置双人同屏人数上限（1..2）与零滞后帧对齐环帧数（0..2）。 */
    fun setPipelineOptions(maxFaces: Int = 1, frameDelayFrames: Int = 0) {
        val clampedFaces = maxFaces.coerceIn(1, 2)
        val clampedDelay = frameDelayFrames.coerceIn(0, 4)
        if (this.maxFaces != clampedFaces) {
            this.maxFaces = clampedFaces
            faceWorker?.setMaxFaces(clampedFaces)
        }
        this.frameDelayFrames = clampedDelay
        pipelineOptionsDirty = true
    }

    fun perfStats(): BeautyPerfStats = cachedStats

    override fun resetTracking() {
        resetPending.set(true)
        faceWorker?.reset()
        bodyWorker?.reset()
        personSegmenter?.state?.reset()
        hairSegmenter?.state?.reset()
    }

    override fun process(inputTexture: Int, width: Int, height: Int, timestampNs: Long): Int {
        return processInternal(
            isOes = false,
            texId = inputTexture,
            stMatrix16 = null,
            width = width,
            height = height,
        )
    }

    /** CameraX / Camera2 / WebRTC `GL_TEXTURE_EXTERNAL_OES` 一步直入处理（GL 线程）。 */
    fun processOes(oesTexture: Int, stMatrix16: FloatArray?, width: Int, height: Int, timestampNs: Long): Int {
        return processInternal(
            isOes = true,
            texId = oesTexture,
            stMatrix16 = stMatrix16,
            width = width,
            height = height,
        )
    }

    private fun processInternal(
        isOes: Boolean,
        texId: Int,
        stMatrix16: FloatArray?,
        width: Int,
        height: Int,
    ): Int {
        val h = handle
        if (h == 0L || texId == 0 || width <= 0 || height <= 0) return if (isOes) 0 else texId

        val nowMs = clockMs()
        val frameTsMs = nextMonotonicTs(nowMs)

        if (resetPending.compareAndSet(true, false)) {
            NativeBeautyBridge.nativeResetTracking(h)
            lastMaskStamp = Long.MIN_VALUE
            lastHairStamp = Long.MIN_VALUE
            lastFaceSampleMs = Long.MIN_VALUE
            lastBodySampleMs = Long.MIN_VALUE
        }

        syncDynamicExtensions(h)

        val cfg = currentConfig
        val hasStickers = stickers.any { it != null && it.enabled && it.glTextureId != 0 }
        val wantVirtualBg = (virtualBgMode == VirtualBackgroundMode.VIRTUAL_IMAGE || virtualBgMode == VirtualBackgroundMode.VIRTUAL_BLUR) && virtualBgTex != 0
        val wantFace = faceHealthy && (needsFace(cfg) || (cfg.enabled && !cfg.compareOriginal && hasStickers))
        val wantBody = bodyHealthy && needsBody(cfg)
        val wantSeg = segHealthy && (needsSegmentation(cfg) || (cfg.enabled && !cfg.compareOriginal && wantVirtualBg))
        val wantHair = hairHealthy && HairMaskTuning.wants(hairHealthy, true, wantFace, cfg)
        val wantMakeup = cfg.enabled && !cfg.compareOriginal && cfg.hasMakeup()

        consumeFaceResult(h, wantFace)
        consumeBodyResult(h, wantBody)
        val activeMaskTex = consumePersonMask(wantSeg, frameTsMs, nowMs)
        val (activeHairTex, activeHairStamp) = consumeHairMask(wantHair, frameTsMs, nowMs)

        if (!isOes) {
            driveSampling(h, texId, width, height, frameTsMs, nowMs, wantFace, wantBody, wantSeg, wantHair)
        }

        val activeLutTex = updateLutTexture(cfg, nowMs)
        if (wantMakeup) updateMakeupAtlases()
        syncNativeAssets(h, activeLutTex)

        val ratio = try { bitrateRatio() } catch (_: Exception) { 1f }
        val outTex = if (isOes) {
            NativeBeautyBridge.nativeProcessOesFrame(
                handle = h,
                oesGlTex = texId,
                stMatrix16 = stMatrix16,
                width = width,
                height = height,
                frameTsMs = frameTsMs,
                nowMs = nowMs,
                personMaskGlTex = activeMaskTex,
                hairMaskGlTex = activeHairTex,
                hairMaskStampMs = activeHairStamp,
                lightLuma = lightLuma,
                bitrateRatio = ratio,
            )
        } else {
            NativeBeautyBridge.nativeProcessFrameEx(
                handle = h,
                inputGlTex = texId,
                width = width,
                height = height,
                frameTsMs = frameTsMs,
                nowMs = nowMs,
                personMaskGlTex = activeMaskTex,
                hairMaskGlTex = activeHairTex,
                hairMaskStampMs = activeHairStamp,
                lightLuma = lightLuma,
                bitrateRatio = ratio,
            )
        }

        NativeBeautyBridge.nativeGetStats(h, statsScratch)
        cachedStats = BeautyPerfStats.fromRaw(statsScratch)

        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        return if (outTex != 0) outTex else if (isOes) 0 else texId
    }

    private fun syncDynamicExtensions(h: Long) {
        if (virtualBgDirty) {
            virtualBgDirty = false
            NativeBeautyBridge.nativeSetVirtualBackground(
                handle = h,
                bgGlTex = virtualBgTex,
                bgAspect = virtualBgAspect,
                mode = virtualBgMode.wireName,
                strength = virtualBgStrength,
            )
        }
        if (pipelineOptionsDirty) {
            pipelineOptionsDirty = false
            NativeBeautyBridge.nativeSetPipelineOptions(h, maxFaces, frameDelayFrames)
        }
        if (stickersDirty) {
            stickersDirty = false
            NativeBeautyBridge.nativeClearStickers(h)
            for (slot in 0 until MAX_STICKERS) {
                val s = stickers[slot] ?: continue
                if (s.enabled && s.glTextureId != 0) {
                    NativeBeautyBridge.nativeSetSticker(
                        handle = h,
                        slot = slot,
                        enabled = true,
                        anchor = s.anchor.id,
                        texGlId = s.glTextureId,
                        scale = s.scale,
                        offsetX = s.offsetX,
                        offsetY = s.offsetY,
                        rotationRad = s.rotationRad,
                        opacity = s.opacity,
                        textureAspect = s.textureAspect,
                    )
                }
            }
        }
    }

    override fun onGlReleased(contextAlive: Boolean) {
        try { if (light.dark) onLowLight(false) } catch (_: Exception) {}
        light.reset()
        lightLuma = Float.NaN

        faceWorker?.close(); faceWorker = null
        bodyWorker?.close(); bodyWorker = null
        personSegmenter?.close(); personSegmenter = null
        hairSegmenter?.close(); hairSegmenter = null
        atlasDecoder.cancel()

        if (contextAlive) {
            faceSampler.delete()
            lutPass.delete()
            hairTex.delete()
            deleteMaskTexture()
            deleteMakeupAtlases()
        } else {
            faceSampler.release()
            lutPass.release()
            hairTex.release()
            maskTex = 0; maskW = 0; maskH = 0; lastMaskStamp = Long.MIN_VALUE
            lastHairStamp = Long.MIN_VALUE
            blushTex = 0; eyebrowTex = 0; eyeMakeupTex = 0; contourTex = 0; irisTex = 0
        }
        invalidateBoundAssets()

        val h = handle
        handle = 0L
        if (h != 0L) {
            try { NativeBeautyBridge.nativeReleaseGpu(h, contextAlive) } catch (_: Exception) {}
            try { NativeBeautyBridge.nativeDestroy(h, contextAlive) } catch (_: Exception) {}
        }
    }

    private fun nextMonotonicTs(nowMs: Long): Long {
        val ts = if (lastFrameTsMs == Long.MIN_VALUE || nowMs > lastFrameTsMs) nowMs else lastFrameTsMs + 1L
        lastFrameTsMs = ts
        return ts
    }

    private fun consumeFaceResult(h: Long, wantFace: Boolean) {
        val res = faceWorker?.pollResult() ?: return
        if (!wantFace) return
        val pts = res.mpXy
        if (pts != null && res.faceCount > 0 && res.pointsPerFace >= 468) {
            if (res.faceCount == 1) {
                NativeBeautyBridge.nativeSubmitFaceLandmarks(
                    h, res.sampleTsMs, pts, res.pointsPerFace, res.width, res.height, res.receivedMs,
                )
            } else {
                NativeBeautyBridge.nativeSubmitMultiFaceLandmarks(
                    h, res.sampleTsMs, pts, res.faceCount, res.pointsPerFace, res.width, res.height, res.receivedMs,
                )
            }
        } else {
            NativeBeautyBridge.nativeSubmitFaceMiss(h, res.sampleTsMs)
        }
    }

    private fun consumeBodyResult(h: Long, wantBody: Boolean) {
        val res = bodyWorker?.pollResult() ?: return
        if (!wantBody) return
        val pose = res.poseXyv
        if (pose != null && res.pointCount >= 33) {
            NativeBeautyBridge.nativeSubmitBodyPose(h, res.sampleTsMs, pose, res.pointCount, res.width, res.height)
        }
    }

    private fun consumePersonMask(wantSeg: Boolean, frameTsMs: Long, nowMs: Long): Int {
        val seg = personSegmenter
        seg?.state?.enable(wantSeg)
        if (!wantSeg || seg == null) return 0
        val snap = seg.state.snapshotAt(frameTsMs, nowMs)
        val mask = snap.mask
        if (mask != null && snap.sampleTimestamp != lastMaskStamp) {
            uploadPersonMask(mask)
            lastMaskStamp = snap.sampleTimestamp
        }
        return if (mask != null && maskTex != 0) maskTex else 0
    }

    private fun consumeHairMask(wantHair: Boolean, frameTsMs: Long, nowMs: Long): Pair<Int, Long> {
        val hair = hairSegmenter
        hair?.state?.enable(wantHair)
        if (!wantHair || hair == null) return 0 to Long.MIN_VALUE
        if (!HairMaskTuning.healthy(hair.inferenceP50Ms())) {
            hairHealthy = false
            hair.state.enable(false)
            return 0 to Long.MIN_VALUE
        }
        val snap = hair.state.snapshotAt(frameTsMs, nowMs)
        val mask = snap.mask
        if (mask != null && snap.sampleTimestamp != lastHairStamp) {
            try {
                hairTex.upload(mask)
                lastHairStamp = snap.sampleTimestamp
            } catch (_: Exception) {
                hairHealthy = false
                return 0 to Long.MIN_VALUE
            }
        }
        return if (mask != null && hairTex.texture != 0) (hairTex.texture to lastHairStamp) else (0 to Long.MIN_VALUE)
    }

    private fun driveSampling(
        h: Long,
        inputTexture: Int,
        width: Int,
        height: Int,
        frameTsMs: Long,
        nowMs: Long,
        wantFace: Boolean,
        wantBody: Boolean,
        wantSeg: Boolean,
        wantHair: Boolean,
    ) {
        val ctx = appContext ?: return
        if (wantFace && faceWorker == null) {
            faceWorker = RawFaceWorker(ctx, maxFaces) {
                faceHealthy = false
                if (faceFailureSent.compareAndSet(false, true)) onFaceFailure()
            }
        }
        if (wantBody && bodyWorker == null) {
            bodyWorker = RawBodyWorker(ctx) { bodyHealthy = false }
        }
        if (wantSeg && personSegmenter == null) {
            personSegmenter = PersonSegmenter(ctx) { segHealthy = false }
            personSegmenter?.state?.enable(true)
        }
        if (wantHair && hairSegmenter == null) {
            hairSegmenter = HairSegmenter(ctx) { hairHealthy = false }
            hairSegmenter?.state?.enable(true)
        }

        val faceInterval = if (faceWorker?.gpu == true) FACE_GPU_INTERVAL_MS else FACE_CPU_INTERVAL_MS
        val doFace = wantFace && faceWorker?.canAccept() == true &&
            (lastFaceSampleMs == Long.MIN_VALUE || frameTsMs - lastFaceSampleMs >= faceInterval)
        val doBody = wantBody && bodyWorker?.canAccept() == true &&
            (lastBodySampleMs == Long.MIN_VALUE || frameTsMs - lastBodySampleMs >= BODY_INTERVAL_MS)
        val segTicket = if (wantSeg) personSegmenter?.state?.begin(frameTsMs, width, height, SEG_INTERVAL_MS) else null
        val hairInterval = HairMaskTuning.interval(hairSegmenter?.inferenceP50Ms() ?: Float.NaN, wantSeg)
        val hairTicket = if (wantHair) hairSegmenter?.state?.begin(frameTsMs, width, height, hairInterval) else null
        val doLight = lightHealthy &&
            (lastLightSampleMs == Long.MIN_VALUE || nowMs - lastLightSampleMs >= LowLightDetector.INTERVAL_MS)

        if (!doFace && !doBody && segTicket == null && hairTicket == null && !doLight) return

        val crop = if (doFace) {
            roiScratch[0] = 0f; roiScratch[1] = 0f; roiScratch[2] = 1f; roiScratch[3] = 1f
            NativeBeautyBridge.nativeSelectFaceRoi(h, width, height, nowMs, roiScratch)
            roiToCropRect(roiScratch)
        } else {
            FaceRoiSelector.CropRect.FULL
        }

        val captured = try {
            faceSampler.capture(inputTexture, width, height, restoreFbo = 0, sync = false, faceCrop = crop)
            true
        } catch (_: Exception) {
            false
        }
        if (!captured) {
            segTicket?.let { personSegmenter?.state?.complete(it, null, nowMs) }
            hairTicket?.let { hairSegmenter?.state?.complete(it, null, nowMs) }
            return
        }

        if (doLight) {
            lastLightSampleMs = nowMs
            try {
                faceSampler.staging?.let { sample ->
                    val measured = LowLightDetector.meanLuma(sample.pixels, sample.width, sample.height)
                    if (measured.isFinite()) {
                        lightLuma = if (lightLuma.isNaN()) measured else lightLuma + 0.35f * (measured - lightLuma)
                        light.onSample(measured)?.let { dark -> try { onLowLight(dark) } catch (_: Exception) {} }
                    }
                }
            } catch (_: Exception) {
                lightHealthy = false
            }
        }
        if (doFace) {
            lastFaceSampleMs = frameTsMs
            try { faceWorker?.submit(faceSampler.lease(FaceTextureSampler.FACE), frameTsMs, width, height) }
            catch (_: Exception) { faceHealthy = false; if (faceFailureSent.compareAndSet(false, true)) onFaceFailure() }
        }
        if (doBody) {
            lastBodySampleMs = frameTsMs
            try { bodyWorker?.submit(faceSampler.lease(FaceTextureSampler.BODY), frameTsMs, width, height) }
            catch (_: Exception) { bodyHealthy = false }
        }
        if (segTicket != null) {
            try { personSegmenter?.submit(faceSampler.lease(FaceTextureSampler.SEGMENT), segTicket) }
            catch (_: Exception) { personSegmenter?.state?.complete(segTicket, null, nowMs); segHealthy = false }
        }
        if (hairTicket != null) {
            try { hairSegmenter?.submit(faceSampler.lease(FaceTextureSampler.HAIR), hairTicket, null) }
            catch (_: Exception) { hairSegmenter?.state?.complete(hairTicket, null, nowMs); hairHealthy = false }
        }
    }

    private fun updateLutTexture(cfg: VideoEffectsConfig, nowMs: Long): Int {
        val url = if (cfg.enabled && !cfg.compareOriginal) cfg.lutUrl else ""
        if (url.isNotEmpty() && url == lutFailed && nowMs >= lutRetryAt) {
            lutFailed = ""
            lutLoader.restart { currentConfig.let { if (it.enabled) it.lutUrl else "" } }
        }
        if (lutHealthy && url.isNotEmpty() && url != lutPass.loadedUrl && url != lutFailed) {
            lutLoader.take(url)?.let { res ->
                val bmp = res.bitmap
                if (bmp == null) {
                    lutFailed = url
                    lutRetryAt = nowMs + VideoEffectsFilter.LUT_RETRY_MS
                } else {
                    try { lutPass.upload(url, bmp) }
                    catch (_: Exception) { lutHealthy = false }
                    finally { bmp.recycle() }
                }
            }
        }
        return lutPass.select(if (lutHealthy && url.isNotEmpty() && url != lutFailed) url else "")
    }

    private fun updateMakeupAtlases() {
        val mgr = assets ?: return
        atlasDecoder.request(mgr)
        while (true) {
            val decoded = atlasDecoder.poll() ?: break
            val bmp = decoded.bitmap ?: continue
            try {
                val tex = uploadRgbaBitmap(bmp)
                when (decoded.index) {
                    0 -> { if (blushTex != 0) glDeleteTextures(1, intArrayOf(blushTex), 0); blushTex = tex }
                    1 -> { if (eyebrowTex != 0) glDeleteTextures(1, intArrayOf(eyebrowTex), 0); eyebrowTex = tex }
                    2 -> { if (eyeMakeupTex != 0) glDeleteTextures(1, intArrayOf(eyeMakeupTex), 0); eyeMakeupTex = tex }
                    3 -> { if (contourTex != 0) glDeleteTextures(1, intArrayOf(contourTex), 0); contourTex = tex }
                    4 -> { if (irisTex != 0) glDeleteTextures(1, intArrayOf(irisTex), 0); irisTex = tex }
                }
            } finally {
                bmp.recycle()
            }
        }
    }

    private fun syncNativeAssets(h: Long, activeLutTex: Int) {
        if (activeLutTex == lastBoundLut && blushTex == lastBoundBlush &&
            eyebrowTex == lastBoundEyebrow && eyeMakeupTex == lastBoundEye &&
            contourTex == lastBoundContour && irisTex == lastBoundIris
        ) return
        lastBoundLut = activeLutTex; lastBoundBlush = blushTex
        lastBoundEyebrow = eyebrowTex; lastBoundEye = eyeMakeupTex
        lastBoundContour = contourTex; lastBoundIris = irisTex
        NativeBeautyBridge.nativeSetAssets(h, activeLutTex, blushTex, eyebrowTex, eyeMakeupTex, contourTex, irisTex)
    }

    private fun invalidateBoundAssets() {
        lastBoundLut = -1; lastBoundBlush = -1; lastBoundEyebrow = -1
        lastBoundEye = -1; lastBoundContour = -1; lastBoundIris = -1
    }

    private fun uploadPersonMask(mask: PersonMask) {
        glActiveTexture(GL_TEXTURE0)
        val fresh = maskTex == 0 || mask.width != maskW || mask.height != maskH
        if (maskTex == 0) {
            val ids = IntArray(1); glGenTextures(1, ids, 0); maskTex = ids[0]
            glBindTexture(GL_TEXTURE_2D, maskTex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        } else {
            glBindTexture(GL_TEXTURE_2D, maskTex)
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        val buf = ByteBuffer.wrap(mask.pixels)
        if (fresh) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, mask.width, mask.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buf)
            maskW = mask.width; maskH = mask.height
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mask.width, mask.height, GL_LUMINANCE, GL_UNSIGNED_BYTE, buf)
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
    }

    private fun deleteMaskTexture() {
        if (maskTex != 0) glDeleteTextures(1, intArrayOf(maskTex), 0)
        maskTex = 0; maskW = 0; maskH = 0; lastMaskStamp = Long.MIN_VALUE
    }

    private fun deleteMakeupAtlases() {
        val ids = intArrayOf(blushTex, eyebrowTex, eyeMakeupTex, contourTex, irisTex).filter { it != 0 }.toIntArray()
        if (ids.isNotEmpty()) glDeleteTextures(ids.size, ids, 0)
        blushTex = 0; eyebrowTex = 0; eyeMakeupTex = 0; contourTex = 0; irisTex = 0
    }

    private fun uploadRgbaBitmap(bitmap: Bitmap): Int {
        val ids = IntArray(1); glGenTextures(1, ids, 0); val tex = ids[0]
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

    private class RawFaceWorker(context: Context, initialMaxFaces: Int, private val onFailure: () -> Unit) {
        class Result(val epoch: Long, val sampleTsMs: Long, val receivedMs: Long, val width: Int, val height: Int, val mpXy: FloatArray?, val faceCount: Int, val pointsPerFace: Int)
        private val appContext = context.applicationContext
        private val closed = AtomicBoolean(false)
        private val inFlight = AtomicBoolean(false)
        private val epoch = AtomicLong(0L)
        private val latest = AtomicReference<Result?>(null)
        private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "lb-sdk-face").apply { isDaemon = true } }
        private var detector: FaceLandmarker? = null
        private var detectorNumFaces = initialMaxFaces.coerceIn(1, 2)
        @Volatile private var desiredNumFaces = detectorNumFaces
        private val policy = sharedGpuPolicy(appContext)
        @Volatile var gpu = policy.allowGpu(); private set
        private var gpuConfirmed = false

        fun setMaxFaces(n: Int) { desiredNumFaces = n.coerceIn(1, 2) }
        fun canAccept() = !closed.get() && !inFlight.get()
        fun reset() { epoch.incrementAndGet(); latest.set(null) }
        fun pollResult(): Result? = latest.getAndSet(null)?.takeIf { it.epoch == epoch.get() }
        fun submit(sample: FaceTextureSampler.Sample, sampleTsMs: Long, w: Int, h: Int) {
            if (closed.get() || !inFlight.compareAndSet(false, true)) return
            val ep = epoch.get()
            try { worker.execute { run(sample, ep, sampleTsMs, w, h) } } catch (_: RejectedExecutionException) { inFlight.set(false) }
        }
        private fun run(sample: FaceTextureSampler.Sample, ep: Long, ts: Long, w: Int, h: Int) {
            try {
                if (closed.get()) return
                val img = ByteBufferImageBuilder(sample.pixels, sample.width, sample.height, MPImage.IMAGE_FORMAT_RGBA).build()
                var outFlat: FloatArray? = null
                var outFaces = 0
                var outPtsPerFace = 0
                try {
                    val faces = try { task().detectForVideo(img, ts).faceLandmarks() } catch (e: Throwable) {
                        if (!gpu) throw e
                        gpu = false; policy.onGpuJavaFailure(); try { detector?.close() } catch (_: Throwable) {}; detector = null
                        return
                    }
                    if (gpu && !gpuConfirmed) { gpuConfirmed = true; policy.onGpuSuccess() }
                    val crop = sample.crop
                    val takeFaces = minOf(faces.size, desiredNumFaces)
                    if (takeFaces > 0) {
                        val ptsPer = faces[0].size
                        if (ptsPer >= 468) {
                            val flat = FloatArray(takeFaces * ptsPer * 2)
                            var validFaces = 0
                            for (fi in 0 until takeFaces) {
                                val f = faces[fi]
                                if (f.size < ptsPer) continue
                                val base = validFaces * ptsPer * 2
                                for (i in 0 until ptsPer) {
                                    val lm = f[i]
                                    flat[base + i * 2] = if (crop.isFullFrame) lm.x() else crop.unprojectX(lm.x())
                                    flat[base + i * 2 + 1] = if (crop.isFullFrame) lm.y() else crop.unprojectY(lm.y())
                                }
                                validFaces++
                            }
                            if (validFaces > 0) {
                                outFlat = flat
                                outFaces = validFaces
                                outPtsPerFace = ptsPer
                            }
                        }
                    }
                } finally { img.close() }
                if (!closed.get() && ep == epoch.get()) {
                    latest.set(Result(ep, ts, SystemClock.uptimeMillis(), w, h, outFlat, outFaces, outPtsPerFace))
                }
            } catch (_: Throwable) { if (!closed.get()) onFailure() } finally { inFlight.set(false) }
        }
        private fun task(): FaceLandmarker {
            val wanted = desiredNumFaces
            if (detector != null && detectorNumFaces != wanted) {
                try { detector?.close() } catch (_: Throwable) {}
                detector = null
            }
            detector?.let { return it }
            detectorNumFaces = wanted
            if (gpu && policy.allowGpu()) {
                policy.beforeGpuAttempt()
                try { return create(Delegate.GPU, wanted).also { detector = it } } catch (_: Throwable) { gpu = false; policy.onGpuJavaFailure() }
            }
            gpu = false; return create(Delegate.CPU, wanted).also { detector = it }
        }
        private fun create(d: Delegate, numFaces: Int) = FaceLandmarker.createFromOptions(appContext,
            FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").setDelegate(d).build())
                .setRunningMode(RunningMode.VIDEO).setNumFaces(numFaces)
                .setMinFaceDetectionConfidence(0.6f).setMinFacePresenceConfidence(0.6f).setMinTrackingConfidence(0.5f).build())
        fun close() {
            if (!closed.compareAndSet(false, true)) return
            epoch.incrementAndGet(); latest.set(null)
            try { worker.execute { detector?.close(); detector = null } } catch (_: Exception) {}
            worker.shutdown()
        }
    }

    private class RawBodyWorker(context: Context, private val onFailure: () -> Unit) {
        class Result(val epoch: Long, val sampleTsMs: Long, val width: Int, val height: Int, val poseXyv: FloatArray?, val pointCount: Int)
        private val appContext = context.applicationContext
        private val closed = AtomicBoolean(false)
        private val inFlight = AtomicBoolean(false)
        private val epoch = AtomicLong(0L)
        private val latest = AtomicReference<Result?>(null)
        private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "lb-sdk-body").apply { isDaemon = true } }
        private var detector: PoseLandmarker? = null

        fun canAccept() = !closed.get() && !inFlight.get()
        fun reset() { epoch.incrementAndGet(); latest.set(null) }
        fun pollResult(): Result? = latest.getAndSet(null)?.takeIf { it.epoch == epoch.get() }
        fun submit(sample: FaceTextureSampler.Sample, ts: Long, w: Int, h: Int) {
            if (closed.get() || !inFlight.compareAndSet(false, true)) return
            val ep = epoch.get()
            try { worker.execute { run(sample, ep, ts, w, h) } } catch (_: RejectedExecutionException) { inFlight.set(false) }
        }
        private fun run(sample: FaceTextureSampler.Sample, ep: Long, ts: Long, w: Int, h: Int) {
            try {
                if (closed.get()) return
                val task = detector ?: PoseLandmarker.createFromOptions(appContext,
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath(BodyPoseTracker.MODEL).build())
                        .setRunningMode(RunningMode.VIDEO).setNumPoses(1)
                        .setMinPoseDetectionConfidence(0.5f).setMinPosePresenceConfidence(0.5f).setMinTrackingConfidence(0.5f).build()).also { detector = it }
                val img = ByteBufferImageBuilder(sample.pixels, sample.width, sample.height, MPImage.IMAGE_FORMAT_RGBA).build()
                val raw = try {
                    task.detectForVideo(img, ts).landmarks().firstOrNull()?.let { p ->
                        val n = minOf(p.size, 33); val out = FloatArray(n * 3)
                        for (i in 0 until n) { out[i * 3] = p[i].x(); out[i * 3 + 1] = p[i].y(); out[i * 3 + 2] = p[i].visibility().orElse(0f) }
                        out to n
                    }
                } finally { img.close() }
                if (!closed.get() && ep == epoch.get()) latest.set(Result(ep, ts, w, h, raw?.first, raw?.second ?: 0))
            } catch (_: Throwable) { if (!closed.get()) onFailure() } finally { inFlight.set(false) }
        }
        fun close() {
            if (!closed.compareAndSet(false, true)) return
            epoch.incrementAndGet(); latest.set(null)
            try { worker.execute { detector?.close(); detector = null } } catch (_: Exception) {}
            worker.shutdown()
        }
    }

    private class AtlasDecoder {
        class Decoded(val index: Int, val bitmap: Bitmap?)
        private val ready = ConcurrentLinkedQueue<Pair<Int, Decoded>>()
        @Volatile private var gen = 0
        private var requested = false
        fun request(mgr: AssetManager) {
            if (requested) return; requested = true; val g = gen
            DECODER.execute {
                for (i in ATLAS_PATHS.indices) {
                    if (g != gen) return@execute
                    val bmp = try {
                        val o = BitmapFactory.Options().apply { inPremultiplied = false; inScaled = false; inPreferredConfig = Bitmap.Config.ARGB_8888 }
                        mgr.open(ATLAS_PATHS[i]).use { BitmapFactory.decodeStream(it, null, o) }
                    } catch (_: Throwable) { null }
                    if (g != gen) { bmp?.recycle(); return@execute }
                    ready.add(g to Decoded(i, bmp))
                }
            }
        }
        fun poll(): Decoded? {
            while (true) {
                val (g, d) = ready.poll() ?: return null
                if (g == gen) return d
                d.bitmap?.recycle()
            }
        }
        fun cancel() { gen++; requested = false; while (true) { ready.poll()?.second?.bitmap?.recycle() ?: break } }
    }

    private class PrefsStore(private val prefs: SharedPreferences) : GpuDelegatePolicy.Store {
        override fun getString(key: String): String? = try { prefs.getString(key, null) } catch (_: ClassCastException) { null }
        override fun getInt(key: String, default: Int): Int = try { prefs.getInt(key, default) } catch (_: ClassCastException) { default }
        override fun put(changes: Map<String, Any?>, sync: Boolean) {
            val e = prefs.edit()
            for ((k, v) in changes) when (v) { null -> e.remove(k); is Int -> e.putInt(k, v); is String -> e.putString(k, v); else -> e.putString(k, v.toString()) }
            if (sync) e.commit() else e.apply()
        }
    }

    companion object {
        const val SCALAR_COUNT = 45
        const val MAX_STICKERS = 4
        private const val FACE_GPU_INTERVAL_MS = 1L
        private const val FACE_CPU_INTERVAL_MS = 45L
        private const val BODY_INTERVAL_MS = VideoEffectsFilter.BODY_INTERVAL_MS
        private const val SEG_INTERVAL_MS = VideoEffectsFilter.SEG_INTERVAL_MS
        private const val PREFS = "livebeauty_face_tracker"
        private val ATLAS_PATHS = arrayOf(
            "makeup/blush_atlas.png", "makeup/eyebrow_atlas.png",
            "makeup/eye_makeup.png", "makeup/contour_highlight.png", "makeup/iris_atlas.png",
        )
        private val DECODER = Executors.newSingleThreadExecutor { r -> Thread(r, "lb-sdk-atlas").apply { isDaemon = true } }
        @Volatile private var cachedPolicy: GpuDelegatePolicy? = null
        private fun sharedGpuPolicy(context: Context): GpuDelegatePolicy = cachedPolicy ?: synchronized(this) {
            cachedPolicy ?: GpuDelegatePolicy(
                PrefsStore(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)),
                Build.FINGERPRINT ?: "", Process.myPid(), Process.getStartUptimeMillis(),
            ).also { cachedPolicy = it }
        }

        fun roiToCropRect(rect4: FloatArray): FaceRoiSelector.CropRect {
            if (rect4.size < 4) return FaceRoiSelector.CropRect.FULL
            val l = rect4[0]; val t = rect4[1]; val w = rect4[2]; val h = rect4[3]
            if (!l.isFinite() || !t.isFinite() || !w.isFinite() || !h.isFinite()) return FaceRoiSelector.CropRect.FULL
            if (w <= 0.01f || h <= 0.01f || l < 0f || t < 0f || l + w > 1.001f || t + h > 1.001f) return FaceRoiSelector.CropRect.FULL
            if (l <= 1e-4f && t <= 1e-4f && w >= 0.999f && h >= 0.999f) return FaceRoiSelector.CropRect.FULL
            return FaceRoiSelector.CropRect(l, t, w, h)
        }

        fun needsFace(cfg: VideoEffectsConfig) = cfg.enabled && (
            cfg.hasFaceReshape() || cfg.hasMakeup() || cfg.whiten > 0f ||
                cfg.darkCircles > 0f || cfg.brightEyes > 0f || cfg.teethWhiten > 0f ||
                cfg.nasolabial > 0f || cfg.rosy > 0f || cfg.radiance > 0f ||
                (cfg.skinOnly && cfg.smoothing > 0f))

        fun needsBody(cfg: VideoEffectsConfig) = cfg.enabled && (
            cfg.waistSlim > 0f || cfg.bustEnhance > 0f || cfg.legLengthen > 0f ||
                cfg.shoulderSlim > 0f || cfg.neckLengthen > 0f)

        fun needsSegmentation(cfg: VideoEffectsConfig) = cfg.enabled && cfg.backgroundBlur > 0f
    }
}
