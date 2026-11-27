#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "gles_device.h"
#include "livebeauty/config.h"
#include "livebeauty/engine.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/face_mesh_deformer.h"

namespace {

struct JniEngineHandle {
    livebeauty::rhi::gles::GlesDevice device;
    livebeauty::LiveBeautyEngine engine;
    livebeauty::rhi::PipelineAssets assets{};
    livebeauty::BackgroundMode bgModeOverride = livebeauty::BackgroundMode::kNone;
    float virtualBgStrengthOverride = 1.0f;
    livebeauty::ChromaKeyColor chromaKeyColorOverride = livebeauty::ChromaKeyColor::kGreen;
    float chromaKeySimilarityOverride = 0.38f;
    float chromaKeySmoothnessOverride = 0.12f;
    float chromaKeyDespillOverride = 0.75f;
    int maxFacesOverride = 1;
    int frameDelayFramesOverride = 0;
    livebeauty::math::RelightStyle relightStyleOverride = livebeauty::math::RelightStyle::kNatural;
    livebeauty::math::PerformanceTier performanceTierOverride = livebeauty::math::PerformanceTier::kAuto;
    float hairDyeOverride = 0.0f;
    livebeauty::HairDyeColor hairDyeColorOverride = livebeauty::HairDyeColor::kMilkTeaBrown;
    float hairShineOverride = 0.0f;
    bool preferHalfFloatOverride = true;

    JniEngineHandle() : device(), engine(device) {}
};

inline JniEngineHandle* fromHandle(jlong ptr) {
    return reinterpret_cast<JniEngineHandle*>(ptr);
}

std::string toStdString(JNIEnv* env, jstring js) {
    if (!js || env->ExceptionCheck()) return {};
    const char* chars = env->GetStringUTFChars(js, nullptr);
    if (!chars) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        return {};
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(js, chars);
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeCreate(JNIEnv*, jclass) {
    try {
        auto* handle = new JniEngineHandle();
        return reinterpret_cast<jlong>(handle);
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeDestroy(JNIEnv*, jclass, jlong ptr, jboolean contextAlive) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->device.releaseAll(contextAlive == JNI_TRUE);
    handle->engine.releaseGpuResources();
    delete handle;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeResetTracking(JNIEnv*, jclass, jlong ptr) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.resetTracking();
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeReleaseGpu(JNIEnv*, jclass, jlong ptr, jboolean contextAlive) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    // Clear GlesDevice tables first (with glDelete* only if contextAlive), then reset Engine/RenderPipeline handles unconditionally
    handle->device.releaseAll(contextAlive == JNI_TRUE);
    handle->engine.releaseGpuResources();
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAssets(
    JNIEnv*, jclass, jlong ptr,
    jint lutGlTex, jint blushGlTex, jint eyebrowGlTex,
    jint eyeMakeupGlTex, jint contourHighlightGlTex, jint irisGlTex) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->assets.lutTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(lutGlTex));
    handle->assets.blushAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(blushGlTex));
    handle->assets.eyebrowAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(eyebrowGlTex));
    handle->assets.eyeMakeupAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(eyeMakeupGlTex));
    handle->assets.contourHighlightAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(contourHighlightGlTex));
    handle->assets.irisAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(irisGlTex));
    handle->engine.setAssets(handle->assets);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetVirtualBackground(
    JNIEnv* env, jclass, jlong ptr,
    jint bgGlTex, jfloat bgAspect, jstring modeStr, jfloat strength) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->assets.virtualBgTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(bgGlTex));
    handle->assets.virtualBgAspect = bgAspect;
    handle->engine.setAssets(handle->assets);

    handle->bgModeOverride = livebeauty::parseBackgroundMode(toStdString(env, modeStr));
    handle->virtualBgStrengthOverride = strength;
    auto cfg = handle->engine.config();
    cfg.backgroundMode = handle->bgModeOverride;
    cfg.virtualBgStrength = handle->virtualBgStrengthOverride;
    handle->engine.setConfig(cfg);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetSticker(
    JNIEnv*, jclass, jlong ptr,
    jint slot, jboolean enabled, jint anchor, jint texGlId,
    jfloat scale, jfloat offsetX, jfloat offsetY,
    jfloat rotationRad, jfloat opacity, jfloat textureAspect) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    livebeauty::math::StickerLayerConfig layer{};
    layer.enabled = (enabled == JNI_TRUE);
    layer.anchor = static_cast<livebeauty::math::StickerAnchor>(anchor);
    layer.texture = handle->device.wrapExternalTexture(static_cast<uint32_t>(texGlId));
    layer.scale = scale;
    layer.offsetX = offsetX;
    layer.offsetY = offsetY;
    layer.rotationRad = rotationRad;
    layer.opacity = opacity;
    layer.textureAspect = textureAspect;
    handle->engine.setSticker(slot, layer);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeClearStickers(
    JNIEnv*, jclass, jlong ptr) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.clearStickers();
}

// Pack all 40 scalar floats in one flat JNI array + 9 enum strings to avoid per-field JNI overhead.
JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetConfig(
    JNIEnv* env, jclass, jlong ptr,
    jboolean enabled, jboolean skinOnly, jboolean compareOriginal,
    jfloatArray scalars,
    jstring filterStr, jstring lutUrlStr,
    jstring lipColorStr, jstring lipFinishStr,
    jstring blushStyleStr, jstring blushColorStr,
    jstring browStyleStr, jstring browColorStr,
    jstring shadowColorStr, jstring lensStyleStr) {
    auto* handle = fromHandle(ptr);
    if (!handle || !scalars) return;
    const jsize n = env->GetArrayLength(scalars);
    if (n < 39) return;
    jfloat s[64];
    std::memset(s, 0, sizeof(s));
    env->GetFloatArrayRegion(scalars, 0, n < 64 ? n : 64, s);

    livebeauty::BeautyConfig cfg{};
    cfg.enabled = (enabled == JNI_TRUE);
    cfg.skinOnly = (skinOnly == JNI_TRUE);
    cfg.compareOriginal = (compareOriginal == JNI_TRUE);
    cfg.smoothing = s[0];
    cfg.brightness = s[1];
    cfg.sharpness = s[2];
    cfg.saturation = s[3];
    cfg.filterIntensity = s[4];
    cfg.faceSlim = s[5];
    cfg.eyeEnlarge = s[6];
    cfg.whiten = s[7];
    cfg.rosy = s[8];
    cfg.waistSlim = s[9];
    cfg.bustEnhance = s[10];
    cfg.legLengthen = s[11];
    cfg.shoulderSlim = s[12];
    cfg.neckLengthen = s[13];
    cfg.chinShape = s[14];
    cfg.foreheadShape = s[15];
    cfg.noseSlim = s[16];
    cfg.vFace = s[17];
    cfg.faceNarrow = s[18];
    cfg.faceSmall = s[19];
    cfg.cheekboneSlim = s[20];
    cfg.eyeDistance = s[21];
    cfg.eyeCorner = s[22];
    cfg.noseLength = s[23];
    cfg.mouthSize = s[24];
    cfg.mouthSmile = s[25];
    cfg.cranialTop = s[26];
    cfg.darkCircles = s[27];
    cfg.brightEyes = s[28];
    cfg.teethWhiten = s[29];
    cfg.nasolabial = s[30];
    cfg.backgroundBlur = s[31];
    cfg.lutIntensity = s[32];
    cfg.skinTexture = s[33];
    cfg.radiance = s[34];
    cfg.lipstick = s[35];
    cfg.blush = s[36];
    cfg.eyebrow = s[37];
    cfg.eyeshadow = s[38];
    if (n >= 45) {
        cfg.eyeliner = s[39];
        cfg.aegyoSal = s[40];
        cfg.contour = s[41];
        cfg.highlight = s[42];
        cfg.contactLens = s[43];
        cfg.catchlight = s[44];
    }
    if (n >= 51) {
        cfg.relight = s[45];
        cfg.oilControl = s[46];
        cfg.neckTone = s[47];
        cfg.ambientMakeupBlend = s[48];
        cfg.matteTemporal = s[49];
        cfg.blemishRemove = s[50];
    }

    cfg.filter = livebeauty::parseFilterStyle(toStdString(env, filterStr));
    cfg.lutUrl = toStdString(env, lutUrlStr);
    cfg.lipstickColor = livebeauty::parseLipstickColor(toStdString(env, lipColorStr));
    cfg.lipstickFinish = livebeauty::parseLipstickFinish(toStdString(env, lipFinishStr));
    cfg.blushStyle = livebeauty::parseBlushStyle(toStdString(env, blushStyleStr));
    cfg.blushColor = livebeauty::parseBlushColor(toStdString(env, blushColorStr));
    cfg.eyebrowStyle = livebeauty::parseEyebrowStyle(toStdString(env, browStyleStr));
    cfg.eyebrowColor = livebeauty::parseEyebrowColor(toStdString(env, browColorStr));
    cfg.eyeshadowColor = livebeauty::parseEyeshadowColor(toStdString(env, shadowColorStr));
    cfg.contactLensStyle = livebeauty::parseContactLensStyle(toStdString(env, lensStyleStr));
    cfg.backgroundMode = handle->bgModeOverride;
    cfg.virtualBgStrength = handle->virtualBgStrengthOverride;
    cfg.chromaKeyColor = handle->chromaKeyColorOverride;
    cfg.chromaKeySimilarity = handle->chromaKeySimilarityOverride;
    cfg.chromaKeySmoothness = handle->chromaKeySmoothnessOverride;
    cfg.chromaKeyDespill = handle->chromaKeyDespillOverride;
    cfg.maxFaces = handle->maxFacesOverride;
    cfg.frameDelayFrames = handle->frameDelayFramesOverride;
    cfg.relightStyle = handle->relightStyleOverride;
    cfg.performanceTier = handle->performanceTierOverride;
    cfg.hairDye = (n >= 52) ? s[51] : handle->hairDyeOverride;
    cfg.hairDyeColor = handle->hairDyeColorOverride;
    cfg.hairShine = (n >= 53) ? s[52] : handle->hairShineOverride;
    if (n >= 58) {
        cfg.bloom = s[53];
        cfg.filmGrain = s[54];
        cfg.colorTemp = s[55];
        cfg.colorTint = s[56];
        cfg.autoWhiteBalance = s[57];
    }
    cfg.preferHalfFloat = handle->preferHalfFloatOverride;

    handle->engine.setConfig(cfg);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitFaceLandmarks(
    JNIEnv* env, jclass, jlong ptr, jlong sampleTsMs, jfloatArray mpXyArr,
    jint pointCount, jint width, jint height, jlong receivedMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !mpXyArr || pointCount < 468 || pointCount > 1024) return JNI_FALSE;
    if (env->GetArrayLength(mpXyArr) < pointCount * 2) return JNI_FALSE;
    jfloat* data = env->GetFloatArrayElements(mpXyArr, nullptr);
    if (!data) return JNI_FALSE;
    const bool ok = handle->engine.submitFaceLandmarks(sampleTsMs, data, pointCount, width, height, receivedMs);
    env->ReleaseFloatArrayElements(mpXyArr, data, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitFaceMiss(
    JNIEnv*, jclass, jlong ptr, jlong sampleTsMs) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.submitFaceMiss(sampleTsMs);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitBodyPose(
    JNIEnv* env, jclass, jlong ptr, jlong sampleTsMs, jfloatArray poseXyzvArr,
    jint pointCount, jint width, jint height) {
    auto* handle = fromHandle(ptr);
    if (!handle || !poseXyzvArr || pointCount < 33) return JNI_FALSE;
    if (env->GetArrayLength(poseXyzvArr) < 33 * 3) return JNI_FALSE;
    jfloat* raw = env->GetFloatArrayElements(poseXyzvArr, nullptr);
    if (!raw) return JNI_FALSE;
    livebeauty::math::PoseLandmark pts[33];
    for (int i = 0; i < 33; ++i) {
        pts[i].x = raw[i * 3 + 0];
        pts[i].y = raw[i * 3 + 1];
        pts[i].visibility = raw[i * 3 + 2];
    }
    env->ReleaseFloatArrayElements(poseXyzvArr, raw, JNI_ABORT);
    return handle->engine.submitBodyPose(sampleTsMs, pts, 33, width, height) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSelectFaceRoi(
    JNIEnv* env, jclass, jlong ptr, jint width, jint height, jlong nowMs, jfloatArray outRect4) {
    auto* handle = fromHandle(ptr);
    if (!handle || !outRect4 || env->GetArrayLength(outRect4) < 4) return;
    const auto r = handle->engine.selectFaceRoi(width, height, nowMs);
    const jfloat vals[4] = {r.left, r.top, r.width, r.height};
    env->SetFloatArrayRegion(outRect4, 0, 4, vals);
}

JNIEXPORT jint JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeProcessFrame(
    JNIEnv*, jclass, jlong ptr, jint inputGlTex, jint width, jint height,
    jlong frameTsMs, jlong nowMs, jint personMaskGlTex, jfloat lightLuma, jfloat bitrateRatio) {
    auto* handle = fromHandle(ptr);
    if (!handle || inputGlTex == 0) return inputGlTex;
    const auto inTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(inputGlTex));
    const auto maskTex = (personMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(personMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto outTex = handle->engine.processFrame(
        inTex, width, height, frameTsMs, nowMs, maskTex, lightLuma, bitrateRatio);
    return static_cast<jint>(handle->device.unwrapGlTexture(outTex));
}

JNIEXPORT jint JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeProcessFrameEx(
    JNIEnv*, jclass, jlong ptr, jint inputGlTex, jint width, jint height,
    jlong frameTsMs, jlong nowMs, jint personMaskGlTex, jint hairMaskGlTex,
    jlong hairMaskStampMs, jfloat lightLuma, jfloat bitrateRatio) {
    auto* handle = fromHandle(ptr);
    if (!handle || inputGlTex == 0) return inputGlTex;
    const auto inTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(inputGlTex));
    const auto maskTex = (personMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(personMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto hairTex = (hairMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(hairMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto outTex = handle->engine.processFrame(
        inTex, width, height, frameTsMs, nowMs, maskTex, lightLuma, bitrateRatio, hairTex, hairMaskStampMs);
    return static_cast<jint>(handle->device.unwrapGlTexture(outTex));
}

JNIEXPORT jint JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeProcessOesFrame(
    JNIEnv* env, jclass, jlong ptr, jint oesGlTex, jfloatArray stMatrix16,
    jint width, jint height, jlong frameTsMs, jlong nowMs,
    jint personMaskGlTex, jint hairMaskGlTex, jlong hairMaskStampMs,
    jfloat lightLuma, jfloat bitrateRatio) {
    auto* handle = fromHandle(ptr);
    if (!handle || oesGlTex == 0 || width <= 0 || height <= 0) return 0;
    float mat[16];
    const float* matPtr = nullptr;
    if (stMatrix16 && env->GetArrayLength(stMatrix16) >= 16) {
        env->GetFloatArrayRegion(stMatrix16, 0, 16, mat);
        matPtr = mat;
    }
    const auto inTex = handle->device.convertOesTo2D(static_cast<uint32_t>(oesGlTex), width, height, matPtr);
    if (inTex == livebeauty::rhi::kInvalidTexture) return 0;
    const auto maskTex = (personMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(personMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto hairTex = (hairMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(hairMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto outTex = handle->engine.processFrame(
        inTex, width, height, frameTsMs, nowMs, maskTex, lightLuma, bitrateRatio, hairTex, hairMaskStampMs);
    return static_cast<jint>(handle->device.unwrapGlTexture(outTex));
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetPipelineOptions(
    JNIEnv*, jclass, jlong ptr, jint maxFaces, jint frameDelayFrames) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->maxFacesOverride = (maxFaces >= 2) ? 2 : 1;
    handle->frameDelayFramesOverride = (frameDelayFrames < 0) ? 0 : (frameDelayFrames > 4 ? 4 : frameDelayFrames);
    auto cfg = handle->engine.config();
    cfg.maxFaces = handle->maxFacesOverride;
    cfg.frameDelayFrames = handle->frameDelayFramesOverride;
    handle->engine.setConfig(cfg);
}

JNIEXPORT jint JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitMultiFaceLandmarks(
    JNIEnv* env, jclass, jlong ptr, jlong sampleTsMs, jfloatArray flatMpXy,
    jint faceCount, jint pointsPerFace, jint width, jint height, jlong receivedMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !flatMpXy || faceCount <= 0 || pointsPerFace < 468) return 0;
    const int nFaces = (faceCount > livebeauty::math::MultiFaceTracker::kMaxFaces)
        ? livebeauty::math::MultiFaceTracker::kMaxFaces : faceCount;
    const jsize needed = static_cast<jsize>(nFaces * pointsPerFace * 2);
    if (env->GetArrayLength(flatMpXy) < needed) return 0;
    jfloat* data = env->GetFloatArrayElements(flatMpXy, nullptr);
    if (!data) return 0;
    livebeauty::math::MultiFaceTracker::RawFaceInput inputs[livebeauty::math::MultiFaceTracker::kMaxFaces];
    for (int i = 0; i < nFaces; ++i) {
        inputs[i].mpXy = data + static_cast<size_t>(i * pointsPerFace * 2);
        inputs[i].pointCount = pointsPerFace;
    }
    const int accepted = handle->engine.submitMultiFaceLandmarks(
        sampleTsMs, inputs, nFaces, width, height, receivedMs);
    env->ReleaseFloatArrayElements(flatMpXy, data, JNI_ABORT);
    return accepted;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetProOptions(
    JNIEnv* env, jclass, jlong ptr, jstring relightStyleStr, jstring performanceTierStr) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->relightStyleOverride = livebeauty::parseRelightStyle(toStdString(env, relightStyleStr));
    handle->performanceTierOverride = livebeauty::parsePerformanceTier(toStdString(env, performanceTierStr));
    auto cfg = handle->engine.config();
    cfg.relightStyle = handle->relightStyleOverride;
    cfg.performanceTier = handle->performanceTierOverride;
    handle->engine.setConfig(cfg);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetHairStyle(
    JNIEnv* env, jclass, jlong ptr, jfloat hairDye, jstring hairDyeColorStr, jfloat hairShine, jboolean preferHalfFloat) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->hairDyeOverride = hairDye;
    handle->hairDyeColorOverride = livebeauty::parseHairDyeColor(toStdString(env, hairDyeColorStr));
    handle->hairShineOverride = hairShine;
    handle->preferHalfFloatOverride = (preferHalfFloat == JNI_TRUE);
    auto cfg = handle->engine.config();
    cfg.hairDye = handle->hairDyeOverride;
    cfg.hairDyeColor = handle->hairDyeColorOverride;
    cfg.hairShine = handle->hairShineOverride;
    cfg.preferHalfFloat = handle->preferHalfFloatOverride;
    handle->engine.setConfig(cfg);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetHardwareProfile(
    JNIEnv* env, jclass, jlong ptr, jstring gpuRenderer, jint cpuCores, jint totalRamMb) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->engine.setHardwareProfile(toStdString(env, gpuRenderer), cpuCores, totalRamMb);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAmbientRgb(
    JNIEnv*, jclass, jlong ptr, jfloat r, jfloat g, jfloat b) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->engine.setAmbientRgb(r, g, b);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetOcclusionMask(
    JNIEnv*, jclass, jlong ptr, jint occlusionMaskGlTex) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    const auto tex = (occlusionMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(occlusionMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    handle->engine.setOcclusionMask(tex);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAiResidual(
    JNIEnv*, jclass, jlong ptr, jint residualGlTex,
    jfloat roiLeft, jfloat roiTop, jfloat roiWidth, jfloat roiHeight, jfloat strength) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    const auto tex = (residualGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(residualGlTex))
        : livebeauty::rhi::kInvalidTexture;
    livebeauty::CropRect roi{roiLeft, roiTop, roiWidth, roiHeight};
    handle->engine.setAiResidual(tex, roi, strength);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAr3dAccessory(
    JNIEnv*, jclass, jlong ptr,
    jint slot, jboolean enabled, jint anchor, jint texGlId,
    jfloat scale, jfloat offsetX, jfloat offsetY, jfloat offsetZ,
    jfloat yawRad, jfloat pitchRad, jfloat rollRad,
    jfloat opacity, jfloat textureAspect,
    jfloat springStiffness, jfloat springDamping, jboolean headOcclusion) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    livebeauty::math::Ar3dAccessoryConfig cfg{};
    cfg.enabled = (enabled == JNI_TRUE);
    cfg.anchor = static_cast<livebeauty::math::Ar3dAnchor>(anchor);
    cfg.texture = (texGlId != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(texGlId))
        : livebeauty::rhi::kInvalidTexture;
    cfg.scale = scale;
    cfg.offsetX = offsetX;
    cfg.offsetY = offsetY;
    cfg.offsetZ = offsetZ;
    cfg.yawOffsetRad = yawRad;
    cfg.pitchOffsetRad = pitchRad;
    cfg.rollOffsetRad = rollRad;
    cfg.opacity = opacity;
    cfg.textureAspect = textureAspect;
    cfg.springStiffness = springStiffness;
    cfg.springDamping = springDamping;
    cfg.headOcclusion = (headOcclusion == JNI_TRUE);
    handle->engine.setAr3dAccessory(slot, cfg);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAr3dAccessoryEx(
    JNIEnv*, jclass, jlong ptr,
    jint slot, jboolean enabled, jint anchor, jint meshPreset, jint triggerRule, jint texGlId,
    jfloat scale, jfloat offsetX, jfloat offsetY, jfloat offsetZ,
    jfloat yawRad, jfloat pitchRad, jfloat rollRad,
    jfloat opacity, jfloat textureAspect,
    jfloat springStiffness, jfloat springDamping, jboolean headOcclusion) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    livebeauty::math::Ar3dAccessoryConfig cfg{};
    cfg.enabled = (enabled == JNI_TRUE);
    cfg.anchor = static_cast<livebeauty::math::Ar3dAnchor>(std::clamp(static_cast<int>(anchor), 0, 5));
    cfg.meshPreset = static_cast<livebeauty::math::Ar3dMeshPreset>(std::clamp(static_cast<int>(meshPreset), 0, 2));
    cfg.texture = (texGlId != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(texGlId))
        : livebeauty::rhi::kInvalidTexture;
    cfg.scale = scale;
    cfg.offsetX = offsetX;
    cfg.offsetY = offsetY;
    cfg.offsetZ = offsetZ;
    cfg.yawOffsetRad = yawRad;
    cfg.pitchOffsetRad = pitchRad;
    cfg.rollOffsetRad = rollRad;
    cfg.opacity = opacity;
    cfg.textureAspect = textureAspect;
    cfg.springStiffness = springStiffness;
    cfg.springDamping = springDamping;
    cfg.headOcclusion = (headOcclusion == JNI_TRUE);
    const auto trig = static_cast<livebeauty::bundle::BundleTriggerRule>(std::clamp(static_cast<int>(triggerRule), 0, 10));
    handle->engine.setAr3dAccessory(slot, cfg, trig);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeClearAr3dAccessories(
    JNIEnv*, jclass, jlong ptr) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.clearAr3dAccessories();
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitFaceBlendshapes(
    JNIEnv*, jclass, jlong ptr,
    jfloat smile, jfloat jawOpen, jfloat browUp, jfloat blink, jfloat cheekPuff, jfloat tongueOut) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    livebeauty::math::BlendshapeInput bs{smile, jawOpen, browUp, blink, cheekPuff, tongueOut};
    handle->engine.submitFaceBlendshapes(bs);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetExpressionReactive(
    JNIEnv*, jclass, jlong ptr, jboolean enabled) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.setExpressionReactive(enabled == JNI_TRUE);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitAuxRgbaSample(
    JNIEnv* env, jclass, jlong ptr, jobject rgbaDirectBuf, jint srcWidth, jint srcHeight, jfloat retouchStrength) {
    auto* handle = fromHandle(ptr);
    if (!handle || !rgbaDirectBuf || srcWidth <= 0 || srcHeight <= 0 || srcWidth > 8192 || srcHeight > 8192) return JNI_FALSE;
    auto* ptrBytes = static_cast<const uint8_t*>(env->GetDirectBufferAddress(rgbaDirectBuf));
    const jlong cap = env->GetDirectBufferCapacity(rgbaDirectBuf);
    const jlong needed = static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight) * 4;
    if (!ptrBytes || cap < needed) return JNI_FALSE;
    return handle->engine.submitAuxRgbaSample(ptrBytes, srcWidth, srcHeight, retouchStrength) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitFaceParsingClassMap(
    JNIEnv* env, jclass, jlong ptr, jbyteArray classIdsArr, jint srcWidth, jint srcHeight, jlong sampleTsMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !classIdsArr || srcWidth <= 0 || srcHeight <= 0 || srcWidth > 8192 || srcHeight > 8192) return JNI_FALSE;
    const jlong len = static_cast<jlong>(env->GetArrayLength(classIdsArr));
    const jlong needed = static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight);
    if (len < needed) return JNI_FALSE;
    jbyte* raw = env->GetByteArrayElements(classIdsArr, nullptr);
    if (!raw) return JNI_FALSE;
    const bool ok = handle->engine.submitFaceParsingClassMap(
        reinterpret_cast<const uint8_t*>(raw), srcWidth, srcHeight, sampleTsMs);
    env->ReleaseByteArrayElements(classIdsArr, raw, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitNeuralResidualTensor(
    JNIEnv* env, jclass, jlong ptr, jfloatArray tensorNchwArr, jint channels, jint srcWidth, jint srcHeight,
    jfloat roiLeft, jfloat roiTop, jfloat roiWidth, jfloat roiHeight, jfloat strength) {
    auto* handle = fromHandle(ptr);
    if (!handle || !tensorNchwArr || (channels != 3 && channels != 4) ||
        srcWidth <= 0 || srcHeight <= 0 || srcWidth > 8192 || srcHeight > 8192) {
        return JNI_FALSE;
    }
    const jlong len = static_cast<jlong>(env->GetArrayLength(tensorNchwArr));
    const jlong needed = static_cast<jlong>(channels) * static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight);
    if (len < needed) return JNI_FALSE;
    jfloat* raw = env->GetFloatArrayElements(tensorNchwArr, nullptr);
    if (!raw) return JNI_FALSE;
    livebeauty::CropRect roi{roiLeft, roiTop, roiWidth, roiHeight};
    const bool ok = handle->engine.submitNeuralResidualTensor(raw, channels, srcWidth, srcHeight, roi, strength);
    env->ReleaseFloatArrayElements(tensorNchwArr, raw, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

static jboolean applyBundleInternal(
    JNIEnv* env, JniEngineHandle* handle, jstring manifestJsonStr,
    jobjectArray assetPathsArr, jintArray texGlIdsArr) {
    try {
        const std::string json = toStdString(env, manifestJsonStr);
        std::vector<std::pair<std::string, uint32_t>> texTable;
        if (assetPathsArr && texGlIdsArr) {
            const jsize nPaths = env->GetArrayLength(assetPathsArr);
            const jsize nIds = env->GetArrayLength(texGlIdsArr);
            const jsize n = std::min(nPaths, nIds);
            if (n > 0) {
                jint* ids = env->GetIntArrayElements(texGlIdsArr, nullptr);
                if (ids) {
                    texTable.reserve(static_cast<size_t>(n));
                    for (jsize i = 0; i < n; ++i) {
                        auto jPath = static_cast<jstring>(env->GetObjectArrayElement(assetPathsArr, i));
                        if (jPath) {
                            texTable.emplace_back(toStdString(env, jPath), static_cast<uint32_t>(ids[i]));
                            env->DeleteLocalRef(jPath);
                        }
                    }
                    env->ReleaseIntArrayElements(texGlIdsArr, ids, JNI_ABORT);
                }
            }
        }
        livebeauty::bundle::BundleTextureResolver resolver = [&](const std::string& path) -> livebeauty::rhi::TextureId {
            for (const auto& kv : texTable) {
                if (kv.first == path && kv.second != 0) {
                    return handle->device.wrapExternalTexture(kv.second);
                }
            }
            return livebeauty::rhi::kInvalidTexture;
        };
        const bool ok = handle->engine.applyEffectBundleJson(json, resolver, nullptr);
        if (ok) {
            const auto& cfg = handle->engine.config();
            handle->hairDyeOverride = cfg.hairDye;
            handle->hairDyeColorOverride = cfg.hairDyeColor;
            handle->hairShineOverride = cfg.hairShine;
            handle->relightStyleOverride = cfg.relightStyle;
            handle->bgModeOverride = cfg.backgroundMode;
            handle->virtualBgStrengthOverride = cfg.virtualBgStrength;
            handle->chromaKeyColorOverride = cfg.chromaKeyColor;
            handle->chromaKeySimilarityOverride = cfg.chromaKeySimilarity;
            handle->chromaKeySmoothnessOverride = cfg.chromaKeySmoothness;
            handle->chromaKeyDespillOverride = cfg.chromaKeyDespill;
            handle->performanceTierOverride = cfg.performanceTier;
            handle->preferHalfFloatOverride = cfg.preferHalfFloat;
            handle->maxFacesOverride = cfg.maxFaces;
            handle->frameDelayFramesOverride = cfg.frameDelayFrames;
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeApplyEffectBundleJson(
    JNIEnv* env, jclass, jlong ptr, jstring manifestJsonStr) {
    auto* handle = fromHandle(ptr);
    if (!handle || !manifestJsonStr) return JNI_FALSE;
    return applyBundleInternal(env, handle, manifestJsonStr, nullptr, nullptr);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeApplyEffectBundleJsonWithAssets(
    JNIEnv* env, jclass, jlong ptr, jstring manifestJsonStr,
    jobjectArray assetPathsArr, jintArray texGlIdsArr) {
    auto* handle = fromHandle(ptr);
    if (!handle || !manifestJsonStr) return JNI_FALSE;
    return applyBundleInternal(env, handle, manifestJsonStr, assetPathsArr, texGlIdsArr);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeGetStats(
    JNIEnv* env, jclass, jlong ptr, jfloatArray outStats) {
    auto* handle = fromHandle(ptr);
    if (!handle || !outStats) return;
    const jsize len = env->GetArrayLength(outStats);
    if (len < 8) return;
    const auto& st = handle->engine.stats();
    jfloat vals[11] = {
        st.lastFrameMs,
        st.emaFrameMs,
        st.fps,
        static_cast<float>(st.loadLevel),
        st.facePresence,
        st.hairShown,
        static_cast<float>(st.activePassesMask),
        static_cast<float>(st.frameCount),
        static_cast<float>(st.activeFaces),
        static_cast<float>(st.displayedTimestampMs >= 0 ? st.displayedTimestampMs : 0),
        static_cast<float>(st.resolvedTier),
    };
    const jsize count = (len >= 11) ? 11 : ((len >= 10) ? 10 : 8);
    env->SetFloatArrayRegion(outStats, 0, count, vals);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetChromaKey(
    JNIEnv* env, jclass, jlong ptr,
    jstring keyColorStr, jfloat similarity, jfloat smoothness, jfloat despill) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->chromaKeyColorOverride = livebeauty::parseChromaKeyColor(toStdString(env, keyColorStr));
    handle->chromaKeySimilarityOverride = std::clamp(similarity, 0.05f, 0.95f);
    handle->chromaKeySmoothnessOverride = std::clamp(smoothness, 0.01f, 0.80f);
    handle->chromaKeyDespillOverride = std::clamp(despill, 0.0f, 1.0f);
    auto cfg = handle->engine.config();
    cfg.chromaKeyColor = handle->chromaKeyColorOverride;
    cfg.chromaKeySimilarity = handle->chromaKeySimilarityOverride;
    cfg.chromaKeySmoothness = handle->chromaKeySmoothnessOverride;
    cfg.chromaKeyDespill = handle->chromaKeyDespillOverride;
    handle->engine.setConfig(cfg);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitMulticlassLogits(
    JNIEnv* env, jclass, jlong ptr, jfloatArray logitsNhwc6Arr,
    jint srcWidth, jint srcHeight, jlong sampleTsMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !logitsNhwc6Arr || srcWidth <= 0 || srcHeight <= 0 || srcWidth > 2048 || srcHeight > 2048) {
        return JNI_FALSE;
    }
    const jlong len = static_cast<jlong>(env->GetArrayLength(logitsNhwc6Arr));
    const jlong needed = static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight) * 6LL;
    if (len < needed) return JNI_FALSE;
    jfloat* raw = env->GetFloatArrayElements(logitsNhwc6Arr, nullptr);
    if (!raw) return JNI_FALSE;
    const bool ok = handle->engine.submitMulticlass6Logits(raw, srcWidth, srcHeight, sampleTsMs);
    env->ReleaseFloatArrayElements(logitsNhwc6Arr, raw, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeTriggerGesture(
    JNIEnv*, jclass, jlong ptr, jint gestureId, jfloat normX, jfloat normY, jlong nowMs) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    const int clamped = std::clamp(static_cast<int>(gestureId), 0, 4);
    handle->engine.triggerGesture(
        static_cast<livebeauty::math::GestureType>(clamped), normX, normY, nowMs);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetHandShield(
    JNIEnv*, jclass, jlong ptr, jfloat hand0X, jfloat hand0Y, jfloat hand1X, jfloat hand1Y) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->engine.setHandShieldCoords(hand0X, hand0Y, hand1X, hand1Y);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitBodyMiss(
    JNIEnv*, jclass, jlong ptr, jlong sampleTsMs) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.submitBodyMiss(sampleTsMs);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetHostLoadLevel(
    JNIEnv*, jclass, jlong ptr, jint level) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.setHostLoadLevel(static_cast<int>(level));
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativePrewarm(
    JNIEnv*, jclass, jlong ptr, jint startIdx, jint count) {
    auto* handle = fromHandle(ptr);
    if (handle) handle->engine.prewarmShaders(static_cast<int>(startIdx), static_cast<int>(count));
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeProcessFrameToFbo(
    JNIEnv*, jclass, jlong ptr, jint inputGlTex, jint width, jint height,
    jlong frameTsMs, jlong nowMs, jint personMaskGlTex, jint hairMaskGlTex,
    jlong hairMaskStampMs, jfloat lightLuma, jfloat bitrateRatio, jint dstFbo) {
    auto* handle = fromHandle(ptr);
    if (!handle || inputGlTex == 0 || dstFbo < 0) return JNI_FALSE;
    const auto inTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(inputGlTex));
    const auto maskTex = (personMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(personMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto hairTex = (hairMaskGlTex != 0)
        ? handle->device.wrapExternalTexture(static_cast<uint32_t>(hairMaskGlTex))
        : livebeauty::rhi::kInvalidTexture;
    const auto outRt = handle->device.wrapExternalFramebuffer(static_cast<uint32_t>(dstFbo), width, height);
    const auto outTex = handle->engine.processFrame(
        inTex, width, height, frameTsMs, nowMs, maskTex, lightLuma, bitrateRatio, hairTex, hairMaskStampMs, outRt);
    // When beauty actually processed and wrote to outRt, textureOf(outRt) == kInvalidTexture (outTex != inTex).
    return (outTex != inTex) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitMulticlassLogitsDirect(
    JNIEnv* env, jclass, jlong ptr, jobject logitsDirectBuf,
    jint srcWidth, jint srcHeight, jint padX, jint padY, jint validW, jint validH, jlong sampleTsMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !logitsDirectBuf || srcWidth <= 0 || srcHeight <= 0 ||
        srcWidth > 2048 || srcHeight > 2048 || validW <= 0 || validH <= 0) {
        return JNI_FALSE;
    }
    auto* raw = static_cast<const float*>(env->GetDirectBufferAddress(logitsDirectBuf));
    const jlong capBytes = env->GetDirectBufferCapacity(logitsDirectBuf);
    const jlong neededBytes = static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight) * 6LL * static_cast<jlong>(sizeof(float));
    if (!raw || capBytes < neededBytes) return JNI_FALSE;
    const bool ok = handle->engine.submitMulticlass6LogitsRegion(
        raw, srcWidth, srcHeight, padX, padY, validW, validH, sampleTsMs);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetCameraShaking(
    JNIEnv*, jclass, jlong ptr, jboolean shaking) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->engine.setCameraShaking(shaking == JNI_TRUE);
}

}  // extern "C"
