#include <jni.h>

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

    JniEngineHandle() : device(), engine(device) {}
};

inline JniEngineHandle* fromHandle(jlong ptr) {
    return reinterpret_cast<JniEngineHandle*>(ptr);
}

std::string toStdString(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* chars = env->GetStringUTFChars(js, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(js, chars);
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeCreate(JNIEnv*, jclass) {
    auto* handle = new JniEngineHandle();
    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeDestroy(JNIEnv*, jclass, jlong ptr, jboolean contextAlive) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    handle->device.releaseAll(contextAlive == JNI_TRUE);
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
    if (contextAlive == JNI_TRUE) {
        handle->engine.releaseGpuResources();
    }
    handle->device.releaseAll(contextAlive == JNI_TRUE);
}

JNIEXPORT void JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSetAssets(
    JNIEnv*, jclass, jlong ptr,
    jint lutGlTex, jint blushGlTex, jint eyebrowGlTex,
    jint eyeMakeupGlTex, jint contourHighlightGlTex, jint irisGlTex) {
    auto* handle = fromHandle(ptr);
    if (!handle) return;
    livebeauty::rhi::PipelineAssets assets{};
    assets.lutTex = handle->device.wrapExternalTexture(static_cast<uint32_t>(lutGlTex));
    assets.blushAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(blushGlTex));
    assets.eyebrowAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(eyebrowGlTex));
    assets.eyeMakeupAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(eyeMakeupGlTex));
    assets.contourHighlightAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(contourHighlightGlTex));
    assets.irisAtlas = handle->device.wrapExternalTexture(static_cast<uint32_t>(irisGlTex));
    handle->engine.setAssets(assets);
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
    jfloat s[48];
    env->GetFloatArrayRegion(scalars, 0, n < 48 ? n : 48, s);

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

    handle->engine.setConfig(cfg);
}

JNIEXPORT jboolean JNICALL
Java_io_bebebus_livebeauty_NativeBeautyBridge_nativeSubmitFaceLandmarks(
    JNIEnv* env, jclass, jlong ptr, jlong sampleTsMs, jfloatArray mpXyArr,
    jint pointCount, jint width, jint height, jlong receivedMs) {
    auto* handle = fromHandle(ptr);
    if (!handle || !mpXyArr || pointCount < 468) return JNI_FALSE;
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

}  // extern "C"
