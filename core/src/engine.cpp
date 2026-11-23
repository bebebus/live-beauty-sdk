#include "livebeauty/engine.h"

#include <chrono>

namespace livebeauty {

LiveBeautyEngine::LiveBeautyEngine(rhi::IRhiDevice& device) : pipeline_(device) {}

void LiveBeautyEngine::setConfig(const BeautyConfig& config) {
    config_ = config;
}

void LiveBeautyEngine::resetTracking() {
    timeline_.reset();
    roiSelector_.reset();
    bodySmoother_.reset();
    guidedSpread_.reset();
    pipeline_.resetTemporal();
    hasPendingBody_ = false;
    pendingBodyTs_ = kTimeInvalid;
}

bool LiveBeautyEngine::submitFaceLandmarks(
    int64_t sampleTsMs, const float* mpXy, int pointCount,
    int frameWidth, int frameHeight, int64_t receivedMs) {
    if (frameWidth <= 0 || frameHeight <= 0) return false;
    const float aspect = static_cast<float>(frameWidth) / static_cast<float>(frameHeight);
    if (frameWidth != lastWidth_ || frameHeight != lastHeight_) {
        timeline_.setAspect(aspect);
        lastWidth_ = frameWidth;
        lastHeight_ = frameHeight;
    }

    math::FaceMeshFrame mesh{};
    if (!math::FaceMeshFrame::fromMediaPipe(mpXy, pointCount, aspect, mesh)) {
        submitFaceMiss(sampleTsMs);
        return false;
    }

    float state41[math::FaceGeometryConstants::kStateSize];
    if (!math::FaceMeshFrame::faceGeometryFromMediaPipe(mpXy, pointCount, aspect, state41)) {
        if (!math::FaceMeshFrame::fillGeometryState(
                mesh.landmarks.data(), math::FaceMeshFrame::kCoordCount,
                mesh.axisX(), mesh.axisY(), aspect, 0.0f, state41)) {
            submitFaceMiss(sampleTsMs);
            return false;
        }
    }

    timeline_.add(sampleTsMs, state41, &mesh);
    roiSelector_.onFace(mesh.centerX(), mesh.centerY(), mesh.faceWidth(), sampleTsMs, receivedMs);
    return true;
}

void LiveBeautyEngine::submitFaceMiss(int64_t sampleTsMs) {
    timeline_.addMiss(sampleTsMs);
    roiSelector_.onMiss();
}

bool LiveBeautyEngine::submitBodyPose(
    int64_t sampleTsMs, const math::PoseLandmark* pose33, int count,
    int frameWidth, int frameHeight) {
    if (frameWidth <= 0 || frameHeight <= 0) return false;
    const float aspect = static_cast<float>(frameWidth) / static_cast<float>(frameHeight);
    math::BodyGeometry bg{};
    if (!math::BodyGeometry::fromPose(pose33, count, aspect, bg)) {
        hasPendingBody_ = false;
        return false;
    }
    bg.toState(pendingBodyState_.data());
    pendingBodyTs_ = sampleTsMs;
    hasPendingBody_ = true;
    return true;
}

void LiveBeautyEngine::submitBodyMiss(int64_t /*sampleTsMs*/) {
    hasPendingBody_ = false;
}

CropRect LiveBeautyEngine::selectFaceRoi(int frameWidth, int frameHeight, int64_t nowMs) {
    return roiSelector_.select(frameWidth, frameHeight, nowMs);
}

rhi::TextureId LiveBeautyEngine::processFrame(
    rhi::TextureId inputTexture,
    int width, int height,
    int64_t frameTimestampMs,
    int64_t nowMs,
    rhi::TextureId personMaskTex,
    float lightLuma,
    float bitrateRatio) {
    if (width <= 0 || height <= 0 || inputTexture == rhi::kInvalidTexture) return inputTexture;
    if (width != lastWidth_ || height != lastHeight_) {
        timeline_.setAspect(static_cast<float>(width) / static_cast<float>(height));
        lastWidth_ = width;
        lastHeight_ = height;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const bool processing = config_.isProcessing();
    loadFades_.update(governor_.level(), nowMs);

    timeline_.update(nowMs);
    const float* state41 = timeline_.at(frameTimestampMs);
    math::FaceTimeline::SmoothedMeshView meshView{};
    const bool hasMesh = timeline_.meshAt(frameTimestampMs, meshView);
    if (hasMesh) {
        guidedSpread_.update(meshView.faceWidth() * static_cast<float>(width));
    }

    const float* bodyState = bodySmoother_.update(
        hasPendingBody_ ? pendingBodyState_.data() : nullptr, pendingBodyTs_, nowMs);

    rhi::FrameInputs in{};
    in.inputTexture = inputTexture;
    in.width = width;
    in.height = height;
    in.nowMs = nowMs;
    in.hasFaceState = (state41 != nullptr);
    in.faceState41 = state41;
    in.hasFaceMesh = hasMesh;
    in.landmarks478x2 = hasMesh ? meshView.landmarks : nullptr;
    in.faceMeta15 = hasMesh ? meshView.meta : nullptr;
    in.hasIris = hasMesh && meshView.hasIris;
    in.facePresence = timeline_.presence();
    in.steadyFace = timeline_.steadyFace();
    in.steadyEye = timeline_.steadyEye();
    in.eyeMakeupKeep = timeline_.eyeMakeupKeep();
    in.denoiseKeep = timeline_.denoiseKeep();
    in.hasBody = (bodyState != nullptr);
    in.bodyState7 = bodyState;
    in.bodyPresence = bodySmoother_.presence();
    in.personMaskTex = personMaskTex;
    in.loadFadeBody = loadFades_.body();
    in.loadFadeFace = loadFades_.face();
    in.loadFadeSmooth = loadFades_.smooth();
    in.guidedSpread = guidedSpread_.value();
    in.lightLuma = lightLuma;
    in.bitratePressure = math::BitrateEffectsTuning::pressure(bitrateRatio);

    const rhi::TextureId outTex = pipeline_.render(config_, in);

    const auto t1 = std::chrono::steady_clock::now();
    const int64_t workNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    governor_.onFrame(frameTimestampMs * 1000000LL, workNs, processing, nowMs);
    return outTex;
}

}  // namespace livebeauty
