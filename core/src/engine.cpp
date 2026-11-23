#include "livebeauty/engine.h"

#include <algorithm>
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
    hairGate_.reset();
    pipeline_.resetTemporal();
    hasPendingBody_ = false;
    pendingBodyTs_ = kTimeInvalid;
}

void LiveBeautyEngine::setSticker(int slot, const math::StickerLayerConfig& layer) {
    if (slot < 0 || slot >= math::kMaxStickerLayers) return;
    stickers_[slot] = layer;
    int maxActive = 0;
    for (int i = 0; i < math::kMaxStickerLayers; ++i) {
        if (stickers_[i].enabled && stickers_[i].texture != rhi::kInvalidTexture) {
            maxActive = i + 1;
        }
    }
    stickerCount_ = maxActive;
}

void LiveBeautyEngine::clearStickers() {
    for (auto& s : stickers_) s = math::StickerLayerConfig{};
    stickerCount_ = 0;
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
    float bitrateRatio,
    rhi::TextureId hairMaskTex,
    int64_t hairMaskStampMs) {
    if (width <= 0 || height <= 0 || inputTexture == rhi::kInvalidTexture) return inputTexture;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    if (width != lastWidth_ || height != lastHeight_) {
        timeline_.setAspect(aspect);
        lastWidth_ = width;
        lastHeight_ = height;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const bool processing = config_.isProcessing() || (config_.enabled && !config_.compareOriginal && stickerCount_ > 0);
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

    // Hair semantic temporal gate & rigid head-motion compensation affine
    const bool hasHairTex = (hairMaskTex != rhi::kInvalidTexture);
    const int64_t effHairStamp = (hairMaskStampMs != kTimeInvalid) ? hairMaskStampMs : frameTimestampMs;
    const float hairLevel = hairGate_.update(
        hasHairTex, effHairStamp, frameTimestampMs,
        loadFades_.body(), timeline_.latest(), timeline_.missSince(), nowMs);
    const float neckWeight = (bodyState != nullptr && config_.neckLengthen > 0.0f) ? bodySmoother_.presence() : 0.0f;
    const float hairShown = hasHairTex ? hairGate_.shown(neckWeight) : 0.0f;
    const float hairBgShown = hasHairTex ? hairLevel : 0.0f;

    std::array<float, math::HairMaskTuning::kAffineSize> hairAffine{};
    math::HairMaskTuning::identity(hairAffine.data());
    if (hasHairTex && hairLevel > 1e-4f && state41 != nullptr) {
        float poseDisplay[math::HairMaskTuning::kPoseSize];
        for (int i = 0; i < math::HairMaskTuning::kPoseSize; ++i) {
            poseDisplay[i] = state41[22 + i];
        }
        float poseMask[math::HairMaskTuning::kPoseSize];
        const int64_t newest = timeline_.latest();
        bool hasMaskPose = false;
        if (newest != kTimeInvalid) {
            if (effHairStamp <= newest) {
                hasMaskPose = timeline_.poseAt(effHairStamp, poseMask);
            } else if (timeline_.poseAt(newest, poseMask)) {
                if (frameTimestampMs > newest) {
                    const float g = std::min(1.0f,
                        static_cast<float>(effHairStamp - newest) / static_cast<float>(frameTimestampMs - newest));
                    for (int i = 0; i < math::HairMaskTuning::kPoseSize; ++i) {
                        poseMask[i] += g * (poseDisplay[i] - poseMask[i]);
                    }
                }
                hasMaskPose = true;
            }
        }
        if (hasMaskPose) {
            math::HairMaskTuning::affine(poseMask, poseDisplay, aspect, hairAffine.data());
        }
    }

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
    in.hairMaskTex = hairMaskTex;
    in.hairShown = hairShown;
    in.hairBgShown = hairBgShown;
    in.hairAffine = hairAffine;
    in.stickers = stickers_;
    in.stickerCount = stickerCount_;
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

    // Update real-time telemetry
    const float frameMs = static_cast<float>(workNs) * 1e-6f;
    stats_.lastFrameMs = frameMs;
    stats_.emaFrameMs = (stats_.frameCount == 0) ? frameMs : (stats_.emaFrameMs * 0.9f + frameMs * 0.1f);
    if (lastStatsNowMs_ != kTimeInvalid && nowMs > lastStatsNowMs_) {
        const float instFps = 1000.0f / static_cast<float>(nowMs - lastStatsNowMs_);
        stats_.fps = (stats_.fps <= 0.0f) ? instFps : (stats_.fps * 0.9f + instFps * 0.1f);
    }
    lastStatsNowMs_ = nowMs;
    stats_.loadLevel = governor_.level();
    stats_.facePresence = timeline_.presence();
    stats_.hairShown = hairShown;
    stats_.activePassesMask = pipeline_.lastActivePassesMask();
    stats_.frameCount++;

    return outTex;
}

}  // namespace livebeauty
