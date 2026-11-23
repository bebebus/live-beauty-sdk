#include "livebeauty/engine.h"

#include <algorithm>
#include <chrono>

namespace livebeauty {

LiveBeautyEngine::LiveBeautyEngine(rhi::IRhiDevice& device) : pipeline_(device) {}

void LiveBeautyEngine::setConfig(const BeautyConfig& config) {
    config_ = config;
}

void LiveBeautyEngine::resetTracking() {
    faces_.reset();
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
    math::MultiFaceTracker::RawFaceInput raw{mpXy, pointCount};
    return submitMultiFaceLandmarks(sampleTsMs, &raw, 1, frameWidth, frameHeight, receivedMs) > 0;
}

int LiveBeautyEngine::submitMultiFaceLandmarks(
    int64_t sampleTsMs, const math::MultiFaceTracker::RawFaceInput* faces, int faceCount,
    int frameWidth, int frameHeight, int64_t receivedMs) {
    if (frameWidth <= 0 || frameHeight <= 0) return 0;
    const float aspect = static_cast<float>(frameWidth) / static_cast<float>(frameHeight);
    if (frameWidth != lastWidth_ || frameHeight != lastHeight_) {
        faces_.setAspect(aspect);
        lastWidth_ = frameWidth;
        lastHeight_ = frameHeight;
    }

    const int cappedCount = std::min(faceCount, std::clamp(config_.maxFaces, 1, math::MultiFaceTracker::kMaxFaces));
    float cx = 0.0f, cy = 0.0f, fw = 0.0f;
    const int valid = faces_.submitFaces(
        sampleTsMs, faces, cappedCount, frameWidth, frameHeight, receivedMs, &cx, &cy, &fw);
    if (valid > 0 && fw > 0.0f && cappedCount == 1) {
        roiSelector_.onFace(cx, cy, fw, sampleTsMs, receivedMs);
    } else if (valid == 0 || cappedCount > 1) {
        // Multi-face mode uses full frame so secondary faces outside primary ROI remain visible
        roiSelector_.onMiss();
    }
    return valid;
}

void LiveBeautyEngine::submitFaceMiss(int64_t sampleTsMs) {
    faces_.submitAllMiss(sampleTsMs);
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
    if (config_.maxFaces >= 2) return CropRect::full();
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
        faces_.setAspect(aspect);
        lastWidth_ = width;
        lastHeight_ = height;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const bool processing = config_.isProcessing() || (config_.enabled && !config_.compareOriginal && stickerCount_ > 0);
    loadFades_.update(governor_.level(), nowMs);

    // Optional cross-backend FrameDelayRing (0..4 frames; aligns displayed camera frame with async MediaPipe landmarks)
    rhi::TextureId displayTex = inputTexture;
    int64_t displayTsMs = frameTimestampMs;
    const int delayFrames = processing ? std::clamp(config_.frameDelayFrames, 0, 4) : 0;
    if (delayFrames > 0) {
        pipeline_.pushFrameDelay(inputTexture, frameTimestampMs, delayFrames, width, height, displayTex, displayTsMs);
    }

    faces_.update(nowMs);
    auto& primaryTl = faces_.slot(0);
    if (primaryTl.runId() != deformerRunId_) {
        deformerRunId_ = primaryTl.runId();
        pipeline_.resetDeformerTemporal();
    }
    const float* state41 = primaryTl.at(displayTsMs);
    const float* dilatedState41 = (state41 != nullptr)
        ? primaryTl.dilate(state41, dilatedState_.data())
        : nullptr;
    math::FaceTimeline::SmoothedMeshView meshView{};
    const bool hasMesh = primaryTl.meshAt(displayTsMs, meshView);
    if (hasMesh) {
        guidedSpread_.update(meshView.faceWidth() * static_cast<float>(width));
    }

    // Optional secondary face (slot 1)
    auto& secondaryTl = faces_.slot(1);
    const float* secState41 = (config_.maxFaces >= 2) ? secondaryTl.at(displayTsMs) : nullptr;
    math::FaceTimeline::SmoothedMeshView secMeshView{};
    const bool hasSecMesh = (config_.maxFaces >= 2) && secondaryTl.meshAt(displayTsMs, secMeshView);

    const float* bodyState = bodySmoother_.update(
        hasPendingBody_ ? pendingBodyState_.data() : nullptr, pendingBodyTs_, nowMs);

    // Hair semantic temporal gate & rigid head-motion compensation affine (using aligned displayTsMs)
    const bool hasHairTex = (hairMaskTex != rhi::kInvalidTexture);
    const int64_t effHairStamp = (hairMaskStampMs != kTimeInvalid) ? hairMaskStampMs : displayTsMs;
    const float hairLevel = hairGate_.update(
        hasHairTex, effHairStamp, displayTsMs,
        loadFades_.body(), primaryTl.latest(), primaryTl.missSince(), nowMs);
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
        const int64_t newest = primaryTl.latest();
        bool hasMaskPose = false;
        if (newest != kTimeInvalid) {
            if (effHairStamp <= newest) {
                hasMaskPose = primaryTl.poseAt(effHairStamp, poseMask);
            } else if (primaryTl.poseAt(newest, poseMask)) {
                if (displayTsMs > newest) {
                    const float g = std::min(1.0f,
                        static_cast<float>(effHairStamp - newest) / static_cast<float>(displayTsMs - newest));
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
    in.inputTexture = displayTex;
    in.width = width;
    in.height = height;
    in.nowMs = nowMs;
    in.hasFaceState = (state41 != nullptr);
    in.faceState41 = state41;
    in.dilatedFaceState41 = dilatedState41;
    in.hasFaceMesh = hasMesh;
    in.landmarks478x2 = hasMesh ? meshView.landmarks : nullptr;
    in.faceMeta15 = hasMesh ? meshView.meta : nullptr;
    in.hasIris = hasMesh && meshView.hasIris;
    in.facePresence = primaryTl.presence();
    in.steadyFace = primaryTl.steadyFace();
    in.steadyEye = primaryTl.steadyEye();
    in.eyeMakeupKeep = primaryTl.eyeMakeupKeep();
    in.denoiseKeep = primaryTl.denoiseKeep();

    in.secondaryFace.hasState = (secState41 != nullptr);
    in.secondaryFace.state41 = secState41;
    in.secondaryFace.hasMesh = hasSecMesh;
    in.secondaryFace.landmarks478x2 = hasSecMesh ? secMeshView.landmarks : nullptr;
    in.secondaryFace.meta15 = hasSecMesh ? secMeshView.meta : nullptr;
    in.secondaryFace.hasIris = hasSecMesh && secMeshView.hasIris;
    in.secondaryFace.presence = (config_.maxFaces >= 2) ? secondaryTl.presence() : 0.0f;
    in.secondaryFace.steadyFace = secondaryTl.steadyFace();
    in.secondaryFace.steadyEye = secondaryTl.steadyEye();
    in.secondaryFace.eyeMakeupKeep = secondaryTl.eyeMakeupKeep();

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

    const float frameMs = static_cast<float>(workNs) * 1e-6f;
    stats_.lastFrameMs = frameMs;
    stats_.emaFrameMs = (stats_.frameCount == 0) ? frameMs : (stats_.emaFrameMs * 0.9f + frameMs * 0.1f);
    if (lastStatsNowMs_ != kTimeInvalid && nowMs > lastStatsNowMs_) {
        const float instFps = 1000.0f / static_cast<float>(nowMs - lastStatsNowMs_);
        stats_.fps = (stats_.fps <= 0.0f) ? instFps : (stats_.fps * 0.9f + instFps * 0.1f);
    }
    lastStatsNowMs_ = nowMs;
    stats_.loadLevel = governor_.level();
    stats_.facePresence = std::max(primaryTl.presence(), (config_.maxFaces >= 2) ? secondaryTl.presence() : 0.0f);
    stats_.hairShown = hairShown;
    stats_.activePassesMask = pipeline_.lastActivePassesMask() | (delayFrames > 0 ? rhi::kPassFrameDelay : 0u);
    stats_.frameCount++;
    stats_.activeFaces = faces_.activeFaceCount();
    stats_.displayedTimestampMs = displayTsMs;

    return outTex;
}

}  // namespace livebeauty
