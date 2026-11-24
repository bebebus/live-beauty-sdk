#include "livebeauty/engine.h"

#include <algorithm>
#include <chrono>

namespace livebeauty {

LiveBeautyEngine::LiveBeautyEngine(rhi::IRhiDevice& device)
    : device_(device), pipeline_(device) {}

LiveBeautyEngine::~LiveBeautyEngine() {
    releaseGpuResources();
}

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
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        occlusionSolver_.reset();
        faceParsingMapper_.reset();
        expressionBus_.reset();
        hasLatestFaceBounds_ = false;
        latestFaceRollRad_ = 0.0f;
        semanticOccActive_ = false;
        stagedOccDirty_ = false;
        stagedParseHairDirty_ = false;
        stagedParseHairTsMs_ = kTimeInvalid;
        stagedResidualDirty_ = false;
        stagedResidualActive_ = false;
    }
    autoResidualStrength_ = 0.0f;
    autoParseHairValid_ = false;
    autoParseHairTsMs_ = kTimeInvalid;
}

void LiveBeautyEngine::releaseGpuResources() {
    if (autoOccTex_ != rhi::kInvalidTexture) {
        device_.destroyTexture(autoOccTex_);
        autoOccTex_ = rhi::kInvalidTexture;
        autoOccTexSize_ = 0;
    }
    if (autoParseHairTex_ != rhi::kInvalidTexture) {
        device_.destroyTexture(autoParseHairTex_);
        autoParseHairTex_ = rhi::kInvalidTexture;
    }
    autoParseHairValid_ = false;
    autoParseHairTsMs_ = kTimeInvalid;
    if (autoResidualTex_ != rhi::kInvalidTexture) {
        device_.destroyTexture(autoResidualTex_);
        autoResidualTex_ = rhi::kInvalidTexture;
    }
    autoResidualStrength_ = 0.0f;
    pipeline_.releaseGpuResources();
}

void LiveBeautyEngine::setSticker(
    int slot, const math::StickerLayerConfig& layer, bundle::BundleTriggerRule trigger) {
    if (slot < 0 || slot >= math::kMaxStickerLayers) return;
    stickers_[slot] = layer;
    stickerTriggers_[slot] = trigger;
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
    stickerTriggers_.fill(bundle::BundleTriggerRule::kAlways);
    stickerCount_ = 0;
}

void LiveBeautyEngine::setAr3dAccessory(
    int slot, const math::Ar3dAccessoryConfig& layer, bundle::BundleTriggerRule trigger) {
    if (slot < 0 || slot >= math::kMaxAr3dLayers) return;
    math::Ar3dAccessoryConfig stored = layer;
    const auto* candVerts = !layer.ownedVertices.empty() ? layer.ownedVertices.data() : layer.customVertices;
    const int candVertCount = !layer.ownedVertices.empty() ? static_cast<int>(layer.ownedVertices.size()) : layer.customVertexCount;
    const auto* candIdx = !layer.ownedIndices.empty() ? layer.ownedIndices.data() : layer.customIndices;
    const int candIdxCount = !layer.ownedIndices.empty() ? static_cast<int>(layer.ownedIndices.size()) : layer.customIndexCount;
    if (candVerts != nullptr || candIdx != nullptr || candVertCount > 0 || candIdxCount > 0) {
        if (!math::Ar3dSolver::validateCustomMesh(candVerts, candVertCount, candIdx, candIdxCount)) {
            ar3dAccessories_[slot] = math::Ar3dAccessoryConfig{};
            ar3dTriggers_[slot] = bundle::BundleTriggerRule::kAlways;
        } else {
            stored.ownedVertices.assign(candVerts, candVerts + candVertCount);
            stored.ownedIndices.assign(candIdx, candIdx + candIdxCount);
            stored.customVertices = stored.ownedVertices.data();
            stored.customVertexCount = candVertCount;
            stored.customIndices = stored.ownedIndices.data();
            stored.customIndexCount = candIdxCount;
            stored.customMeshGeneration = ++nextAr3dMeshGen_;
            ar3dAccessories_[slot] = std::move(stored);
            ar3dTriggers_[slot] = trigger;
        }
    } else {
        stored.ownedVertices.clear();
        stored.ownedIndices.clear();
        stored.customVertices = nullptr;
        stored.customVertexCount = 0;
        stored.customIndices = nullptr;
        stored.customIndexCount = 0;
        stored.customMeshGeneration = 0;
        ar3dAccessories_[slot] = std::move(stored);
        ar3dTriggers_[slot] = trigger;
    }
    int maxActive = 0;
    for (int i = 0; i < math::kMaxAr3dLayers; ++i) {
        if (ar3dAccessories_[i].enabled && ar3dAccessories_[i].texture != rhi::kInvalidTexture) {
            maxActive = i + 1;
        }
    }
    ar3dCount_ = maxActive;
}

void LiveBeautyEngine::clearAr3dAccessories() {
    for (auto& a : ar3dAccessories_) a = math::Ar3dAccessoryConfig{};
    ar3dTriggers_.fill(bundle::BundleTriggerRule::kAlways);
    ar3dCount_ = 0;
}

bool LiveBeautyEngine::applyEffectBundle(
    const bundle::EffectBundleManifest& manifest,
    const bundle::BundleTextureResolver& textureResolver) {
    if (manifest.hasBeautySection) {
        BeautyConfig merged = manifest.beauty;
        if (!manifest.overridesBackground) {
            merged.backgroundBlur = config_.backgroundBlur;
            merged.backgroundMode = config_.backgroundMode;
            merged.virtualBgStrength = config_.virtualBgStrength;
        }
        if (!manifest.overridesPipelineOptions) {
            merged.performanceTier = config_.performanceTier;
            merged.preferHalfFloat = config_.preferHalfFloat;
            merged.maxFaces = config_.maxFaces;
            merged.frameDelayFrames = config_.frameDelayFrames;
        }
        setConfig(merged);
    }
    clearStickers();
    for (const auto& s : manifest.stickers) {
        auto layer = s.layer;
        if (layer.texture == rhi::kInvalidTexture && textureResolver && !s.assetPath.empty()) {
            layer.texture = textureResolver(s.assetPath);
        }
        setSticker(s.slot, layer, s.trigger);
    }
    clearAr3dAccessories();
    for (const auto& a : manifest.ar3d) {
        auto layer = a.layer;
        if (layer.texture == rhi::kInvalidTexture && textureResolver && !a.assetPath.empty()) {
            layer.texture = textureResolver(a.assetPath);
        }
        setAr3dAccessory(a.slot, layer, a.trigger);
    }
    return true;
}

bool LiveBeautyEngine::applyEffectBundleJson(
    const std::string& manifestJson,
    const bundle::BundleTextureResolver& textureResolver,
    std::string* outError) {
    bundle::EffectBundleManifest parsed{};
    if (!bundle::EffectBundleParser::parseJson(manifestJson, parsed, outError)) {
        return false;
    }
    return applyEffectBundle(parsed, textureResolver);
}

void LiveBeautyEngine::setHardwareProfile(const std::string& gpuRenderer, int cpuCores, int totalRamMb) {
    hardwareTier_ = math::DeviceTierPolicy::classify(gpuRenderer, cpuCores, totalRamMb);
}

void LiveBeautyEngine::setAmbientRgb(float r, float g, float b) {
    ambientRgb_ = {
        detail::clampf(r, 0.65f, 1.35f),
        detail::clampf(g, 0.65f, 1.35f),
        detail::clampf(b, 0.65f, 1.35f),
    };
}

void LiveBeautyEngine::setOcclusionMask(rhi::TextureId occlusionMaskTex) {
    occlusionMaskTex_ = occlusionMaskTex;
}

void LiveBeautyEngine::setAiResidual(rhi::TextureId residualTex, const CropRect& roi, float strength) {
    aiResidualTex_ = residualTex;
    aiResidualRect_ = roi;
    aiResidualStrength_ = detail::clampf(strength, 0.0f, 1.0f);
}

void LiveBeautyEngine::submitFaceBlendshapes(const math::BlendshapeInput& blendshapes) {
    std::lock_guard<std::mutex> lock(auxMutex_);
    expressionBus_.update(&blendshapes);
}

math::ExpressionReaction LiveBeautyEngine::expressionReaction() const {
    std::lock_guard<std::mutex> lock(auxMutex_);
    return expressionBus_.latest();
}

bool LiveBeautyEngine::submitAuxRgbaSample(
    const uint8_t* rgbaTopDown,
    int srcWidth,
    int srcHeight,
    float retouchStrength) {
    if (!rgbaTopDown || srcWidth <= 0 || srcHeight <= 0) return false;
    std::lock_guard<std::mutex> lock(auxMutex_);
    if (!hasLatestFaceBounds_) {
        stagedResidualActive_ = false;
        stagedResidualDirty_ = true;
        return false;
    }
    const CropRect bounds = latestFaceBounds_;
    const float rollRad = latestFaceRollRad_;
    // Do not overwrite 128x128 19-class semantic parsing mask when active
    if (!semanticOccActive_) {
        occlusionSolver_.compute(rgbaTopDown, srcWidth, srcHeight, &bounds, stagedOccMask_.data(), rollRad);
        stagedOccSize_ = math::FaceOcclusionSolver::kMaskSize;
        stagedOccDirty_ = true;
    }
    const float clampedStrength = detail::clampf(retouchStrength, 0.0f, 1.0f);
    std::array<uint8_t, math::FaceRoiResidualSolver::kRoiPixels * 4> biRes{};
    std::array<uint8_t, math::NeuralSkinResidualAdapter::kOutPixels * 4> microRes{};
    if (clampedStrength > 0.02f &&
        residualSolver_.compute(rgbaTopDown, srcWidth, srcHeight, bounds, clampedStrength, biRes.data()) &&
        neuralResidualAdapter_.evaluateMicroKernel(rgbaTopDown, srcWidth, srcHeight, bounds, clampedStrength, microRes.data())) {
        // Combine bilateral chroma de-speckling + multi-scale depthwise-separable structural shadow lift
        for (size_t i = 0; i < stagedResidualRgba_.size(); i += 4) {
            for (size_t c = 0; c < 3; ++c) {
                const int d0 = static_cast<int>(biRes[i + c]) - 128;
                const int d1 = static_cast<int>(microRes[i + c]) - 128;
                stagedResidualRgba_[i + c] = static_cast<uint8_t>(std::clamp(128 + d0 + (d1 * 3) / 4, 0, 255));
            }
            stagedResidualRgba_[i + 3] = std::max(biRes[i + 3], microRes[i + 3]);
        }
        stagedResidualActive_ = true;
        stagedResidualRect_ = bounds;
        stagedResidualStrength_ = clampedStrength;
    } else {
        stagedResidualActive_ = false;
        stagedResidualStrength_ = 0.0f;
    }
    stagedResidualDirty_ = true;
    return true;
}

bool LiveBeautyEngine::submitFaceParsingClassMap(
    const uint8_t* classIds, int srcWidth, int srcHeight, int64_t sampleTsMs) {
    if (!classIds || srcWidth <= 0 || srcHeight <= 0) return false;
    std::lock_guard<std::mutex> lock(auxMutex_);
    if (!faceParsingMapper_.mapClassIds(
            classIds, srcWidth, srcHeight, stagedOccMask_.data(), stagedParseHair_.data(), nullptr)) {
        return false;
    }
    semanticOccActive_ = true;
    stagedOccSize_ = math::FaceParsingMapper::kOutSize;
    stagedOccDirty_ = true;
    stagedParseHairDirty_ = true;
    stagedParseHairTsMs_ = sampleTsMs;
    return true;
}

bool LiveBeautyEngine::submitFaceParsingLogits(
    const float* logitsNchw, int numClasses, int srcWidth, int srcHeight, int64_t sampleTsMs) {
    if (!logitsNchw || numClasses < 19 || srcWidth <= 0 || srcHeight <= 0) return false;
    std::lock_guard<std::mutex> lock(auxMutex_);
    if (!faceParsingMapper_.mapLogitsNchw(
            logitsNchw, numClasses, srcWidth, srcHeight, stagedOccMask_.data(), stagedParseHair_.data(), nullptr)) {
        return false;
    }
    semanticOccActive_ = true;
    stagedOccSize_ = math::FaceParsingMapper::kOutSize;
    stagedOccDirty_ = true;
    stagedParseHairDirty_ = true;
    stagedParseHairTsMs_ = sampleTsMs;
    return true;
}

bool LiveBeautyEngine::submitNeuralResidualTensor(
    const float* tensorNchw,
    int channels,
    int srcWidth,
    int srcHeight,
    const CropRect& roiRect,
    float strength) {
    std::lock_guard<std::mutex> lock(auxMutex_);
    const float clamped = detail::clampf(strength, 0.0f, 1.0f);
    if (clamped <= 0.01f) {
        stagedResidualActive_ = false;
        stagedResidualStrength_ = 0.0f;
        stagedResidualDirty_ = true;
        return true;
    }
    if (!math::NeuralSkinResidualAdapter::packExternalTensorNchw(
            tensorNchw, channels, srcWidth, srcHeight, stagedResidualRgba_.data())) {
        return false;
    }
    stagedResidualActive_ = true;
    stagedResidualRect_ = roiRect;
    stagedResidualStrength_ = clamped;
    stagedResidualDirty_ = true;
    return true;
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
    if (valid > 0 && fw > 0.0f && faces && faceCount > 0) {
        // Select the raw input closest to primary slot(0)'s top-down center (cx, 1 - cy)
        const float targetTopDownCy = 1.0f - cy;
        int bestIdx = -1;
        float bestDist2 = 1e9f;
        for (int i = 0; i < cappedCount; ++i) {
            if (!faces[i].mpXy || faces[i].pointCount < 468) continue;
            CropRect cand{};
            if (!math::FaceOcclusionSolver::boundsFromLandmarks(faces[i].mpXy, faces[i].pointCount, cand)) continue;
            const float candCx = cand.left + 0.5f * cand.width;
            const float candCy = cand.top + 0.5f * cand.height;
            const float d2 = (candCx - cx) * (candCx - cx) + (candCy - targetTopDownCy) * (candCy - targetTopDownCy);
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestIdx = i;
            }
        }
        if (bestIdx >= 0) {
            CropRect fb{};
            float roll = 0.0f;
            if (math::FaceOcclusionSolver::boundsFromLandmarks(
                    faces[bestIdx].mpXy, faces[bestIdx].pointCount, fb, &roll)) {
                std::lock_guard<std::mutex> lock(auxMutex_);
                latestFaceBounds_ = fb;
                latestFaceRollRad_ = roll;
                hasLatestFaceBounds_ = true;
            }
        }
    } else if (valid == 0) {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasLatestFaceBounds_ = false;
        latestFaceRollRad_ = 0.0f;
        expressionBus_.update(nullptr);
    }
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
    std::lock_guard<std::mutex> lock(auxMutex_);
    hasLatestFaceBounds_ = false;
    expressionBus_.update(nullptr);
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
    const math::TierCapabilities tierCaps =
        math::DeviceTierPolicy::resolve(config_.performanceTier, hardwareTier_, governor_.level());
    const math::PerformanceTier resolvedTier = tierCaps.resolvedTier;
    const bool processing = config_.isProcessing() ||
        (config_.enabled && !config_.compareOriginal &&
         (stickerCount_ > 0 || ar3dCount_ > 0 ||
          (aiResidualTex_ != rhi::kInvalidTexture && aiResidualStrength_ > 1e-4f) ||
          (autoResidualTex_ != rhi::kInvalidTexture && autoResidualStrength_ > 1e-4f)));
    loadFades_.update(governor_.level(), nowMs);

    // Sync staged C++17 + NEON occlusion & residual maps onto GPU RHI textures
    math::ExpressionReaction exprReaction{};
    bool faceBoundsAlive = false;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        exprReaction = expressionBus_.latest();
        faceBoundsAlive = hasLatestFaceBounds_;
        if (stagedOccDirty_) {
            stagedOccDirty_ = false;
            const int occS = stagedOccSize_;
            if (autoOccTex_ != rhi::kInvalidTexture && autoOccTexSize_ != occS) {
                device_.destroyTexture(autoOccTex_);
                autoOccTex_ = rhi::kInvalidTexture;
            }
            if (autoOccTex_ == rhi::kInvalidTexture) {
                autoOccTex_ = device_.createTexture2D(occS, occS, rhi::TextureFormat::kR8, stagedOccMask_.data());
                autoOccTexSize_ = occS;
            } else {
                device_.updateTexture2D(autoOccTex_, occS, occS, rhi::TextureFormat::kR8, stagedOccMask_.data());
            }
        }
        if (stagedParseHairDirty_) {
            stagedParseHairDirty_ = false;
            constexpr int kHairS = math::FaceParsingMapper::kOutSize;
            if (autoParseHairTex_ == rhi::kInvalidTexture) {
                autoParseHairTex_ = device_.createTexture2D(kHairS, kHairS, rhi::TextureFormat::kR8, stagedParseHair_.data());
            } else {
                device_.updateTexture2D(autoParseHairTex_, kHairS, kHairS, rhi::TextureFormat::kR8, stagedParseHair_.data());
            }
            autoParseHairValid_ = (autoParseHairTex_ != rhi::kInvalidTexture);
            autoParseHairTsMs_ = stagedParseHairTsMs_;
        }
        if (stagedResidualDirty_) {
            stagedResidualDirty_ = false;
            if (stagedResidualActive_) {
                constexpr int kResS = math::FaceRoiResidualSolver::kRoiSize;
                if (autoResidualTex_ == rhi::kInvalidTexture) {
                    autoResidualTex_ = device_.createTexture2D(kResS, kResS, rhi::TextureFormat::kRgba8, stagedResidualRgba_.data());
                } else {
                    device_.updateTexture2D(autoResidualTex_, kResS, kResS, rhi::TextureFormat::kRgba8, stagedResidualRgba_.data());
                }
                autoResidualRect_ = stagedResidualRect_;
                autoResidualStrength_ = stagedResidualStrength_;
            } else {
                autoResidualStrength_ = 0.0f;
            }
        }
    }

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

    // Optional secondary face (slot 1), gated by resolved hardware tier
    const int effectiveMaxFaces = std::min(config_.maxFaces, tierCaps.maxFacesCap);
    auto& secondaryTl = faces_.slot(1);
    const float* secState41 = (effectiveMaxFaces >= 2) ? secondaryTl.at(displayTsMs) : nullptr;
    math::FaceTimeline::SmoothedMeshView secMeshView{};
    const bool hasSecMesh = (effectiveMaxFaces >= 2) && secondaryTl.meshAt(displayTsMs, secMeshView);

    const float* bodyState = bodySmoother_.update(
        hasPendingBody_ ? pendingBodyState_.data() : nullptr, pendingBodyTs_, nowMs);

    // Hair semantic temporal gate & rigid head-motion compensation affine (using aligned displayTsMs)
    const rhi::TextureId resolvedHairTex = (hairMaskTex != rhi::kInvalidTexture)
        ? hairMaskTex
        : (autoParseHairValid_ ? autoParseHairTex_ : rhi::kInvalidTexture);
    const int64_t resolvedHairStampArg = (hairMaskTex != rhi::kInvalidTexture) ? hairMaskStampMs : autoParseHairTsMs_;
    const bool hasHairTex = (resolvedHairTex != rhi::kInvalidTexture);
    const int64_t effHairStamp = (resolvedHairStampArg != kTimeInvalid) ? resolvedHairStampArg : displayTsMs;
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
    in.hairMaskTex = resolvedHairTex;
    bool semanticOccAlive = false;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        semanticOccAlive = semanticOccActive_;
    }
    in.occlusionMaskTex = (occlusionMaskTex_ != rhi::kInvalidTexture)
        ? occlusionMaskTex_
        : ((faceBoundsAlive || semanticOccAlive) && hasMesh
               ? autoOccTex_
               : rhi::kInvalidTexture);
    in.hairShown = hairShown;
    in.hairBgShown = hairBgShown;
    in.hairAffine = hairAffine;
    in.ambientRgb = ambientRgb_;
    if (aiResidualTex_ != rhi::kInvalidTexture && aiResidualStrength_ > 1e-4f) {
        in.aiResidualTex = aiResidualTex_;
        in.aiResidualRect = aiResidualRect_;
        in.aiResidualStrength = aiResidualStrength_;
    } else if (faceBoundsAlive && hasMesh && autoResidualTex_ != rhi::kInvalidTexture && autoResidualStrength_ > 1e-4f) {
        in.aiResidualTex = autoResidualTex_;
        in.aiResidualRect = autoResidualRect_;
        in.aiResidualStrength = autoResidualStrength_;
    }
    std::array<math::StickerLayerConfig, math::kMaxStickerLayers> modulatedStickers = stickers_;
    for (int i = 0; i < stickerCount_; ++i) {
        if (!modulatedStickers[i].enabled) continue;
        if (stickerTriggers_[i] != bundle::BundleTriggerRule::kAlways) {
            const float gain = bundle::EffectBundleParser::evaluateTriggerGain(stickerTriggers_[i], exprReaction);
            modulatedStickers[i].opacity *= gain;
            if (modulatedStickers[i].opacity <= 1e-4f) modulatedStickers[i].enabled = false;
        }
    }
    in.stickers = modulatedStickers;
    in.stickerCount = stickerCount_;

    std::array<math::Ar3dAccessoryConfig, math::kMaxAr3dLayers> modulatedAr3d = ar3dAccessories_;
    for (int i = 0; i < ar3dCount_; ++i) {
        if (!modulatedAr3d[i].enabled) continue;
        if (ar3dTriggers_[i] != bundle::BundleTriggerRule::kAlways) {
            const float gain = bundle::EffectBundleParser::evaluateTriggerGain(ar3dTriggers_[i], exprReaction);
            modulatedAr3d[i].opacity *= gain;
            if (modulatedAr3d[i].opacity <= 1e-4f) {
                modulatedAr3d[i].enabled = false;
                continue;
            }
        }
        if (expressionReactive_ && exprReaction.trigger != math::ExpressionTrigger::kNone) {
            modulatedAr3d[i].scale *= exprReaction.scaleMultiplier;
            modulatedAr3d[i].offsetY += exprReaction.offsetYBoost;
            modulatedAr3d[i].opacity = detail::clampf(modulatedAr3d[i].opacity * exprReaction.opacityBoost, 0.0f, 1.0f);
        }
    }
    in.ar3dAccessories = modulatedAr3d;
    in.ar3dCount = ar3dCount_;
    in.tierCaps = tierCaps;
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
    stats_.resolvedTier = static_cast<int>(resolvedTier);
    stats_.facePresence = std::max(primaryTl.presence(), (effectiveMaxFaces >= 2) ? secondaryTl.presence() : 0.0f);
    stats_.hairShown = hairShown;
    stats_.activePassesMask = pipeline_.lastActivePassesMask() | (delayFrames > 0 ? rhi::kPassFrameDelay : 0u);
    stats_.frameCount++;
    stats_.activeFaces = faces_.activeFaceCount();
    stats_.displayedTimestampMs = displayTsMs;

    return outTex;
}

}  // namespace livebeauty
