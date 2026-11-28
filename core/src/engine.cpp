#include "livebeauty/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace livebeauty {

namespace {

// Restrict full-frame semantic occlusion visibility masks (255=visible, 0=occluded) strictly to active
// face ellipse(s) so 6-class clothes (class 4) / accessories (class 5) / hair (class 1) on the neck,
// collarbone, or chest NEVER zero out rtSkinMask_.a (nonHair) in kCompositePro and punch dark
// un-whitened shadow patches across the chest.
void restrictOccMaskToFaces(
    uint8_t* occMaskR8,
    int size,
    const CropRect* primaryBounds,
    const CropRect* secondaryBounds) {
    if (!occMaskR8 || size <= 0) return;
    auto validBounds = [](const CropRect* b) {
        return b && b->width >= 0.04f && b->height >= 0.04f;
    };
    const bool hasP = validBounds(primaryBounds);
    const bool hasS = validBounds(secondaryBounds);
    if (!hasP && !hasS) {
        std::fill(occMaskR8, occMaskR8 + static_cast<size_t>(size) * static_cast<size_t>(size), uint8_t{255});
        return;
    }
    auto evalKeep = [](float u, float v, const CropRect& b) {
        const float cx = b.left + 0.5f * b.width;
        const float cy = b.top + 0.5f * b.height;
        const float rx = std::max(0.50f * b.width, 0.02f);
        const float ry = std::max(0.50f * b.height, 0.02f);
        const float dx = (u - cx) / rx;
        const float dy = (v - cy) / ry;
        return 1.0f - detail::smoothstep(0.85f, 1.05f, dx * dx + dy * dy);
    };
    const float invSize = 1.0f / static_cast<float>(size);
    for (int y = 0; y < size; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) * invSize;
        const int row = y * size;
        for (int x = 0; x < size; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) * invSize;
            float keep = 0.0f;
            if (hasP) keep = std::max(keep, evalKeep(u, v, *primaryBounds));
            if (hasS) keep = std::max(keep, evalKeep(u, v, *secondaryBounds));
            if (keep <= 1e-4f) {
                occMaskR8[row + x] = 255;
            } else if (keep < 0.999f) {
                const float vis = static_cast<float>(occMaskR8[row + x]);
                occMaskR8[row + x] = static_cast<uint8_t>(
                    std::clamp(static_cast<int>(std::lround(255.0f - (255.0f - vis) * keep)), 0, 255));
            }
        }
    }
}

}  // namespace

LiveBeautyEngine::LiveBeautyEngine(rhi::IRhiDevice& device)
    : device_(device), pipeline_(device) {}

LiveBeautyEngine::~LiveBeautyEngine() {
    releaseGpuResources();
}

void LiveBeautyEngine::setConfig(const BeautyConfig& config) {
    if (config.hasBodyReshape() && !config_.hasBodyReshape()) {
        bodySmoother_.reset();
        hasPendingBody_ = false;
        pendingBodyTs_ = kTimeInvalid;
    }
    config_ = config;
    blemishActive_.store(config_.blemishRemove > 0.05f, std::memory_order_relaxed);
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
    pendingBodyVis_ = 1.0f;
    pendingKneeVis_ = 0.0f;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        {
            std::lock_guard<std::mutex> mlock(mapperMutex_);
            occlusionSolver_.reset();
            faceParsingMapper_.reset();
        }
        expressionBus_.reset();
        particleSolver_.reset();
        activeGesture_ = math::GestureType::kNone;
        activeGestureUntilMs_ = kTimeInvalid;
        handShieldCoords_ = {-10.0f, -10.0f, -10.0f, -10.0f};
        hasLatestFaceBounds_ = false;
        hasLatestSecFaceBounds_ = false;
        latestFaceRollRad_ = 0.0f;
        semanticOccActive_ = false;
        lastSemanticSubmitMs_ = kTimeInvalid;
        stagedOccDirty_ = false;
        stagedOccTsMs_ = kTimeInvalid;
        stagedParseHairDirty_ = false;
        stagedParseHairTsMs_ = kTimeInvalid;
        stagedParsePersonDirty_ = false;
        stagedResidualDirty_ = false;
        stagedResidualActive_ = false;
    }
    cameraShaking_.store(false, std::memory_order_relaxed);
    autoOccTsMs_ = kTimeInvalid;
    autoResidualStrength_ = 0.0f;
    autoParseHairValid_ = false;
    autoParseHairTsMs_ = kTimeInvalid;
    autoParsePersonValid_ = false;
    lastAmbientNowMs_ = kTimeInvalid;
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
    if (autoParsePersonTex_ != rhi::kInvalidTexture) {
        device_.destroyTexture(autoParsePersonTex_);
        autoParsePersonTex_ = rhi::kInvalidTexture;
    }
    autoParsePersonValid_ = false;
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
            merged.chromaKeyColor = config_.chromaKeyColor;
            merged.chromaKeySimilarity = config_.chromaKeySimilarity;
            merged.chromaKeySmoothness = config_.chromaKeySmoothness;
            merged.chromaKeyDespill = config_.chromaKeyDespill;
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

void LiveBeautyEngine::setHostLoadLevel(int level) {
    hostLoadLevel_.store(std::clamp(level, 0, 3), std::memory_order_relaxed);
}

void LiveBeautyEngine::prewarmShaders(int startIdx, int count) {
    const int total = static_cast<int>(rhi::ShaderId::kCount);
    const int from = std::max(0, startIdx);
    const int to = std::min(total, from + std::max(0, count));
    for (int i = from; i < to; ++i) {
        device_.prewarmShader(static_cast<rhi::ShaderId>(i));
    }
}

void LiveBeautyEngine::setAmbientRgb(float r, float g, float b) {
    targetAmbientRgb_ = {
        detail::clampf(r, 0.65f, 1.35f),
        detail::clampf(g, 0.65f, 1.35f),
        detail::clampf(b, 0.65f, 1.35f),
    };
    if (lastAmbientNowMs_ == kTimeInvalid) {
        ambientRgb_ = targetAmbientRgb_;
    }
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
    bool hasBounds = false;
    CropRect bounds{};
    float rollRad = 0.0f;
    bool semanticAlive = false;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasBounds = hasLatestFaceBounds_;
        bounds = latestFaceBounds_;
        rollRad = latestFaceRollRad_;
        semanticAlive = semanticOccActive_;
        if (!hasBounds) {
            stagedResidualActive_ = false;
            stagedResidualDirty_ = true;
            return false;
        }
    }

    // Run heavy NEON occlusion & ROI residual compute under mapperMutex_ (never blocks GL thread's auxMutex_).
    std::array<uint8_t, math::FaceOcclusionSolver::kMaskPixels> localOcc{};
    float localLowerOcc = 0.0f;
    bool computedOcc = false;

    const float clampedStrength = detail::clampf(retouchStrength, 0.0f, 1.0f);
    const bool wantResidual = (clampedStrength > 0.02f) && blemishActive_.load(std::memory_order_relaxed);
    std::array<uint8_t, math::FaceRoiResidualSolver::kRoiPixels * 4> localResidual{};
    bool computedResidual = false;
    {
        std::lock_guard<std::mutex> mlock(mapperMutex_);
        if (!semanticAlive) {
            occlusionSolver_.compute(rgbaTopDown, srcWidth, srcHeight, &bounds, localOcc.data(), rollRad);
            localLowerOcc = occlusionSolver_.lowerFaceOcclusion();
            computedOcc = true;
        }
        if (wantResidual) {
            std::array<uint8_t, math::FaceRoiResidualSolver::kRoiPixels * 4> biRes{};
            std::array<uint8_t, math::NeuralSkinResidualAdapter::kOutPixels * 4> microRes{};
            if (residualSolver_.compute(rgbaTopDown, srcWidth, srcHeight, bounds, clampedStrength, biRes.data(), rollRad) &&
                neuralResidualAdapter_.evaluateMicroKernel(rgbaTopDown, srcWidth, srcHeight, bounds, clampedStrength, microRes.data(), rollRad)) {
                for (size_t i = 0; i < localResidual.size(); i += 4) {
                    for (size_t c = 0; c < 3; ++c) {
                        const int d0 = static_cast<int>(biRes[i + c]) - 128;
                        const int d1 = static_cast<int>(microRes[i + c]) - 128;
                        localResidual[i + c] = static_cast<uint8_t>(std::clamp(128 + d0 + (d1 * 3) / 4, 0, 255));
                    }
                    localResidual[i + 3] = std::max(biRes[i + 3], microRes[i + 3]);
                }
                computedResidual = true;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        if (computedOcc && !semanticOccActive_) {
            std::copy(localOcc.begin(), localOcc.end(), stagedOccMask_.begin());
            stagedOccSize_ = math::FaceOcclusionSolver::kMaskSize;
            stagedOccTsMs_ = kTimeInvalid;
            stagedLowerFaceOcc_ = localLowerOcc;
            stagedSecLowerFaceOcc_ = 0.0f;
            stagedOccDirty_ = true;
        }
        if (computedResidual) {
            stagedResidualRgba_ = localResidual;
            stagedResidualActive_ = true;
            stagedResidualRect_ = bounds;
            stagedResidualStrength_ = clampedStrength;
        } else {
            stagedResidualActive_ = false;
            stagedResidualStrength_ = 0.0f;
        }
        stagedResidualDirty_ = true;
    }
    return true;
}

bool LiveBeautyEngine::submitFaceParsingClassMap(
    const uint8_t* classIds, int srcWidth, int srcHeight, int64_t sampleTsMs) {
    if (!classIds || srcWidth <= 0 || srcHeight <= 0) return false;
    bool hasP = false, hasS = false;
    CropRect pBounds{}, sBounds{};
    int64_t prevSemMs = kTimeInvalid;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasP = hasLatestFaceBounds_;
        pBounds = latestFaceBounds_;
        hasS = hasLatestSecFaceBounds_;
        sBounds = latestSecFaceBounds_;
        prevSemMs = lastSemanticSubmitMs_;
    }

    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localOcc{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localHair{};
    {
        std::lock_guard<std::mutex> mlock(mapperMutex_);
        if (prevSemMs != kTimeInvalid && std::llabs(sampleTsMs - prevSemMs) > 800) {
            faceParsingMapper_.reset();
        }
        if (!faceParsingMapper_.mapClassIds(
                classIds, srcWidth, srcHeight, localOcc.data(), localHair.data(), nullptr)) {
            return false;
        }
    }
    const int outS = math::FaceParsingMapper::kOutSize;
    const float lowerOcc = hasP ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, pBounds) : 0.0f;
    const float secLowerOcc = hasS ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, sBounds) : 0.0f;
    restrictOccMaskToFaces(localOcc.data(), outS, hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr);

    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        stagedOccMask_ = localOcc;
        stagedParseHair_ = localHair;
        semanticOccActive_ = true;
        lastSemanticSubmitMs_ = sampleTsMs;
        stagedOccSize_ = outS;
        stagedOccTsMs_ = sampleTsMs;
        stagedLowerFaceOcc_ = lowerOcc;
        stagedSecLowerFaceOcc_ = secLowerOcc;
        stagedOccDirty_ = true;
        stagedParseHairDirty_ = true;
        stagedParseHairTsMs_ = sampleTsMs;
    }
    return true;
}

bool LiveBeautyEngine::submitFaceParsingLogits(
    const float* logitsNchw, int numClasses, int srcWidth, int srcHeight, int64_t sampleTsMs) {
    if (!logitsNchw || numClasses < 19 || srcWidth <= 0 || srcHeight <= 0) return false;
    bool hasP = false, hasS = false;
    CropRect pBounds{}, sBounds{};
    int64_t prevSemMs = kTimeInvalid;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasP = hasLatestFaceBounds_;
        pBounds = latestFaceBounds_;
        hasS = hasLatestSecFaceBounds_;
        sBounds = latestSecFaceBounds_;
        prevSemMs = lastSemanticSubmitMs_;
    }

    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localOcc{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localHair{};
    {
        std::lock_guard<std::mutex> mlock(mapperMutex_);
        if (prevSemMs != kTimeInvalid && std::llabs(sampleTsMs - prevSemMs) > 800) {
            faceParsingMapper_.reset();
        }
        if (!faceParsingMapper_.mapLogitsNchw(
                logitsNchw, numClasses, srcWidth, srcHeight, localOcc.data(), localHair.data(), nullptr)) {
            return false;
        }
    }
    const int outS = math::FaceParsingMapper::kOutSize;
    const float lowerOcc = hasP ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, pBounds) : 0.0f;
    const float secLowerOcc = hasS ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, sBounds) : 0.0f;
    restrictOccMaskToFaces(localOcc.data(), outS, hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr);

    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        stagedOccMask_ = localOcc;
        stagedParseHair_ = localHair;
        semanticOccActive_ = true;
        lastSemanticSubmitMs_ = sampleTsMs;
        stagedOccSize_ = outS;
        stagedOccTsMs_ = sampleTsMs;
        stagedLowerFaceOcc_ = lowerOcc;
        stagedSecLowerFaceOcc_ = secLowerOcc;
        stagedOccDirty_ = true;
        stagedParseHairDirty_ = true;
        stagedParseHairTsMs_ = sampleTsMs;
    }
    return true;
}

bool LiveBeautyEngine::submitMulticlass6Logits(
    const float* logitsNhwc6, int srcWidth, int srcHeight, int64_t sampleTsMs) {
    return submitMulticlass6LogitsRegion(logitsNhwc6, srcWidth, srcHeight, 0, 0, srcWidth, srcHeight, sampleTsMs);
}

bool LiveBeautyEngine::submitMulticlass6LogitsRegion(
    const float* logitsNhwc6,
    int srcWidth, int srcHeight,
    int padX, int padY, int validW, int validH,
    int64_t sampleTsMs) {
    if (!logitsNhwc6 || srcWidth <= 0 || srcHeight <= 0 || validW <= 0 || validH <= 0) return false;
    bool hasP = false, hasS = false;
    CropRect pBounds{}, sBounds{};
    int64_t prevSemMs = kTimeInvalid;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasP = hasLatestFaceBounds_;
        pBounds = latestFaceBounds_;
        hasS = hasLatestSecFaceBounds_;
        sBounds = latestSecFaceBounds_;
        prevSemMs = lastSemanticSubmitMs_;
    }

    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localOcc{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localHair{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localPerson{};
    {
        std::lock_guard<std::mutex> mlock(mapperMutex_);
        if (prevSemMs != kTimeInvalid && std::llabs(sampleTsMs - prevSemMs) > 800) {
            faceParsingMapper_.reset();
        }
        if (!faceParsingMapper_.mapMulticlass6LogitsNhwcRegion(
                logitsNhwc6, srcWidth, srcHeight, padX, padY, validW, validH,
                localOcc.data(), localHair.data(), localPerson.data(),
                hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr)) {
            return false;
        }
    }
    const int outS = math::FaceParsingMapper::kOutSize;
    const float lowerOcc = hasP ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, pBounds) : 0.0f;
    const float secLowerOcc = hasS ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, sBounds) : 0.0f;
    restrictOccMaskToFaces(localOcc.data(), outS, hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr);

    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        stagedOccMask_ = localOcc;
        stagedParseHair_ = localHair;
        stagedParsePerson_ = localPerson;
        semanticOccActive_ = true;
        lastSemanticSubmitMs_ = sampleTsMs;
        stagedOccSize_ = outS;
        stagedOccTsMs_ = sampleTsMs;
        stagedLowerFaceOcc_ = lowerOcc;
        stagedSecLowerFaceOcc_ = secLowerOcc;
        stagedOccDirty_ = true;
        stagedParseHairDirty_ = true;
        stagedParseHairTsMs_ = sampleTsMs;
        stagedParsePersonDirty_ = true;
    }
    return true;
}

bool LiveBeautyEngine::submitMulticlass6Planar(
    const float* const planes[6], int srcWidth, int srcHeight, int64_t sampleTsMs) {
    if (!planes || srcWidth <= 0 || srcHeight <= 0) return false;
    bool hasP = false, hasS = false;
    CropRect pBounds{}, sBounds{};
    int64_t prevSemMs = kTimeInvalid;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasP = hasLatestFaceBounds_;
        pBounds = latestFaceBounds_;
        hasS = hasLatestSecFaceBounds_;
        sBounds = latestSecFaceBounds_;
        prevSemMs = lastSemanticSubmitMs_;
    }

    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localOcc{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localHair{};
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> localPerson{};
    {
        std::lock_guard<std::mutex> mlock(mapperMutex_);
        if (prevSemMs != kTimeInvalid && std::llabs(sampleTsMs - prevSemMs) > 800) {
            faceParsingMapper_.reset();
        }
        if (!faceParsingMapper_.mapMulticlass6PlanarProbabilities(
                planes, srcWidth, srcHeight,
                localOcc.data(), localHair.data(), localPerson.data(),
                hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr)) {
            return false;
        }
    }
    const int outS = math::FaceParsingMapper::kOutSize;
    const float lowerOcc = hasP ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, pBounds) : 0.0f;
    const float secLowerOcc = hasS ? math::FaceParsingMapper::estimateLowerFaceOcclusion(localOcc.data(), outS, sBounds) : 0.0f;
    restrictOccMaskToFaces(localOcc.data(), outS, hasP ? &pBounds : nullptr, hasS ? &sBounds : nullptr);

    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        stagedOccMask_ = localOcc;
        stagedParseHair_ = localHair;
        stagedParsePerson_ = localPerson;
        semanticOccActive_ = true;
        lastSemanticSubmitMs_ = sampleTsMs;
        stagedOccSize_ = outS;
        stagedOccTsMs_ = sampleTsMs;
        stagedLowerFaceOcc_ = lowerOcc;
        stagedSecLowerFaceOcc_ = secLowerOcc;
        stagedOccDirty_ = true;
        stagedParseHairDirty_ = true;
        stagedParseHairTsMs_ = sampleTsMs;
        stagedParsePersonDirty_ = true;
    }
    return true;
}

void LiveBeautyEngine::triggerGesture(
    math::GestureType gesture, float normX, float normY, int64_t nowMs) {
    std::lock_guard<std::mutex> lock(auxMutex_);
    if (gesture == math::GestureType::kNone) return;
    particleSolver_.triggerBurst(gesture, normX, normY, nowMs);
    activeGesture_ = gesture;
    activeGestureUntilMs_ = nowMs + 1200;
}

void LiveBeautyEngine::setHandShieldCoords(
    float hand0X, float hand0Y, float hand1X, float hand1Y) {
    std::lock_guard<std::mutex> lock(auxMutex_);
    const bool h0Valid = (hand0X >= 0.0f && hand0X <= 1.0f && hand0Y >= 0.0f && hand0Y <= 1.0f);
    const bool h1Valid = (hand1X >= 0.0f && hand1X <= 1.0f && hand1Y >= 0.0f && hand1Y <= 1.0f);
    handShieldCoords_ = {
        h0Valid ? hand0X : -10.0f,
        h0Valid ? (1.0f - hand0Y) : -10.0f,
        h1Valid ? hand1X : -10.0f,
        h1Valid ? (1.0f - hand1Y) : -10.0f,
    };
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

    const int effLoadLevel = std::max(governor_.level(), hostLoadLevel_.load(std::memory_order_relaxed));
    const auto tierCaps = math::DeviceTierPolicy::resolve(config_.performanceTier, hardwareTier_, effLoadLevel);
    const int effectiveMaxFaces = std::min(
        std::clamp(config_.maxFaces, 1, math::MultiFaceTracker::kMaxFaces), tierCaps.maxFacesCap);
    const int cappedCount = std::min(faceCount, effectiveMaxFaces);
    float cx = 0.0f, cy = 0.0f, fw = 0.0f;
    const int valid = faces_.submitFaces(
        sampleTsMs, faces, cappedCount, frameWidth, frameHeight, receivedMs, &cx, &cy, &fw, effectiveMaxFaces);
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
                    faces[bestIdx].mpXy, faces[bestIdx].pointCount, fb, &roll, aspect)) {
                std::lock_guard<std::mutex> lock(auxMutex_);
                latestFaceBounds_ = fb;
                latestFaceRollRad_ = roll;
                hasLatestFaceBounds_ = true;
            }
            // Track secondary face bounds independently for dual-face lower-face occlusion gating
            bool foundSec = false;
            CropRect secFb{};
            for (int i = 0; i < cappedCount; ++i) {
                if (i == bestIdx || !faces[i].mpXy || faces[i].pointCount < 468) continue;
                if (math::FaceOcclusionSolver::boundsFromLandmarks(faces[i].mpXy, faces[i].pointCount, secFb, nullptr, aspect)) {
                    foundSec = true;
                    break;
                }
            }
            std::lock_guard<std::mutex> lock(auxMutex_);
            hasLatestSecFaceBounds_ = foundSec;
            if (foundSec) latestSecFaceBounds_ = secFb;
        }
    } else if (valid > 0 && fw <= 0.0f) {
        // Primary slot 0 was not assigned in this submission (only secondary face present): clear stale primary bounds.
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasLatestFaceBounds_ = false;
        latestFaceRollRad_ = 0.0f;
    } else if (valid == 0) {
        std::lock_guard<std::mutex> lock(auxMutex_);
        hasLatestFaceBounds_ = false;
        hasLatestSecFaceBounds_ = false;
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
    hasLatestSecFaceBounds_ = false;
    stagedLowerFaceOcc_ = 0.0f;
    stagedSecLowerFaceOcc_ = 0.0f;
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
    pendingBodyVis_ = bg.visibility;
    pendingKneeVis_ = bg.kneeVisibility;
    pendingBodyTs_ = sampleTsMs;
    hasPendingBody_ = true;
    return true;
}

void LiveBeautyEngine::submitBodyMiss(int64_t /*sampleTsMs*/) {
    hasPendingBody_ = false;
    pendingBodyVis_ = 0.0f;
    pendingKneeVis_ = 0.0f;
}

CropRect LiveBeautyEngine::selectFaceRoi(int frameWidth, int frameHeight, int64_t nowMs) {
    if (config_.maxFaces >= 2) return CropRect::full();
    if (cameraShaking_.load(std::memory_order_relaxed)) {
        return CropRect::full();
    }
    if (faces_.slot(0).speed() > 1.15f) {
        roiSelector_.onMiss();
        return CropRect::full();
    }
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
    int64_t hairMaskStampMs,
    rhi::RenderTargetId outputRt) {
    if (width <= 0 || height <= 0 || inputTexture == rhi::kInvalidTexture) return inputTexture;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    if (width != lastWidth_ || height != lastHeight_) {
        faces_.setAspect(aspect);
        lastWidth_ = width;
        lastHeight_ = height;
    }

    // Smooth ambient RGB damping (tau = 4000ms) prevents sudden WB flicker when moving across lamps.
    if (lastAmbientNowMs_ != kTimeInvalid && nowMs > lastAmbientNowMs_) {
        const float dtAmb = static_cast<float>(std::min<int64_t>(nowMs - lastAmbientNowMs_, 500));
        const float aAmb = 1.0f - std::exp(-dtAmb / 4000.0f);
        ambientRgb_.x += aAmb * (targetAmbientRgb_.x - ambientRgb_.x);
        ambientRgb_.y += aAmb * (targetAmbientRgb_.y - ambientRgb_.y);
        ambientRgb_.z += aAmb * (targetAmbientRgb_.z - ambientRgb_.z);
    } else {
        ambientRgb_ = targetAmbientRgb_;
    }
    lastAmbientNowMs_ = nowMs;

    const auto t0 = std::chrono::steady_clock::now();
    const int effLoadLevel = std::max(governor_.level(), hostLoadLevel_.load(std::memory_order_relaxed));
    const math::TierCapabilities tierCaps =
        math::DeviceTierPolicy::resolve(config_.performanceTier, hardwareTier_, effLoadLevel);
    const math::PerformanceTier resolvedTier = tierCaps.resolvedTier;

    // Sync staged C++17 + NEON occlusion, multiclass parsing, particles & residual maps onto GPU RHI textures
    math::ExpressionReaction exprReaction{};
    math::GestureType activeGesture = math::GestureType::kNone;
    Vec4 handShield{-10.0f, -10.0f, -10.0f, -10.0f};
    int particleQuads = 0;
    bool faceBoundsAlive = false;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        exprReaction = expressionBus_.latest();
        faceBoundsAlive = hasLatestFaceBounds_;
        handShield = handShieldCoords_;
        if (activeGestureUntilMs_ != kTimeInvalid && nowMs <= activeGestureUntilMs_) {
            activeGesture = activeGesture_;
        }
        if (config_.enabled && !config_.compareOriginal && particleSolver_.hasActiveParticles(nowMs)) {
            particleQuads = particleSolver_.buildBatch(nowMs, aspect, particleVerts_.data());
        }
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
            autoOccTsMs_ = stagedOccTsMs_;
            lowerFaceOcc_ = stagedLowerFaceOcc_;
            secLowerFaceOcc_ = stagedSecLowerFaceOcc_;
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
        if (stagedParsePersonDirty_) {
            stagedParsePersonDirty_ = false;
            constexpr int kPerS = math::FaceParsingMapper::kOutSize;
            if (autoParsePersonTex_ == rhi::kInvalidTexture) {
                autoParsePersonTex_ = device_.createTexture2D(kPerS, kPerS, rhi::TextureFormat::kR8, stagedParsePerson_.data());
            } else {
                device_.updateTexture2D(autoParsePersonTex_, kPerS, kPerS, rhi::TextureFormat::kR8, stagedParsePerson_.data());
            }
            autoParsePersonValid_ = (autoParsePersonTex_ != rhi::kInvalidTexture);
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

    const bool processing = config_.isProcessing() ||
        (config_.enabled && !config_.compareOriginal &&
         (stickerCount_ > 0 || ar3dCount_ > 0 || particleQuads > 0 ||
          (aiResidualTex_ != rhi::kInvalidTexture && aiResidualStrength_ > 1e-4f) ||
          (autoResidualTex_ != rhi::kInvalidTexture && autoResidualStrength_ > 1e-4f)));
    loadFades_.update(effLoadLevel, nowMs);

    // Optional cross-backend FrameDelayRing (0..4 frames; aligns displayed camera frame with async MediaPipe landmarks)
    rhi::TextureId displayTex = inputTexture;
    int64_t displayTsMs = frameTimestampMs;
    const int delayFrames = processing ? std::clamp(config_.frameDelayFrames, 0, 4) : 0;
    if (delayFrames > 0) {
        pipeline_.pushFrameDelay(inputTexture, frameTimestampMs, delayFrames, width, height, displayTex, displayTsMs);
    }

    faces_.update(nowMs);
    const int effectiveMaxFaces = std::min(config_.maxFaces, tierCaps.maxFacesCap);
    if (effectiveMaxFaces < 2) {
        faces_.promoteSecondaryIfPrimaryEmpty();
    }
    auto& primaryTl = faces_.slot(0);
    const uint64_t promoCount = faces_.promotionCount();
    if (primaryTl.runId() != deformerRunId_ || promoCount != deformerPromotionCount_) {
        deformerRunId_ = primaryTl.runId();
        deformerPromotionCount_ = promoCount;
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
    auto& secondaryTl = faces_.slot(1);
    const float* secState41 = (effectiveMaxFaces >= 2) ? secondaryTl.at(displayTsMs) : nullptr;
    math::FaceTimeline::SmoothedMeshView secMeshView{};
    const bool hasSecMesh = (effectiveMaxFaces >= 2) && secondaryTl.meshAt(displayTsMs, secMeshView);

    // Stale body pose timeout (>250ms without fresh pose -> fade out body warp cleanly)
    if (hasPendingBody_ && pendingBodyTs_ != kTimeInvalid && frameTimestampMs - pendingBodyTs_ > 250) {
        hasPendingBody_ = false;
    }
    const float* bodyState = bodySmoother_.update(
        hasPendingBody_ ? pendingBodyState_.data() : nullptr, pendingBodyTs_, nowMs,
        pendingBodyVis_, pendingKneeVis_);

    // Hair semantic temporal gate & rigid head-motion compensation affine (using aligned displayTsMs).
    // Internal 11Hz multiclass hair mask only binds when hairDye / hairShine is active so dynamic-INT8
    // hairline fluctuations never punch jumping nonHair holes into forehead/face-contour whitening.
    const bool wantAutoHair = config_.enabled && !config_.compareOriginal &&
                              (config_.hairDye > 1e-4f || config_.hairShine > 1e-4f);
    const rhi::TextureId resolvedHairTex = (hairMaskTex != rhi::kInvalidTexture)
        ? hairMaskTex
        : ((autoParseHairValid_ && wantAutoHair) ? autoParseHairTex_ : rhi::kInvalidTexture);
    const int64_t resolvedHairStampArg = (hairMaskTex != rhi::kInvalidTexture) ? hairMaskStampMs : autoParseHairTsMs_;
    const bool hasHairTex = (resolvedHairTex != rhi::kInvalidTexture);
    const int64_t effHairStamp = (resolvedHairStampArg != kTimeInvalid) ? resolvedHairStampArg : displayTsMs;
    // Active hair dye / shine is a core user-visible feature (not optional body warp): follow loadFades_.smooth()
    // so mild thermal loadLevel == 1 never silently switches hair coloring off.
    const float hairLoadKeep = wantAutoHair ? loadFades_.smooth() : loadFades_.body();
    const float hairLevel = hairGate_.update(
        hasHairTex, effHairStamp, displayTsMs,
        hairLoadKeep, primaryTl.latest(), primaryTl.missSince(), nowMs);
    const float neckWeight = (bodyState != nullptr && config_.neckLengthen > 0.0f) ? bodySmoother_.presence() : 0.0f;
    const float hairShown = hasHairTex ? hairGate_.shown(neckWeight) : 0.0f;
    const float hairBgShown = hasHairTex ? hairLevel : 0.0f;

    const bool shaking = cameraShaking_.load(std::memory_order_relaxed);
    if (shaking) {
        pipeline_.resetPixelTemporal();
    }
    const float shakeGate = shaking ? 0.0f : 1.0f;

    auto computePoseAffine = [&](int64_t stamp, std::array<float, math::HairMaskTuning::kAffineSize>& outAff) -> bool {
        math::HairMaskTuning::identity(outAff.data());
        if (state41 == nullptr || stamp == kTimeInvalid) return true;
        float poseDisplay[math::HairMaskTuning::kPoseSize];
        for (int i = 0; i < math::HairMaskTuning::kPoseSize; ++i) {
            poseDisplay[i] = state41[22 + i];
        }
        float poseMask[math::HairMaskTuning::kPoseSize];
        const int64_t newest = primaryTl.latest();
        bool hasMaskPose = false;
        if (newest != kTimeInvalid) {
            if (stamp <= newest) {
                hasMaskPose = primaryTl.poseAt(stamp, poseMask);
            } else if (primaryTl.poseAt(newest, poseMask)) {
                if (displayTsMs > newest) {
                    const float g = std::min(1.0f,
                        static_cast<float>(stamp - newest) / static_cast<float>(displayTsMs - newest));
                    for (int i = 0; i < math::HairMaskTuning::kPoseSize; ++i) {
                        poseMask[i] += g * (poseDisplay[i] - poseMask[i]);
                    }
                }
                hasMaskPose = true;
            }
        }
        if (hasMaskPose) {
            return math::HairMaskTuning::affine(poseMask, poseDisplay, aspect, outAff.data());
        }
        return (newest == kTimeInvalid);
    };

    std::array<float, math::HairMaskTuning::kAffineSize> hairAffine{};
    math::HairMaskTuning::identity(hairAffine.data());
    bool hairAffineValid = true;
    if (hasHairTex && hairLevel > 1e-4f) {
        hairAffineValid = computePoseAffine(effHairStamp, hairAffine);
    }

    std::array<float, math::HairMaskTuning::kAffineSize> occAffine{};
    math::HairMaskTuning::identity(occAffine.data());
    bool occAffineValid = true;
    if (autoOccTex_ != rhi::kInvalidTexture && autoOccTsMs_ != kTimeInvalid) {
        occAffineValid = computePoseAffine(autoOccTsMs_, occAffine);
    }

    rhi::FrameInputs in{};
    in.inputTexture = displayTex;
    in.width = width;
    in.height = height;
    in.nowMs = nowMs;
    in.outputRt = outputRt;
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
    in.denoiseKeep = primaryTl.denoiseKeep() * shakeGate;

    const float asyncMaskKeep = hairAffineValid ? (in.denoiseKeep * shakeGate) : 0.0f;
    // Active hair styling and background hair merge already track live head motion via rigid hairAffine;
    // decouple both from the 16ms denoiseKeep_ micro-motion gate so slight head sway never flashes hair masks.
    const float hairMotionKeep = (hairAffineValid ? 1.0f : 0.0f) * shakeGate * in.steadyFace;
    // v3.9.5 (seg-3): When hairMask is supplied solely for background bokeh (without hairDye/hairShine
    // or virtual background), feed it ONLY to kBgHairMerge (effHairBgShown) and never punch holes in rtSkinMask_.a.
    const bool blurOnlyHair = !wantAutoHair && config_.backgroundBlur > 1e-4f &&
                              config_.backgroundMode != BackgroundMode::kVirtualImage &&
                              config_.backgroundMode != BackgroundMode::kVirtualBlur;
    const float effHairShown = blurOnlyHair ? 0.0f : (hairShown * (wantAutoHair ? hairMotionKeep : asyncMaskKeep));
    const float effHairBgShown = hairBgShown * hairMotionKeep;

    in.secondaryFace.hasState = (secState41 != nullptr);
    in.secondaryFace.state41 = secState41;
    in.secondaryFace.hasMesh = hasSecMesh;
    in.secondaryFace.landmarks478x2 = hasSecMesh ? secMeshView.landmarks : nullptr;
    in.secondaryFace.meta15 = hasSecMesh ? secMeshView.meta : nullptr;
    in.secondaryFace.hasIris = hasSecMesh && secMeshView.hasIris;
    in.secondaryFace.presence = (effectiveMaxFaces >= 2) ? secondaryTl.presence() : 0.0f;
    in.secondaryFace.steadyFace = secondaryTl.steadyFace();
    in.secondaryFace.steadyEye = secondaryTl.steadyEye();
    in.secondaryFace.eyeMakeupKeep = secondaryTl.eyeMakeupKeep();
    in.primaryPromotionCount = promoCount;

    in.hasBody = (bodyState != nullptr);
    in.bodyState7 = bodyState;
    in.bodyPresence = bodySmoother_.presence();
    in.handShieldCoords = handShield;
    const bool bgMaskNeeded = config_.enabled && !config_.compareOriginal &&
        ((config_.backgroundMode == BackgroundMode::kVirtualImage ||
          config_.backgroundMode == BackgroundMode::kVirtualBlur) ||
         config_.backgroundBlur > 1e-4f);
    in.personMaskTex = (personMaskTex != rhi::kInvalidTexture)
        ? personMaskTex
        : ((autoParsePersonValid_ && (bgMaskNeeded || asyncMaskKeep > 1e-4f))
            ? autoParsePersonTex_
            : rhi::kInvalidTexture);
    in.hairMaskTex = (effHairShown > 1e-4f || effHairBgShown > 1e-4f) ? resolvedHairTex : rhi::kInvalidTexture;
    bool semanticOccAlive = false;
    {
        std::lock_guard<std::mutex> lock(auxMutex_);
        if (semanticOccActive_ && lastSemanticSubmitMs_ != kTimeInvalid &&
            std::llabs(displayTsMs - lastSemanticSubmitMs_) > 800) {
            // Multiclass stopped arriving (>800ms): yield cleanly back to fallback occlusionSolver_
            // and clear secondary-face occlusion state (v3.9.5 segmentation-c2).
            semanticOccActive_ = false;
            stagedSecLowerFaceOcc_ = 0.0f;
            secLowerFaceOcc_ = 0.0f;
        }
        semanticOccAlive = semanticOccActive_;
    }
    const bool wantAutoOcc = config_.enabled && !config_.compareOriginal && config_.hasMakeup();
    if (occlusionMaskTex_ != rhi::kInvalidTexture) {
        in.occlusionMaskTex = occlusionMaskTex_;
        in.occlusionShown = 1.0f;
    } else if (wantAutoOcc && (faceBoundsAlive || semanticOccAlive) && hasMesh && autoOccTex_ != rhi::kInvalidTexture) {
        const float occFresh = (autoOccTsMs_ != kTimeInvalid)
            ? math::HairMaskTuning::fresh(static_cast<float>(autoOccTsMs_ - displayTsMs))
            : 1.0f;
        const float occMotionKeep = (occAffineValid ? 1.0f : 0.0f) * shakeGate * in.steadyFace;
        in.occlusionShown = detail::clampf(occFresh * occMotionKeep, 0.0f, 1.0f);
        in.occlusionMaskTex = (in.occlusionShown > 1e-4f) ? autoOccTex_ : rhi::kInvalidTexture;
        if (in.occlusionShown > 1e-4f) {
            in.lowerMakeupKeep = math::BeautyQualityTuning::lowerFaceMakeupKeep(lowerFaceOcc_ * in.occlusionShown);
            if (hasSecMesh && effectiveMaxFaces >= 2) {
                const float secOccShown = detail::clampf(occFresh * shakeGate * in.secondaryFace.steadyFace, 0.0f, 1.0f);
                in.secondaryFace.lowerMakeupKeep =
                    math::BeautyQualityTuning::lowerFaceMakeupKeep(secLowerFaceOcc_ * secOccShown);
            }
        }
    }
    in.hairShown = effHairShown;
    in.hairBgShown = effHairBgShown;
    in.hairAffine = hairAffine;
    in.occAffine = occAffine;
    in.hasOccAffine = (autoOccTex_ != rhi::kInvalidTexture && autoOccTsMs_ != kTimeInvalid);
    in.ambientRgb = ambientRgb_;
    const bool wantAutoResidual = config_.enabled && !config_.compareOriginal &&
                                  config_.blemishRemove > 0.05f &&
                                  config_.backgroundMode == BackgroundMode::kNone;
    if (aiResidualTex_ != rhi::kInvalidTexture && aiResidualStrength_ > 1e-4f) {
        in.aiResidualTex = aiResidualTex_;
        in.aiResidualRect = aiResidualRect_;
        in.aiResidualStrength = aiResidualStrength_;
    } else if (wantAutoResidual && faceBoundsAlive && hasMesh && autoResidualTex_ != rhi::kInvalidTexture && autoResidualStrength_ > 1e-4f) {
        const float rcx = autoResidualRect_.left + 0.5f * autoResidualRect_.width;
        const float rcy = autoResidualRect_.top + 0.5f * autoResidualRect_.height;
        const float curCx = meshView.centerX();
        const float curCy = 1.0f - meshView.centerY();
        const float dx = curCx - rcx;
        const float dy = curCy - rcy;
        const float fw = std::max(meshView.faceWidth(), 0.04f);
        const float centerShift = std::hypot(dx, dy / std::max(aspect, 0.1f));
        const float shiftKeep = (1.0f - detail::smoothstep(0.025f * fw, 0.075f * fw, centerShift)) *
                                (1.0f - detail::smoothstep(0.12f, 0.30f, primaryTl.speed()));
        const float effResStrength = autoResidualStrength_ * shiftKeep * shakeGate * in.steadyFace;
        if (effResStrength > 1e-4f) {
            in.aiResidualTex = autoResidualTex_;
            in.aiResidualRect = autoResidualRect_;
            in.aiResidualRect.left += dx;
            in.aiResidualRect.top += dy;
            in.aiResidualStrength = effResStrength;
        }
    }
    std::array<math::StickerLayerConfig, math::kMaxStickerLayers> modulatedStickers = stickers_;
    for (int i = 0; i < stickerCount_; ++i) {
        if (!modulatedStickers[i].enabled) continue;
        if (stickerTriggers_[i] != bundle::BundleTriggerRule::kAlways) {
            const float gain = bundle::EffectBundleParser::evaluateTriggerGain(
                stickerTriggers_[i], exprReaction, activeGesture);
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
            const float gain = bundle::EffectBundleParser::evaluateTriggerGain(
                ar3dTriggers_[i], exprReaction, activeGesture);
            modulatedAr3d[i].opacity *= gain;
            if (modulatedAr3d[i].opacity <= 1e-4f) {
                modulatedAr3d[i].enabled = false;
                continue;
            }
        }
        if (expressionReactive_) {
            modulatedAr3d[i].scale *= exprReaction.scaleMultiplier;
            modulatedAr3d[i].offsetY += exprReaction.offsetYBoost;
            modulatedAr3d[i].opacity = detail::clampf(modulatedAr3d[i].opacity * exprReaction.opacityBoost, 0.0f, 1.0f);
        }
    }
    in.ar3dAccessories = modulatedAr3d;
    in.ar3dCount = ar3dCount_;
    in.particleVertices = (particleQuads > 0) ? particleVerts_.data() : nullptr;
    in.particleQuadCount = particleQuads;
    in.tierCaps = tierCaps;
    in.loadFadeBody = loadFades_.body();
    in.loadFadeFace = loadFades_.face();
    in.loadFadeSmooth = loadFades_.smooth();
    in.guidedSpread = guidedSpread_.value();
    in.lightLuma = lightLuma;
    in.bitratePressure = math::BitrateEffectsTuning::pressure(bitrateRatio);

    device_.beginFrameGpuTimer();
    const rhi::TextureId outTex = pipeline_.render(config_, in);
    device_.endFrameGpuTimer();

    const auto t1 = std::chrono::steady_clock::now();
    const int64_t cpuWorkNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int64_t gpuWorkNs = device_.lastGpuFrameTimeNs();
    const int64_t workNs = std::max(cpuWorkNs, gpuWorkNs);
    governor_.onFrame(frameTimestampMs * 1000000LL, workNs, processing, nowMs);

    const float frameMs = static_cast<float>(workNs) * 1e-6f;
    stats_.lastFrameMs = frameMs;
    stats_.emaFrameMs = (stats_.frameCount == 0) ? frameMs : (stats_.emaFrameMs * 0.9f + frameMs * 0.1f);
    if (lastStatsNowMs_ != kTimeInvalid && nowMs > lastStatsNowMs_) {
        const float instFps = 1000.0f / static_cast<float>(nowMs - lastStatsNowMs_);
        stats_.fps = (stats_.fps <= 0.0f) ? instFps : (stats_.fps * 0.9f + instFps * 0.1f);
    }
    lastStatsNowMs_ = nowMs;
    stats_.loadLevel = std::max(governor_.level(), hostLoadLevel_.load(std::memory_order_relaxed));
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
