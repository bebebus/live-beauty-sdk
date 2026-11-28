#include "livebeauty/rhi/render_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace livebeauty::rhi {

namespace {

constexpr int kMaskVertexCount = math::FaceMeshTopology::kFaceVertexCount + math::FaceMeshTopology::kOvalVertexCount;
constexpr int kCanonicalIndexCount = math::FaceMeshTopology::kCanonicalTriangleCount * 3;
constexpr int kInnerFeatherIndexCount = math::FaceMeshTopology::kInnerBufferTriangleCount * 3;

float gauss2d(float u, float v, float cu, float cv, float su, float sv) {
    const float du = (u - cu) / su;
    const float dv = (v - cv) / sv;
    return std::exp(-0.5f * (du * du + dv * dv));
}

void buildSkinMaskWeights(float* out504x3) {
    static const std::unordered_set<int> kEyeCore = {
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    };
    static const std::unordered_set<int> kBrowCore = {52, 65, 53, 105, 63, 282, 295, 283, 334, 293};
    static const std::unordered_set<int> kBrowBorder = {46, 55, 70, 66, 107, 276, 285, 300, 296, 336};
    static const std::unordered_set<int> kLipOuter = {
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    };
    std::unordered_set<int> lipInnerAndMid = {
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    };
    for (uint16_t s : math::FaceMeshTopology::kLipBandIndices) {
        if (!kLipOuter.count(s)) lipInnerAndMid.insert(s);
    }
    std::unordered_set<int> ovalSet(
        math::FaceMeshTopology::kFaceOval.begin(), math::FaceMeshTopology::kFaceOval.end());

    const auto& uv = math::FaceMeshTopology::kCanonicalUv;
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        const float u = uv[i * 2];
        const float v = uv[i * 2 + 1];
        float skin = 1.0f, excl = 0.0f;
        if (kEyeCore.count(i) || lipInnerAndMid.count(i)) {
            skin = 0.0f; excl = 1.0f;
        } else if (kLipOuter.count(i)) {
            skin = 0.22f; excl = 0.78f;
        } else if (kBrowCore.count(i)) {
            skin = 0.88f; excl = 0.52f;
        } else if (kBrowBorder.count(i)) {
            skin = 0.95f; excl = 0.20f;
        } else if (ovalSet.count(i)) {
            skin = 0.84f;
            excl = 0.0f;
        }
        const float bridge = gauss2d(u, v, 0.50f, 0.44f, 0.032f, 0.11f);
        const float forehead = gauss2d(u, v, 0.50f, 0.22f, 0.11f, 0.065f) * 0.85f;
        const float cheekR = gauss2d(u, v, 0.36f, 0.48f, 0.065f, 0.052f) * 0.85f;
        const float cheekL = gauss2d(u, v, 0.64f, 0.48f, 0.065f, 0.052f) * 0.85f;
        const float chin = gauss2d(u, v, 0.50f, 0.82f, 0.055f, 0.042f) * 0.70f;
        const float hiPlane = std::max({bridge, forehead, cheekR, cheekL, chin}) * skin;
        out504x3[i * 3 + 0] = skin;
        out504x3[i * 3 + 1] = excl;
        out504x3[i * 3 + 2] = hiPlane;
    }
    for (int j = 0; j < math::FaceMeshTopology::kOvalVertexCount; ++j) {
        const int base = (math::FaceMeshTopology::kMidBufferOffset + j) * 3;
        out504x3[base + 0] = 0.0f;
        out504x3[base + 1] = 0.0f;
        out504x3[base + 2] = 0.0f;
    }
}

}  // namespace

RenderPipeline::RenderPipeline(IRhiDevice& device) : device_(device) {}

RenderPipeline::~RenderPipeline() {
    releaseGpuResources();
}

void RenderPipeline::resetTemporal() {
    deformer_.resetTemporal();
    deformer2_.resetTemporal();
    relightSolver_.resetTemporal();
    denoiseHistoryValid_ = false;
    matteHistoryValid_ = false;
    for (auto& s : ar3dSprings_) s.reset();
    delayWriteIdx_ = 0;
    delayCount_ = 0;
    for (int i = 0; i < kMaxDelaySlots; ++i) delayStamps_[i] = kTimeInvalid;
    lastWroteOutputRt_ = false;
    lastValidOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
    lastValidAxis_ = {1.0f, 0.0f};
    bodyPriorPresence_ = 0.0f;
    lastFaceSeenMs_ = kTimeInvalid;
    hasLastValidLandmarks_ = false;
    lastValidFrontCw_ = false;
    secLastValidOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
    secLastValidAxis_ = {1.0f, 0.0f};
    secBodyPriorPresence_ = 0.0f;
    secLastFaceSeenMs_ = kTimeInvalid;
    hasSecLastValidLandmarks_ = false;
    secLastValidFrontCw_ = false;
    ghostOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
    ghostAxis_ = {1.0f, 0.0f};
    ghostPresence_ = 0.0f;
    lastPriorPromotionCount_ = 0;
    priorLastNowMs_ = kTimeInvalid;
    priorInitialized_ = false;
    neckBandRatio_ = 0.30f;
    chinClearanceState_ = 1.0f;
    neckBandValid_ = false;
    neckBandLastValidMs_ = kTimeInvalid;
    neckBandLastNowMs_ = kTimeInvalid;
}

void RenderPipeline::resetDeformerTemporal() {
    deformer_.resetTemporal();
    deformer2_.resetTemporal();
    relightSolver_.resetTemporal();
}

void RenderPipeline::resetPixelTemporal() {
    denoiseHistoryValid_ = false;
    matteHistoryValid_ = false;
}

void RenderPipeline::releaseGpuResources() {
    auto killRt = [&](RenderTargetId& rt) {
        if (rt != kInvalidRenderTarget) {
            device_.destroyRenderTarget(rt);
            rt = kInvalidRenderTarget;
        }
    };
    killRt(rtBody_);
    killRt(rtWarp_);
    killRt(rtComposite_);
    killRt(rtMakeup_);
    killRt(rtDenoise_[0]);
    killRt(rtDenoise_[1]);
    for (auto& rt : rtGuided_) killRt(rt);
    for (auto& rt : rtMacro_) killRt(rt);
    killRt(rtSkinMask_);
    killRt(rtSkinMaskTmp_);
    for (auto& rt : rtBg_) killRt(rt);
    killRt(rtBgHair_);
    killRt(rtBgOut_);
    for (auto& rt : rtMatte_) killRt(rt);
    for (auto& rt : rtMatteRgb_) killRt(rt);
    killRt(rtHeadDepth_);
    for (auto& rt : rtDelay_) killRt(rt);
    delayW_ = delayH_ = 0;
    delayWriteIdx_ = delayCount_ = 0;
    for (int i = 0; i < kMaxDelaySlots; ++i) delayStamps_[i] = kTimeInvalid;

    if (ownedFallbackTex_ != kInvalidTexture) {
        device_.destroyTexture(ownedFallbackTex_);
        ownedFallbackTex_ = kInvalidTexture;
    }

    auto killBuf = [&](BufferId& b) {
        if (b != kInvalidBuffer) {
            device_.destroyBuffer(b);
            b = kInvalidBuffer;
        }
    };
    killBuf(vbMeshWarp_);
    killBuf(vbSkinMask_);
    killBuf(vbNeckApron_);
    killBuf(vbRetouch_);
    killBuf(vbMakeup_);
    killBuf(vbSticker_);
    killBuf(vbAr3dHead_);
    killBuf(vbAr3dQuad_);
    killBuf(ibAr3dGlasses_);
    killBuf(ibAr3dTiara_);
    for (int i = 0; i < math::kMaxAr3dLayers; ++i) {
        killBuf(ibAr3dCustom_[i]);
        ibAr3dCustomPtr_[i] = nullptr;
        ibAr3dCustomCount_[i] = 0;
        ibAr3dCustomGen_[i] = 0;
    }
    killBuf(ibFullMesh_);
    killBuf(ibNeckApron_);
    killBuf(ibSkinRetouch_);
    killBuf(ibEyeOpening_);
    killBuf(ibInnerMouth_);
    killBuf(ibSkinFace_);
    killBuf(ibLipBand_);
    killBuf(ibRightEye_);
    killBuf(ibLeftEye_);
    killBuf(ibSticker_);
    killBuf(vbParticles_);
    killBuf(ibParticles_);

    staticBuffersReady_ = false;
    width_ = height_ = 0;
    halfFloatActive_ = false;
    denoiseHistoryValid_ = false;
    matteHistoryValid_ = false;
}

void RenderPipeline::ensureStaticBuffers() {
    if (!staticSkinWeightsReady_) {
        buildSkinMaskWeights(staticSkinWeights_.data());
        staticSkinWeightsReady_ = true;
    }
    if (staticBuffersReady_) return;

    vbMeshWarp_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceMeshTopology::kTotalVertexCount * 4);
    vbSkinMask_ = device_.createDynamicVertexBuffer(sizeof(float) * kMaskVertexCount * 5);
    vbNeckApron_ = device_.createDynamicVertexBuffer(sizeof(float) * neckApronVerts_.size());
    vbRetouch_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceRetouchSubMesh::kVertexDataSize);
    vbMakeup_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceMakeupUv::kVertexDataSize);
    vbSticker_ = device_.createDynamicVertexBuffer(sizeof(float) * math::StickerTransform::kQuadFloats);
    vbAr3dHead_ = device_.createDynamicVertexBuffer(sizeof(float) * math::Ar3dSolver::kHeadShellFloats);
    vbAr3dQuad_ = device_.createDynamicVertexBuffer(sizeof(float) * math::Ar3dSolver::kMaxMeshFloats);
    {
        const math::Ar3dMeshVertex* gv = nullptr;
        const uint16_t* gi = nullptr;
        int gvc = 0, gic = 0;
        math::Ar3dSolver::getCurvedGlassesMesh(gv, gvc, gi, gic);
        ibAr3dGlasses_ = device_.createStaticIndexBuffer(gi, static_cast<size_t>(gic));

        const math::Ar3dMeshVertex* tv = nullptr;
        const uint16_t* ti = nullptr;
        int tvc = 0, tic = 0;
        math::Ar3dSolver::getCurvedTiaraMesh(tv, tvc, ti, tic);
        ibAr3dTiara_ = device_.createStaticIndexBuffer(ti, static_cast<size_t>(tic));
    }

    ibFullMesh_ = device_.createStaticIndexBuffer(
        math::FaceMeshTopology::kFullIndices.data(),
        static_cast<int>(math::FaceMeshTopology::kFullIndices.size()));
    std::array<uint16_t, kNeckIndexCount> neckIdx{};
    int nIdx = 0;
    for (int r = 0; r < kNeckRows - 1; ++r) {
        for (int c = 0; c < kNeckJawCount - 1; ++c) {
            const uint16_t a = static_cast<uint16_t>(r * kNeckJawCount + c);
            const uint16_t b = static_cast<uint16_t>(a + 1);
            const uint16_t d = static_cast<uint16_t>((r + 1) * kNeckJawCount + c);
            const uint16_t e = static_cast<uint16_t>(d + 1);
            neckIdx[nIdx++] = a; neckIdx[nIdx++] = d; neckIdx[nIdx++] = b;
            neckIdx[nIdx++] = b; neckIdx[nIdx++] = d; neckIdx[nIdx++] = e;
        }
    }
    ibNeckApron_ = device_.createStaticIndexBuffer(neckIdx.data(), kNeckIndexCount);
    const auto& skinRetouch = math::FaceRetouchSubMesh::skinRetouchIndices();
    ibSkinRetouch_ = device_.createStaticIndexBuffer(skinRetouch.data(), static_cast<int>(skinRetouch.size()));
    const auto& eyeOpen = math::FaceMakeupUv::eyeOpeningIndices();
    ibEyeOpening_ = device_.createStaticIndexBuffer(eyeOpen.data(), static_cast<int>(eyeOpen.size()));
    const auto& innerMouth = math::FaceMakeupUv::innerMouthIndices();
    ibInnerMouth_ = device_.createStaticIndexBuffer(innerMouth.data(), static_cast<int>(innerMouth.size()));
    const auto& skinFace = math::FaceMakeupUv::skinFaceIndices();
    ibSkinFace_ = device_.createStaticIndexBuffer(skinFace.data(), static_cast<int>(skinFace.size()));
    ibLipBand_ = device_.createStaticIndexBuffer(
        math::FaceMeshTopology::kLipBandIndices.data(),
        static_cast<int>(math::FaceMeshTopology::kLipBandIndices.size()));
    ibRightEye_ = device_.createStaticIndexBuffer(
        math::FaceMeshTopology::kRightEyeIndices.data(),
        static_cast<int>(math::FaceMeshTopology::kRightEyeIndices.size()));
    ibLeftEye_ = device_.createStaticIndexBuffer(
        math::FaceMeshTopology::kLeftEyeIndices.data(),
        static_cast<int>(math::FaceMeshTopology::kLeftEyeIndices.size()));
    static constexpr uint16_t kStickerIndices[6] = {0, 1, 2, 2, 1, 3};
    ibSticker_ = device_.createStaticIndexBuffer(kStickerIndices, 6);
    vbParticles_ = device_.createDynamicVertexBuffer(
        sizeof(float) * static_cast<size_t>(math::ParticleBurstSolver::kMaxVertexFloats));
    const auto& particleIdx = math::ParticleBurstSolver::quadIndices();
    ibParticles_ = device_.createStaticIndexBuffer(particleIdx.data(), particleIdx.size());

    const uint8_t fallbackPx[4] = {0, 0, 0, 255};
    ownedFallbackTex_ = device_.createTexture2D(1, 1, TextureFormat::kRgba8, fallbackPx);

    staticBuffersReady_ = true;
}

void RenderPipeline::ensureResources(int width, int height, bool useHalfFloat) {
    ensureStaticBuffers();
    const bool sizeChanged = (width != width_ || height != height_);
    const bool fmtChanged = (useHalfFloat != halfFloatActive_);
    if (!sizeChanged && !fmtChanged) return;

    auto recreateFmt = [&](RenderTargetId& rt, int w, int h, TextureFormat fmt) {
        if (rt != kInvalidRenderTarget) device_.destroyRenderTarget(rt);
        rt = device_.createRenderTarget(w, h, fmt);
    };
    const TextureFormat interFmt = useHalfFloat ? TextureFormat::kRgba16f : TextureFormat::kRgba8;
    halfFloatActive_ = useHalfFloat;

    if (sizeChanged) {
        width_ = width;
        height_ = height;
        guidedW_ = std::max(1, (width + 2) / 3);
        guidedH_ = std::max(1, (height + 2) / 3);
        macroW_ = std::max(1, (guidedW_ + 2) / 3);
        macroH_ = std::max(1, (guidedH_ + 2) / 3);
        skinW_ = std::max(1, (width + 3) / 4);
        skinH_ = std::max(1, (height + 3) / 4);
        bgW_ = std::max(1, (width + 3) / 4);
        bgH_ = std::max(1, (height + 3) / 4);

        // Full-res warp/body targets stay RGBA8 (input is 8-bit SDR; avoids 2x full-res FP16 bandwidth).
        // Only downsampled dual-band guided filter targets (1/3 + 1/9 res) use RGBA16F for variance precision.
        recreateFmt(rtBody_, width_, height_, TextureFormat::kRgba8);
        recreateFmt(rtWarp_, width_, height_, TextureFormat::kRgba8);
        recreateFmt(rtComposite_, width_, height_, TextureFormat::kRgba8);
        recreateFmt(rtMakeup_, width_, height_, TextureFormat::kRgba8);
        // v3.9.5 (tone-2): Stage 0 8-bit temporal denoise removed (sub-0.5-LSB rounding & wasted warmup copy);
        // skipping rtDenoise_[0..1] saves 2x full-res RGBA8 render targets (~16.6MB @ 1080p).
        for (int i = 0; i < 3; ++i) recreateFmt(rtGuided_[i], guidedW_, guidedH_, interFmt);
        for (int i = 0; i < 2; ++i) recreateFmt(rtMacro_[i], macroW_, macroH_, interFmt);
        recreateFmt(rtSkinMask_, skinW_, skinH_, TextureFormat::kRgba8);
        recreateFmt(rtSkinMaskTmp_, skinW_, skinH_, TextureFormat::kRgba8);
        for (int i = 0; i < 2; ++i) recreateFmt(rtBg_[i], bgW_, bgH_, TextureFormat::kRgba8);
        recreateFmt(rtBgHair_, bgW_, bgH_, TextureFormat::kRgba8);
        recreateFmt(rtBgOut_, width_, height_, TextureFormat::kRgba8);
        for (int i = 0; i < 2; ++i) recreateFmt(rtMatte_[i], bgW_, bgH_, TextureFormat::kRgba8);
        for (int i = 0; i < 2; ++i) recreateFmt(rtMatteRgb_[i], bgW_, bgH_, TextureFormat::kRgba8);
        recreateFmt(rtHeadDepth_, skinW_, skinH_, TextureFormat::kRgba8);
        denoiseHistoryValid_ = false;
        matteHistoryValid_ = false;
    } else {
        // Format-only change: only recreate downsampled guided/macro targets; keep full-res targets & temporal history intact.
        for (int i = 0; i < 3; ++i) recreateFmt(rtGuided_[i], guidedW_, guidedH_, interFmt);
        for (int i = 0; i < 2; ++i) recreateFmt(rtMacro_[i], macroW_, macroH_, interFmt);
    }
}

void RenderPipeline::pushFrameDelay(
    TextureId inputTex, int64_t inputTsMs, int delayFrames, int width, int height,
    TextureId& outTex, int64_t& outTsMs) {
    const int d = std::clamp(delayFrames, 0, kMaxDelaySlots - 1);
    if (d <= 0 || inputTex == kInvalidTexture || width <= 0 || height <= 0) {
        outTex = inputTex;
        outTsMs = inputTsMs;
        return;
    }
    if (delayW_ != width || delayH_ != height || rtDelay_[0] == kInvalidRenderTarget) {
        for (int i = 0; i < kMaxDelaySlots; ++i) {
            if (rtDelay_[i] != kInvalidRenderTarget) device_.destroyRenderTarget(rtDelay_[i]);
            rtDelay_[i] = device_.createRenderTarget(width, height, TextureFormat::kRgba8);
            delayStamps_[i] = kTimeInvalid;
        }
        delayW_ = width;
        delayH_ = height;
        delayWriteIdx_ = 0;
        delayCount_ = 0;
    }

    const int writeSlot = delayWriteIdx_;
    RenderState st{};
    TextureBindings tb{};
    UniformBlock ub{};
    tb.bind(0, inputTex);
    device_.beginPass(rtDelay_[writeSlot], width, height, false);
    device_.drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);
    device_.endPass();
    delayStamps_[writeSlot] = inputTsMs;

    delayWriteIdx_ = (writeSlot + 1) % kMaxDelaySlots;
    if (delayCount_ < kMaxDelaySlots) ++delayCount_;

    const int effectiveDelay = std::min(d, delayCount_ - 1);
    const int readSlot = (writeSlot - effectiveDelay + kMaxDelaySlots) % kMaxDelaySlots;
    outTex = device_.textureOf(rtDelay_[readSlot]);
    outTsMs = (delayStamps_[readSlot] != kTimeInvalid) ? delayStamps_[readSlot] : inputTsMs;
}

void RenderPipeline::renderGuidedFilter(
    TextureId srcTex, int width, int height, float eps, float evenness, float spread,
    TextureId hairWeightTex) {
    RenderState st{};
    TextureBindings tb{};
    UniformBlock ub{};
    const bool hairWeighted = (hairWeightTex != kInvalidTexture);

    // P1: 1/3 downsample with 2/3 texel tap scale (exact 3x3 box filter).
    tb.bind(0, srcTex);
    ub.push(math::BeautyQualityTuning::kDownTapScale / static_cast<float>(width),
            math::BeautyQualityTuning::kDownTapScale / static_cast<float>(height), 0.0f, 0.0f);
    device_.beginPass(rtGuided_[0], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kDownsample, st, tb, ub);
    device_.endPass();

    // P2: Horizontal blur -> rtGuided_[1] (hair-weighted when semantic hair A channel is active)
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[0]));
    if (hairWeighted) tb.bind(1, hairWeightTex);
    ub.push(spread / static_cast<float>(guidedW_), 0.0f, 0.0f, 0.0f);
    device_.beginPass(rtGuided_[1], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(hairWeighted ? ShaderId::kGuidedWeightBlur : ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();

    // P3: Vertical blur + energy diff against rtGuided_[0] -> rtGuided_[2]
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[1]));
    tb.bind(1, device_.textureOf(rtGuided_[0]));
    ub.push(0.0f, spread / static_cast<float>(guidedH_), 0.0f, 0.0f);
    device_.beginPass(rtGuided_[2], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(hairWeighted ? ShaderId::kGuidedWeightDiff : ShaderId::kGuidedDiff, st, tb, ub);
    device_.endPass();

    // Dual-band 1/9 macro evenness (strictly isolated macroTextures to prevent FBO feedback loop).
    if (evenness > 0.0f) {
        tb.clear(); ub.clear();
        tb.bind(0, device_.textureOf(rtGuided_[2]));
        ub.push(math::BeautyQualityTuning::kDownTapScale / static_cast<float>(guidedW_),
                math::BeautyQualityTuning::kDownTapScale / static_cast<float>(guidedH_), 0.0f, 0.0f);
        device_.beginPass(rtMacro_[0], macroW_, macroH_, false);
        device_.drawFullScreenQuad(ShaderId::kDownsample, st, tb, ub);
        device_.endPass();

        tb.clear(); ub.clear();
        tb.bind(0, device_.textureOf(rtMacro_[0]));
        ub.push(1.55f / static_cast<float>(macroW_), 0.0f, 0.0f, 0.0f);
        device_.beginPass(rtMacro_[1], macroW_, macroH_, false);
        device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
        device_.endPass();

        tb.clear(); ub.clear();
        tb.bind(0, device_.textureOf(rtMacro_[1]));
        ub.push(0.0f, 1.55f / static_cast<float>(macroH_), 0.0f, 0.0f);
        device_.beginPass(rtMacro_[0], macroW_, macroH_, false);
        device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
        device_.endPass();
    }

    // P4: Blur energy alpha horizontally -> rtGuided_[1]
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[2]));
    ub.push(spread / static_cast<float>(guidedW_), 0.0f, 0.0f, 0.0f);
    device_.beginPass(rtGuided_[1], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurAlpha, st, tb, ub);
    device_.endPass();

    // P5: Guided coeff + macro evenness folded into (1-a)*evenMean -> rtGuided_[0]
    // Keep vertex uU0 (dir) and fragment uU1 (eps, evenness) strictly stage-disjoint so GLES2 mediump drivers always link.
    // Always bind isolated rtMacro_[0] texture even when evenness==0 to prevent GLES feedback loop.
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[1]));
    tb.bind(1, device_.textureOf(rtMacro_[0]));
    if (hairWeighted) tb.bind(2, hairWeightTex);
    ub.push(0.0f, spread / static_cast<float>(guidedH_), 0.0f, 0.0f);
    ub.push(eps, evenness, 0.0f, 0.0f);
    device_.beginPass(rtGuided_[0], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(hairWeighted ? ShaderId::kGuidedWeightCoeff : ShaderId::kGuidedCoeff, st, tb, ub);
    device_.endPass();

    // P6/P7: Final smooth of guided coefficients (b, a) -> rtGuided_[1] -> rtGuided_[0]
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[0]));
    ub.push(spread / static_cast<float>(guidedW_), 0.0f, 0.0f, 0.0f);
    device_.beginPass(rtGuided_[1], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();

    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[1]));
    ub.push(0.0f, spread / static_cast<float>(guidedH_), 0.0f, 0.0f);
    device_.beginPass(rtGuided_[0], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();
}

void RenderPipeline::renderSkinMask(
    const float* landmarks478x2, float axisX, float axisY, float yawWeight,
    bool frontFaceCw, int width, int height,
    float relightStrength, math::RelightStyle relightStyle, bool useRelight3d,
    float neckToneStrength, const float* bodyState7,
    bool clearFirst) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float axisNorm = std::hypot(axisX, axisY);
    const float ux = (axisNorm > 1e-4f) ? (axisX / axisNorm) : 1.0f;
    const float uy = (axisNorm > 1e-4f) ? (axisY / axisNorm) : 0.0f;
    const float vx = -uy;
    const float vy = ux;

    const float m234x = landmarks478x2[234 * 2], m234y = landmarks478x2[234 * 2 + 1] / aspect;
    const float m454x = landmarks478x2[454 * 2], m454y = landmarks478x2[454 * 2 + 1] / aspect;
    const float m10x = landmarks478x2[10 * 2], m10y = landmarks478x2[10 * 2 + 1] / aspect;
    const float m152x = landmarks478x2[152 * 2], m152y = landmarks478x2[152 * 2 + 1] / aspect;
    const float fw = std::max(std::hypot(m454x - m234x, m454y - m234y), 0.03f);
    const float fh = std::max(std::hypot(m10x - m152x, m10y - m152y), 0.03f);
    const float cx = 0.25f * (m234x + m454x + m10x + m152x);
    const float cy = 0.25f * (m234y + m454y + m10y + m152y);

    // P1: Real 3D rotated canonical vertex normals for pose-reactive studio relighting
    bool computed3dRelight = false;
    if (useRelight3d && relightStrength > 1e-4f) {
        const auto rawPose3d = math::FaceRelightSolver::estimateHeadPose(
            landmarks478x2, aspect, axisX, axisY, yawWeight);
        if (rawPose3d.valid) {
            const auto pose3d = relightSolver_.smoothPose(rawPose3d, clearFirst ? 0 : 1);
            relightSolver_.computeVertexRelight(
                pose3d, relightStyle, relightStrength, relightBuf_.data(), frontFaceCw);
            computed3dRelight = true;
            lastActivePassesMask_ |= kPassRelight3d;
        }
    }

    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        const int k = i * 5;
        const float skinW = staticSkinWeights_[i * 3 + 0];
        skinMaskVerts_[k + 0] = landmarks478x2[i * 2] * 2.0f - 1.0f;
        skinMaskVerts_[k + 1] = landmarks478x2[i * 2 + 1] * 2.0f - 1.0f;
        skinMaskVerts_[k + 2] = skinW;
        skinMaskVerts_[k + 3] = staticSkinWeights_[i * 3 + 1];
        skinMaskVerts_[k + 4] = computed3dRelight
            ? (relightBuf_[i] * skinW)
            : staticSkinWeights_[i * 3 + 2];
    }

    // Match SkinMaskPass.kt outer crown feather ring: canonical-radial hybrid direction + forehead crown extension.
    const float mirrorSign = frontFaceCw ? -1.0f : 1.0f;
    const auto& canonicalUv = math::FaceMeshTopology::kCanonicalUv;
    for (int j = 0; j < math::FaceMeshTopology::kOvalVertexCount; ++j) {
        const int oi = math::FaceMeshTopology::kFaceOval[j];
        const float ox = landmarks478x2[oi * 2];
        const float oy = landmarks478x2[oi * 2 + 1] / aspect;
        const float cNu = (canonicalUv[oi * 2] - 0.5f) * 2.0f * mirrorSign;
        const float cNv = (0.5f - canonicalUv[oi * 2 + 1]) * 2.0f;
        const float cLen = std::max(std::hypot(cNu, cNv), 1e-4f);
        const float cxDir = (cNu * ux + cNv * vx) / cLen;
        const float cyDir = (cNu * uy + cNv * vy) / cLen;
        const float dx = ox - cx;
        const float dy = oy - cy;
        const float len = std::hypot(dx, dy);
        const float radX = (len > 1e-4f) ? (dx / len) : cxDir;
        const float radY = (len > 1e-4f) ? (dy / len) : cyDir;
        const float align = detail::clampf(radX * cxDir + radY * cyDir, 0.0f, 1.0f);
        const float wRad = 0.65f * align;
        const float mx = cxDir * (1.0f - wRad) + radX * wRad;
        const float my = cyDir * (1.0f - wRad) + radY * wRad;
        const float mNorm = std::max(std::hypot(mx, my), 1e-4f);
        const float rx = mx / mNorm, ry = my / mNorm;
        const float dv = dx * vx + dy * vy;
        const float crownMix = detail::clampf((dv / fh - 0.22f) / 0.22f, 0.0f, 1.0f);
        const float pad = (1.0f - crownMix) * (0.045f * fw) + crownMix * (0.08f * fh);
        const float dirX = (1.0f - crownMix * 0.50f) * rx + (crownMix * 0.50f) * vx;
        const float dirY = (1.0f - crownMix * 0.50f) * ry + (crownMix * 0.50f) * vy;
        const float dirNorm = std::max(std::hypot(dirX, dirY), 1e-4f);
        const float fx = ox + (dirX / dirNorm) * pad;
        const float fy = (oy + (dirY / dirNorm) * pad) * aspect;
        const int vi = math::FaceMeshTopology::kMidBufferOffset + j;
        const int k = vi * 5;
        skinMaskVerts_[k + 0] = fx * 2.0f - 1.0f;
        skinMaskVerts_[k + 1] = fy * 2.0f - 1.0f;
        skinMaskVerts_[k + 2] = 0.0f;
        skinMaskVerts_[k + 3] = 0.0f;
        skinMaskVerts_[k + 4] = 0.0f;
    }

    device_.updateVertexBuffer(vbSkinMask_, skinMaskVerts_.data(), sizeof(float) * skinMaskVerts_.size());

    // P1: Seamless neck-face skin apron (strictly <= 0.40 faceM so CompositePro polyGate enforces skin-chroma gating
    // and guarantees 0/255 bleed on collars/shirts while unifying face & neck tone/smoothness).
    const float neckS = detail::clampf(neckToneStrength, 0.0f, 1.0f);
    const bool drawNeck = (neckS > 1e-4f);
    if (drawNeck) {
        static constexpr int kJawIndices[kNeckJawCount] = {
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
        };
        const float downX = -vx;
        const float downY = -vy;
        float neckLen = 0.58f * fh;
        if (bodyState7 && bodyState7[4] > 0.05f) {
            neckLen = std::min(0.65f * fh, 0.36f * bodyState7[4]);
        }
        const float rowDrop[kNeckRows] = {0.0f, 0.52f * neckLen, 1.0f * neckLen};
        const float rowTaper[kNeckRows] = {0.0f, 0.14f, 0.24f};
        const float rowWeight[kNeckRows] = {0.40f * neckS, 0.36f * neckS, 0.0f};
        for (int r = 0; r < kNeckRows; ++r) {
            for (int c = 0; c < kNeckJawCount; ++c) {
                const int lmIdx = kJawIndices[c];
                const float jx = landmarks478x2[lmIdx * 2];
                const float jy = landmarks478x2[lmIdx * 2 + 1] / aspect;
                // Taper gently toward chin vertical line so apron stays strictly on neck cylinder
                const float lateral = (jx - m152x) * ux + (jy - m152y) * uy;
                const float px = jx - ux * lateral * rowTaper[r] + downX * rowDrop[r];
                const float py = (jy - uy * lateral * rowTaper[r] + downY * rowDrop[r]) * aspect;
                const float tEdge = static_cast<float>(c) / static_cast<float>(kNeckJawCount - 1);
                const float sideFade = std::sin(3.14159265f * tEdge);
                const float w = rowWeight[r] * (0.25f + 0.75f * std::sqrt(std::max(sideFade, 0.0f)));
                const int k = (r * kNeckJawCount + c) * 5;
                neckApronVerts_[k + 0] = px * 2.0f - 1.0f;
                neckApronVerts_[k + 1] = py * 2.0f - 1.0f;
                neckApronVerts_[k + 2] = w;
                neckApronVerts_[k + 3] = 0.0f;
                neckApronVerts_[k + 4] = w * 0.30f;
            }
        }
        device_.updateVertexBuffer(vbNeckApron_, neckApronVerts_.data(), sizeof(float) * neckApronVerts_.size());
    }

    // Painter's order: neck apron first (no cull), then outer feather ring (no cull), then 898 face triangles (cull back).
    TextureBindings tb{};
    UniformBlock ub{};
    device_.beginPass(rtSkinMask_, skinW_, skinH_, clearFirst ? LoadOp::kClear : LoadOp::kLoad, {0.0f, 0.0f, 0.0f, 1.0f});

    RenderState featherState{};
    featherState.cullBack = false;
    if (drawNeck) {
        device_.drawIndexed(
            ShaderId::kSkinMask, featherState, vbNeckApron_, ibNeckApron_,
            0, kNeckIndexCount, VertexLayout::kSkinMaskPos2W3, tb, ub);
    }
    device_.drawIndexed(
        ShaderId::kSkinMask, featherState, vbSkinMask_, ibFullMesh_,
        kCanonicalIndexCount, kInnerFeatherIndexCount, VertexLayout::kSkinMaskPos2W3, tb, ub);

    RenderState innerState{};
    innerState.cullBack = true;
    innerState.frontFaceCw = frontFaceCw;
    device_.drawIndexed(
        ShaderId::kSkinMask, innerState, vbSkinMask_, ibFullMesh_,
        0, kCanonicalIndexCount, VertexLayout::kSkinMaskPos2W3, tb, ub);

    device_.endPass();
}

void RenderPipeline::blurSkinMask() {
    // 2-pass H+V Gaussian anti-alias feathering matching SkinMaskPass.kt (1.10 / skinW, 1.10 / skinH).
    RenderState st{};
    TextureBindings tb{};
    UniformBlock ub{};
    tb.bind(0, device_.textureOf(rtSkinMask_));
    ub.push(1.10f / static_cast<float>(skinW_), 0.0f, 0.0f, 0.0f);
    device_.beginPass(rtSkinMaskTmp_, skinW_, skinH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();

    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtSkinMaskTmp_));
    ub.push(0.0f, 1.10f / static_cast<float>(skinH_), 0.0f, 0.0f);
    device_.beginPass(rtSkinMask_, skinW_, skinH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();
}

void RenderPipeline::renderMakeup(
    RenderTargetId targetRt, TextureId baseTex, TextureId semanticTex, int width, int height,
    const BeautyConfig& cfg, const Vec3& ambientRgb, bool enableAmbient,
    const float* landmarks478x2, const float* meta15,
    bool hasIris, bool frontFaceCw, float presence, float steadyFace, float steadyEye, float eyeMakeupKeep,
    float lowerMakeupKeep, bool copyBaseFirst) {
    math::FaceMakeupUv::packVertices(landmarks478x2, math::FaceMeshFrame::kCoordCount, makeupVerts_.data());
    device_.updateVertexBuffer(vbMakeup_, makeupVerts_.data(), sizeof(float) * makeupVerts_.size());

    const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
        ? assets_.whiteAlphaFallbackTex
        : ownedFallbackTex_;
    const TextureId maskTex = (semanticTex != kInvalidTexture) ? semanticTex : fallbackTex;
    const float ambW = enableAmbient ? detail::clampf(cfg.ambientMakeupBlend, 0.0f, 1.0f) : 0.0f;
    const float invW = 1.0f / static_cast<float>(std::max(width, 1));
    const float invH = 1.0f / static_cast<float>(std::max(height, 1));

    // Copy base composite to targetRt first so layered alpha-blended makeup reads clean baseTex without feedback loop.
    device_.beginPass(targetRt, width, height, copyBaseFirst ? LoadOp::kDontCare : LoadOp::kLoad);
    if (copyBaseFirst) {
        RenderState copyState{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, copyState, tb, ub);
    }

    RenderState layerState{};
    layerState.blendEnable = true;
    layerState.cullBack = true;
    layerState.frontFaceCw = frontFaceCw;

    // Broad cheek/face makeup (blush, contour, highlight, lipstick) follows steadyFace * lowerMakeupKeep so brief
    // eye-radius extrapolation never pumps cheek beauty while heavy hand/cup lower-face occlusion mutes cleanly;
    // eyebrow follows steadyFace, eye makeup follows eyeMakeupKeep, and corneal lens/catchlight keeps strict steadyEye.
    const float faceGain = presence * steadyFace;
    const float lowerGain = faceGain * detail::clampf(lowerMakeupKeep, 0.0f, 1.0f);
    const float irisGain = presence * steadyEye;
    const float eyeGain = presence * eyeMakeupKeep;
    const int skinIdxCount = static_cast<int>(math::FaceMakeupUv::skinFaceIndices().size());

    auto drawSkinLayer = [&](TextureId atlas, Vec3 rgb, float alpha, Vec4 channel, Vec4 mode,
                             float poreKeep, float creaseShadow) {
        if (atlas == kInvalidTexture || alpha <= 1e-4f) return;
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        tb.bind(1, atlas);
        tb.bind(2, maskTex);
        ub.push(rgb.x, rgb.y, rgb.z, alpha);
        ub.push(channel);
        ub.push(mode);
        ub.push(ambientRgb.x, ambientRgb.y, ambientRgb.z, ambW);
        ub.push(invW, invH, poreKeep, creaseShadow);
        device_.drawIndexed(
            ShaderId::kMakeupLayer, layerState, vbMakeup_, ibSkinFace_,
            0, skinIdxCount, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
    };

    // 1. Blush (SoftLight mode=0, preserves micro-pore texture)
    if (cfg.blush > 0.0f) {
        drawSkinLayer(assets_.blushAtlas, math::FaceMakeupUv::blushRgb(cfg.blushColor),
                      cfg.blush * 0.72f * lowerGain, math::FaceMakeupUv::blushChannel(cfg.blushStyle),
                      {0.0f, 0.0f, 0.0f, 0.0f}, 0.55f, 0.0f);
    }
    // 2. Contour & Highlight (contour_highlight R=contour, G=highlight; mode.z=2 guards dark eye-socket/nose/beard shadows)
    if (cfg.contour > 0.0f) {
        drawSkinLayer(assets_.contourHighlightAtlas, {0.42f, 0.30f, 0.24f},
                      cfg.contour * 0.55f * lowerGain, {1.0f, 0.0f, 0.0f, 0.0f},
                      {1.0f, 0.0f, 2.0f, 0.0f}, 0.50f, 0.0f);
    }
    if (cfg.highlight > 0.0f) {
        drawSkinLayer(assets_.contourHighlightAtlas, {1.0f, 0.97f, 0.92f},
                      cfg.highlight * 0.60f * lowerGain, {0.0f, 1.0f, 0.0f, 0.0f},
                      {2.0f, 0.0f, 0.0f, 0.0f}, 0.45f, 0.0f);
    }
    // 3. Eyebrow (split-eye HD UV mode.y=1)
    if (cfg.eyebrow > 0.0f) {
        drawSkinLayer(assets_.eyebrowAtlas, math::FaceMakeupUv::eyebrowRgb(cfg.eyebrowColor),
                      cfg.eyebrow * 0.68f * faceGain, math::FaceMakeupUv::eyebrowChannel(cfg.eyebrowStyle),
                      {1.0f, 1.0f, 0.0f, 0.0f}, 0.35f, 0.0f);
    }
    // 4. Eyeshadow (dual layer with eyelid skin gate mode.z=1, crease micro-shadow & eyeMakeupKeep velocity fade)
    if (cfg.eyeshadow > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, math::FaceMakeupUv::eyeshadowMainRgb(cfg.eyeshadowColor),
                      cfg.eyeshadow * 0.64f * eyeGain, {1.0f, 0.0f, 0.0f, 0.0f},
                      {1.0f, 1.0f, 1.0f, 0.0f}, 0.45f, 0.45f);
        drawSkinLayer(assets_.eyeMakeupAtlas, math::FaceMakeupUv::eyeshadowDeepRgb(cfg.eyeshadowColor),
                      cfg.eyeshadow * 0.56f * eyeGain, {0.0f, 1.0f, 0.0f, 0.0f},
                      {1.0f, 1.0f, 1.0f, 0.0f}, 0.40f, 0.55f);
    }
    // 5. Eyeliner & Aegyo-sal
    if (cfg.eyeliner > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, {0.10f, 0.08f, 0.08f},
                      cfg.eyeliner * 0.78f * eyeGain, {0.0f, 0.0f, 1.0f, 0.0f},
                      {1.0f, 1.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
    }
    if (cfg.aegyoSal > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, {1.0f, 0.94f, 0.88f},
                      cfg.aegyoSal * 0.62f * eyeGain, {0.0f, 0.0f, 0.0f, 1.0f},
                      {2.0f, 1.0f, 0.0f, 0.0f}, 0.35f, 0.0f);
    }
    // 6. Physically-based Translucent Lipstick (120 lip-band triangles, open-mouth geometric cavity cutout, ambient WB & 6-DoF yaw glaze)
    const float rawSignedYaw = meta15 ? detail::clampf(meta15[math::FaceMeshFrame::kMetaSignedYaw], -1.0f, 1.0f) : 0.0f;
    const float uvLipYaw = (frontFaceCw ? 1.0f : -1.0f) * rawSignedYaw;
    if (cfg.lipstick > 0.0f && assets_.contourHighlightAtlas != kInvalidTexture) {
        const Vec3 lipRgb = math::FaceMakeupUv::lipstickRgb(cfg.lipstickColor);
        const float lipAlpha = cfg.lipstick * 0.75f * lowerGain;
        if (lipAlpha > 1e-4f) {
            const float aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1));
            const float mw = std::hypot(landmarks478x2[291 * 2] - landmarks478x2[61 * 2],
                                        (landmarks478x2[291 * 2 + 1] - landmarks478x2[61 * 2 + 1]) / aspect);
            const float mh = std::hypot(landmarks478x2[14 * 2] - landmarks478x2[13 * 2],
                                        (landmarks478x2[14 * 2 + 1] - landmarks478x2[13 * 2 + 1]) / aspect);
            const float openGate = math::BeautyQualityTuning::mouthOpenGate(mh / std::max(mw, 1e-3f));
            const float mCx = 0.5f * (landmarks478x2[13 * 2] + landmarks478x2[14 * 2]);
            const float mCy = 0.5f * (landmarks478x2[13 * 2 + 1] + landmarks478x2[14 * 2 + 1]);
            const float innerW = std::hypot(landmarks478x2[308 * 2] - landmarks478x2[78 * 2],
                                            (landmarks478x2[308 * 2 + 1] - landmarks478x2[78 * 2 + 1]) / aspect);
            const float mRx = std::max(innerW * 0.46f, 0.002f);
            const float mRy = std::max(mh * 0.48f, 0.001f);
            const float ax = meta15 ? meta15[math::FaceMeshFrame::kMetaAxisX] : 1.0f;
            const float ay = meta15 ? meta15[math::FaceMeshFrame::kMetaAxisY] : 0.0f;

            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, baseTex);
            tb.bind(1, assets_.contourHighlightAtlas);
            tb.bind(2, maskTex);
            ub.push(lipRgb.x, lipRgb.y, lipRgb.z, lipAlpha);
            ub.push(cfg.lipstickFinish == LipstickFinish::kGlossy ? 1.0f : 0.0f, invW, invH, uvLipYaw);
            ub.push(ambientRgb.x, ambientRgb.y, ambientRgb.z, ambW);
            ub.push(mCx, mCy, mRx, mRy);
            ub.push(openGate, aspect, ax, ay);
            device_.drawIndexed(
                ShaderId::kMakeupLip, layerState, vbMakeup_, ibLipBand_,
                0, math::FaceMeshTopology::kLipBandIndexCount, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
        }
    }
    // 7. Contact lens & corneal parallax catchlights (14+14 eye opening triangles, blink gated)
    if ((cfg.contactLens > 0.0f || cfg.catchlight > 0.0f) && hasIris && assets_.irisAtlas != kInvalidTexture) {
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const Vec2 tile = math::FaceMakeupUv::irisTile(cfg.contactLensStyle);
        // v3.9.6 (makeup-m1): Piecewise perceptual calibration keeps built-in presets (0.29..0.46)
        // in the exact calibrated range (0.19..0.25) while preserving full 1.0 reach at slider max.
        auto calIris = [](float s, float k) -> float {
            const float c = detail::clampf(s, 0.0f, 1.0f);
            return (c <= 0.5f) ? (k * c) : (0.5f * k + (1.0f - 0.5f * k) * (c - 0.5f) / 0.5f);
        };
        const float lensS = calIris(cfg.contactLens, 0.65f) * irisGain;
        const float clS = calIris(cfg.catchlight, 0.55f) * irisGain;

        auto drawEye = [&](int irisBase, BufferId eyeIb, int eyeCount, float openVal, float gateVal) {
            float circle[4];
            if (!math::FaceMakeupUv::computeIrisCircle(landmarks478x2, math::FaceMeshFrame::kCoordCount, irisBase, aspect, circle)) {
                return;
            }
            const float effOpen = openVal * gateVal;
            if (effOpen <= 0.05f) return;
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, baseTex);
            tb.bind(1, assets_.irisAtlas);
            ub.push(circle[0], circle[1], circle[2], circle[3]);
            ub.push(tile.x, tile.y, effOpen, rawSignedYaw);
            ub.push(lensS, clS, 0.0f, 0.0f);
            device_.drawIndexed(
                ShaderId::kMakeupIris, layerState, vbMakeup_, eyeIb,
                0, eyeCount, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
        };
        drawEye(468, ibRightEye_, math::FaceMeshTopology::kRightEyeIndexCount,
                meta15[math::FaceMeshFrame::kMetaRightEyeOpen], meta15[math::FaceMeshFrame::kMetaRightIrisGate]);
        drawEye(473, ibLeftEye_, math::FaceMeshTopology::kLeftEyeIndexCount,
                meta15[math::FaceMeshFrame::kMetaLeftEyeOpen], meta15[math::FaceMeshFrame::kMetaLeftIrisGate]);
    }

    device_.endPass();
}

void RenderPipeline::renderHairAlpha(
    TextureId hairMaskTex, float hairShown, const float* hairAffine8,
    TextureId occlusionMaskTex, float occlusionShown, const float* occAffine8,
    const float* ovalState41, float aspect, bool clearRgbToZero,
    bool meshWarped, bool frontFaceCw) {
    const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
        ? assets_.whiteAlphaFallbackTex
        : ownedFallbackTex_;
    const bool hasHair = (hairMaskTex != kInvalidTexture && hairShown > 1e-4f);
    const float occStrength = (occlusionMaskTex != kInvalidTexture)
        ? detail::clampf(occlusionShown, 0.0f, 1.0f)
        : 0.0f;
    const bool hasOcc = (occStrength > 1e-4f);

    RenderState st{};
    st.colorMaskAlphaOnly = !clearRgbToZero;
    TextureBindings tb{};
    UniformBlock ub{};
    tb.bind(0, hasHair ? hairMaskTex : fallbackTex);
    tb.bind(1, hasOcc ? occlusionMaskTex : fallbackTex);
    ub.push(hairAffine8[0], hairAffine8[1], hairAffine8[2], hasHair ? hairShown : 0.0f);
    ub.push(hairAffine8[4], hairAffine8[5], hairAffine8[6], occStrength);
    if (ovalState41) {
        ub.push(ovalState41[22], ovalState41[23], ovalState41[24], ovalState41[25]);
        ub.push(ovalState41[26], ovalState41[27], aspect, 1.0f);
    } else {
        ub.push(0.5f, 0.5f, 0.25f, 0.30f);
        ub.push(1.0f, 0.0f, aspect, 0.0f);
    }
    if (occAffine8) {
        ub.push(occAffine8[0], occAffine8[1], occAffine8[2], 1.0f);
        ub.push(occAffine8[4], occAffine8[5], occAffine8[6], 0.0f);
    } else {
        ub.push(1.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 1.0f, 0.0f, 0.0f);
    }
    device_.beginPass(rtSkinMask_, skinW_, skinH_, clearRgbToZero ? LoadOp::kClear : LoadOp::kLoad, {0.0f, 0.0f, 0.0f, 1.0f});
    device_.drawFullScreenQuad(ShaderId::kHairAlpha, st, tb, ub);
    if (meshWarped && (hasHair || hasOcc)) {
        // Follow primary face mesh deformation in alpha channel (matching SemanticAlphaPass.kt).
        RenderState outerSt = st;
        outerSt.colorMaskAlphaOnly = true;
        outerSt.cullBack = false;
        const int totalIndices = static_cast<int>(math::FaceMeshTopology::kFullIndices.size());
        device_.drawIndexed(
            ShaderId::kHairAlpha, outerSt, vbMeshWarp_, ibFullMesh_,
            kCanonicalIndexCount, totalIndices - kCanonicalIndexCount,
            VertexLayout::kMeshPos2Uv2, tb, ub);
        RenderState innerSt = outerSt;
        innerSt.cullBack = true;
        innerSt.frontFaceCw = frontFaceCw;
        device_.drawIndexed(
            ShaderId::kHairAlpha, innerSt, vbMeshWarp_, ibFullMesh_,
            0, kCanonicalIndexCount, VertexLayout::kMeshPos2Uv2, tb, ub);
    }
    device_.endPass();
}

TextureId RenderPipeline::renderTemporalMatte(TextureId rawMaskTex, TextureId currentRgbTex, int srcW, int srcH,
                                              float matteTemporal, float matteSpatial) {
    const int nextParity = 1 - matteParity_;
    // 4-tap box downsample of current RGB to bgW_ x bgH_ for anti-aliased RGB-guided bilateral & temporal motion diff.
    {
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentRgbTex);
        ub.push(1.0f / static_cast<float>(std::max(srcW, 1)),
                1.0f / static_cast<float>(std::max(srcH, 1)), 0.0f, 0.0f);
        device_.beginPass(rtMatteRgb_[nextParity], bgW_, bgH_, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kDownsample, st, tb, ub);
        device_.endPass();
    }
    {
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        const TextureId curRgbSmall = device_.textureOf(rtMatteRgb_[nextParity]);
        const TextureId prevMask = matteHistoryValid_ ? device_.textureOf(rtMatte_[matteParity_]) : rawMaskTex;
        const TextureId prevRgb = matteHistoryValid_ ? device_.textureOf(rtMatteRgb_[matteParity_]) : curRgbSmall;
        tb.bind(0, rawMaskTex);
        tb.bind(1, curRgbSmall);
        tb.bind(2, prevMask);
        tb.bind(3, prevRgb);
        // No 0.36 floor: when motion pushes matteTemporal -> 0, temporal blend drops cleanly to 0;
        // uU1.w carries matteSpatial [0..1] so overload L0<->L1 smoothly fades spatial bilateral back to raw cMask!
        const float temporalWeight = detail::clampf(matteTemporal, 0.0f, 0.84f);
        const float spatialWeight = detail::clampf(matteSpatial, 0.0f, 1.0f);
        ub.push(1.0f / static_cast<float>(bgW_), 1.0f / static_cast<float>(bgH_),
                temporalWeight, (matteHistoryValid_ && temporalWeight > 1e-4f) ? 1.0f : 0.0f);
        ub.push(0.055f, 0.022f, 0.080f, spatialWeight);
        device_.beginPass(rtMatte_[nextParity], bgW_, bgH_, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kMatteRefine, st, tb, ub);
        device_.endPass();
    }
    matteParity_ = nextParity;
    matteHistoryValid_ = (matteTemporal > 1e-4f && matteSpatial > 0.25f);
    return device_.textureOf(rtMatte_[matteParity_]);
}

void RenderPipeline::renderAr3dAccessories(
    RenderTargetId targetRt, TextureId baseTex, bool copyBaseFirst,
    int width, int height, const FrameInputs& in,
    const float* landmarks478x2, const math::HeadPose3D& pose,
    bool frontFaceCw, float presence) {
    if (!landmarks478x2 || !pose.valid || in.ar3dCount <= 0) return;
    const int n = std::min(in.ar3dCount, math::kMaxAr3dLayers);
    bool needHeadDepth = false;
    for (int i = 0; i < n; ++i) {
        const auto& layer = in.ar3dAccessories[i];
        if (layer.enabled && layer.texture != kInvalidTexture && layer.opacity > 1e-4f && layer.headOcclusion) {
            needHeadDepth = true;
            break;
        }
    }
    if (needHeadDepth) {
        math::Ar3dSolver::buildHeadDepthShell(landmarks478x2, pose, ar3dHeadVerts_.data(), frontFaceCw);
        device_.updateVertexBuffer(vbAr3dHead_, ar3dHeadVerts_.data(), sizeof(float) * ar3dHeadVerts_.size());
        RenderState depthSt{};
        depthSt.cullBack = true;
        depthSt.frontFaceCw = frontFaceCw;
        TextureBindings tb{};
        UniformBlock ub{};
        device_.beginPass(rtHeadDepth_, skinW_, skinH_, true, {0.0f, 0.0f, 0.0f, 1.0f});
        device_.drawIndexed(
            ShaderId::kHeadDepthShell, depthSt, vbAr3dHead_, ibFullMesh_,
            0, kCanonicalIndexCount, VertexLayout::kAr3dPos2Uv2Z1, tb, ub);
        device_.endPass();
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    device_.beginPass(targetRt, width, height, copyBaseFirst ? LoadOp::kDontCare : LoadOp::kLoad);
    if (copyBaseFirst) {
        RenderState copySt{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
    }
    RenderState st{};
    st.blendEnable = true;
    const TextureId headDepthTex = device_.textureOf(rtHeadDepth_);
    for (int i = 0; i < n; ++i) {
        const auto& layer = in.ar3dAccessories[i];
        if (!layer.enabled || layer.texture == kInvalidTexture || layer.opacity <= 1e-4f) continue;
        float anchorX = 0.5f, anchorY = 0.5f;
        int vertCount = 0;
        const uint16_t* meshIdx = nullptr;
        int idxCount = 0;
        if (!math::Ar3dSolver::buildAccessoryMesh(
                layer, landmarks478x2, pose, aspect, nullptr,
                ar3dMeshVerts_.data(), math::Ar3dSolver::kMaxMeshVertices,
                vertCount, meshIdx, idxCount, anchorX, anchorY, frontFaceCw)) {
            continue;
        }
        if (layer.springStiffness > 1e-3f) {
            ar3dSprings_[i].update(
                anchorX, anchorY, pose.rollRad, layer.springStiffness, layer.springDamping, in.nowMs);
            math::Ar3dSolver::buildAccessoryMesh(
                layer, landmarks478x2, pose, aspect, &ar3dSprings_[i],
                ar3dMeshVerts_.data(), math::Ar3dSolver::kMaxMeshVertices,
                vertCount, meshIdx, idxCount, anchorX, anchorY, frontFaceCw);
        }
        BufferId activeIb = ibSticker_;
        if (math::Ar3dSolver::validateCustomMesh(
                layer.customVertices, layer.customVertexCount,
                layer.customIndices, layer.customIndexCount)) {
            if (ibAr3dCustom_[i] == kInvalidBuffer ||
                ibAr3dCustomPtr_[i] != meshIdx ||
                ibAr3dCustomCount_[i] != idxCount ||
                ibAr3dCustomGen_[i] != layer.customMeshGeneration) {
                if (ibAr3dCustom_[i] != kInvalidBuffer) device_.destroyBuffer(ibAr3dCustom_[i]);
                ibAr3dCustom_[i] = device_.createStaticIndexBuffer(meshIdx, static_cast<size_t>(idxCount));
                ibAr3dCustomPtr_[i] = meshIdx;
                ibAr3dCustomCount_[i] = idxCount;
                ibAr3dCustomGen_[i] = layer.customMeshGeneration;
            }
            activeIb = ibAr3dCustom_[i];
        } else if (layer.meshPreset == math::Ar3dMeshPreset::kCurvedGlasses) {
            activeIb = ibAr3dGlasses_;
        } else if (layer.meshPreset == math::Ar3dMeshPreset::kCurvedTiara) {
            activeIb = ibAr3dTiara_;
        }

        device_.updateVertexBuffer(
            vbAr3dQuad_, ar3dMeshVerts_.data(),
            sizeof(float) * static_cast<size_t>(vertCount * math::Ar3dSolver::kFloatsPerVertex));
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, layer.texture);
        tb.bind(1, headDepthTex);
        ub.push(detail::clampf(layer.opacity * presence, 0.0f, 1.0f),
                layer.headOcclusion ? 1.0f : 0.0f, 0.0f, 0.0f);
        device_.drawIndexed(
            ShaderId::kAr3dMesh, st, vbAr3dQuad_, activeIb, 0, idxCount,
            VertexLayout::kAr3dPos2Uv2Z1, tb, ub);
    }
    device_.endPass();
}

void RenderPipeline::renderStickers(
    RenderTargetId targetRt, TextureId baseTex, bool copyBaseFirst,
    int width, int height, const FrameInputs& in,
    const float* landmarks478x2, const float* meta15, float presence) {
    if (!landmarks478x2 || !meta15 || in.stickerCount <= 0) return;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    device_.beginPass(targetRt, width, height, copyBaseFirst ? LoadOp::kDontCare : LoadOp::kLoad);
    if (copyBaseFirst) {
        RenderState copySt{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
    }
    RenderState st{};
    st.blendEnable = true;
    const int n = std::min(in.stickerCount, math::kMaxStickerLayers);
    float quadVerts[math::StickerTransform::kQuadFloats];
    for (int i = 0; i < n; ++i) {
        const auto& layer = in.stickers[i];
        if (!math::StickerTransform::buildQuadVertices(layer, landmarks478x2, meta15, aspect, quadVerts)) {
            continue;
        }
        device_.updateVertexBuffer(vbSticker_, quadVerts, sizeof(quadVerts));
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, layer.texture);
        ub.push(detail::clampf(layer.opacity * presence, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f);
        device_.drawIndexed(
            ShaderId::kStickerQuad, st, vbSticker_, ibSticker_, 0, 6,
            VertexLayout::kQuadPos2Uv2, tb, ub);
    }
    device_.endPass();
}

void RenderPipeline::renderParticles(
    RenderTargetId targetRt, TextureId baseTex, bool copyBaseFirst,
    int width, int height, const float* particleVerts, int quadCount) {
    if (!particleVerts || quadCount <= 0) return;
    const int nQuads = std::min(quadCount, math::ParticleBurstSolver::kMaxParticles);
    const size_t floatCount = static_cast<size_t>(
        nQuads * math::ParticleBurstSolver::kVertsPerQuad * math::ParticleBurstSolver::kFloatsPerVertex);
    device_.updateVertexBuffer(vbParticles_, particleVerts, sizeof(float) * floatCount);

    device_.beginPass(targetRt, width, height, copyBaseFirst ? LoadOp::kDontCare : LoadOp::kLoad);
    if (copyBaseFirst) {
        RenderState copySt{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
    }
    RenderState st{};
    st.blendEnable = true;
    TextureBindings emptyTb{};
    UniformBlock emptyUb{};
    device_.drawIndexed(
        ShaderId::kParticleBurst, st, vbParticles_, ibParticles_,
        0, nQuads * math::ParticleBurstSolver::kIndicesPerQuad,
        VertexLayout::kRetouchPos2Uv2A4, emptyTb, emptyUb);
    device_.endPass();
}

TextureId RenderPipeline::render(const BeautyConfig& config, const FrameInputs& in) {
    lastActivePassesMask_ = 0;
    lastWroteOutputRt_ = false;
    if (in.inputTexture == kInvalidTexture || in.width <= 0 || in.height <= 0) {
        return in.inputTexture;
    }
    bool hasAnySticker = false;
    const int stickerN = std::min(in.stickerCount, math::kMaxStickerLayers);
    for (int i = 0; i < stickerN; ++i) {
        if (in.stickers[i].enabled && in.stickers[i].texture != kInvalidTexture && in.stickers[i].opacity > 1e-4f) {
            hasAnySticker = true;
            break;
        }
    }
    bool hasAnyAr3d = false;
    const int ar3dN = std::min(in.ar3dCount, math::kMaxAr3dLayers);
    for (int i = 0; i < ar3dN; ++i) {
        if (in.ar3dAccessories[i].enabled && in.ar3dAccessories[i].texture != kInvalidTexture &&
            in.ar3dAccessories[i].opacity > 1e-4f) {
            hasAnyAr3d = true;
            break;
        }
    }
    const bool hasAiResidual = (in.aiResidualTex != kInvalidTexture && in.aiResidualStrength > 1e-4f);
    const bool hasParticles = (in.particleQuadCount > 0 && in.particleVertices != nullptr);
    if (!config.isProcessing() &&
        !(config.enabled && !config.compareOriginal &&
          (hasAnySticker || hasAnyAr3d || hasAiResidual || hasParticles))) {
        return in.inputTexture;
    }

    const bool useHalfFloat = config.preferHalfFloat && (in.tierCaps.hardwareTier != math::PerformanceTier::kLite);
    ensureResources(in.width, in.height, useHalfFloat);
    const float aspect = static_cast<float>(in.width) / static_cast<float>(in.height);
    TextureId currentTex = in.inputTexture;

    // 0. Stage 0 8-bit Temporal Denoise removed in v3.9.5 (tone-2):
    // 8-bit quantization rounded away <0.5 LSB corrections while paying full-res 10-tap + warmup PassThrough.
    denoiseHistoryValid_ = false;

    // 1. Optional Background Bokeh / Virtual Background / Studio Chroma Key Replacement
    // Privacy invariant: backgroundBlur and virtual background replacement must NEVER fade out
    // or expose the user's real room when loadLevel >= 1 (loadFadeBody -> 0). Under load, only
    // bokeh pass count / hair merge is simplified while keeping full privacy opacity.
    const float bgBlur = detail::clampf(config.backgroundBlur, 0.0f, 1.0f);
    const float vbgStrength = detail::clampf(config.virtualBgStrength, 0.0f, 1.0f);
    const bool wantChromaKey = (config.backgroundMode == BackgroundMode::kChromaKey) &&
                               assets_.virtualBgTex != kInvalidTexture &&
                               vbgStrength > 1e-4f;
    const bool wantVirtualImage = !wantChromaKey &&
                                  (config.backgroundMode == BackgroundMode::kVirtualImage ||
                                   config.backgroundMode == BackgroundMode::kVirtualBlur) &&
                                  assets_.virtualBgTex != kInvalidTexture &&
                                  vbgStrength > 1e-4f;
    const bool wantBgBlur = !wantChromaKey && !wantVirtualImage && bgBlur > 1e-4f;
    const float bgFacePresence = in.hasFaceState ? detail::clampf(in.facePresence, 0.0f, 1.0f) : 0.0f;
    const float ovalCx = in.hasFaceState ? in.faceState41[22] : 0.5f;
    const float ovalCy = in.hasFaceState ? in.faceState41[23] : 0.5f;
    const float ovalRx = in.hasFaceState ? in.faceState41[24] : 0.25f;
    const float ovalRy = in.hasFaceState ? in.faceState41[25] : 0.30f;
    const float ax = in.hasFaceState ? in.faceState41[26] : 1.0f;
    const float ay = in.hasFaceState ? in.faceState41[27] : 0.0f;
    const bool hasSecOval = in.secondaryFace.hasState && in.secondaryFace.state41 != nullptr &&
                            in.secondaryFace.presence > 1e-4f;
    const float secOvalCx = hasSecOval ? in.secondaryFace.state41[22] : 0.5f;
    const float secOvalCy = hasSecOval ? in.secondaryFace.state41[23] : 0.5f;
    const float secOvalRx = hasSecOval ? in.secondaryFace.state41[24] : 0.25f;
    const float secOvalRy = hasSecOval ? in.secondaryFace.state41[25] : 0.30f;
    const float secAx = hasSecOval ? in.secondaryFace.state41[26] : 1.0f;
    const float secAy = hasSecOval ? in.secondaryFace.state41[27] : 0.0f;
    const float bgSecPresence = hasSecOval ? detail::clampf(in.secondaryFace.presence, 0.0f, 1.0f) : 0.0f;
    if (wantChromaKey) {
        matteHistoryValid_ = false;
        const float bgAspect = (assets_.virtualBgAspect > 0.05f) ? assets_.virtualBgAspect : aspect;
        float uvScaleX = 1.0f, uvScaleY = 1.0f;
        if (bgAspect > aspect) {
            uvScaleX = aspect / bgAspect;
        } else {
            uvScaleY = bgAspect / aspect;
        }
        const float uvOffX = 0.5f * (1.0f - uvScaleX);
        const float uvOffY = 0.5f * (1.0f - uvScaleY);
        const bool isBlue = (config.chromaKeyColor == ChromaKeyColor::kBlue);
        const Vec3 keyRgb = isBlue ? Vec3{0.08f, 0.22f, 0.82f} : Vec3{0.05f, 0.72f, 0.10f};

        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, assets_.virtualBgTex);
        ub.push(vbgStrength,
                detail::clampf(config.chromaKeySimilarity, 0.05f, 0.95f),
                detail::clampf(config.chromaKeySmoothness, 0.01f, 0.80f),
                detail::clampf(config.chromaKeyDespill, 0.0f, 1.0f));
        ub.push(uvScaleX, uvScaleY, uvOffX, uvOffY);
        ub.push(keyRgb.x, keyRgb.y, keyRgb.z, isBlue ? 1.0f : 0.0f);
        ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
        ub.push(ax, ay, aspect, bgFacePresence);
        device_.beginPass(rtBgOut_, in.width, in.height, false);
        device_.drawFullScreenQuad(ShaderId::kChromaKey, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(rtBgOut_);
        lastActivePassesMask_ |= (kPassBg | kPassChromaKey);
    } else if ((wantVirtualImage || wantBgBlur) && in.personMaskTex != kInvalidTexture) {
        lastActivePassesMask_ |= kPassBg;

        // 1a. Optional Hair-Person Mask Merge (soft bokeh only; smoothly attenuated by loadFadeBody so no step jump at 0.5)
        TextureId effectiveMaskTex = in.personMaskTex;
        const float effHairBg = in.hairBgShown * detail::clampf(in.loadFadeBody, 0.0f, 1.0f);
        if (!wantVirtualImage && effHairBg > 1e-4f && in.hairMaskTex != kInvalidTexture) {
            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, in.personMaskTex);
            tb.bind(1, in.hairMaskTex);
            ub.push(in.hairAffine[0], in.hairAffine[1], in.hairAffine[2], effHairBg);
            ub.push(in.hairAffine[4], in.hairAffine[5], in.hairAffine[6], 0.0f);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, in.hasFaceState ? 1.0f : 0.0f);
            device_.beginPass(rtBgHair_, bgW_, bgH_, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgHairMerge, st, tb, ub);
            device_.endPass();
            effectiveMaskTex = device_.textureOf(rtBgHair_);
        }

        // 1b. P2: RGB-Guided Joint Bilateral + Motion-Adaptive Temporal Portrait Matting Refinement
        // v3.9.7 (segmentation-k1/gpuperf-k1): Smoothly fade BOTH temporal weight AND spatial bilateral weight (uU1.w)
        // with loadFadeBody before skipping kDownsample + kMatteRefine at <= 1e-3f (zero 4-16px edge jump on L0<->L1!).
        const float mt = detail::clampf(config.matteTemporal, 0.0f, 1.0f);
        const float matteSpatialFade = detail::smoothstep(0.05f, 0.95f, in.loadFadeBody);
        const float matteTemporalFade = detail::smoothstep(0.25f, 0.75f, in.loadFadeBody);
        const float effMatteTemporal =
            (mt > 1e-4f ? (0.36f + 0.48f * mt) : 0.0f) * detail::clampf(in.denoiseKeep, 0.0f, 1.0f) * matteTemporalFade;
        if (in.tierCaps.enableMatteRefine && matteSpatialFade > 1e-3f) {
            effectiveMaskTex = renderTemporalMatte(
                effectiveMaskTex, currentTex, in.width, in.height, effMatteTemporal, matteSpatialFade);
            lastActivePassesMask_ |= kPassMatteRefine;
        } else {
            matteHistoryValid_ = false;
        }

        if (wantVirtualImage) {
            TextureId bgSource = assets_.virtualBgTex;
            if (config.backgroundMode == BackgroundMode::kVirtualBlur && in.loadFadeBody > 0.5f) {
                RenderState st{};
                TextureBindings tb{};
                UniformBlock ub{};
                tb.bind(0, assets_.virtualBgTex);
                device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);
                device_.endPass();
                static constexpr float kVbgSteps[2] = {1.0f, 1.5f};
                for (int pass = 0; pass < 2; ++pass) {
                    const float stepScale = kVbgSteps[pass];
                    tb.clear(); ub.clear();
                    tb.bind(0, device_.textureOf(rtBg_[0]));
                    ub.push(stepScale / static_cast<float>(bgW_), 0.0f, 0.0f, 0.0f);
                    device_.beginPass(rtBg_[1], bgW_, bgH_, LoadOp::kDontCare);
                    device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                    device_.endPass();
                    tb.clear(); ub.clear();
                    tb.bind(0, device_.textureOf(rtBg_[1]));
                    ub.push(0.0f, stepScale / static_cast<float>(bgH_), 0.0f, 0.0f);
                    device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
                    device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                    device_.endPass();
                }
                bgSource = device_.textureOf(rtBg_[0]);
            }

            // Compute centered aspect-fill UV scale/offset for virtual background image
            const float bgAspect = (assets_.virtualBgAspect > 0.05f) ? assets_.virtualBgAspect : aspect;
            float uvScaleX = 1.0f, uvScaleY = 1.0f;
            if (bgAspect > aspect) {
                uvScaleX = aspect / bgAspect;
            } else {
                uvScaleY = bgAspect / aspect;
            }
            const float uvOffX = 0.5f * (1.0f - uvScaleX);
            const float uvOffY = 0.5f * (1.0f - uvScaleY);

            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, currentTex);
            tb.bind(1, bgSource);
            tb.bind(2, effectiveMaskTex);
            // Calibrated sigmoid mask gate [0.42, 0.72] suppresses sub-threshold room/shadow ripples to 0%,
            // while tight face-oval + throat shields prevent virtual bg from ever occluding tracked faces.
            ub.push(vbgStrength, 0.42f, 0.72f, bgFacePresence);
            ub.push(uvScaleX, uvScaleY, uvOffX, uvOffY);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, bgSecPresence);
            ub.push(secOvalCx, secOvalCy, secOvalRx, secOvalRy);
            ub.push(secAx, secAy, 0.0f, 0.0f);
            device_.beginPass(rtBgOut_, in.width, in.height, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
            device_.endPass();
            currentTex = device_.textureOf(rtBgOut_);
        } else {
            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, currentTex);
            tb.bind(1, effectiveMaskTex);
            ub.push(bgBlur, 0.38f, 0.68f, 0.0f);
            ub.push(1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height), 0.0f, 0.0f);
            device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgPremask, st, tb, ub);
            device_.endPass();

            // v3.9.5 (gpuperf-n1/segmentation-n1/n3): Strictly continuous & monotonic anti-aliasing bokeh ladder.
            // Normal (loadFadeBody=1): 3 passes (1.0, 1.0+0.5*b, 1.0+1.0*b), sigma 11.7..17.8px, zero jump at 0.55.
            // Overload transition: pass 2 step smoothly scales to 0 (identity) with loadFadeBody and skips once <= 0.02,
            // settling on privacy-safe 2-pass (1.0, 1.0) (sigma 9.6px) with zero visual step discontinuity!
            const float fadeB = detail::clampf(in.loadFadeBody, 0.0f, 1.0f);
            float steps[3] = {
                1.0f,
                1.0f + 0.50f * bgBlur * fadeB,
                (1.0f + 1.00f * bgBlur) * fadeB,
            };
            const int passCount = (steps[2] > 0.02f) ? 3 : 2;
            for (int pass = 0; pass < passCount; ++pass) {
                const float stepScale = steps[pass];
                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtBg_[0]));
                ub.push(stepScale / static_cast<float>(bgW_), 0.0f, 0.0f, 0.0f);
                device_.beginPass(rtBg_[1], bgW_, bgH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();

                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtBg_[1]));
                ub.push(0.0f, stepScale / static_cast<float>(bgH_), 0.0f, 0.0f);
                device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();
            }

            tb.clear(); ub.clear();
            tb.bind(0, currentTex);
            tb.bind(1, device_.textureOf(rtBg_[0]));
            tb.bind(2, effectiveMaskTex);
            ub.push(bgBlur, 0.38f, 0.68f, in.hasFaceState ? 0.65f : 0.0f);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, bgFacePresence);
            device_.beginPass(rtBgOut_, in.width, in.height, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgMix, st, tb, ub);
            device_.endPass();
            currentTex = device_.textureOf(rtBgOut_);
        }
    } else {
        matteHistoryValid_ = false;
    }

    // 2. Optional 5-axis Body Warp + Chin-Aware NeckLift alignment for face landmarks/state
    const float bodyMod = in.bodyPresence * in.loadFadeBody;
    float appliedNeck = 0.0f;
    float neckBandOverride = 0.0f;
    if (in.hasBody && in.bodyState7 && bodyMod > 1e-4f && config.hasBodyReshape()) {
        const float legStrength = config.legLengthen * 0.28f * bodyMod * in.bodyState7[6];
        const float waist = config.waistSlim * 0.16f * bodyMod;
        const float bust = config.bustEnhance * 0.14f * bodyMod;
        const float shoulder = config.shoulderSlim * 0.12f * bodyMod;
        float neck = config.neckLengthen * 0.22f * bodyMod;

        // v3.9.6 (shape-n3/shape-m4): Stateful 150ms smoothed chin-aware neckBand ratio AND chinClearance
        // prevent single-frame neck jump during head nod or transient face loss (< 1000ms).
        const float torso = in.bodyState7[4];
        const float axLen = std::hypot(in.bodyState7[2], in.bodyState7[3]);
        if (neck > 1e-4f && torso > 0.04f && axLen > 1e-4f) {
            const float dtNeckMs = (neckBandLastNowMs_ != kTimeInvalid && in.nowMs > neckBandLastNowMs_)
                ? static_cast<float>(std::min<int64_t>(in.nowMs - neckBandLastNowMs_, 250))
                : 33.0f;
            neckBandLastNowMs_ = in.nowMs;
            const float alphaNeck = 1.0f - std::exp(-dtNeckMs / 150.0f);
            bool gotChin = false;
            if (in.hasFaceMesh && in.landmarks478x2) {
                const float bax = in.bodyState7[2] / axLen;
                const float bay = in.bodyState7[3] / axLen;
                const float hipX = in.bodyState7[0];
                const float hipY = in.bodyState7[1] / aspect;
                const float chinX = in.landmarks478x2[152 * 2];
                const float chinY = in.landmarks478x2[152 * 2 + 1] / aspect;
                const float uChin = -((chinX - hipX) * bax + (chinY - hipY) * bay + torso);
                const float targetRatio = detail::clampf((uChin - 0.03f * torso) / torso, 0.06f, 0.30f);
                const float targetClearance = detail::smoothstep(0.02f * torso, 0.09f * torso, uChin);
                if (!neckBandValid_) {
                    neckBandRatio_ = targetRatio;
                    chinClearanceState_ = targetClearance;
                    neckBandValid_ = true;
                } else {
                    neckBandRatio_ += alphaNeck * (targetRatio - neckBandRatio_);
                    chinClearanceState_ += alphaNeck * (targetClearance - chinClearanceState_);
                }
                neckBandLastValidMs_ = in.nowMs;
                gotChin = true;
            }
            if (!gotChin) {
                // Hold last valid ratio & chinClearance during transient face loss (< 1000ms); ramp smoothly afterward.
                if (neckBandValid_ && neckBandLastValidMs_ != kTimeInvalid &&
                    in.nowMs - neckBandLastValidMs_ > 1000) {
                    const float alphaBack = 1.0f - std::exp(-dtNeckMs / 300.0f);
                    neckBandRatio_ += alphaBack * (0.30f - neckBandRatio_);
                    chinClearanceState_ += alphaBack * (1.0f - chinClearanceState_);
                }
            }
            neck *= (0.25f + 0.75f * detail::clampf(chinClearanceState_, 0.0f, 1.0f));
            neckBandOverride = neckBandRatio_ * torso;
        }
        appliedNeck = neck;

        const bool hasHandShield = (in.handShieldCoords.x >= 0.0f || in.handShieldCoords.z >= 0.0f);
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        ub.push(in.bodyState7[0], in.bodyState7[1],
                in.bodyState7[2] * (1.0f + legStrength), in.bodyState7[3] * (1.0f + legStrength));
        ub.push(in.bodyState7[4], in.bodyState7[5], waist, bust);
        ub.push(shoulder, neck, aspect, hasHandShield ? 1.0f : 0.0f);
        ub.push(in.handShieldCoords);
        ub.push(neckBandOverride, 0.0f, 0.0f, 0.0f);
        device_.beginPass(rtBody_, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kBodyWarp, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(rtBody_);
        lastActivePassesMask_ |= kPassBody;
    }

    // Apply NeckLift to face landmarks & 41-D state if neck lengthening ran.
    const float* activeLandmarks = in.landmarks478x2;
    const float* activeFaceState = in.faceState41;
    if (appliedNeck > 0.0f && in.bodyState7) {
        if (in.hasFaceMesh && in.landmarks478x2 && in.faceMeta15) {
            math::NeckLift::applyLandmarks(
                in.landmarks478x2, math::FaceMeshFrame::kCoordCount,
                in.faceMeta15[math::FaceMeshFrame::kMetaCenterX],
                in.faceMeta15[math::FaceMeshFrame::kMetaCenterY],
                in.bodyState7, appliedNeck, aspect, liftedLandmarks_.data(), neckBandOverride);
            activeLandmarks = liftedLandmarks_.data();
        }
        if (in.hasFaceState && in.faceState41) {
            math::NeckLift::applyFaceState(
                in.faceState41, in.bodyState7, appliedNeck, aspect, liftedState_.data(), neckBandOverride);
            activeFaceState = liftedState_.data();
        }
    }

    // 3. 15-axis 478-pt Diffeomorphic Face Mesh Warp + 468-pt In-Place Anatomical Retouch
    const float effFacePresence = in.facePresence * in.loadFadeFace;
    bool meshWarped = false;
    bool frontFaceCw = false;
    const float* postWarpLandmarks = activeLandmarks;

    const auto deformParams = math::FaceMeshDeformer::Params::fromConfig(config);
    const bool wantWarp = in.hasFaceMesh && activeLandmarks && in.faceMeta15 &&
                          effFacePresence > 1e-4f && deformParams.isActive();
    // v3.7.0 Implicit natural retouch synergy (strictly non-darkening & 0/255 on flat skin)
    const bool hasImplicitRetouch = config.enabled && !config.compareOriginal && config.smoothing > 1e-4f;
    const bool wantRetouch = in.hasFaceMesh && activeLandmarks && in.faceMeta15 &&
                             effFacePresence > 1e-4f && (config.hasRetouch() || hasImplicitRetouch);
    const auto computeMouthOpenGate = [aspect](const float* lm) -> float {
        const float mw = std::hypot(lm[291 * 2] - lm[61 * 2], (lm[291 * 2 + 1] - lm[61 * 2 + 1]) / aspect);
        const float mh = std::hypot(lm[14 * 2] - lm[13 * 2], (lm[14 * 2 + 1] - lm[13 * 2 + 1]) / aspect);
        return math::BeautyQualityTuning::mouthOpenGate(mh / std::max(mw, 1e-3f));
    };

    if (wantWarp) {
        meshWarped = deformer_.deform(
            activeLandmarks, aspect,
            in.faceMeta15[math::FaceMeshFrame::kMetaAxisX], in.faceMeta15[math::FaceMeshFrame::kMetaAxisY],
            in.faceMeta15[math::FaceMeshFrame::kMetaFaceWidth], in.faceMeta15[math::FaceMeshFrame::kMetaFaceHeight],
            in.faceMeta15[math::FaceMeshFrame::kMetaLeftWeight], in.faceMeta15[math::FaceMeshFrame::kMetaRightWeight],
            in.faceMeta15[math::FaceMeshFrame::kMetaYawWeight],
            effFacePresence, in.steadyFace, in.steadyEye, deformParams, in.nowMs);
    }
    if (meshWarped) {
        frontFaceCw = deformer_.frontFaceCw();
        postWarpLandmarks = deformer_.warpedLandmarks();
        if (in.faceMeta15 && math::FaceMeshFrame::fillGeometryState(
                postWarpLandmarks, math::FaceMeshFrame::kCoordCount,
                in.faceMeta15[math::FaceMeshFrame::kMetaAxisX],
                in.faceMeta15[math::FaceMeshFrame::kMetaAxisY],
                aspect, 0.0f, warpedState_.data())) {
            activeFaceState = warpedState_.data();
        }
    } else if (in.hasFaceMesh && activeLandmarks) {
        frontFaceCw = math::FaceMeshFrame::isFrontFaceCw(activeLandmarks, math::FaceMeshFrame::kCoordCount);
    }

    if (meshWarped || wantRetouch) {
        const TextureId sourceBeforeWarp = currentTex;
        device_.beginPass(rtWarp_, in.width, in.height, LoadOp::kDontCare);
        // When meshWarped is active, kMeshWarp covers 100% of [-1,1]^2 with fold-free triangles;
        // only draw fullscreen kPassThrough base when running retouch-only without mesh warp.
        if (!meshWarped) {
            RenderState copySt{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, sourceBeforeWarp);
            device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        if (meshWarped) {
            device_.updateVertexBuffer(
                vbMeshWarp_, deformer_.vertexData(),
                sizeof(float) * static_cast<size_t>(deformer_.vertexDataFloatCount()));
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, sourceBeforeWarp);

            // Painter's order: outer buffer + background first (no cull), then 898 foreground face triangles (cull back).
            RenderState outerSt{};
            outerSt.cullBack = false;
            const int totalIndices = static_cast<int>(math::FaceMeshTopology::kFullIndices.size());
            device_.drawIndexed(
                ShaderId::kMeshWarp, outerSt, vbMeshWarp_, ibFullMesh_,
                kCanonicalIndexCount, totalIndices - kCanonicalIndexCount,
                VertexLayout::kMeshPos2Uv2, tb, ub);

            RenderState innerSt{};
            innerSt.cullBack = true;
            innerSt.frontFaceCw = frontFaceCw;
            device_.drawIndexed(
                ShaderId::kMeshWarp, innerSt, vbMeshWarp_, ibFullMesh_,
                0, kCanonicalIndexCount, VertexLayout::kMeshPos2Uv2, tb, ub);
        }

        // In-place 468-pt anatomical sub-mesh retouch directly on rtWarp_ (zero extra 1080p FBO).
        if (wantRetouch) {
            const float baseDc = hasImplicitRetouch ? std::max(config.darkCircles, config.smoothing * 0.35f) : config.darkCircles;
            const float baseNf = hasImplicitRetouch ? std::max(config.nasolabial, config.smoothing * 0.25f) : config.nasolabial;
            const float baseBe = hasImplicitRetouch ? std::max(config.brightEyes, config.smoothing * 0.40f) : config.brightEyes;
            const float dc = detail::clampf(baseDc * effFacePresence * in.steadyFace, 0.0f, 1.0f);
            const float nf = detail::clampf(baseNf * effFacePresence * in.steadyFace, 0.0f, 1.0f);
            const float be = detail::clampf(baseBe * effFacePresence * in.steadyEye, 0.0f, 1.0f);
            const float mouthGate = (config.teethWhiten > 1e-4f) ? computeMouthOpenGate(postWarpLandmarks) : 0.0f;
            const float tw = detail::clampf(
                config.teethWhiten * mouthGate * effFacePresence * in.steadyEye * detail::clampf(in.lowerMakeupKeep, 0.0f, 1.0f),
                0.0f, 1.0f);
            const float ax = in.faceMeta15[math::FaceMeshFrame::kMetaAxisX];
            const float ay = in.faceMeta15[math::FaceMeshFrame::kMetaAxisY];
            const float norm = std::hypot(ax, ay);
            if (norm > 1e-4f) {
                const float ux = ax / norm, uy = ay / norm;
                math::FaceRetouchSubMesh::packVertices(
                    postWarpLandmarks, meshWarped ? deformer_.vertexData() : nullptr,
                    ux, uy, -uy, ux,
                    in.faceMeta15[math::FaceMeshFrame::kMetaFaceWidth],
                    in.faceMeta15[math::FaceMeshFrame::kMetaFaceHeight],
                    aspect, frontFaceCw ? -1.0f : 1.0f, dc, nf, retouchVerts_.data());
                device_.updateVertexBuffer(vbRetouch_, retouchVerts_.data(), sizeof(float) * retouchVerts_.size());

                RenderState retouchSt{};
                retouchSt.blendEnable = true;
                retouchSt.cullBack = true;
                retouchSt.frontFaceCw = frontFaceCw;
                TextureBindings tb{};
                tb.bind(0, sourceBeforeWarp);

                if (dc > 1e-4f || nf > 1e-4f) {
                    UniformBlock ub{};
                    ub.push(dc, nf, 0.0f, 0.0f);
                    device_.drawIndexed(
                        ShaderId::kRetouchSkin, retouchSt, vbRetouch_, ibSkinRetouch_,
                        0, static_cast<int>(math::FaceRetouchSubMesh::skinRetouchIndices().size()),
                        VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                }
                if (be > 1e-4f) {
                    UniformBlock ub{};
                    ub.push(be, 0.0f, 1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height));
                    device_.drawIndexed(
                        ShaderId::kRetouchEyeTeeth, retouchSt, vbRetouch_, ibEyeOpening_,
                        0, static_cast<int>(math::FaceMakeupUv::eyeOpeningIndices().size()),
                        VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                }
                if (tw > 1e-4f) {
                    UniformBlock ub{};
                    ub.push(tw, 1.0f, 0.0f, 0.0f);
                    device_.drawIndexed(
                        ShaderId::kRetouchEyeTeeth, retouchSt, vbRetouch_, ibInnerMouth_,
                        0, static_cast<int>(math::FaceMakeupUv::innerMouthIndices().size()),
                        VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                }
            }
        }
        device_.endPass();
        currentTex = device_.textureOf(rtWarp_);
        if (meshWarped) lastActivePassesMask_ |= kPassFaceWarp;
        if (wantRetouch) lastActivePassesMask_ |= kPassRetouch;
    }

    // 3b. Optional Secondary Face (Dual-Face Co-Streaming): independent warp + retouch reusing rtBody_ (zero extra FBO)
    const auto& sec = in.secondaryFace;
    const float effSecPresence = (config.maxFaces >= 2) ? (sec.presence * in.loadFadeFace) : 0.0f;
    const float effSmoothFacePresence = in.facePresence * in.loadFadeSmooth;
    const float effSmoothSecPresence = (config.maxFaces >= 2) ? (sec.presence * in.loadFadeSmooth) : 0.0f;
    bool secActive = sec.hasMesh && sec.landmarks478x2 && sec.meta15 && effSecPresence > 1e-4f;
    const bool secSmoothActive = sec.hasMesh && sec.landmarks478x2 && sec.meta15 && effSmoothSecPresence > 1e-4f;
    bool secWarped = false;
    bool secFrontCw = secSmoothActive
        ? math::FaceMeshFrame::isFrontFaceCw(sec.landmarks478x2, math::FaceMeshFrame::kCoordCount)
        : false;
    const float* postSecLandmarks = sec.landmarks478x2;
    if (secActive) {
        lastActivePassesMask_ |= kPassDualFace;
        const bool wantSecWarp = deformParams.isActive();
        const bool wantSecRetouch = config.hasRetouch() || hasImplicitRetouch;
        if (wantSecWarp) {
            secWarped = deformer2_.deform(
                sec.landmarks478x2, aspect,
                sec.meta15[math::FaceMeshFrame::kMetaAxisX], sec.meta15[math::FaceMeshFrame::kMetaAxisY],
                sec.meta15[math::FaceMeshFrame::kMetaFaceWidth], sec.meta15[math::FaceMeshFrame::kMetaFaceHeight],
                sec.meta15[math::FaceMeshFrame::kMetaLeftWeight], sec.meta15[math::FaceMeshFrame::kMetaRightWeight],
                sec.meta15[math::FaceMeshFrame::kMetaYawWeight],
                effSecPresence, sec.steadyFace, sec.steadyEye, deformParams, in.nowMs);
        }
        if (secWarped) {
            secFrontCw = deformer2_.frontFaceCw();
            postSecLandmarks = deformer2_.warpedLandmarks();
        } else {
            secFrontCw = math::FaceMeshFrame::isFrontFaceCw(sec.landmarks478x2, math::FaceMeshFrame::kCoordCount);
        }
        if (secWarped || wantSecRetouch) {
            // Ping-pong into rtBody_ when currentTex != rtBody_, else into rtWarp_ (strictly avoids feedback loop)
            const RenderTargetId secDstRt = (currentTex == device_.textureOf(rtBody_)) ? rtWarp_ : rtBody_;
            const TextureId srcBeforeSec = currentTex;
            device_.beginPass(secDstRt, in.width, in.height, LoadOp::kDontCare);
            if (!secWarped) {
                RenderState copySt{}; TextureBindings tb{}; UniformBlock ub{}; tb.bind(0, srcBeforeSec);
                device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
            }
            if (secWarped) {
                device_.updateVertexBuffer(
                    vbMeshWarp_, deformer2_.vertexData(),
                    sizeof(float) * static_cast<size_t>(deformer2_.vertexDataFloatCount()));
                TextureBindings tb{}; UniformBlock ub{}; tb.bind(0, srcBeforeSec);
                RenderState outerSt{}; outerSt.cullBack = false;
                const int totalIndices = static_cast<int>(math::FaceMeshTopology::kFullIndices.size());
                device_.drawIndexed(ShaderId::kMeshWarp, outerSt, vbMeshWarp_, ibFullMesh_,
                                    kCanonicalIndexCount, totalIndices - kCanonicalIndexCount,
                                    VertexLayout::kMeshPos2Uv2, tb, ub);
                RenderState innerSt{}; innerSt.cullBack = true; innerSt.frontFaceCw = secFrontCw;
                device_.drawIndexed(ShaderId::kMeshWarp, innerSt, vbMeshWarp_, ibFullMesh_,
                                    0, kCanonicalIndexCount, VertexLayout::kMeshPos2Uv2, tb, ub);
                lastActivePassesMask_ |= kPassFaceWarp;
            }
            if (wantSecRetouch) {
                const float baseDc = hasImplicitRetouch ? std::max(config.darkCircles, config.smoothing * 0.35f) : config.darkCircles;
                const float baseNf = hasImplicitRetouch ? std::max(config.nasolabial, config.smoothing * 0.25f) : config.nasolabial;
                const float baseBe = hasImplicitRetouch ? std::max(config.brightEyes, config.smoothing * 0.40f) : config.brightEyes;
                const float dc = detail::clampf(baseDc * effSecPresence * sec.steadyFace, 0.0f, 1.0f);
                const float nf = detail::clampf(baseNf * effSecPresence * sec.steadyFace, 0.0f, 1.0f);
                const float be = detail::clampf(baseBe * effSecPresence * sec.steadyEye, 0.0f, 1.0f);
                const float secMouthGate = (config.teethWhiten > 1e-4f) ? computeMouthOpenGate(postSecLandmarks) : 0.0f;
                const float tw = detail::clampf(config.teethWhiten * secMouthGate * effSecPresence * sec.steadyEye, 0.0f, 1.0f);
                const float ax = sec.meta15[math::FaceMeshFrame::kMetaAxisX];
                const float ay = sec.meta15[math::FaceMeshFrame::kMetaAxisY];
                const float norm = std::hypot(ax, ay);
                if (norm > 1e-4f) {
                    const float ux = ax / norm, uy = ay / norm;
                    math::FaceRetouchSubMesh::packVertices(
                        postSecLandmarks, secWarped ? deformer2_.vertexData() : nullptr,
                        ux, uy, -uy, ux,
                        sec.meta15[math::FaceMeshFrame::kMetaFaceWidth],
                        sec.meta15[math::FaceMeshFrame::kMetaFaceHeight],
                        aspect, secFrontCw ? -1.0f : 1.0f, dc, nf, retouchVerts_.data());
                    device_.updateVertexBuffer(vbRetouch_, retouchVerts_.data(), sizeof(float) * retouchVerts_.size());
                    RenderState retouchSt{}; retouchSt.blendEnable = true; retouchSt.cullBack = true; retouchSt.frontFaceCw = secFrontCw;
                    TextureBindings tb{}; tb.bind(0, srcBeforeSec);
                    if (dc > 1e-4f || nf > 1e-4f) {
                        UniformBlock ub{}; ub.push(dc, nf, 0.0f, 0.0f);
                        device_.drawIndexed(ShaderId::kRetouchSkin, retouchSt, vbRetouch_, ibSkinRetouch_,
                                            0, static_cast<int>(math::FaceRetouchSubMesh::skinRetouchIndices().size()),
                                            VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                    }
                    if (be > 1e-4f) {
                        UniformBlock ub{}; ub.push(be, 0.0f, 1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height));
                        device_.drawIndexed(ShaderId::kRetouchEyeTeeth, retouchSt, vbRetouch_, ibEyeOpening_,
                                            0, static_cast<int>(math::FaceMakeupUv::eyeOpeningIndices().size()),
                                            VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                    }
                    if (tw > 1e-4f) {
                        UniformBlock ub{}; ub.push(tw, 1.0f, 0.0f, 0.0f);
                        device_.drawIndexed(ShaderId::kRetouchEyeTeeth, retouchSt, vbRetouch_, ibInnerMouth_,
                                            0, static_cast<int>(math::FaceMakeupUv::innerMouthIndices().size()),
                                            VertexLayout::kRetouchPos2Uv2A4, tb, ub);
                    }
                    lastActivePassesMask_ |= kPassRetouch;
                }
            }
            device_.endPass();
            currentTex = device_.textureOf(secDstRt);
        }
    }

    // 3c. v3.9.7 (skin-k1/tone-k1/tone-k2/tone-m1): Compute portrait spatial prior, brief-miss hold,
    //     dual-face promotion swap, and ghost cross-fade BEFORE Stage 4 so polygon skin mask (with full
    //     lip/eye/brow exclM) stays rendered throughout brief-miss hold and ramps smoothly with neck!
    {
        constexpr int64_t kPriorHoldMs = 1200;
        const float dtPriorMs = (priorLastNowMs_ != kTimeInvalid && in.nowMs > priorLastNowMs_)
            ? static_cast<float>(std::min<int64_t>(in.nowMs - priorLastNowMs_, 250))
            : 33.0f;
        priorLastNowMs_ = in.nowMs;
        const bool coldStart = !priorInitialized_;
        priorInitialized_ = true;
        const float decayPrior = std::exp(-dtPriorMs / 350.0f);
        const float risePrior = 1.0f - std::exp(-dtPriorMs / 160.0f);

        if (!coldStart && in.primaryPromotionCount != lastPriorPromotionCount_) {
            std::swap(lastValidOval_, secLastValidOval_);
            std::swap(lastValidAxis_, secLastValidAxis_);
            std::swap(bodyPriorPresence_, secBodyPriorPresence_);
            std::swap(lastFaceSeenMs_, secLastFaceSeenMs_);
            std::swap(lastValidLandmarks_, secLastValidLandmarks_);
            std::swap(lastValidMeta_, secLastValidMeta_);
            std::swap(hasLastValidLandmarks_, hasSecLastValidLandmarks_);
            std::swap(lastValidFrontCw_, secLastValidFrontCw_);
        }
        lastPriorPromotionCount_ = in.primaryPromotionCount;

        if (ghostPresence_ > 0.001f) {
            ghostPresence_ *= decayPrior;
            if (ghostPresence_ < 0.002f) {
                ghostPresence_ = 0.0f;
                ghostOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // Always use un-dilated activeFaceState for lastValidOval_ so timeline clear never switches oval source!
        const bool hasPrimaryOval = (activeFaceState &&
                                     activeFaceState[24] > 0.001f &&
                                     activeFaceState[25] > 0.001f);
        const float targetP = detail::clampf(effSmoothFacePresence, 0.0f, 1.0f);
        if (hasPrimaryOval && in.facePresence >= 0.5f) {
            if (!coldStart && bodyPriorPresence_ > 0.05f && lastValidOval_[2] > 0.001f) {
                const float dcx = activeFaceState[22] - lastValidOval_[0];
                const float dcy = (activeFaceState[23] - lastValidOval_[1]) / std::max(aspect, 0.01f);
                const float dist = std::hypot(dcx, dcy);
                const float refRx = std::max(lastValidOval_[2], activeFaceState[24]);
                if (dist > 0.45f * refRx) {
                    ghostOval_ = lastValidOval_;
                    ghostAxis_ = lastValidAxis_;
                    ghostPresence_ = bodyPriorPresence_ * decayPrior;
                    bodyPriorPresence_ = 0.0f;
                }
            }
            lastValidOval_ = {activeFaceState[22], activeFaceState[23],
                              activeFaceState[24], activeFaceState[25]};
            lastValidAxis_ = {activeFaceState[26], activeFaceState[27]};
            lastFaceSeenMs_ = in.nowMs;
            if (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15) {
                std::memcpy(lastValidLandmarks_.data(), postWarpLandmarks,
                            sizeof(float) * math::FaceMeshFrame::kCoordCount);
                std::memcpy(lastValidMeta_.data(), in.faceMeta15,
                            sizeof(float) * math::FaceMeshFrame::kMetaSize);
                lastValidFrontCw_ = frontFaceCw;
                hasLastValidLandmarks_ = true;
            }
            if (coldStart) {
                bodyPriorPresence_ = targetP;
            } else if (targetP >= bodyPriorPresence_) {
                bodyPriorPresence_ += risePrior * (targetP - bodyPriorPresence_);
            } else {
                bodyPriorPresence_ = targetP + (bodyPriorPresence_ - targetP) * decayPrior;
            }
        } else {
            const bool inHold = (bodyPriorPresence_ > 0.001f && lastValidOval_[2] > 0.001f &&
                                 lastFaceSeenMs_ != kTimeInvalid &&
                                 in.nowMs >= lastFaceSeenMs_ &&
                                 (in.nowMs - lastFaceSeenMs_) <= kPriorHoldMs &&
                                 in.loadFadeSmooth > 0.05f);
            if (inHold) {
                const float holdCap = detail::clampf(in.loadFadeSmooth, 0.0f, 1.0f);
                if (bodyPriorPresence_ > holdCap) {
                    bodyPriorPresence_ = holdCap + (bodyPriorPresence_ - holdCap) * decayPrior;
                }
            } else {
                const float floorTarget = hasPrimaryOval ? targetP : 0.0f;
                if (floorTarget >= bodyPriorPresence_) {
                    bodyPriorPresence_ += risePrior * (floorTarget - bodyPriorPresence_);
                } else {
                    bodyPriorPresence_ = floorTarget + (bodyPriorPresence_ - floorTarget) * decayPrior;
                }
                if (bodyPriorPresence_ < 0.002f) {
                    bodyPriorPresence_ = 0.0f;
                    lastValidOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
                    lastFaceSeenMs_ = kTimeInvalid;
                    hasLastValidLandmarks_ = false;
                }
            }
        }

        const float* secPriorState = (hasSecOval && config.maxFaces >= 2) ? in.secondaryFace.state41 : nullptr;
        const bool hasSecValidOval = (secPriorState && secPriorState[24] > 0.001f && secPriorState[25] > 0.001f);
        const float targetS = detail::clampf(effSmoothSecPresence, 0.0f, 1.0f);
        if (hasSecValidOval && in.secondaryFace.presence >= 0.5f) {
            if (!coldStart && secBodyPriorPresence_ > 0.05f && secLastValidOval_[2] > 0.001f) {
                const float dcx = secPriorState[22] - secLastValidOval_[0];
                const float dcy = (secPriorState[23] - secLastValidOval_[1]) / std::max(aspect, 0.01f);
                const float dist = std::hypot(dcx, dcy);
                const float refRx = std::max(secLastValidOval_[2], secPriorState[24]);
                if (dist > 0.45f * refRx) {
                    secBodyPriorPresence_ = 0.0f;
                }
            }
            secLastValidOval_ = {secPriorState[22], secPriorState[23], secPriorState[24], secPriorState[25]};
            secLastValidAxis_ = {secPriorState[26], secPriorState[27]};
            secLastFaceSeenMs_ = in.nowMs;
            if (secSmoothActive && postSecLandmarks && sec.meta15) {
                std::memcpy(secLastValidLandmarks_.data(), postSecLandmarks,
                            sizeof(float) * math::FaceMeshFrame::kCoordCount);
                std::memcpy(secLastValidMeta_.data(), sec.meta15,
                            sizeof(float) * math::FaceMeshFrame::kMetaSize);
                secLastValidFrontCw_ = secFrontCw;
                hasSecLastValidLandmarks_ = true;
            }
            if (coldStart) {
                secBodyPriorPresence_ = targetS;
            } else if (targetS >= secBodyPriorPresence_) {
                secBodyPriorPresence_ += risePrior * (targetS - secBodyPriorPresence_);
            } else {
                secBodyPriorPresence_ = targetS + (secBodyPriorPresence_ - targetS) * decayPrior;
            }
        } else {
            const bool secInHold = (secBodyPriorPresence_ > 0.001f && secLastValidOval_[2] > 0.001f &&
                                    secLastFaceSeenMs_ != kTimeInvalid &&
                                    in.nowMs >= secLastFaceSeenMs_ &&
                                    (in.nowMs - secLastFaceSeenMs_) <= kPriorHoldMs &&
                                    in.loadFadeSmooth > 0.05f);
            if (secInHold) {
                const float holdCap = detail::clampf(in.loadFadeSmooth, 0.0f, 1.0f);
                if (secBodyPriorPresence_ > holdCap) {
                    secBodyPriorPresence_ = holdCap + (secBodyPriorPresence_ - holdCap) * decayPrior;
                }
            } else {
                const float floorTargetS = hasSecValidOval ? targetS : 0.0f;
                if (floorTargetS >= secBodyPriorPresence_) {
                    secBodyPriorPresence_ += risePrior * (floorTargetS - secBodyPriorPresence_);
                } else {
                    secBodyPriorPresence_ = floorTargetS + (secBodyPriorPresence_ - floorTargetS) * decayPrior;
                }
                if (secBodyPriorPresence_ < 0.002f) {
                    secBodyPriorPresence_ = 0.0f;
                    secLastValidOval_ = {0.0f, 0.0f, 0.0f, 0.0f};
                    secLastFaceSeenMs_ = kTimeInvalid;
                    hasSecLastValidLandmarks_ = false;
                }
            }
        }
    }

    // 4. Polygon Skin Mask (468+36 vertices + P1 3D normal relight + neck apron + 2-pass feather blur)
    //    + P2 unified Hair & Occlusion Semantic Alpha (mesh-warped)
    bool polySkinMaskReady = false;
    bool hasSemanticTex = false;
    const float effRelight = in.tierCaps.enableRelight3d
        ? detail::clampf(config.relight * in.loadFadeFace, 0.0f, 1.0f) : 0.0f;
    const float effNeckTone = in.tierCaps.enableNeckTone
        ? detail::clampf(std::max(config.neckTone, config.whiten * 0.45f * in.steadyFace) * in.loadFadeSmooth, 0.0f, 1.0f) : 0.0f;
    const float primaryPolyPres = (in.facePresence >= 0.5f)
        ? std::min(effSmoothFacePresence, bodyPriorPresence_)
        : bodyPriorPresence_;
    const float secPolyPres = (in.secondaryFace.presence >= 0.5f)
        ? std::min(effSmoothSecPresence, secBodyPriorPresence_)
        : secBodyPriorPresence_;
    const float* primaryMaskLm = (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && in.facePresence >= 0.5f)
        ? postWarpLandmarks
        : (hasLastValidLandmarks_ ? lastValidLandmarks_.data() : nullptr);
    const float* primaryMaskMeta = (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && in.facePresence >= 0.5f)
        ? in.faceMeta15
        : (hasLastValidLandmarks_ ? lastValidMeta_.data() : nullptr);
    const bool primaryMaskCw = (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && in.facePresence >= 0.5f)
        ? frontFaceCw : lastValidFrontCw_;
    // v3.9.6/v3.9.7 (tone-m2/skin-d1/skin-k1/tone-k1): Gate polygon skin mask on primaryPolyPres (which stays
    // active through L2 and brief-miss hold via cached landmarks) so lips/eyes/brows never lose exclM protection.
    if (primaryMaskLm && primaryMaskMeta && primaryPolyPres > 1e-4f) {
        renderSkinMask(primaryMaskLm,
                       primaryMaskMeta[math::FaceMeshFrame::kMetaAxisX],
                       primaryMaskMeta[math::FaceMeshFrame::kMetaAxisY],
                       primaryMaskMeta[math::FaceMeshFrame::kMetaYawWeight],
                       primaryMaskCw, in.width, in.height,
                       effRelight, config.relightStyle, in.tierCaps.enableRelight3d,
                       effNeckTone, in.hasBody ? in.bodyState7 : nullptr,
                       /*clearFirst=*/true);
        polySkinMaskReady = true;
        hasSemanticTex = true;
        lastActivePassesMask_ |= kPassSkinMask;
    }
    const float* secMaskLm = (secSmoothActive && postSecLandmarks && sec.meta15 && in.secondaryFace.presence >= 0.5f)
        ? postSecLandmarks
        : (hasSecLastValidLandmarks_ ? secLastValidLandmarks_.data() : nullptr);
    const float* secMaskMeta = (secSmoothActive && postSecLandmarks && sec.meta15 && in.secondaryFace.presence >= 0.5f)
        ? sec.meta15
        : (hasSecLastValidLandmarks_ ? secLastValidMeta_.data() : nullptr);
    const bool secMaskCw = (secSmoothActive && postSecLandmarks && sec.meta15 && in.secondaryFace.presence >= 0.5f)
        ? secFrontCw : secLastValidFrontCw_;
    if (secMaskLm && secMaskMeta && secPolyPres > 1e-4f) {
        renderSkinMask(secMaskLm,
                       secMaskMeta[math::FaceMeshFrame::kMetaAxisX],
                       secMaskMeta[math::FaceMeshFrame::kMetaAxisY],
                       secMaskMeta[math::FaceMeshFrame::kMetaYawWeight],
                       secMaskCw, in.width, in.height,
                       effRelight, config.relightStyle, in.tierCaps.enableRelight3d,
                       effNeckTone, nullptr,
                       /*clearFirst=*/!polySkinMaskReady);
        polySkinMaskReady = true;
        hasSemanticTex = true;
        lastActivePassesMask_ |= kPassSkinMask;
    }
    if (polySkinMaskReady) {
        blurSkinMask();
    }
    TextureId hairWeightTex = kInvalidTexture;
    // v3.9.7 (skin-e1): Keep 128^2 kHairAlpha & hair-weighted guided filter active at L2 when hairDye/hairShine
    // is active on Standard->Lite downshift so bangs/side hair never hard-cut +17lv on L2 entry/exit.
    const bool hasHairStyleActive = (config.hairDye > 1e-4f || config.hairShine > 1e-4f) && in.loadFadeSmooth > 1e-4f;
    const bool hasHairAlphaInput =
        (in.tierCaps.enableHairAlpha || hasHairStyleActive) &&
        in.hairMaskTex != kInvalidTexture && in.hairShown > 1e-4f;
    const bool hasOccAlphaInput = in.occlusionMaskTex != kInvalidTexture && in.occlusionShown > 1e-4f;
    bool primaryMeshVboRestored = false;
    if (hasHairAlphaInput) {
        // Re-upload primary mesh VBO if secondary face overwrote vbMeshWarp_ so hair alpha follows primary face warp.
        if (meshWarped && secWarped) {
            device_.updateVertexBuffer(
                vbMeshWarp_, deformer_.vertexData(),
                sizeof(float) * static_cast<size_t>(deformer_.vertexDataFloatCount()));
            primaryMeshVboRestored = true;
        }
        // Stage 4 writes ONLY real hairMask into rtSkinMask_.a (for Stage 5 hair-weighted guided filter).
        // Occlusion masks (6-class hair/clothes/accessories or chroma occluders) are deferred to Stage 6c
        // (after kCompositePro, right before Stage 7 makeup) so they NEVER punch dark un-whitened shadow
        // patches into face/hairline/glasses/collar skin during kCompositePro.
        renderHairAlpha(
            in.hairMaskTex, in.hairShown, in.hairAffine.data(),
            kInvalidTexture, 0.0f, nullptr,
            activeFaceState, aspect, !polySkinMaskReady, meshWarped, frontFaceCw);
        hasSemanticTex = true;
        hairWeightTex = device_.textureOf(rtSkinMask_);
        lastActivePassesMask_ |= kPassHairAlpha;
    }

    // 4b. P3: Multi-scale isolated blemish / acne / redness healer (strictly inside face skin mask, pre-guided)
    const float maxFacePresence = std::max(effFacePresence, effSecPresence);
    const float effBlemish = detail::clampf(config.blemishRemove * in.loadFadeBody, 0.0f, 1.0f);
    if (in.tierCaps.enableBlemishHeal && effBlemish > 1e-4f && polySkinMaskReady && maxFacePresence > 1e-4f) {
        const RenderTargetId dstRt = (currentTex == device_.textureOf(rtWarp_)) ? rtBody_ : rtWarp_;
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, device_.textureOf(rtSkinMask_));
        ub.push(1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height),
                effBlemish, maxFacePresence);
        device_.beginPass(dstRt, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kBlemishHeal, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(dstRt);
        lastActivePassesMask_ |= kPassBlemishHeal;
    }

    // 4c. P3: Pluggable Face-ROI Neural Residual Delta Fusion (strictly bounded & skin-mask gated)
    if (hasAiResidual) {
        const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
            ? assets_.whiteAlphaFallbackTex
            : ownedFallbackTex_;
        const RenderTargetId dstRt = (currentTex == device_.textureOf(rtWarp_)) ? rtBody_ : rtWarp_;
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, in.aiResidualTex);
        tb.bind(2, hasSemanticTex ? device_.textureOf(rtSkinMask_) : fallbackTex);
        ub.push(in.aiResidualRect.left, in.aiResidualRect.top,
                in.aiResidualRect.width, in.aiResidualRect.height);
        ub.push(detail::clampf(in.aiResidualStrength, 0.0f, 1.0f),
                polySkinMaskReady ? 1.0f : 0.0f, 0.0f, 0.0f);
        device_.beginPass(dstRt, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kAiResidualBlend, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(dstRt);
        lastActivePassesMask_ |= kPassAiResidual;
    }

    // 5. Dual-Band Guided Filter Pre-Pass (1/3 + 1/9, hair-weighted when hair semantic A channel is active)
    const float effSmooth = detail::clampf(config.smoothing * in.loadFadeSmooth, 0.0f, 1.0f);
    if (effSmooth > 1e-4f) {
        const float eps = math::BeautyQualityTuning::guidedEps(effSmooth);
        const float evenness = (in.tierCaps.hardwareTier != math::PerformanceTier::kLite)
            ? (math::BeautyQualityTuning::macroEvenness(effSmooth) * in.loadFadeFace)
            : 0.0f;
        renderGuidedFilter(currentTex, in.width, in.height, eps, evenness, in.guidedSpread, hairWeightTex);
        lastActivePassesMask_ |= kPassGuided;
    }

    // Pre-compute post-composite stage activity so the true final full-res pass can render directly
    // into in.outputRt (eliminates 1 full-res 1080p host blit when outputRt is provided).
    static const std::string kEmptyLutUrl;
    const bool preLutBound = (assets_.lutTex != kInvalidTexture && !config.lutUrl.empty());
    const float preLutStr = preLutBound ? detail::clampf(config.lutIntensity, 0.0f, 1.0f) : 0.0f;
    float preAutoBloom = 0.0f, preAutoGrain = 0.0f;
    math::BeautyQualityTuning::resolveStyleAtmosphere(
        preLutBound ? config.lutUrl : kEmptyLutUrl, preLutStr,
        static_cast<int>(config.filter), config.filterIntensity,
        preAutoBloom, preAutoGrain);
    const bool preAllowBloom = (in.tierCaps.resolvedTier != math::PerformanceTier::kLite);
    const float preEffBloom = preAllowBloom
        ? detail::clampf(std::max(config.bloom, preAutoBloom) * in.loadFadeBody, 0.0f, 1.0f)
        : 0.0f;
    const float preEffGrain = detail::clampf(std::max(config.filmGrain, preAutoGrain) * in.loadFadeSmooth, 0.0f, 0.50f);
    float autoTemp = 0.0f, autoTint = 0.0f;
    math::BeautyQualityTuning::estimateAutoWhiteBalance(in.ambientRgb, config.autoWhiteBalance, autoTemp, autoTint);
    const float preEffTemp = detail::clampf(config.colorTemp + autoTemp, -1.0f, 1.0f);
    const float preEffTint = detail::clampf(config.colorTint + autoTint, -1.0f, 1.0f);
    const bool preHasCat02 = (std::fabs(preEffTemp) > 1e-4f || std::fabs(preEffTint) > 1e-4f);

    const float effHairDye = detail::clampf(config.hairDye * in.loadFadeSmooth, 0.0f, 1.0f);
    const float effHairShine = detail::clampf(config.hairShine * in.loadFadeSmooth, 0.0f, 1.0f);
    const bool willRunHairDye = (effHairDye > 1e-4f || effHairShine > 1e-4f) &&
                                in.hairMaskTex != kInvalidTexture && in.hairShown > 1e-4f;
    const bool willRenderPrimaryMakeup =
        in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && effFacePresence > 1e-4f && config.hasMakeup();
    const bool willRenderSecondaryMakeup =
        secActive && postSecLandmarks && config.hasMakeup();
    const bool willRunPrimarySticker =
        hasAnySticker && in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && effFacePresence > 1e-4f;
    const bool willRunSecondarySticker =
        hasAnySticker && secActive && postSecLandmarks;
    math::HeadPose3D cachedAr3dPose{};
    const bool willRunAr3d =
        in.tierCaps.enableAr3d && hasAnyAr3d && in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 &&
        effFacePresence > 1e-4f &&
        (cachedAr3dPose = math::FaceRelightSolver::estimateHeadPose(
             postWarpLandmarks, aspect,
             in.faceMeta15[math::FaceMeshFrame::kMetaAxisX],
             in.faceMeta15[math::FaceMeshFrame::kMetaAxisY],
             in.faceMeta15[math::FaceMeshFrame::kMetaYawWeight]),
         cachedAr3dPose.valid);
    const bool willRunParticles = hasParticles && config.enabled && !config.compareOriginal;
    const bool willRunStage7 = willRenderPrimaryMakeup || willRenderSecondaryMakeup ||
                               willRunPrimarySticker || willRunSecondarySticker ||
                               willRunAr3d || willRunParticles;
    const bool hasPostCompositeLayers = willRunHairDye || willRunStage7;

    const bool fuseAtmInComposite = config.enabled && !config.compareOriginal &&
        preEffBloom <= 1e-4f && (preEffGrain > 1e-4f || preHasCat02) && !hasPostCompositeLayers;
    const bool willRunStage8 = config.enabled && !config.compareOriginal && !fuseAtmInComposite &&
                               (preEffBloom > 1e-4f || preEffGrain >= 0.17f || preHasCat02);

    // 6. Main Composite Pro + 64^3 3D LUT (13 vec4 + 4 samplers)
    {
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
            ? assets_.whiteAlphaFallbackTex
            : ownedFallbackTex_;
        const TextureId lutTex = (assets_.lutTex != kInvalidTexture) ? assets_.lutTex : fallbackTex;
        const float lutStrength = preLutStr;

        tb.bind(0, currentTex);
        tb.bind(1, effSmooth > 1e-4f ? device_.textureOf(rtGuided_[0]) : currentTex);
        tb.bind(2, hasSemanticTex ? device_.textureOf(rtSkinMask_) : fallbackTex);
        tb.bind(3, lutTex);

        // v3.9.6 (tone-m2/skin-d1/tone-m1/skin-m3/skin-m5/skin-m7):
        // - Track portrait prior against effSmoothFacePresence (loadFadeSmooth, NOT loadFadeFace) so L2 keeps
        //   face & neck smoothing/whitening active while only L3 fades them out.
        // - Hold last valid oval & presence for 1200ms during brief head turns/look-down so beauty never breathes,
        //   then decay smoothly (tau=350ms) to 0 for genuinely empty rooms; couple polygon faceM reacquire with
        //   bodyPriorPresence_ so post-long-absence reacquire ramps smoothly with zero jaw-neck seam.
        // - Pack explicit secondary prior flag into uU5.z whenever polySkinMaskReady || framePresence <= 0.001f.
        // - Pass pure lightLuma-derived lowLightT into uU12.w (decoupled from bitrate pressure).
        const float activeStyle = (lutStrength > 0.0f) ? 0.0f : static_cast<float>(static_cast<int>(config.filter));
        const float activeFilterIntensity = (lutStrength > 0.0f) ? 0.0f : config.filterIntensity;
        const float stylePacked = activeStyle + (config.skinOnly ? 16.0f : 0.0f);
        const float effTexture = math::BitrateEffectsTuning::texture(config.skinTexture, in.bitratePressure);
        const float effSharpness = math::BitrateEffectsTuning::sharpness(config.sharpness, in.bitratePressure);
        const float coreThresh = math::BeautyQualityTuning::coreThreshold(in.lightLuma, in.bitratePressure);
        const float effOilControl = in.tierCaps.enableOilControl
            ? detail::clampf(config.oilControl * in.loadFadeSmooth, 0.0f, 1.0f) : 0.0f;
        const float microDetailKeep = effTexture * 0.45f;
        const float* fallbackEllipseState = (!meshWarped && appliedNeck <= 0.0f && in.dilatedFaceState41)
            ? in.dilatedFaceState41 : activeFaceState;

        // Default auto ambient scene lift disabled (prevents greying/brightening real room background on AE cameras).
        const float ambientLift = 0.0f;

        float cat02[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float fusedCat02Full = 0.0f;
        float fusedPackedGrain = 0.0f;
        if (fuseAtmInComposite) {
            if (preHasCat02) {
                math::BeautyQualityTuning::cat02WhiteBalanceMatrix(preEffTemp, preEffTint, cat02);
                fusedCat02Full = 1.0f;
            }
            const float grainPhase = std::fmod(static_cast<float>(std::max<int64_t>(0, in.nowMs) % 60000LL) * 0.018f, 1.0f);
            fusedPackedGrain = (preEffGrain > 1e-4f)
                ? (std::floor(grainPhase * 63.0f) + detail::clampf(preEffGrain, 0.0f, 0.50f))
                : 0.0f;
            lastActivePassesMask_ |= kPassAtmosphere;
        }

        const float bodyPriorMute = 1.0f - detail::clampf(bodyPriorPresence_, 0.0f, 1.0f);
        const float smoothPolyPres = std::max(primaryPolyPres, secPolyPres);
        const float framePresence = (polySkinMaskReady && smoothPolyPres > 0.0f)
            ? (smoothPolyPres + 4.0f) : smoothPolyPres;

        const bool hasSecSlot = (secBodyPriorPresence_ > 0.001f && secLastValidOval_[2] > 0.001f);
        const bool hasGhostSlot = (!hasSecSlot && ghostPresence_ > 0.001f && ghostOval_[2] > 0.001f);
        const bool packSecPrior = (hasSecSlot || hasGhostSlot) &&
                                  (polySkinMaskReady || framePresence <= 0.001f);
        const float secPriorFlag = packSecPrior ? 1.0f : 0.0f;
        const auto& extraOval = hasSecSlot ? secLastValidOval_ : ghostOval_;
        const auto& extraAxis = hasSecSlot ? secLastValidAxis_ : ghostAxis_;
        const float extraPres = hasSecSlot ? secBodyPriorPresence_ : ghostPresence_;
        const float lowLightT = std::isfinite(in.lightLuma)
            ? (1.0f - detail::smoothstep(0.12f, 0.28f, in.lightLuma))
            : 0.0f;

        ub.push(1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height), aspect, framePresence);
        ub.push(stylePacked, effSmooth, config.brightness, effTexture);
        ub.push(config.radiance, effSharpness, config.saturation, activeFilterIntensity);
        ub.push(config.whiten, config.rosy, effOilControl, effRelight);
        // v3.9.7 (tone-k2): Emit activeFaceState while tracked (presence >= 0.5), and lock immediately onto
        // un-dilated lastValidOval_ upon entering hold/decay so timeline clear at 500ms never snaps oval source!
        if (in.facePresence >= 0.5f && activeFaceState) {
            const float* eyeMouthState = fallbackEllipseState ? fallbackEllipseState : activeFaceState;
            ub.push(cat02[0], cat02[1], cat02[2], activeFaceState[26]);
            ub.push(fusedPackedGrain, bodyPriorMute, secPriorFlag, activeFaceState[27]);
            ub.push(activeFaceState[22], activeFaceState[23], activeFaceState[24], activeFaceState[25]);
            ub.push(eyeMouthState[28], eyeMouthState[29], eyeMouthState[30], eyeMouthState[31]);
            if (packSecPrior) {
                ub.push(extraOval[0], extraOval[1], extraOval[2], extraOval[3]);
                ub.push(extraAxis[0], extraAxis[1], extraPres, 0.0f);
            } else {
                ub.push(eyeMouthState[32], eyeMouthState[33], eyeMouthState[34], eyeMouthState[35]);
                ub.push(eyeMouthState[36], eyeMouthState[37], eyeMouthState[38], eyeMouthState[39]);
            }
        } else if (bodyPriorPresence_ > 0.001f && lastValidOval_[2] > 0.001f) {
            ub.push(cat02[0], cat02[1], cat02[2], lastValidAxis_[0]);
            ub.push(fusedPackedGrain, bodyPriorMute, secPriorFlag, lastValidAxis_[1]);
            ub.push(lastValidOval_[0], lastValidOval_[1], lastValidOval_[2], lastValidOval_[3]);
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            if (packSecPrior) {
                ub.push(extraOval[0], extraOval[1], extraOval[2], extraOval[3]);
                ub.push(extraAxis[0], extraAxis[1], extraPres, 0.0f);
            } else {
                ub.push(0.0f, 0.0f, 0.0f, 0.0f);
                ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            }
        } else {
            ub.push(cat02[0], cat02[1], cat02[2], 0.0f);
            ub.push(fusedPackedGrain, 1.0f, secPriorFlag, 0.0f);
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            if (packSecPrior) {
                ub.push(extraOval[0], extraOval[1], extraOval[2], extraOval[3]);
                ub.push(extraAxis[0], extraAxis[1], extraPres, 0.0f);
            } else {
                ub.push(0.0f, 0.0f, 0.0f, 0.0f);
                ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        ub.push(lutStrength, coreThresh, microDetailKeep, ambientLift);
        ub.push(cat02[3], cat02[4], cat02[5], fusedCat02Full);
        ub.push(cat02[6], cat02[7], cat02[8], lowLightT);

        const bool compositeIsFinal = !hasPostCompositeLayers && !willRunStage8 &&
                                      (in.outputRt != kInvalidRenderTarget);
        const RenderTargetId compositeDstRt = compositeIsFinal ? in.outputRt : rtComposite_;
        device_.beginPass(compositeDstRt, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(compositeDstRt);
        if (compositeIsFinal) lastWroteOutputRt_ = true;
        lastActivePassesMask_ |= kPassComposite;
    }

    // 6b. v3.4: Real-Time Strand-Preserving Hair Dyeing & Crown Angel-Ring Shine
    if (willRunHairDye) {
        const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
            ? assets_.whiteAlphaFallbackTex
            : ownedFallbackTex_;
        const Vec3 dyeRgb = math::FaceMakeupUv::hairDyeRgb(config.hairDyeColor);
        float ux = 1.0f, uy = 0.0f;
        float crownCx = 0.5f, crownCy = 0.72f, crownRx = 0.28f, crownRy = 0.14f;
        float signedYaw = 0.0f;
        if (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && effFacePresence > 1e-4f) {
            const float ax = in.faceMeta15[math::FaceMeshFrame::kMetaAxisX];
            const float ay = in.faceMeta15[math::FaceMeshFrame::kMetaAxisY];
            const float n = std::hypot(ax, ay);
            if (n > 1e-4f) { ux = ax / n; uy = ay / n; }
            const float vx = -uy, vy = ux;
            const float fw = in.faceMeta15[math::FaceMeshFrame::kMetaFaceWidth];
            const float fh = in.faceMeta15[math::FaceMeshFrame::kMetaFaceHeight];
            crownCx = postWarpLandmarks[10 * 2] + vx * (0.08f * fh);
            crownCy = postWarpLandmarks[10 * 2 + 1] + (vy * (0.08f * fh)) * aspect;
            crownRx = std::max(fw * 0.72f, 0.08f);
            crownRy = std::max(fh * 0.28f, 0.06f);
            signedYaw = detail::clampf(in.faceMeta15[math::FaceMeshFrame::kMetaSignedYaw], -1.0f, 1.0f);
        } else if (activeFaceState) {
            ux = activeFaceState[26];
            uy = activeFaceState[27];
            const float vx = -uy, vy = ux;
            crownCx = activeFaceState[22] + vx * (activeFaceState[25] * 0.82f);
            crownCy = activeFaceState[23] + (vy * (activeFaceState[25] * 0.82f)) * aspect;
            crownRx = std::max(activeFaceState[24] * 1.35f, 0.12f);
            crownRy = std::max(activeFaceState[25] * 0.55f, 0.08f);
        }

        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, in.hairMaskTex);
        tb.bind(2, polySkinMaskReady ? device_.textureOf(rtSkinMask_) : fallbackTex);
        ub.push(in.hairAffine[0], in.hairAffine[1], in.hairAffine[2], effHairDye * in.hairShown);
        ub.push(in.hairAffine[4], in.hairAffine[5], in.hairAffine[6], effHairShine * in.hairShown);
        ub.push(dyeRgb.x, dyeRgb.y, dyeRgb.z, polySkinMaskReady ? 1.0f : 0.0f);
        ub.push(crownCx, crownCy, crownRx, crownRy);
        ub.push(ux, uy, aspect, signedYaw);
        // uU5: (pixelTexelX, pixelTexelY, maskMatteRadiusU, maskMatteRadiusV) for RGB-guided boundary matting
        ub.push(1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height),
                1.6f / 128.0f, 1.6f / 128.0f);
        const bool hairDyeIsFinal = !willRunStage7 && !willRunStage8 &&
                                    (in.outputRt != kInvalidRenderTarget);
        const RenderTargetId hairDyeDstRt = hairDyeIsFinal ? in.outputRt : rtBgOut_;
        device_.beginPass(hairDyeDstRt, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kHairDye, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(hairDyeDstRt);
        if (hairDyeIsFinal) lastWroteOutputRt_ = true;
        lastActivePassesMask_ |= kPassHairDye;
    }

    // 6c. v3.9.3: Composite makeup occlusion mask into rtSkinMask_.a strictly AFTER kCompositePro & kHairDye,
    //     right before Stage 7 UV Makeup, using independent occAffine for occlusionMaskTex.
    if (hasOccAlphaInput && (willRenderPrimaryMakeup || willRenderSecondaryMakeup)) {
        if (meshWarped && secWarped && !primaryMeshVboRestored) {
            device_.updateVertexBuffer(
                vbMeshWarp_, deformer_.vertexData(),
                sizeof(float) * static_cast<size_t>(deformer_.vertexDataFloatCount()));
            primaryMeshVboRestored = true;
        }
        renderHairAlpha(
            hasHairAlphaInput ? in.hairMaskTex : kInvalidTexture,
            hasHairAlphaInput ? in.hairShown : 0.0f,
            in.hairAffine.data(),
            in.occlusionMaskTex, in.occlusionShown,
            in.hasOccAffine ? in.occAffine.data() : nullptr,
            activeFaceState, aspect, !polySkinMaskReady, meshWarped, frontFaceCw);
        hasSemanticTex = true;
        lastActivePassesMask_ |= kPassHairAlpha;
    }

    // 7. 6-Layer Real-Time UV Makeup + Stickers + 3D AR + Particles
    const RenderTargetId stage7DstRt = (!willRunStage8 && in.outputRt != kInvalidRenderTarget)
        ? in.outputRt : rtMakeup_;
    bool makeupRendered = false;
    const TextureId compositeBaseTex = currentTex;
    const TextureId semanticForMakeup = hasSemanticTex ? device_.textureOf(rtSkinMask_) : kInvalidTexture;
    if (willRenderPrimaryMakeup) {
        renderMakeup(
            stage7DstRt, compositeBaseTex, semanticForMakeup, in.width, in.height,
            config, in.ambientRgb, in.tierCaps.enableAmbientMakeup,
            postWarpLandmarks, in.faceMeta15,
            in.hasIris, frontFaceCw, effFacePresence, in.steadyFace, in.steadyEye, in.eyeMakeupKeep,
            in.lowerMakeupKeep, /*copyBaseFirst=*/true);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassMakeup;
    }
    if (willRenderSecondaryMakeup) {
        renderMakeup(
            stage7DstRt, compositeBaseTex, semanticForMakeup, in.width, in.height,
            config, in.ambientRgb, in.tierCaps.enableAmbientMakeup,
            postSecLandmarks, sec.meta15,
            sec.hasIris, secFrontCw, effSecPresence, sec.steadyFace, sec.steadyEye, sec.eyeMakeupKeep,
            sec.lowerMakeupKeep, /*copyBaseFirst=*/!makeupRendered);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassMakeup;
    }

    // 7b. Face-Anchored 2D AR Stickers (primary + optional secondary face sharing stage7DstRt)
    if (willRunPrimarySticker) {
        renderStickers(stage7DstRt, currentTex, !makeupRendered, in.width, in.height, in,
                       postWarpLandmarks, in.faceMeta15, effFacePresence);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassSticker;
    }
    if (hasAnySticker && secActive && postSecLandmarks) {
        renderStickers(stage7DstRt, currentTex, !makeupRendered, in.width, in.height, in,
                       postSecLandmarks, sec.meta15, effSecPresence);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassSticker;
    }

    // 7c. P4: Lightweight 3D Head-Occluded AR Accessories + Spring Pendulum Physics
    if (willRunAr3d) {
        renderAr3dAccessories(
            stage7DstRt, currentTex, !makeupRendered, in.width, in.height, in,
            postWarpLandmarks, cachedAr3dPose, frontFaceCw, effFacePresence);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassAr3d;
    }

    // 7d. v3.6: Batched 2D Analytical SDF Interactive Gesture Particle Bursts
    if (willRunParticles) {
        renderParticles(stage7DstRt, currentTex, !makeupRendered, in.width, in.height,
                        in.particleVertices, in.particleQuadCount);
        currentTex = device_.textureOf(stage7DstRt);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassParticles;
    }
    if (makeupRendered && stage7DstRt == in.outputRt && in.outputRt != kInvalidRenderTarget) {
        lastWroteOutputRt_ = true;
    }

    // 8. v3.8/v3.9: Cinematic Atmosphere Post-Processing (Soft-Knee Multi-Scale Bloom + Non-Darkening Filmic Grain + CIE CAT02 WB)
    if (willRunStage8) {
        const float effBloom = preEffBloom;
        const float effGrain = preEffGrain;
        const float effTemp = preEffTemp;
        const float effTint = preEffTint;
        const bool hasCat02 = preHasCat02;

        TextureId tightBloomTex = currentTex;
        TextureId wideBloomTex = currentTex;
        if (effBloom > 1e-4f) {
            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            // 8a. Soft-knee bright extract at 1/4 res (rtBg_[0]); threshold 0.80/knee 0.10 avoids lifting face highlights
            tb.bind(0, currentTex);
            ub.push(0.80f, 0.10f, 1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height));
            device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kPostBloomExtract, st, tb, ub);
            device_.endPass();

            // 8b. Tight glow separable H+V blur at 1/4 res (step=1.0 prevents comb aliasing)
            tb.clear(); ub.clear();
            tb.bind(0, device_.textureOf(rtBg_[0]));
            ub.push(1.0f / static_cast<float>(bgW_), 0.0f, 0.0f, 0.0f);
            device_.beginPass(rtBg_[1], bgW_, bgH_, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
            device_.endPass();

            tb.clear(); ub.clear();
            tb.bind(0, device_.textureOf(rtBg_[1]));
            ub.push(0.0f, 1.0f / static_cast<float>(bgH_), 0.0f, 0.0f);
            device_.beginPass(rtBg_[0], bgW_, bgH_, LoadOp::kDontCare);
            device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
            device_.endPass();

            tightBloomTex = device_.textureOf(rtBg_[0]);
            const bool useWidePyramid = (effBloom > 0.22f) &&
                (in.tierCaps.resolvedTier == math::PerformanceTier::kUltra) &&
                (in.loadFadeBody >= 0.99f);
            if (useWidePyramid) {
                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtBg_[0]));
                ub.push(1.0f / static_cast<float>(bgW_), 1.0f / static_cast<float>(bgH_), 0.0f, 0.0f);
                device_.beginPass(rtMacro_[0], macroW_, macroH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kDownsample, st, tb, ub);
                device_.endPass();

                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtMacro_[0]));
                ub.push(1.0f / static_cast<float>(macroW_), 0.0f, 0.0f, 0.0f);
                device_.beginPass(rtMacro_[1], macroW_, macroH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();

                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtMacro_[1]));
                ub.push(0.0f, 1.0f / static_cast<float>(macroH_), 0.0f, 0.0f);
                device_.beginPass(rtMacro_[0], macroW_, macroH_, LoadOp::kDontCare);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();

                wideBloomTex = device_.textureOf(rtMacro_[0]);
            } else {
                wideBloomTex = tightBloomTex;
            }
        }

        float cat02[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        if (hasCat02) {
            math::BeautyQualityTuning::cat02WhiteBalanceMatrix(effTemp, effTint, cat02);
        }
        const RenderTargetId fallbackPostRt = (currentTex == device_.textureOf(rtBgOut_)) ? rtComposite_ : rtBgOut_;
        const RenderTargetId postDstRt = (in.outputRt != kInvalidRenderTarget) ? in.outputRt : fallbackPostRt;
        const float grainPhase = std::fmod(static_cast<float>(std::max<int64_t>(0, in.nowMs) % 60000LL) * 0.018f, 1.0f);
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, tightBloomTex);
        tb.bind(2, wideBloomTex);
        ub.push(effBloom, effGrain, grainPhase, hasCat02 ? 1.0f : 0.0f);
        ub.push(cat02[0], cat02[1], cat02[2], static_cast<float>(in.width) * 0.5f);
        ub.push(cat02[3], cat02[4], cat02[5], static_cast<float>(in.height) * 0.5f);
        ub.push(cat02[6], cat02[7], cat02[8], 0.0f);
        device_.beginPass(postDstRt, in.width, in.height, LoadOp::kDontCare);
        device_.drawFullScreenQuad(ShaderId::kPostAtmosphere, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(postDstRt);
        if (postDstRt == in.outputRt && in.outputRt != kInvalidRenderTarget) {
            lastWroteOutputRt_ = true;
        }
        lastActivePassesMask_ |= kPassAtmosphere;
    }

    return currentTex;
}

}  // namespace livebeauty::rhi
