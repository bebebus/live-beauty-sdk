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
        if (kEyeCore.count(i) || kBrowCore.count(i) || lipInnerAndMid.count(i)) {
            skin = 0.0f; excl = 1.0f;
        } else if (kBrowBorder.count(i) || kLipOuter.count(i)) {
            skin = 0.22f; excl = 0.78f;
        } else if (ovalSet.count(i)) {
            skin = (v < 0.32f) ? 0.80f : 0.68f;
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
    denoiseHistoryValid_ = false;
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
    for (auto& rt : rtBg_) killRt(rt);
    killRt(rtBgHair_);
    killRt(rtBgOut_);

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
    killBuf(vbRetouch_);
    killBuf(vbMakeup_);
    killBuf(vbSticker_);
    killBuf(ibFullMesh_);
    killBuf(ibSkinRetouch_);
    killBuf(ibEyeOpening_);
    killBuf(ibInnerMouth_);
    killBuf(ibSkinFace_);
    killBuf(ibLipBand_);
    killBuf(ibRightEye_);
    killBuf(ibLeftEye_);
    killBuf(ibSticker_);

    staticBuffersReady_ = false;
    width_ = height_ = 0;
    denoiseHistoryValid_ = false;
}

void RenderPipeline::ensureStaticBuffers() {
    if (!staticSkinWeightsReady_) {
        buildSkinMaskWeights(staticSkinWeights_.data());
        staticSkinWeightsReady_ = true;
    }
    if (staticBuffersReady_) return;

    vbMeshWarp_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceMeshTopology::kTotalVertexCount * 4);
    vbSkinMask_ = device_.createDynamicVertexBuffer(sizeof(float) * kMaskVertexCount * 5);
    vbRetouch_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceRetouchSubMesh::kVertexDataSize);
    vbMakeup_ = device_.createDynamicVertexBuffer(sizeof(float) * math::FaceMakeupUv::kVertexDataSize);
    vbSticker_ = device_.createDynamicVertexBuffer(sizeof(float) * math::StickerTransform::kQuadFloats);

    ibFullMesh_ = device_.createStaticIndexBuffer(
        math::FaceMeshTopology::kFullIndices.data(),
        static_cast<int>(math::FaceMeshTopology::kFullIndices.size()));
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

    const uint8_t fallbackPx[4] = {0, 0, 0, 255};
    ownedFallbackTex_ = device_.createTexture2D(1, 1, TextureFormat::kRgba8, fallbackPx);

    staticBuffersReady_ = true;
}

void RenderPipeline::ensureResources(int width, int height) {
    ensureStaticBuffers();
    if (width == width_ && height == height_) return;

    auto recreate = [&](RenderTargetId& rt, int w, int h) {
        if (rt != kInvalidRenderTarget) device_.destroyRenderTarget(rt);
        rt = device_.createRenderTarget(w, h, TextureFormat::kRgba8);
    };

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

    recreate(rtBody_, width_, height_);
    recreate(rtWarp_, width_, height_);
    recreate(rtComposite_, width_, height_);
    recreate(rtMakeup_, width_, height_);
    recreate(rtDenoise_[0], width_, height_);
    recreate(rtDenoise_[1], width_, height_);
    for (int i = 0; i < 3; ++i) recreate(rtGuided_[i], guidedW_, guidedH_);
    for (int i = 0; i < 2; ++i) recreate(rtMacro_[i], macroW_, macroH_);
    recreate(rtSkinMask_, skinW_, skinH_);
    for (int i = 0; i < 2; ++i) recreate(rtBg_[i], bgW_, bgH_);
    recreate(rtBgHair_, bgW_, bgH_);
    recreate(rtBgOut_, width_, height_);
    denoiseHistoryValid_ = false;
}

void RenderPipeline::renderGuidedFilter(
    TextureId srcTex, int width, int height, float eps, float evenness, float spread) {
    RenderState st{};
    TextureBindings tb{};
    UniformBlock ub{};

    // P1: 1/3 downsample with 2/3 texel tap scale (exact 3x3 box filter).
    tb.bind(0, srcTex);
    ub.push(math::BeautyQualityTuning::kDownTapScale / static_cast<float>(width),
            math::BeautyQualityTuning::kDownTapScale / static_cast<float>(height), 0.0f, 0.0f);
    device_.beginPass(rtGuided_[0], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kDownsample, st, tb, ub);
    device_.endPass();

    // P2: Horizontal blur -> rtGuided_[1]
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[0]));
    ub.push(spread / static_cast<float>(guidedW_), 0.0f, 0.0f, 0.0f);
    device_.beginPass(rtGuided_[1], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
    device_.endPass();

    // P3: Vertical blur + energy diff against rtGuided_[0] -> rtGuided_[2]
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[1]));
    tb.bind(1, device_.textureOf(rtGuided_[0]));
    ub.push(0.0f, spread / static_cast<float>(guidedH_), 0.0f, 0.0f);
    device_.beginPass(rtGuided_[2], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kGuidedDiff, st, tb, ub);
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
        ub.push(1.25f / static_cast<float>(macroW_), 0.0f, 0.0f, 0.0f);
        device_.beginPass(rtMacro_[1], macroW_, macroH_, false);
        device_.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
        device_.endPass();

        tb.clear(); ub.clear();
        tb.bind(0, device_.textureOf(rtMacro_[1]));
        ub.push(0.0f, 1.25f / static_cast<float>(macroH_), 0.0f, 0.0f);
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
    // Always bind isolated rtMacro_[0] texture even when evenness==0 to prevent GLES feedback loop.
    tb.clear(); ub.clear();
    tb.bind(0, device_.textureOf(rtGuided_[1]));
    tb.bind(1, device_.textureOf(rtMacro_[0]));
    ub.push(0.0f, spread / static_cast<float>(guidedH_), eps, evenness);
    device_.beginPass(rtGuided_[0], guidedW_, guidedH_, false);
    device_.drawFullScreenQuad(ShaderId::kGuidedCoeff, st, tb, ub);
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

void RenderPipeline::renderSkinMask(const float* landmarks478x2, bool frontFaceCw, int width, int height) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    // Pack 468 face vertices + 36 outer feather vertices.
    float cx = 0.0f, cy = 0.0f;
    for (int k = 0; k < math::FaceMeshTopology::kOvalVertexCount; ++k) {
        const int oi = math::FaceMeshTopology::kFaceOval[k];
        cx += landmarks478x2[oi * 2];
        cy += landmarks478x2[oi * 2 + 1];
    }
    cx /= static_cast<float>(math::FaceMeshTopology::kOvalVertexCount);
    cy /= static_cast<float>(math::FaceMeshTopology::kOvalVertexCount);

    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        const int k = i * 5;
        skinMaskVerts_[k + 0] = landmarks478x2[i * 2] * 2.0f - 1.0f;
        skinMaskVerts_[k + 1] = landmarks478x2[i * 2 + 1] * 2.0f - 1.0f;
        skinMaskVerts_[k + 2] = staticSkinWeights_[i * 3 + 0];
        skinMaskVerts_[k + 3] = staticSkinWeights_[i * 3 + 1];
        skinMaskVerts_[k + 4] = staticSkinWeights_[i * 3 + 2];
    }
    for (int j = 0; j < math::FaceMeshTopology::kOvalVertexCount; ++j) {
        const int oi = math::FaceMeshTopology::kFaceOval[j];
        const float ox = landmarks478x2[oi * 2];
        const float oy = landmarks478x2[oi * 2 + 1];
        const float dx = ox - cx;
        const float dy = (oy - cy) / aspect;
        const float len = std::max(std::hypot(dx, dy), 1e-4f);
        const float pad = 0.03f;
        const float fx = ox + (dx / len) * pad;
        const float fy = oy + (dy / len) * pad * aspect;
        const int vi = math::FaceMeshTopology::kMidBufferOffset + j;
        const int k = vi * 5;
        skinMaskVerts_[k + 0] = fx * 2.0f - 1.0f;
        skinMaskVerts_[k + 1] = fy * 2.0f - 1.0f;
        skinMaskVerts_[k + 2] = 0.0f;
        skinMaskVerts_[k + 3] = 0.0f;
        skinMaskVerts_[k + 4] = 0.0f;
    }

    device_.updateVertexBuffer(vbSkinMask_, skinMaskVerts_.data(), sizeof(float) * skinMaskVerts_.size());

    // Painter's order: outer feather ring first (no cull), then 898 foreground face triangles (back-face culled).
    TextureBindings tb{};
    UniformBlock ub{};
    device_.beginPass(rtSkinMask_, skinW_, skinH_, true, {0.0f, 0.0f, 0.0f, 1.0f});

    RenderState featherState{};
    featherState.cullBack = false;
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

void RenderPipeline::renderMakeup(
    RenderTargetId targetRt, TextureId baseTex, int width, int height,
    const BeautyConfig& cfg, const float* landmarks478x2, const float* meta15,
    bool hasIris, bool frontFaceCw, float presence, float steadyEye, float eyeMakeupKeep) {
    math::FaceMakeupUv::packVertices(landmarks478x2, math::FaceMeshFrame::kCoordCount, makeupVerts_.data());
    device_.updateVertexBuffer(vbMakeup_, makeupVerts_.data(), sizeof(float) * makeupVerts_.size());

    // Copy base composite to targetRt first so layered alpha-blended makeup reads clean baseTex without feedback loop.
    {
        RenderState copyState{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        device_.beginPass(targetRt, width, height, false);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, copyState, tb, ub);
    }

    RenderState layerState{};
    layerState.blendEnable = true;
    layerState.cullBack = true;
    layerState.frontFaceCw = frontFaceCw;

    const float faceGain = presence * steadyEye;
    const float eyeGain = presence * eyeMakeupKeep;
    const int skinIdxCount = static_cast<int>(math::FaceMakeupUv::skinFaceIndices().size());

    auto drawSkinLayer = [&](TextureId atlas, Vec3 rgb, float alpha, Vec4 channel, Vec4 mode) {
        if (atlas == kInvalidTexture || alpha <= 1e-4f) return;
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, baseTex);
        tb.bind(1, atlas);
        ub.push(rgb.x, rgb.y, rgb.z, alpha);
        ub.push(channel);
        ub.push(mode);
        device_.drawIndexed(
            ShaderId::kMakeupLayer, layerState, vbMakeup_, ibSkinFace_,
            0, skinIdxCount, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
    };

    // 1. Blush (SoftLight mode=0)
    if (cfg.blush > 0.0f) {
        drawSkinLayer(assets_.blushAtlas, math::FaceMakeupUv::blushRgb(cfg.blushColor),
                      cfg.blush * 0.72f * faceGain, math::FaceMakeupUv::blushChannel(cfg.blushStyle),
                      {0.0f, 0.0f, 0.0f, 0.0f});
    }
    // 2. Contour & Highlight (contour_highlight R=contour, G=highlight)
    if (cfg.contour > 0.0f) {
        drawSkinLayer(assets_.contourHighlightAtlas, {0.42f, 0.30f, 0.24f},
                      cfg.contour * 0.55f * faceGain, {1.0f, 0.0f, 0.0f, 0.0f},
                      {1.0f, 0.0f, 0.0f, 0.0f});
    }
    if (cfg.highlight > 0.0f) {
        drawSkinLayer(assets_.contourHighlightAtlas, {1.0f, 0.97f, 0.92f},
                      cfg.highlight * 0.60f * faceGain, {0.0f, 1.0f, 0.0f, 0.0f},
                      {2.0f, 0.0f, 0.0f, 0.0f});
    }
    // 3. Eyebrow (split-eye HD UV mode.y=1)
    if (cfg.eyebrow > 0.0f) {
        drawSkinLayer(assets_.eyebrowAtlas, math::FaceMakeupUv::eyebrowRgb(cfg.eyebrowColor),
                      cfg.eyebrow * 0.68f * faceGain, math::FaceMakeupUv::eyebrowChannel(cfg.eyebrowStyle),
                      {1.0f, 1.0f, 0.0f, 0.0f});
    }
    // 4. Eyeshadow (dual layer with eyelid skin gate mode.z=1 and eyeMakeupKeep velocity fade)
    if (cfg.eyeshadow > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, math::FaceMakeupUv::eyeshadowMainRgb(cfg.eyeshadowColor),
                      cfg.eyeshadow * 0.64f * eyeGain, {1.0f, 0.0f, 0.0f, 0.0f},
                      {1.0f, 1.0f, 1.0f, 0.0f});
        drawSkinLayer(assets_.eyeMakeupAtlas, math::FaceMakeupUv::eyeshadowDeepRgb(cfg.eyeshadowColor),
                      cfg.eyeshadow * 0.56f * eyeGain, {0.0f, 1.0f, 0.0f, 0.0f},
                      {1.0f, 1.0f, 1.0f, 0.0f});
    }
    // 5. Eyeliner & Aegyo-sal
    if (cfg.eyeliner > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, {0.10f, 0.08f, 0.08f},
                      cfg.eyeliner * 0.78f * eyeGain, {0.0f, 0.0f, 1.0f, 0.0f},
                      {1.0f, 1.0f, 0.0f, 0.0f});
    }
    if (cfg.aegyoSal > 0.0f) {
        drawSkinLayer(assets_.eyeMakeupAtlas, {1.0f, 0.94f, 0.88f},
                      cfg.aegyoSal * 0.62f * eyeGain, {0.0f, 0.0f, 0.0f, 1.0f},
                      {2.0f, 1.0f, 0.0f, 0.0f});
    }
    // 6. Physically-based Beer-Lambert Lipstick (120 lip-band triangles only)
    if (cfg.lipstick > 0.0f && assets_.contourHighlightAtlas != kInvalidTexture) {
        const Vec3 lipRgb = math::FaceMakeupUv::lipstickRgb(cfg.lipstickColor);
        const float lipAlpha = cfg.lipstick * 0.85f * faceGain;
        if (lipAlpha > 1e-4f) {
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, baseTex);
            tb.bind(1, assets_.contourHighlightAtlas);
            ub.push(lipRgb.x, lipRgb.y, lipRgb.z, lipAlpha);
            ub.push(cfg.lipstickFinish == LipstickFinish::kGlossy ? 1.0f : 0.0f,
                    1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height), 0.0f);
            device_.drawIndexed(
                ShaderId::kMakeupLip, layerState, vbMakeup_, ibLipBand_,
                0, math::FaceMeshTopology::kLipBandIndexCount, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
        }
    }
    // 7. Contact lens & twin catchlights (14+14 eye opening triangles, blink gated)
    if ((cfg.contactLens > 0.0f || cfg.catchlight > 0.0f) && hasIris && assets_.irisAtlas != kInvalidTexture) {
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const Vec2 tile = math::FaceMakeupUv::irisTile(cfg.contactLensStyle);
        const float lensS = cfg.contactLens * faceGain;
        const float clS = cfg.catchlight * faceGain;

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
            ub.push(tile.x, tile.y, effOpen, 0.0f);
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
    const float* ovalState41, float aspect, bool clearRgbToZero) {
    RenderState st{};
    st.colorMaskAlphaOnly = !clearRgbToZero;
    TextureBindings tb{};
    UniformBlock ub{};
    tb.bind(0, hairMaskTex);
    ub.push(hairAffine8[0], hairAffine8[1], hairAffine8[2], hairShown);
    ub.push(hairAffine8[4], hairAffine8[5], hairAffine8[6], 0.0f);
    if (ovalState41) {
        ub.push(ovalState41[22], ovalState41[23], ovalState41[24], ovalState41[25]);
        ub.push(ovalState41[26], ovalState41[27], aspect, 1.0f);
    } else {
        ub.push(0.5f, 0.5f, 0.25f, 0.30f);
        ub.push(1.0f, 0.0f, aspect, 0.0f);
    }
    device_.beginPass(rtSkinMask_, skinW_, skinH_, clearRgbToZero, {0.0f, 0.0f, 0.0f, 1.0f});
    device_.drawFullScreenQuad(ShaderId::kHairAlpha, st, tb, ub);
    device_.endPass();
}

void RenderPipeline::renderStickers(
    RenderTargetId targetRt, TextureId baseTex, bool copyBaseFirst,
    int width, int height, const FrameInputs& in,
    const float* landmarks478x2, float presence) {
    if (!landmarks478x2 || !in.faceMeta15 || in.stickerCount <= 0) return;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    device_.beginPass(targetRt, width, height, false);
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
        if (!math::StickerTransform::buildQuadVertices(layer, landmarks478x2, in.faceMeta15, aspect, quadVerts)) {
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

TextureId RenderPipeline::render(const BeautyConfig& config, const FrameInputs& in) {
    lastActivePassesMask_ = 0;
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
    if (!config.isProcessing() && !(config.enabled && !config.compareOriginal && hasAnySticker)) {
        return in.inputTexture;
    }

    ensureResources(in.width, in.height);
    const float aspect = static_cast<float>(in.width) / static_cast<float>(in.height);
    TextureId currentTex = in.inputTexture;

    // 1. Optional Background Bokeh / Virtual Background Replacement (with optional hair-merge flyaway preservation)
    const float bgBlur = config.backgroundBlur * in.loadFadeBody;
    const bool wantVirtualImage = (config.backgroundMode == BackgroundMode::kVirtualImage ||
                                   config.backgroundMode == BackgroundMode::kVirtualBlur) &&
                                  assets_.virtualBgTex != kInvalidTexture &&
                                  config.virtualBgStrength * in.loadFadeBody > 1e-4f;
    const bool wantBgBlur = !wantVirtualImage && bgBlur > 1e-4f;
    if ((wantVirtualImage || wantBgBlur) && in.personMaskTex != kInvalidTexture) {
        lastActivePassesMask_ |= kPassBg;
        const float ovalCx = in.hasFaceState ? in.faceState41[22] : 0.5f;
        const float ovalCy = in.hasFaceState ? in.faceState41[23] : 0.5f;
        const float ovalRx = in.hasFaceState ? in.faceState41[24] : 0.25f;
        const float ovalRy = in.hasFaceState ? in.faceState41[25] : 0.30f;
        const float ax = in.hasFaceState ? in.faceState41[26] : 1.0f;
        const float ay = in.hasFaceState ? in.faceState41[27] : 0.0f;

        // 1a. Optional Hair-Person Mask Merge (protects fine flyaway hair strands during bokeh/virtual replacement)
        TextureId effectiveMaskTex = in.personMaskTex;
        if (in.hairMaskTex != kInvalidTexture && in.hairBgShown > 1e-4f) {
            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, in.personMaskTex);
            tb.bind(1, in.hairMaskTex);
            ub.push(in.hairAffine[0], in.hairAffine[1], in.hairAffine[2], in.hairBgShown);
            ub.push(in.hairAffine[4], in.hairAffine[5], in.hairAffine[6], 0.0f);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, in.hasFaceState ? 1.0f : 0.0f);
            device_.beginPass(rtBgHair_, bgW_, bgH_, false);
            device_.drawFullScreenQuad(ShaderId::kBgHairMerge, st, tb, ub);
            device_.endPass();
            effectiveMaskTex = device_.textureOf(rtBgHair_);
        }

        if (wantVirtualImage) {
            TextureId bgSource = assets_.virtualBgTex;
            if (config.backgroundMode == BackgroundMode::kVirtualBlur) {
                RenderState st{};
                TextureBindings tb{};
                UniformBlock ub{};
                tb.bind(0, assets_.virtualBgTex);
                device_.beginPass(rtBg_[0], bgW_, bgH_, false);
                device_.drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);
                device_.endPass();
                for (int pass = 0; pass < 2; ++pass) {
                    const float stepScale = (pass == 0) ? 1.0f : 2.0f;
                    tb.clear(); ub.clear();
                    tb.bind(0, device_.textureOf(rtBg_[0]));
                    ub.push(stepScale / static_cast<float>(bgW_), 0.0f, 0.0f, 0.0f);
                    device_.beginPass(rtBg_[1], bgW_, bgH_, false);
                    device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                    device_.endPass();
                    tb.clear(); ub.clear();
                    tb.bind(0, device_.textureOf(rtBg_[1]));
                    ub.push(0.0f, stepScale / static_cast<float>(bgH_), 0.0f, 0.0f);
                    device_.beginPass(rtBg_[0], bgW_, bgH_, false);
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
            ub.push(detail::clampf(config.virtualBgStrength * in.loadFadeBody, 0.0f, 1.0f),
                    0.35f, 0.70f, in.hasFaceState ? 0.85f : 0.0f);
            ub.push(uvScaleX, uvScaleY, uvOffX, uvOffY);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, 0.0f);
            device_.beginPass(rtBgOut_, in.width, in.height, false);
            device_.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
            device_.endPass();
            currentTex = device_.textureOf(rtBgOut_);
        } else {
            RenderState st{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, currentTex);
            tb.bind(1, effectiveMaskTex);
            ub.push(bgBlur, 0.35f, 0.70f, 0.0f);
            device_.beginPass(rtBg_[0], bgW_, bgH_, false);
            device_.drawFullScreenQuad(ShaderId::kBgPremask, st, tb, ub);
            device_.endPass();

            for (int pass = 0; pass < 2; ++pass) {
                const float stepScale = (pass == 0) ? 1.0f : 2.0f;
                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtBg_[0]));
                ub.push(stepScale / static_cast<float>(bgW_), 0.0f, 0.0f, 0.0f);
                device_.beginPass(rtBg_[1], bgW_, bgH_, false);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();

                tb.clear(); ub.clear();
                tb.bind(0, device_.textureOf(rtBg_[1]));
                ub.push(0.0f, stepScale / static_cast<float>(bgH_), 0.0f, 0.0f);
                device_.beginPass(rtBg_[0], bgW_, bgH_, false);
                device_.drawFullScreenQuad(ShaderId::kBgBlur, st, tb, ub);
                device_.endPass();
            }

            tb.clear(); ub.clear();
            tb.bind(0, currentTex);
            tb.bind(1, device_.textureOf(rtBg_[0]));
            tb.bind(2, effectiveMaskTex);
            ub.push(bgBlur, 0.35f, 0.70f, in.hasFaceState ? 0.85f : 0.0f);
            ub.push(ovalCx, ovalCy, ovalRx, ovalRy);
            ub.push(ax, ay, aspect, 0.0f);
            device_.beginPass(rtBgOut_, in.width, in.height, false);
            device_.drawFullScreenQuad(ShaderId::kBgMix, st, tb, ub);
            device_.endPass();
            currentTex = device_.textureOf(rtBgOut_);
        }
    }

    // 2. Optional 5-axis Body Warp + NeckLift alignment for face landmarks/state
    const float bodyMod = in.bodyPresence * in.loadFadeBody;
    float appliedNeck = 0.0f;
    if (in.hasBody && in.bodyState7 && bodyMod > 1e-4f && config.hasBodyReshape()) {
        const float legStrength = config.legLengthen * 0.28f * bodyMod * in.bodyState7[6];
        const float waist = config.waistSlim * 0.16f * bodyMod;
        const float bust = config.bustEnhance * 0.14f * bodyMod;
        const float shoulder = config.shoulderSlim * 0.12f * bodyMod;
        const float neck = config.neckLengthen * 0.22f * bodyMod;
        appliedNeck = neck;

        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        ub.push(in.bodyState7[0], in.bodyState7[1],
                in.bodyState7[2] * (1.0f + legStrength), in.bodyState7[3] * (1.0f + legStrength));
        ub.push(in.bodyState7[4], in.bodyState7[5], waist, bust);
        ub.push(shoulder, neck, aspect, 0.0f);
        device_.beginPass(rtBody_, in.width, in.height, false);
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
                in.bodyState7, appliedNeck, aspect, liftedLandmarks_.data());
            activeLandmarks = liftedLandmarks_.data();
        }
        if (in.hasFaceState && in.faceState41) {
            math::NeckLift::applyFaceState(in.faceState41, in.bodyState7, appliedNeck, aspect, liftedState_.data());
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
    const bool wantRetouch = in.hasFaceMesh && activeLandmarks && in.faceMeta15 &&
                             effFacePresence > 1e-4f && config.hasRetouch();

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
    } else if (in.hasFaceMesh && activeLandmarks) {
        frontFaceCw = math::FaceMeshFrame::isFrontFaceCw(activeLandmarks, math::FaceMeshFrame::kCoordCount);
    }

    if (meshWarped || wantRetouch) {
        const TextureId sourceBeforeWarp = currentTex;
        device_.beginPass(rtWarp_, in.width, in.height, false);
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
        } else {
            RenderState copySt{};
            TextureBindings tb{};
            UniformBlock ub{};
            tb.bind(0, sourceBeforeWarp);
            device_.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }

        // In-place 468-pt anatomical sub-mesh retouch directly on rtWarp_ (zero extra 1080p FBO).
        if (wantRetouch) {
            const float dc = detail::clampf(config.darkCircles * effFacePresence * in.steadyFace, 0.0f, 1.0f);
            const float nf = detail::clampf(config.nasolabial * effFacePresence * in.steadyFace, 0.0f, 1.0f);
            const float be = detail::clampf(config.brightEyes * effFacePresence * in.steadyEye, 0.0f, 1.0f);
            const float tw = detail::clampf(config.teethWhiten * effFacePresence * in.steadyEye, 0.0f, 1.0f);
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
                    ub.push(be, 0.0f, 0.0f, 0.0f);
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

    // 4. Polygon Skin Mask (468+36 vertices) + Optional Hair Semantic Alpha (A channel = nonHair)
    bool polySkinMaskReady = false;
    bool hasSemanticTex = false;
    if (in.hasFaceMesh && postWarpLandmarks && effFacePresence > 1e-4f) {
        renderSkinMask(postWarpLandmarks, frontFaceCw, in.width, in.height);
        polySkinMaskReady = true;
        hasSemanticTex = true;
        lastActivePassesMask_ |= kPassSkinMask;
    }
    if (in.hairMaskTex != kInvalidTexture && in.hairShown > 1e-4f) {
        renderHairAlpha(
            in.hairMaskTex, in.hairShown, in.hairAffine.data(),
            activeFaceState, aspect, !polySkinMaskReady);
        hasSemanticTex = true;
        lastActivePassesMask_ |= kPassHairAlpha;
    }

    // 5. Dual-Band Guided Filter Pre-Pass (1/3 + 1/9)
    const float effSmooth = detail::clampf(config.smoothing * in.loadFadeSmooth, 0.0f, 1.0f);
    if (effSmooth > 1e-4f) {
        const float eps = math::BeautyQualityTuning::guidedEps(effSmooth);
        const float evenness = math::BeautyQualityTuning::macroEvenness(effSmooth);
        renderGuidedFilter(currentTex, in.width, in.height, eps, evenness, in.guidedSpread);
        lastActivePassesMask_ |= kPassGuided;
    }

    // 6. Main Composite Pro + 64^3 3D LUT (15 active uniforms)
    {
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        const TextureId fallbackTex = (assets_.whiteAlphaFallbackTex != kInvalidTexture)
            ? assets_.whiteAlphaFallbackTex
            : ownedFallbackTex_;
        const TextureId lutTex = (assets_.lutTex != kInvalidTexture) ? assets_.lutTex : fallbackTex;
        const float lutStrength = (assets_.lutTex != kInvalidTexture && !config.lutUrl.empty())
            ? detail::clampf(config.lutIntensity, 0.0f, 1.0f) : 0.0f;

        tb.bind(0, currentTex);
        tb.bind(1, effSmooth > 1e-4f ? device_.textureOf(rtGuided_[0]) : currentTex);
        tb.bind(2, hasSemanticTex ? device_.textureOf(rtSkinMask_) : fallbackTex);
        tb.bind(3, lutTex);

        const float framePresence = (polySkinMaskReady && effFacePresence > 0.0f)
            ? (effFacePresence + 4.0f) : effFacePresence;
        const float stylePacked = static_cast<float>(static_cast<int>(config.filter)) + (config.skinOnly ? 16.0f : 0.0f);
        const float effTexture = math::BitrateEffectsTuning::texture(config.skinTexture, in.bitratePressure);
        const float effSharpness = math::BitrateEffectsTuning::sharpness(config.sharpness, in.bitratePressure);
        const float coreThresh = math::BeautyQualityTuning::coreThreshold(in.lightLuma, in.bitratePressure);

        // Uniform layout matching FRAGMENT_PRO (11 vec4 + 4 samplers = 15 active uniforms):
        // [0] uFrame, [1] uParams, [2] uDetails, [3] uTone, [4..5] uEyes[2],
        // [6] uOval, [7] uMouth, [8..9] uEyeMask[2], [10] uGrade
        ub.push(1.0f / static_cast<float>(in.width), 1.0f / static_cast<float>(in.height), aspect, framePresence);
        ub.push(stylePacked, effSmooth, config.brightness, effTexture);
        ub.push(config.radiance, effSharpness, config.saturation, config.filterIntensity);
        ub.push(config.whiten, config.rosy, 0.0f, 0.0f);
        if (activeFaceState) {
            ub.push(activeFaceState[16], activeFaceState[17], activeFaceState[18], activeFaceState[26]);
            ub.push(activeFaceState[19], activeFaceState[20], activeFaceState[21], activeFaceState[27]);
            ub.push(activeFaceState[22], activeFaceState[23], activeFaceState[24], activeFaceState[25]);
            ub.push(activeFaceState[28], activeFaceState[29], activeFaceState[30], activeFaceState[31]);
            ub.push(activeFaceState[32], activeFaceState[33], activeFaceState[34], activeFaceState[35]);
            ub.push(activeFaceState[36], activeFaceState[37], activeFaceState[38], activeFaceState[39]);
        } else {
            for (int i = 0; i < 6; ++i) ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        }
        ub.push(lutStrength, coreThresh, 0.0f, 0.0f);

        device_.beginPass(rtComposite_, in.width, in.height, false);
        device_.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        device_.endPass();
        currentTex = device_.textureOf(rtComposite_);
        lastActivePassesMask_ |= kPassComposite;
    }

    // 7. 6-Layer Real-Time UV Makeup
    bool makeupRendered = false;
    if (in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && effFacePresence > 1e-4f && config.hasMakeup()) {
        renderMakeup(
            rtMakeup_, currentTex, in.width, in.height,
            config, postWarpLandmarks, in.faceMeta15,
            in.hasIris, frontFaceCw, effFacePresence, in.steadyEye, in.eyeMakeupKeep);
        currentTex = device_.textureOf(rtMakeup_);
        makeupRendered = true;
        lastActivePassesMask_ |= kPassMakeup;
    }

    // 7b. Face-Anchored 2D AR Stickers (shares rtMakeup_ with zero extra 1080p FBO)
    if (hasAnySticker && in.hasFaceMesh && postWarpLandmarks && in.faceMeta15 && effFacePresence > 1e-4f) {
        renderStickers(rtMakeup_, currentTex, !makeupRendered, in.width, in.height, in, postWarpLandmarks, effFacePresence);
        currentTex = device_.textureOf(rtMakeup_);
        lastActivePassesMask_ |= kPassSticker;
    }

    // 8. Adaptive Temporal Denoise
    const float denoiseWeight = math::BitrateEffectsTuning::denoise(in.bitratePressure) * in.denoiseKeep * in.loadFadeSmooth;
    if (denoiseWeight > 0.01f && denoiseHistoryValid_) {
        const int dstIdx = 1 - denoiseParity_;
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        tb.bind(1, device_.textureOf(rtDenoise_[denoiseParity_]));
        ub.push(denoiseWeight, 0.03f, 0.10f, 0.0f);
        device_.beginPass(rtDenoise_[dstIdx], in.width, in.height, false);
        device_.drawFullScreenQuad(ShaderId::kTemporalDenoise, st, tb, ub);
        device_.endPass();
        denoiseParity_ = dstIdx;
        currentTex = device_.textureOf(rtDenoise_[dstIdx]);
        lastActivePassesMask_ |= kPassDenoise;
    } else if (denoiseWeight > 0.01f) {
        RenderState st{};
        TextureBindings tb{};
        UniformBlock ub{};
        tb.bind(0, currentTex);
        device_.beginPass(rtDenoise_[denoiseParity_], in.width, in.height, false);
        device_.drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);
        device_.endPass();
        denoiseHistoryValid_ = true;
    }

    return currentTex;
}

}  // namespace livebeauty::rhi
