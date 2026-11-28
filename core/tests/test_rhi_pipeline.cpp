#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "livebeauty/engine.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/rhi/rhi_device.h"
#ifdef LIVEBEAUTY_HAS_WEBGL_RHI
#include "livebeauty/rhi/webgl/webgl2_device.h"
#endif

using namespace livebeauty;
using namespace livebeauty::rhi;
using namespace livebeauty::math;

namespace {

int gFailures = 0;

#define EXPECT_TRUE(cond)                                                                      \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " (" << #cond << ")\n";    \
            ++gFailures;                                                                       \
        }                                                                                      \
    } while (0)

struct DrawCall {
    RenderTargetId targetRt;
    ShaderId shader;
    RenderState state;
    TextureBindings textures;
    UniformBlock uniforms;
    bool indexed;
    int indexOffset;
    int indexCount;
};

class MockRhiDevice final : public IRhiDevice {
public:
    std::vector<DrawCall> calls;
    std::vector<TextureFormat> createdRtFormats;
    bool feedbackLoopDetected = false;
    RenderTargetId activeRt = kInvalidRenderTarget;

    RenderTargetId createRenderTarget(int /*w*/, int /*h*/, TextureFormat fmt) override {
        createdRtFormats.push_back(fmt);
        const RenderTargetId rt = nextId_++;
        const TextureId tex = nextId_++;
        rtToTex_[rt] = tex;
        return rt;
    }
    void destroyRenderTarget(RenderTargetId rt) override { rtToTex_.erase(rt); }
    TextureId textureOf(RenderTargetId rt) const override {
        auto it = rtToTex_.find(rt);
        return it != rtToTex_.end() ? it->second : kInvalidTexture;
    }

    TextureId createTexture2D(int, int, TextureFormat, const uint8_t*) override { return nextId_++; }
    void updateTexture2D(TextureId, int, int, TextureFormat, const uint8_t*) override {}
    void destroyTexture(TextureId) override {}

    BufferId createDynamicVertexBuffer(size_t) override { return nextId_++; }
    void updateVertexBuffer(BufferId, const void*, size_t) override {}
    BufferId createStaticIndexBuffer(const uint16_t*, size_t) override { return nextId_++; }
    void destroyBuffer(BufferId) override {}

    void beginPass(RenderTargetId rt, int, int, LoadOp, const Vec4&) override { activeRt = rt; }
    void drawFullScreenQuad(ShaderId shader, const RenderState& st,
                            const TextureBindings& tb, const UniformBlock& ub) override {
        checkFeedback(tb);
        calls.push_back({activeRt, shader, st, tb, ub, false, 0, 0});
    }
    void drawIndexed(ShaderId shader, const RenderState& st, BufferId, BufferId,
                     int indexOffset, int indexCount, VertexLayout,
                     const TextureBindings& tb, const UniformBlock& ub) override {
        checkFeedback(tb);
        calls.push_back({activeRt, shader, st, tb, ub, true, indexOffset, indexCount});
    }
    void endPass() override { activeRt = kInvalidRenderTarget; }

private:
    void checkFeedback(const TextureBindings& tb) {
        const TextureId dstTex = textureOf(activeRt);
        if (dstTex == kInvalidTexture) return;
        for (int i = 0; i < tb.count; ++i) {
            if (tb.slots[i] == dstTex) feedbackLoopDetected = true;
        }
    }

    uint32_t nextId_ = 100;
    std::unordered_map<RenderTargetId, TextureId> rtToTex_;
};

void makeMediaPipeFace(float aspect, std::array<float, FaceMeshFrame::kCoordCount>& mpXy) {
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float scale = 0.42f / FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float mx = 0.5f + xyz[i * 3] * scale;
        const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
        mpXy[i * 2] = mx;
        mpXy[i * 2 + 1] = 1.0f - (my * aspect);
    }
    for (int i = 468; i < 478; ++i) {
        mpXy[i * 2] = 0.5f;
        mpXy[i * 2 + 1] = 0.45f;
    }
}

void testPassThroughZeroCost() {
    MockRhiDevice mock;
    LiveBeautyEngine engine(mock);
    engine.setConfig(BeautyConfig::original());
    const TextureId inputTex = 42;
    const TextureId outTex = engine.processFrame(inputTex, 720, 1280, 1000, 1000);
    EXPECT_TRUE(outTex == inputTex);
    EXPECT_TRUE(mock.calls.empty());
}

void testFullBeautyPipelineOrderingAndNoFeedbackLoop() {
    MockRhiDevice mock;
    LiveBeautyEngine engine(mock);

    PipelineAssets assets{};
    assets.lutTex = 10;
    assets.blushAtlas = 11;
    assets.eyebrowAtlas = 12;
    assets.eyeMakeupAtlas = 13;
    assets.contourHighlightAtlas = 14;
    assets.irisAtlas = 15;
    engine.setAssets(assets);

    BeautyConfig cfg = BeautyConfig::koreanWaterGlow();
    cfg.cranialTop = 0.6f;
    cfg.nasolabial = 0.4f;
    cfg.eyebrow = 0.35f;
    cfg.contour = 0.3f;
    cfg.highlight = 0.3f;
    cfg.backgroundBlur = 0.5f;
    engine.setConfig(cfg);

    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    makeMediaPipeFace(720.0f / 1280.0f, mpXy);

    // Feed 6 frames so presence ramps up cleanly.
    for (int f = 0; f < 6; ++f) {
        const int64_t ts = 1000 + f * 33;
        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts));
        mock.calls.clear();
        const TextureId out = engine.processFrame(42, 720, 1280, ts, ts, 99);
        EXPECT_TRUE(out != 42);
    }

    EXPECT_TRUE(!mock.feedbackLoopDetected);

    // Verify key passes occurred in strict order:
    // BgMix -> MeshWarp (outer no-cull, then inner cull-back) -> RetouchSkin -> RetouchEyeTeeth
    // -> SkinMask (painter's order) -> GuidedCoeff -> CompositePro (11 vec4) -> MakeupLayer -> MakeupLip -> MakeupIris
    int idxBgMix = -1, idxWarpOuter = -1, idxWarpInner = -1, idxRetouchSkin = -1;
    int idxSkinFeather = -1, idxSkinInner = -1, idxGuidedCoeff = -1;
    int idxComposite = -1, idxLip = -1, idxEyeshadow = -1;

    for (int i = 0; i < static_cast<int>(mock.calls.size()); ++i) {
        const auto& c = mock.calls[i];
        if (c.shader == ShaderId::kBgMix) idxBgMix = i;
        if (c.shader == ShaderId::kMeshWarp && !c.state.cullBack && idxWarpOuter < 0) idxWarpOuter = i;
        if (c.shader == ShaderId::kMeshWarp && c.state.cullBack && idxWarpInner < 0) idxWarpInner = i;
        if (c.shader == ShaderId::kRetouchSkin && idxRetouchSkin < 0) idxRetouchSkin = i;
        if (c.shader == ShaderId::kSkinMask && !c.state.cullBack && idxSkinFeather < 0) idxSkinFeather = i;
        if (c.shader == ShaderId::kSkinMask && c.state.cullBack && idxSkinInner < 0) idxSkinInner = i;
        if (c.shader == ShaderId::kGuidedCoeff) idxGuidedCoeff = i;
        if (c.shader == ShaderId::kCompositePro) {
            idxComposite = i;
            // 13 vec4 (including full 3x3 CAT02 WB) + 4 samplers = 17 uniform vectors.
            EXPECT_TRUE(c.uniforms.count == 13);
            EXPECT_TRUE(c.textures.count == 4);
        }
        if (c.shader == ShaderId::kMakeupLayer && c.uniforms.v[2].z > 0.5f && idxEyeshadow < 0) idxEyeshadow = i;
        if (c.shader == ShaderId::kMakeupLip) idxLip = i;
    }

    EXPECT_TRUE(idxBgMix >= 0);
    EXPECT_TRUE(idxWarpOuter > idxBgMix && idxWarpInner > idxWarpOuter);
    EXPECT_TRUE(idxRetouchSkin > idxWarpInner);
    EXPECT_TRUE(idxSkinFeather > idxRetouchSkin && idxSkinInner > idxSkinFeather);
    EXPECT_TRUE(idxGuidedCoeff > idxSkinInner);
    EXPECT_TRUE(idxComposite > idxGuidedCoeff);
    EXPECT_TRUE(idxEyeshadow > idxComposite);
    EXPECT_TRUE(idxLip > idxEyeshadow);
}

void testHairProtectionVirtualBgStickersAndTelemetry() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId personMask = dev.createTexture2D(256, 256, TextureFormat::kR8, nullptr);
    const TextureId hairMask = dev.createTexture2D(216, 384, TextureFormat::kR8, nullptr);
    const TextureId virtualBg = dev.createTexture2D(1080, 1920, TextureFormat::kRgba8, nullptr);
    const TextureId stickerTex = dev.createTexture2D(256, 256, TextureFormat::kRgba8, nullptr);

    PipelineAssets assets{};
    assets.virtualBgTex = virtualBg;
    assets.virtualBgAspect = 9.0f / 16.0f;
    engine.setAssets(assets);

    auto cfg = BeautyConfig::naturalStudio();
    cfg.backgroundMode = BackgroundMode::kVirtualImage;
    cfg.virtualBgStrength = 1.0f;
    engine.setConfig(cfg);

    math::StickerLayerConfig sticker{};
    sticker.enabled = true;
    sticker.anchor = math::StickerAnchor::kForeheadCrown;
    sticker.texture = stickerTex;
    sticker.scale = 1.0f;
    sticker.opacity = 0.95f;
    engine.setSticker(0, sticker);

    // Feed 12 frames so hair presence & face presence ramp up cleanly
    std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
    const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
    const float aspect = 720.0f / 1280.0f;
    const float scale = 0.40f / math::FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        const float mx = 0.5f + xyz[i * 3] * scale;
        const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
        mpXy[i * 2] = mx;
        mpXy[i * 2 + 1] = 1.0f - (my * aspect);
    }
    for (int i = 468; i < 478; ++i) {
        mpXy[i * 2] = 0.5f;
        mpXy[i * 2 + 1] = 0.45f;
    }

    TextureId out = kInvalidTexture;
    for (int f = 0; f < 12; ++f) {
        const int64_t ts = 2000 + f * 33;
        dev.calls.clear();
        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts));
        out = engine.processFrame(inputTex, 720, 1280, ts, ts, personMask, 0.30f, 1.0f, hairMask, ts);
    }

    EXPECT_TRUE(out != kInvalidTexture && out != inputTex);
    EXPECT_TRUE(!dev.feedbackLoopDetected);

    bool sawHairMerge = false;
    bool sawVirtualMix = false;
    bool sawHairAlpha = false;
    bool sawSticker = false;
    float vbgLo = 0.0f, vbgHi = 0.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBgHairMerge) sawHairMerge = true;
        if (c.shader == ShaderId::kBgVirtualMix) {
            sawVirtualMix = true;
            vbgLo = c.uniforms.v[0].y;
            vbgHi = c.uniforms.v[0].z;
        }
        if (c.shader == ShaderId::kHairAlpha) sawHairAlpha = true;
        if (c.shader == ShaderId::kStickerQuad) sawSticker = true;
    }
    // v3.6.4: kVirtualImage skips async kBgHairMerge so 5-12Hz hair masks never cast lagging dark shadows,
    // while kHairAlpha still protects hair from skin whitening/smoothing and kBgVirtualMix uses [0.42, 0.72].
    EXPECT_TRUE(!sawHairMerge);
    EXPECT_TRUE(sawVirtualMix);
    EXPECT_TRUE(std::fabs(vbgLo - 0.42f) < 1e-4f && std::fabs(vbgHi - 0.72f) < 1e-4f);
    EXPECT_TRUE(sawHairAlpha);
    EXPECT_TRUE(sawSticker);

    const auto& st = engine.stats();
    EXPECT_TRUE(st.frameCount == 12);
    EXPECT_TRUE(st.facePresence > 0.9f);
    EXPECT_TRUE(st.hairShown > 0.5f);
    EXPECT_TRUE((st.activePassesMask & kPassBg) != 0);
    EXPECT_TRUE((st.activePassesMask & kPassHairAlpha) != 0);
    EXPECT_TRUE((st.activePassesMask & kPassSticker) != 0);

    // Switching to BackgroundMode::kBlur still activates kBgHairMerge for soft bokeh flyaway preservation
    cfg.backgroundMode = BackgroundMode::kBlur;
    cfg.backgroundBlur = 0.75f;
    engine.setConfig(cfg);
    dev.calls.clear();
    engine.processFrame(inputTex, 720, 1280, 2430, 2430, personMask, 0.75f, 1.0f, hairMask, 2430);
    bool sawBlurHairMerge = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBgHairMerge) sawBlurHairMerge = true;
    }
    EXPECT_TRUE(sawBlurHairMerge);
}

void testHairWeightedGuidedDualFaceAndFrameDelayRing() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId hairMask = dev.createTexture2D(216, 384, TextureFormat::kR8, nullptr);

    auto cfg = BeautyConfig::koreanWaterGlow();
    cfg.maxFaces = 2;
    cfg.frameDelayFrames = 2;
    engine.setConfig(cfg);

    auto makeFaceMp = [](float cx, float fw) {
        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = fw / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            const float mx = cx + xyz[i * 3] * scale;
            const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
            mpXy[i * 2] = mx;
            mpXy[i * 2 + 1] = 1.0f - (my * aspect);
        }
        for (int i = 468; i < 478; ++i) {
            mpXy[i * 2] = cx;
            mpXy[i * 2 + 1] = 0.45f;
        }
        return mpXy;
    };
    const auto faceLeft = makeFaceMp(0.28f, 0.28f);
    const auto faceRight = makeFaceMp(0.72f, 0.26f);
    math::MultiFaceTracker::RawFaceInput pair[2] = {{faceLeft.data(), 478}, {faceRight.data(), 478}};

    TextureId out = kInvalidTexture;
    int64_t lastTs = 0;
    for (int f = 0; f < 10; ++f) {
        lastTs = 5000 + f * 33;
        dev.calls.clear();
        EXPECT_TRUE(engine.submitMultiFaceLandmarks(lastTs, pair, 2, 720, 1280, lastTs) == 2);
        out = engine.processFrame(inputTex, 720, 1280, lastTs, lastTs, kInvalidTexture, 0.30f, 1.0f, hairMask, lastTs);
    }

    EXPECT_TRUE(out != kInvalidTexture && out != inputTex);
    EXPECT_TRUE(!dev.feedbackLoopDetected);

    bool sawWeightBlur = false, sawWeightDiff = false, sawWeightCoeff = false;
    int skinMaskDraws = 0;
    int hairAlphaIndexedDraws = 0;
    int blurCallsAfterSkinMask = 0;
    int lastSkinMaskIdx = -1;
    int firstHairAlphaIdx = -1;
    for (int i = 0; i < static_cast<int>(dev.calls.size()); ++i) {
        const auto& c = dev.calls[i];
        if (c.shader == ShaderId::kGuidedWeightBlur) sawWeightBlur = true;
        if (c.shader == ShaderId::kGuidedWeightDiff) sawWeightDiff = true;
        if (c.shader == ShaderId::kGuidedWeightCoeff) {
            sawWeightCoeff = true;
            // Verify vertex uU0 (dir) and fragment uU1 (eps, evenness) are strictly disjoint.
            EXPECT_TRUE(c.uniforms.count == 2);
            EXPECT_TRUE(c.uniforms.v[0].z == 0.0f && c.uniforms.v[0].w == 0.0f);
            EXPECT_TRUE(c.uniforms.v[1].x > 0.0f && c.uniforms.v[1].y > 0.0f);
        }
        if (c.shader == ShaderId::kSkinMask) {
            ++skinMaskDraws;
            lastSkinMaskIdx = i;
        }
        if (c.shader == ShaderId::kHairAlpha) {
            if (firstHairAlphaIdx < 0) firstHairAlphaIdx = i;
            if (c.indexed) ++hairAlphaIndexedDraws;
        }
    }
    for (int i = lastSkinMaskIdx + 1; i < firstHairAlphaIdx; ++i) {
        if (dev.calls[i].shader == ShaderId::kBlurRgb) ++blurCallsAfterSkinMask;
    }
    EXPECT_TRUE(sawWeightBlur);
    EXPECT_TRUE(sawWeightDiff);
    EXPECT_TRUE(sawWeightCoeff);
    // Primary + Secondary face each issue auto neck-apron + feather + inner skin-mask draws = 6 draws
    EXPECT_TRUE(skinMaskDraws == 6);
    // SkinMask must execute 2-pass H+V Gaussian feathering BEFORE HairAlpha writes alpha
    EXPECT_TRUE(blurCallsAfterSkinMask == 2);
    // HairAlpha must follow primary face mesh warp (outer + inner indexed passes)
    EXPECT_TRUE(hairAlphaIndexedDraws == 2);

    const auto& st = engine.stats();
    EXPECT_TRUE(st.activeFaces == 2);
    EXPECT_TRUE((st.activePassesMask & kPassDualFace) != 0);
    EXPECT_TRUE((st.activePassesMask & kPassFrameDelay) != 0);
    // FrameDelayRing with delayFrames=2 must display the frame from 2 frames earlier (66 ms lag)
    EXPECT_TRUE(st.displayedTimestampMs == lastTs - 66);
}

void testFourFrameDelayStage0DenoiseAndLutFilterSuppression() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId lutTex = dev.createTexture2D(512, 512, TextureFormat::kRgba8, nullptr);
    PipelineAssets assets{};
    assets.lutTex = lutTex;
    engine.setAssets(assets);

    auto cfg = BeautyConfig::naturalStudio();
    cfg.filter = ColorFilterStyle::kWarm;
    cfg.filterIntensity = 0.8f;
    cfg.frameDelayFrames = 4;
    engine.setConfig(cfg);

    int64_t lastTs = 0;
    for (int f = 0; f < 8; ++f) {
        lastTs = 9000 + f * 33;
        dev.calls.clear();
        // Low light (0.06) triggers Stage 0 temporal denoise
        engine.processFrame(inputTex, 720, 1280, lastTs, lastTs, kInvalidTexture, 0.06f, 1.0f);
    }
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    EXPECT_TRUE(engine.stats().displayedTimestampMs == lastTs - 132);

    int idxDenoise = -1, idxComposite = -1;
    for (int i = 0; i < static_cast<int>(dev.calls.size()); ++i) {
        const auto& c = dev.calls[i];
        if (c.shader == ShaderId::kTemporalDenoise) {
            idxDenoise = i;
        }
        if (c.shader == ShaderId::kCompositePro) {
            idxComposite = i;
            // LUT active -> legacy style (0) + skinOnly (16) = 16, legacy filterIntensity = 0
            EXPECT_TRUE(std::fabs(c.uniforms.v[1].x - 16.0f) < 1e-4f);
            EXPECT_TRUE(c.uniforms.v[2].w == 0.0f);
        }
    }
    // v3.9.5 (tone-2): Stage 0 8-bit temporal denoise removed (zero wasted pass/VRAM).
    EXPECT_TRUE(idxDenoise == -1 && idxComposite >= 0);
}

void testP1ToP4PassesAndHardwareTierGating() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    engine.setHardwareProfile("Adreno (TM) 750", 8, 12288);  // Ultra tier

    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId personMask = dev.createTexture2D(256, 256, TextureFormat::kR8, nullptr);
    const TextureId occMask = dev.createTexture2D(256, 256, TextureFormat::kR8, nullptr);
    const TextureId aiResTex = dev.createTexture2D(128, 128, TextureFormat::kRgba8, nullptr);
    const TextureId arTex = dev.createTexture2D(128, 128, TextureFormat::kRgba8, nullptr);
    const TextureId atlasTex = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);

    PipelineAssets assets{};
    assets.blushAtlas = atlasTex;
    assets.contourHighlightAtlas = atlasTex;
    engine.setAssets(assets);
    engine.setAmbientRgb(1.12f, 1.00f, 0.88f);
    engine.setOcclusionMask(occMask);
    engine.setAiResidual(aiResTex, {0.2f, 0.2f, 0.6f, 0.6f}, 0.75f);

    math::Ar3dAccessoryConfig ear{};
    ear.enabled = true;
    ear.anchor = math::Ar3dAnchor::kLeftEar;
    ear.texture = arTex;
    ear.springStiffness = 20.0f;
    ear.headOcclusion = true;
    engine.setAr3dAccessory(0, ear);

    auto cfg = BeautyConfig::naturalStudio();
    cfg.relight = 0.65f;
    cfg.relightStyle = math::RelightStyle::kStudioSoftbox;
    cfg.oilControl = 0.55f;
    cfg.neckTone = 0.60f;
    cfg.backgroundBlur = 0.70f;
    cfg.matteTemporal = 0.70f;
    cfg.blemishRemove = 0.60f;
    cfg.blush = 0.45f;
    cfg.lipstick = 0.50f;
    cfg.performanceTier = math::PerformanceTier::kAuto;
    engine.setConfig(cfg);

    std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
    const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
    const float aspect = 720.0f / 1280.0f;
    const float scale = 0.36f / math::FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        const float mx = 0.5f + xyz[i * 3] * scale;
        const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
        mpXy[i * 2] = mx;
        mpXy[i * 2 + 1] = 1.0f - (my * aspect);
    }
    for (int i = 468; i < 478; ++i) { mpXy[i * 2] = 0.5f; mpXy[i * 2 + 1] = 0.45f; }

    for (int f = 0; f < 6; ++f) {
        const int64_t ts = 12000 + f * 33;
        dev.calls.clear();
        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts));
        engine.processFrame(inputTex, 720, 1280, ts, ts, personMask, 0.30f, 1.0f);
    }
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    const auto& stUltra = engine.stats();
    EXPECT_TRUE(stUltra.resolvedTier == static_cast<int>(math::PerformanceTier::kUltra));
    EXPECT_TRUE((stUltra.activePassesMask & kPassRelight3d) != 0);
    EXPECT_TRUE((stUltra.activePassesMask & kPassMatteRefine) != 0);
    EXPECT_TRUE((stUltra.activePassesMask & kPassBlemishHeal) != 0);
    EXPECT_TRUE((stUltra.activePassesMask & kPassAiResidual) != 0);
    EXPECT_TRUE((stUltra.activePassesMask & kPassAr3d) != 0);
    EXPECT_TRUE((stUltra.activePassesMask & kPassHairAlpha) != 0);

    bool sawMatte = false, sawBlemish = false, sawAiRes = false, sawHeadDepth = false, sawAr3d = false;
    int skinMaskDraws = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMatteRefine) sawMatte = true;
        if (c.shader == ShaderId::kBlemishHeal) sawBlemish = true;
        if (c.shader == ShaderId::kAiResidualBlend) sawAiRes = true;
        if (c.shader == ShaderId::kHeadDepthShell) sawHeadDepth = true;
        if (c.shader == ShaderId::kAr3dMesh) sawAr3d = true;
        if (c.shader == ShaderId::kSkinMask) ++skinMaskDraws;
    }
    EXPECT_TRUE(sawMatte && sawBlemish && sawAiRes && sawHeadDepth && sawAr3d);
    // SkinMask with neckTone > 0 draws neck apron + outer feather + inner face = 3 draws
    EXPECT_TRUE(skinMaskDraws == 3);

    // Now force Lite tier: heavy passes (Relight3d, MatteRefine, BlemishHeal, Ar3d) must all cleanly gate off!
    cfg.performanceTier = math::PerformanceTier::kLite;
    engine.setConfig(cfg);
    dev.calls.clear();
    engine.processFrame(inputTex, 720, 1280, 12231, 12231, personMask, 0.30f, 1.0f);
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    const auto& stLite = engine.stats();
    EXPECT_TRUE(stLite.resolvedTier == static_cast<int>(math::PerformanceTier::kLite));
    EXPECT_TRUE((stLite.activePassesMask & kPassRelight3d) == 0);
    EXPECT_TRUE((stLite.activePassesMask & kPassMatteRefine) == 0);
    EXPECT_TRUE((stLite.activePassesMask & kPassBlemishHeal) == 0);
    EXPECT_TRUE((stLite.activePassesMask & kPassAr3d) == 0);
}

void testV34HairDyeHalfFloatAndFaceParsingIntegration() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    engine.setHardwareProfile("Apple M3 GPU", 8, 8192);

    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    PipelineAssets assets{};
    assets.contourHighlightAtlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
    assets.blushAtlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
    engine.setAssets(assets);

    auto cfg = BeautyConfig::clearColdWhite();
    cfg.lutUrl = "";
    cfg.hairDye = 0.80f;
    cfg.hairDyeColor = HairDyeColor::kHazeBlue;
    cfg.hairShine = 0.65f;
    cfg.preferHalfFloat = true;
    engine.setConfig(cfg);

    // Feed 19-class CelebAMask-HQ semantic parsing map with hair (class 17) on upper half and skin (class 1) on center
    std::vector<uint8_t> parsing(64 * 64, 0);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            if (y < 24) parsing[y * 64 + x] = 17;       // hair
            else if (y < 52) parsing[y * 64 + x] = 1;   // skin
        }
    }

    std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
    makeMediaPipeFace(720.0f / 1280.0f, mpXy);

    for (int f = 0; f < 8; ++f) {
        const int64_t ts = 20000 + f * 33;
        dev.calls.clear();
        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts));
        EXPECT_TRUE(engine.submitFaceParsingClassMap(parsing.data(), 64, 64, ts));
        const TextureId out = engine.processFrame(inputTex, 720, 1280, ts, ts);
        EXPECT_TRUE(out != kInvalidTexture && out != inputTex);
    }

    EXPECT_TRUE(!dev.feedbackLoopDetected);
    // Verify RGBA16F intermediate RTs were created when preferHalfFloat=true on non-Lite tier
    int fp16Count = 0;
    for (auto fmt : dev.createdRtFormats) {
        if (fmt == TextureFormat::kRgba16f) ++fp16Count;
    }
    EXPECT_TRUE(fp16Count == 5);  // v3.9.4 D1: rtGuided_[0..2], rtMacro_[0..1] FP16; full-res rtBody_/rtWarp_ RGBA8

    int idxComposite = -1, idxHairDye = -1, idxLip = -1;
    for (int i = 0; i < static_cast<int>(dev.calls.size()); ++i) {
        if (dev.calls[i].shader == ShaderId::kCompositePro) idxComposite = i;
        if (dev.calls[i].shader == ShaderId::kHairDye) idxHairDye = i;
        if (dev.calls[i].shader == ShaderId::kMakeupLip) idxLip = i;
    }
    EXPECT_TRUE(idxComposite >= 0);
    EXPECT_TRUE(idxHairDye > idxComposite);
    EXPECT_TRUE(idxLip > idxHairDye);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) != 0);
}

void testV35EffectBundleTriggersAndWebGl2Backend() {
    // 1. Declarative .lbfx EffectBundle with expression-gated sticker & 36-index curved 3D glasses
    {
        MockRhiDevice dev;
        LiveBeautyEngine engine(dev);
        engine.setHardwareProfile("Apple M3 GPU", 8, 8192);

        const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        const TextureId heartTex = dev.createTexture2D(128, 128, TextureFormat::kRgba8, nullptr);
        const TextureId glassesTex = dev.createTexture2D(256, 128, TextureFormat::kRgba8, nullptr);

        const std::string json = R"JSON({
            "schemaVersion": 1,
            "id": "smile_sparkle_glasses",
            "beauty": { "preset": "natural_studio", "lutUrl": "" },
            "stickers": [
                { "slot": 0, "anchor": "cheeks", "assetPath": "heart.png", "trigger": "smile", "opacity": 0.9 }
            ],
            "ar3d": [
                { "slot": 0, "anchor": "glasses_bridge", "meshPreset": "curved_glasses", "assetPath": "glasses.png", "opacity": 0.95 }
            ]
        })JSON";

        std::string err;
        const bool loaded = engine.applyEffectBundleJson(
            json,
            [&](const std::string& path) -> TextureId {
                if (path == "heart.png") return heartTex;
                if (path == "glasses.png") return glassesTex;
                return kInvalidTexture;
            },
            &err);
        EXPECT_TRUE(loaded && err.empty());

        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        makeMediaPipeFace(720.0f / 1280.0f, mpXy);

        // Neutral face (smile = 0): curved_glasses (36 indices) renders, smile-gated sticker stays hidden!
        engine.submitFaceBlendshapes({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        for (int f = 0; f < 5; ++f) {
            const int64_t ts = 30000 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts);
            engine.processFrame(inputTex, 720, 1280, ts, ts);
        }
        bool sawStickerNeutral = false;
        int glassesIndexCount = 0;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kStickerQuad) sawStickerNeutral = true;
            if (c.shader == ShaderId::kAr3dMesh) glassesIndexCount = c.indexCount;
        }
        EXPECT_TRUE(!sawStickerNeutral);
        EXPECT_TRUE(glassesIndexCount == 36);

        // Smiling face (smile = 0.85): smile-gated sticker smoothly activates!
        for (int f = 0; f < 4; ++f) {
            engine.submitFaceBlendshapes({0.85f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            const int64_t ts = 30300 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts);
            engine.processFrame(inputTex, 720, 1280, ts, ts);
        }
        bool sawStickerSmile = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kStickerQuad) sawStickerSmile = true;
        }
        EXPECT_TRUE(sawStickerSmile);
    }

#ifdef LIVEBEAUTY_HAS_WEBGL_RHI
    // 2. WebGL2 / Emscripten RHI backend full pipeline execution (GL_RGBA16F + GL_R8 + 31 shaders)
    {
        struct FakeWebGlState {
            uint32_t nextHandle = 1;
            int rgba16fCount = 0;
            int r8Count = 0;
            int drawCalls = 0;
            int linkedPrograms = 0;
        };
        static FakeWebGlState sGl{};
        sGl = FakeWebGlState{};

        webgl::WebGl2DispatchTable tbl{};
        tbl.genTextures = [](int n, uint32_t* ids) { for (int i = 0; i < n; ++i) ids[i] = sGl.nextHandle++; };
        tbl.deleteTextures = [](int, const uint32_t*) {};
        tbl.bindTexture = [](uint32_t, uint32_t) {};
        tbl.activeTexture = [](uint32_t) {};
        tbl.texParameteri = [](uint32_t, uint32_t, int) {};
        tbl.pixelStorei = [](uint32_t, int) {};
        tbl.texImage2D = [](uint32_t, int, int internalFmt, int, int, int, uint32_t, uint32_t, const void*) {
            if (internalFmt == 0x881A) ++sGl.rgba16fCount;
            if (internalFmt == 0x8229) ++sGl.r8Count;
        };
        tbl.texSubImage2D = [](uint32_t, int, int, int, int, int, uint32_t, uint32_t, const void*) {};
        tbl.genFramebuffers = [](int n, uint32_t* ids) { for (int i = 0; i < n; ++i) ids[i] = sGl.nextHandle++; };
        tbl.deleteFramebuffers = [](int, const uint32_t*) {};
        tbl.bindFramebuffer = [](uint32_t, uint32_t) {};
        tbl.framebufferTexture2D = [](uint32_t, uint32_t, uint32_t, uint32_t, int) {};
        tbl.checkFramebufferStatus = [](uint32_t) -> uint32_t { return 0x8CD5u; };  // GL_FRAMEBUFFER_COMPLETE
        tbl.genBuffers = [](int n, uint32_t* ids) { for (int i = 0; i < n; ++i) ids[i] = sGl.nextHandle++; };
        tbl.deleteBuffers = [](int, const uint32_t*) {};
        tbl.bindBuffer = [](uint32_t, uint32_t) {};
        tbl.bufferData = [](uint32_t, ptrdiff_t, const void*, uint32_t) {};
        tbl.createShader = [](uint32_t) -> uint32_t { return sGl.nextHandle++; };
        tbl.deleteShader = [](uint32_t) {};
        tbl.shaderSource = [](uint32_t, int, const char* const*, const int*) {};
        tbl.compileShader = [](uint32_t) {};
        tbl.getShaderiv = [](uint32_t, uint32_t, int* out) { *out = 1; };
        tbl.createProgram = []() -> uint32_t { return sGl.nextHandle++; };
        tbl.deleteProgram = [](uint32_t) {};
        tbl.attachShader = [](uint32_t, uint32_t) {};
        tbl.linkProgram = [](uint32_t) { ++sGl.linkedPrograms; };
        tbl.getProgramiv = [](uint32_t, uint32_t, int* out) { *out = 1; };
        tbl.useProgram = [](uint32_t) {};
        tbl.getAttribLocation = [](uint32_t, const char*) -> int { return 0; };
        tbl.getUniformLocation = [](uint32_t, const char*) -> int { return 0; };
        tbl.enableVertexAttribArray = [](uint32_t) {};
        tbl.disableVertexAttribArray = [](uint32_t) {};
        tbl.vertexAttribPointer = [](uint32_t, int, uint32_t, uint8_t, int, const void*) {};
        tbl.uniform1i = [](int, int) {};
        tbl.uniform4f = [](int, float, float, float, float) {};
        tbl.viewport = [](int, int, int, int) {};
        tbl.enable = [](uint32_t) {};
        tbl.disable = [](uint32_t) {};
        tbl.blendFuncSeparate = [](uint32_t, uint32_t, uint32_t, uint32_t) {};
        tbl.frontFace = [](uint32_t) {};
        tbl.cullFace = [](uint32_t) {};
        tbl.colorMask = [](uint8_t, uint8_t, uint8_t, uint8_t) {};
        tbl.clearColor = [](float, float, float, float) {};
        tbl.clear = [](uint32_t) {};
        tbl.drawArrays = [](uint32_t, int, int) { ++sGl.drawCalls; };
        tbl.drawElements = [](uint32_t, int, uint32_t, const void*) { ++sGl.drawCalls; };
        tbl.getError = []() -> uint32_t { return 0u; };
        tbl.enableExtension = [](const char*) -> bool { return true; };

        webgl::WebGl2Device webglDev(tbl);
        EXPECT_TRUE(webglDev.supportsHalfFloatColorBuffer());

        LiveBeautyEngine webEngine(webglDev);
        webEngine.setHardwareProfile("WebGL2 Apple GPU", 8, 8192);
        const TextureId videoTex = webglDev.wrapExternalTexture(777u);
        const TextureId r8Mask = webglDev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);

        auto cfg = BeautyConfig::naturalStudio();
        cfg.lutUrl = "";
        cfg.backgroundBlur = 0.45f;
        webEngine.setConfig(cfg);

        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        makeMediaPipeFace(720.0f / 1280.0f, mpXy);
        webEngine.submitFaceLandmarks(1000, mpXy.data(), 478, 720, 1280, 1000);
        const TextureId out = webEngine.processFrame(videoTex, 720, 1280, 1000, 1000, r8Mask);
        EXPECT_TRUE(out != kInvalidTexture && webglDev.unwrapWebGlTexture(out) != 777u);
        EXPECT_TRUE(sGl.rgba16fCount == 5);
        EXPECT_TRUE(sGl.r8Count >= 1);
        EXPECT_TRUE(sGl.linkedPrograms >= 8);
        EXPECT_TRUE(sGl.drawCalls >= 10);
        webEngine.releaseGpuResources();
        webglDev.releaseAll(true);
    }
#endif
}

void testCodeReviewEngineLifecycleAndPriorityGuards() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    engine.setHardwareProfile("Apple M3 GPU", 8, 8192);
    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId arTex = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);

    auto cfg = BeautyConfig::clearColdWhite();
    cfg.lutUrl = "";
    cfg.hairDye = 0.80f;
    cfg.maxFaces = 2;
    engine.setConfig(cfg);

    std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
    makeMediaPipeFace(720.0f / 1280.0f, mpXy);
    std::vector<uint8_t> parsing(32 * 32, 17);  // hair class

    engine.submitFaceLandmarks(1000, mpXy.data(), 478, 720, 1280, 1000);
    engine.submitFaceParsingClassMap(parsing.data(), 32, 32, 1000);
    engine.processFrame(inputTex, 720, 1280, 1000, 1000);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) != 0);

    // 1. resetTracking() must clear autoParseHairValid_ so ghost hair mask never persists
    engine.resetTracking();
    engine.submitFaceLandmarks(2000, mpXy.data(), 478, 720, 1280, 2000);
    engine.processFrame(inputTex, 720, 1280, 2000, 2000);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) == 0);

    // 2. Stack-scoped custom 3D mesh is deep-copied safely; OOB custom indices are rejected
    {
        std::vector<math::Ar3dMeshVertex> stackVerts = {
            {-0.4f, -0.4f, 0.0f, 0.0f, 0.0f},
            { 0.4f, -0.4f, 0.0f, 1.0f, 0.0f},
            { 0.0f,  0.4f, 0.0f, 0.5f, 1.0f},
        };
        std::vector<uint16_t> stackIdx = {0, 1, 2};
        math::Ar3dAccessoryConfig customAcc{};
        customAcc.enabled = true;
        customAcc.texture = arTex;
        customAcc.customVertices = stackVerts.data();
        customAcc.customVertexCount = 3;
        customAcc.customIndices = stackIdx.data();
        customAcc.customIndexCount = 3;
        engine.setAr3dAccessory(0, customAcc);

        std::vector<uint16_t> badIdx = {0, 1, 99};
        customAcc.customIndices = badIdx.data();
        engine.setAr3dAccessory(1, customAcc);  // invalid OOB custom mesh must be disabled safely
    }
    dev.calls.clear();
    engine.submitFaceLandmarks(2033, mpXy.data(), 478, 720, 1280, 2033);
    engine.processFrame(inputTex, 720, 1280, 2033, 2033);
    int ar3dDraws = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kAr3dMesh) {
            ++ar3dDraws;
            EXPECT_TRUE(c.indexCount == 3);
        }
    }
    EXPECT_TRUE(ar3dDraws == 1);

    // 3. applyEffectBundleJson without pipelineOptions preserves existing runtime maxFaces=2
    EXPECT_TRUE(engine.applyEffectBundleJson(R"({"schemaVersion":1,"beauty":{"smoothing":0.5}})", nullptr, nullptr));
    EXPECT_TRUE(engine.config().maxFaces == 2);
}

void testV36ChromaKeyParticlesMulticlassAndHandShield() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    engine.setHardwareProfile("Adreno 750", 8, 12288);
    const TextureId inputTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId vbgTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId stickerTex = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);

    PipelineAssets assets{};
    assets.virtualBgTex = vbgTex;
    assets.virtualBgAspect = 720.0f / 1280.0f;
    engine.setAssets(assets);

    // 1. Chroma Key works without personMaskTex, and ParticleBurst works with zero feedback loop
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.backgroundMode = BackgroundMode::kChromaKey;
    cfg.chromaKeyColor = ChromaKeyColor::kGreen;
    cfg.virtualBgStrength = 1.0f;
    cfg.waistSlim = 0.50f;
    engine.setConfig(cfg);

    std::array<PoseLandmark, 33> pose{};
    for (auto& p : pose) p = {0.5f, 0.5f, 0.9f};
    pose[11] = {0.35f, 0.32f, 0.95f};
    pose[12] = {0.65f, 0.32f, 0.95f};
    pose[23] = {0.39f, 0.58f, 0.95f};
    pose[24] = {0.61f, 0.58f, 0.95f};
    pose[27] = {0.40f, 0.90f, 0.95f};
    pose[28] = {0.60f, 0.90f, 0.95f};
    EXPECT_TRUE(engine.submitBodyPose(1000, pose.data(), 33, 720, 1280));
    engine.setHandShieldCoords(0.38f, 0.50f, -1.0f, -1.0f);
    engine.triggerGesture(GestureType::kHeart, 0.50f, 0.45f, 1000);

    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    makeMediaPipeFace(720.0f / 1280.0f, mpXy);
    engine.submitFaceLandmarks(1000, mpXy.data(), 478, 720, 1280, 1000);

    StickerLayerConfig heartCrown{};
    heartCrown.enabled = true;
    heartCrown.texture = stickerTex;
    heartCrown.opacity = 1.0f;
    engine.setSticker(0, heartCrown, bundle::BundleTriggerRule::kGestureHeart);

    // Warm up 1 frame at t=1000ms so BodyGeometrySmoother fade-in dt > 0 at t=1150ms
    engine.processFrame(inputTex, 720, 1280, 1000, 1000);
    dev.calls.clear();
    const TextureId outTex = engine.processFrame(inputTex, 720, 1280, 1150, 1150);
    EXPECT_TRUE(outTex != inputTex);
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    const uint32_t mask = engine.stats().activePassesMask;
    EXPECT_TRUE((mask & kPassChromaKey) != 0);
    EXPECT_TRUE((mask & kPassParticles) != 0);
    EXPECT_TRUE((mask & kPassSticker) != 0);
    EXPECT_TRUE((mask & kPassBody) != 0);

    bool sawHandShieldUniform = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBodyWarp && c.uniforms.count >= 4 && c.uniforms.v[2].w > 0.5f) {
            sawHandShieldUniform = std::fabs(c.uniforms.v[3].x - 0.38f) < 1e-4f &&
                                   std::fabs(c.uniforms.v[3].y - 0.50f) < 1e-4f;
        }
    }
    EXPECT_TRUE(sawHandShieldUniform);

    // 2. Multiclass 6-way NHWC logits auto-supply both hairMaskTex and fallback personMaskTex
    std::vector<float> logits(16 * 16 * 6, -3.0f);
    for (int i = 0; i < 16 * 16; ++i) {
        logits[static_cast<size_t>(i * 6 + (i < 128 ? 1 : 3))] = 4.0f;
    }
    EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, 1200));
    cfg.backgroundMode = BackgroundMode::kVirtualImage;
    cfg.hairDye = 0.75f;
    engine.setConfig(cfg);
    engine.processFrame(inputTex, 720, 1280, 1250, 1250);
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBg) != 0);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) != 0);

    // v3.6.5: When hairDye == 0 and hasMakeup() == false (e.g. cold-porcelain whiten + virtual background),
    // internal 11Hz multiclass hair/occlusion masks MUST NOT inject jumping nonHair holes via kHairAlpha!
    cfg.hairDye = 0.0f;
    cfg.whiten = 0.65f;
    engine.setConfig(cfg);
    engine.processFrame(inputTex, 720, 1280, 1266, 1266);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairAlpha) == 0);
    cfg.hairDye = 0.75f;
    engine.setConfig(cfg);

    // 3. v3.6.2 Camera Shake Anti-Ghosting Gate:
    //    - IMU setCameraShaking(true) immediately flushes temporal denoise/matte and suppresses stale async hair/occlusion masks;
    //    - Large >10% face displacement since mask capture (exceeding HairMaskTuning::kAffineMaxShift) suppresses stale mask instead of snapping to identity!
    cfg.smoothing = 0.65f;
    cfg.matteTemporal = 0.70f;
    cfg.lipstick = 0.70f;
    cfg.noseSlim = 0.60f;
    cfg.eyeEnlarge = 0.60f;
    PipelineAssets lipAssets = assets;
    lipAssets.contourHighlightAtlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
    engine.setAssets(lipAssets);
    engine.setConfig(cfg);
    // Activate far-field ROI crop first, then verify setCameraShaking(true) forces full-frame ROI immediately
    std::array<float, FaceMeshFrame::kCoordCount> farMp{};
    {
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = 0.22f / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            farMp[i * 2] = 0.50f + xyz[i * 3] * scale;
            farMp[i * 2 + 1] = 1.0f - ((0.50f / aspect) + xyz[i * 3 + 1] * scale) * aspect;
        }
        for (int i = 468; i < 478; ++i) {
            farMp[i * 2] = 0.50f;
            farMp[i * 2 + 1] = 0.50f;
        }
    }
    for (int k = 0; k < 4; ++k) {
        engine.submitFaceLandmarks(1215 + k * 20, farMp.data(), 478, 720, 1280, 1215 + k * 20);
    }
    EXPECT_TRUE(!engine.selectFaceRoi(720, 1280, 1278).isFullFrame());

    engine.setCameraShaking(true);
    EXPECT_TRUE(engine.selectFaceRoi(720, 1280, 1280).isFullFrame());
    dev.calls.clear();
    engine.processFrame(inputTex, 720, 1280, 1283, 1283, kInvalidTexture, 0.05f, 1.0f);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassDenoise) == 0);
    // v3.9.5 (segmentation-v2): During camera shake, spatial RGB-guided matte refine stays active
    // while temporal history blending drops strictly to 0 (no ghosting!).
    EXPECT_TRUE((engine.stats().activePassesMask & kPassMatteRefine) != 0);
    bool sawZeroTemporalMatteOnShake = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMatteRefine && c.uniforms.count >= 1) {
            sawZeroTemporalMatteOnShake = (c.uniforms.v[0].z <= 1e-4f && c.uniforms.v[0].w <= 1e-4f);
        }
    }
    EXPECT_TRUE(sawZeroTemporalMatteOnShake);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) == 0);
    // v3.6.3: Fallback autoParsePersonTex_ MUST remain bound during camera shake so virtual background never flashes off!
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBg) != 0);
    // v3.6.6: Gyro shake alone must NOT black out live-tracked lip/nose/eye beauty (kMakeupLip & kMeshWarp stay active)!
    bool sawShakeLip = false;
    bool sawShakeWarp = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMakeupLip) sawShakeLip = true;
        if (c.shader == ShaderId::kMeshWarp) sawShakeWarp = true;
    }
    EXPECT_TRUE(sawShakeLip);
    EXPECT_TRUE(sawShakeWarp);
    engine.setCameraShaking(false);

    // Shift face by +0.25 X (> kAffineMaxShift = 0.10) while multiclass mask stays at t=1200ms (x=0.50):
    // stale hair mask must NOT render uncompensated ghost at old position!
    for (int i = 0; i < 478; ++i) mpXy[i * 2] += 0.25f;
    engine.submitFaceLandmarks(1350, mpXy.data(), 478, 720, 1280, 1350);
    engine.processFrame(inputTex, 720, 1280, 1350, 1350);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) == 0);

    // 4. v3.6.3 Virtual Background Load-Governor Privacy Invariant:
    //    Even when loadFadeBody == 0.0f (EffectsLoadGovernor level >= 1), kVirtualBlur / kVirtualImage
    //    MUST stay 100% active at full strength while skipping optional 4-pass background bokeh blur!
    {
        MockRhiDevice pipeDev;
        RenderPipeline pipe(pipeDev);
        PipelineAssets pAssets{};
        pAssets.virtualBgTex = pipeDev.createTexture2D(540, 960, TextureFormat::kRgba8, nullptr);
        pAssets.virtualBgAspect = 9.0f / 16.0f;
        pipe.setAssets(pAssets);

        BeautyConfig pCfg{};
        pCfg.enabled = true;
        pCfg.backgroundMode = BackgroundMode::kVirtualBlur;
        pCfg.virtualBgStrength = 1.0f;
        pCfg.matteTemporal = 0.0f;

        FrameInputs pIn{};
        pIn.inputTexture = pipeDev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        pIn.personMaskTex = pipeDev.createTexture2D(160, 288, TextureFormat::kR8, nullptr);
        pIn.width = 720;
        pIn.height = 1280;
        pIn.loadFadeBody = 0.0f;  // Simulates governor level >= 1!

        pipeDev.calls.clear();
        const TextureId outTex = pipe.render(pCfg, pIn);
        EXPECT_TRUE(outTex != pIn.inputTexture);
        EXPECT_TRUE((pipe.lastActivePassesMask() & kPassBg) != 0);
        bool sawVbgMix = false;
        bool sawBgBlur = false;
        float appliedStrength = 0.0f;
        for (const auto& call : pipeDev.calls) {
            if (call.shader == ShaderId::kBgVirtualMix) {
                sawVbgMix = true;
                appliedStrength = call.uniforms.v[0].x;
            }
            if (call.shader == ShaderId::kBgBlur) sawBgBlur = true;
        }
        EXPECT_TRUE(sawVbgMix);
        EXPECT_TRUE(!sawBgBlur);
        EXPECT_TRUE(std::fabs(appliedStrength - 1.0f) < 1e-4f);
        pipe.releaseGpuResources();
    }
}

void testV3610CheekMakeupSteadyFaceAndAutoOccDecoupling() {
    // 1. Cheek blush & contour must follow steadyFace (NOT steadyEye!), so brief eye-radius extrapolation
    //    never pumps cheek makeup on/off while contact lens still respects steadyEye.
    {
        MockRhiDevice dev;
        RenderPipeline pipe(dev);
        const TextureId inTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        const TextureId atlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
        PipelineAssets assets{};
        assets.blushAtlas = atlas;
        assets.contourHighlightAtlas = atlas;
        assets.irisAtlas = atlas;
        pipe.setAssets(assets);

        BeautyConfig cfg{};
        cfg.enabled = true;
        cfg.skinOnly = true;
        cfg.smoothing = 0.55f;
        cfg.blush = 0.40f;
        cfg.contour = 0.30f;
        cfg.contactLens = 0.40f;

        std::array<float, math::FaceMeshFrame::kCoordCount> lm{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = 0.40f / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            lm[i * 2] = 0.50f + xyz[i * 3] * scale;
            lm[i * 2 + 1] = 0.50f + xyz[i * 3 + 1] * scale * aspect;
        }
        math::FaceMeshFrame::synthesizeIris(lm.data(), 468, 33, 133, 159, 145, aspect);
        math::FaceMeshFrame::synthesizeIris(lm.data(), 473, 362, 263, 386, 374, aspect);
        float meta[math::FaceMeshFrame::kMetaSize]{};
        EXPECT_TRUE(math::FaceMeshFrame::fillMeta(lm.data(), math::FaceMeshFrame::kCoordCount, aspect, meta));

        FrameInputs in{};
        in.inputTexture = inTex;
        in.width = 720;
        in.height = 1280;
        in.nowMs = 1000;
        in.hasFaceMesh = true;
        in.landmarks478x2 = lm.data();
        in.faceMeta15 = meta;
        in.hasIris = true;
        in.facePresence = 1.0f;
        in.steadyFace = 1.0f;
        in.steadyEye = 0.25f;  // Simulates brief eye-scale extrapolation dip!
        in.eyeMakeupKeep = 0.25f;

        dev.calls.clear();
        pipe.render(cfg, in);

        float blushAlpha = 0.0f;
        float contourAlpha = 0.0f;
        float irisStrength = 1.0f;
        int makeupLayerIdx = 0;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kMakeupLayer) {
                if (makeupLayerIdx == 0) blushAlpha = c.uniforms.v[0].w;
                else if (makeupLayerIdx == 1) contourAlpha = c.uniforms.v[0].w;
                ++makeupLayerIdx;
            } else if (c.shader == ShaderId::kMakeupIris) {
                irisStrength = c.uniforms.v[2].x;
            }
        }
        // Blush & contour stay at full steadyFace strength (~0.40*0.72=0.288, ~0.30*0.55=0.165), not collapsed to 25%!
        EXPECT_TRUE(blushAlpha >= 0.27f);
        EXPECT_TRUE(contourAlpha >= 0.15f);
        // Contact lens still respects steadyEye (0.40 * 0.25 = 0.10)
        EXPECT_TRUE(irisStrength <= 0.12f);
        pipe.releaseGpuResources();
    }

    // 2. Internal 10Hz autoOccTex_ (from submitAuxRgbaSample under a makeup preset) MUST NOT hijack
    //    renderGuidedFilter into kGuidedWeightBlur/Diff/Coeff (causing full-cheek smoothing switches when hairMaskTex is absent!).
    {
        MockRhiDevice dev;
        LiveBeautyEngine engine(dev);
        const TextureId inTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        const TextureId atlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
        PipelineAssets assets{};
        assets.blushAtlas = atlas;
        assets.contourHighlightAtlas = atlas;
        engine.setAssets(assets);

        BeautyConfig cfg{};
        cfg.enabled = true;
        cfg.skinOnly = true;
        cfg.smoothing = 0.60f;
        cfg.whiten = 0.45f;
        cfg.blush = 0.28f;
        cfg.lipstick = 0.35f;
        engine.setConfig(cfg);

        std::array<float, math::FaceMeshFrame::kCoordCount> mp{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = 0.40f / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            mp[i * 2] = 0.50f + xyz[i * 3] * scale;
            mp[i * 2 + 1] = 1.0f - (0.50f + xyz[i * 3 + 1] * scale * aspect);
        }
        for (int i = 468; i < 478; ++i) { mp[i * 2] = 0.50f; mp[i * 2 + 1] = 0.45f; }

        EXPECT_TRUE(engine.submitFaceLandmarks(1000, mp.data(), 478, 720, 1280, 1000));
        std::vector<uint8_t> auxRgba(108 * 192 * 4, 180);
        EXPECT_TRUE(engine.submitAuxRgbaSample(auxRgba.data(), 108, 192, 0.40f));

        dev.calls.clear();
        engine.processFrame(inTex, 720, 1280, 1033, 1033);
        bool sawWeightedGuided = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kGuidedWeightBlur ||
                c.shader == ShaderId::kGuidedWeightDiff ||
                c.shader == ShaderId::kGuidedWeightCoeff) {
                sawWeightedGuided = true;
            }
        }
        EXPECT_TRUE(!sawWeightedGuided);
        engine.releaseGpuResources();
    }
}

void testV3613HairDyeStabilityDuringSlightHeadSwayAndLoadLevel1() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId inTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);

    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.hairDye = 0.80f;
    cfg.hairDyeColor = HairDyeColor::kHazeBlue;
    cfg.hairShine = 0.45f;
    engine.setConfig(cfg);

    const float aspect = 720.0f / 1280.0f;
    auto makeFaceAt = [&](float cx) {
        std::array<float, math::FaceMeshFrame::kCoordCount> mp{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float scale = 0.36f / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            mp[i * 2] = cx + xyz[i * 3] * scale;
            mp[i * 2 + 1] = 1.0f - (0.50f + xyz[i * 3 + 1] * scale * aspect);
        }
        for (int i = 468; i < 478; ++i) {
            mp[i * 2] = cx;
            mp[i * 2 + 1] = 0.45f;
        }
        return mp;
    };

    std::vector<float> logits(16 * 16 * 6, -3.5f);
    for (int i = 0; i < 16 * 16; ++i) {
        logits[static_cast<size_t>(i * 6 + (i < 128 ? 1 : 3))] = 4.0f;
    }

    // Warm up at rest so hairGate presence ramps to 1.0
    for (int f = 0; f < 6; ++f) {
        const int64_t t = 1000 + f * 33;
        auto mp = makeFaceAt(0.50f);
        EXPECT_TRUE(engine.submitFaceLandmarks(t, mp.data(), 478, 720, 1280, t));
        if (f == 0 || f == 3) {
            EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, t));
        }
        engine.processFrame(inTex, 720, 1280, t, t);
    }

    // 1. Gentle continuous head sway (+0.015 screenW/frame = 0.45 screenW/s -> forces denoiseKeep == 0.0)
    //    with multiclass refreshing every 2 frames (~66ms): kPassHairDye MUST stay 100% active on EVERY frame
    //    and push non-zero uU5.zw RGB boundary matting radius!
    int activeFrames = 0;
    bool sawMatteRadius = false;
    float cx = 0.50f;
    for (int f = 0; f < 10; ++f) {
        const int64_t t = 1200 + f * 33;
        cx += (f < 5 ? 0.014f : -0.014f);
        auto mp = makeFaceAt(cx);
        EXPECT_TRUE(engine.submitFaceLandmarks(t, mp.data(), 478, 720, 1280, t));
        if (f % 2 == 0) {
            EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, t));
        }
        dev.calls.clear();
        engine.processFrame(inTex, 720, 1280, t, t);
        if ((engine.stats().activePassesMask & kPassHairDye) != 0) {
            ++activeFrames;
        }
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kHairDye && c.uniforms.count >= 6 &&
                c.uniforms.v[5].z > 0.005f && c.uniforms.v[5].w > 0.005f) {
                sawMatteRadius = true;
            }
        }
    }
    EXPECT_TRUE(activeFrames == 10);
    EXPECT_TRUE(sawMatteRadius);

    // 2. Hard camera shake suppresses kPassHairDye, and releasing shake recovers cleanly on next frame
    engine.setCameraShaking(true);
    engine.processFrame(inTex, 720, 1280, 1550, 1550);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) == 0);
    engine.setCameraShaking(false);
    auto mpRec = makeFaceAt(cx);
    EXPECT_TRUE(engine.submitFaceLandmarks(1583, mpRec.data(), 478, 720, 1280, 1583));
    EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, 1583));
    engine.processFrame(inTex, 720, 1280, 1583, 1583);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) != 0);

    // 3. Mild thermal load (loadFadeBody == 0.0 while loadFadeSmooth == 1.0):
    //    RenderPipeline hair dye MUST remain active at full smooth fade!
    {
        MockRhiDevice pipeDev;
        RenderPipeline pipe(pipeDev);
        BeautyConfig pCfg{};
        pCfg.enabled = true;
        pCfg.hairDye = 0.80f;
        pCfg.hairDyeColor = HairDyeColor::kHazeBlue;
        FrameInputs pIn{};
        pIn.inputTexture = pipeDev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        pIn.hairMaskTex = pipeDev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);
        pIn.hairShown = 1.0f;
        math::HairMaskTuning::identity(pIn.hairAffine.data());
        pIn.width = 720;
        pIn.height = 1280;
        pIn.loadFadeBody = 0.0f;
        pIn.loadFadeSmooth = 1.0f;
        const TextureId outTex = pipe.render(pCfg, pIn);
        EXPECT_TRUE(outTex != pIn.inputTexture);
        EXPECT_TRUE((pipe.lastActivePassesMask() & kPassHairDye) != 0);
    }

    // 4. v3.6.15 Ambient scene lift wiring in uGrade.w (uU10.w): indoor lightLuma pushes >0.35, sunny 0.80 pushes 0.0
    {
        MockRhiDevice pipeDev;
        RenderPipeline pipe(pipeDev);
        BeautyConfig pCfg{};
        pCfg.enabled = true;
        pCfg.smoothing = 0.50f;
        FrameInputs pIn{};
        pIn.inputTexture = pipeDev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        pIn.width = 720;
        pIn.height = 1280;
        pIn.lightLuma = 0.25f;
        pipeDev.calls.clear();
        pipe.render(pCfg, pIn);
        float indoorAmb = -1.0f;
        for (const auto& c : pipeDev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 11) {
                indoorAmb = c.uniforms.v[10].w;
            }
        }
        EXPECT_TRUE(std::fabs(indoorAmb) <= 1e-5f);  // v3.9.4 A1: auto ambient lift disabled to protect real background

        pIn.lightLuma = 0.80f;
        pipeDev.calls.clear();
        pipe.render(pCfg, pIn);
        float sunnyAmb = -1.0f;
        for (const auto& c : pipeDev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 11) {
                sunnyAmb = c.uniforms.v[10].w;
            }
        }
        EXPECT_TRUE(std::fabs(sunnyAmb) <= 1e-5f);
    }

    engine.releaseGpuResources();
}

void testV370ImplicitRetouchOpenMouthGateAndIndoorDenoise() {
    MockRhiDevice dev;
    RenderPipeline pipe(dev);
    const TextureId inTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);

    std::array<float, math::FaceMeshFrame::kCoordCount> lm{};
    const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
    const float aspect = 720.0f / 1280.0f;
    const float scale = 0.40f / math::FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        lm[i * 2] = 0.50f + xyz[i * 3] * scale;
        lm[i * 2 + 1] = 0.50f + xyz[i * 3 + 1] * scale * aspect;
    }
    math::FaceMeshFrame::synthesizeIris(lm.data(), 468, 33, 133, 159, 145, aspect);
    math::FaceMeshFrame::synthesizeIris(lm.data(), 473, 362, 263, 386, 374, aspect);
    float meta[math::FaceMeshFrame::kMetaSize]{};
    EXPECT_TRUE(math::FaceMeshFrame::fillMeta(lm.data(), math::FaceMeshFrame::kCoordCount, aspect, meta));

    FrameInputs in{};
    in.inputTexture = inTex;
    in.width = 720;
    in.height = 1280;
    in.nowMs = 1000;
    in.hasFaceMesh = true;
    in.landmarks478x2 = lm.data();
    in.faceMeta15 = meta;
    in.hasIris = true;
    in.facePresence = 1.0f;
    in.steadyFace = 1.0f;
    in.steadyEye = 1.0f;
    in.denoiseKeep = 1.0f;
    in.loadFadeSmooth = 1.0f;
    in.loadFadeFace = 1.0f;

    // 1. Implicit natural retouch synergy activates automatically with smoothing=0.60 even when explicit retouch sliders are 0
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.smoothing = 0.60f;
    cfg.darkCircles = 0.0f;
    cfg.nasolabial = 0.0f;
    cfg.brightEyes = 0.0f;
    cfg.teethWhiten = 0.0f;

    // v3.9.4 A5: normal indoor light (0.24) must NOT burn a full-res denoise pass; extreme darkness (0.08) engages it.
    in.lightLuma = 0.24f;
    pipe.render(cfg, in);
    in.nowMs = 1033;
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassDenoise) == 0);

    in.lightLuma = 0.08f;  // v3.9.5 (tone-2): Stage 0 8-bit temporal denoise removed (zero wasted warmup/pass).
    pipe.render(cfg, in);
    in.nowMs = 1066;
    dev.calls.clear();
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassRetouch) != 0);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassDenoise) == 0);

    bool sawSkinRetouch = false;
    bool sawEyeSparkleWithTexel = false;
    bool sawTeethDraw = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kRetouchSkin && c.uniforms.count >= 1) {
            sawSkinRetouch = (std::fabs(c.uniforms.v[0].x - 0.21f) < 0.01f &&
                              std::fabs(c.uniforms.v[0].y - 0.15f) < 0.01f);
        } else if (c.shader == ShaderId::kRetouchEyeTeeth && c.uniforms.count >= 1) {
            if (c.uniforms.v[0].y < 0.5f) {
                sawEyeSparkleWithTexel = (std::fabs(c.uniforms.v[0].x - 0.24f) < 0.01f &&
                                          c.uniforms.v[0].z > 0.001f && c.uniforms.v[0].w > 0.0005f);
            } else {
                sawTeethDraw = true;
            }
        }
    }
    EXPECT_TRUE(sawSkinRetouch);
    EXPECT_TRUE(sawEyeSparkleWithTexel);
    EXPECT_TRUE(!sawTeethDraw);

    // 2. Bright daylight (lightLuma=0.45) and camera shake (denoiseKeep=0.0) both keep kPassDenoise == 0
    in.lightLuma = 0.45f;
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassDenoise) == 0);
    in.lightLuma = 0.24f;
    in.denoiseKeep = 0.0f;
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassDenoise) == 0);
    in.denoiseKeep = 1.0f;

    // 3. Open-mouth teeth whitening gate: closed mouth suppresses teeth draw, open mouth activates it
    cfg.teethWhiten = 0.85f;
    // Closed mouth: lm[13] == lm[14]
    const float midLipY = 0.5f * (lm[13 * 2 + 1] + lm[14 * 2 + 1]);
    lm[13 * 2 + 1] = midLipY;
    lm[14 * 2 + 1] = midLipY;
    dev.calls.clear();
    pipe.render(cfg, in);
    bool sawTeethClosed = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kRetouchEyeTeeth && c.uniforms.count >= 1 && c.uniforms.v[0].y > 0.5f) {
            sawTeethClosed = true;
        }
    }
    EXPECT_TRUE(!sawTeethClosed);

    // Open mouth: gap/width = 0.18 (> 0.135 -> mouthOpenGate == 1.0)
    const float mw = std::hypot(lm[291 * 2] - lm[61 * 2], (lm[291 * 2 + 1] - lm[61 * 2 + 1]) / aspect);
    lm[13 * 2 + 1] = midLipY + 0.09f * mw * aspect;
    lm[14 * 2 + 1] = midLipY - 0.09f * mw * aspect;
    dev.calls.clear();
    pipe.render(cfg, in);
    float teethStrength = 0.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kRetouchEyeTeeth && c.uniforms.count >= 1 && c.uniforms.v[0].y > 0.5f) {
            teethStrength = c.uniforms.v[0].x;
        }
    }
    EXPECT_TRUE(teethStrength >= 0.84f);

    // 4. smoothing=0 + explicit retouch=0 -> kPassRetouch == 0
    cfg.smoothing = 0.0f;
    cfg.teethWhiten = 0.0f;
    cfg.whiten = 0.30f;
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassRetouch) == 0);

    pipe.releaseGpuResources();
}

void testV380AtmospherePostPipelineOpenMouthLipAndOcclusionMute() {
    MockRhiDevice dev;
    RenderPipeline pipe(dev);
    const TextureId inTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId atlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, nullptr);
    const TextureId lutTex = dev.createTexture2D(512, 512, TextureFormat::kRgba8, nullptr);

    PipelineAssets assets{};
    assets.blushAtlas = atlas;
    assets.contourHighlightAtlas = atlas;
    assets.eyeMakeupAtlas = atlas;
    assets.eyebrowAtlas = atlas;
    assets.lutTex = lutTex;
    pipe.setAssets(assets);

    std::array<float, math::FaceMeshFrame::kCoordCount> lm{};
    const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
    const float aspect = 720.0f / 1280.0f;
    const float scale = 0.40f / math::FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        lm[i * 2] = 0.50f + xyz[i * 3] * scale;
        lm[i * 2 + 1] = 0.50f + xyz[i * 3 + 1] * scale * aspect;
    }
    math::FaceMeshFrame::synthesizeIris(lm.data(), 468, 33, 133, 159, 145, aspect);
    math::FaceMeshFrame::synthesizeIris(lm.data(), 473, 362, 263, 386, 374, aspect);
    float meta[math::FaceMeshFrame::kMetaSize]{};
    EXPECT_TRUE(math::FaceMeshFrame::fillMeta(lm.data(), math::FaceMeshFrame::kCoordCount, aspect, meta));

    FrameInputs in{};
    in.inputTexture = inTex;
    in.width = 720;
    in.height = 1280;
    in.nowMs = 1000;
    in.hasFaceMesh = true;
    in.landmarks478x2 = lm.data();
    in.faceMeta15 = meta;
    in.facePresence = 1.0f;
    in.steadyFace = 1.0f;
    in.steadyEye = 1.0f;
    in.eyeMakeupKeep = 1.0f;
    in.lowerMakeupKeep = 1.0f;
    in.loadFadeBody = 1.0f;
    in.loadFadeFace = 1.0f;
    in.loadFadeSmooth = 1.0f;

    // 1. Natural mode (no LUT, bloom=0, filmGrain=0, colorTemp=0): kPassAtmosphere == 0 (zero cost!)
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.smoothing = 0.50f;
    dev.calls.clear();
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassAtmosphere) == 0);
    EXPECT_TRUE(!dev.feedbackLoopDetected);

    // 2. Cinema LUT alone no longer forces invisible auto-bloom/grain (v3.9.4 D4); explicit bloom + filmGrain engages multi-scale bloom
    cfg.lutUrl = "asset://luts/night_vibe.png";
    cfg.lutIntensity = 0.80f;
    dev.calls.clear();
    pipe.render(cfg, in);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassAtmosphere) == 0);

    cfg.bloom = 0.30f;
    cfg.filmGrain = 0.18f;
    dev.calls.clear();
    const TextureId cinemaOut = pipe.render(cfg, in);
    EXPECT_TRUE(cinemaOut != inTex);
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassAtmosphere) != 0);
    bool sawBloomExtract = false;
    bool sawAtmosphere = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kPostBloomExtract) sawBloomExtract = true;
        if (c.shader == ShaderId::kPostAtmosphere && c.uniforms.count >= 1) {
            sawAtmosphere = (c.uniforms.v[0].x > 0.20f && c.uniforms.v[0].y > 0.08f);
        }
    }
    EXPECT_TRUE(sawBloomExtract);
    EXPECT_TRUE(sawAtmosphere);

    // 3. Lite tier throttles multi-pass bloom extract off while preserving lightweight grain / CAT02 WB
    in.tierCaps.resolvedTier = math::PerformanceTier::kLite;
    cfg.colorTemp = 0.35f;
    dev.calls.clear();
    pipe.render(cfg, in);
    bool sawLiteExtract = false;
    bool sawLiteCat02 = false;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kPostBloomExtract) sawLiteExtract = true;
        if (c.shader == ShaderId::kPostAtmosphere && c.uniforms.count >= 1) {
            sawLiteCat02 = (c.uniforms.v[0].x == 0.0f && c.uniforms.v[0].w > 0.5f);
        } else if (c.shader == ShaderId::kCompositePro && c.uniforms.count == 13) {
            // v3.9.4 fused zero-extra-pass full 3x3 CAT02 WB + filmic grain inside kCompositePro (uU4/uU11/uU12)
            sawLiteCat02 = (c.uniforms.v[11].w > 0.5f);
        }
    }
    EXPECT_TRUE(!sawLiteExtract);
    EXPECT_TRUE(sawLiteCat02);
    in.tierCaps.resolvedTier = math::PerformanceTier::kStandard;
    cfg.lutUrl.clear();
    cfg.bloom = 0.0f;
    cfg.filmGrain = 0.0f;
    cfg.colorTemp = 0.0f;

    // 4. Open-mouth geometric cutout uniforms wired to kMakeupLip (closed -> 0, open -> ~1)
    cfg.lipstick = 0.65f;
    const float midLipY = 0.5f * (lm[13 * 2 + 1] + lm[14 * 2 + 1]);
    lm[13 * 2 + 1] = midLipY;
    lm[14 * 2 + 1] = midLipY;
    dev.calls.clear();
    pipe.render(cfg, in);
    float openUniformClosed = -1.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMakeupLip && c.uniforms.count >= 5) {
            openUniformClosed = c.uniforms.v[4].x;
        }
    }
    EXPECT_TRUE(openUniformClosed >= 0.0f && openUniformClosed < 0.01f);

    const float mw = std::hypot(lm[291 * 2] - lm[61 * 2], (lm[291 * 2 + 1] - lm[61 * 2 + 1]) / aspect);
    lm[13 * 2 + 1] = midLipY + 0.09f * mw * aspect;
    lm[14 * 2 + 1] = midLipY - 0.09f * mw * aspect;
    dev.calls.clear();
    pipe.render(cfg, in);
    float openUniformOpen = 0.0f;
    float mouthRy = 0.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMakeupLip && c.uniforms.count >= 5) {
            mouthRy = c.uniforms.v[3].w;
            openUniformOpen = c.uniforms.v[4].x;
        }
    }
    EXPECT_TRUE(openUniformOpen >= 0.95f && mouthRy > 0.005f);

    // 5. Heavy lower-face occlusion mute (lowerMakeupKeep = 0): mutes blush/contour/lipstick/teeth
    //    while keeping eyeshadow & eyeliner 100% active!
    cfg.blush = 0.50f;
    cfg.contour = 0.45f;
    cfg.highlight = 0.40f;
    cfg.eyeshadow = 0.55f;
    cfg.eyeliner = 0.50f;
    cfg.teethWhiten = 0.80f;
    in.lowerMakeupKeep = 0.0f;
    dev.calls.clear();
    pipe.render(cfg, in);
    bool sawLipMuted = false;
    bool sawTeethMuted = false;
    int makeupLayerDraws = 0;  // Only eyeshadow (2 draws) + eyeliner (1 draw) = 3 draws (blush/contour/highlight muted!)
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kMakeupLip) sawLipMuted = true;
        if (c.shader == ShaderId::kRetouchEyeTeeth && c.uniforms.count >= 1 && c.uniforms.v[0].y > 0.5f) {
            sawTeethMuted = true;
        }
        if (c.shader == ShaderId::kMakeupLayer) ++makeupLayerDraws;
    }
    EXPECT_TRUE(!sawLipMuted);
    EXPECT_TRUE(!sawTeethMuted);
    EXPECT_TRUE(makeupLayerDraws == 3);

    pipe.releaseGpuResources();
}

void testV390PassFusionAutoWhiteBalanceFastBloomAndDualFaceOcclusion() {
    MockRhiDevice dev;
    RenderPipeline pipe(dev);
    PipelineAssets assets{};
    assets.blushAtlas = dev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
    assets.eyebrowAtlas = dev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
    assets.eyeMakeupAtlas = dev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
    assets.contourHighlightAtlas = dev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
    assets.irisAtlas = dev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
    assets.whiteAlphaFallbackTex = dev.createTexture2D(1, 1, TextureFormat::kRgba8, nullptr);
    pipe.setAssets(assets);

    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    FrameInputs in{};
    in.inputTexture = camTex;
    in.width = 720;
    in.height = 1280;
    in.nowMs = 420;
    in.ambientRgb = {1.24f, 1.00f, 0.76f};  // Warm tungsten lighting
    in.tierCaps = math::DeviceTierPolicy::capabilitiesFor(math::PerformanceTier::kStandard);

    // 1. Pass Fusion: Auto CAT02 WB + filmGrain with no Bloom and no post-composite layers fuses directly
    //    into kCompositePro -> ZERO kPostAtmosphere draw calls (saves 1 full-res render pass)!
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.smoothing = 0.45f;
    cfg.autoWhiteBalance = 0.85f;
    cfg.filmGrain = 0.10f;
    dev.calls.clear();
    pipe.render(cfg, in);
    int compositeDraws = 0, postAtmDraws = 0;
    float fusedCat02Flag = 0.0f, fusedGrainVal = 0.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kCompositePro) {
            ++compositeDraws;
            if (c.uniforms.count == 13) {
                fusedCat02Flag = (c.uniforms.v[11].w > 0.5f) ? 1.0f : 0.0f;
                fusedGrainVal = c.uniforms.v[5].x - std::floor(c.uniforms.v[5].x);
            }
        }
        if (c.shader == ShaderId::kPostAtmosphere) ++postAtmDraws;
    }
    EXPECT_TRUE(compositeDraws == 1);
    EXPECT_TRUE(postAtmDraws == 0);
    EXPECT_TRUE(fusedCat02Flag > 0.5f && fusedGrainVal > 0.05f);
    EXPECT_TRUE((pipe.lastActivePassesMask() & kPassAtmosphere) != 0u);

    // 2. Fast-path single-level Bloom on Standard tier skips 1/9-res kDownsample after kPostBloomExtract (saves 3 passes!)
    cfg.bloom = 0.35f;
    dev.calls.clear();
    pipe.render(cfg, in);
    int bloomExtractStd = 0, bloomDownsampleStd = 0, postAtmStd = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kPostBloomExtract) ++bloomExtractStd;
        if (bloomExtractStd > 0 && c.shader == ShaderId::kDownsample) ++bloomDownsampleStd;
        if (c.shader == ShaderId::kPostAtmosphere) ++postAtmStd;
    }
    EXPECT_TRUE(bloomExtractStd == 1 && postAtmStd == 1);
    EXPECT_TRUE(bloomDownsampleStd == 0);

    // Ultra tier + strong bloom enables the full 2-level wide halo pyramid
    in.tierCaps = math::DeviceTierPolicy::capabilitiesFor(math::PerformanceTier::kUltra);
    dev.calls.clear();
    pipe.render(cfg, in);
    int bloomExtractUltra = 0, bloomDownsampleUltra = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kPostBloomExtract) ++bloomExtractUltra;
        if (bloomExtractUltra > 0 && c.shader == ShaderId::kDownsample) ++bloomDownsampleUltra;
    }
    EXPECT_TRUE(bloomDownsampleUltra == 1);

    // 3. Dual-face independent lower-face occlusion via LiveBeautyEngine::submitMulticlass6Planar:
    //    Primary face (left half) unoccluded -> lipstick drawn; Secondary face (right half) masked -> lipstick muted, eye makeup kept!
    {
        MockRhiDevice engDev;
        LiveBeautyEngine engine(engDev);
        BeautyConfig ecfg{};
        ecfg.enabled = true;
        ecfg.maxFaces = 2;
        ecfg.performanceTier = math::PerformanceTier::kUltra;
        ecfg.lipstick = 0.65f;
        ecfg.eyeshadow = 0.50f;
        engine.setConfig(ecfg);
        PipelineAssets engAssets{};
        engAssets.blushAtlas = engDev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
        engAssets.eyebrowAtlas = engDev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
        engAssets.eyeMakeupAtlas = engDev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
        engAssets.contourHighlightAtlas = engDev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
        engAssets.irisAtlas = engDev.createTexture2D(16, 16, TextureFormat::kRgba8, nullptr);
        engAssets.whiteAlphaFallbackTex = engDev.createTexture2D(1, 1, TextureFormat::kRgba8, nullptr);
        engine.setAssets(engAssets);

        auto makeCanonFace = [](float cx, float fw) {
            std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
            const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
            const float aspect = 720.0f / 1280.0f;
            const float scale = fw / math::FaceMeshFilter::canonicalFaceWidth();
            for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
                const float mx = cx + xyz[i * 3] * scale;
                const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
                mpXy[i * 2] = mx;
                mpXy[i * 2 + 1] = 1.0f - (my * aspect);
            }
            for (int i = 468; i < 478; ++i) {
                mpXy[i * 2] = cx;
                mpXy[i * 2 + 1] = 0.45f;
            }
            return mpXy;
        };
        const auto fLeft = makeCanonFace(0.26f, 0.26f);
        const auto fRight = makeCanonFace(0.74f, 0.24f);
        math::MultiFaceTracker::RawFaceInput rawFaces[2] = {
            {fLeft.data(), 478},
            {fRight.data(), 478},
        };
        const TextureId engCam = engDev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
        int64_t lastTs = 5000;
        for (int k = 0; k < 8; ++k) {
            lastTs = 5000 + k * 33;
            EXPECT_TRUE(engine.submitMultiFaceLandmarks(lastTs, rawFaces, 2, 720, 1280, lastTs) == 2);
            engine.processFrame(engCam, 720, 1280, lastTs, lastTs);
        }

        // Planar 6-class confidence maps: left half bare skin (p3=1), right half lower-face occluded by mask (p5=1)
        constexpr int kW = 32, kH = 32;
        std::vector<float> p0(kW * kH, 0.0f), p1(kW * kH, 0.0f), p2(kW * kH, 0.0f);
        std::vector<float> p3(kW * kH, 0.0f), p4(kW * kH, 0.0f), p5(kW * kH, 0.0f);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int idx = y * kW + x;
                if (x >= kW / 2 && y >= kH / 2) {
                    p5[idx] = 0.98f; p0[idx] = 0.02f;  // Right face lower-half masked
                } else {
                    p3[idx] = 0.98f; p0[idx] = 0.02f;  // Bare skin
                }
            }
        }
        const float* planes[6] = {p0.data(), p1.data(), p2.data(), p3.data(), p4.data(), p5.data()};
        EXPECT_TRUE(engine.submitMulticlass6Planar(planes, kW, kH, lastTs));

        engDev.calls.clear();
        engine.processFrame(engCam, 720, 1280, lastTs + 33, lastTs + 33);
        int lipDraws = 0;
        int eyeLayerDraws = 0;
        for (const auto& c : engDev.calls) {
            if (c.shader == ShaderId::kMakeupLip) ++lipDraws;
            if (c.shader == ShaderId::kMakeupLayer) ++eyeLayerDraws;
        }
        // Only primary (left) face gets lipstick (1 draw), while BOTH faces get eyeshadow (2 layers * 2 faces = 4 draws)!
        EXPECT_TRUE(lipDraws == 1);
        EXPECT_TRUE(eyeLayerDraws == 4);
        engine.releaseGpuResources();
    }

    pipe.releaseGpuResources();
}

void testV393OcclusionDeferredAfterCompositeBeforeMakeup() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId atlasTex = dev.createTexture2D(32, 32, TextureFormat::kRgba8, nullptr);
    PipelineAssets assets{};
    assets.blushAtlas = atlasTex;
    assets.contourHighlightAtlas = atlasTex;
    assets.whiteAlphaFallbackTex = dev.createTexture2D(1, 1, TextureFormat::kRgba8, nullptr);
    engine.setAssets(assets);

    std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
    const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
    const float aspect = 720.0f / 1280.0f;
    const float scale = 0.38f / math::FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
        mpXy[i * 2] = 0.50f + xyz[i * 3] * scale;
        mpXy[i * 2 + 1] = 1.0f - ((0.50f / aspect) + xyz[i * 3 + 1] * scale) * aspect;
    }
    for (int i = 468; i < 478; ++i) { mpXy[i * 2] = 0.50f; mpXy[i * 2 + 1] = 0.45f; }

    // 1. Makeup + 6-class semantic occlusion (no hairDye):
    //    kHairAlpha MUST run strictly AFTER kCompositePro and BEFORE kMakeupLayer so occlusion NEVER
    //    darkens skin whitening/smoothing in kCompositePro while still blocking UV makeup!
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.performanceTier = math::PerformanceTier::kUltra;
    cfg.smoothing = 0.55f;
    cfg.whiten = 0.50f;
    cfg.blush = 0.35f;
    cfg.lipstick = 0.40f;
    engine.setConfig(cfg);

    std::vector<float> logits(16 * 16 * 6, -4.0f);
    for (int i = 0; i < 16 * 16; ++i) {
        logits[static_cast<size_t>(i * 6 + (i < 64 ? 1 : 3))] = 5.0f;
    }
    for (int f = 0; f < 5; ++f) {
        const int64_t ts = 7000 + f * 33;
        engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts);
        engine.processFrame(camTex, 720, 1280, ts, ts);
    }
    EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, 7165));
    engine.submitFaceLandmarks(7165, mpXy.data(), 478, 720, 1280, 7165);
    dev.calls.clear();
    engine.processFrame(camTex, 720, 1280, 7165, 7165);
    EXPECT_TRUE(!dev.feedbackLoopDetected);

    int idxComposite = -1, idxHairAlpha = -1, idxMakeup = -1;
    int hairAlphaCount = 0;
    float hairAlphaOccWeight = 0.0f;
    for (int i = 0; i < static_cast<int>(dev.calls.size()); ++i) {
        const auto& c = dev.calls[i];
        if (c.shader == ShaderId::kCompositePro) idxComposite = i;
        if (c.shader == ShaderId::kHairAlpha) {
            if (idxHairAlpha < 0) idxHairAlpha = i;
            ++hairAlphaCount;
            if (c.uniforms.count >= 2) hairAlphaOccWeight = c.uniforms.v[1].w;
        }
        if ((c.shader == ShaderId::kMakeupLayer || c.shader == ShaderId::kMakeupLip) && idxMakeup < 0) {
            idxMakeup = i;
        }
    }
    EXPECT_TRUE(idxComposite >= 0 && idxHairAlpha >= 0 && idxMakeup >= 0);
    EXPECT_TRUE(idxHairAlpha > idxComposite && idxHairAlpha < idxMakeup);
    EXPECT_TRUE(hairAlphaOccWeight > 0.5f);

    // 2. Hair dye + makeup + 6-class logits:
    //    Stage 4 runs kHairAlpha with hairMask ONLY (uU1.w == 0) before kCompositePro;
    //    Stage 6c runs kHairAlpha with hairMask + occMask (uU1.w > 0) after kCompositePro and before kMakeupLayer!
    cfg.hairDye = 0.70f;
    engine.setConfig(cfg);
    EXPECT_TRUE(engine.submitMulticlass6Logits(logits.data(), 16, 16, 7198));
    engine.submitFaceLandmarks(7198, mpXy.data(), 478, 720, 1280, 7198);
    dev.calls.clear();
    engine.processFrame(camTex, 720, 1280, 7198, 7198);
    EXPECT_TRUE(!dev.feedbackLoopDetected);

    int idxPreHairAlpha = -1, idxPostHairAlpha = -1;
    float preOccW = -1.0f, postOccW = -1.0f;
    idxComposite = -1;
    idxMakeup = -1;
    for (int i = 0; i < static_cast<int>(dev.calls.size()); ++i) {
        const auto& c = dev.calls[i];
        if (c.shader == ShaderId::kCompositePro) idxComposite = i;
        if ((c.shader == ShaderId::kMakeupLayer || c.shader == ShaderId::kMakeupLip) && idxMakeup < 0) idxMakeup = i;
        if (c.shader == ShaderId::kHairAlpha && c.uniforms.count >= 2) {
            if (idxComposite < 0) {
                idxPreHairAlpha = i;
                preOccW = c.uniforms.v[1].w;
            } else {
                idxPostHairAlpha = i;
                postOccW = c.uniforms.v[1].w;
            }
        }
    }
    EXPECT_TRUE(idxPreHairAlpha >= 0 && idxPreHairAlpha < idxComposite);
    EXPECT_TRUE(std::fabs(preOccW) < 1e-4f);
    EXPECT_TRUE(idxPostHairAlpha > idxComposite && idxPostHairAlpha < idxMakeup);
    EXPECT_TRUE(postOccW > 0.5f);

    engine.releaseGpuResources();
}

void testV394AuditFixesPipelineAndGpuPerf() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    engine.setHardwareProfile("Adreno (TM) 750", 8, 12288);
    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId extOutRt = dev.createRenderTarget(720, 1280, TextureFormat::kRgba8);

    // 1. Default Natural look: zero denoise at indoor luma, zero auto ambient lift, zero auto bloom,
    //    5 FP16 RTs (rtBody_/rtWarp_ RGBA8), zero redundant kPassThrough in rtWarp_, and zero-copy outputRt.
    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.smoothing = 0.55f;
    cfg.whiten = 0.30f;
    cfg.rosy = 0.20f;
    cfg.faceSlim = 0.25f;
    cfg.eyeEnlarge = 0.15f;
    engine.setConfig(cfg);

    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    makeMediaPipeFace(720.0f / 1280.0f, mpXy);
    for (int f = 0; f < 4; ++f) {
        const int64_t ts = 1000 + f * 33;
        engine.submitFaceLandmarks(ts, mpXy.data(), 478, 720, 1280, ts);
        dev.calls.clear();
        engine.processFrame(camTex, 720, 1280, ts, ts, kInvalidTexture, 0.30f, 1.0f,
                            kInvalidTexture, kTimeInvalid, extOutRt);
    }
    EXPECT_TRUE(!dev.feedbackLoopDetected);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassDenoise) == 0);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassAtmosphere) == 0);
    int compositeToExtRt = 0;
    int passThroughInWarp = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kCompositePro) {
            if (c.targetRt == extOutRt) ++compositeToExtRt;
            EXPECT_TRUE(std::fabs(c.uniforms.v[10].w) < 1e-5f);  // ambientLift == 0
        }
        if (c.shader == ShaderId::kPassThrough) ++passThroughInWarp;
    }
    EXPECT_TRUE(compositeToExtRt == 1);
    EXPECT_TRUE(passThroughInWarp == 0);

    // 2. D5: Host load level switching 0 -> 2 -> 0 must NOT recreate any render targets!
    const size_t rtCountBefore = dev.createdRtFormats.size();
    engine.setHostLoadLevel(2);
    EXPECT_TRUE(engine.loadLevel() >= 2);
    engine.processFrame(camTex, 720, 1280, 1200, 1200);
    engine.setHostLoadLevel(0);
    engine.processFrame(camTex, 720, 1280, 1233, 1233);
    EXPECT_TRUE(dev.createdRtFormats.size() == rtCountBefore);

    // 3. C3 (shape-4): Stale body pose (>250ms) and submitBodyMiss fade out body warp cleanly
    cfg.waistSlim = 0.50f;
    engine.setConfig(cfg);
    math::PoseLandmark pose33[33]{};
    pose33[11] = {0.35f, 0.35f, 0.95f};
    pose33[12] = {0.65f, 0.35f, 0.95f};
    pose33[23] = {0.40f, 0.65f, 0.95f};
    pose33[24] = {0.60f, 0.65f, 0.95f};
    EXPECT_TRUE(engine.submitBodyPose(2000, pose33, 33, 720, 1280));
    engine.processFrame(camTex, 720, 1280, 2000, 2000);
    engine.processFrame(camTex, 720, 1280, 2033, 2033);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBody) != 0);
    // Advance > 600ms without submitting any new body pose -> stale timeout kicks in and fades out kPassBody
    for (int f = 1; f <= 20; ++f) {
        engine.processFrame(camTex, 720, 1280, 2033 + f * 40, 2033 + f * 40);
    }
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBody) == 0);

    // 4. C6 (seg-4 / seg-v3): Background blur stays active under loadLevel=2 (privacy!) and pass 0 uses step=1.0
    cfg.waistSlim = 0.0f;
    cfg.backgroundBlur = 0.70f;
    engine.setConfig(cfg);
    const TextureId personTex = dev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);
    engine.setHostLoadLevel(0);
    dev.calls.clear();
    engine.processFrame(camTex, 720, 1280, 3500, 3500, personTex);
    int bgBlurCallsNormal = 0;
    float firstStepNormal = -1.0f;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBgBlur) {
            if (bgBlurCallsNormal == 0) firstStepNormal = c.uniforms.v[0].x * 180.0f;
            ++bgBlurCallsNormal;
        }
    }
    EXPECT_TRUE(bgBlurCallsNormal == 6);  // 3 H+V passes at bgBlur=0.70 > 0.55
    EXPECT_TRUE(std::fabs(firstStepNormal - 1.0f) < 0.02f);

    engine.setHostLoadLevel(2);
    for (int f = 1; f <= 20; ++f) {
        dev.calls.clear();
        engine.processFrame(camTex, 720, 1280, 3500 + f * 40, 3500 + f * 40, personTex);
    }
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBg) != 0);  // Privacy preserved!
    int bgBlurCallsLoad = 0;
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBgBlur) ++bgBlurCallsLoad;
    }
    EXPECT_TRUE(bgBlurCallsLoad == 4);  // v3.9.5 (segmentation-n1): Privacy-safe 2 H+V passes (sigma 9.6px) under overload

    engine.releaseGpuResources();
}

void testV395ReverificationFixesPipeline() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId personTex = dev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);
    const TextureId hairTex = dev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);
    const TextureId irisAtlas = dev.createTexture2D(256, 256, TextureFormat::kRgba8, nullptr);
    PipelineAssets assets{};
    assets.irisAtlas = irisAtlas;
    engine.setAssets(assets);

    auto makeFaceMp = [](float cx, float fw) {
        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = fw / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            mpXy[i * 2] = cx + xyz[i * 3] * scale;
            mpXy[i * 2 + 1] = 1.0f - (((0.50f / aspect) + xyz[i * 3 + 1] * scale) * aspect);
        }
        for (int i = 468; i < 478; ++i) {
            mpXy[i * 2] = cx + ((i >= 473) ? 0.06f : -0.06f);
            mpXy[i * 2 + 1] = 0.44f;
        }
        return mpXy;
    };

    // 1. skin-n1/tone-n1/tone-m1: Brief face miss (<1200ms) holds prior steady (zero beauty breathing);
    //    extended absence (>1200ms, 90 frames ~3s) decays bodyPriorMute smoothly (delta <= 0.11/frame)
    //    to >= 0.95 and NEVER snaps portraitPrior back to 1.0 when timeline clears!
    {
        auto cfg = BeautyConfig::naturalStudio();
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.30f);
        for (int f = 0; f < 12; ++f) {
            const int64_t ts = 1000 + f * 33;
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        float prevMute = -1.0f;
        float maxDelta = 0.0f;
        float lastMute = 0.0f;
        float muteAt600ms = 1.0f;
        for (int f = 0; f < 90; ++f) {
            const int64_t ts = 1400 + f * 33;
            dev.calls.clear();
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 7) {
                    const float mute = c.uniforms.v[5].y;
                    if (prevMute >= 0.0f) {
                        maxDelta = std::max(maxDelta, std::fabs(mute - prevMute));
                    }
                    prevMute = mute;
                    lastMute = mute;
                    if (f == 18) muteAt600ms = mute;  // ~600ms brief miss
                    // Whenever oval clears (uU6.z == 0), bodyPriorMute MUST be 1.0 (prior=0, never 1.0!)
                    if (c.uniforms.v[6].z <= 0.001f) {
                        EXPECT_TRUE(mute >= 0.99f);
                    }
                }
            }
        }
        EXPECT_TRUE(muteAt600ms <= 0.05f);
        EXPECT_TRUE(maxDelta <= 0.11f);
        EXPECT_TRUE(lastMute >= 0.95f);
    }

    // 2. skin-n4/tone-n2: Dual-face secondary portrait prior packed into uU8/uU9 when polySkinMaskReady
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.maxFaces = 2;
        engine.setConfig(cfg);
        const auto mpL = makeFaceMp(0.28f, 0.26f);
        const auto mpR = makeFaceMp(0.72f, 0.26f);
        math::MultiFaceTracker::RawFaceInput pair[2] = {{mpL.data(), 478}, {mpR.data(), 478}};
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 3000 + f * 33;
            dev.calls.clear();
            engine.submitMultiFaceLandmarks(ts, pair, 2, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        bool sawSecPrior = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 10) {
                if (c.uniforms.v[0].w >= 3.5f && c.uniforms.v[8].z > 0.05f && c.uniforms.v[9].z > 0.5f) {
                    sawSecPrior = true;
                }
            }
        }
        EXPECT_TRUE(sawSecPrior);
    }

    // 3. makeup-n1: Perceptual calibration of contactLens (*0.65) & catchlight (*0.55)
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.maxFaces = 1;
        cfg.contactLens = 0.35f;
        cfg.catchlight = 0.40f;
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.32f);
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 4000 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        bool sawCalibratedIris = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kMakeupIris && c.uniforms.count >= 3) {
                const float lensS = c.uniforms.v[2].x;
                const float clS = c.uniforms.v[2].y;
                if (lensS > 0.18f && lensS < 0.25f && clS > 0.18f && clS < 0.25f) {
                    sawCalibratedIris = true;
                }
            }
        }
        EXPECT_TRUE(sawCalibratedIris);
    }

    // 4. gpuperf-n1 + segmentation-n1 + seg-3: continuous 3-pass bokeh across 0.54/0.56 & blurOnlyHair skips kHairAlpha
    {
        auto cfg = BeautyConfig::naturalStudio();
        cfg.contactLens = 0.0f;
        cfg.catchlight = 0.0f;
        cfg.hairDye = 0.0f;
        cfg.hairShine = 0.0f;
        cfg.backgroundBlur = 0.54f;
        engine.setConfig(cfg);
        const auto mpBlur = makeFaceMp(0.50f, 0.30f);
        for (int f = 0; f < 6; ++f) {
            const int64_t ts = 5000 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mpBlur.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts, personTex, 0.30f, 1.0f, hairTex, ts);
        }
        int blur54 = 0;
        float step2_54 = 0.0f;
        bool sawHairMerge = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kBgBlur) {
                if (blur54 == 4) step2_54 = c.uniforms.v[0].x * 180.0f;
                ++blur54;
            }
            if (c.shader == ShaderId::kBgHairMerge) sawHairMerge = true;
        }
        EXPECT_TRUE(blur54 == 6);
        EXPECT_TRUE(sawHairMerge);
        EXPECT_TRUE((engine.stats().activePassesMask & kPassHairAlpha) == 0);  // seg-3: blur-only skips rtSkinMask_.a

        cfg.backgroundBlur = 0.56f;
        engine.setConfig(cfg);
        dev.calls.clear();
        engine.submitFaceLandmarks(5200, mpBlur.data(), 478, 720, 1280, 5200);
        engine.processFrame(camTex, 720, 1280, 5200, 5200, personTex, 0.30f, 1.0f, hairTex, 5200);
        int blur56 = 0;
        float step2_56 = 0.0f;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kBgBlur) {
                if (blur56 == 4) step2_56 = c.uniforms.v[0].x * 180.0f;
                ++blur56;
            }
        }
        EXPECT_TRUE(blur56 == 6);
        EXPECT_TRUE(std::fabs(step2_56 - step2_54) < 0.03f);  // Continuous across 0.55!
    }

    // 5. gpuperf-n4: lastWroteOutputRt() accurately distinguishes rendered outputRt vs passthrough
    {
        const RenderTargetId extRt = dev.createRenderTarget(720, 1280, TextureFormat::kRgba8);
        auto cfg = BeautyConfig::naturalStudio();
        engine.setConfig(cfg);
        engine.processFrame(camTex, 720, 1280, 6000, 6000, kInvalidTexture, 0.30f, 1.0f, kInvalidTexture, kTimeInvalid, extRt);
        EXPECT_TRUE(engine.lastWroteOutputRt());

        cfg.compareOriginal = true;
        engine.setConfig(cfg);
        engine.processFrame(camTex, 720, 1280, 6033, 6033, kInvalidTexture, 0.30f, 1.0f, kInvalidTexture, kTimeInvalid, extRt);
        EXPECT_TRUE(!engine.lastWroteOutputRt());
        dev.destroyRenderTarget(extRt);
    }

    engine.releaseGpuResources();
}

void testV396AuditFixesPipeline() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId personTex = dev.createTexture2D(256, 256, TextureFormat::kR8, nullptr);
    const TextureId irisTex = dev.createTexture2D(512, 512, TextureFormat::kRgba8, nullptr);
    PipelineAssets assets{};
    assets.irisAtlas = irisTex;
    engine.setAssets(assets);
    engine.setHardwareProfile("Adreno (TM) 740", 8, 12288);

    auto makeFaceMp = [](float cx, float fw) {
        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = fw / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            mpXy[i * 2] = cx + xyz[i * 3] * scale;
            mpXy[i * 2 + 1] = 1.0f - (((0.50f / aspect) + xyz[i * 3 + 1] * scale) * aspect);
        }
        for (int i = 468; i < 478; ++i) {
            mpXy[i * 2] = cx + ((i >= 473) ? 0.06f : -0.06f);
            mpXy[i * 2 + 1] = 0.44f;
        }
        return mpXy;
    };

    // 1. tone-m2 / skin-d1 (P1): L2 overload pauses face warp/makeup while keeping skin smoothing & whitening intact!
    {
        auto cfg = BeautyConfig::naturalStudio();
        cfg.faceSlim = 0.5f;
        engine.setConfig(cfg);
        engine.setHostLoadLevel(2);
        const auto mp = makeFaceMp(0.50f, 0.30f);
        float lastMute = 1.0f;
        float lastOvalZ = 0.0f;
        float lastFramePres = 0.0f;
        uint32_t lastMask = 0;
        for (int f = 0; f < 65; ++f) {
            const int64_t ts = 1000 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            lastMask = engine.stats().activePassesMask;
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 7) {
                    lastFramePres = c.uniforms.v[0].w;
                    lastMute = c.uniforms.v[5].y;
                    lastOvalZ = c.uniforms.v[6].z;
                }
            }
        }
        EXPECT_TRUE((lastMask & kPassFaceWarp) == 0);  // L2 pauses face reshape
        EXPECT_TRUE((lastMask & kPassSkinMask) != 0);  // L2 keeps polygon skin mask active
        EXPECT_TRUE(lastMute <= 0.05f);                // L2 keeps body/neck prior active
        EXPECT_TRUE(lastOvalZ > 0.05f);
        EXPECT_TRUE(lastFramePres >= 4.90f);           // L2 keeps face smoothing & whitening active

        // L3 fades smoothing & body prior cleanly down to ~0
        engine.setHostLoadLevel(3);
        for (int f = 0; f < 80; ++f) {
            const int64_t ts = 3200 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 7) {
                    lastMute = c.uniforms.v[5].y;
                }
            }
        }
        EXPECT_TRUE(lastMute >= 0.90f);
        engine.setHostLoadLevel(0);
    }

    // 2. tone-m1 (P1): Long >2s absence decays smoothly to 0; reacquire ramps faceM & bodyPrior TOGETHER (no jaw seam/snap)
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.30f);
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 6000 + f * 33;
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        // Leave frame for ~2.7s (82 frames)
        for (int f = 0; f < 82; ++f) {
            const int64_t ts = 6350 + f * 33;
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        // Reacquire: faceM (uU0.w - 4.0) and bodyPrior (1 - uU5.y) must ramp smoothly together!
        float prevFacePres = 0.0f;
        float maxFaceStep = 0.0f;
        float maxJawMismatch = 0.0f;
        float finalFacePres = 0.0f;
        for (int f = 0; f < 25; ++f) {
            const int64_t ts = 9100 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 6) {
                    const float fp = (c.uniforms.v[0].w >= 3.5f) ? (c.uniforms.v[0].w - 4.0f) : c.uniforms.v[0].w;
                    const float bp = 1.0f - c.uniforms.v[5].y;
                    maxFaceStep = std::max(maxFaceStep, std::fabs(fp - prevFacePres));
                    maxJawMismatch = std::max(maxJawMismatch, std::fabs(fp - bp));
                    prevFacePres = fp;
                    finalFacePres = fp;
                }
            }
        }
        EXPECT_TRUE(maxFaceStep <= 0.30f);     // Smooth ~tau=120ms ramp, NEVER snaps 0 -> 1 in 1 frame
        EXPECT_TRUE(maxJawMismatch <= 0.06f);  // Face and neck ramp together (zero jaw seam)
        EXPECT_TRUE(finalFacePres >= 0.90f);
    }

    // 3. skin-m3/skin-m5/tone-m3/tone-m5: Explicit uU5.z secondary flag (0 on single-face; 1 on dual-face & dual-loss decay)
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.maxFaces = 1;
        engine.setConfig(cfg);
        const auto mpL = makeFaceMp(0.28f, 0.26f);
        const auto mpR = makeFaceMp(0.72f, 0.26f);
        dev.calls.clear();
        engine.submitFaceLandmarks(11000, mpL.data(), 478, 720, 1280, 11000);
        engine.processFrame(camTex, 720, 1280, 11000, 11000);
        bool singleFlagZero = false;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 6) {
                singleFlagZero = (c.uniforms.v[5].z < 0.5f);
            }
        }
        EXPECT_TRUE(singleFlagZero);

        // Dual-face then simultaneous miss: uU5.z stays 1.0f during hold & smooth decay (zero 1-frame snap!)
        engine.resetTracking();
        cfg.maxFaces = 2;
        engine.setConfig(cfg);
        math::MultiFaceTracker::RawFaceInput pair[2] = {{mpL.data(), 478}, {mpR.data(), 478}};
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 12000 + f * 33;
            engine.submitMultiFaceLandmarks(ts, pair, 2, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        float prevSecP = 1.0f;
        float maxSecDelta = 0.0f;
        bool sawDualLossSecFlag = false;
        for (int f = 0; f < 85; ++f) {
            const int64_t ts = 12350 + f * 33;
            dev.calls.clear();
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 10) {
                    const float flag = c.uniforms.v[5].z;
                    const float sp = (flag > 0.5f) ? c.uniforms.v[9].z : 0.0f;
                    if (f >= 20 && flag > 0.5f && sp > 0.05f) sawDualLossSecFlag = true;
                    maxSecDelta = std::max(maxSecDelta, std::fabs(sp - prevSecP));
                    prevSecP = sp;
                }
            }
        }
        EXPECT_TRUE(sawDualLossSecFlag);
        EXPECT_TRUE(maxSecDelta <= 0.12f);
    }

    // 4. skin-m7: uU12.w carries pure lightLuma lowLightT (decoupled from bitrate pressure)
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.30f);
        dev.calls.clear();
        engine.submitFaceLandmarks(16000, mp.data(), 478, 720, 1280, 16000);
        // Bright scene (0.50) + terrible network (bitrateRatio = 0.2 -> bitratePressure = 1.0)
        engine.processFrame(camTex, 720, 1280, 16000, 16000, kInvalidTexture, 0.50f, 0.20f);
        float brightLowLightT = -1.0f;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 13) {
                brightLowLightT = c.uniforms.v[12].w;
            }
        }
        EXPECT_TRUE(brightLowLightT >= 0.0f && brightLowLightT <= 0.01f);

        dev.calls.clear();
        engine.submitFaceLandmarks(16033, mp.data(), 478, 720, 1280, 16033);
        engine.processFrame(camTex, 720, 1280, 16033, 16033, kInvalidTexture, 0.10f, 1.0f);
        float darkLowLightT = -1.0f;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 13) {
                darkLowLightT = c.uniforms.v[12].w;
            }
        }
        EXPECT_TRUE(darkLowLightT >= 0.99f);
    }

    // 5. gpuperf-m3 / cpuperf-m5: kMatteRefine smoothly skips once loadFadeBody <= 0.25 at L1+
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.backgroundBlur = 0.6f;
        engine.setConfig(cfg);
        engine.setHostLoadLevel(0);
        engine.processFrame(camTex, 720, 1280, 17000, 17000, personTex);
        EXPECT_TRUE((engine.stats().activePassesMask & kPassMatteRefine) != 0);
        engine.setHostLoadLevel(1);
        for (int f = 1; f <= 60; ++f) {
            const int64_t ts = 17000 + f * 33;
            engine.processFrame(camTex, 720, 1280, ts, ts, personTex);
        }
        EXPECT_TRUE((engine.stats().activePassesMask & kPassMatteRefine) == 0);
        engine.setHostLoadLevel(0);
    }

    // 6. makeup-m1: Piecewise iris curve reaches >= 0.95 at slider 1.0 while keeping presets calibrated
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.contactLens = 1.0f;
        cfg.catchlight = 1.0f;
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.30f);
        float maxLens = 0.0f, maxCl = 0.0f;
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 20000 + f * 33;
            dev.calls.clear();
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kMakeupIris && c.uniforms.count >= 3) {
                    maxLens = std::max(maxLens, c.uniforms.v[2].x);
                    maxCl = std::max(maxCl, c.uniforms.v[2].y);
                }
            }
        }
        EXPECT_TRUE(maxLens >= 0.95f);
        EXPECT_TRUE(maxCl >= 0.95f);
    }

    // 7. shape-m4: Chin-nod neck attenuation holds smoothly across transient face loss (< 1000ms)
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.neckLengthen = 1.0f;
        engine.setConfig(cfg);
        // Pose with shoulders at y=0.52 (top-down), hips at y=0.80 -> torso ~ 0.50 in aspect-normalized Y
        std::array<math::PoseLandmark, 33> pose{};
        for (auto& pt : pose) { pt.x = 0.50f; pt.y = 0.60f; pt.visibility = 0.95f; }
        pose[11] = {0.38f, 0.52f, 0.98f};
        pose[12] = {0.62f, 0.52f, 0.98f};
        pose[23] = {0.42f, 0.80f, 0.98f};
        pose[24] = {0.58f, 0.80f, 0.98f};
        // Face nodded low so chin landmark 152 sits close to shoulder line (top-down y ~ 0.51)
        auto mpNod = makeFaceMp(0.50f, 0.30f);
        const float dyShift = 0.51f - mpNod[152 * 2 + 1];
        for (int i = 0; i < 478; ++i) mpNod[i * 2 + 1] += dyShift;

        float neckWithFace = -1.0f;
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 22000 + f * 33;
            dev.calls.clear();
            engine.submitBodyPose(ts, pose.data(), 33, 720, 1280);
            engine.submitFaceLandmarks(ts, mpNod.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kBodyWarp && c.uniforms.count >= 3) {
                    neckWithFace = c.uniforms.v[2].y;
                }
            }
        }
        // Now submit face miss for 6 frames (~200ms): neck strength must NOT jump up to 4x!
        float maxNeckDelta = 0.0f;
        float prevNeck = neckWithFace;
        for (int f = 0; f < 6; ++f) {
            const int64_t ts = 22350 + f * 33;
            dev.calls.clear();
            engine.submitBodyPose(ts, pose.data(), 33, 720, 1280);
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kBodyWarp && c.uniforms.count >= 3) {
                    maxNeckDelta = std::max(maxNeckDelta, std::fabs(c.uniforms.v[2].y - prevNeck));
                    prevNeck = c.uniforms.v[2].y;
                }
            }
        }
        EXPECT_TRUE(neckWithFace > 0.0f && neckWithFace < 0.14f);  // Attenuated by nod
        EXPECT_TRUE(maxNeckDelta < 0.02f);                          // Zero jump on brief face miss!
    }

    engine.releaseGpuResources();
}

void testV397AuditFixesPipeline() {
    MockRhiDevice dev;
    LiveBeautyEngine engine(dev);
    const TextureId camTex = dev.createTexture2D(720, 1280, TextureFormat::kRgba8, nullptr);
    const TextureId personTex = dev.createTexture2D(256, 256, TextureFormat::kR8, nullptr);
    const TextureId hairTex = dev.createTexture2D(128, 128, TextureFormat::kR8, nullptr);
    engine.setHardwareProfile("Adreno (TM) 740", 8, 12288);

    auto makeFaceMp = [](float cx, float cyTopDown, float fw) {
        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float aspect = 720.0f / 1280.0f;
        const float scale = fw / math::FaceMeshFilter::canonicalFaceWidth();
        const float cyGl = 1.0f - cyTopDown;
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            mpXy[i * 2] = cx + xyz[i * 3] * scale;
            mpXy[i * 2 + 1] = 1.0f - (((cyGl / aspect) + xyz[i * 3 + 1] * scale) * aspect);
        }
        for (int i = 468; i < 478; ++i) {
            mpXy[i * 2] = cx + ((i >= 473) ? 0.06f : -0.06f);
            mpXy[i * 2 + 1] = cyTopDown - 0.06f;
        }
        return mpXy;
    };

    auto evalPrior = [](float ux, float uy, float aspect, const Vec4& oval, float ax, float ay, float pres) -> float {
        if (oval.z <= 0.001f || oval.w <= 0.001f || pres <= 0.001f) return 0.0f;
        const float fLen = std::hypot(ax, ay);
        const float fux = fLen > 0.001f ? (ax / fLen) : 1.0f;
        const float fuy = fLen > 0.001f ? (ay / fLen) : 0.0f;
        float bx = 0.70f + 0.30f * fux, by = 0.30f * fuy;
        const float bLen = std::max(std::hypot(bx, by), 1e-4f);
        bx /= bLen; by /= bLen;
        const float dpx = ux - oval.x, dpy = (uy - oval.y) / std::max(aspect, 0.0001f);
        const float pa = (dpx * fux + dpy * fuy) / std::max(oval.z, 0.0001f);
        const float pb = (-dpx * fuy + dpy * fux) / std::max(oval.w, 0.0001f);
        const float bodyRx = std::max(oval.z, oval.w * 0.77f);
        const float ba = (dpx * bx + dpy * by) / std::max(bodyRx, 0.0001f);
        const float bb = (-dpx * by + dpy * bx) / std::max(oval.w, 0.0001f);
        const float headHalo = (1.0f - detail::smoothstep(1.02f, 1.56f, std::hypot(pa, pb))) *
                               (1.0f - detail::smoothstep(0.75f, 1.12f, pb));
        const float down = -bb;
        const float sideReach = 1.35f + (2.55f - 1.35f) * detail::smoothstep(0.55f, 1.35f, down);
        const float neckChest = detail::smoothstep(0.28f, 0.92f, down) * (1.0f - detail::smoothstep(3.60f, 5.00f, down)) *
                                (1.0f - detail::smoothstep(sideReach - 0.45f, sideReach + 0.55f, std::fabs(ba)));
        return detail::clampf(std::max(headHalo, neckChest) * detail::clampf(pres, 0.0f, 1.0f), 0.0f, 1.0f);
    };

    auto evalPriorGrid = [&](const UniformBlock& u, std::vector<float>& outGrid) {
        constexpr int kGx = 24, kGy = 32;
        outGrid.resize(kGx * kGy);
        const float aspect = u.v[0].z;
        const float pPres = detail::clampf(1.0f - u.v[5].y, 0.0f, 1.0f);
        const bool hasSec = u.v[5].z > 0.5f;
        for (int gy = 0; gy < kGy; ++gy) {
            const float vy = (gy + 0.5f) / kGy;
            for (int gx = 0; gx < kGx; ++gx) {
                const float vx = (gx + 0.5f) / kGx;
                float p = evalPrior(vx, vy, aspect, u.v[6], u.v[4].w, u.v[5].w, pPres);
                if (hasSec) {
                    p = std::max(p, evalPrior(vx, vy, aspect, u.v[8], u.v[9].x, u.v[9].y, u.v[9].z));
                }
                outGrid[gy * kGx + gx] = p;
            }
        }
    };

    // 1. skin-k1 / tone-k1 (P2): During brief 600ms face miss (hold window), polygon skin mask MUST remain active
    //    (kPassSkinMask + uU0.w >= 4.5) so lips/eyes/brows retain exclM and are NOT whitened/smoothed!
    {
        auto cfg = BeautyConfig::naturalStudio();
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.50f, 0.30f);
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 1000 + f * 33;
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
        }
        float minFramePresInHold = 10.0f;
        bool allHoldFramesHadSkinMask = true;
        for (int f = 0; f < 18; ++f) {  // ~600ms miss
            const int64_t ts = 1350 + f * 33;
            dev.calls.clear();
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            if ((engine.stats().activePassesMask & kPassSkinMask) == 0) {
                allHoldFramesHadSkinMask = false;
            }
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 6) {
                    minFramePresInHold = std::min(minFramePresInHold, c.uniforms.v[0].w);
                }
            }
        }
        EXPECT_TRUE(allHoldFramesHadSkinMask);
        EXPECT_TRUE(minFramePresInHold >= 4.50f);
    }

    // 2. tone-k2 (P2 - tone-n1 regression): Relocation after 600ms miss, A->B swap after 400ms,
    //    and dual-face primary promotion MUST NOT single-frame teleport portraitPrior (max |dPrior| <= 0.25)!
    {
        auto runPriorMaxJump = [&](auto&& feedSeq) -> float {
            engine.resetTracking();
            std::vector<float> prevGrid, curGrid;
            float maxJump = 0.0f;
            bool havePrev = false;
            feedSeq([&](int64_t ts, bool recordJump) {
                dev.calls.clear();
                engine.processFrame(camTex, 720, 1280, ts, ts);
                for (const auto& c : dev.calls) {
                    if (c.shader == ShaderId::kCompositePro && c.uniforms.count >= 10) {
                        evalPriorGrid(c.uniforms, curGrid);
                        if (havePrev && recordJump) {
                            for (size_t i = 0; i < curGrid.size(); ++i) {
                                maxJump = std::max(maxJump, std::fabs(curGrid[i] - prevGrid[i]));
                            }
                        }
                        prevGrid = curGrid;
                        havePrev = true;
                    }
                }
            });
            return maxJump;
        };

        auto cfg = BeautyConfig::naturalStudio();
        cfg.maxFaces = 1;
        engine.setConfig(cfg);
        const auto mpA = makeFaceMp(0.30f, 0.50f, 0.26f);
        const auto mpB = makeFaceMp(0.72f, 0.50f, 0.26f);

        // (a) Single-face miss 600ms then reappear 0.42 away (relocation)
        const float jumpReloc = runPriorMaxJump([&](auto step) {
            for (int f = 0; f < 8; ++f) {
                const int64_t ts = 4000 + f * 33;
                engine.submitFaceLandmarks(ts, mpA.data(), 478, 720, 1280, ts);
                step(ts, f > 0);
            }
            for (int f = 0; f < 18; ++f) {
                const int64_t ts = 4270 + f * 33;
                engine.submitFaceMiss(ts);
                step(ts, true);
            }
            for (int f = 0; f < 12; ++f) {
                const int64_t ts = 4870 + f * 33;
                engine.submitFaceLandmarks(ts, mpB.data(), 478, 720, 1280, ts);
                step(ts, true);
            }
        });
        EXPECT_TRUE(jumpReloc <= 0.25f);

        // (b) Dual-face primary leaves -> secondary promotes cleanly with smooth prior!
        cfg.maxFaces = 2;
        engine.setConfig(cfg);
        math::MultiFaceTracker::RawFaceInput pair[2] = {{mpA.data(), 478}, {mpB.data(), 478}};
        math::MultiFaceTracker::RawFaceInput onlyB[1] = {{mpB.data(), 478}};
        const float jumpPromo = runPriorMaxJump([&](auto step) {
            for (int f = 0; f < 10; ++f) {
                const int64_t ts = 7000 + f * 33;
                engine.submitMultiFaceLandmarks(ts, pair, 2, 720, 1280, ts);
                step(ts, f > 0);
            }
            for (int f = 0; f < 30; ++f) {
                const int64_t ts = 7350 + f * 33;
                engine.submitMultiFaceLandmarks(ts, onlyB, 1, 720, 1280, ts);
                step(ts, true);
            }
        });
        EXPECT_TRUE(jumpPromo <= 0.25f);
        cfg.maxFaces = 1;
        engine.setConfig(cfg);
    }

    // 3. segmentation-k1 / gpuperf-k1 (P2/P3): kMatteRefine spatial weight uU1.w fades smoothly during L0 -> L1
    //    (maxFrameDelta <= 0.20, drops < 0.15 before kPassMatteRefine skips!).
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.backgroundBlur = 0.6f;
        engine.setConfig(cfg);
        engine.setHostLoadLevel(0);
        dev.calls.clear();
        engine.processFrame(camTex, 720, 1280, 10000, 10000, personTex);
        float prevSpatial = -1.0f;
        for (const auto& c : dev.calls) {
            if (c.shader == ShaderId::kMatteRefine && c.uniforms.count >= 2) {
                prevSpatial = c.uniforms.v[1].w;
            }
        }
        EXPECT_TRUE(prevSpatial >= 0.95f);  // Full spatial weight at L0 (v3.9.6 passed 0.0f!)

        engine.setHostLoadLevel(1);
        float maxDelta = 0.0f;
        float lastSpatialBeforeSkip = prevSpatial;
        bool skipped = false;
        for (int f = 1; f <= 40; ++f) {
            const int64_t ts = 10000 + f * 33;
            dev.calls.clear();
            engine.processFrame(camTex, 720, 1280, ts, ts, personTex);
            bool ran = false;
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kMatteRefine && c.uniforms.count >= 2) {
                    const float cur = c.uniforms.v[1].w;
                    maxDelta = std::max(maxDelta, std::fabs(cur - prevSpatial));
                    prevSpatial = cur;
                    lastSpatialBeforeSkip = cur;
                    ran = true;
                }
            }
            if (!ran) {
                skipped = true;
                break;
            }
        }
        EXPECT_TRUE(skipped);
        EXPECT_TRUE(maxDelta <= 0.20f);
        EXPECT_TRUE(lastSpatialBeforeSkip <= 0.15f);
        engine.setHostLoadLevel(0);
    }

    // 4. skin-e1 (P2): Mali-G57 (Standard) + hairDye=0.4 + hairMask keeps kPassHairAlpha active on L1 -> L2!
    {
        engine.resetTracking();
        engine.setHardwareProfile("Mali-G57 MC2", 8, 6144);
        auto cfg = BeautyConfig::naturalStudio();
        cfg.hairDye = 0.4f;
        engine.setConfig(cfg);
        const auto mp = makeFaceMp(0.50f, 0.50f, 0.30f);
        engine.setHostLoadLevel(1);
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 13000 + f * 33;
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts, kInvalidTexture, 0.30f, 1.0f, hairTex, ts);
        }
        EXPECT_TRUE((engine.stats().activePassesMask & kPassHairAlpha) != 0);
        engine.setHostLoadLevel(2);
        bool allL2HadHairAlpha = true;
        for (int f = 0; f < 15; ++f) {
            const int64_t ts = 13350 + f * 33;
            engine.submitFaceLandmarks(ts, mp.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts, kInvalidTexture, 0.30f, 1.0f, hairTex, ts);
            if ((engine.stats().activePassesMask & kPassHairAlpha) == 0) {
                allL2HadHairAlpha = false;
            }
        }
        EXPECT_TRUE(allL2HadHairAlpha);
        engine.setHostLoadLevel(0);
        engine.setHardwareProfile("Adreno (TM) 740", 8, 12288);
    }

    // 5. shape-k3 (m4 strengthened): >=25 consecutive face miss frames (>800ms, past FaceTimeline 500ms clear!)
    //    must keep chin-nod attenuated neck strength smooth with maxNeckDelta < 0.02!
    {
        engine.resetTracking();
        auto cfg = BeautyConfig::naturalStudio();
        cfg.neckLengthen = 1.0f;
        engine.setConfig(cfg);
        std::array<math::PoseLandmark, 33> pose{};
        for (auto& pt : pose) { pt.x = 0.50f; pt.y = 0.60f; pt.visibility = 0.95f; }
        pose[11] = {0.38f, 0.52f, 0.98f};
        pose[12] = {0.62f, 0.52f, 0.98f};
        pose[23] = {0.42f, 0.80f, 0.98f};
        pose[24] = {0.58f, 0.80f, 0.98f};
        auto mpNod = makeFaceMp(0.50f, 0.50f, 0.30f);
        const float dyShift = 0.51f - mpNod[152 * 2 + 1];
        for (int i = 0; i < 478; ++i) mpNod[i * 2 + 1] += dyShift;

        float prevNeck = -1.0f;
        for (int f = 0; f < 10; ++f) {
            const int64_t ts = 16000 + f * 33;
            dev.calls.clear();
            engine.submitBodyPose(ts, pose.data(), 33, 720, 1280);
            engine.submitFaceLandmarks(ts, mpNod.data(), 478, 720, 1280, ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kBodyWarp && c.uniforms.count >= 3) prevNeck = c.uniforms.v[2].y;
            }
        }
        EXPECT_TRUE(prevNeck > 0.0f && prevNeck < 0.14f);
        float maxNeckDelta = 0.0f;
        for (int f = 0; f < 25; ++f) {  // 825ms miss: crosses FaceTimeline 500ms mesh clear around f=15!
            const int64_t ts = 16350 + f * 33;
            dev.calls.clear();
            engine.submitBodyPose(ts, pose.data(), 33, 720, 1280);
            engine.submitFaceMiss(ts);
            engine.processFrame(camTex, 720, 1280, ts, ts);
            for (const auto& c : dev.calls) {
                if (c.shader == ShaderId::kBodyWarp && c.uniforms.count >= 3) {
                    maxNeckDelta = std::max(maxNeckDelta, std::fabs(c.uniforms.v[2].y - prevNeck));
                    prevNeck = c.uniforms.v[2].y;
                }
            }
        }
        EXPECT_TRUE(maxNeckDelta < 0.02f);
    }

    engine.releaseGpuResources();
}

}  // namespace

int main() {
    testPassThroughZeroCost();
    testFullBeautyPipelineOrderingAndNoFeedbackLoop();
    testHairProtectionVirtualBgStickersAndTelemetry();
    testHairWeightedGuidedDualFaceAndFrameDelayRing();
    testFourFrameDelayStage0DenoiseAndLutFilterSuppression();
    testP1ToP4PassesAndHardwareTierGating();
    testV34HairDyeHalfFloatAndFaceParsingIntegration();
    testV35EffectBundleTriggersAndWebGl2Backend();
    testCodeReviewEngineLifecycleAndPriorityGuards();
    testV36ChromaKeyParticlesMulticlassAndHandShield();
    testV3610CheekMakeupSteadyFaceAndAutoOccDecoupling();
    testV3613HairDyeStabilityDuringSlightHeadSwayAndLoadLevel1();
    testV370ImplicitRetouchOpenMouthGateAndIndoorDenoise();
    testV380AtmospherePostPipelineOpenMouthLipAndOcclusionMute();
    testV390PassFusionAutoWhiteBalanceFastBloomAndDualFaceOcclusion();
    testV393OcclusionDeferredAfterCompositeBeforeMakeup();
    testV394AuditFixesPipelineAndGpuPerf();
    testV395ReverificationFixesPipeline();
    testV396AuditFixesPipeline();
    testV397AuditFixesPipeline();
    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty RHI & RenderPipeline tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
