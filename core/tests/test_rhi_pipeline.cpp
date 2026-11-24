#include <cmath>
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

    void beginPass(RenderTargetId rt, int, int, bool, const Vec4&) override { activeRt = rt; }
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
            // 11 vec4 + 4 samplers = exact 15 uniform vectors budget.
            EXPECT_TRUE(c.uniforms.count == 11);
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
            EXPECT_TRUE(c.uniforms.v[0].x <= 0.181f);
            EXPECT_TRUE(std::fabs(c.uniforms.v[0].y - 0.010f) < 1e-4f);
            EXPECT_TRUE(std::fabs(c.uniforms.v[0].z - 0.032f) < 1e-4f);
        }
        if (c.shader == ShaderId::kCompositePro) {
            idxComposite = i;
            // LUT active -> legacy style (0) + skinOnly (16) = 16, legacy filterIntensity = 0
            EXPECT_TRUE(std::fabs(c.uniforms.v[1].x - 16.0f) < 1e-4f);
            EXPECT_TRUE(c.uniforms.v[2].w == 0.0f);
        }
    }
    EXPECT_TRUE(idxDenoise >= 0 && idxComposite > idxDenoise);
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
    EXPECT_TRUE(fp16Count >= 7);  // rtBody_, rtWarp_, rtGuided_[0..2], rtMacro_[0..1]

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
        EXPECT_TRUE(sGl.rgba16fCount >= 7);
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

    // 3. v3.6.2 Camera Shake Anti-Ghosting Gate:
    //    - IMU setCameraShaking(true) immediately flushes temporal denoise/matte and suppresses stale async hair/occlusion masks;
    //    - Large >10% face displacement since mask capture (exceeding HairMaskTuning::kAffineMaxShift) suppresses stale mask instead of snapping to identity!
    cfg.smoothing = 0.65f;
    cfg.matteTemporal = 0.70f;
    engine.setConfig(cfg);
    engine.setCameraShaking(true);
    engine.processFrame(inputTex, 720, 1280, 1283, 1283, kInvalidTexture, 0.05f, 1.0f);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassDenoise) == 0);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassMatteRefine) == 0);
    EXPECT_TRUE((engine.stats().activePassesMask & kPassHairDye) == 0);
    // v3.6.3: Fallback autoParsePersonTex_ MUST remain bound during camera shake so virtual background never flashes off!
    EXPECT_TRUE((engine.stats().activePassesMask & kPassBg) != 0);
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
    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty RHI & RenderPipeline tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
