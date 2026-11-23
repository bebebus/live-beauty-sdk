#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "livebeauty/engine.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/rhi/rhi_device.h"

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
    bool feedbackLoopDetected = false;
    RenderTargetId activeRt = kInvalidRenderTarget;

    RenderTargetId createRenderTarget(int w, int h, TextureFormat /*fmt*/) override {
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
    for (const auto& c : dev.calls) {
        if (c.shader == ShaderId::kBgHairMerge) sawHairMerge = true;
        if (c.shader == ShaderId::kBgVirtualMix) sawVirtualMix = true;
        if (c.shader == ShaderId::kHairAlpha) sawHairAlpha = true;
        if (c.shader == ShaderId::kStickerQuad) sawSticker = true;
    }
    EXPECT_TRUE(sawHairMerge);
    EXPECT_TRUE(sawVirtualMix);
    EXPECT_TRUE(sawHairAlpha);
    EXPECT_TRUE(sawSticker);

    const auto& st = engine.stats();
    EXPECT_TRUE(st.frameCount == 12);
    EXPECT_TRUE(st.facePresence > 0.9f);
    EXPECT_TRUE(st.hairShown > 0.5f);
    EXPECT_TRUE((st.activePassesMask & kPassBg) != 0);
    EXPECT_TRUE((st.activePassesMask & kPassHairAlpha) != 0);
    EXPECT_TRUE((st.activePassesMask & kPassSticker) != 0);
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
    // Primary + Secondary face each issue feather + inner skin-mask draws = 4 draws
    EXPECT_TRUE(skinMaskDraws == 4);
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

}  // namespace

int main() {
    testPassThroughZeroCost();
    testFullBeautyPipelineOrderingAndNoFeedbackLoop();
    testHairProtectionVirtualBgStickersAndTelemetry();
    testHairWeightedGuidedDualFaceAndFrameDelayRing();
    testFourFrameDelayStage0DenoiseAndLutFilterSuppression();
    testP1ToP4PassesAndHardwareTierGating();
    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty RHI & RenderPipeline tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
