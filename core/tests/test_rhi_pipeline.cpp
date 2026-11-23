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

}  // namespace

int main() {
    testPassThroughZeroCost();
    testFullBeautyPipelineOrderingAndNoFeedbackLoop();
    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty RHI & RenderPipeline tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
