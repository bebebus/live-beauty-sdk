#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/engine.h"
#include "livebeauty/math/face_makeup_uv.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "metal_device.h"

using namespace livebeauty;
using namespace livebeauty::rhi;
using namespace livebeauty::rhi::metal;
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

struct Rgba {
    uint8_t r, g, b, a;
};

TextureId makeSolidTex(MetalDevice& dev, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    std::vector<uint8_t> px(static_cast<size_t>(w * h * 4));
    for (int i = 0; i < w * h; ++i) {
        px[i * 4 + 0] = r;
        px[i * 4 + 1] = g;
        px[i * 4 + 2] = b;
        px[i * 4 + 3] = a;
    }
    return dev.createTexture2D(w, h, TextureFormat::kRgba8, px.data());
}

// Render a single fullscreen triangle pair using Makeup vertex layout (screenUv + makeupUv) into a 16x16 RT and read center pixel.
Rgba evalMakeupShader(MetalDevice& dev, ShaderId shader, Rgba baseColor, Rgba atlasColor,
                      const UniformBlock& ub, float makeupU = 0.5f, float makeupV = 0.5f) {
    constexpr int kSz = 16;
    const TextureId baseTex = makeSolidTex(dev, kSz, kSz, baseColor.r, baseColor.g, baseColor.b, baseColor.a);
    const TextureId atlasTex = makeSolidTex(dev, kSz, kSz, atlasColor.r, atlasColor.g, atlasColor.b, atlasColor.a);
    const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

    // 4 vertices * 6 floats: (ndcX, ndcY, screenU, screenV, makeupU, makeupV)
    const float verts[24] = {
        -1.0f, -1.0f, 0.0f, 0.0f, makeupU, makeupV,
         1.0f, -1.0f, 1.0f, 0.0f, makeupU, makeupV,
        -1.0f,  1.0f, 0.0f, 1.0f, makeupU, makeupV,
         1.0f,  1.0f, 1.0f, 1.0f, makeupU, makeupV,
    };
    const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
    const BufferId vb = dev.createDynamicVertexBuffer(sizeof(verts));
    dev.updateVertexBuffer(vb, verts, sizeof(verts));
    const BufferId ib = dev.createStaticIndexBuffer(indices, 6);

    // Clear RT with baseColor first so discard/blend behaves identically to real makeup pass.
    dev.beginPass(rt, kSz, kSz, true,
                  {baseColor.r / 255.0f, baseColor.g / 255.0f, baseColor.b / 255.0f, 1.0f});
    RenderState st{};
    st.blendEnable = true;
    st.cullBack = false;
    TextureBindings tb{};
    tb.bind(0, baseTex);
    tb.bind(1, atlasTex);
    dev.drawIndexed(shader, st, vb, ib, 0, 6, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
    dev.endPass();

    std::vector<uint8_t> out(kSz * kSz * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
    const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
    const Rgba result{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};

    dev.destroyBuffer(vb);
    dev.destroyBuffer(ib);
    dev.destroyRenderTarget(rt);
    dev.destroyTexture(baseTex);
    dev.destroyTexture(atlasTex);
    return result;
}

void testLipProbesL1L2L6(MetalDevice& dev) {
    const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
    const Vec3 rose = FaceMakeupUv::lipstickRgb(LipstickColor::kRoseBean);

    UniformBlock ubCherryGloss{};
    ubCherryGloss.push(cherry.x, cherry.y, cherry.z, 0.85f);
    ubCherryGloss.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);

    // L1: Dark lip parting seam (45,25,27) & oral cavity (22,14,15) must not brighten (dY <= 0.02).
    const Rgba l1Seam = evalMakeupShader(dev, ShaderId::kMakeupLip, {45, 25, 27, 255}, {0, 0, 255, 255}, ubCherryGloss, 0.5f, 0.70f);
    const float ySeam0 = (0.299f * 45 + 0.587f * 25 + 0.114f * 27) / 255.0f;
    const float ySeam1 = (0.299f * l1Seam.r + 0.587f * l1Seam.g + 0.114f * l1Seam.b) / 255.0f;
    EXPECT_TRUE(ySeam1 - ySeam0 <= 0.02f);

    const Rgba l1Cavity = evalMakeupShader(dev, ShaderId::kMakeupLip, {22, 14, 15, 255}, {0, 0, 255, 255}, ubCherryGloss, 0.5f, 0.70f);
    EXPECT_TRUE(std::abs(int(l1Cavity.r) - 22) <= 2 && std::abs(int(l1Cavity.g) - 14) <= 2);

    // L2: Exposed teeth (222,216,206) must stay white (delta(R-G) < 8).
    const Rgba l2Teeth = evalMakeupShader(dev, ShaderId::kMakeupLip, {222, 216, 206, 255}, {0, 0, 255, 255}, ubCherryGloss, 0.5f, 0.70f);
    EXPECT_TRUE((int(l2Teeth.r) - int(l2Teeth.g)) - (222 - 216) < 8);

    // L6: Bright front camera (210,152,150) glossy rose_bean & cherry_red must not wash out to pastel pink.
    UniformBlock ubRoseGloss{};
    ubRoseGloss.push(rose.x, rose.y, rose.z, 0.85f);
    ubRoseGloss.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);
    const Rgba l6Rose = evalMakeupShader(dev, ShaderId::kMakeupLip, {210, 152, 150, 255}, {0, 0, 255, 255}, ubRoseGloss, 0.5f, 0.73f);
    EXPECT_TRUE(l6Rose.r < 205 && l6Rose.g >= l6Rose.b - 4);

    const Rgba l6Cherry = evalMakeupShader(dev, ShaderId::kMakeupLip, {210, 152, 150, 255}, {0, 0, 255, 255}, ubCherryGloss, 0.5f, 0.73f);
    EXPECT_TRUE(l6Cherry.r > l6Cherry.g + 50 && l6Cherry.g >= l6Cherry.b - 4);
}

void testEyeshadowSkinGateProbesE1ToE4(MetalDevice& dev) {
    const Vec3 shadow = FaceMakeupUv::eyeshadowMainRgb(EyeshadowColor::kEarth);
    UniformBlock ub{};
    ub.push(shadow.x, shadow.y, shadow.z, 0.64f);
    ub.push(1.0f, 0.0f, 0.0f, 0.0f);  // R channel
    ub.push(1.0f, 0.0f, 1.0f, 0.0f);  // mode.x=1 (Multiply+SoftLight), mode.y=0 (direct UV), mode.z=1 (eyelid skin gate)

    // E1: Eyelid warm skin (195,148,128) gets natural soft powder tinting.
    const Rgba e1 = evalMakeupShader(dev, ShaderId::kMakeupLayer, {195, 148, 128, 255}, {255, 0, 0, 255}, ub);
    EXPECT_TRUE(e1.r < 190 && e1.g < 143);

    // E2: Bright & shadowed sclera -> exact 0/255 change.
    const Rgba e2a = evalMakeupShader(dev, ShaderId::kMakeupLayer, {210, 206, 204, 255}, {255, 0, 0, 255}, ub);
    EXPECT_TRUE(e2a.r == 210 && e2a.g == 206 && e2a.b == 204);
    const Rgba e2b = evalMakeupShader(dev, ShaderId::kMakeupLayer, {165, 162, 164, 255}, {255, 0, 0, 255}, ub);
    EXPECT_TRUE(e2b.r == 165 && e2b.g == 162 && e2b.b == 164);

    // E3: Dark iris/lashes (38,30,28) -> exact 0/255 change.
    const Rgba e3 = evalMakeupShader(dev, ShaderId::kMakeupLayer, {38, 30, 28, 255}, {255, 0, 0, 255}, ub);
    EXPECT_TRUE(e3.r == 38 && e3.g == 30 && e3.b == 28);

    // E4: Neutral grey wall (190,192,196) -> exact 0/255 change.
    const Rgba e4 = evalMakeupShader(dev, ShaderId::kMakeupLayer, {190, 192, 196, 255}, {255, 0, 0, 255}, ub);
    EXPECT_TRUE(e4.r == 190 && e4.g == 192 && e4.b == 196);
}

void testRetouchDarkSkinProbeR1(MetalDevice& dev) {
    constexpr int kSz = 16;
    const TextureId srcTex = makeSolidTex(dev, kSz, kSz, 70, 56, 48, 255);
    const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

    const float verts[32] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -0.05f,
         1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, -0.05f,
        -1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -0.05f,
         1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, -0.05f,
    };
    const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
    const BufferId vb = dev.createDynamicVertexBuffer(sizeof(verts));
    dev.updateVertexBuffer(vb, verts, sizeof(verts));
    const BufferId ib = dev.createStaticIndexBuffer(indices, 6);

    dev.beginPass(rt, kSz, kSz, true, {70 / 255.0f, 56 / 255.0f, 48 / 255.0f, 1.0f});
    RenderState st{};
    st.blendEnable = true;
    TextureBindings tb{};
    tb.bind(0, srcTex);
    UniformBlock ub{};
    ub.push(1.0f, 0.0f, 0.0f, 0.0f);
    dev.drawIndexed(ShaderId::kRetouchSkin, st, vb, ib, 0, 6, VertexLayout::kRetouchPos2Uv2A4, tb, ub);
    dev.endPass();

    std::vector<uint8_t> out(kSz * kSz * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
    const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
    // Uniform dark skin with zero under-eye deficit must have exact 0/255 change (no reverse-panda orange bar).
    EXPECT_TRUE(out[idx] == 70 && out[idx + 1] == 56 && out[idx + 2] == 48);

    dev.destroyBuffer(vb);
    dev.destroyBuffer(ib);
    dev.destroyRenderTarget(rt);
    dev.destroyTexture(srcTex);
}

void testEndToEndMetalBeautyEngine(MetalDevice& dev) {
    constexpr int kW = 180;
    constexpr int kH = 320;
    LiveBeautyEngine engine(dev);

    // Synthesize portrait skin gradient frame.
    std::vector<uint8_t> framePx(kW * kH * 4);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int i = (y * kW + x) * 4;
            framePx[i + 0] = static_cast<uint8_t>(185 + ((x + y) % 16));
            framePx[i + 1] = static_cast<uint8_t>(142 + ((x * 2) % 12));
            framePx[i + 2] = static_cast<uint8_t>(122 + (y % 10));
            framePx[i + 3] = 255;
        }
    }
    const TextureId inputTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, framePx.data());
    const TextureId dummyAtlas = makeSolidTex(dev, 8, 8, 128, 128, 128, 128);

    PipelineAssets assets{};
    assets.blushAtlas = dummyAtlas;
    assets.eyebrowAtlas = dummyAtlas;
    assets.eyeMakeupAtlas = dummyAtlas;
    assets.contourHighlightAtlas = dummyAtlas;
    assets.irisAtlas = dummyAtlas;
    engine.setAssets(assets);

    BeautyConfig cfg = BeautyConfig::koreanWaterGlow();
    cfg.cranialTop = 0.5f;
    cfg.nasolabial = 0.4f;
    cfg.lutUrl = "";  // no LUT texture uploaded in synthetic test
    engine.setConfig(cfg);

    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float aspect = static_cast<float>(kW) / static_cast<float>(kH);
    const float scale = 0.45f / FaceMeshFilter::canonicalFaceWidth();
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

    TextureId outTex = kInvalidTexture;
    for (int f = 0; f < 6; ++f) {
        const int64_t ts = 1000 + f * 33;
        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts));
        outTex = engine.processFrame(inputTex, kW, kH, ts, ts);
    }
    EXPECT_TRUE(outTex != kInvalidTexture && outTex != inputTex);

    std::vector<uint8_t> outPx(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(outTex, outPx.data(), kW, kH));
    // Verify center face pixel was brightened/smoothed naturally and is non-zero.
    const int cIdx = ((kH / 2) * kW + (kW / 2)) * 4;
    EXPECT_TRUE(outPx[cIdx] > framePx[cIdx]);  // whitening & radiance active
    EXPECT_TRUE(outPx[cIdx + 3] == 255);

    engine.releaseGpuResources();
    dev.destroyTexture(dummyAtlas);
    dev.destroyTexture(inputTex);
}

}  // namespace

int main() {
    @autoreleasepool {
        MetalDevice dev;
        if (!dev.isReady()) {
            std::cout << "SKIP  No Metal GPU device available on this host\n";
            return 0;
        }
        testLipProbesL1L2L6(dev);
        testEyeshadowSkinGateProbesE1ToE4(dev);
        testRetouchDarkSkinProbeR1(dev);
        testEndToEndMetalBeautyEngine(dev);

        if (gFailures == 0) {
            std::cout << "PASS  all Apple Metal GPU pixel gates (L1/L2/L6, E1-E4, R1, E2E KoreanWaterGlow) passed!\n";
            return 0;
        }
        std::cerr << "FAIL  " << gFailures << " Metal GPU pixel gate assertion(s) failed.\n";
        return 1;
    }
}
