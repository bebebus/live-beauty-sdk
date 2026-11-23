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

// V1 + H1: Virtual background replacement replaces background pixels with virtualBgTex while preserving foreground & hair flyaways.
void testVirtualBackgroundAndHairFlyawayProbeV1H1(MetalDevice& dev) {
    constexpr int kW = 64, kH = 64;
    // Camera input: solid red foreground person (220, 40, 40)
    std::vector<uint8_t> camPx(kW * kH * 4);
    for (int i = 0; i < kW * kH; ++i) {
        camPx[i * 4 + 0] = 220; camPx[i * 4 + 1] = 40; camPx[i * 4 + 2] = 40; camPx[i * 4 + 3] = 255;
    }
    // Virtual background: solid blue (30, 80, 230)
    std::vector<uint8_t> vbgPx(kW * kH * 4);
    for (int i = 0; i < kW * kH; ++i) {
        vbgPx[i * 4 + 0] = 30; vbgPx[i * 4 + 1] = 80; vbgPx[i * 4 + 2] = 230; vbgPx[i * 4 + 3] = 255;
    }
    // Person mask (R8 top-down): left third = person (255), right two-thirds = background (0)
    std::vector<uint8_t> personPx(kW * kH, 0);
    // Hair mask (R8 top-down): middle third = hair flyaway (255), right third = 0
    std::vector<uint8_t> hairPx(kW * kH, 0);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            if (x < 20) personPx[y * kW + x] = 255;
            if (x >= 22 && x < 42) hairPx[y * kW + x] = 255;
        }
    }

    const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
    const TextureId vbgTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, vbgPx.data());
    const TextureId personTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, personPx.data());
    const TextureId hairTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, hairPx.data());

    // Step 1: Merge person + hair mask
    const RenderTargetId mergedRt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
    {
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, personTex);
        tb.bind(1, hairTex);
        UniformBlock ub{};
        ub.push(1.0f, 0.0f, 0.0f, 1.0f);  // identity affine + shown=1.0
        ub.push(0.0f, 1.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.5f, 0.5f);
        ub.push(1.0f, 0.0f, 1.0f, 0.0f);  // hasFace=0 -> full head prior
        dev.beginPass(mergedRt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBgHairMerge, st, tb, ub);
        dev.endPass();
    }

    // Step 2: Virtual background mix
    const RenderTargetId outRt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
    {
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, camTex);
        tb.bind(1, vbgTex);
        tb.bind(2, dev.textureOf(mergedRt));
        UniformBlock ub{};
        ub.push(1.0f, 0.35f, 0.70f, 0.0f);
        ub.push(1.0f, 1.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.25f, 0.25f);
        ub.push(1.0f, 0.0f, 1.0f, 0.0f);
        dev.beginPass(outRt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
        dev.endPass();
    }

    std::vector<uint8_t> out(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(outRt), out.data(), kW, kH));
    const int fgIdx = (32 * kW + 10) * 4;
    const int hairIdx = (32 * kW + 32) * 4;
    const int bgIdx = (32 * kW + 54) * 4;
    // Foreground person kept red
    EXPECT_TRUE(out[fgIdx + 0] > 200 && out[fgIdx + 2] < 60);
    // Hair flyaway preserved via BgHairMerge (stays red instead of swallowed by blue bg)
    EXPECT_TRUE(out[hairIdx + 0] > 200 && out[hairIdx + 2] < 60);
    // Background replaced by blue virtual image
    EXPECT_TRUE(out[bgIdx + 2] > 210 && out[bgIdx + 0] < 50);

    dev.destroyRenderTarget(outRt);
    dev.destroyRenderTarget(mergedRt);
    dev.destroyTexture(hairTex);
    dev.destroyTexture(personTex);
    dev.destroyTexture(vbgTex);
    dev.destroyTexture(camTex);
}

// S1: 2D AR face sticker alpha-blends cleanly onto center frame.
void testStickerOverlayProbeS1(MetalDevice& dev) {
    constexpr int kW = 64, kH = 64;
    std::vector<uint8_t> basePx(kW * kH * 4);
    std::vector<uint8_t> stickerPx(kW * kH * 4);
    for (int i = 0; i < kW * kH; ++i) {
        basePx[i * 4 + 0] = 40; basePx[i * 4 + 1] = 40; basePx[i * 4 + 2] = 40; basePx[i * 4 + 3] = 255;
        stickerPx[i * 4 + 0] = 255; stickerPx[i * 4 + 1] = 220; stickerPx[i * 4 + 2] = 0; stickerPx[i * 4 + 3] = 255;
    }
    const TextureId baseTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, basePx.data());
    const TextureId stickerTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, stickerPx.data());
    const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

    // Quad covering center 50% (-0.5..+0.5 in NDC)
    static constexpr float kCenterQuad[16] = {
        -0.5f, -0.5f, 0.0f, 1.0f,
         0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f,  0.5f, 0.0f, 0.0f,
         0.5f,  0.5f, 1.0f, 0.0f,
    };
    static constexpr uint16_t kIdx[6] = {0, 1, 2, 2, 1, 3};
    const BufferId vb = dev.createDynamicVertexBuffer(sizeof(kCenterQuad));
    dev.updateVertexBuffer(vb, kCenterQuad, sizeof(kCenterQuad));
    const BufferId ib = dev.createStaticIndexBuffer(kIdx, 6);

    dev.beginPass(rt, kW, kH, false, {});
    {
        RenderState copySt{};
        TextureBindings tb{}; tb.bind(0, baseTex);
        UniformBlock ub{};
        dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
    }
    {
        RenderState blendSt{}; blendSt.blendEnable = true;
        TextureBindings tb{}; tb.bind(0, stickerTex);
        UniformBlock ub{}; ub.push(1.0f, 0.0f, 0.0f, 0.0f);
        dev.drawIndexed(ShaderId::kStickerQuad, blendSt, vb, ib, 0, 6, VertexLayout::kQuadPos2Uv2, tb, ub);
    }
    dev.endPass();

    std::vector<uint8_t> out(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
    const int centerIdx = (32 * kW + 32) * 4;
    const int cornerIdx = (4 * kW + 4) * 4;
    // Center covered by yellow sticker
    EXPECT_TRUE(out[centerIdx + 0] > 240 && out[centerIdx + 1] > 200 && out[centerIdx + 2] < 20);
    // Corner remains dark grey base
    EXPECT_TRUE(out[cornerIdx + 0] < 50 && out[cornerIdx + 1] < 50 && out[cornerIdx + 2] < 50);

    dev.destroyBuffer(ib);
    dev.destroyBuffer(vb);
    dev.destroyRenderTarget(rt);
    dev.destroyTexture(stickerTex);
    dev.destroyTexture(baseTex);
}

// H2: Hair-weighted guided filter prevents dark hair pixels (alpha=0) from bleeding gray halo into adjacent fair skin (alpha=255).
void testHairWeightedGuidedNoBleedProbeH2(MetalDevice& dev) {
    constexpr int kW = 64, kH = 64;
    // Left half = pitch-black hair (10,10,10), Right half = fair warm skin (215,175,155)
    std::vector<uint8_t> imgPx(kW * kH * 4);
    std::vector<uint8_t> weightPx(kW * kH * 4);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int i = (y * kW + x) * 4;
            if (x < 32) {
                imgPx[i+0] = 10; imgPx[i+1] = 10; imgPx[i+2] = 10; imgPx[i+3] = 255;
                weightPx[i+0] = 0; weightPx[i+1] = 0; weightPx[i+2] = 0; weightPx[i+3] = 0;  // hair
            } else {
                imgPx[i+0] = 215; imgPx[i+1] = 175; imgPx[i+2] = 155; imgPx[i+3] = 255;
                weightPx[i+0] = 255; weightPx[i+1] = 0; weightPx[i+2] = 0; weightPx[i+3] = 255;  // skin
            }
        }
    }
    const TextureId imgTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, imgPx.data());
    const TextureId weightTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, weightPx.data());
    const RenderTargetId rtBlur = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
    const RenderTargetId rtDiff = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
    const RenderTargetId rtUnwBlur = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
    const RenderTargetId rtUnwDiff = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

    // 1. Hair-weighted guided filter (kGuidedWeightBlur -> kGuidedWeightDiff)
    {
        RenderState st{};
        TextureBindings tb{}; tb.bind(0, imgTex); tb.bind(1, weightTex);
        UniformBlock ub{}; ub.push(1.25f / kW, 0.0f, 0.0f, 0.0f);
        dev.beginPass(rtBlur, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kGuidedWeightBlur, st, tb, ub);
        dev.endPass();
    }
    {
        RenderState st{};
        TextureBindings tb{}; tb.bind(0, dev.textureOf(rtBlur)); tb.bind(1, imgTex);
        UniformBlock ub{}; ub.push(0.0f, 1.25f / kH, 0.0f, 0.0f);
        dev.beginPass(rtDiff, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kGuidedWeightDiff, st, tb, ub);
        dev.endPass();
    }
    // 2. Unweighted reference (kBlurRgb -> kGuidedDiff)
    {
        RenderState st{};
        TextureBindings tb{}; tb.bind(0, imgTex);
        UniformBlock ub{}; ub.push(1.25f / kW, 0.0f, 0.0f, 0.0f);
        dev.beginPass(rtUnwBlur, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBlurRgb, st, tb, ub);
        dev.endPass();
    }
    {
        RenderState st{};
        TextureBindings tb{}; tb.bind(0, dev.textureOf(rtUnwBlur)); tb.bind(1, imgTex);
        UniformBlock ub{}; ub.push(0.0f, 1.25f / kH, 0.0f, 0.0f);
        dev.beginPass(rtUnwDiff, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kGuidedDiff, st, tb, ub);
        dev.endPass();
    }

    std::vector<uint8_t> outW(kW * kH * 4, 0), outU(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rtDiff), outW.data(), kW, kH));
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rtUnwDiff), outU.data(), kW, kH));
    // Adjacent to black hair boundary: hair-weighted mean dramatically eliminates dark-hair bleed vs unweighted!
    const int adj1 = (32 * kW + 33) * 4;
    const int adj2 = (32 * kW + 34) * 4;
    EXPECT_TRUE(static_cast<int>(outW[adj1 + 0]) - static_cast<int>(outU[adj1 + 0]) >= 35);
    EXPECT_TRUE(outW[adj2 + 0] >= 205);
    EXPECT_TRUE(outW[adj2 + 1] >= 168);

    dev.destroyRenderTarget(rtUnwDiff);
    dev.destroyRenderTarget(rtUnwBlur);
    dev.destroyRenderTarget(rtDiff);
    dev.destroyRenderTarget(rtBlur);
    dev.destroyTexture(weightTex);
    dev.destroyTexture(imgTex);
}

// M1: Dual-face co-streaming independently brightens/beautifies both faces simultaneously on Metal GPU.
void testDualFaceCoStreamingMetalProbeM1(MetalDevice& dev) {
    constexpr int kW = 160, kH = 240;
    std::vector<uint8_t> framePx(kW * kH * 4);
    for (int i = 0; i < kW * kH; ++i) {
        framePx[i * 4 + 0] = 178;
        framePx[i * 4 + 1] = 142;
        framePx[i * 4 + 2] = 122;
        framePx[i * 4 + 3] = 255;
    }
    const TextureId inputTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, framePx.data());
    LiveBeautyEngine engine(dev);
    BeautyConfig cfg = BeautyConfig::naturalStudio();
    cfg.maxFaces = 2;
    cfg.whiten = 0.70f;
    cfg.lutUrl = "";
    engine.setConfig(cfg);

    auto makeFace = [](float cx, float fw) {
        std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        const float aspect = static_cast<float>(kW) / static_cast<float>(kH);
        const float scale = fw / FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
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
    const auto fL = makeFace(0.27f, 0.30f);
    const auto fR = makeFace(0.73f, 0.30f);
    MultiFaceTracker::RawFaceInput pair[2] = {{fL.data(), 478}, {fR.data(), 478}};

    TextureId outTex = kInvalidTexture;
    for (int f = 0; f < 6; ++f) {
        const int64_t ts = 3000 + f * 33;
        EXPECT_TRUE(engine.submitMultiFaceLandmarks(ts, pair, 2, kW, kH, ts) == 2);
        outTex = engine.processFrame(inputTex, kW, kH, ts, ts);
    }
    EXPECT_TRUE(outTex != kInvalidTexture && outTex != inputTex);
    EXPECT_TRUE(engine.activeFaceCount() == 2);

    std::vector<uint8_t> outPx(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(outTex, outPx.data(), kW, kH));
    const int leftFaceIdx = ((kH / 2) * kW + static_cast<int>(kW * 0.27f)) * 4;
    const int rightFaceIdx = ((kH / 2) * kW + static_cast<int>(kW * 0.73f)) * 4;
    EXPECT_TRUE(outPx[leftFaceIdx] > framePx[leftFaceIdx]);
    EXPECT_TRUE(outPx[rightFaceIdx] > framePx[rightFaceIdx]);

    engine.releaseGpuResources();
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
        testVirtualBackgroundAndHairFlyawayProbeV1H1(dev);
        testStickerOverlayProbeS1(dev);
        testHairWeightedGuidedNoBleedProbeH2(dev);
        testDualFaceCoStreamingMetalProbeM1(dev);

        if (gFailures == 0) {
            std::cout << "PASS  all Apple Metal GPU pixel gates (L1/L2/L6, E1-E4, R1, V1/H1, H2, S1, M1, E2E) passed!\n";
            return 0;
        }
        std::cerr << "FAIL  " << gFailures << " Metal GPU pixel gate assertion(s) failed.\n";
        return 1;
    }
}
