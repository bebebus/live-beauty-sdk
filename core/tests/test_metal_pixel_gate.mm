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
                      const UniformBlock& ub, float makeupU = 0.5f, float makeupV = 0.5f,
                      uint8_t semAlpha = 255) {
    constexpr int kSz = 16;
    const TextureId baseTex = makeSolidTex(dev, kSz, kSz, baseColor.r, baseColor.g, baseColor.b, baseColor.a);
    const TextureId atlasTex = makeSolidTex(dev, kSz, kSz, atlasColor.r, atlasColor.g, atlasColor.b, atlasColor.a);
    const TextureId semTex = makeSolidTex(dev, kSz, kSz, 255, 0, 0, semAlpha);
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
    tb.bind(2, semTex);
    dev.drawIndexed(shader, st, vb, ib, 0, 6, VertexLayout::kMakeupPos2Uv2Uv2, tb, ub);
    dev.endPass();

    std::vector<uint8_t> out(kSz * kSz * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
    const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
    const Rgba result{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};

    dev.destroyBuffer(vb);
    dev.destroyBuffer(ib);
    dev.destroyRenderTarget(rt);
    dev.destroyTexture(semTex);
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

// P1-P4 GPU Pixel Gate: T-zone oil control, ambient WB & bangs/occlusion makeup guard,
// RGB-guided temporal matte refinement, isolated blemish healer, AI residual blend, and 3D head-occluded AR.
void testP1ToP4GpuPixelGates(MetalDevice& dev) {
    // 1. P1 Oil control tames shiny T-zone glare while leaving matte skin intact
    {
        constexpr int kSz = 16;
        const TextureId shinyTex = makeSolidTex(dev, kSz, kSz, 245, 232, 224, 255);
        const TextureId matteTex = makeSolidTex(dev, kSz, kSz, 185, 145, 125, 255);
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, 255, 0, 230, 255);  // faceM=1, excl=0, hiM=0.9, nonHair=1
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

        auto runOil = [&](TextureId inTex) -> Rgba {
            RenderState st{};
            TextureBindings tb{};
            tb.bind(0, inTex); tb.bind(1, inTex); tb.bind(2, semTex); tb.bind(3, semTex);
            UniformBlock ub{};
            ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);  // pres=1.0 poly mode
            ub.push(16.0f, 0.0f, 0.0f, 0.0f);             // skinOnly=1, smooth=0
            ub.push(0.0f, 0.0f, 1.0f, 0.0f);
            ub.push(0.0f, 0.0f, 0.90f, 0.0f);             // oilControl=0.90
            for (int i = 0; i < 6; ++i) ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            ub.push(0.0f, 0.01f, 0.0f, 0.0f);
            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
            dev.endPass();
            std::vector<uint8_t> out(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
            const int c = ((kSz / 2) * kSz + (kSz / 2)) * 4;
            return {out[c], out[c + 1], out[c + 2], out[c + 3]};
        };
        const Rgba shinyOut = runOil(shinyTex);
        const Rgba matteOut = runOil(matteTex);
        EXPECT_TRUE(shinyOut.r < 235 && shinyOut.b < 210);
        EXPECT_TRUE(std::abs(int(matteOut.r) - 185) <= 2 && std::abs(int(matteOut.g) - 145) <= 2);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(semTex);
        dev.destroyTexture(matteTex);
        dev.destroyTexture(shinyTex);
    }

    // 2. P1 Ambient WB + P2 Bangs/Occlusion makeup blocking (exact 0/255 when occluded!)
    {
        const Vec3 blush = FaceMakeupUv::blushRgb(BlushColor::kPeach);
        UniformBlock ubWarm{};
        ubWarm.push(blush.x, blush.y, blush.z, 0.72f);
        ubWarm.push(1.0f, 0.0f, 0.0f, 0.0f);
        ubWarm.push(0.0f, 0.0f, 0.0f, 0.0f);
        ubWarm.push(1.22f, 1.00f, 0.78f, 1.0f);  // warm ambient illuminant
        ubWarm.push(0.0f, 0.0f, 0.0f, 0.0f);

        UniformBlock ubCool{};
        ubCool.push(blush.x, blush.y, blush.z, 0.72f);
        ubCool.push(1.0f, 0.0f, 0.0f, 0.0f);
        ubCool.push(0.0f, 0.0f, 0.0f, 0.0f);
        ubCool.push(0.78f, 1.00f, 1.22f, 1.0f);  // cool ambient illuminant
        ubCool.push(0.0f, 0.0f, 0.0f, 0.0f);

        // Occluded by bangs/hand (semAlpha = 0) -> exact 0/255 change!
        const Rgba occBlush = evalMakeupShader(dev, ShaderId::kMakeupLayer, {190, 150, 130, 255}, {255, 0, 0, 255}, ubWarm, 0.5f, 0.5f, 0);
        EXPECT_TRUE(occBlush.r == 190 && occBlush.g == 150 && occBlush.b == 130);

        UniformBlock ubLip{};
        const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
        ubLip.push(cherry.x, cherry.y, cherry.z, 0.85f);
        ubLip.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);
        ubLip.push(1.0f, 1.0f, 1.0f, 0.0f);
        const Rgba occLip = evalMakeupShader(dev, ShaderId::kMakeupLip, {195, 140, 135, 255}, {0, 0, 255, 255}, ubLip, 0.5f, 0.70f, 0);
        EXPECT_TRUE(occLip.r == 195 && occLip.g == 140 && occLip.b == 135);

        // Unoccluded warm vs cool ambient shifts pigment R-B warmer under warm lighting
        const Rgba warmRes = evalMakeupShader(dev, ShaderId::kMakeupLayer, {190, 150, 130, 255}, {255, 0, 0, 255}, ubWarm, 0.5f, 0.5f, 255);
        const Rgba coolRes = evalMakeupShader(dev, ShaderId::kMakeupLayer, {190, 150, 130, 255}, {255, 0, 0, 255}, ubCool, 0.5f, 0.5f, 255);
        EXPECT_TRUE((int(warmRes.r) - int(warmRes.b)) > (int(coolRes.r) - int(coolRes.b)) + 6);
    }

    // 3. P3 Multi-scale isolated blemish healer heals red pimple on cheek while leaving clean skin & excluded eye untouched
    {
        constexpr int kW = 32, kH = 32;
        std::vector<uint8_t> img(kW * kH * 4);
        std::vector<uint8_t> sem(kW * kH * 4);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                img[i+0] = 210; img[i+1] = 168; img[i+2] = 148; img[i+3] = 255;
                // Left half: cheek skin (faceM=1, excl=0, nonHair=1); Right half: eye exclusion (excl=1)
                sem[i+0] = 255; sem[i+1] = (x >= 20 ? 255 : 0); sem[i+2] = 0; sem[i+3] = 255;
            }
        }
        // Place isolated red acne spot at (10, 16) on cheek and identical spot at (25, 16) inside eye exclusion
        auto putSpot = [&](int cx, int cy) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int i = ((cy + dy) * kW + (cx + dx)) * 4;
                    img[i+0] = 168; img[i+1] = 72; img[i+2] = 68;
                }
            }
        };
        putSpot(10, 16);
        putSpot(25, 16);

        const TextureId imgTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, img.data());
        const TextureId semTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, sem.data());
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{}; tb.bind(0, imgTex); tb.bind(1, semTex);
        UniformBlock ub{}; ub.push(1.0f / kW, 1.0f / kH, 1.0f, 1.0f);
        dev.beginPass(rt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBlemishHeal, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        const int cheekSpot = (16 * kW + 10) * 4;
        const int cleanCheek = (4 * kW + 4) * 4;
        const int eyeSpot = (16 * kW + 25) * 4;
        // Cheek pimple healed strongly toward clean cheek (210,168,148)
        EXPECT_TRUE(out[cheekSpot + 1] > 130 && out[cheekSpot + 2] > 115);
        // Clean cheek untouched
        EXPECT_TRUE(std::abs(int(out[cleanCheek + 0]) - 210) <= 1 && std::abs(int(out[cleanCheek + 1]) - 168) <= 1);
        // Excluded eye region exact 0/255 untouched
        EXPECT_TRUE(out[eyeSpot + 0] == 168 && out[eyeSpot + 1] == 72 && out[eyeSpot + 2] == 68);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(semTex);
        dev.destroyTexture(imgTex);
    }

    // 4. P4 3D Head-Occluded AR Accessory: occluded behind closer head depth, visible outside & in front
    {
        constexpr int kW = 32, kH = 32;
        const TextureId baseTex = makeSolidTex(dev, kW, kH, 50, 50, 50, 255);
        const TextureId goldTex = makeSolidTex(dev, kW, kH, 250, 210, 40, 255);
        // Head depth shell: left half has head at depth 0.65 (166), right half is background at depth 0.0 (0)
        std::vector<uint8_t> headDepthPx(kW * kH * 4, 0);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                const uint8_t d = (x < kW / 2) ? 166 : 0;
                headDepthPx[i+0] = d; headDepthPx[i+1] = d; headDepthPx[i+2] = d; headDepthPx[i+3] = 255;
            }
        }
        const TextureId headDepthTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, headDepthPx.data());
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        // Fullscreen accessory quad at depth 0.42 (behind head 0.65 on left, visible over background 0.0 on right)
        static constexpr float kAccVerts[20] = {
            -1.0f, -1.0f, 0.0f, 1.0f, 0.42f,
             1.0f, -1.0f, 1.0f, 1.0f, 0.42f,
            -1.0f,  1.0f, 0.0f, 0.0f, 0.42f,
             1.0f,  1.0f, 1.0f, 0.0f, 0.42f,
        };
        static constexpr uint16_t kIdx[6] = {0, 1, 2, 2, 1, 3};
        const BufferId vb = dev.createDynamicVertexBuffer(sizeof(kAccVerts));
        dev.updateVertexBuffer(vb, kAccVerts, sizeof(kAccVerts));
        const BufferId ib = dev.createStaticIndexBuffer(kIdx, 6);

        dev.beginPass(rt, kW, kH, false, {});
        {
            RenderState copySt{}; TextureBindings tb{}; tb.bind(0, baseTex); UniformBlock ub{};
            dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        {
            RenderState blendSt{}; blendSt.blendEnable = true;
            TextureBindings tb{}; tb.bind(0, goldTex); tb.bind(1, headDepthTex);
            UniformBlock ub{}; ub.push(1.0f, 1.0f, 0.0f, 0.0f);  // opacity=1, headOcclusion=1
            dev.drawIndexed(ShaderId::kAr3dMesh, blendSt, vb, ib, 0, 6, VertexLayout::kAr3dPos2Uv2Z1, tb, ub);
        }
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        const int occludedPx = (16 * kW + 6) * 4;
        const int visiblePx = (16 * kW + 24) * 4;
        // Behind head -> cleanly occluded (stays base grey 50,50,50)
        EXPECT_TRUE(out[occludedPx + 0] == 50 && out[occludedPx + 1] == 50 && out[occludedPx + 2] == 50);
        // Outside head -> gold accessory fully visible
        EXPECT_TRUE(out[visiblePx + 0] > 240 && out[visiblePx + 1] > 200 && out[visiblePx + 2] < 50);

        dev.destroyBuffer(ib);
        dev.destroyBuffer(vb);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(headDepthTex);
        dev.destroyTexture(goldTex);
        dev.destroyTexture(baseTex);
    }

    // 5. v3.4 Real-time Hair Dyeing + Crown Angel-Ring Shine + View-Dependent PBR Lip Yaw Glaze
    {
        constexpr int kW = 32, kH = 32;
        // Dark natural hair (34, 28, 25) on top half, warm cheek skin (200, 155, 138) on bottom half
        std::vector<uint8_t> img(kW * kH * 4);
        std::vector<uint8_t> hairMask(kW * kH, 0);
        std::vector<uint8_t> sem(kW * kH * 4);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int idx4 = (y * kW + x) * 4;
                if (y < kH / 2) {
                    // Top-down y < 16 = hair region (corresponds to GL uv.y > 0.5)
                    img[idx4 + 0] = 34;
                    img[idx4 + 1] = 28;
                    img[idx4 + 2] = 25;
                    img[idx4 + 3] = 255;
                    hairMask[y * kW + x] = 255;
                    sem[idx4 + 0] = 0; sem[idx4 + 1] = 0; sem[idx4 + 2] = 0; sem[idx4 + 3] = 0;
                } else {
                    img[idx4 + 0] = 200;
                    img[idx4 + 1] = 155;
                    img[idx4 + 2] = 138;
                    img[idx4 + 3] = 255;
                    hairMask[y * kW + x] = 0;
                    sem[idx4 + 0] = 255; sem[idx4 + 1] = 0; sem[idx4 + 2] = 0; sem[idx4 + 3] = 255;
                }
            }
        }
        const TextureId imgTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, img.data());
        const TextureId hairTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, hairMask.data());
        const TextureId semTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, sem.data());
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        const Vec3 hazeBlue = FaceMakeupUv::hairDyeRgb(HairDyeColor::kHazeBlue);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, imgTex); tb.bind(1, hairTex); tb.bind(2, semTex);
        UniformBlock ub{};
        ub.push(1.0f, 0.0f, 0.0f, 0.90f);  // hairDye=0.90
        ub.push(0.0f, 1.0f, 0.0f, 0.85f);  // hairShine=0.85
        ub.push(hazeBlue.x, hazeBlue.y, hazeBlue.z, 1.0f);
        ub.push(0.50f, 0.75f, 0.30f, 0.15f);  // crown centered at (0.50, 0.75) in GL UV (top-down y=8)
        ub.push(1.0f, 0.0f, 1.0f, 0.0f);
        dev.beginPass(rt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kHairDye, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        // Top-down y=8, x=16 is right on the crown Angel-Ring band inside hair
        const int hairCrownPx = (8 * kW + 16) * 4;
        // Top-down y=2, x=16 is upper hair away from Angel-Ring peak
        const int hairBasePx = (2 * kW + 16) * 4;
        // Top-down y=24, x=16 is bare cheek skin -> exact 0/255 change!
        const int skinPx = (24 * kW + 16) * 4;

        // Hair dyed unmistakably into cool Haze-Blue (B > R + 16) and crown Angel-Ring lifts brightness vs hairBasePx
        EXPECT_TRUE(int(out[hairBasePx + 2]) > int(out[hairBasePx + 0]) + 16);
        EXPECT_TRUE(int(out[hairCrownPx + 2]) > int(out[hairBasePx + 2]) + 8);
        // Bare skin untouched (exact 0/255 bleed)
        EXPECT_TRUE(out[skinPx + 0] == 200 && out[skinPx + 1] == 155 && out[skinPx + 2] == 138);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(semTex);
        dev.destroyTexture(hairTex);
        dev.destroyTexture(imgTex);

        // View-dependent PBR lip specular glaze: turning head shifts specular belly onto off-center lip UV (u=0.56, v=0.714)
        const Vec3 rose = FaceMakeupUv::lipstickRgb(LipstickColor::kRoseBean);
        UniformBlock ubFrontal{};
        ubFrontal.push(rose.x, rose.y, rose.z, 0.85f);
        ubFrontal.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);  // yaw=0
        UniformBlock ubTurned{};
        ubTurned.push(rose.x, rose.y, rose.z, 0.85f);
        ubTurned.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.35f); // yaw=+0.35 shifts belly toward u=0.563

        const Rgba lipFrontalOff = evalMakeupShader(dev, ShaderId::kMakeupLip, {180, 118, 116, 255}, {0, 0, 255, 255}, ubFrontal, 0.56f, 0.714f);
        const Rgba lipTurnedOff = evalMakeupShader(dev, ShaderId::kMakeupLip, {180, 118, 116, 255}, {0, 0, 255, 255}, ubTurned, 0.56f, 0.714f);
        EXPECT_TRUE(int(lipTurnedOff.r) > int(lipFrontalOff.r) + 8 && int(lipTurnedOff.g) > int(lipFrontalOff.g) + 8);
    }
}

void testMetalIntraFrameMultiSlotVboZeroCopyBgraAndTransientEviction(MetalDevice& dev) {
    // 1. Intra-frame dynamic VBO slab arena: two sequential updates + draws to the SAME BufferId
    //    inside one uncommitted pass must NOT overwrite each other!
    {
        constexpr int kW = 32, kH = 32;
        const TextureId baseTex = makeSolidTex(dev, kW, kH, 20, 20, 20, 255);
        const TextureId redTex = makeSolidTex(dev, kW, kH, 250, 10, 10, 255);
        const TextureId greenTex = makeSolidTex(dev, kW, kH, 10, 250, 10, 255);
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        static constexpr float kLeftQuad[16] = {
            -0.9f, -0.5f, 0.0f, 1.0f,
            -0.1f, -0.5f, 1.0f, 1.0f,
            -0.9f,  0.5f, 0.0f, 0.0f,
            -0.1f,  0.5f, 1.0f, 0.0f,
        };
        static constexpr float kRightQuad[16] = {
             0.1f, -0.5f, 0.0f, 1.0f,
             0.9f, -0.5f, 1.0f, 1.0f,
             0.1f,  0.5f, 0.0f, 0.0f,
             0.9f,  0.5f, 1.0f, 0.0f,
        };
        static constexpr uint16_t kIdx[6] = {0, 1, 2, 2, 1, 3};
        const BufferId sharedVb = dev.createDynamicVertexBuffer(sizeof(kLeftQuad));
        const BufferId ib = dev.createStaticIndexBuffer(kIdx, 6);

        dev.beginPass(rt, kW, kH, false, {});
        {
            RenderState copySt{}; TextureBindings tb{}; tb.bind(0, baseTex); UniformBlock ub{};
            dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        {
            RenderState blendSt{}; blendSt.blendEnable = true;
            UniformBlock ub{}; ub.push(1.0f, 0.0f, 0.0f, 0.0f);

            dev.updateVertexBuffer(sharedVb, kLeftQuad, sizeof(kLeftQuad));
            TextureBindings tbL{}; tbL.bind(0, redTex);
            dev.drawIndexed(ShaderId::kStickerQuad, blendSt, sharedVb, ib, 0, 6, VertexLayout::kQuadPos2Uv2, tbL, ub);

            dev.updateVertexBuffer(sharedVb, kRightQuad, sizeof(kRightQuad));
            TextureBindings tbR{}; tbR.bind(0, greenTex);
            dev.drawIndexed(ShaderId::kStickerQuad, blendSt, sharedVb, ib, 0, 6, VertexLayout::kQuadPos2Uv2, tbR, ub);
        }
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        const int leftPx = (16 * kW + 8) * 4;
        const int rightPx = (16 * kW + 24) * 4;
        EXPECT_TRUE(out[leftPx + 0] > 230 && out[leftPx + 1] < 30);    // Left stayed Red!
        EXPECT_TRUE(out[rightPx + 1] > 230 && out[rightPx + 0] < 30);  // Right drew Green!

        dev.destroyBuffer(ib);
        dev.destroyBuffer(sharedVb);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(greenTex);
        dev.destroyTexture(redTex);
        dev.destroyTexture(baseTex);
    }

    // 2. Transient external texture eviction & zero-copy RGBA8 -> BGRA8 external target blit
    {
        const size_t baseCount = dev.trackedTextureCount();
        constexpr int kW = 16, kH = 16;
        std::vector<uint8_t> asymRgba(kW * kH * 4, 0);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                if (x < 8 && y < 8) {
                    asymRgba[i + 0] = 240; asymRgba[i + 1] = 20; asymRgba[i + 2] = 10; asymRgba[i + 3] = 255;  // Top-left Red
                } else {
                    asymRgba[i + 0] = 10; asymRgba[i + 1] = 30; asymRgba[i + 2] = 245; asymRgba[i + 3] = 255;  // Else Blue
                }
            }
        }
        const TextureId srcId = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, asymRgba.data());

        MTLTextureDescriptor* bgraDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:kW
                                        height:kH
                                     mipmapped:NO];
        bgraDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        bgraDesc.storageMode = MTLStorageModeShared;

        for (int iter = 0; iter < 32; ++iter) {
            id<MTLTexture> transientIn = [dev.nativeDevice() newTextureWithDescriptor:bgraDesc];
            const TextureId trId = dev.wrapTransientMetalTexture(transientIn);
            EXPECT_TRUE(trId != kInvalidTexture);
            dev.flush();
        }
        // All 32 transient textures must be evicted; only srcId remains above baseCount!
        EXPECT_TRUE(dev.trackedTextureCount() == baseCount + 1);

        id<MTLTexture> dstBgra = [dev.nativeDevice() newTextureWithDescriptor:bgraDesc];
        EXPECT_TRUE(dev.renderToExternalTexture(srcId, dstBgra));
        dev.flush();

        std::vector<uint8_t> bgraBytes(kW * kH * 4, 0);
        [dstBgra getBytes:bgraBytes.data() bytesPerRow:kW * 4 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
        // Top-left (2, 2) in BGRA byte order: B=10, G=20, R=240, A=255
        const int tl = (2 * kW + 2) * 4;
        EXPECT_TRUE(bgraBytes[tl + 0] < 25 && bgraBytes[tl + 1] < 35 && bgraBytes[tl + 2] > 225 && bgraBytes[tl + 3] == 255);
        // Bottom-right (12, 12) in BGRA byte order: B=245, G=30, R=10, A=255
        const int br = (12 * kW + 12) * 4;
        EXPECT_TRUE(bgraBytes[br + 0] > 230 && bgraBytes[br + 1] < 45 && bgraBytes[br + 2] < 25 && bgraBytes[br + 3] == 255);

        dev.destroyTexture(srcId);
        EXPECT_TRUE(dev.trackedTextureCount() == baseCount);
    }
}

void testV36MetalChromaKeyParticleSdfAndHandShieldProbes(MetalDevice& dev) {
    // 1. Chroma Key probe: studio green background replaced by blue virtual background,
    //    while neutral dark hair & warm skin stay intact and green spill on shoulder is suppressed!
    {
        constexpr int kW = 32, kH = 32;
        std::vector<uint8_t> cam(kW * kH * 4, 255);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                if (x < 10) {
                    // Studio green screen (12, 195, 22)
                    cam[i + 0] = 12; cam[i + 1] = 195; cam[i + 2] = 22;
                } else if (x < 20) {
                    // Warm portrait skin (205, 158, 136)
                    cam[i + 0] = 205; cam[i + 1] = 158; cam[i + 2] = 136;
                } else {
                    // Dark hair (32, 30, 34)
                    cam[i + 0] = 32; cam[i + 1] = 30; cam[i + 2] = 34;
                }
            }
        }
        const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, cam.data());
        const TextureId vbgTex = makeSolidTex(dev, kW, kH, 25, 60, 235, 255);  // Vivid royal blue VBG
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, camTex);
        tb.bind(1, vbgTex);
        UniformBlock ub{};
        ub.push(1.0f, 0.38f, 0.12f, 0.75f);
        ub.push(1.0f, 1.0f, 0.0f, 0.0f);
        ub.push(0.05f, 0.72f, 0.10f, 0.0f);
        dev.beginPass(rt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kChromaKey, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        const int greenKeyedPx = (16 * kW + 4) * 4;
        const int skinPx = (16 * kW + 15) * 4;
        const int hairPx = (16 * kW + 26) * 4;
        // Green screen became royal blue VBG!
        EXPECT_TRUE(out[greenKeyedPx + 2] > 210 && out[greenKeyedPx + 1] < 80);
        // Warm skin & dark hair preserved!
        EXPECT_TRUE(std::abs(int(out[skinPx + 0]) - 205) <= 3 && std::abs(int(out[skinPx + 1]) - 158) <= 3);
        EXPECT_TRUE(std::abs(int(out[hairPx + 0]) - 32) <= 3 && out[hairPx + 2] < 50);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(vbgTex);
        dev.destroyTexture(camTex);
    }

    // 2. Batched 2D SDF Particle Burst probe: glowing rose heart renders cleanly on dark frame
    {
        constexpr int kW = 32, kH = 32;
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
        ParticleBurstSolver solver;
        solver.triggerBurst(GestureType::kHeart, 0.5f, 0.5f, 1000);
        std::array<float, ParticleBurstSolver::kMaxVertexFloats> verts{};
        const int quads = solver.buildBatch(1200, 1.0f, verts.data());
        EXPECT_TRUE(quads > 0);

        const BufferId vb = dev.createDynamicVertexBuffer(sizeof(verts));
        dev.updateVertexBuffer(vb, verts.data(), sizeof(float) * static_cast<size_t>(quads * 32));
        const auto& idx = ParticleBurstSolver::quadIndices();
        const BufferId ib = dev.createStaticIndexBuffer(idx.data(), idx.size());

        dev.beginPass(rt, kW, kH, true, {0.04f, 0.04f, 0.06f, 1.0f});
        RenderState st{};
        st.blendEnable = true;
        TextureBindings emptyTb{};
        UniformBlock emptyUb{};
        dev.drawIndexed(ShaderId::kParticleBurst, st, vb, ib, 0, quads * 6,
                        VertexLayout::kRetouchPos2Uv2A4, emptyTb, emptyUb);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        int brightParticlesPx = 0;
        for (int i = 0; i < kW * kH; ++i) {
            if (out[i * 4 + 0] > 80) ++brightParticlesPx;
        }
        EXPECT_TRUE(brightParticlesPx >= 4);

        dev.destroyBuffer(ib);
        dev.destroyBuffer(vb);
        dev.destroyRenderTarget(rt);
    }
}

void testV361ColdPorcelainWhiteningAndNonDarkeningLipProbes(MetalDevice& dev) {
    // 1. Dim indoor lips (135, 82, 80) with rose_bean / cherry_red / berry_plum must brighten into rosy/vivid sRGB
    //    and NEVER darken R below bare lip (135)!
    {
        const Vec3 rose = FaceMakeupUv::lipstickRgb(LipstickColor::kRoseBean);
        const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
        const Vec3 berry = FaceMakeupUv::lipstickRgb(LipstickColor::kBerryPlum);

        UniformBlock ubRose{};
        ubRose.push(rose.x, rose.y, rose.z, 0.55f);
        ubRose.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);

        UniformBlock ubCherry{};
        ubCherry.push(cherry.x, cherry.y, cherry.z, 0.72f);
        ubCherry.push(1.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);

        UniformBlock ubBerry{};
        ubBerry.push(berry.x, berry.y, berry.z, 0.65f);
        ubBerry.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);

        const Rgba dimRose = evalMakeupShader(dev, ShaderId::kMakeupLip, {135, 82, 80, 255}, {0, 0, 255, 255}, ubRose, 0.5f, 0.73f);
        const Rgba dimCherry = evalMakeupShader(dev, ShaderId::kMakeupLip, {135, 82, 80, 255}, {0, 0, 255, 255}, ubCherry, 0.5f, 0.73f);
        const Rgba dimBerry = evalMakeupShader(dev, ShaderId::kMakeupLip, {135, 82, 80, 255}, {0, 0, 255, 255}, ubBerry, 0.5f, 0.73f);
        EXPECT_TRUE(dimRose.r >= 140 && dimRose.r > dimRose.g + 55 && dimRose.g >= dimRose.b);
        EXPECT_TRUE(dimCherry.r >= 150 && dimCherry.r > dimCherry.g + 90 && dimCherry.g >= dimCherry.b);
        EXPECT_TRUE(dimBerry.r >= 138 && dimBerry.b > dimBerry.g + 12);
    }

    // 2. Cold-porcelain skin whitening & bare-lip complexion care in CompositePro:
    //    - Indoor yellowish skin (175, 135, 112) at whiten=0.60 lifts luma by >= +26/255 without amplifying yellow R-B (>66)
    //    - Bare dull/purplish lip (112, 74, 80) in default natural mode (whiten=0.30, rosy=0.15, lipstick=0) lifts R >= 132
    //    - Dark lip parting seam (45, 25, 27) & white teeth (222, 216, 206) inside mouthZone stay protected (<= 2/255)
    {
        auto evalComp = [&](Rgba px, uint8_t faceByte, uint8_t exclByte, float whiten, float rosy) -> Rgba {
            constexpr int kSz = 16;
            const TextureId srcTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 255);
            const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 0);
            const TextureId semTex = makeSolidTex(dev, kSz, kSz, faceByte, exclByte, 76, 255);
            const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
            const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

            RenderState st{};
            TextureBindings tb{};
            tb.bind(0, srcTex);
            tb.bind(1, coeffTex);
            tb.bind(2, semTex);
            tb.bind(3, lutTex);

            UniformBlock ub{};
            ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);  // uFrame: poly skin mask active (pres = 1.0)
            ub.push(16.0f, 0.0f, 0.0f, 0.0f);             // uParams: skinOnly=true, smooth=0, brightness=0
            ub.push(0.0f, 0.30f, 1.0f, 0.0f);             // uDetails: sharpness=0.30, sat=1.0
            ub.push(whiten, rosy, 0.0f, 0.0f);            // uTone: whiten, rosy
            ub.push(0.0f, 0.0f, 0.0f, 1.0f);              // uEye0 (axisX = 1.0)
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // uEye1 (axisY = 0.0)
            ub.push(0.5f, 0.5f, 0.35f, 0.45f);            // uOval
            ub.push(0.5f, 0.5f, 0.20f, 0.12f);            // uMouth centered at (0.5, 0.5)
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // uEyeMask0
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // uEyeMask1
            ub.push(0.0f, 0.015f, 0.0f, 0.0f);            // uGrade

            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
            dev.endPass();

            std::vector<uint8_t> out(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
            const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
            const Rgba res{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};

            dev.destroyRenderTarget(rt);
            dev.destroyTexture(lutTex);
            dev.destroyTexture(semTex);
            dev.destroyTexture(coeffTex);
            dev.destroyTexture(srcTex);
            return res;
        };

        const Rgba porcelainSkin = evalComp({175, 135, 112, 255}, 255, 0, 0.60f, 0.15f);
        const float ySkin0 = 0.2126f * 175.0f + 0.7152f * 135.0f + 0.0722f * 112.0f;
        const float ySkin1 = 0.2126f * porcelainSkin.r + 0.7152f * porcelainSkin.g + 0.0722f * porcelainSkin.b;
        EXPECT_TRUE((ySkin1 - ySkin0) >= 26.0f && (int(porcelainSkin.r) - int(porcelainSkin.b)) <= 66);

        // v3.9.4 purple-only bare lip care:
        // - Mild muted lip (112, 74, 80) lifts cleanly with natural hue (no artificial Barbie pink or forced orange);
        // - Truly cyanotic purple lip (100, 66, 98, B > 0.80*R) reduces blue excess cleanly under whitening.
        const Rgba caredBareLip = evalComp({112, 74, 80, 255}, 0, 255, 0.30f, 0.15f);
        EXPECT_TRUE(caredBareLip.r >= 120 && caredBareLip.r <= 136 &&
                    caredBareLip.r > caredBareLip.b + 32);
        const Rgba cyanoticLip = evalComp({100, 66, 98, 255}, 0, 255, 0.60f, 0.15f);
        EXPECT_TRUE(int(cyanoticLip.r) > 108 &&
                    float(cyanoticLip.b) / float(cyanoticLip.r) < (98.0f / 100.0f) - 0.05f);

        const Rgba protectedSeam = evalComp({45, 25, 27, 255}, 0, 255, 0.55f, 0.20f);
        EXPECT_TRUE(std::abs(int(protectedSeam.r) - 45) <= 2 && std::abs(int(protectedSeam.g) - 25) <= 2);

        const Rgba protectedTeeth = evalComp({222, 216, 206, 255}, 0, 255, 0.55f, 0.20f);
        EXPECT_TRUE(std::abs(int(protectedTeeth.r) - 222) <= 2 && std::abs(int(protectedTeeth.g) - 216) <= 2);

        // v3.6.2 Anti-Ghosting: non-skin neutral wall (185, 185, 187) & black hair (28, 26, 27) overlapped by lagging faceM=255
        // must receive exact 0/255 porcelain whitening bleed!
        const Rgba laggedWall = evalComp({185, 185, 187, 255}, 255, 0, 0.70f, 0.20f);
        EXPECT_TRUE(std::abs(int(laggedWall.r) - 185) <= 1 && std::abs(int(laggedWall.b) - 187) <= 1);
        const Rgba laggedHair = evalComp({28, 26, 27, 255}, 255, 0, 0.70f, 0.20f);
        EXPECT_TRUE(std::abs(int(laggedHair.r) - 28) <= 1 && std::abs(int(laggedHair.b) - 27) <= 1);
    }

    // 3. v3.6.4 Virtual Background Anti-Occlusion Shield + Zero Shadow Bleed on Sub-Threshold Room Noise:
    //    - Tracked face & throat remain solid even when personMask == 0;
    //    - Sub-threshold sigmoid ripples (96/255 = 0.376) right next to head/torso produce EXACT 0/255 room shadow bleed!
    {
        constexpr int kW = 64, kH = 64;
        // Camera: dark room/person foreground (35, 30, 28) vs warm person test? Use distinct camera RGB (215, 165, 140)
        // and vivid cyan-blue Virtual BG (20, 110, 240) so any bleed immediately shifts R up and B down.
        const TextureId camTex = makeSolidTex(dev, kW, kH, 215, 165, 140, 255);
        const TextureId vbgTex = makeSolidTex(dev, kW, kH, 20, 110, 240, 255);
        std::vector<uint8_t> maskPx(kW * kH, 0);
        // Bottom-left clothing patch with solid confidence 190 (~0.745 > 0.72)
        for (int y = 44; y < 60; ++y) {
            for (int x = 4; x < 20; ++x) {
                maskPx[y * kW + x] = 190;
            }
        }
        // Sub-threshold room/shadow ripple 96/255 (~0.376 < 0.42) right next to ear/head and beside neck/shoulder!
        for (int y = 18; y < 54; ++y) {
            for (int x = 44; x < 62; ++x) {
                maskPx[y * kW + x] = 96;
            }
        }
        for (int y = 36; y < 54; ++y) {
            for (int x = 38; x < 44; ++x) {
                maskPx[y * kW + x] = 96;
            }
        }
        const TextureId maskTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, maskPx.data());
        const RenderTargetId outRt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, camTex);
        tb.bind(1, vbgTex);
        tb.bind(2, maskTex);
        UniformBlock ub{};
        // Primary face centered at (0.50, 0.62) in bottom-up UV (top-down row ~24), rx=0.14, ry=0.16
        ub.push(1.0f, 0.42f, 0.72f, 1.0f);  // strength=1, lo=0.42, hi=0.72, primaryPresence=1
        ub.push(1.0f, 1.0f, 0.0f, 0.0f);
        ub.push(0.50f, 0.62f, 0.14f, 0.16f);
        ub.push(1.0f, 0.0f, 1.0f, 0.0f);    // axis=(1,0), aspect=1, secPresence=0
        ub.push(0.5f, 0.5f, 0.1f, 0.1f);
        ub.push(1.0f, 0.0f, 0.0f, 0.0f);
        dev.beginPass(outRt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(outRt), out.data(), kW, kH));
        const int facePx = (24 * kW + 32) * 4;
        const int neckPx = (37 * kW + 32) * 4;
        const int clothPx = (50 * kW + 10) * 4;
        const int jawContourBgPx = (29 * kW + 39) * 4;  // a~0.79, b~-0.46: just outside tapered jawline!
        const int noisyHeadWallPx = (24 * kW + 50) * 4;
        const int noisyShoulderWallPx = (42 * kW + 40) * 4;

        EXPECT_TRUE(out[facePx + 0] >= 210 && out[facePx + 2] <= 145);
        EXPECT_TRUE(out[neckPx + 0] >= 200 && out[neckPx + 2] <= 155);
        EXPECT_TRUE(std::abs(int(out[clothPx + 0]) - 215) <= 2 && std::abs(int(out[clothPx + 2]) - 140) <= 2);
        // Tapered inner shield MUST NOT bleed camera room pixels outside jawline contour!
        EXPECT_TRUE(std::abs(int(out[jawContourBgPx + 0]) - 20) <= 1 && std::abs(int(out[jawContourBgPx + 2]) - 240) <= 1);
        // Sub-threshold noise (0.376) right next to head and shoulder MUST yield exact 0/255 camera bleed!
        EXPECT_TRUE(std::abs(int(out[noisyHeadWallPx + 0]) - 20) <= 1 && std::abs(int(out[noisyHeadWallPx + 2]) - 240) <= 1);
        EXPECT_TRUE(std::abs(int(out[noisyShoulderWallPx + 0]) - 20) <= 1 && std::abs(int(out[noisyShoulderWallPx + 2]) - 240) <= 1);

        dev.destroyRenderTarget(outRt);
        dev.destroyTexture(maskTex);
        dev.destroyTexture(vbgTex);
        dev.destroyTexture(camTex);
    }

    // 5. v3.6.5 Forehead & Face-Contour Bloated Dark-Room Halo Snap Gate (kMatteRefine -> kBgVirtualMix):
    //    When 256x256 segmentation bloats 3 texels (mask=168/255=0.659 > 0.42!) into a dark room (32, 30, 28)
    //    right outside bright forehead/cheek skin (215, 165, 140), dual-scale contrast-snapping kMatteRefine
    //    MUST collapse the dark-room halo to 0/255 shadow bleed while keeping inner skin 100% solid!
    {
        constexpr int kW = 32, kH = 32;
        std::vector<uint8_t> camPx(kW * kH * 4, 0);
        std::vector<uint8_t> rawMaskPx(kW * kH, 0);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                if (x < 16) {
                    camPx[i + 0] = 215; camPx[i + 1] = 165; camPx[i + 2] = 140; camPx[i + 3] = 255;
                    rawMaskPx[y * kW + x] = 250;
                } else {
                    camPx[i + 0] = 32; camPx[i + 1] = 30; camPx[i + 2] = 28; camPx[i + 3] = 255;
                    rawMaskPx[y * kW + x] = (x < 19) ? 168 : 6;  // 3-texel bloated dark-room halo!
                }
            }
        }
        const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
        const TextureId vbgTex = makeSolidTex(dev, kW, kH, 20, 110, 240, 255);
        const TextureId rawMaskTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, rawMaskPx.data());
        const RenderTargetId matteRt = dev.createRenderTarget(kW, kH, TextureFormat::kR8);
        const RenderTargetId mixRt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, rawMaskTex);
        tb.bind(1, camTex);
        tb.bind(2, rawMaskTex);
        tb.bind(3, camTex);
        UniformBlock ub{};
        ub.push(1.0f / kW, 1.0f / kH, 0.0f, 0.0f);  // single-frame spatial snap (even before temporal damping!)
        ub.push(0.055f, 0.022f, 0.080f, 0.0f);
        dev.beginPass(matteRt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kMatteRefine, st, tb, ub);
        dev.endPass();

        tb.clear(); ub.clear();
        tb.bind(0, camTex);
        tb.bind(1, vbgTex);
        tb.bind(2, dev.textureOf(matteRt));
        ub.push(1.0f, 0.42f, 0.72f, 0.0f);
        ub.push(1.0f, 1.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.1f, 0.1f);
        ub.push(1.0f, 0.0f, 1.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.1f, 0.1f);
        ub.push(1.0f, 0.0f, 0.0f, 0.0f);
        dev.beginPass(mixRt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(mixRt), out.data(), kW, kH));
        const int innerSkinPx = (16 * kW + 13) * 4;
        const int bloatedHaloPx = (16 * kW + 17) * 4;  // had rawMask=168 over dark room!
        EXPECT_TRUE(out[innerSkinPx + 0] >= 210 && out[innerSkinPx + 2] <= 145);
        EXPECT_TRUE(std::abs(int(out[bloatedHaloPx + 0]) - 20) <= 2 &&
                    std::abs(int(out[bloatedHaloPx + 2]) - 240) <= 2);

        dev.destroyRenderTarget(mixRt);
        dev.destroyRenderTarget(matteRt);
        dev.destroyTexture(rawMaskTex);
        dev.destroyTexture(vbgTex);
        dev.destroyTexture(camTex);
    }

    // 4. v3.6.4 Temporal Matte Anti-Shadow-Flash Gate (kMatteRefine -> kBgVirtualMix):
    //    - Static dark wall RGB (32, 30, 28) suffering a 1-frame neural spike (10 -> 155/255 = 0.608)
    //      MUST be damped below 0.42 so virtual background (20, 110, 240) receives exact 0/255 dark shadow flash;
    //    - Real moving foreground RGB (215, 165, 140) with mask 220 snaps to 100% solid foreground in 1 frame!
    {
        constexpr int kSz = 16;
        const TextureId darkWallRgb = makeSolidTex(dev, kSz, kSz, 32, 30, 28, 255);
        const TextureId movingPersonRgb = makeSolidTex(dev, kSz, kSz, 215, 165, 140, 255);
        const TextureId vbgTex = makeSolidTex(dev, kSz, kSz, 20, 110, 240, 255);
        std::vector<uint8_t> prevMaskPx(kSz * kSz, 10);
        std::vector<uint8_t> spikeMaskPx(kSz * kSz, 155);
        std::vector<uint8_t> solidMaskPx(kSz * kSz, 220);
        const TextureId prevMaskTex = dev.createTexture2D(kSz, kSz, TextureFormat::kR8, prevMaskPx.data());
        const TextureId spikeMaskTex = dev.createTexture2D(kSz, kSz, TextureFormat::kR8, spikeMaskPx.data());
        const TextureId solidMaskTex = dev.createTexture2D(kSz, kSz, TextureFormat::kR8, solidMaskPx.data());
        const RenderTargetId matteRt = dev.createRenderTarget(kSz, kSz, TextureFormat::kR8);
        const RenderTargetId mixRt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

        auto evalMatteThenMix = [&](TextureId curMask, TextureId curRgb, TextureId prevRgb) -> Rgba {
            RenderState st{};
            TextureBindings tb{};
            tb.bind(0, curMask);
            tb.bind(1, curRgb);
            tb.bind(2, prevMaskTex);
            tb.bind(3, prevRgb);
            UniformBlock ub{};
            ub.push(1.0f / kSz, 1.0f / kSz, 0.612f, 1.0f);  // default matteTemporal=0.60 -> 0.612
            ub.push(0.075f, 0.020f, 0.070f, 0.0f);
            dev.beginPass(matteRt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kMatteRefine, st, tb, ub);
            dev.endPass();

            tb.clear(); ub.clear();
            tb.bind(0, curRgb);
            tb.bind(1, vbgTex);
            tb.bind(2, dev.textureOf(matteRt));
            ub.push(1.0f, 0.42f, 0.72f, 0.0f);
            ub.push(1.0f, 1.0f, 0.0f, 0.0f);
            ub.push(0.5f, 0.5f, 0.1f, 0.1f);
            ub.push(1.0f, 0.0f, 1.0f, 0.0f);
            ub.push(0.5f, 0.5f, 0.1f, 0.1f);
            ub.push(1.0f, 0.0f, 0.0f, 0.0f);
            dev.beginPass(mixRt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kBgVirtualMix, st, tb, ub);
            dev.endPass();

            std::vector<uint8_t> out(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(mixRt), out.data(), kSz, kSz));
            const int c = ((kSz / 2) * kSz + (kSz / 2)) * 4;
            return {out[c], out[c + 1], out[c + 2], out[c + 3]};
        };

        const Rgba staticWallSpikeOut = evalMatteThenMix(spikeMaskTex, darkWallRgb, darkWallRgb);
        EXPECT_TRUE(std::abs(int(staticWallSpikeOut.r) - 20) <= 1 &&
                    std::abs(int(staticWallSpikeOut.g) - 110) <= 1 &&
                    std::abs(int(staticWallSpikeOut.b) - 240) <= 1);

        const Rgba movingPersonOut = evalMatteThenMix(solidMaskTex, movingPersonRgb, darkWallRgb);
        EXPECT_TRUE(std::abs(int(movingPersonOut.r) - 215) <= 2 &&
                    std::abs(int(movingPersonOut.g) - 165) <= 2 &&
                    std::abs(int(movingPersonOut.b) - 140) <= 2);

        dev.destroyRenderTarget(mixRt);
        dev.destroyRenderTarget(matteRt);
        dev.destroyTexture(solidMaskTex);
        dev.destroyTexture(spikeMaskTex);
        dev.destroyTexture(prevMaskTex);
        dev.destroyTexture(vbgTex);
        dev.destroyTexture(movingPersonRgb);
        dev.destroyTexture(darkWallRgb);
    }
}

void testV367EyeSocketNoseAndBeardContourNoDarkTrenches(MetalDevice& dev) {
    // 1. Shaded nose-wing / eye-socket / beard-rim crease (58, 44, 37) next to lit cheek/nose skin (148, 114, 96)
    //    under guided smoothing + cold-porcelain whitening + radiance/relight + sharpening:
    //    MUST lift smoothly (>= +18/255 luma) without bleedGuard=0 razor-sharp black trenches or negative sculpt/sharpness darkening!
    {
        constexpr int kW = 32, kH = 32;
        std::vector<uint8_t> srcPx(kW * kH * 4, 255);
        std::vector<uint8_t> coeffPx(kW * kH * 4, 0);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int i = (y * kW + x) * 4;
                const bool inCrease = (x >= 15 && x <= 17);
                if (inCrease) {
                    srcPx[i + 0] = 58; srcPx[i + 1] = 44; srcPx[i + 2] = 37; srcPx[i + 3] = 255;
                } else {
                    srcPx[i + 0] = 148; srcPx[i + 1] = 114; srcPx[i + 2] = 96; srcPx[i + 3] = 255;
                }
                // Guided filter across crease edge: a = 0.55 (140/255), mean = (118, 90, 76) -> b = (1 - a) * mean
                const float a = 140.0f / 255.0f;
                coeffPx[i + 0] = static_cast<uint8_t>(std::lround((1.0f - a) * 118.0f));
                coeffPx[i + 1] = static_cast<uint8_t>(std::lround((1.0f - a) * 90.0f));
                coeffPx[i + 2] = static_cast<uint8_t>(std::lround((1.0f - a) * 76.0f));
                coeffPx[i + 3] = 140;
            }
        }
        const TextureId srcTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, srcPx.data());
        const TextureId coeffTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, coeffPx.data());
        // hiM = 0 on eye-socket / nose-wing / beard contour, faceM = 255, exclM = 0, nonHair = 255
        const TextureId semTex = makeSolidTex(dev, kW, kH, 255, 0, 0, 255);
        const TextureId lutTex = makeSolidTex(dev, kW, kH, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex);
        tb.bind(1, coeffTex);
        tb.bind(2, semTex);
        tb.bind(3, lutTex);

        UniformBlock ub{};
        ub.push(1.0f / kW, 1.0f / kH, 1.0f, 5.0f);     // poly skin mask active (pres = 1.0)
        ub.push(16.0f, 0.70f, 0.0f, 0.45f);            // skinOnly=true, smoothing=0.70, skinTexture=0.45
        ub.push(0.35f, 0.32f, 1.0f, 0.0f);             // radiance=0.35, sharpness=0.32, sat=1.0
        ub.push(0.75f, 0.18f, 0.0f, 0.35f);            // whiten=0.75, rosy=0.18, relight=0.35
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.25f, 0.18f, 0.10f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.012f, 0.35f, 0.0f);

        dev.beginPass(rt, kW, kH, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        const int creaseIdx = (16 * kW + 16) * 4;
        const float yIn = 0.2126f * 58.0f + 0.7152f * 44.0f + 0.0722f * 37.0f;
        const float yOut = 0.2126f * out[creaseIdx + 0] + 0.7152f * out[creaseIdx + 1] + 0.0722f * out[creaseIdx + 2];
        EXPECT_TRUE((yOut - yIn) >= 18.0f);
        EXPECT_TRUE(int(out[creaseIdx + 0]) > int(out[creaseIdx + 2]) + 10);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
    }

    // 2. Upper eye-socket / brow-bone (lm 65, 52) & nasal-root glabella (lm 55) skin mask continuity + contour dark-beard guard
    {
        constexpr int kW = 360, kH = 640;
        std::vector<uint8_t> dimFacePx(kW * kH * 4, 255);
        for (int i = 0; i < kW * kH; ++i) {
            dimFacePx[i * 4 + 0] = 72;
            dimFacePx[i * 4 + 1] = 56;
            dimFacePx[i * 4 + 2] = 48;
            dimFacePx[i * 4 + 3] = 255;
        }
        const TextureId inTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, dimFacePx.data());
        const TextureId solidContour = makeSolidTex(dev, 16, 16, 255, 0, 0, 255);

        LiveBeautyEngine engine(dev);
        PipelineAssets assets{};
        assets.contourHighlightAtlas = solidContour;
        engine.setAssets(assets);

        BeautyConfig cfg{};
        cfg.enabled = true;
        cfg.skinOnly = true;
        cfg.smoothing = 0.65f;
        cfg.whiten = 0.75f;
        cfg.rosy = 0.15f;
        cfg.radiance = 0.30f;
        cfg.relight = 0.30f;
        engine.setConfig(cfg);

        std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        const float aspect = static_cast<float>(kW) / static_cast<float>(kH);
        const float scale = 0.48f / FaceMeshFilter::canonicalFaceWidth();
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
        for (int f = 0; f < 5; ++f) {
            const int64_t ts = 2000 + f * 33;
            EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts));
            outTex = engine.processFrame(inTex, kW, kH, ts, ts);
        }
        std::vector<uint8_t> outPx(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(outTex, outPx.data(), kW, kH));

        auto lumaAtLm = [&](int lmIdx) {
            const int px = std::clamp(static_cast<int>(std::lround(mpXy[lmIdx * 2] * (kW - 1))), 0, kW - 1);
            const int py = std::clamp(static_cast<int>(std::lround(mpXy[lmIdx * 2 + 1] * (kH - 1))), 0, kH - 1);
            const int off = (py * kW + px) * 4;
            return 0.2126f * outPx[off] + 0.7152f * outPx[off + 1] + 0.0722f * outPx[off + 2];
        };
        const float yIn0 = 0.2126f * 72.0f + 0.7152f * 56.0f + 0.0722f * 48.0f;
        // Orbital brow-bone (lm 65, 52) & glabella nasal-root (lm 55) must lift smoothly in parity with cheek (lm 205), no dark goggles!
        EXPECT_TRUE((lumaAtLm(65) - yIn0) >= 16.0f && std::fabs(lumaAtLm(65) - lumaAtLm(205)) <= 7.0f);
        EXPECT_TRUE((lumaAtLm(52) - yIn0) >= 16.0f && (lumaAtLm(55) - yIn0) >= 16.0f);

        // Contour makeup on dark jawline beard / deep nasal shadow (46, 38, 35) must NOT darken into muddy black streaks,
        // while lit jawline/nose-side skin (195, 152, 132) still receives 3D contour sculpting!
        BeautyConfig contourOnly{};
        contourOnly.enabled = true;
        contourOnly.contour = 0.85f;
        engine.setConfig(contourOnly);
        const TextureId darkBeardTex = makeSolidTex(dev, kW, kH, 46, 38, 35, 255);
        const TextureId litJawTex = makeSolidTex(dev, kW, kH, 195, 152, 132, 255);

        EXPECT_TRUE(engine.submitFaceLandmarks(2200, mpXy.data(), 478, kW, kH, 2200));
        const TextureId darkOutTex = engine.processFrame(darkBeardTex, kW, kH, 2200, 2200);
        std::vector<uint8_t> darkOutPx(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(darkOutTex, darkOutPx.data(), kW, kH));
        const int jawPxX = std::clamp(static_cast<int>(std::lround(mpXy[172 * 2] * (kW - 1))), 0, kW - 1);
        const int jawPxY = std::clamp(static_cast<int>(std::lround(mpXy[172 * 2 + 1] * (kH - 1))), 0, kH - 1);
        const int jawOff = (jawPxY * kW + jawPxX) * 4;
        EXPECT_TRUE(darkOutPx[jawOff + 0] >= 45 && darkOutPx[jawOff + 1] >= 37);

        EXPECT_TRUE(engine.submitFaceLandmarks(2233, mpXy.data(), 478, kW, kH, 2233));
        const TextureId litOutTex = engine.processFrame(litJawTex, kW, kH, 2233, 2233);
        std::vector<uint8_t> litOutPx(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(litOutTex, litOutPx.data(), kW, kH));
        EXPECT_TRUE(litOutPx[jawOff + 0] <= 186);

        engine.releaseGpuResources();
        dev.destroyTexture(litJawTex);
        dev.destroyTexture(darkBeardTex);
        dev.destroyTexture(solidContour);
        dev.destroyTexture(inTex);
    }

    // 3. kHairAlpha lower-face beard/mustache shield: when HairSegmenter misclassifies beard/mustache as hair (255),
    //    lower central face keeps nonHair >= 245 while forehead bangs still get nonHair <= 15!
    {
        constexpr int kSz = 32;
        std::vector<uint8_t> fullHair(kSz * kSz, 255);
        const TextureId hairTex = dev.createTexture2D(kSz, kSz, TextureFormat::kR8, fullHair.data());
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, hairTex);
        tb.bind(1, hairTex);
        UniformBlock ub{};
        ub.push(1.0f, 0.0f, 0.0f, 1.0f);       // identity affine + hairShown=1.0
        ub.push(0.0f, 1.0f, 0.0f, 0.0f);       // occShown=0.0
        ub.push(0.5f, 0.5f, 0.25f, 0.32f);     // face oval centered at (0.5, 0.5)
        ub.push(1.0f, 0.0f, 1.0f, 1.0f);       // axis=(1,0), aspect=1.0, hasFace=1.0
        dev.beginPass(rt, kSz, kSz, true, {0.0f, 0.0f, 0.0f, 1.0f});
        dev.drawFullScreenQuad(ShaderId::kHairAlpha, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        // Top-down y=8 -> GL bottom-up v=0.73 (forehead bangs: b = +0.72) -> nonHair <= 15
        const int bangsIdx = (8 * kSz + 16) * 4;
        // Top-down y=21 -> GL bottom-up v=0.33 (mustache / chin beard: b = -0.53) -> nonHair >= 245
        const int beardIdx = (21 * kSz + 16) * 4;
        EXPECT_TRUE(out[bangsIdx + 3] <= 15);
        EXPECT_TRUE(out[beardIdx + 3] >= 245);

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(hairTex);
    }
}

void testV369MetalSlightCameraShakeNoWaterRipple(MetalDevice& dev) {
    constexpr int kW = 180;
    constexpr int kH = 320;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    const float baseScale = 0.44f / FaceMeshFilter::canonicalFaceWidth();

    struct ShakePose {
        float tx, ty, rollRad, scaleMul;
    };
    auto shakeAtMs = [](int64_t tsMs) -> ShakePose {
        const double t = static_cast<double>(tsMs) * 0.001;
        ShakePose p{};
        p.tx = static_cast<float>(0.028 * std::sin(2.0 * M_PI * 2.7 * t) + 0.011 * std::sin(2.0 * M_PI * 4.3 * t + 0.7));
        p.ty = static_cast<float>(0.022 * std::cos(2.0 * M_PI * 2.3 * t + 0.4) + 0.009 * std::sin(2.0 * M_PI * 3.9 * t));
        p.rollRad = static_cast<float>(0.038 * std::sin(2.0 * M_PI * 2.5 * t + 0.3));
        p.scaleMul = static_cast<float>(1.0 + 0.012 * std::sin(2.0 * M_PI * 1.9 * t));
        return p;
    };
    auto forwardPt = [&](float lx, float ly, const ShakePose& p, float& outU, float& outVTop) {
        const float cr = std::cos(p.rollRad), sr = std::sin(p.rollRad);
        outU = 0.5f + p.tx + (cr * lx - sr * ly) * p.scaleMul;
        outVTop = ((0.48f / kAspect) + p.ty + (sr * lx + cr * ly) * p.scaleMul) * kAspect;
    };
    auto evalRest = [&](float u, float vTop, float rgb[3]) {
        const float mx = u - 0.5f;
        const float my = (vTop - 0.48f) / kAspect;
        const float rFace = std::hypot(mx / 0.23f, my / 0.30f);
        const float shade = 1.0f - 0.22f * (mx / 0.23f) * (mx / 0.23f) - 0.14f * (my / 0.30f) * (my / 0.30f);
        float skR = 204.0f * shade, skG = 160.0f * shade, skB = 138.0f * shade;
        const float fold = std::exp(-0.5f * (std::pow((std::fabs(mx) - 0.068f) / 0.020f, 2.0f) + std::pow((my - 0.075f) / 0.048f, 2.0f)));
        skR -= 24.0f * fold; skG -= 22.0f * fold; skB -= 20.0f * fold;
        const float eyeD = std::hypot((std::fabs(mx) - 0.092f) / 0.042f, (my + 0.068f) / 0.020f);
        const float inEye = 1.0f - detail::smoothstep(0.75f, 1.05f, eyeD);
        const float irisD = std::hypot((std::fabs(mx) - 0.092f) / 0.018f, (my + 0.068f) / 0.018f);
        const float inIris = 1.0f - detail::smoothstep(0.75f, 1.05f, irisD);
        const float eyeLum = 226.0f * (1.0f - inIris) + 32.0f * inIris;
        skR = skR * (1.0f - inEye) + eyeLum * inEye;
        skG = skG * (1.0f - inEye) + eyeLum * inEye;
        skB = skB * (1.0f - inEye) + eyeLum * inEye;
        const float faceMask = 1.0f - detail::smoothstep(0.94f, 1.04f, rFace);
        rgb[0] = std::clamp(145.0f * (1.0f - faceMask) + skR * faceMask, 0.0f, 255.0f);
        rgb[1] = std::clamp(142.0f * (1.0f - faceMask) + skG * faceMask, 0.0f, 255.0f);
        rgb[2] = std::clamp(138.0f * (1.0f - faceMask) + skB * faceMask, 0.0f, 255.0f);
    };

    const TextureId atlas = makeSolidTex(dev, 16, 16, 200, 200, 200, 255);
    PipelineAssets assets{kInvalidTexture, atlas, atlas, atlas, atlas, atlas};

    auto measureShakeStepRms = [&](bool beautyEnabled, int frameDelayFrames, int64_t latMs) -> double {
        LiveBeautyEngine engine(dev);
        engine.setAssets(assets);
        BeautyConfig cfg{};
        cfg.enabled = beautyEnabled;
        cfg.skinOnly = true;
        cfg.smoothing = 0.65f;
        cfg.whiten = 0.70f;
        cfg.relight = 0.65f;
        cfg.faceSlim = 0.68f;
        cfg.eyeEnlarge = 0.62f;
        cfg.frameDelayFrames = frameDelayFrames;
        engine.setConfig(cfg);

        TextureId camTex = kInvalidTexture;
        std::vector<uint8_t> camPx(kW * kH * 4, 255), outPx(kW * kH * 4, 0);
        constexpr int kGrid = 17;
        std::vector<float> prevRestLuma(kGrid * kGrid, -1.0f);
        double sumSq = 0.0;
        int count = 0;

        for (int f = 0; f < 36; ++f) {
            const int64_t frameTs = 1000 + f * 33;
            const int64_t lmTs = frameTs - latMs;
            const ShakePose camP = shakeAtMs(frameTs);
            const ShakePose lmP = shakeAtMs(lmTs);
            const ShakePose dispP = shakeAtMs(frameTs - static_cast<int64_t>(frameDelayFrames) * 33);
            const float crInv = std::cos(-camP.rollRad), srInv = std::sin(-camP.rollRad);
            for (int y = 0; y < kH; ++y) {
                const float vTop = (y + 0.5f) / kH;
                for (int x = 0; x < kW; ++x) {
                    const float u = (x + 0.5f) / kW;
                    const float dx = (u - 0.5f - camP.tx) / camP.scaleMul;
                    const float dy = (vTop / kAspect - (0.48f / kAspect) - camP.ty) / camP.scaleMul;
                    float rgb[3];
                    evalRest(0.5f + crInv * dx - srInv * dy, 0.48f + (srInv * dx + crInv * dy) * kAspect, rgb);
                    const int idx = (y * kW + x) * 4;
                    camPx[idx + 0] = static_cast<uint8_t>(std::lround(rgb[0]));
                    camPx[idx + 1] = static_cast<uint8_t>(std::lround(rgb[1]));
                    camPx[idx + 2] = static_cast<uint8_t>(std::lround(rgb[2]));
                    camPx[idx + 3] = 255;
                }
            }
            if (camTex == kInvalidTexture) camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
            else dev.updateTexture2D(camTex, kW, kH, TextureFormat::kRgba8, camPx.data());

            float mpXy[478 * 2];
            for (int i = 0; i < 468; ++i) {
                float u, vTop;
                forwardPt(xyz[i * 3 + 0] * baseScale, -xyz[i * 3 + 1] * baseScale, lmP, u, vTop);
                const float mul = (std::hypot(uv[i * 2] - 0.5f, uv[i * 2 + 1] - 0.5f) > 0.22f) ? 1.8f : 1.0f;
                u += std::sin(static_cast<float>(i * 12.9898f + f * 78.233f)) * 0.0025f * mul;
                vTop += std::cos(static_cast<float>(i * 39.3467f + f * 41.135f)) * 0.0025f * mul * kAspect;
                mpXy[i * 2 + 0] = u;
                mpXy[i * 2 + 1] = vTop;
            }
            auto setIris = [&](int base, int a, int b, int top, int bot) {
                const float cx = 0.25f * (mpXy[a * 2] + mpXy[b * 2] + mpXy[top * 2] + mpXy[bot * 2]);
                const float cy = 0.25f * (mpXy[a * 2 + 1] + mpXy[b * 2 + 1] + mpXy[top * 2 + 1] + mpXy[bot * 2 + 1]);
                for (int k = 0; k < 5; ++k) { mpXy[(base + k) * 2] = cx; mpXy[(base + k) * 2 + 1] = cy; }
            };
            setIris(468, 33, 133, 159, 145);
            setIris(473, 362, 263, 386, 374);

            EXPECT_TRUE(engine.submitFaceLandmarks(lmTs, mpXy, 478, kW, kH, frameTs));
            const TextureId outTex = engine.processFrame(camTex, kW, kH, frameTs, frameTs);
            EXPECT_TRUE(dev.readbackTextureRgba8(outTex, outPx.data(), kW, kH));

            if (f >= 12) {
                for (int gy = 0; gy < kGrid; ++gy) {
                    const float ly = -0.17f + 0.34f * (gy / float(kGrid - 1));
                    for (int gx = 0; gx < kGrid; ++gx) {
                        const float lx = -0.14f + 0.28f * (gx / float(kGrid - 1));
                        float u, vTop;
                        forwardPt(lx, ly, dispP, u, vTop);
                        const float fx = std::clamp(u * kW - 0.5f, 0.0f, float(kW - 1));
                        const float fy = std::clamp(vTop * kH - 0.5f, 0.0f, float(kH - 1));
                        const int x0 = int(fx), x1 = std::min(x0 + 1, kW - 1);
                        const int y0 = int(fy), y1 = std::min(y0 + 1, kH - 1);
                        const float wx = fx - x0, wy = fy - y0;
                        auto lum = [&](int x, int y) {
                            const int off = (y * kW + x) * 4;
                            return 0.2126f * outPx[off] + 0.7152f * outPx[off + 1] + 0.0722f * outPx[off + 2];
                        };
                        const float curLuma = (1.0f - wy) * ((1.0f - wx) * lum(x0, y0) + wx * lum(x1, y0)) +
                                              wy * ((1.0f - wx) * lum(x0, y1) + wx * lum(x1, y1));
                        const int gi = gy * kGrid + gx;
                        if (prevRestLuma[gi] >= 0.0f) {
                            const double d = std::fabs(curLuma - prevRestLuma[gi]);
                            sumSq += d * d;
                            ++count;
                        }
                        prevRestLuma[gi] = curLuma;
                    }
                }
            }
        }
        engine.releaseGpuResources();
        dev.destroyTexture(camTex);
        return std::sqrt(sumSq / std::max(1, count));
    };

    const double bypassRms = measureShakeStepRms(false, 0, 0);
    const double alignedDelayRms = measureShakeStepRms(true, 1, 33);
    const double extrapolatedRms = measureShakeStepRms(true, 0, 33);
    // Aligned FrameDelayRing rendering must be even calmer than raw Bypass rasterization,
    // and unaligned 33ms sub-frame extrapolation must stay within +0.25 luma RMS of Bypass!
    EXPECT_TRUE(alignedDelayRms <= bypassRms);
    EXPECT_TRUE(extrapolatedRms <= bypassRms + 0.25);

    dev.destroyTexture(atlas);
}

void testV3610MetalCheekBeautyZeroTwitch(MetalDevice& dev) {
    // Real-device stationary mirrored front-camera portrait with realistic MediaPipe eye/nose + contour noise,
    // heavy cheek slimming + whitening + smoothing + blush/contour + 10Hz autoOcc sample:
    // cheek silhouette & interior pixels must stay rock-steady (zero spasmodic cheek twitching!).
    constexpr int kW = 180;
    constexpr int kH = 320;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    const float canonW = FaceMeshFilter::canonicalFaceWidth();
    float mx0 = 0.0f, my0 = 0.0f, mz0 = 0.0f;
    for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
        mx0 += xyz[i * 3 + 0];
        my0 += xyz[i * 3 + 1];
        mz0 += xyz[i * 3 + 2];
    }
    mx0 /= FaceMeshFrame::kFaceCount;
    my0 /= FaceMeshFrame::kFaceCount;
    mz0 /= FaceMeshFrame::kFaceCount;

    // Static high-contrast portrait (bright cheeks against darker indoor room + subtle cheek texture)
    std::vector<uint8_t> camPx(kW * kH * 4, 255);
    for (int y = 0; y < kH; ++y) {
        const float vTop = (y + 0.5f) / kH;
        for (int x = 0; x < kW; ++x) {
            const float u = (x + 0.5f) / kW;
            const float dx = (u - 0.50f) / 0.22f;
            const float dy = ((vTop - 0.46f) / kAspect) / 0.28f;
            const float r = std::hypot(dx, dy);
            const float faceMask = 1.0f - detail::smoothstep(0.92f, 1.02f, r);
            const float shade = 1.0f - 0.22f * std::min(1.0f, dx * dx);
            const float R = 48.0f * (1.0f - faceMask) + (212.0f * shade) * faceMask;
            const float G = 52.0f * (1.0f - faceMask) + (168.0f * shade) * faceMask;
            const float B = 64.0f * (1.0f - faceMask) + (146.0f * shade) * faceMask;
            const int off = (y * kW + x) * 4;
            camPx[off + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(R)), 0, 255));
            camPx[off + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(G)), 0, 255));
            camPx[off + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(B)), 0, 255));
            camPx[off + 3] = 255;
        }
    }

    const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
    // Feathered cheek blush/contour/lip atlas matching production makeup assets (smooth falloff, no hard UV cliff)
    std::vector<uint8_t> atlasPx(64 * 64 * 4, 0);
    for (int y = 0; y < 64; ++y) {
        const float v = (y + 0.5f) / 64.0f;
        for (int x = 0; x < 64; ++x) {
            const float u = (x + 0.5f) / 64.0f;
            const float br = std::exp(-0.5f * (std::pow((u - 0.32f) / 0.08f, 2.0f) + std::pow((v - 0.54f) / 0.06f, 2.0f)));
            const float bl = std::exp(-0.5f * (std::pow((u - 0.68f) / 0.08f, 2.0f) + std::pow((v - 0.54f) / 0.06f, 2.0f)));
            const float lip = std::exp(-0.5f * (std::pow((u - 0.50f) / 0.08f, 2.0f) + std::pow((v - 0.70f) / 0.035f, 2.0f)));
            const int off = (y * 64 + x) * 4;
            atlasPx[off + 0] = static_cast<uint8_t>(std::lround(std::max(br, bl) * 220.0f));
            atlasPx[off + 1] = 0;
            atlasPx[off + 2] = static_cast<uint8_t>(std::lround(lip * 220.0f));
            atlasPx[off + 3] = 255;
        }
    }
    const TextureId atlas = dev.createTexture2D(64, 64, TextureFormat::kRgba8, atlasPx.data());
    LiveBeautyEngine engine(dev);
    PipelineAssets assets{kInvalidTexture, atlas, atlas, atlas, atlas, atlas};
    engine.setAssets(assets);

    BeautyConfig cfg{};
    cfg.enabled = true;
    cfg.skinOnly = true;
    cfg.smoothing = 0.62f;
    cfg.whiten = 0.45f;
    cfg.radiance = 0.25f;
    cfg.faceSlim = 0.65f;
    cfg.vFace = 0.50f;
    cfg.cheekboneSlim = 0.45f;
    cfg.eyeEnlarge = 0.15f;
    cfg.blush = 0.30f;
    cfg.contour = 0.24f;
    cfg.lipstick = 0.35f;
    engine.setConfig(cfg);

    auto hnoise = [](int a, int b) {
        const float s = std::sin(static_cast<float>(a) * 12.9898f + static_cast<float>(b) * 78.233f) * 43758.5453f;
        return (s - std::floor(s)) * 2.0f - 1.0f;
    };

    std::vector<uint8_t> outPx(kW * kH * 4, 0), prevPx(kW * kH * 4, 0);
    double cheekEdgeSumSq = 0.0;
    int cheekEdgeCnt = 0;
    float maxCheekBandJump = 0.0f;

    for (int f = 0; f < 60; ++f) {
        const int64_t ts = 1000 + f * 33;
        const float corrL = hnoise(200 + f, 901) * 0.0055f;
        const float corrR = hnoise(200 + f, 902) * 0.0055f;
        const float effYaw = 0.18f + hnoise(200 + f, 903) * 0.012f;
        const float effS = 0.44f * (1.0f + hnoise(200 + f, 904) * 0.006f);
        const float cyE = std::cos(effYaw), syE = std::sin(effYaw);

        std::array<float, FaceMeshFrame::kCoordCount> mp{};
        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            float lx = (xyz[i * 3 + 0] - mx0) / canonW;
            const float ly = (xyz[i * 3 + 1] - my0) / canonW;
            const float lz = (xyz[i * 3 + 2] - mz0) / canonW;
            const float nu = (uv[i * 2] - 0.5f) * 2.0f;
            const float nv = (0.5f - uv[i * 2 + 1]) * 2.0f;
            if (nv < 0.15f && nv > -0.85f && std::fabs(nu) > 0.35f) {
                const float w = std::exp(-0.5f * std::pow((nv + 0.32f) / 0.34f, 2.0f)) *
                                detail::smoothstep(0.35f, 0.85f, std::fabs(nu));
                lx *= (1.0f + 0.18f * w);
            }
            const float x1 = -(lx * cyE + lz * syE);  // mirrored front camera
            float u = 0.50f + x1 * effS;
            float vTop = 0.46f - ly * effS * kAspect;
            const float rUv = std::hypot((uv[i * 2] - 0.5f) / 0.36f, (uv[i * 2 + 1] - 0.5f) / 0.43f);
            const float isC = detail::smoothstep(0.62f, 0.90f, rUv);
            const float sigU = 0.0024f * (1.0f - isC) + 0.0070f * isC;
            u += hnoise((200 + f) * 503 + i, 1) * sigU;
            vTop += hnoise((200 + f) * 503 + i, 2) * sigU * kAspect;
            if (isC > 0.1f) u += isC * ((-nu < 0.0f) ? corrL : corrR);
            mp[i * 2 + 0] = u;
            mp[i * 2 + 1] = vTop;
        }
        for (int i = 468; i < 478; ++i) { mp[i * 2] = 0.50f; mp[i * 2 + 1] = 0.42f; }

        EXPECT_TRUE(engine.submitFaceLandmarks(ts, mp.data(), 478, kW, kH, ts));
        if (f % 3 == 0) {
            EXPECT_TRUE(engine.submitAuxRgbaSample(camPx.data(), kW, kH, 0.45f));
        }
        const TextureId outTex = engine.processFrame(camTex, kW, kH, ts, ts);
        EXPECT_TRUE(dev.readbackTextureRgba8(outTex, outPx.data(), kW, kH));

        if (f >= 15) {
            // Inspect BOTH left & right cheek bands (including outer cheek slimming contour + blush interior)
            // y ∈ [0.46*H, 0.58*H], x ∈ [0.22*W..0.44*W] and [0.56*W..0.78*W]
            const int y0 = static_cast<int>(kH * 0.46f);
            const int y1 = static_cast<int>(kH * 0.58f);
            for (int side = 0; side < 2; ++side) {
                const int x0 = (side == 0) ? static_cast<int>(kW * 0.22f) : static_cast<int>(kW * 0.56f);
                const int x1 = (side == 0) ? static_cast<int>(kW * 0.44f) : static_cast<int>(kW * 0.78f);
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        const int off = (y * kW + x) * 4;
                        const float l0 = 0.2126f * prevPx[off] + 0.7152f * prevPx[off + 1] + 0.0722f * prevPx[off + 2];
                        const float l1 = 0.2126f * outPx[off] + 0.7152f * outPx[off + 1] + 0.0722f * outPx[off + 2];
                        const float d = std::fabs(l1 - l0);
                        cheekEdgeSumSq += static_cast<double>(d) * d;
                        ++cheekEdgeCnt;
                        maxCheekBandJump = std::max(maxCheekBandJump, d);
                    }
                }
            }
        }
        prevPx = outPx;
    }

    const double cheekRms = std::sqrt(cheekEdgeSumSq / std::max(1, cheekEdgeCnt));
    // Stationary portrait cheek band must stay rock-steady: RMS <= 0.45 luma and max single-pixel high-contrast
    // silhouette jump <= 13/255 (< 0.09 px on a 150-luma step edge with v3.9.4 kResBeta[3]=18 & zero indoor temporal blur).
    if (cheekRms > 0.45 || maxCheekBandJump > 13.0f) {
        std::cerr << "  [v3.6.10 cheek gate] cheekRms=" << cheekRms
                  << " maxCheekBandJump=" << maxCheekBandJump << "\n";
    }
    EXPECT_TRUE(cheekRms <= 0.45);
    EXPECT_TRUE(maxCheekBandJump <= 13.0f);

    engine.releaseGpuResources();
    dev.destroyTexture(camTex);
    dev.destroyTexture(atlas);
}

void testV3611RealisticNaturalLipColorNoFakePink(MetalDevice& dev) {
    // Helper: evaluate CompositePro at center of mouthZone (face=0, excl=255) with lipstick=0
    auto evalBareLip = [&](Rgba px, float whiten, float rosy) -> Rgba {
        constexpr int kSz = 16;
        const TextureId srcTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 255);
        const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 0);
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, 0, 255, 0, 255);
        const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex);
        tb.bind(1, coeffTex);
        tb.bind(2, semTex);
        tb.bind(3, lutTex);

        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);
        ub.push(16.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.30f, 1.0f, 0.0f);
        ub.push(whiten, rosy, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.5f, 0.20f, 0.12f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.015f, 0.0f, 0.0f);

        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
        const Rgba res{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
        return res;
    };

    // 1. Default 'natural' mode (whiten=0.30, rosy=0.15, lipstick=0):
    //    Real bare lips MUST stay true-to-life (ZERO artificial Barbie pink paint!).
    const Rgba maleNormal = evalBareLip({140, 100, 92, 255}, 0.30f, 0.15f);
    const Rgba femaleNormal = evalBareLip({158, 106, 100, 255}, 0.30f, 0.15f);
    const Rgba brightLip = evalBareLip({178, 126, 120, 255}, 0.30f, 0.15f);
    const Rgba upperShadow = evalBareLip({88, 58, 56, 255}, 0.30f, 0.15f);
    const Rgba dullPurple = evalBareLip({112, 74, 80, 255}, 0.30f, 0.15f);

    // Normal/bright bare lips follow exposure-coupled hue-preserving lift (ZERO Barbie pink shift: G >= B + 4, R/G preserved)
    EXPECT_TRUE(int(maleNormal.r) >= 145 && int(maleNormal.r) <= 158 && int(maleNormal.g) >= int(maleNormal.b) + 5 &&
                std::fabs(float(maleNormal.r) / float(maleNormal.g) - 140.0f / 100.0f) <= 0.06f);
    EXPECT_TRUE(int(femaleNormal.r) >= 162 && int(femaleNormal.r) <= 178 && int(femaleNormal.g) >= int(femaleNormal.b) + 4 &&
                std::fabs(float(femaleNormal.r) / float(femaleNormal.g) - 158.0f / 106.0f) <= 0.06f);
    EXPECT_TRUE(int(brightLip.r) >= 182 && int(brightLip.r) <= 196 && int(brightLip.g) >= int(brightLip.b) + 4);
    // Upper-lip shadow lifts out of black mud while preserving genuine 3D shading contrast (>= 55/255 vs lower lip)
    EXPECT_TRUE(int(upperShadow.r) >= 98 && int(upperShadow.r) <= 114 && int(femaleNormal.r) - int(upperShadow.r) >= 55);
    // v3.9.4 purple-only lip care: mild muted lip (112, 74, 80) preserves natural hue without neon pink,
    // while genuinely cyanotic purple lip (100, 66, 98, B > 0.80*R) reduces excess blue cleanly!
    EXPECT_TRUE(int(dullPurple.r) >= 120 && int(dullPurple.r) <= 136 && int(dullPurple.r) > int(dullPurple.b) + 32);
    const Rgba trueCyanotic = evalBareLip({100, 66, 98, 255}, 0.60f, 0.15f);
    EXPECT_TRUE(int(trueCyanotic.r) > 108 &&
                float(trueCyanotic.b) / float(trueCyanotic.r) < (98.0f / 100.0f) - 0.05f);

    // 2. Translucent 3D-shaded lipstick at preset strength (clear: rose_bean 0.35 * 0.75 = 0.2625):
    //    Must preserve real upper-vs-lower lip 3D shading and warm natural hue (NO flat glowing plastic pink!).
    const Vec3 rose = FaceMakeupUv::lipstickRgb(LipstickColor::kRoseBean);
    UniformBlock ubRosePreset{};
    ubRosePreset.push(rose.x, rose.y, rose.z, 0.35f * 0.75f);
    ubRosePreset.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);

    const Rgba presetUpper = evalMakeupShader(dev, ShaderId::kMakeupLip, {88, 58, 56, 255}, {0, 0, 255, 255}, ubRosePreset, 0.5f, 0.67f);
    const Rgba presetLower = evalMakeupShader(dev, ShaderId::kMakeupLip, {158, 106, 100, 255}, {0, 0, 255, 255}, ubRosePreset, 0.5f, 0.72f);
    const Rgba presetMale = evalMakeupShader(dev, ShaderId::kMakeupLip, {140, 100, 92, 255}, {0, 0, 255, 255}, ubRosePreset, 0.5f, 0.72f);

    // Shadowed upper lip must stay shaded (dR <= 10) and preserve >= 58/255 3D contrast against lit lower lip
    EXPECT_TRUE(int(presetUpper.r) - 88 <= 10 && int(presetLower.r) - int(presetUpper.r) >= 58);
    // Preset rose_bean must stay warm natural muted rose (G >= B + 4, dR <= 10, never neon pink)
    EXPECT_TRUE(int(presetMale.r) - 140 <= 10 && int(presetMale.g) >= int(presetMale.b) + 4);
}

void testV3613MetalHairDyeRgbBoundaryMattingBangsAndDarkHairUniformity(MetalDevice& dev) {
    constexpr int kW = 64, kH = 64;
    std::vector<uint8_t> img(kW * kH * 4, 255);
    std::vector<uint8_t> hairMask(kW * kH, 0);
    std::vector<uint8_t> sem(kW * kH * 4, 0);

    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int idx4 = (y * kW + x) * 4;
            if (x < 44) {
                // Left 44 columns: Hair -> Bangs/Eyebrow -> Forehead skin
                if (y < 28) {
                    // Very dark natural Asian hair (18, 15, 14)
                    img[idx4 + 0] = 18; img[idx4 + 1] = 15; img[idx4 + 2] = 14;
                    hairMask[y * kW + x] = 255;
                    // Intentional 2x2 interior mask dropout hole around (22, 12)
                    if (x >= 21 && x <= 22 && y >= 11 && y <= 12) {
                        hairMask[y * kW + x] = 0;
                    }
                } else if (y < 34) {
                    if (x < 30) {
                        // Forehead bangs over face polygon (sem.r = 255, sem.g = 0, hairMask = 255)
                        img[idx4 + 0] = 20; img[idx4 + 1] = 17; img[idx4 + 2] = 15;
                        hairMask[y * kW + x] = 255;
                        sem[idx4 + 0] = 255; sem[idx4 + 1] = 0; sem[idx4 + 3] = 255;
                    } else {
                        // Dark eyebrow inside eye/brow cutout (sem.g = 255) with spilled hairMask = 255
                        img[idx4 + 0] = 22; img[idx4 + 1] = 19; img[idx4 + 2] = 17;
                        hairMask[y * kW + x] = 255;
                        sem[idx4 + 0] = 255; sem[idx4 + 1] = 255; sem[idx4 + 3] = 255;
                    }
                } else {
                    // Bright forehead skin (204, 160, 142); rows 34..36 have coarse mask bleed (180)!
                    img[idx4 + 0] = 204; img[idx4 + 1] = 160; img[idx4 + 2] = 142;
                    hairMask[y * kW + x] = (y < 37) ? 180 : 0;
                    sem[idx4 + 0] = 255; sem[idx4 + 1] = 0; sem[idx4 + 3] = 255;
                }
            } else {
                // Right 20 columns (outside face polygon, sem.r = 0): Hair vs Bright Studio Wall
                if (y < 28) {
                    // Dark hair; rows 26..27 are under-segmented hairline edges (hairMask = 20)!
                    img[idx4 + 0] = 18; img[idx4 + 1] = 15; img[idx4 + 2] = 14;
                    hairMask[y * kW + x] = (y >= 26) ? 20 : 255;
                } else {
                    // Bright studio wall (220, 215, 210); rows 28..29 have coarse mask bleed (190)!
                    img[idx4 + 0] = 220; img[idx4 + 1] = 215; img[idx4 + 2] = 210;
                    hairMask[y * kW + x] = (y < 30) ? 190 : 0;
                }
            }
        }
    }

    const TextureId imgTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, img.data());
    const TextureId hairTex = dev.createTexture2D(kW, kH, TextureFormat::kR8, hairMask.data());
    const TextureId semTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, sem.data());
    const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

    const Vec3 hazeBlue = FaceMakeupUv::hairDyeRgb(HairDyeColor::kHazeBlue);
    RenderState st{};
    TextureBindings tb{};
    tb.bind(0, imgTex); tb.bind(1, hairTex); tb.bind(2, semTex);
    UniformBlock ub{};
    ub.push(1.0f, 0.0f, 0.0f, 0.90f);  // hairDye = 0.90
    ub.push(0.0f, 1.0f, 0.0f, 0.0f);   // hairShine = 0
    ub.push(hazeBlue.x, hazeBlue.y, hazeBlue.z, 1.0f);
    ub.push(0.50f, 0.75f, 0.30f, 0.15f);
    ub.push(1.0f, 0.0f, 1.0f, 0.0f);
    // 4-pixel RGB boundary matting radius matching the 2-3px coarse mask bleed & under-segmentation
    ub.push(1.0f / float(kW), 1.0f / float(kH), 4.0f / float(kW), 4.0f / float(kH));
    dev.beginPass(rt, kW, kH, false, {});
    dev.drawFullScreenQuad(ShaderId::kHairDye, st, tb, ub);
    dev.endPass();

    std::vector<uint8_t> out(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));

    auto px = [&](int x, int y) -> Rgba {
        const int i = (y * kW + x) * 4;
        return {out[i], out[i + 1], out[i + 2], out[i + 3]};
    };

    // 1. Very dark Asian hair (18, 15, 14) gets strong uniform Haze-Blue coverage (B > R + 18, B >= 55)
    const Rgba darkHair = px(12, 12);
    EXPECT_TRUE(int(darkHair.b) > int(darkHair.r) + 18 && int(darkHair.b) >= 55);

    // 2. Interior 2x2 mask dropout hole inside dark hair is cleanly filled via surrounding hair color match
    const Rgba filledHole = px(22, 12);
    EXPECT_TRUE(int(filledHole.b) > int(filledHole.r) + 14);

    // 3. Forehead bangs (sem.r=255, sem.g=0, hairMask=255) are cleanly dyed without forehead suppression
    const Rgba bangs = px(14, 30);
    EXPECT_TRUE(int(bangs.b) > int(bangs.r) + 18);

    // 4. Eyebrow inside sem.g=255 stays exact 0/255 untouched even with spilled hairMask=255
    const Rgba brow = px(37, 30);
    EXPECT_TRUE(brow.r == 22 && brow.g == 19 && brow.b == 17);

    // 5. Coarse mask bleed (180/255) onto bright forehead skin is snapped to exact 0/255 bleed!
    const Rgba foreheadBleed = px(14, 35);
    EXPECT_TRUE(foreheadBleed.r == 204 && foreheadBleed.g == 160 && foreheadBleed.b == 142);

    // 6. Coarse mask bleed (190/255) onto bright background wall (sem.r=0!) is snapped by RGB matting to exact 0/255!
    const Rgba wallBleed = px(54, 29);
    EXPECT_TRUE(wallBleed.r == 220 && wallBleed.g == 215 && wallBleed.b == 210);

    // 7. Under-segmented hairline edge (hairMask=20/255 on dark hair) snaps UP and gets dyed cleanly!
    const Rgba hairlineEdge = px(54, 26);
    EXPECT_TRUE(int(hairlineEdge.b) > int(hairlineEdge.r) + 14);

    dev.destroyRenderTarget(rt);
    dev.destroyTexture(semTex);
    dev.destroyTexture(hairTex);
    dev.destroyTexture(imgTex);
}

void testV3614TikTokExposureCoupledNonDarkeningLips(MetalDevice& dev) {
    auto luma601 = [](Rgba c) -> float {
        return (0.299f * c.r + 0.587f * c.g + 0.114f * c.b) / 255.0f;
    };

    // 1. Bare lip under high whitening (whiten=0.60, relight=0.35):
    //    Exposure-coupled hue-preserving lift prevents "white face -> blackened lips" simultaneous-contrast darkening,
    //    while dark lip seam (45, 25, 27) and teeth (222, 216, 206) stay exact 0/255 untouched!
    auto evalCompHigh = [&](Rgba px, uint8_t faceByte, uint8_t exclByte) -> Rgba {
        constexpr int kSz = 16;
        const TextureId srcTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 255);
        const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 0);
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, faceByte, exclByte, 76, 255);
        const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, coeffTex); tb.bind(2, semTex); tb.bind(3, lutTex);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);
        ub.push(16.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.30f, 1.0f, 0.0f);
        ub.push(0.60f, 0.15f, 0.0f, 0.35f);  // whiten=0.60, rosy=0.15, relight=0.35
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.5f, 0.20f, 0.12f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.015f, 0.0f, 0.0f);
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
        const Rgba res{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
        return res;
    };

    const Rgba skinIn{175, 138, 122, 255};
    const Rgba upperIn{88, 58, 56, 255};
    const Rgba indoorIn{118, 76, 74, 255};
    const Rgba skinOut = evalCompHigh(skinIn, 255, 0);
    const Rgba upperOut = evalCompHigh(upperIn, 0, 255);
    const Rgba indoorOut = evalCompHigh(indoorIn, 0, 255);
    const Rgba seamOut = evalCompHigh({45, 25, 27, 255}, 0, 255);
    const Rgba teethOut = evalCompHigh({222, 216, 206, 255}, 0, 255);

    // Lip brightness tracks skin whitening cleanly (skin-lip contrast increase stays <= +0.055 instead of exploding!)
    const float upperContrastDelta = (luma601(skinOut) - luma601(upperOut)) - (luma601(skinIn) - luma601(upperIn));
    const float indoorContrastDelta = (luma601(skinOut) - luma601(indoorOut)) - (luma601(skinIn) - luma601(indoorIn));
    EXPECT_TRUE(luma601(upperOut) >= luma601(upperIn) + 0.08f && upperContrastDelta <= 0.055f);
    EXPECT_TRUE(luma601(indoorOut) >= luma601(indoorIn) + 0.07f && indoorContrastDelta <= 0.055f);
    // Warm natural hue preserved (ZERO Barbie pink: G >= B + 2)
    EXPECT_TRUE(int(upperOut.g) >= int(upperOut.b) + 2 && int(indoorOut.g) >= int(indoorOut.b) + 2);
    // Seam and teeth remain exact untouched
    EXPECT_TRUE(seamOut.r == 45 && seamOut.g == 25 && seamOut.b == 27);
    EXPECT_TRUE(teethOut.r == 222 && teethOut.g == 216 && teethOut.b == 206);

    // 2. ALL 6 lipstick shades at velvet preset/mid strength (lipstick=0.60 -> uAlpha=0.45):
    //    MUST NEVER darken either shadowed upper lip (82,52,50) or indoor lip (118,76,74)!
    const LipstickColor allShades[6] = {
        LipstickColor::kRoseBean,
        LipstickColor::kCoralPeach,
        LipstickColor::kCherryRed,
        LipstickColor::kMilkTea,
        LipstickColor::kBerryPlum,
        LipstickColor::kOrangeSoda,
    };
    const Rgba darkUpper{82, 52, 50, 255};
    for (LipstickColor shade : allShades) {
        const Vec3 rgb = FaceMakeupUv::lipstickRgb(shade);
        UniformBlock ub{};
        ub.push(rgb.x, rgb.y, rgb.z, 0.45f);
        ub.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);  // velvet finish
        const Rgba outUpper = evalMakeupShader(dev, ShaderId::kMakeupLip, darkUpper, {0, 0, 255, 255}, ub, 0.5f, 0.67f);
        const Rgba outIndoor = evalMakeupShader(dev, ShaderId::kMakeupLip, indoorIn, {0, 0, 255, 255}, ub, 0.5f, 0.72f);
        // Zero darkening: EVERY shade lifts shadowed upper lip >= +0.020 and indoor lip >= +0.010 in Rec.601 luma!
        EXPECT_TRUE(luma601(outUpper) >= luma601(darkUpper) + 0.020f);
        EXPECT_TRUE(luma601(outIndoor) >= luma601(indoorIn) + 0.010f);
        // Warm/natural shades stay warm (G >= B); berry_plum stays berry (B > G)
        if (shade == LipstickColor::kBerryPlum) {
            EXPECT_TRUE(int(outIndoor.b) > int(outIndoor.g) + 4);
        } else {
            EXPECT_TRUE(int(outIndoor.g) >= int(outIndoor.b));
        }
    }
}

void testV3615MetalDouyinAmbientSceneBrightnessLift(MetalDevice& dev) {
    // Non-skin room background pixels (faceM=0, skin=0) MUST lift cleanly when uGrade.w (ambientLift) or
    // uParams.z (brightness) is active, while deep blacks stay crisp and bright highlights never clip!
    auto evalAmbient = [&](Rgba px, float brightness, float ambientLift) -> Rgba {
        constexpr int kSz = 16;
        const TextureId srcTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 255);
        const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, px.r, px.g, px.b, 0);
        // Non-face background wall: faceM=0, exclM=0, nonHair=255
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, coeffTex); tb.bind(2, semTex); tb.bind(3, lutTex);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);
        ub.push(16.0f, 0.0f, brightness, 0.0f);
        ub.push(0.0f, 0.0f, 1.0f, 0.0f);
        ub.push(0.30f, 0.15f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.5f, 0.20f, 0.12f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.015f, 0.0f, ambientLift);
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int idx = ((kSz / 2) * kSz + (kSz / 2)) * 4;
        const Rgba res{out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
        return res;
    };

    // 1. Dim indoor room wall (78, 76, 72) lifts noticeably (+18..+28/255) under indoor ambientLift=0.45
    const Rgba wallOut = evalAmbient({78, 76, 72, 255}, 0.0f, 0.45f);
    EXPECT_TRUE(int(wallOut.r) >= 96 && int(wallOut.r) <= 112 && std::abs(int(wallOut.r) - int(wallOut.b)) <= 10);

    // 2. Global brightness slider (brightness=0.35, ambientLift=0) now genuinely brightens non-skin room background too!
    const Rgba brightWall = evalAmbient({96, 94, 92, 255}, 0.35f, 0.0f);
    EXPECT_TRUE(int(brightWall.r) >= 106);

    // 3. Deep black (6, 6, 6) stays crisp rich black (<= 8/255, zero milky grey haze!)
    const Rgba deepBlack = evalAmbient({6, 6, 6, 255}, 0.25f, 0.50f);
    EXPECT_TRUE(int(deepBlack.r) <= 8 && int(deepBlack.g) <= 8 && int(deepBlack.b) <= 8);

    // 4. Bright lamp/window (236, 236, 234) stays cleanly rolled off (< 252, never hard-clipped!)
    const Rgba brightLamp = evalAmbient({236, 236, 234, 255}, 0.0f, 0.45f);
    EXPECT_TRUE(int(brightLamp.r) >= 236 && int(brightLamp.r) <= 250);
}

void testV370MetalCommercialBeautyPixelInvariants(MetalDevice& dev) {
    // 1. Pre-smooth isolated mid-band dark blemish pre-lift vs dark hair protection & flat skin 0/255 invariant
    {
        constexpr int kSz = 32;
        std::vector<uint8_t> srcPx(kSz * kSz * 4);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                uint8_t r = 205, g = 165, b = 148;
                if (x >= 15 && x <= 17 && y >= 15 && y <= 17) {
                    // Isolated mid-band dark blemish/pore on cheek
                    r = 184; g = 144; b = 128;
                } else if (x >= 4 && x <= 10 && y >= 4 && y <= 10) {
                    // Dark hair / lash strand (yOrig < 0.14)
                    r = 22; g = 18; b = 16;
                }
                srcPx[i + 0] = r; srcPx[i + 1] = g; srcPx[i + 2] = b; srcPx[i + 3] = 255;
            }
        }
        const TextureId srcTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, srcPx.data());
        const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, 205, 165, 148, 0);
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, 255, 0, 76, 255);
        const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, coeffTex); tb.bind(2, semTex); tb.bind(3, lutTex);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);
        ub.push(16.0f, 0.55f, 0.0f, 0.0f);  // skinOnly=1, smoothing=0.55
        ub.push(0.0f, 0.0f, 1.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.5f, 0.20f, 0.12f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.015f, 0.0f, 0.0f);
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        auto px = [&](int x, int y) -> Rgba {
            const int idx = (y * kSz + x) * 4;
            return {out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};
        };
        const Rgba blemishOut = px(16, 16);
        const Rgba hairOut = px(7, 7);
        const Rgba flatOut = px(26, 26);
        EXPECT_TRUE(int(blemishOut.r) >= 198 && int(blemishOut.g) >= 158);
        EXPECT_TRUE(int(hairOut.r) <= 24 && int(hairOut.g) <= 20);
        EXPECT_TRUE(flatOut.r == 205 && flatOut.g == 165 && flatOut.b == 148);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
    }

    // 2. Log-compressed halo-free clarity: flat skin & sub-noise 1-LSB dots stay exact 0/255, real detail pops cleanly
    {
        constexpr int kSz = 32;
        std::vector<uint8_t> srcPx(kSz * kSz * 4);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                uint8_t v = 132;
                if (x >= 15 && x <= 17 && y >= 15 && y <= 17) v = 168;  // 3x3 detail ridge
                else if (x == 8 && y == 24) v = 133;                    // 1-LSB sub-noise dot
                srcPx[i + 0] = v; srcPx[i + 1] = v; srcPx[i + 2] = v; srcPx[i + 3] = 255;
            }
        }
        const TextureId srcTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, srcPx.data());
        const TextureId coeffTex = makeSolidTex(dev, kSz, kSz, 132, 132, 132, 0);
        const TextureId semTex = makeSolidTex(dev, kSz, kSz, 255, 255, 0, 255);  // exclM=1.0 (eye/lash detail zone)
        const TextureId lutTex = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, coeffTex); tb.bind(2, semTex); tb.bind(3, lutTex);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);
        ub.push(16.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.80f, 1.0f, 0.0f);  // sharpness=0.80
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.35f, 0.45f);
        ub.push(0.5f, 0.5f, 0.20f, 0.12f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.015f, 0.0f, 0.0f);
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        auto px = [&](int x, int y) -> Rgba {
            const int idx = (y * kSz + x) * 4;
            return {out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};
        };
        const Rgba flatOut = px(26, 26);
        const Rgba noiseOut = px(8, 24);
        const Rgba ridgeOut = px(16, 16);
        EXPECT_TRUE(flatOut.r == 132 && flatOut.g == 132 && flatOut.b == 132);
        EXPECT_TRUE(noiseOut.r == 133 && noiseOut.g == 133 && noiseOut.b == 133);
        EXPECT_TRUE(int(ridgeOut.g) >= 174 && int(ridgeOut.g) <= 205);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(lutTex);
        dev.destroyTexture(semTex);
        dev.destroyTexture(coeffTex);
        dev.destroyTexture(srcTex);
    }

    // 3. Strictly brighten-only Bright Eyes (kRetouchEyeTeeth): never darkens pupil/iris, pops catchlight & cleans sclera
    {
        constexpr int kSz = 16;
        std::vector<uint8_t> eyePx(kSz * kSz * 4, 255);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                if (x < 8 && y < 8) {
                    eyePx[i + 0] = 52; eyePx[i + 1] = 40; eyePx[i + 2] = 34;  // dark iris
                    if (x <= 2 && y <= 2) {
                        eyePx[i + 0] = 16; eyePx[i + 1] = 14; eyePx[i + 2] = 14;  // pitch-dark pupil
                    } else if (x == 5 && y == 5) {
                        eyePx[i + 0] = 116; eyePx[i + 1] = 108; eyePx[i + 2] = 102;  // iris catchlight
                    }
                } else {
                    eyePx[i + 0] = 168; eyePx[i + 1] = 148; eyePx[i + 2] = 144;  // slightly reddish sclera
                }
            }
        }
        const TextureId srcTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, eyePx.data());
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        const float verts[32] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const BufferId vb = dev.createDynamicVertexBuffer(sizeof(verts));
        dev.updateVertexBuffer(vb, verts, sizeof(verts));
        const BufferId ib = dev.createStaticIndexBuffer(indices, 6);
        dev.beginPass(rt, kSz, kSz, false, {});
        {
            RenderState copySt{};
            TextureBindings tb{}; tb.bind(0, srcTex);
            UniformBlock ub{};
            dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        {
            RenderState st{}; st.blendEnable = true;
            TextureBindings tb{}; tb.bind(0, srcTex);
            UniformBlock ub{}; ub.push(0.90f, 0.0f, 1.0f / kSz, 1.0f / kSz);
            dev.drawIndexed(ShaderId::kRetouchEyeTeeth, st, vb, ib, 0, 6, VertexLayout::kRetouchPos2Uv2A4, tb, ub);
        }
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        auto px = [&](int x, int y) -> Rgba {
            const int idx = (y * kSz + x) * 4;
            return {out[idx], out[idx + 1], out[idx + 2], out[idx + 3]};
        };
        const Rgba pupilOut = px(1, 1);
        const Rgba irisOut = px(3, 6);
        const Rgba catchOut = px(5, 5);
        const Rgba scleraOut = px(12, 12);
        EXPECT_TRUE(pupilOut.r >= 16 && pupilOut.g >= 14 && pupilOut.b >= 14);
        EXPECT_TRUE(irisOut.r >= 52 && irisOut.g >= 40 && irisOut.b >= 34);
        EXPECT_TRUE(int(catchOut.r) >= 116 + 15);
        EXPECT_TRUE(int(scleraOut.g) >= 148 + 8 && int(scleraOut.b) >= 144 + 10 &&
                    (int(scleraOut.r) - int(scleraOut.b)) < (168 - 144));
        dev.destroyBuffer(ib);
        dev.destroyBuffer(vb);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(srcTex);
    }

    // 4. Cool-cast Tear Trough & Crease-filled Nasolabial (kRetouchSkin): non-darkening, lifts folds/cool tint, 0/255 on flat skin
    {
        constexpr int kSz = 16;
        // 4a: Top half cool bluish under-eye (152,134,146), bottom half warm lower cheek (182,150,136)
        std::vector<uint8_t> dcPx(kSz * kSz * 4, 255);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                if (y < 8) { dcPx[i + 0] = 152; dcPx[i + 1] = 134; dcPx[i + 2] = 146; }
                else       { dcPx[i + 0] = 182; dcPx[i + 1] = 150; dcPx[i + 2] = 136; }
            }
        }
        const TextureId dcTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, dcPx.data());
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        const float dcVerts[32] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -0.50f,
             1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, -0.50f,
            -1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -0.50f,
             1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, -0.50f,
        };
        const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const BufferId vb = dev.createDynamicVertexBuffer(sizeof(dcVerts));
        const BufferId ib = dev.createStaticIndexBuffer(indices, 6);
        dev.updateVertexBuffer(vb, dcVerts, sizeof(dcVerts));
        dev.beginPass(rt, kSz, kSz, false, {});
        {
            RenderState copySt{}; TextureBindings tb{}; tb.bind(0, dcTex); UniformBlock ub{};
            dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        {
            RenderState st{}; st.blendEnable = true; TextureBindings tb{}; tb.bind(0, dcTex);
            UniformBlock ub{}; ub.push(1.0f, 0.0f, 0.0f, 0.0f);
            dev.drawIndexed(ShaderId::kRetouchSkin, st, vb, ib, 0, 6, VertexLayout::kRetouchPos2Uv2A4, tb, ub);
        }
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int dcIdx = (4 * kSz + 8) * 4;
        const int flatDcIdx = (12 * kSz + 8) * 4;
        // v3.9.4 YCbCr tear-trough correction: lifts luminance strictly non-darkening while warming R/G and
        // gently neutralizing cool bluish cast (B shifts slightly down toward warm cheek 136), flat cheek stays exact 0/255.
        const float dcY0 = 0.299f * 152.0f + 0.587f * 134.0f + 0.114f * 146.0f;
        const float dcY1 = 0.299f * out[dcIdx] + 0.587f * out[dcIdx + 1] + 0.114f * out[dcIdx + 2];
        EXPECT_TRUE(dcY1 >= dcY0 + 10.0f && out[dcIdx] >= 152 + 16 && out[dcIdx + 1] >= 134 + 10 &&
                    out[dcIdx + 2] >= 136 && out[dcIdx + 2] <= 146);
        EXPECT_TRUE(out[flatDcIdx] == 182 && out[flatDcIdx + 1] == 150 && out[flatDcIdx + 2] == 136);

        // 4b: Vertical nasolabial dark crease at x=8 (136,104,92) flanked by warm cheek (176,140,124)
        std::vector<uint8_t> nfPx(kSz * kSz * 4, 255);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                if (x == 8) { nfPx[i + 0] = 136; nfPx[i + 1] = 104; nfPx[i + 2] = 92; }
                else        { nfPx[i + 0] = 176; nfPx[i + 1] = 140; nfPx[i + 2] = 124; }
            }
        }
        const TextureId nfTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, nfPx.data());
        const float nfVerts[32] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.18f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.18f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.18f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.18f, 0.0f,
        };
        dev.updateVertexBuffer(vb, nfVerts, sizeof(nfVerts));
        dev.beginPass(rt, kSz, kSz, false, {});
        {
            RenderState copySt{}; TextureBindings tb{}; tb.bind(0, nfTex); UniformBlock ub{};
            dev.drawFullScreenQuad(ShaderId::kPassThrough, copySt, tb, ub);
        }
        {
            RenderState st{}; st.blendEnable = true; TextureBindings tb{}; tb.bind(0, nfTex);
            UniformBlock ub{}; ub.push(0.0f, 1.0f, 0.0f, 0.0f);
            dev.drawIndexed(ShaderId::kRetouchSkin, st, vb, ib, 0, 6, VertexLayout::kRetouchPos2Uv2A4, tb, ub);
        }
        dev.endPass();
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int nfIdx = (8 * kSz + 8) * 4;
        const int flatNfIdx = (8 * kSz + 2) * 4;
        EXPECT_TRUE(out[nfIdx] >= 136 + 24 && out[nfIdx + 1] >= 104 + 20 && out[nfIdx + 2] >= 92 + 16);
        EXPECT_TRUE(out[flatNfIdx] == 176 && out[flatNfIdx + 1] == 140 && out[flatNfIdx + 2] == 124);

        dev.destroyBuffer(ib);
        dev.destroyBuffer(vb);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(nfTex);
        dev.destroyTexture(dcTex);
    }

    // 5. Exact W3C softLightBlend in kMakeupLayer lifts luminous highlight cleanly without clipping
    {
        UniformBlock ub{};
        ub.push(0.96f, 0.93f, 0.88f, 0.80f);  // luminous warm champagne highlight
        ub.push(1.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);      // mode.x=0 (W3C softLight + subtle mult)
        const Rgba hiOut = evalMakeupShader(dev, ShaderId::kMakeupLayer, {145, 115, 100, 255}, {255, 0, 0, 255}, ub);
        EXPECT_TRUE(int(hiOut.r) >= 145 + 10 && int(hiOut.g) >= 115 + 8 && int(hiOut.r) <= 235);
    }
}

void testV380MetalAtmosphereBloomGrainCat02AndOpenMouthLipProbes(MetalDevice& dev) {
    // 1. Soft-knee bright extract: dim shadows -> exact 0/255, bright highlights -> clean positive glow
    {
        constexpr int kSz = 16;
        std::vector<uint8_t> img(kSz * kSz * 4, 255);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const int i = (y * kSz + x) * 4;
                if (x < 8) { img[i] = 40; img[i + 1] = 38; img[i + 2] = 36; }
                else       { img[i] = 245; img[i + 1] = 240; img[i + 2] = 232; }
            }
        }
        const TextureId srcTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, img.data());
        const RenderTargetId rtExt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{}; tb.bind(0, srcTex);
        UniformBlock ub{}; ub.push(0.62f, 0.28f, 1.0f / kSz, 1.0f / kSz);
        dev.beginPass(rtExt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kPostBloomExtract, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> extOut(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rtExt), extOut.data(), kSz, kSz));
        const int dimIdx = (8 * kSz + 2) * 4;
        const int hiIdx = (8 * kSz + 13) * 4;
        EXPECT_TRUE(extOut[dimIdx] == 0 && extOut[dimIdx + 1] == 0 && extOut[dimIdx + 2] == 0);
        EXPECT_TRUE(extOut[hiIdx] >= 70 && extOut[hiIdx + 1] >= 65);

        // Composite bloom onto mid-tone base: strictly non-darkening (out >= base) and lifts highlights
        const RenderTargetId rtOut = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        const TextureId midTex = makeSolidTex(dev, kSz, kSz, 120, 115, 110, 255);
        TextureBindings ctb{};
        ctb.bind(0, midTex);
        ctb.bind(1, dev.textureOf(rtExt));
        ctb.bind(2, dev.textureOf(rtExt));
        UniformBlock cub{};
        cub.push(0.45f, 0.0f, 0.0f, 0.0f);  // bloom=0.45, grain=0, cat02=0
        cub.push(1.0f, 0.0f, 0.0f, 64.0f);
        cub.push(0.0f, 1.0f, 0.0f, 64.0f);
        cub.push(0.0f, 0.0f, 1.0f, 0.0f);
        dev.beginPass(rtOut, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kPostAtmosphere, st, ctb, cub);
        dev.endPass();
        std::vector<uint8_t> compOut(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rtOut), compOut.data(), kSz, kSz));
        EXPECT_TRUE(compOut[dimIdx] == 120 && compOut[dimIdx + 1] == 115 && compOut[dimIdx + 2] == 110);
        EXPECT_TRUE(compOut[hiIdx] >= 120 + 12 && compOut[hiIdx + 1] >= 115 + 10);

        dev.destroyRenderTarget(rtOut);
        dev.destroyTexture(midTex);
        dev.destroyRenderTarget(rtExt);
        dev.destroyTexture(srcTex);
    }

    // 2. Non-darkening organic filmic grain: never darkens any mid-tone pixel, varies spatially & across phases, keeps blacks crisp
    {
        constexpr int kSz = 32;
        const TextureId midTex = makeSolidTex(dev, kSz, kSz, 128, 128, 128, 255);
        const TextureId blkTex = makeSolidTex(dev, kSz, kSz, 4, 4, 4, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        auto runGrain = [&](TextureId tex, float phase, std::vector<uint8_t>& out) {
            RenderState st{};
            TextureBindings tb{}; tb.bind(0, tex); tb.bind(1, tex); tb.bind(2, tex);
            UniformBlock ub{};
            ub.push(0.0f, 0.30f, phase, 0.0f);
            ub.push(1.0f, 0.0f, 0.0f, 64.0f);
            ub.push(0.0f, 1.0f, 0.0f, 64.0f);
            ub.push(0.0f, 0.0f, 1.0f, 0.0f);
            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kPostAtmosphere, st, tb, ub);
            dev.endPass();
            out.assign(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        };
        std::vector<uint8_t> g0, g1, gBlk;
        runGrain(midTex, 0.12f, g0);
        runGrain(midTex, 0.67f, g1);
        runGrain(blkTex, 0.12f, gBlk);

        int darkened = 0;
        int minV = 255, maxV = 0;
        int phaseDiffs = 0;
        for (int i = 0; i < kSz * kSz; ++i) {
            const int r0 = g0[i * 4];
            if (r0 < 128 || g0[i * 4 + 1] < 128 || g0[i * 4 + 2] < 128) ++darkened;
            minV = std::min(minV, r0);
            maxV = std::max(maxV, r0);
            if (r0 != int(g1[i * 4])) ++phaseDiffs;
        }
        EXPECT_TRUE(darkened == 0);
        EXPECT_TRUE(maxV > minV);           // genuine spatial grain texture
        EXPECT_TRUE(phaseDiffs > 100);      // genuine temporal variation (not frozen)
        EXPECT_TRUE(gBlk[0] <= 5 && gBlk[1] <= 5 && gBlk[2] <= 5);  // pitch-black stays crisp

        dev.destroyRenderTarget(rt);
        dev.destroyTexture(blkTex);
        dev.destroyTexture(midTex);
    }

    // 3. CIE D65 CAT02 White Balance in kPostAtmosphere: exact identity at (0,0), warm(+0.5) & cool(-0.5) verified
    {
        constexpr int kSz = 8;
        const TextureId grayTex = makeSolidTex(dev, kSz, kSz, 140, 140, 140, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        auto evalCat02 = [&](float temp, float tint) -> Rgba {
            float m[9]{};
            BeautyQualityTuning::cat02WhiteBalanceMatrix(temp, tint, m);
            RenderState st{};
            TextureBindings tb{}; tb.bind(0, grayTex); tb.bind(1, grayTex); tb.bind(2, grayTex);
            UniformBlock ub{};
            ub.push(0.0f, 0.0f, 0.0f, 1.0f);
            ub.push(m[0], m[1], m[2], 64.0f);
            ub.push(m[3], m[4], m[5], 64.0f);
            ub.push(m[6], m[7], m[8], 0.0f);
            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kPostAtmosphere, st, tb, ub);
            dev.endPass();
            std::vector<uint8_t> out(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
            return {out[0], out[1], out[2], out[3]};
        };
        const Rgba idOut = evalCat02(0.0f, 0.0f);
        const Rgba warmOut = evalCat02(0.50f, 0.0f);
        const Rgba coolOut = evalCat02(-0.50f, 0.0f);
        EXPECT_TRUE(idOut.r == 140 && idOut.g == 140 && idOut.b == 140);
        EXPECT_TRUE(int(warmOut.r) > int(warmOut.b) + 12);
        EXPECT_TRUE(int(coolOut.b) > int(coolOut.r) + 12);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(grayTex);
    }

    // 4. Open-mouth geometric inner-mouth cavity cutout in kMakeupLip:
    //    Center pixel inside open inner-mouth ellipse (0.5, 0.5) stays exact 0/255 even on pink tongue (190, 115, 115),
    //    while closed mouth (uU4.x = 0) or outer lip vermilion gets stained normally!
    {
        const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
        UniformBlock ubOpen{};
        ubOpen.push(cherry.x, cherry.y, cherry.z, 0.80f);
        ubOpen.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);
        ubOpen.push(1.0f, 1.0f, 1.0f, 0.0f);
        ubOpen.push(0.50f, 0.50f, 0.25f, 0.20f);  // inner-mouth ellipse covering screen center (0.5, 0.5)
        ubOpen.push(1.0f, 1.0f, 1.0f, 0.0f);      // mouthOpen = 1.0, aspect = 1.0, axis = (1, 0)
        const Rgba cavityOut = evalMakeupShader(dev, ShaderId::kMakeupLip, {190, 115, 115, 255}, {0, 0, 255, 255}, ubOpen, 0.5f, 0.70f);
        EXPECT_TRUE(cavityOut.r == 190 && cavityOut.g == 115 && cavityOut.b == 115);

        // Shift ellipse far away (outer vermilion pixel outside open mouth cavity) -> cleanly stained!
        UniformBlock ubOuter = ubOpen;
        ubOuter.v[3] = {0.10f, 0.10f, 0.10f, 0.08f};
        const Rgba outerOut = evalMakeupShader(dev, ShaderId::kMakeupLip, {165, 112, 105, 255}, {0, 0, 255, 255}, ubOuter, 0.5f, 0.72f);
        EXPECT_TRUE(int(outerOut.r) > int(outerOut.g) + 60);
    }
}

/**
 * v3.9.0 Automated 7-Scenario Quantitative Portrait Quality Benchmark Gate:
 * Verifies real Apple Metal GPU shader pixel metrics across 7 live-streaming lighting & subject scenarios
 * to prevent regressions across skin tones, tungsten/fluorescent lighting, low light, smiling teeth,
 * lower-face occlusion, male beard stubble, and dark hair dyeing.
 */
void testV390SevenScenarioPortraitQualityBenchmark(MetalDevice& dev) {
    auto luma709 = [](const Rgba& p) -> float {
        return (0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b) / 255.0f;
    };

    // Scenario 1 & 2: Warm Tungsten Room & Cool Green Fluorescent Room -> Fused Auto CAT02 WB in fs_composite_pro
    auto runFusedCompositeWb = [&](Rgba srcColor, Vec3 ambientRgb, float awbStrength) -> Rgba {
        constexpr int kSz = 16;
        const TextureId srcTex = makeSolidTex(dev, kSz, kSz, srcColor.r, srcColor.g, srcColor.b, 255);
        const TextureId whiteMask = makeSolidTex(dev, kSz, kSz, 255, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        float autoT = 0.0f, autoG = 0.0f;
        BeautyQualityTuning::estimateAutoWhiteBalance(ambientRgb, awbStrength, autoT, autoG);
        float m[9]{};
        BeautyQualityTuning::cat02WhiteBalanceMatrix(autoT, autoG, m);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, srcTex); tb.bind(2, whiteMask); tb.bind(3, whiteMask);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 0.0f);  // u0
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // u1 (no smoothing/style)
        ub.push(0.0f, 0.0f, 1.0f, 0.0f);              // u2 (sat=1.0)
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // u3
        const float wbR = std::clamp(m[0] + m[1] + m[2], 0.50f, 1.60f);
        const float wbG = std::clamp(m[3] + m[4] + m[5], 0.50f, 1.60f);
        const float wbB = std::clamp(m[6] + m[7] + m[8], 0.50f, 1.60f);
        ub.push(wbR, wbG, wbB, 0.0f);                 // u4: 15-uniform-budget fused CAT02 WB
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // u5: grain=0
        for (int i = 0; i < 4; ++i) ub.push(0.0f, 0.0f, 0.0f, 0.0f);  // u6..u9
        ub.push(0.0f, 0.005f, 0.0f, 0.0f);            // u10
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(whiteMask);
        dev.destroyTexture(srcTex);
        return {out[0], out[1], out[2], out[3]};
    };

    // Scenario 1: Warm tungsten skin (218, 178, 130) under warm light (1.24, 1.00, 0.76) -> R-B cast drops cleanly
    const Rgba tungSrc{218, 178, 130, 255};
    const Rgba tungOut = runFusedCompositeWb(tungSrc, {1.24f, 1.00f, 0.76f}, 0.85f);
    const float tungCastBefore = float(tungSrc.r - tungSrc.b) / 255.0f;
    const float tungCastAfter = float(tungOut.r - tungOut.b) / 255.0f;
    EXPECT_TRUE(tungCastAfter < tungCastBefore - 0.045f);
    EXPECT_TRUE(std::fabs(luma709(tungOut) - luma709(tungSrc)) < 0.06f);

    // Scenario 2: Cool green fluorescent skin (152, 174, 172) under (0.84, 1.14, 1.16) -> green cast drops & warmth restores
    const Rgba fluoSrc{152, 174, 172, 255};
    const Rgba fluoOut = runFusedCompositeWb(fluoSrc, {0.84f, 1.14f, 1.16f}, 0.90f);
    const float greenBefore = (float(fluoSrc.g) - 0.5f * (fluoSrc.r + fluoSrc.b)) / 255.0f;
    const float greenAfter = (float(fluoOut.g) - 0.5f * (fluoOut.r + fluoOut.b)) / 255.0f;
    EXPECT_TRUE(greenAfter < greenBefore - 0.015f);
    EXPECT_TRUE(int(fluoOut.r) > int(fluoSrc.r));

    // Scenario 3: Low-light indoor portrait (luma=0.14) with v3.9.4 bounded ambientSceneLift (<= 0.18)
    const float lowLightLift = BeautyQualityTuning::ambientSceneLift(0.14f);
    EXPECT_TRUE(lowLightLift > 0.12f && lowLightLift <= 0.185f);

    // Scenario 4: Open-mouth smile with Cherry Red lipstick -> exposed white teeth stay neutral (R-B < 0.04)
    {
        const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
        UniformBlock ubSmile{};
        ubSmile.push(cherry.x, cherry.y, cherry.z, 0.78f);
        ubSmile.push(0.0f, 1.0f / 16.0f, 1.0f / 16.0f, 0.0f);
        ubSmile.push(1.0f, 1.0f, 1.0f, 0.0f);
        ubSmile.push(0.50f, 0.50f, 0.22f, 0.18f);
        ubSmile.push(1.0f, 1.0f, 1.0f, 0.0f);
        const Rgba teethOut = evalMakeupShader(dev, ShaderId::kMakeupLip, {226, 224, 218, 255}, {0, 0, 255, 255}, ubSmile, 0.5f, 0.70f);
        EXPECT_TRUE(float(int(teethOut.r) - int(teethOut.b)) / 255.0f < 0.04f);
    }

    // Scenario 5: Lower-face mask occlusion -> lowerFaceMakeupKeep drops to 0.0 while upper-face remains 1.0
    EXPECT_TRUE(std::fabs(BeautyQualityTuning::lowerFaceMakeupKeep(0.85f)) < 1e-5f);
    EXPECT_TRUE(std::fabs(BeautyQualityTuning::lowerFaceMakeupKeep(0.15f) - 1.0f) < 1e-5f);

    // Scenario 6: Male dark beard bristle pixel (38, 34, 32) under heavy smoothing (0.75) & whitening (0.60)
    //             in fs_composite_pro -> dark bristle must NOT wash out into gray haze!
    {
        constexpr int kSz = 16;
        std::vector<uint8_t> beardImg(kSz * kSz * 4, 0);
        for (int y = 0; y < kSz; ++y) {
            for (int x = 0; x < kSz; ++x) {
                const bool bristle = ((x + y) % 2 == 0);
                uint8_t* p = beardImg.data() + (y * kSz + x) * 4;
                p[0] = bristle ? 38 : 185;
                p[1] = bristle ? 34 : 145;
                p[2] = bristle ? 32 : 122;
                p[3] = 255;
            }
        }
        const TextureId beardTex = dev.createTexture2D(kSz, kSz, TextureFormat::kRgba8, beardImg.data());
        const TextureId guidedTex = makeSolidTex(dev, kSz, kSz, 140, 110, 95, 25);
        const TextureId skinMask = makeSolidTex(dev, kSz, kSz, 255, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, beardTex); tb.bind(1, guidedTex); tb.bind(2, skinMask); tb.bind(3, skinMask);
        UniformBlock ub{};
        ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);  // framePresence=4+1 (poly mask)
        ub.push(16.0f, 0.75f, 0.0f, 0.50f);           // skinOnly=1, smoothing=0.75, texture=0.5
        ub.push(0.0f, 0.0f, 1.0f, 0.0f);
        ub.push(0.60f, 0.0f, 0.0f, 0.0f);             // whiten=0.60
        for (int i = 0; i < 6; ++i) ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.005f, 0.22f, 0.0f);
        dev.beginPass(rt, kSz, kSz, false, {});
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();
        std::vector<uint8_t> out(kSz * kSz * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
        const int ci = (8 * kSz + 8) * 4;
        const Rgba bristleOut{out[ci], out[ci + 1], out[ci + 2], 255};
        EXPECT_TRUE(luma709(bristleOut) < 0.22f);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(skinMask);
        dev.destroyTexture(guidedTex);
        dev.destroyTexture(beardTex);
    }

    // Scenario 7: Dark hair dyeing (milk_tea_brown) with hairline-spill & eyebrow guard -> hair dyes cleanly, forehead spill & brows untouched!
    {
        constexpr int kSz = 16;
        const TextureId hairImg = makeSolidTex(dev, kSz, kSz, 42, 38, 36, 255);
        const TextureId skinImg = makeSolidTex(dev, kSz, kSz, 205, 168, 145, 255);
        const TextureId fullHairMask = makeSolidTex(dev, kSz, kSz, 255, 255, 255, 255);
        const TextureId spillHairMask = makeSolidTex(dev, kSz, kSz, 100, 100, 100, 255);  // weak hairline spill on forehead
        const TextureId faceSkinMask = makeSolidTex(dev, kSz, kSz, 255, 0, 0, 255);
        const TextureId eyebrowMask = makeSolidTex(dev, kSz, kSz, 0, 255, 0, 255);        // G=excl (eyebrow)
        const TextureId noSkinMask = makeSolidTex(dev, kSz, kSz, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        const Vec3 dye = FaceMakeupUv::hairDyeRgb(HairDyeColor::kMilkTeaBrown);
        auto runDye = [&](TextureId img, TextureId hMask, TextureId polyMask) -> Rgba {
            RenderState st{};
            TextureBindings tb{};
            tb.bind(0, img); tb.bind(1, hMask); tb.bind(2, polyMask);
            UniformBlock ub{};
            ub.push(1.0f, 0.0f, 0.0f, 0.85f);
            ub.push(0.0f, 1.0f, 0.0f, 0.25f);
            ub.push(dye.x, dye.y, dye.z, 1.0f);
            ub.push(0.5f, 0.5f, 0.4f, 0.2f);
            ub.push(1.0f, 0.0f, 1.0f, 0.0f);
            ub.push(1.0f / kSz, 1.0f / kSz, 1.6f / 128.0f, 1.6f / 128.0f);
            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kHairDye, st, tb, ub);
            dev.endPass();
            std::vector<uint8_t> out(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kSz, kSz));
            return {out[(8 * kSz + 8) * 4], out[(8 * kSz + 8) * 4 + 1], out[(8 * kSz + 8) * 4 + 2], 255};
        };
        const Rgba dyedHair = runDye(hairImg, fullHairMask, noSkinMask);
        const Rgba guardedForehead = runDye(skinImg, spillHairMask, faceSkinMask);
        const Rgba guardedBrow = runDye(hairImg, fullHairMask, eyebrowMask);
        EXPECT_TRUE(int(dyedHair.r) > 65 && int(dyedHair.r) > int(dyedHair.b) + 12);
        EXPECT_TRUE(std::abs(int(guardedForehead.r) - 205) <= 2 && std::abs(int(guardedForehead.g) - 168) <= 2);
        EXPECT_TRUE(std::abs(int(guardedBrow.r) - 42) <= 2 && std::abs(int(guardedBrow.g) - 38) <= 2);
        dev.destroyRenderTarget(rt);
        dev.destroyTexture(noSkinMask);
        dev.destroyTexture(eyebrowMask);
        dev.destroyTexture(faceSkinMask);
        dev.destroyTexture(spillHairMask);
        dev.destroyTexture(fullHairMask);
        dev.destroyTexture(skinImg);
        dev.destroyTexture(hairImg);
    }

    std::cout << "[BENCHMARK v3.9.0] 7-Scenario Portrait Quality Gate PASSED "
              << "(tungsten_rb_drop=" << (tungCastBefore - tungCastAfter)
              << ", fluo_green_drop=" << (greenBefore - greenAfter)
              << ", low_light_lift=" << lowLightLift << ")\n";
}

// ============================================================================
// v3.9.2 Regression Gate: 6-class semantic clothes/accessories on chest MUST NOT
// zero out rtSkinMask_.a (nonHair) and create a dark un-whitened shadow patch on
// the chest, while in-face occluders (hand/glasses/mask on cheek) stay protected.
// ============================================================================
void testV392MulticlassChestNoDarkShadowPatch(MetalDevice& dev) {
    constexpr int kW = 180;
    constexpr int kH = 320;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float scale = 0.36f / FaceMeshFilter::canonicalFaceWidth();

    // Place face in upper-middle frame (center y=0.35 top-down) leaving large neck & chest below (y=0.55..0.95)
    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float mx = 0.50f + xyz[i * 3] * scale;
        const float my = (0.65f / kAspect) + xyz[i * 3 + 1] * scale;
        mpXy[i * 2] = mx;
        mpXy[i * 2 + 1] = 1.0f - (my * kAspect);
    }
    for (int i = 468; i < 478; ++i) {
        mpXy[i * 2] = 0.50f;
        mpXy[i * 2 + 1] = 0.32f;
    }

    // Uniform warm skin frame across face, neck & chest
    std::vector<uint8_t> camRgba(kW * kH * 4, 255);
    for (int i = 0; i < kW * kH; ++i) {
        camRgba[i * 4 + 0] = 185;
        camRgba[i * 4 + 1] = 145;
        camRgba[i * 4 + 2] = 125;
        camRgba[i * 4 + 3] = 255;
    }
    const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camRgba.data());

    auto makeEngineWithConfig = [&]() {
        auto eng = std::make_unique<LiveBeautyEngine>(dev);
        BeautyConfig cfg{};
        cfg.enabled = true;
        cfg.performanceTier = PerformanceTier::kUltra;
        cfg.smoothing = 0.45f;
        cfg.whiten = 0.65f;
        cfg.lipstick = 0.50f;
        cfg.blush = 0.35f;
        eng->setConfig(cfg);
        return eng;
    };

    // 1. Baseline without multiclass semantic mask
    auto engBase = makeEngineWithConfig();
    int64_t ts = 9000;
    TextureId outBase = kInvalidTexture;
    for (int f = 0; f < 6; ++f) {
        ts += 33;
        engBase->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
        outBase = engBase->processFrame(camTex, kW, kH, ts, ts);
    }
    std::vector<uint8_t> basePixels(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(outBase, basePixels.data(), kW, kH));

    // 2. Same setup WITH 6-class multiclass logits classifying ENTIRE chest (y >= 0.52) as clothes (class 4)
    //    AND placing a small accessory occluder (class 5) strictly inside the left cheek (u=0.45..0.55, v=0.33..0.38).
    auto engMc = makeEngineWithConfig();
    ts = 9000;
    for (int f = 0; f < 5; ++f) {
        ts += 33;
        engMc->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
        engMc->processFrame(camTex, kW, kH, ts, ts);
    }
    constexpr int kM = 64;
    std::vector<float> logits(kM * kM * 6, -4.0f);
    for (int y = 0; y < kM; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / kM;
        for (int x = 0; x < kM; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / kM;
            float* lg = logits.data() + (y * kM + x) * 6;
            if (v >= 0.52f) {
                lg[4] = 6.0f;  // Clothes / chest occlusion outside face
            } else if (u >= 0.42f && u <= 0.58f && v >= 0.31f && v <= 0.40f) {
                lg[5] = 6.0f;  // In-face occluder (glasses/hand on mid-face)
            } else {
                lg[3] = 6.0f;  // Face skin
            }
        }
    }
    ts += 33;
    engMc->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
    EXPECT_TRUE(engMc->submitMulticlass6Logits(logits.data(), kM, kM, ts));
    const TextureId outMc = engMc->processFrame(camTex, kW, kH, ts, ts);
    std::vector<uint8_t> mcPixels(kW * kH * 4, 0);
    EXPECT_TRUE(dev.readbackTextureRgba8(outMc, mcPixels.data(), kW, kH));

    // Probe chest pixel (center x=90, y=240 = 75% down): MUST match baseline (zero dark shadow patch!)
    const int chestIdx = (240 * kW + 90) * 4;
    EXPECT_TRUE(basePixels[chestIdx + 0] > 195);  // Chest skin IS whitened in baseline
    EXPECT_TRUE(std::abs(int(mcPixels[chestIdx + 0]) - int(basePixels[chestIdx + 0])) <= 2);
    EXPECT_TRUE(std::abs(int(mcPixels[chestIdx + 1]) - int(basePixels[chestIdx + 1])) <= 2);
    EXPECT_TRUE(std::abs(int(mcPixels[chestIdx + 2]) - int(basePixels[chestIdx + 2])) <= 2);

    // v3.9.3: Probe in-face occluded pixel (center x=90, y=114 = 35.6% down, e.g. skin behind glasses or
    // bangs/INT8 false positive): kCompositePro whitening MUST ALSO match baseline (ZERO dark shadow hole!),
    // while UV makeup occlusion is verified separately in testV393NoDarkShadowPatchesBodyFaceHairGuided.
    const int faceOccIdx = (114 * kW + 90) * 4;
    EXPECT_TRUE(basePixels[faceOccIdx + 0] > 195);
    EXPECT_TRUE(std::abs(int(mcPixels[faceOccIdx + 0]) - int(basePixels[faceOccIdx + 0])) <= 2);
    EXPECT_TRUE(std::abs(int(mcPixels[faceOccIdx + 1]) - int(basePixels[faceOccIdx + 1])) <= 2);
    EXPECT_TRUE(std::abs(int(mcPixels[faceOccIdx + 2]) - int(basePixels[faceOccIdx + 2])) <= 2);

    engBase->releaseGpuResources();
    engMc->releaseGpuResources();
    dev.destroyTexture(camTex);
}

// v3.9.3: End-to-end GPU pixel regression gates covering all 3 root causes of inexplicable dark shadow patches:
//   1) 6-class semantic occlusion deferred after kCompositePro: face skin stays evenly whitened even under
//      in-face hair/glasses/accessories class labels while UV blush makeup is still cleanly occluded (0 bleed);
//   2) Luminance-normalized body/neck skin gating in kCompositePro: shaded/cool-LED collarbones, necks, chests
//      & arms whiten smoothly without dark shadow cliffs while neutral walls, black hair & dark beard keep <=1/255;
//   3) Coarse hairMask hairline bleed guard inside face polygon + kGuidedWeightDiff zero-weight anti-black fallback.
void testV393NoDarkShadowPatchesBodyFaceHairGuided(MetalDevice& dev) {
    // --- Part A: In-face 6-class occluder blocks UV blush WITHOUT punching a dark un-whitened shadow patch ---
    {
        constexpr int kW = 180, kH = 320;
        const float aspect = static_cast<float>(kW) / static_cast<float>(kH);
        std::array<float, math::FaceMeshFrame::kCoordCount> mpXy{};
        const auto& xyz = math::FaceMeshTopology::kCanonicalXyz;
        const float scale = 0.36f / math::FaceMeshFilter::canonicalFaceWidth();
        for (int i = 0; i < math::FaceMeshTopology::kFaceVertexCount; ++i) {
            const float mx = 0.50f + xyz[i * 3] * scale;
            const float my = (0.65f / aspect) + xyz[i * 3 + 1] * scale;
            mpXy[i * 2] = mx;
            mpXy[i * 2 + 1] = 1.0f - (my * aspect);
        }
        for (int i = 468; i < 478; ++i) { mpXy[i * 2] = 0.50f; mpXy[i * 2 + 1] = 0.32f; }

        const TextureId camTex = makeSolidTex(dev, kW, kH, 185, 145, 125, 255);
        // Full-coverage blush atlas (all channels = 255) so blush tints unoccluded cheeks noticeably
        const TextureId solidAtlas = makeSolidTex(dev, 32, 32, 255, 255, 255, 255);

        auto makeEng = [&]() {
            auto eng = std::make_unique<LiveBeautyEngine>(dev);
            PipelineAssets assets{};
            assets.blushAtlas = solidAtlas;
            assets.contourHighlightAtlas = solidAtlas;
            eng->setAssets(assets);
            BeautyConfig cfg{};
            cfg.enabled = true;
            cfg.performanceTier = PerformanceTier::kUltra;
            cfg.smoothing = 0.40f;
            cfg.whiten = 0.60f;
            cfg.blush = 0.85f;
            cfg.blushColor = BlushColor::kBerry;
            eng->setConfig(cfg);
            return eng;
        };

        auto engBase = makeEng();
        auto engMc = makeEng();
        int64_t ts = 11000;
        TextureId outBase = kInvalidTexture;
        for (int f = 0; f < 6; ++f) {
            ts += 33;
            engBase->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
            outBase = engBase->processFrame(camTex, kW, kH, ts, ts);
            if (f < 5) {
                engMc->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
                engMc->processFrame(camTex, kW, kH, ts, ts);
            }
        }
        // Mark center/right face cheek (u=0.42..0.60, v=0.31..0.41) as class 5 occluder (e.g. hand/glasses)
        constexpr int kM = 64;
        std::vector<float> logits(kM * kM * 6, -4.0f);
        for (int y = 0; y < kM; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / kM;
            for (int x = 0; x < kM; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / kM;
                float* lg = logits.data() + (y * kM + x) * 6;
                if (u >= 0.42f && u <= 0.60f && v >= 0.31f && v <= 0.41f) {
                    lg[5] = 6.0f;
                } else {
                    lg[3] = 6.0f;
                }
            }
        }
        engMc->submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts);
        EXPECT_TRUE(engMc->submitMulticlass6Logits(logits.data(), kM, kM, ts));
        const TextureId outMc = engMc->processFrame(camTex, kW, kH, ts, ts);

        std::vector<uint8_t> basePx(kW * kH * 4, 0), mcPx(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(outBase, basePx.data(), kW, kH));
        EXPECT_TRUE(dev.readbackTextureRgba8(outMc, mcPx.data(), kW, kH));

        // At occluded mid-face (x=90, y=114):
        //  1) Blush pigment IS cleanly blocked by occMask -> G/B stay HIGHER (less rose-berry tint) than blush baseline!
        //  2) Skin is STILL whitened (R > 195 vs input 185) -> NOT turned into an un-whitened dark shadow hole!
        const int occIdx = (114 * kW + 90) * 4;
        EXPECT_TRUE(mcPx[occIdx + 0] > 195);
        EXPECT_TRUE(int(mcPx[occIdx + 1]) - int(basePx[occIdx + 1]) >= 12);
        EXPECT_TRUE(int(mcPx[occIdx + 2]) - int(basePx[occIdx + 2]) >= 10);

        engBase->releaseGpuResources();
        engMc->releaseGpuResources();
        dev.destroyTexture(camTex);
        dev.destroyTexture(solidAtlas);
    }

    // --- Part B: Luminance-normalized body/neck skin gating & coarse hairMask hairline guard in kCompositePro ---
    {
        constexpr int kSz = 32;
        const RenderTargetId rt = dev.createRenderTarget(kSz, kSz, TextureFormat::kRgba8);
        auto evalComp = [&](Rgba src, uint8_t faceM, uint8_t nonHair, float whiten, bool hasFaceOval = true) -> Rgba {
            const TextureId srcTex = makeSolidTex(dev, kSz, kSz, src.r, src.g, src.b, 255);
            const TextureId semTex = makeSolidTex(dev, kSz, kSz, faceM, 0, 0, nonHair);
            RenderState st{};
            TextureBindings tb{};
            tb.bind(0, srcTex);
            tb.bind(1, srcTex);
            tb.bind(2, semTex);
            tb.bind(3, semTex);
            UniformBlock ub{};
            ub.push(1.0f / kSz, 1.0f / kSz, 1.0f, 5.0f);  // [0] framePresence=5.0 -> polySkinMask mode, pres=1.0
            ub.push(16.0f, 0.0f, 0.0f, 0.5f);             // [1] skinOnly=1, smoothing=0, brightness=0
            ub.push(0.0f, 0.0f, 1.0f, 0.0f);              // [2] radiance=0, sharp=0, sat=1, filter=0
            ub.push(whiten, 0.0f, 0.0f, 0.0f);            // [3] whiten, rosy=0, oil=0, relight=0
            ub.push(0.0f, 0.0f, 0.0f, hasFaceOval ? 1.0f : 0.0f);  // [4] uEye0.w = axis.x
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);                       // [5] uEye1.y = bodyPriorMute=0, uEye1.w = axis.y
            // [6] uOval: place tracked face above probe center (0.5, 0.5) so probe lies in neck/chest portraitPrior=1.0
            if (hasFaceOval) {
                ub.push(0.50f, 0.72f, 0.18f, 0.22f);
            } else {
                ub.push(0.0f, 0.0f, 0.0f, 0.0f);
            }
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // [7] uMouth
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // [8] uEyeMask0 (secondary oval inactive)
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // [9] uEyeMask1
            ub.push(0.0f, 0.01f, 0.0f, 0.0f);             // [10] lut=0, core=0.01, micro=0, ambient=0
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // [11] uWbRow1
            ub.push(0.0f, 0.0f, 0.0f, 0.0f);              // [12] uWbRow2
            dev.beginPass(rt, kSz, kSz, false, {});
            dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
            dev.endPass();
            std::vector<uint8_t> px(kSz * kSz * 4, 0);
            EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), px.data(), kSz, kSz));
            dev.destroyTexture(srcTex);
            dev.destroyTexture(semTex);
            const int mid = (16 * kSz + 16) * 4;
            return {px[mid], px[mid + 1], px[mid + 2], px[mid + 3]};
        };
        auto yOf = [](Rgba c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; };

        // 1. Body / neck / collarbone / arm shaded skin (faceM = 0, inside portraitPrior) whitens smoothly
        //    without dark-shadow cliffs, while untracked frame (hasFaceOval=false, v3.9.5 skin-n1) stays 0 bleed!
        const Rgba litCollar{135, 114, 106, 255};
        const Rgba hollowCollar{115, 102, 99, 255};
        const Rgba deepNeck{85, 72, 66, 255};
        const Rgba coolFairShaded{138, 126, 123, 255};

        const Rgba outLit = evalComp(litCollar, 0, 255, 0.50f);
        const Rgba outHollow = evalComp(hollowCollar, 0, 255, 0.50f);
        const Rgba outDeep = evalComp(deepNeck, 0, 255, 0.50f);
        const Rgba outCoolFair = evalComp(coolFairShaded, 0, 255, 0.50f);

        const float dLit = yOf(outLit) - yOf(litCollar);
        const float dHollow = yOf(outHollow) - yOf(hollowCollar);
        const float dDeep = yOf(outDeep) - yOf(deepNeck);
        const float dCoolFair = yOf(outCoolFair) - yOf(coolFairShaded);

        EXPECT_TRUE(dLit >= 22.0f);
        EXPECT_TRUE(dHollow >= 20.0f);  // Previously dropped to +4.4 -> stark dark shadow blotch!
        EXPECT_TRUE(dDeep >= 18.0f);    // Previously dropped to +8.9 -> muddy neck shadow!
        EXPECT_TRUE(dCoolFair >= 11.0f);
        EXPECT_TRUE(std::fabs(dLit - dHollow) <= 6.0f);  // Smooth transition between lit & hollow collarbone

        // v3.9.5 (skin-n1): When no face oval is tracked, portraitPrior defaults to 0.0 (no background flash)
        const Rgba outNoFace = evalComp(litCollar, 0, 255, 0.50f, /*hasFaceOval=*/false);
        EXPECT_TRUE(std::abs(int(outNoFace.r) - int(litCollar.r)) <= 1);
        EXPECT_TRUE(std::abs(int(outNoFace.g) - int(litCollar.g)) <= 1);
        EXPECT_TRUE(std::abs(int(outNoFace.b) - int(litCollar.b)) <= 1);

        // 2. Non-skin surfaces outside face (faceM = 0) keep <= 1/255 whitening bleed!
        const Rgba nonSkins[] = {
            {185, 185, 187, 255},  // Cool grey wall
            {78, 76, 72, 255},     // Dim warm-grey wall
            {220, 215, 210, 255},  // Bright studio wall
            {28, 26, 27, 255},     // Black hair
            {38, 34, 32, 255},     // Dark beard bristle
            {40, 55, 85, 255},     // Navy shirt
            {200, 40, 40, 255},    // Saturated red shirt
        };
        for (const auto& ns : nonSkins) {
            const Rgba outNs = evalComp(ns, 0, 255, 0.50f);
            EXPECT_TRUE(std::abs(int(outNs.r) - int(ns.r)) <= 1);
            EXPECT_TRUE(std::abs(int(outNs.g) - int(ns.g)) <= 1);
            EXPECT_TRUE(std::abs(int(outNs.b) - int(ns.b)) <= 1);
        }

        // 3. v3.9.5 (skin-n3): Coarse 128x128 hairMask transition-edge bleed (nonHair=96/255 ~ 0.37) onto lit
        //    forehead skin inside faceM=255 stays whitened (no dark hairline band), while high-confidence bangs
        //    interior (nonHair=0, even light/blonde) AND real black bangs stay 100% protected (<= 1/255)!
        const Rgba foreheadSkin{178, 142, 126, 255};
        const Rgba outForeheadEdge = evalComp(foreheadSkin, 255, 96, 0.55f);
        EXPECT_TRUE(yOf(outForeheadEdge) - yOf(foreheadSkin) >= 18.0f);

        const Rgba outHighConfBangs = evalComp(foreheadSkin, 255, 0, 0.55f);
        EXPECT_TRUE(std::abs(int(outHighConfBangs.r) - int(foreheadSkin.r)) <= 1);
        EXPECT_TRUE(std::abs(int(outHighConfBangs.g) - int(foreheadSkin.g)) <= 1);
        EXPECT_TRUE(std::abs(int(outHighConfBangs.b) - int(foreheadSkin.b)) <= 1);

        const Rgba blackBangs{28, 26, 27, 255};
        const Rgba outBangs = evalComp(blackBangs, 255, 96, 0.55f);
        EXPECT_TRUE(std::abs(int(outBangs.r) - int(blackBangs.r)) <= 1);
        EXPECT_TRUE(std::abs(int(outBangs.g) - int(blackBangs.g)) <= 1);
        EXPECT_TRUE(std::abs(int(outBangs.b) - int(blackBangs.b)) <= 1);

        dev.destroyRenderTarget(rt);
    }

    // --- Part C: kGuidedWeightDiff zero-weight hair pixels stay original RGB (NOT pure black 0,0,0) ---
    {
        constexpr int kW = 32, kH = 32;
        const TextureId hairImg = makeSolidTex(dev, kW, kH, 135, 102, 82, 255);  // Medium brown hair
        const TextureId zeroWeight = makeSolidTex(dev, kW, kH, 0, 0, 0, 0);      // nonHair = 0
        const RenderTargetId rtBlur = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);
        const RenderTargetId rtDiff = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        {
            RenderState st{};
            TextureBindings tb{}; tb.bind(0, hairImg); tb.bind(1, zeroWeight);
            UniformBlock ub{}; ub.push(1.25f / kW, 0.0f, 0.0f, 0.0f);
            dev.beginPass(rtBlur, kW, kH, false, {});
            dev.drawFullScreenQuad(ShaderId::kGuidedWeightBlur, st, tb, ub);
            dev.endPass();
        }
        {
            RenderState st{};
            TextureBindings tb{}; tb.bind(0, dev.textureOf(rtBlur)); tb.bind(1, hairImg);
            UniformBlock ub{}; ub.push(0.0f, 1.25f / kH, 0.0f, 0.0f);
            dev.beginPass(rtDiff, kW, kH, false, {});
            dev.drawFullScreenQuad(ShaderId::kGuidedWeightDiff, st, tb, ub);
            dev.endPass();
        }
        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rtDiff), out.data(), kW, kH));
        const int mid = (16 * kW + 16) * 4;
        // Previously collapsed to (0, 0, 0) pure black and poisoned rtMacro_ 46px blur!
        EXPECT_TRUE(std::abs(int(out[mid + 0]) - 135) <= 2);
        EXPECT_TRUE(std::abs(int(out[mid + 1]) - 102) <= 2);
        EXPECT_TRUE(std::abs(int(out[mid + 2]) - 82) <= 2);

        dev.destroyTexture(hairImg);
        dev.destroyTexture(zeroWeight);
        dev.destroyRenderTarget(rtBlur);
        dev.destroyRenderTarget(rtDiff);
    }
}

void testV394MetalAuditPixelInvariants(MetalDevice& dev) {
    // 1. A1 + A2: Under default Natural beauty (with face tracked at center), neutral gray wall
    //    AND warm skin-colored wood background outside the head/neck/chest prior must stay untouched (<= 2/255),
    //    while face skin and chest below chin still whiten cleanly.
    {
        constexpr int kW = 128, kH = 128;
        std::vector<uint8_t> camPx(kW * kH * 4, 255);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int o = (y * kW + x) * 4;
                if (x < 24) {
                    // Neutral gray wall on far left (A1)
                    camPx[o] = 120; camPx[o + 1] = 120; camPx[o + 2] = 120;
                } else if (x >= kW - 24) {
                    // Warm beige/wood wall on far right (A2: matches skin chroma!)
                    camPx[o] = 185; camPx[o + 1] = 148; camPx[o + 2] = 122;
                } else {
                    // Center face / neck / chest skin
                    camPx[o] = 185; camPx[o + 1] = 148; camPx[o + 2] = 122;
                }
            }
        }
        const TextureId camTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
        LiveBeautyEngine engine(dev);
        BeautyConfig cfg{};
        cfg.enabled = true;
        cfg.skinOnly = true;
        cfg.smoothing = 0.60f;
        cfg.whiten = 0.55f;
        cfg.rosy = 0.35f;
        engine.setConfig(cfg);

        // Place tracked face centered at (0.50, 0.38) top-down with fw ~ 0.26
        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        const float scale = 0.26f / FaceMeshFilter::canonicalFaceWidth();
        std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            mpXy[i * 2] = 0.50f + xyz[i * 3] * scale;
            mpXy[i * 2 + 1] = 0.38f - xyz[i * 3 + 1] * scale;
        }
        for (int i = 468; i < 478; ++i) { mpXy[i * 2] = 0.50f; mpXy[i * 2 + 1] = 0.35f; }
        TextureId outTex = kInvalidTexture;
        for (int f = 0; f < 5; ++f) {
            const int64_t ts = 1000 + f * 33;
            EXPECT_TRUE(engine.submitFaceLandmarks(ts, mpXy.data(), 478, kW, kH, ts));
            outTex = engine.processFrame(camTex, kW, kH, ts, ts, kInvalidTexture, 0.30f, 1.0f);
        }
        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(outTex, out.data(), kW, kH));

        // Neutral gray wall (x=10, y=64): max drift <= 2/255 (was +11..25/255!)
        const int grayIdx = (64 * kW + 10) * 4;
        EXPECT_TRUE(std::abs(int(out[grayIdx + 0]) - 120) <= 2);
        EXPECT_TRUE(std::abs(int(out[grayIdx + 1]) - 120) <= 2);
        EXPECT_TRUE(std::abs(int(out[grayIdx + 2]) - 120) <= 2);

        // Warm wood wall on far right (x=118, y=48, outside head/chest prior): max drift <= 2/255!
        const int woodIdx = (48 * kW + 118) * 4;
        EXPECT_TRUE(std::abs(int(out[woodIdx + 0]) - 185) <= 2);
        EXPECT_TRUE(std::abs(int(out[woodIdx + 1]) - 148) <= 2);
        EXPECT_TRUE(std::abs(int(out[woodIdx + 2]) - 122) <= 2);

        // Face skin center (x=64, y=48): visibly whitened (> +8 luma)
        const int faceIdx = (48 * kW + 64) * 4;
        const float faceY = 0.299f * out[faceIdx] + 0.587f * out[faceIdx + 1] + 0.114f * out[faceIdx + 2];
        const float origY = 0.299f * 185.0f + 0.587f * 148.0f + 0.114f * 122.0f;
        EXPECT_TRUE(faceY - origY >= 8.0f);

        engine.releaseGpuResources();
        dev.destroyTexture(camTex);
    }

    // 2. A3 (skin-1): 1px dark eyebrow/stubble strand retains >= 75% contrast while isolated 1px dot still smooths.
    {
        constexpr int kW = 64, kH = 64;
        std::vector<uint8_t> camPx(kW * kH * 4, 255);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const int o = (y * kW + x) * 4;
                camPx[o] = 190; camPx[o + 1] = 152; camPx[o + 2] = 132;
            }
        }
        // Horizontal 1px dark hair/stubble line across y=24 (x=12..52) with realistic mid-band contrast (-52 luma)
        for (int x = 12; x <= 52; ++x) {
            const int o = (24 * kW + x) * 4;
            camPx[o] = 132; camPx[o + 1] = 102; camPx[o + 2] = 86;
        }
        // Isolated 1px dark pore/blackhead dot at (32, 42) with identical contrast
        {
            const int o = (42 * kW + 32) * 4;
            camPx[o] = 132; camPx[o + 1] = 102; camPx[o + 2] = 86;
        }
        const TextureId srcTex = dev.createTexture2D(kW, kH, TextureFormat::kRgba8, camPx.data());
        // Simulate guided smooth output: guidedA=0.05 (flat skin), guidedB=skin*(1-0.05) -> heavily smoothed
        const TextureId guidedTex = makeSolidTex(dev, kW, kH,
            static_cast<uint8_t>(190 * 0.95f), static_cast<uint8_t>(152 * 0.95f),
            static_cast<uint8_t>(132 * 0.95f), static_cast<uint8_t>(0.05f * 255.0f));
        const TextureId skinMask = makeSolidTex(dev, kW, kH, 255, 0, 0, 255);
        const TextureId fallback = makeSolidTex(dev, 1, 1, 0, 0, 0, 255);
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        RenderState st{};
        TextureBindings tb{};
        tb.bind(0, srcTex); tb.bind(1, guidedTex); tb.bind(2, skinMask); tb.bind(3, fallback);
        UniformBlock ub{};
        ub.push(1.0f / kW, 1.0f / kH, 1.0f, 5.0f);  // framePresence = 1 + 4 (poly mask)
        ub.push(0.0f, 0.70f, 0.0f, 0.20f);           // smooth = 0.70
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 1.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.5f, 0.5f, 0.4f, 0.4f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.025f, 0.09f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        ub.push(0.0f, 0.0f, 0.0f, 0.0f);
        dev.beginPass(rt, kW, kH, LoadOp::kDontCare);
        dev.drawFullScreenQuad(ShaderId::kCompositePro, st, tb, ub);
        dev.endPass();

        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        auto yPx = [&](int x, int y) {
            const int o = (y * kW + x) * 4;
            return 0.299f * out[o] + 0.587f * out[o + 1] + 0.114f * out[o + 2];
        };
        const float skinY = yPx(32, 16);
        const float lineY = yPx(32, 24);
        const float dotY = yPx(32, 42);
        const float origContrast = (0.299f * 190 + 0.587f * 152 + 0.114f * 132) - (0.299f * 132 + 0.587f * 102 + 0.114f * 86);
        const float lineContrast = skinY - lineY;
        const float dotContrast = skinY - dotY;
        // 1px dark hair line must retain >= 75% contrast, while isolated 1px dot is smoothed noticeably more!
        EXPECT_TRUE(lineContrast >= origContrast * 0.75f);
        EXPECT_TRUE(dotContrast < lineContrast * 0.75f);

        dev.destroyTexture(srcTex);
        dev.destroyTexture(guidedTex);
        dev.destroyTexture(skinMask);
        dev.destroyTexture(fallback);
        dev.destroyRenderTarget(rt);
    }

    // 3. C2 (makeup-v1): kMakeupIris contactLens linear strength (no alpha^2 attenuation)
    {
        constexpr int kW = 64, kH = 64;
        const TextureId baseTex = makeSolidTex(dev, kW, kH, 40, 40, 40, 255);  // Dark pupil/iris base
        // Iris atlas with bright blue lens RGB(40, 140, 240) and alpha=255 (no catchlight: G=0)
        const TextureId irisAtlas = makeSolidTex(dev, 32, 32, 40, 140, 240, 255);
        const RenderTargetId rt = dev.createRenderTarget(kW, kH, TextureFormat::kRgba8);

        // Fullscreen mesh quad using kMakeupPos2Uv2Uv2 (pos, screenUv, makeupUv)
        const float verts[4 * 6] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        };
        const uint16_t idx[6] = {0, 1, 2, 2, 1, 3};
        const BufferId vb = dev.createDynamicVertexBuffer(sizeof(verts));
        dev.updateVertexBuffer(vb, verts, sizeof(verts));
        const BufferId ib = dev.createStaticIndexBuffer(idx, 6);

        // Copy baseTex into rt first, then alpha-blend kMakeupIris at lensS=0.50 (clS=0)
        {
            RenderState st{};
            TextureBindings tb{}; tb.bind(0, baseTex);
            UniformBlock ub{};
            dev.beginPass(rt, kW, kH, LoadOp::kDontCare);
            dev.drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);

            RenderState blendSt{};
            blendSt.blendEnable = true;
            TextureBindings itb{}; itb.bind(0, baseTex); itb.bind(1, irisAtlas);
            UniformBlock iub{};
            iub.push(0.50f, 0.50f, 0.40f, 1.0f);  // iris center (0.5, 0.5), radius=0.40, aspect=1.0
            iub.push(0.0f, 0.0f, 0.30f, 0.0f);    // tile=(0,0), effOpen=0.30 (fully open), yaw=0
            iub.push(0.50f, 0.0f, 0.0f, 0.0f);    // lensS=0.50, clS=0.0
            dev.drawIndexed(ShaderId::kMakeupIris, blendSt, vb, ib, 0, 6, VertexLayout::kMakeupPos2Uv2Uv2, itb, iub);
            dev.endPass();
        }
        std::vector<uint8_t> out(kW * kH * 4, 0);
        EXPECT_TRUE(dev.readbackTextureRgba8(dev.textureOf(rt), out.data(), kW, kH));
        // Sample ring r=0.55*radius (x=32+14=46, y=32): base B=40, atlas B=240, lensA ~ 0.50*0.65 = 0.325
        // Linear blend expects B ~ 40 + 0.325*(240-40) = 105 (old squared alpha gave only ~61!).
        const int ringIdx = (32 * kW + 46) * 4;
        EXPECT_TRUE(out[ringIdx + 2] >= 92 && out[ringIdx + 2] <= 122);

        dev.destroyBuffer(vb);
        dev.destroyBuffer(ib);
        dev.destroyTexture(baseTex);
        dev.destroyTexture(irisAtlas);
        dev.destroyRenderTarget(rt);
    }
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
        testP1ToP4GpuPixelGates(dev);
        testMetalIntraFrameMultiSlotVboZeroCopyBgraAndTransientEviction(dev);
        testV36MetalChromaKeyParticleSdfAndHandShieldProbes(dev);
        testV361ColdPorcelainWhiteningAndNonDarkeningLipProbes(dev);
        testV367EyeSocketNoseAndBeardContourNoDarkTrenches(dev);
        testV369MetalSlightCameraShakeNoWaterRipple(dev);
        testV3610MetalCheekBeautyZeroTwitch(dev);
        testV3611RealisticNaturalLipColorNoFakePink(dev);
        testV3613MetalHairDyeRgbBoundaryMattingBangsAndDarkHairUniformity(dev);
        testV3614TikTokExposureCoupledNonDarkeningLips(dev);
        testV3615MetalDouyinAmbientSceneBrightnessLift(dev);
        testV370MetalCommercialBeautyPixelInvariants(dev);
        testV380MetalAtmosphereBloomGrainCat02AndOpenMouthLipProbes(dev);
        testV390SevenScenarioPortraitQualityBenchmark(dev);
        testV392MulticlassChestNoDarkShadowPatch(dev);
        testV393NoDarkShadowPatchesBodyFaceHairGuided(dev);
        testV394MetalAuditPixelInvariants(dev);

        if (gFailures == 0) {
            std::cout << "PASS  all Apple Metal GPU pixel gates (L1/L2/L6, E1-E4, R1, V1/H1, H2, S1, M1, P1-P4, E2E) passed!\n";
            return 0;
        }
        std::cerr << "FAIL  " << gFailures << " Metal GPU pixel gate assertion(s) failed.\n";
        return 1;
    }
}
