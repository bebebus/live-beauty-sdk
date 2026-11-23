#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "livebeauty/types.h"

namespace livebeauty::rhi {

using TextureId = uint32_t;
using RenderTargetId = uint32_t;
using BufferId = uint32_t;

constexpr TextureId kInvalidTexture = 0;
constexpr RenderTargetId kInvalidRenderTarget = 0;
constexpr BufferId kInvalidBuffer = 0;

enum class TextureFormat : uint8_t {
    kRgba8 = 0,
    kR8 = 1,
};

enum class VertexLayout : uint8_t {
    kQuadPos2Uv2 = 0,       // 4 floats: ndcX, ndcY, u, v
    kMeshPos2Uv2 = 1,       // 4 floats: dstNdcX, dstNdcY, srcU, srcV (FaceMeshDeformer)
    kSkinMaskPos2W3 = 2,    // 5 floats: dstNdcX, dstNdcY, skin, excl, hiPlane
    kRetouchPos2Uv2A4 = 3,  // 8 floats: dstNdcX, dstNdcY, srcU, srcV, wTear, wFold, offU, offV
    kMakeupPos2Uv2Uv2 = 4,  // 6 floats: dstNdcX, dstNdcY, screenU, screenV, makeupU, makeupV
    kAr3dPos2Uv2Z1 = 5,     // 5 floats: dstNdcX, dstNdcY, u, v, depth01 (P4 3D AR & head occluder)
};

enum class ShaderId : uint8_t {
    kPassThrough = 0,
    kDownsample,
    kBlurRgb,
    kBlurAlpha,
    kGuidedDiff,
    kGuidedCoeff,
    kGuidedWeightBlur,
    kGuidedWeightDiff,
    kGuidedWeightCoeff,
    kMeshWarp,
    kSkinMask,
    kRetouchSkin,
    kRetouchEyeTeeth,
    kMakeupLayer,
    kMakeupLip,
    kMakeupIris,
    kBodyWarp,
    kCompositePro,
    kTemporalDenoise,
    kBgPremask,
    kBgBlur,
    kBgMix,
    kHairAlpha,
    kBgHairMerge,
    kBgVirtualMix,
    kStickerQuad,
    kMatteRefine,
    kBlemishHeal,
    kAiResidualBlend,
    kHeadDepthShell,
    kAr3dMesh,
    kCount,
};

struct RenderState {
    bool blendEnable = false;
    bool cullBack = false;
    bool frontFaceCw = false;
    bool colorMaskAlphaOnly = false;
};

struct UniformBlock {
    static constexpr int kMaxVec4 = 16;
    std::array<Vec4, kMaxVec4> v{};
    int count = 0;

    void clear() { count = 0; }
    void push(float x, float y, float z, float w) {
        if (count < kMaxVec4) v[count++] = {x, y, z, w};
    }
    void push(const Vec4& vec) {
        if (count < kMaxVec4) v[count++] = vec;
    }
};

struct TextureBindings {
    static constexpr int kMaxSlots = 4;
    std::array<TextureId, kMaxSlots> slots = {
        kInvalidTexture, kInvalidTexture, kInvalidTexture, kInvalidTexture,
    };
    int count = 0;

    void clear() {
        slots.fill(kInvalidTexture);
        count = 0;
    }
    void bind(int slot, TextureId tex) {
        if (slot >= 0 && slot < kMaxSlots) {
            slots[slot] = tex;
            if (slot + 1 > count) count = slot + 1;
        }
    }
};

}  // namespace livebeauty::rhi
