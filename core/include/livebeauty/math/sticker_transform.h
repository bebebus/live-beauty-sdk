#pragma once

#include <cstdint>
#include "livebeauty/rhi/rhi_types.h"

namespace livebeauty::math {

enum class StickerAnchor : int {
    kForeheadCrown = 0,  // Crown / ears / headband anchored above forehead (landmark 10)
    kEyesBridge    = 1,  // Glasses / shades anchored at inter-pupillary midpoint
    kNoseTip       = 2,  // Cute nose / whiskers anchored at nose tip (landmark 1)
    kCheeks        = 3,  // Mid-face cheek decals
    kFullFace      = 4,  // Full face mask centered on face oval
};

struct StickerLayerConfig {
    bool enabled = false;
    StickerAnchor anchor = StickerAnchor::kForeheadCrown;
    rhi::TextureId texture = rhi::kInvalidTexture;
    float scale = 1.0f;          // Relative scale multiplier (1.0 = default anatomical fit)
    float offsetX = 0.0f;        // Offset along face right axis in units of faceWidth
    float offsetY = 0.0f;        // Offset along face up axis in units of faceWidth
    float rotationRad = 0.0f;    // Additional CCW rotation in radians
    float opacity = 1.0f;        // [0..1]
    float textureAspect = 1.0f;  // textureWidth / textureHeight
};

static constexpr int kMaxStickerLayers = 4;

/**
 * Computes rigid 2D AR sticker quad vertices (4 vertices x 4 floats: ndcX, ndcY, u, v)
 * anchored to post-warp 478-pt face landmarks so stickers stay locked to head motion,
 * scale, rotation, and facial reshaping without jitter.
 */
struct StickerTransform {
    static constexpr int kQuadFloats = 16;  // 4 vertices * (ndcX, ndcY, u, v)

    static bool buildQuadVertices(
        const StickerLayerConfig& cfg,
        const float* landmarks478x2,
        const float* faceMeta15,
        float aspect,
        float outQuad16[kQuadFloats]);
};

}  // namespace livebeauty::math
