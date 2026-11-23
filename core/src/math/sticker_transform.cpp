#include "livebeauty/math/sticker_transform.h"

#include <cmath>
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

bool StickerTransform::buildQuadVertices(
    const StickerLayerConfig& cfg,
    const float* landmarks478x2,
    const float* faceMeta15,
    float aspect,
    float outQuad16[kQuadFloats]) {
    if (!cfg.enabled || cfg.texture == rhi::kInvalidTexture || cfg.opacity <= 1e-4f ||
        !landmarks478x2 || !faceMeta15 || !(aspect > 0.0f)) {
        return false;
    }

    const float axRaw = faceMeta15[FaceMeshFrame::kMetaAxisX];
    const float ayRaw = faceMeta15[FaceMeshFrame::kMetaAxisY];
    const float norm = std::hypot(axRaw, ayRaw);
    const float faceW = faceMeta15[FaceMeshFrame::kMetaFaceWidth];
    if (!(norm > 1e-5f) || !(faceW > 1e-4f)) return false;

    // Metric-space orthonormal face basis (x = image width, y = image height / aspect).
    // Right axis = (rx, ry), Up axis = (ux, uy) pointing toward forehead (positive GL Y).
    float rx = axRaw / norm;
    float ry = ayRaw / norm;
    float ux = -ry;
    float uy = rx;
    // Ensure up axis points from chin (152) toward forehead (10) in GL y-up metric space.
    const float fhDx = landmarks478x2[10 * 2 + 0] - landmarks478x2[152 * 2 + 0];
    const float fhDy = (landmarks478x2[10 * 2 + 1] - landmarks478x2[152 * 2 + 1]) / aspect;
    if (fhDx * ux + fhDy * uy < 0.0f) {
        ux = -ux;
        uy = -uy;
        rx = -rx;
        ry = -ry;
    }

    auto pt = [&](int idx) -> Vec2 {
        return {landmarks478x2[idx * 2 + 0], landmarks478x2[idx * 2 + 1] / aspect};
    };

    Vec2 anchorM{};
    float baseWidthM = faceW;
    switch (cfg.anchor) {
        case StickerAnchor::kForeheadCrown: {
            const Vec2 forehead = pt(10);
            const Vec2 chin = pt(152);
            const float faceH = std::hypot(forehead.x - chin.x, forehead.y - chin.y);
            anchorM = {forehead.x + ux * (faceH * 0.22f), forehead.y + uy * (faceH * 0.22f)};
            baseWidthM = faceW * 1.15f;
            break;
        }
        case StickerAnchor::kEyesBridge: {
            const Vec2 rOuter = pt(33);
            const Vec2 lOuter = pt(263);
            anchorM = {(rOuter.x + lOuter.x) * 0.5f, (rOuter.y + lOuter.y) * 0.5f};
            baseWidthM = std::hypot(lOuter.x - rOuter.x, lOuter.y - rOuter.y) * 1.48f;
            break;
        }
        case StickerAnchor::kNoseTip: {
            anchorM = pt(1);
            baseWidthM = faceW * 0.42f;
            break;
        }
        case StickerAnchor::kCheeks: {
            const Vec2 rCheek = pt(205);
            const Vec2 lCheek = pt(425);
            anchorM = {(rCheek.x + lCheek.x) * 0.5f, (rCheek.y + lCheek.y) * 0.5f};
            baseWidthM = faceW * 0.92f;
            break;
        }
        case StickerAnchor::kFullFace:
        default: {
            const Vec2 forehead = pt(10);
            const Vec2 chin = pt(152);
            anchorM = {(forehead.x + chin.x) * 0.5f, (forehead.y + chin.y) * 0.5f};
            baseWidthM = faceW * 1.18f;
            break;
        }
    }

    // Apply user local offset in face basis.
    anchorM.x += (rx * cfg.offsetX + ux * cfg.offsetY) * faceW;
    anchorM.y += (ry * cfg.offsetX + uy * cfg.offsetY) * faceW;

    // Optional extra rotation around anchor.
    const float cosR = std::cos(cfg.rotationRad);
    const float sinR = std::sin(cfg.rotationRad);
    const float brx = rx * cosR + ux * sinR;
    const float bry = ry * cosR + uy * sinR;
    const float bux = -rx * sinR + ux * cosR;
    const float buy = -ry * sinR + uy * cosR;

    const float texAspect = (cfg.textureAspect > 0.05f && std::isfinite(cfg.textureAspect))
        ? cfg.textureAspect : 1.0f;
    const float halfW = 0.5f * baseWidthM * detail::clampf(cfg.scale, 0.1f, 4.0f);
    const float halfH = halfW / texAspect;

    // 4 corners in triangle-strip / 2-triangle quad order: BL, BR, TL, TR.
    // Sticker texture v=0 is top (standard PNG/Android upload), so TL has v=0, BL has v=1.
    const float signs[4][4] = {
        {-1.0f, -1.0f, 0.0f, 1.0f},  // BL
        { 1.0f, -1.0f, 1.0f, 1.0f},  // BR
        {-1.0f,  1.0f, 0.0f, 0.0f},  // TL
        { 1.0f,  1.0f, 1.0f, 0.0f},  // TR
    };
    for (int i = 0; i < 4; ++i) {
        const float sx = signs[i][0] * halfW;
        const float sy = signs[i][1] * halfH;
        const float uScreen = anchorM.x + brx * sx + bux * sy;
        const float vScreen = (anchorM.y + bry * sx + buy * sy) * aspect;
        outQuad16[i * 4 + 0] = uScreen * 2.0f - 1.0f;
        outQuad16[i * 4 + 1] = vScreen * 2.0f - 1.0f;
        outQuad16[i * 4 + 2] = signs[i][2];
        outQuad16[i * 4 + 3] = signs[i][3];
    }
    return true;
}

}  // namespace livebeauty::math
