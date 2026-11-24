#include "livebeauty/math/face_parsing.h"

#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LB_PARSE_NEON 1
#else
#define LB_PARSE_NEON 0
#endif

namespace livebeauty::math {

namespace {

constexpr float kParseEmaAlpha = 0.55f;
constexpr float kInv255 = 1.0f / 255.0f;

inline bool isOccluderClass(uint8_t cls) {
    // Explicit foreground occluders within 19+1 taxonomy; >=20 (e.g. ignore_index=255) stays neutral.
    return cls == static_cast<uint8_t>(FaceParsingClass::kEyeglasses) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kEarring) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kInnerMouth) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kNecklace) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kCloth) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kHair) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kHat) ||
           cls == static_cast<uint8_t>(FaceParsingClass::kHand);
}

inline bool isForegroundPersonClass(uint8_t cls) {
    return cls >= 1 && cls <= 18;
}

void boxBlur3x3AndEma(
    const uint8_t* raw,
    float* ema,
    float alpha,
    uint8_t* outR8,
    int size) {
    if (!outR8) return;
    const int total = size * size;
    for (int y = 0; y < size; ++y) {
        const int y0 = std::max(0, y - 1);
        const int y1 = std::min(size - 1, y + 1);
        for (int x = 0; x < size; ++x) {
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(size - 1, x + 1);
            int sum = 0, cnt = 0;
            for (int ny = y0; ny <= y1; ++ny) {
                const int rOff = ny * size;
                for (int nx = x0; nx <= x1; ++nx) {
                    sum += raw[rOff + nx];
                    ++cnt;
                }
            }
            const float cur = static_cast<float>(sum) / static_cast<float>(cnt);
            const int idx = y * size + x;
            const float sm = ema[idx] * (1.0f - alpha) + cur * alpha;
            ema[idx] = sm;
            outR8[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(sm)), 0, 255));
        }
    }
    (void)total;
}

}  // namespace

// ============================================================================
// FaceParsingMapper
// ============================================================================

FaceParsingMapper::FaceParsingMapper() {
    emaOcc_.fill(255.0f);
    emaHair_.fill(0.0f);
    emaPerson_.fill(0.0f);
}

void FaceParsingMapper::reset() {
    hasOccHistory_ = false;
    hasHairHistory_ = false;
    hasPersonHistory_ = false;
    emaOcc_.fill(255.0f);
    emaHair_.fill(0.0f);
    emaPerson_.fill(0.0f);
}

bool FaceParsingMapper::mapClassIds(
    const uint8_t* classIds,
    int srcW,
    int srcH,
    uint8_t* outOcclusionR8,
    uint8_t* outHairR8,
    uint8_t* outPersonR8) {
    if (!classIds || srcW <= 0 || srcH <= 0 || srcW > 8192 || srcH > 8192 || !outOcclusionR8) return false;

    for (int y = 0; y < kOutSize; ++y) {
        const int sy = std::clamp(static_cast<int>((static_cast<int64_t>(y) * srcH) / kOutSize), 0, srcH - 1);
        const size_t srcRow = static_cast<size_t>(sy) * static_cast<size_t>(srcW);
        const int dstRow = y * kOutSize;
        for (int x = 0; x < kOutSize; ++x) {
            const int sx = std::clamp(static_cast<int>((static_cast<int64_t>(x) * srcW) / kOutSize), 0, srcW - 1);
            const uint8_t cls = classIds[srcRow + static_cast<size_t>(sx)];
            const int idx = dstRow + x;
            // Background (0) and ignore labels (>=20) stay neutral (255) for occlusion;
            // explicit occluders (eyeglasses, inner-mouth teeth, hair bangs, hat, cloth, hand) go to 0!
            rawOcc_[idx] = isOccluderClass(cls) ? 0 : 255;
            rawHair_[idx] = (cls == static_cast<uint8_t>(FaceParsingClass::kHair)) ? 255 : 0;
            rawPerson_[idx] = isForegroundPersonClass(cls) ? 255 : 0;
        }
    }

    boxBlur3x3AndEma(rawOcc_.data(), emaOcc_.data(), hasOccHistory_ ? kParseEmaAlpha : 1.0f, outOcclusionR8, kOutSize);
    hasOccHistory_ = true;
    if (outHairR8) {
        boxBlur3x3AndEma(rawHair_.data(), emaHair_.data(), hasHairHistory_ ? kParseEmaAlpha : 1.0f, outHairR8, kOutSize);
        hasHairHistory_ = true;
    }
    if (outPersonR8) {
        boxBlur3x3AndEma(rawPerson_.data(), emaPerson_.data(), hasPersonHistory_ ? kParseEmaAlpha : 1.0f, outPersonR8, kOutSize);
        hasPersonHistory_ = true;
    }
    return true;
}

bool FaceParsingMapper::mapLogitsNchw(
    const float* logitsNchw,
    int numClasses,
    int srcW,
    int srcH,
    uint8_t* outOcclusionR8,
    uint8_t* outHairR8,
    uint8_t* outPersonR8) {
    if (!logitsNchw || numClasses < 19 || srcW <= 0 || srcH <= 0 || srcW > 8192 || srcH > 8192 || !outOcclusionR8) {
        return false;
    }
    const size_t plane = static_cast<size_t>(srcW) * static_cast<size_t>(srcH);
    argmaxScratch_.resize(plane);
    const int cappedClasses = std::min(numClasses, 32);

    for (size_t i = 0; i < plane; ++i) {
        float bestVal = logitsNchw[i];
        uint8_t bestCls = 0;
        for (int c = 1; c < cappedClasses; ++c) {
            const float v = logitsNchw[static_cast<size_t>(c) * plane + i];
            if (v > bestVal) {
                bestVal = v;
                bestCls = static_cast<uint8_t>(c);
            }
        }
        argmaxScratch_[i] = bestCls;
    }
    return mapClassIds(argmaxScratch_.data(), srcW, srcH, outOcclusionR8, outHairR8, outPersonR8);
}

// ============================================================================
// NeuralSkinResidualAdapter
// ============================================================================

bool NeuralSkinResidualAdapter::packExternalTensorNchw(
    const float* tensorNchw,
    int channels,
    int srcW,
    int srcH,
    uint8_t* outRgba64x64) {
    if (!tensorNchw || (channels != 3 && channels != 4) || srcW <= 0 || srcH <= 0 ||
        srcW > 8192 || srcH > 8192 || !outRgba64x64) {
        return false;
    }
    const size_t plane = static_cast<size_t>(srcW) * static_cast<size_t>(srcH);
    const float* chR = tensorNchw;
    const float* chG = tensorNchw + plane;
    const float* chB = tensorNchw + plane * 2;
    const float* chA = (channels == 4) ? (tensorNchw + plane * 3) : nullptr;

    for (int y = 0; y < kOutSize; ++y) {
        const int sy = std::clamp(static_cast<int>((static_cast<int64_t>(y) * srcH) / kOutSize), 0, srcH - 1);
        for (int x = 0; x < kOutSize; ++x) {
            const int sx = std::clamp(static_cast<int>((static_cast<int64_t>(x) * srcW) / kOutSize), 0, srcW - 1);
            const size_t sIdx = static_cast<size_t>(sy) * static_cast<size_t>(srcW) + static_cast<size_t>(sx);
            uint8_t* dst = outRgba64x64 + (y * kOutSize + x) * 4;
            const float dR = detail::clampf(chR[sIdx], -0.24f, 0.24f);
            const float dG = detail::clampf(chG[sIdx], -0.24f, 0.24f);
            const float dB = detail::clampf(chB[sIdx], -0.24f, 0.24f);
            const float conf = chA ? detail::clampf(chA[sIdx], 0.0f, 1.0f) : 1.0f;
            dst[0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dR * 2.0f) * 255.0f)), 0, 255));
            dst[1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dG * 2.0f) * 255.0f)), 0, 255));
            dst[2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dB * 2.0f) * 255.0f)), 0, 255));
            dst[3] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(conf * 255.0f)), 0, 255));
        }
    }
    return true;
}

bool NeuralSkinResidualAdapter::evaluateMicroKernel(
    const uint8_t* rgbaTopDown,
    int srcWidth,
    int srcHeight,
    const CropRect& faceBounds,
    float strength,
    uint8_t* outRgba64x64) {
    if (!rgbaTopDown || !outRgba64x64 || srcWidth <= 0 || srcHeight <= 0) return false;
    if (strength <= 0.01f || faceBounds.width < 0.04f || faceBounds.height < 0.04f) return false;

    const float roiL = detail::clampf(faceBounds.left, 0.0f, 0.96f);
    const float roiT = detail::clampf(faceBounds.top, 0.0f, 0.96f);
    const float roiW = detail::clampf(faceBounds.width, 0.04f, 1.0f - roiL);
    const float roiH = detail::clampf(faceBounds.height, 0.04f, 1.0f - roiT);

    // 1. Sample 64x64 ROI & fine luminance
    for (int y = 0; y < kOutSize; ++y) {
        const float v = detail::clampf(roiT + ((static_cast<float>(y) + 0.5f) / kOutSize) * roiH, 0.0f, 1.0f);
        const int sy = std::clamp(static_cast<int>(v * (srcHeight - 1)), 0, srcHeight - 1);
        for (int x = 0; x < kOutSize; ++x) {
            const float u = detail::clampf(roiL + ((static_cast<float>(x) + 0.5f) / kOutSize) * roiW, 0.0f, 1.0f);
            const int sx = std::clamp(static_cast<int>(u * (srcWidth - 1)), 0, srcWidth - 1);
            const uint8_t* p = rgbaTopDown + (sy * srcWidth + sx) * 4;
            const int idx = y * kOutSize + x;
            const float r = p[0] * kInv255;
            const float g = p[1] * kInv255;
            const float b = p[2] * kInv255;
            rPlane_[idx] = r;
            gPlane_[idx] = g;
            bPlane_[idx] = b;
            lumaFine_[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    // 2. Separable depthwise [1, 2, 1]/4 (mid-band 2px) and dilated [1, 2, 1]/4 stride-3 (structural shadow 6px)
    for (int y = 0; y < kOutSize; ++y) {
        const int ym1 = std::max(0, y - 1), yp1 = std::min(kOutSize - 1, y + 1);
        for (int x = 0; x < kOutSize; ++x) {
            const int xm1 = std::max(0, x - 1), xp1 = std::min(kOutSize - 1, x + 1);
            lumaMid_[y * kOutSize + x] = 0.0625f * (
                lumaFine_[ym1 * kOutSize + xm1] + 2.0f * lumaFine_[ym1 * kOutSize + x] + lumaFine_[ym1 * kOutSize + xp1] +
                2.0f * lumaFine_[y * kOutSize + xm1] + 4.0f * lumaFine_[y * kOutSize + x] + 2.0f * lumaFine_[y * kOutSize + xp1] +
                lumaFine_[yp1 * kOutSize + xm1] + 2.0f * lumaFine_[yp1 * kOutSize + x] + lumaFine_[yp1 * kOutSize + xp1]);
        }
    }
    for (int y = 0; y < kOutSize; ++y) {
        const int ym3 = std::max(0, y - 3), yp3 = std::min(kOutSize - 1, y + 3);
        for (int x = 0; x < kOutSize; ++x) {
            const int xm3 = std::max(0, x - 3), xp3 = std::min(kOutSize - 1, x + 3);
            lumaCoarse_[y * kOutSize + x] = 0.0625f * (
                lumaMid_[ym3 * kOutSize + xm3] + 2.0f * lumaMid_[ym3 * kOutSize + x] + lumaMid_[ym3 * kOutSize + xp3] +
                2.0f * lumaMid_[y * kOutSize + xm3] + 4.0f * lumaMid_[y * kOutSize + x] + 2.0f * lumaMid_[y * kOutSize + xp3] +
                lumaMid_[yp3 * kOutSize + xm3] + 2.0f * lumaMid_[yp3 * kOutSize + x] + lumaMid_[yp3 * kOutSize + xp3]);
        }
    }

    // 3. Pointwise 1x1 residual synthesis: lift tear-trough / nasolabial / acne-pit structural dips (coarse > mid)
    //    while 100% preserving micro-pore detail (lumaFine - lumaMid untouched!)
    const float str = detail::clampf(strength, 0.0f, 1.0f);
    for (int y = 0; y < kOutSize; ++y) {
        const float relY = (static_cast<float>(y) + 0.5f) / kOutSize;
        for (int x = 0; x < kOutSize; ++x) {
            const float relX = (static_cast<float>(x) + 0.5f) / kOutSize;
            const int idx = y * kOutSize + x;
            uint8_t* dst = outRgba64x64 + idx * 4;

            const float ex = (relX - 0.5f) / 0.46f;
            const float ey = (relY - 0.5f) / 0.46f;
            const float ovalDist = ex * ex + ey * ey;
            const bool inEyes = (relY >= 0.27f && relY <= 0.44f && relX >= 0.18f && relX <= 0.82f);
            const bool inMouth = (relY >= 0.67f && relY <= 0.82f && relX >= 0.30f && relX <= 0.70f);
            if (ovalDist >= 1.0f || inEyes || inMouth || lumaFine_[idx] < 0.10f) {
                dst[0] = 128; dst[1] = 128; dst[2] = 128; dst[3] = 0;
                continue;
            }

            // Asymmetric soft-ReLU on structural shadow deficit (lifts hollow shadows without flattening nose highlight!)
            const float shadowDip = std::max(lumaCoarse_[idx] - lumaMid_[idx], 0.0f);
            const float edgeGuard = 1.0f - detail::clampf((shadowDip - 0.09f) / 0.08f, 0.0f, 1.0f);
            const float lift = detail::clampf(shadowDip * 0.78f * edgeGuard * str, 0.0f, 0.14f);

            // Mild redness de-blotching
            const float cR = rPlane_[idx], cG = gPlane_[idx], cB = bPlane_[idx];
            const float excessRed = std::max((cR - cG) - 0.14f, 0.0f);
            const float coolComp = detail::clampf(excessRed * 0.35f * str, 0.0f, 0.05f);

            const float dR = detail::clampf(lift * 0.92f - coolComp * 0.4f, -0.22f, 0.22f);
            const float dG = detail::clampf(lift * 1.00f + coolComp * 0.6f, -0.22f, 0.22f);
            const float dB = detail::clampf(lift * 1.02f + coolComp * 0.5f, -0.22f, 0.22f);

            const float skinLike = detail::clampf((cR - cB) - 0.02f, 0.0f, 0.14f) / 0.14f;
            const float ovalFade = detail::clampf((1.0f - ovalDist) / 0.28f, 0.0f, 1.0f);
            const int conf = std::clamp(static_cast<int>(std::lround(skinLike * ovalFade * 255.0f)), 0, 255);

            dst[0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dR * 2.0f) * 255.0f)), 0, 255));
            dst[1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dG * 2.0f) * 255.0f)), 0, 255));
            dst[2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + dB * 2.0f) * 255.0f)), 0, 255));
            dst[3] = static_cast<uint8_t>(conf);
        }
    }
    return true;
}

}  // namespace livebeauty::math
