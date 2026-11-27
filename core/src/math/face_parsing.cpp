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

constexpr float kParseEmaAlpha = 0.46f;
constexpr float kHairEmaAlpha = 0.74f;
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
            const float effAlpha = (std::fabs(cur - ema[idx]) > 140.0f) ? std::max(alpha, 0.85f) : alpha;
            const float sm = ema[idx] * (1.0f - effAlpha) + cur * effAlpha;
            ema[idx] = sm;
            outR8[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(sm)), 0, 255));
        }
    }
    (void)total;
}

// Center-weighted 3x3 Gaussian [1,2,1; 2,4,2; 1,2,1]/16 + crisp motion-snapping EMA for hair masks:
// preserves sharp hairline edges and eliminates temporal ghost trails on head movement while still
// suppressing 1-2 LSB dynamic-INT8 jitter on stationary hair pixels.
void gaussian3x3AndHairEma(
    const uint8_t* raw,
    float* ema,
    float alpha,
    uint8_t* outR8,
    int size) {
    if (!outR8) return;
    for (int y = 0; y < size; ++y) {
        const int ym = std::max(0, y - 1);
        const int yp = std::min(size - 1, y + 1);
        const int rowM = ym * size;
        const int rowC = y * size;
        const int rowP = yp * size;
        for (int x = 0; x < size; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(size - 1, x + 1);
            const int sum =
                raw[rowM + xm] + 2 * raw[rowM + x] + raw[rowM + xp] +
                2 * raw[rowC + xm] + 4 * raw[rowC + x] + 2 * raw[rowC + xp] +
                raw[rowP + xm] + 2 * raw[rowP + x] + raw[rowP + xp];
            const float cur = static_cast<float>(sum) * (1.0f / 16.0f);
            const int idx = rowC + x;
            const float diff = std::fabs(cur - ema[idx]);
            const float effAlpha = (diff > 56.0f) ? std::max(alpha, 0.96f) : alpha;
            const float sm = ema[idx] * (1.0f - effAlpha) + cur * effAlpha;
            ema[idx] = sm;
            outR8[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(sm)), 0, 255));
        }
    }
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
        gaussian3x3AndHairEma(rawHair_.data(), emaHair_.data(), hasHairHistory_ ? kHairEmaAlpha : 1.0f, outHairR8, kOutSize);
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

namespace {

inline float evalInnerFaceMask(float u, float v, const CropRect* primary, const CropRect* secondary) {
    auto evalOne = [](float px, float py, const CropRect* b) -> float {
        if (!b || b->width < 0.04f || b->height < 0.04f) return 0.0f;
        const float cx = b->left + 0.5f * b->width;
        const float cy = b->top + 0.5f * b->height;
        const float rx = std::max(0.5f * b->width, 0.02f);
        const float ry = std::max(0.5f * b->height, 0.02f);
        const float dx = (px - cx) / rx;
        const float dy = (py - cy) / ry;
        // Inner face core (0.55..0.75) excludes jawline/neck boundary so jaw contour isn't muted.
        return 1.0f - detail::smoothstep(0.55f, 0.75f, dx * dx + dy * dy);
    };
    return std::max(evalOne(u, v, primary), evalOne(u, v, secondary));
}

}  // namespace

bool FaceParsingMapper::mapMulticlass6LogitsNhwc(
    const float* logitsNhwc6,
    int srcW,
    int srcH,
    uint8_t* outOcclusionR8,
    uint8_t* outHairR8,
    uint8_t* outPersonR8,
    const CropRect* primaryFaceBounds,
    const CropRect* secondaryFaceBounds) {
    return mapMulticlass6LogitsNhwcRegion(
        logitsNhwc6, srcW, srcH, 0, 0, srcW, srcH,
        outOcclusionR8, outHairR8, outPersonR8,
        primaryFaceBounds, secondaryFaceBounds);
}

bool FaceParsingMapper::mapMulticlass6LogitsNhwcRegion(
    const float* logitsNhwc6,
    int srcW,
    int srcH,
    int padX,
    int padY,
    int validW,
    int validH,
    uint8_t* outOcclusionR8,
    uint8_t* outHairR8,
    uint8_t* outPersonR8,
    const CropRect* primaryFaceBounds,
    const CropRect* secondaryFaceBounds) {
    if (!logitsNhwc6 || srcW <= 0 || srcH <= 0 || srcW > 4096 || srcH > 4096 || !outOcclusionR8) {
        return false;
    }
    const int rx0 = std::clamp(padX, 0, srcW - 1);
    const int ry0 = std::clamp(padY, 0, srcH - 1);
    const int rw = std::clamp(validW, 1, srcW - rx0);
    const int rh = std::clamp(validH, 1, srcH - ry0);
    const bool useBox2x2 = (rw >= kOutSize * 2 || rh >= kOutSize * 2);
    const float invOut = 1.0f / static_cast<float>(kOutSize);

    for (int y = 0; y < kOutSize; ++y) {
        const float fy = static_cast<float>(ry0) + (static_cast<float>(y) + 0.5f) * static_cast<float>(rh) * invOut;
        // v3.9.5 (segmentation-n4): 2x2 box center offset is -0.5 (not -0.25) so pixel centers align with 0-bias.
        const int sy0 = std::clamp(static_cast<int>(std::floor(fy - 0.5f)), ry0, ry0 + rh - 1);
        const int sy1 = useBox2x2 ? std::clamp(sy0 + 1, ry0, ry0 + rh - 1) : sy0;
        const size_t row0 = static_cast<size_t>(sy0) * static_cast<size_t>(srcW);
        const size_t row1 = static_cast<size_t>(sy1) * static_cast<size_t>(srcW);
        const float vNorm = (static_cast<float>(y) + 0.5f) * invOut;
        const int dstRow = y * kOutSize;
        for (int x = 0; x < kOutSize; ++x) {
            const float fx = static_cast<float>(rx0) + (static_cast<float>(x) + 0.5f) * static_cast<float>(rw) * invOut;
            const int sx0 = std::clamp(static_cast<int>(std::floor(fx - 0.5f)), rx0, rx0 + rw - 1);
            const int sx1 = useBox2x2 ? std::clamp(sx0 + 1, rx0, rx0 + rw - 1) : sx0;

            float lgAvg[6];
            bool finiteAll = true;
            if (useBox2x2 && (sx1 != sx0 || sy1 != sy0)) {
                const float* l00 = logitsNhwc6 + (row0 + static_cast<size_t>(sx0)) * 6u;
                const float* l10 = logitsNhwc6 + (row0 + static_cast<size_t>(sx1)) * 6u;
                const float* l01 = logitsNhwc6 + (row1 + static_cast<size_t>(sx0)) * 6u;
                const float* l11 = logitsNhwc6 + (row1 + static_cast<size_t>(sx1)) * 6u;
                for (int c = 0; c < 6; ++c) {
                    const float avg = 0.25f * (l00[c] + l10[c] + l01[c] + l11[c]);
                    if (!std::isfinite(avg)) { finiteAll = false; break; }
                    lgAvg[c] = avg;
                }
            } else {
                const float* lg = logitsNhwc6 + (row0 + static_cast<size_t>(sx0)) * 6u;
                for (int c = 0; c < 6; ++c) {
                    if (!std::isfinite(lg[c])) { finiteAll = false; break; }
                    lgAvg[c] = lg[c];
                }
            }
            const int idx = dstRow + x;
            if (!finiteAll) {
                rawOcc_[idx] = 255;
                rawHair_[idx] = 0;
                rawPerson_[idx] = 0;
                continue;
            }
            // Numerically stable 6-way softmax: [0:bg, 1:hair, 2:body_skin(incl. hands), 3:face_skin, 4:clothes, 5:accessories]
            float maxL = lgAvg[0];
            for (int c = 1; c < 6; ++c) {
                if (lgAvg[c] > maxL) maxL = lgAvg[c];
            }
            float p[6];
            float sumE = 0.0f;
            for (int c = 0; c < 6; ++c) {
                p[c] = std::exp(std::max(lgAvg[c] - maxL, -20.0f));
                sumE += p[c];
            }
            const float invSum = 1.0f / std::max(sumE, 1e-6f);
            for (int c = 0; c < 6; ++c) p[c] *= invSum;

            // v3.9.5 (makeup-1): Inside inner face core where body_skin dominates face_skin (e.g. hand on lips/chin),
            // relax pOcc threshold proportionally to handInFace so uncertain fingers (p2=0.60, p3=0.30) also occlude
            // lipstick/blush cleanly while keeping non-hand face skin (handInFace=0) at exact original thresholds.
            const float uNorm = (static_cast<float>(x) + 0.5f) * invOut;
            const float inner = evalInnerFaceMask(uNorm, vNorm, primaryFaceBounds, secondaryFaceBounds);
            const float bodyDom = detail::smoothstep(0.10f, 0.30f, p[2] - p[3]);
            const float handInFace = inner * bodyDom;
            const float pSkinOcc = p[3] + p[2] * (1.0f - handInFace);
            const float rawOccluder = p[1] + p[4] + p[5] + p[2] * handInFace;
            const float pOcc = detail::smoothstep(
                0.62f - 0.30f * handInFace,
                0.90f - 0.32f * handInFace,
                rawOccluder - (0.90f - 0.55f * handInFace) * pSkinOcc);
            const float pSkinHair = p[2] + p[3];
            const float pHair = detail::smoothstep(0.32f, 0.72f, p[1] - 0.75f * pSkinHair);
            const float pPerson = detail::clampf(1.0f - p[0], 0.0f, 1.0f);

            rawOcc_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((1.0f - pOcc) * 255.0f)), 0, 255));
            rawHair_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(pHair * 255.0f)), 0, 255));
            rawPerson_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(pPerson * 255.0f)), 0, 255));
        }
    }

    boxBlur3x3AndEma(rawOcc_.data(), emaOcc_.data(), hasOccHistory_ ? kParseEmaAlpha : 1.0f, outOcclusionR8, kOutSize);
    hasOccHistory_ = true;
    if (outHairR8) {
        gaussian3x3AndHairEma(rawHair_.data(), emaHair_.data(), hasHairHistory_ ? kHairEmaAlpha : 1.0f, outHairR8, kOutSize);
        hasHairHistory_ = true;
    }
    if (outPersonR8) {
        boxBlur3x3AndEma(rawPerson_.data(), emaPerson_.data(), hasPersonHistory_ ? kParseEmaAlpha : 1.0f, outPersonR8, kOutSize);
        hasPersonHistory_ = true;
    }
    return true;
}

bool FaceParsingMapper::mapMulticlass6PlanarProbabilities(
    const float* const planes[6],
    int srcW,
    int srcH,
    uint8_t* outOcclusionR8,
    uint8_t* outHairR8,
    uint8_t* outPersonR8,
    const CropRect* primaryFaceBounds,
    const CropRect* secondaryFaceBounds) {
    if (!planes || srcW <= 0 || srcH <= 0 || srcW > 4096 || srcH > 4096 || !outOcclusionR8) {
        return false;
    }
    for (int c = 0; c < 6; ++c) {
        if (!planes[c]) return false;
    }
    const float invOut = 1.0f / static_cast<float>(kOutSize);
    const bool useBox2x2 = (srcW >= kOutSize * 2 || srcH >= kOutSize * 2);

    for (int y = 0; y < kOutSize; ++y) {
        const float fy = (static_cast<float>(y) + 0.5f) * static_cast<float>(srcH) * invOut;
        const int sy0 = std::clamp(static_cast<int>(std::floor(fy - 0.5f)), 0, srcH - 1);
        const int sy1 = useBox2x2 ? std::clamp(sy0 + 1, 0, srcH - 1) : sy0;
        const size_t row0 = static_cast<size_t>(sy0) * static_cast<size_t>(srcW);
        const size_t row1 = static_cast<size_t>(sy1) * static_cast<size_t>(srcW);
        const float vNorm = (static_cast<float>(y) + 0.5f) * invOut;
        const int dstRow = y * kOutSize;
        for (int x = 0; x < kOutSize; ++x) {
            const float fx = (static_cast<float>(x) + 0.5f) * static_cast<float>(srcW) * invOut;
            const int sx0 = std::clamp(static_cast<int>(std::floor(fx - 0.5f)), 0, srcW - 1);
            const int sx1 = useBox2x2 ? std::clamp(sx0 + 1, 0, srcW - 1) : sx0;
            float p[6];
            float sumP = 0.0f;
            bool finiteAll = true;
            const bool doBox = useBox2x2 && (sx1 != sx0 || sy1 != sy0);
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                if (doBox) {
                    const float v00 = planes[c][row0 + static_cast<size_t>(sx0)];
                    const float v10 = planes[c][row0 + static_cast<size_t>(sx1)];
                    const float v01 = planes[c][row1 + static_cast<size_t>(sx0)];
                    const float v11 = planes[c][row1 + static_cast<size_t>(sx1)];
                    if (!std::isfinite(v00) || !std::isfinite(v10) || !std::isfinite(v01) || !std::isfinite(v11)) {
                        finiteAll = false;
                        break;
                    }
                    v = 0.25f * (v00 + v10 + v01 + v11);
                } else {
                    v = planes[c][row0 + static_cast<size_t>(sx0)];
                    if (!std::isfinite(v)) {
                        finiteAll = false;
                        break;
                    }
                }
                p[c] = detail::clampf(v, 0.0f, 1.0f);
                sumP += p[c];
            }
            const int idx = dstRow + x;
            if (!finiteAll || sumP <= 1e-5f) {
                rawOcc_[idx] = 255;
                rawHair_[idx] = 0;
                rawPerson_[idx] = 0;
                continue;
            }
            const float invSum = 1.0f / sumP;
            for (int c = 0; c < 6; ++c) p[c] *= invSum;

            const float uNorm = (static_cast<float>(x) + 0.5f) * invOut;
            const float inner = evalInnerFaceMask(uNorm, vNorm, primaryFaceBounds, secondaryFaceBounds);
            const float bodyDom = detail::smoothstep(0.10f, 0.30f, p[2] - p[3]);
            const float handInFace = inner * bodyDom;
            const float pSkinOcc = p[3] + p[2] * (1.0f - handInFace);
            const float rawOccluder = p[1] + p[4] + p[5] + p[2] * handInFace;
            const float pOcc = detail::smoothstep(
                0.62f - 0.30f * handInFace,
                0.90f - 0.32f * handInFace,
                rawOccluder - (0.90f - 0.55f * handInFace) * pSkinOcc);
            const float pSkinHair = p[2] + p[3];
            const float pHair = detail::smoothstep(0.32f, 0.72f, p[1] - 0.75f * pSkinHair);
            const float pPerson = detail::clampf(1.0f - p[0], 0.0f, 1.0f);

            rawOcc_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((1.0f - pOcc) * 255.0f)), 0, 255));
            rawHair_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(pHair * 255.0f)), 0, 255));
            rawPerson_[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(pPerson * 255.0f)), 0, 255));
        }
    }

    boxBlur3x3AndEma(rawOcc_.data(), emaOcc_.data(), hasOccHistory_ ? kParseEmaAlpha : 1.0f, outOcclusionR8, kOutSize);
    hasOccHistory_ = true;
    if (outHairR8) {
        gaussian3x3AndHairEma(rawHair_.data(), emaHair_.data(), hasHairHistory_ ? kHairEmaAlpha : 1.0f, outHairR8, kOutSize);
        hasHairHistory_ = true;
    }
    if (outPersonR8) {
        boxBlur3x3AndEma(rawPerson_.data(), emaPerson_.data(), hasPersonHistory_ ? kParseEmaAlpha : 1.0f, outPersonR8, kOutSize);
        hasPersonHistory_ = true;
    }
    return true;
}

float FaceParsingMapper::estimateLowerFaceOcclusion(
    const uint8_t* occMaskR8,
    int maskSize,
    const CropRect& faceBounds) {
    if (!occMaskR8 || maskSize <= 0 || faceBounds.width < 0.04f || faceBounds.height < 0.04f) {
        return 0.0f;
    }
    const float xLo = detail::clampf(faceBounds.left + faceBounds.width * 0.18f, 0.0f, 1.0f);
    const float xHi = detail::clampf(faceBounds.left + faceBounds.width * 0.82f, xLo, 1.0f);
    const float yLo = detail::clampf(faceBounds.top + faceBounds.height * 0.48f, 0.0f, 1.0f);
    const float yHi = detail::clampf(faceBounds.top + faceBounds.height * 0.92f, yLo, 1.0f);
    const int ix0 = std::clamp(static_cast<int>(xLo * maskSize), 0, maskSize - 1);
    const int ix1 = std::clamp(static_cast<int>(xHi * maskSize), ix0, maskSize - 1);
    const int iy0 = std::clamp(static_cast<int>(yLo * maskSize), 0, maskSize - 1);
    const int iy1 = std::clamp(static_cast<int>(yHi * maskSize), iy0, maskSize - 1);
    float occSum = 0.0f;
    int cnt = 0;
    for (int y = iy0; y <= iy1; ++y) {
        const int row = y * maskSize;
        for (int x = ix0; x <= ix1; ++x) {
            occSum += 1.0f - static_cast<float>(occMaskR8[row + x]) * kInv255;
            ++cnt;
        }
    }
    return (cnt > 0) ? detail::clampf(occSum / static_cast<float>(cnt), 0.0f, 1.0f) : 0.0f;
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
    uint8_t* outRgba64x64,
    float headRollRad) {
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
    const float cosR = std::cos(-headRollRad);
    const float sinR = std::sin(-headRollRad);
    for (int y = 0; y < kOutSize; ++y) {
        const float rawRelY = (static_cast<float>(y) + 0.5f) / kOutSize;
        for (int x = 0; x < kOutSize; ++x) {
            const float rawRelX = (static_cast<float>(x) + 0.5f) / kOutSize;
            const float cx = rawRelX - 0.5f;
            const float cy = rawRelY - 0.5f;
            const float relX = 0.5f + cx * cosR - cy * sinR;
            const float relY = 0.5f + cx * sinR + cy * cosR;
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
