#include "livebeauty/math/face_aux_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LB_HAS_NEON 1
#else
#define LB_HAS_NEON 0
#endif

namespace livebeauty::math {

namespace {

constexpr float kTemporalAlpha = 0.45f;
constexpr float kExprAlpha = 0.35f;
constexpr float kInv255 = 1.0f / 255.0f;

constexpr float kSkinProbeUvs[16] = {
    0.50f, 0.16f,
    0.38f, 0.18f,
    0.62f, 0.18f,
    0.26f, 0.54f,
    0.34f, 0.58f,
    0.74f, 0.54f,
    0.66f, 0.58f,
    0.50f, 0.88f,
};

constexpr int kRingOffsets[16] = {
    -2, 0,
     2, 0,
     0, -2,
     0,  2,
    -2, -2,
     2, -2,
    -2,  2,
     2,  2,
};

inline void rgbToYCbCr(uint8_t r, uint8_t g, uint8_t b, float& y, float& cb, float& cr) {
    const float rf = static_cast<float>(r);
    const float gf = static_cast<float>(g);
    const float bf = static_cast<float>(b);
    y  = 0.299f * rf + 0.587f * gf + 0.114f * bf;
    cb = 128.0f - 0.168736f * rf - 0.331264f * gf + 0.5f * bf;
    cr = 128.0f + 0.5f * rf - 0.418688f * gf - 0.081312f * bf;
}

}  // namespace

// ============================================================================
// FaceExpressionBus
// ============================================================================

void FaceExpressionBus::reset() {
    smSmile_ = 0.0f;
    smJaw_ = 0.0f;
    smBrow_ = 0.0f;
    smBlink_ = 0.0f;
    smPuff_ = 0.0f;
    smTongue_ = 0.0f;
    animTick_ = 0;
    latest_ = ExpressionReaction{};
}

ExpressionReaction FaceExpressionBus::update(const BlendshapeInput* sample) {
    const BlendshapeInput in = sample ? *sample : BlendshapeInput{};
    smSmile_  += (detail::clampf(in.smile, 0.0f, 1.0f) - smSmile_) * kExprAlpha;
    smJaw_    += (detail::clampf(in.jawOpen, 0.0f, 1.0f) - smJaw_) * kExprAlpha;
    smBrow_   += (detail::clampf(in.browUp, 0.0f, 1.0f) - smBrow_) * kExprAlpha;
    smBlink_  += (detail::clampf(in.blink, 0.0f, 1.0f) - smBlink_) * kExprAlpha;
    smPuff_   += (detail::clampf(in.cheekPuff, 0.0f, 1.0f) - smPuff_) * kExprAlpha;
    smTongue_ += (detail::clampf(in.tongueOut, 0.0f, 1.0f) - smTongue_) * kExprAlpha;

    const float smileAct  = detail::smoothstep(0.32f, 0.78f, smSmile_);
    const float jawAct    = detail::smoothstep(0.28f, 0.75f, smJaw_);
    const float browAct   = detail::smoothstep(0.28f, 0.75f, smBrow_);
    const float puffAct   = detail::smoothstep(0.30f, 0.75f, smPuff_);
    const float tongueAct = detail::smoothstep(0.30f, 0.75f, smTongue_);

    const float energy = std::max({smileAct, jawAct * 0.85f, browAct * 0.90f, puffAct * 0.80f, tongueAct * 0.85f});
    ExpressionTrigger trig = ExpressionTrigger::kNone;
    if (jawAct >= 0.35f) {
        trig = ExpressionTrigger::kMouthOpen;
    } else if (tongueAct >= 0.35f) {
        trig = ExpressionTrigger::kTongueOut;
    } else if (browAct >= 0.35f) {
        trig = ExpressionTrigger::kBrowRaise;
    } else if (puffAct >= 0.35f) {
        trig = ExpressionTrigger::kCheekPuff;
    } else if (smileAct >= 0.30f) {
        trig = ExpressionTrigger::kSmile;
    } else if (smBlink_ >= 0.65f) {
        trig = ExpressionTrigger::kBlink;
    }

    if (trig != ExpressionTrigger::kNone) {
        animTick_ = (animTick_ + 1) & 0x7FFF;
    } else {
        animTick_ = 0;
    }

    ExpressionReaction out{};
    out.scaleMultiplier = 1.0f + energy * 0.14f;
    out.offsetYBoost = -browAct * 0.035f - smileAct * 0.020f;
    out.opacityBoost = 1.0f;
    out.trigger = trig;
    out.spriteFrameOffset = animTick_;
    out.smile = smSmile_;
    out.mouthOpen = smJaw_;
    out.browUp = smBrow_;
    out.blink = smBlink_;
    out.cheekPuff = smPuff_;
    out.tongueOut = smTongue_;
    latest_ = out;
    return out;
}

// ============================================================================
// FaceOcclusionSolver
// ============================================================================

FaceOcclusionSolver::FaceOcclusionSolver() {
    emaMask_.fill(255.0f);
}

void FaceOcclusionSolver::reset() {
    hasHistory_ = false;
    emaMask_.fill(255.0f);
}

bool FaceOcclusionSolver::boundsFromLandmarks(
    const float* mpXy, int pointCount, CropRect& outBounds, float* outRollRad) {
    if (!mpXy || pointCount < 468) return false;
    float minX = 1.0f, minY = 1.0f;
    float maxX = 0.0f, maxY = 0.0f;
    for (int i = 0; i < 468; ++i) {
        const float x = mpXy[i * 2];
        const float y = mpXy[i * 2 + 1];
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    if (maxX <= minX + 0.03f || maxY <= minY + 0.03f) return false;
    const float padX = (maxX - minX) * 0.06f;
    const float padY = (maxY - minY) * 0.06f;
    const float l = detail::clampf(minX - padX, 0.0f, 0.96f);
    const float t = detail::clampf(minY - padY, 0.0f, 0.96f);
    const float r = detail::clampf(maxX + padX, l + 0.04f, 1.0f);
    const float b = detail::clampf(maxY + padY, t + 0.04f, 1.0f);
    outBounds = {l, t, r - l, b - t};
    if (outRollRad) {
        // Vertical face axis from forehead(10) to chin(152) in top-down UV: upright is dx=0, dy>0
        const float dx = mpXy[152 * 2] - mpXy[10 * 2];
        const float dy = mpXy[152 * 2 + 1] - mpXy[10 * 2 + 1];
        *outRollRad = (std::hypot(dx, dy) > 1e-4f) ? std::atan2(-dx, dy) : 0.0f;
    }
    return true;
}

void FaceOcclusionSolver::fillNeutral(uint8_t* outMaskR8) {
    if (!outMaskR8) return;
    const float alpha = hasHistory_ ? kTemporalAlpha : 1.0f;
#if LB_HAS_NEON
    const float32x4_t vOneMinusA = vdupq_n_f32(1.0f - alpha);
    const float32x4_t vAdd = vdupq_n_f32(255.0f * alpha);
    const float32x4_t vHalf = vdupq_n_f32(0.5f);
    for (int i = 0; i < kMaskPixels; i += 4) {
        float32x4_t prev = vld1q_f32(&emaMask_[i]);
        float32x4_t sm = vmlaq_f32(vAdd, prev, vOneMinusA);
        vst1q_f32(&emaMask_[i], sm);
        uint32x4_t u32 = vcvtq_u32_f32(vaddq_f32(sm, vHalf));
        uint16x4_t u16 = vqmovn_u32(u32);
        uint8x8_t u8 = vqmovn_u16(vcombine_u16(u16, u16));
        outMaskR8[i]     = vget_lane_u8(u8, 0);
        outMaskR8[i + 1] = vget_lane_u8(u8, 1);
        outMaskR8[i + 2] = vget_lane_u8(u8, 2);
        outMaskR8[i + 3] = vget_lane_u8(u8, 3);
    }
#else
    for (int i = 0; i < kMaskPixels; ++i) {
        const float v = emaMask_[i] * (1.0f - alpha) + 255.0f * alpha;
        emaMask_[i] = v;
        outMaskR8[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(v)), 0, 255));
    }
#endif
    hasHistory_ = true;
}

void FaceOcclusionSolver::compute(
    const uint8_t* rgbaTopDown,
    int srcWidth,
    int srcHeight,
    const CropRect* faceBounds,
    uint8_t* outMaskR8,
    float headRollRad) {
    if (!outMaskR8) return;
    if (!rgbaTopDown || srcWidth <= 0 || srcHeight <= 0 || !faceBounds ||
        faceBounds->width < 0.04f || faceBounds->height < 0.04f) {
        fillNeutral(outMaskR8);
        return;
    }

    const float fL = detail::clampf(faceBounds->left, 0.0f, 1.0f);
    const float fT = detail::clampf(faceBounds->top, 0.0f, 1.0f);
    const float fR = detail::clampf(faceBounds->left + faceBounds->width, fL, 1.0f);
    const float fB = detail::clampf(faceBounds->top + faceBounds->height, fT, 1.0f);
    const float fW = fR - fL;
    const float fH = fB - fT;
    if (fW < 0.04f || fH < 0.04f) {
        fillNeutral(outMaskR8);
        return;
    }

    const float cosR = std::cos(-headRollRad);
    const float sinR = std::sin(-headRollRad);

    // 1. Sample adaptive skin YCbCr baseline from forehead & cheeks (roll-compensated)
    float sumY = 0.0f, sumCb = 0.0f, sumCr = 0.0f;
    int skinSamples = 0;
    for (int i = 0; i < 16; i += 2) {
        const float dx = kSkinProbeUvs[i] - 0.5f;
        const float dy = kSkinProbeUvs[i + 1] - 0.5f;
        const float rx = 0.5f + (dx * std::cos(headRollRad) - dy * std::sin(headRollRad));
        const float ry = 0.5f + (dx * std::sin(headRollRad) + dy * std::cos(headRollRad));
        const float u = detail::clampf(fL + fW * rx, 0.0f, 0.999f);
        const float v = detail::clampf(fT + fH * ry, 0.0f, 0.999f);
        const int sx = std::clamp(static_cast<int>(u * srcWidth), 0, srcWidth - 1);
        const int sy = std::clamp(static_cast<int>(v * srcHeight), 0, srcHeight - 1);
        const uint8_t* p = rgbaTopDown + (sy * srcWidth + sx) * 4;
        float y, cb, cr;
        rgbToYCbCr(p[0], p[1], p[2], y, cb, cr);
        if (y >= 38.0f && y <= 242.0f && cb >= 72.0f && cb <= 132.0f && cr >= 130.0f && cr <= 182.0f) {
            sumY += y;
            sumCb += cb;
            sumCr += cr;
            ++skinSamples;
        }
    }
    if (skinSamples < 3) {
        sumY = 135.0f * 3.0f;
        sumCb = 108.0f * 3.0f;
        sumCr = 150.0f * 3.0f;
        skinSamples = 3;
    }
    const float refY = sumY / static_cast<float>(skinSamples);
    const float refCb = sumCb / static_cast<float>(skinSamples);
    const float refCr = sumCr / static_cast<float>(skinSamples);

    // 2. Evaluate per-cell foreign occluder score inside roll-compensated face ellipse
    rawOcc_.fill(0);
    const int x0 = std::clamp(static_cast<int>(fL * kMaskSize), 0, kMaskSize - 1);
    const int x1 = std::clamp(static_cast<int>(fR * kMaskSize), x0, kMaskSize - 1);
    const int y0 = std::clamp(static_cast<int>(fT * kMaskSize), 0, kMaskSize - 1);
    const int y1 = std::clamp(static_cast<int>(fB * kMaskSize), y0, kMaskSize - 1);

    for (int my = y0; my <= y1; ++my) {
        const float v = (static_cast<float>(my) + 0.5f) / static_cast<float>(kMaskSize);
        const float rawRelY = (v - fT) / fH;
        const int sy = std::clamp(static_cast<int>(v * srcHeight), 0, srcHeight - 1);
        const int rowOff = my * kMaskSize;
        for (int mx = x0; mx <= x1; ++mx) {
            const float u = (static_cast<float>(mx) + 0.5f) / static_cast<float>(kMaskSize);
            const float rawRelX = (u - fL) / fW;
            const float cx = rawRelX - 0.5f;
            const float cy = rawRelY - 0.5f;
            const float relX = 0.5f + cx * cosR - cy * sinR;
            const float relY = 0.5f + cx * sinR + cy * cosR;
            const float ex = (relX - 0.5f) / 0.48f;
            const float ey = (relY - 0.50f) / 0.48f;
            if (ex * ex + ey * ey > 1.0f) continue;

            const bool inEyesBrow = (relY >= 0.24f && relY <= 0.46f && relX >= 0.16f && relX <= 0.84f);
            const bool inMouth = (relY >= 0.66f && relY <= 0.84f && relX >= 0.30f && relX <= 0.70f);

            const int sx = std::clamp(static_cast<int>(u * srcWidth), 0, srcWidth - 1);
            const uint8_t* p = rgbaTopDown + (sy * srcWidth + sx) * 4;
            float y, cb, cr;
            rgbToYCbCr(p[0], p[1], p[2], y, cb, cr);

            const float chromaDist = std::fabs(cb - refCb) + std::fabs(cr - refCr);
            const float lumaDrop = refY - y;
            const float chromaThr = (inEyesBrow || inMouth) ? 42.0f : 24.0f;
            const bool nonSkinChroma = (cb > 134.0f || cr < 126.0f) && chromaDist > chromaThr;
            const bool darkObject = !inEyesBrow && !inMouth && y < 44.0f && lumaDrop > 68.0f && (cr < 142.0f || cb > 116.0f);

            float foreignScore = 0.0f;
            if (nonSkinChroma) {
                foreignScore = detail::clampf((chromaDist - chromaThr) / 20.0f, 0.0f, 1.0f);
            } else if (darkObject) {
                foreignScore = detail::clampf((lumaDrop - 68.0f) / 30.0f, 0.0f, 1.0f);
            }
            if (foreignScore > 0.25f) {
                rawOcc_[rowOff + mx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(foreignScore * 255.0f)), 0, 255));
            }
        }
    }

    // 3. 3x3 morphological erosion to reject isolated 1px freckles/noise
    openedOcc_.fill(0);
    const int ey0 = std::max(1, y0);
    const int ey1 = std::min(kMaskSize - 1, y1 + 1);
    const int ex0 = std::max(1, x0);
    const int ex1 = std::min(kMaskSize - 1, x1 + 1);
    for (int my = ey0; my < ey1; ++my) {
        const int rowOff = my * kMaskSize;
        for (int mx = ex0; mx < ex1; ++mx) {
            const uint8_t c = rawOcc_[rowOff + mx];
            if (c == 0) continue;
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int nRow = (my + dy) * kMaskSize;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (rawOcc_[nRow + mx + dx] >= 96) ++neighbors;
                }
            }
            if (neighbors >= 5) {
                openedOcc_[rowOff + mx] = c;
            }
        }
    }

    // 4. 3x3 box feather + invert to visibility + NEON temporal EMA
    const float alpha = hasHistory_ ? kTemporalAlpha : 1.0f;
    alignas(16) float targetVis[kMaskPixels];
    for (int my = 0; my < kMaskSize; ++my) {
        const int rowOff = my * kMaskSize;
        for (int mx = 0; mx < kMaskSize; ++mx) {
            int sum = 0;
            int cnt = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = my + dy;
                if (ny < 0 || ny >= kMaskSize) continue;
                const int nRow = ny * kMaskSize;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = mx + dx;
                    if (nx < 0 || nx >= kMaskSize) continue;
                    sum += openedOcc_[nRow + nx];
                    ++cnt;
                }
            }
            const float occ = static_cast<float>(sum) / static_cast<float>(cnt);
            targetVis[rowOff + mx] = detail::clampf(255.0f - occ, 0.0f, 255.0f);
        }
    }

#if LB_HAS_NEON
    const float32x4_t vOneMinusA = vdupq_n_f32(1.0f - alpha);
    const float32x4_t vAlpha = vdupq_n_f32(alpha);
    const float32x4_t vHalf = vdupq_n_f32(0.5f);
    for (int i = 0; i < kMaskPixels; i += 4) {
        float32x4_t prev = vld1q_f32(&emaMask_[i]);
        float32x4_t cur  = vld1q_f32(&targetVis[i]);
        float32x4_t sm   = vmlaq_f32(vmulq_f32(prev, vOneMinusA), cur, vAlpha);
        vst1q_f32(&emaMask_[i], sm);
        uint32x4_t u32 = vcvtq_u32_f32(vaddq_f32(sm, vHalf));
        uint16x4_t u16 = vqmovn_u32(u32);
        uint8x8_t u8   = vqmovn_u16(vcombine_u16(u16, u16));
        outMaskR8[i]     = vget_lane_u8(u8, 0);
        outMaskR8[i + 1] = vget_lane_u8(u8, 1);
        outMaskR8[i + 2] = vget_lane_u8(u8, 2);
        outMaskR8[i + 3] = vget_lane_u8(u8, 3);
    }
#else
    for (int i = 0; i < kMaskPixels; ++i) {
        const float sm = emaMask_[i] * (1.0f - alpha) + targetVis[i] * alpha;
        emaMask_[i] = sm;
        outMaskR8[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(sm)), 0, 255));
    }
#endif
    hasHistory_ = true;
}

// ============================================================================
// FaceRoiResidualSolver
// ============================================================================

bool FaceRoiResidualSolver::compute(
    const uint8_t* rgbaTopDown,
    int srcWidth,
    int srcHeight,
    const CropRect& faceBounds,
    float retouchStrength,
    uint8_t* outRgba64x64) {
    if (!rgbaTopDown || !outRgba64x64 || srcWidth <= 0 || srcHeight <= 0) return false;
    if (retouchStrength <= 0.01f || faceBounds.width < 0.04f || faceBounds.height < 0.04f) return false;

    const float roiL = detail::clampf(faceBounds.left, 0.0f, 0.96f);
    const float roiT = detail::clampf(faceBounds.top, 0.0f, 0.96f);
    const float roiW = detail::clampf(faceBounds.width, 0.04f, 1.0f - roiL);
    const float roiH = detail::clampf(faceBounds.height, 0.04f, 1.0f - roiT);

    // 1. Bilinear sample face ROI into 64x64 planar RGB [0..1]
    for (int y = 0; y < kRoiSize; ++y) {
        const float v = detail::clampf(roiT + ((static_cast<float>(y) + 0.5f) / kRoiSize) * roiH, 0.0f, 1.0f);
        const float syF = v * static_cast<float>(srcHeight - 1);
        const int sy0 = std::clamp(static_cast<int>(syF), 0, srcHeight - 1);
        const int sy1 = std::min(sy0 + 1, srcHeight - 1);
        const float wy = syF - static_cast<float>(sy0);
        const int rowOff = y * kRoiSize;

        for (int x = 0; x < kRoiSize; ++x) {
            const float u = detail::clampf(roiL + ((static_cast<float>(x) + 0.5f) / kRoiSize) * roiW, 0.0f, 1.0f);
            const float sxF = u * static_cast<float>(srcWidth - 1);
            const int sx0 = std::clamp(static_cast<int>(sxF), 0, srcWidth - 1);
            const int sx1 = std::min(sx0 + 1, srcWidth - 1);
            const float wx = sxF - static_cast<float>(sx0);

            const uint8_t* p00 = rgbaTopDown + (sy0 * srcWidth + sx0) * 4;
            const uint8_t* p10 = rgbaTopDown + (sy0 * srcWidth + sx1) * 4;
            const uint8_t* p01 = rgbaTopDown + (sy1 * srcWidth + sx0) * 4;
            const uint8_t* p11 = rgbaTopDown + (sy1 * srcWidth + sx1) * 4;

            const float w00 = (1.0f - wx) * (1.0f - wy);
            const float w10 = wx * (1.0f - wy);
            const float w01 = (1.0f - wx) * wy;
            const float w11 = wx * wy;

            const int idx = rowOff + x;
            patchR_[idx] = (p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11) * kInv255;
            patchG_[idx] = (p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11) * kInv255;
            patchB_[idx] = (p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11) * kInv255;
        }
    }

    // 2. Edge-preserving 8-tap ring bilateral skin residual extraction
    const float str = detail::clampf(retouchStrength, 0.0f, 1.0f);
    const float inv2Sigma2 = 0.5f / (0.095f * 0.095f);

    for (int y = 0; y < kRoiSize; ++y) {
        const float relY = (static_cast<float>(y) + 0.5f) / static_cast<float>(kRoiSize);
        const int rowOff = y * kRoiSize;
        for (int x = 0; x < kRoiSize; ++x) {
            const float relX = (static_cast<float>(x) + 0.5f) / static_cast<float>(kRoiSize);
            const int idx = rowOff + x;
            const float cR = patchR_[idx];
            const float cG = patchG_[idx];
            const float cB = patchB_[idx];
            const float cY = 0.299f * cR + 0.587f * cG + 0.114f * cB;

            const float ex = (relX - 0.5f) / 0.46f;
            const float ey = (relY - 0.5f) / 0.46f;
            const float ovalDist = ex * ex + ey * ey;
            const bool inEyes = (relY >= 0.27f && relY <= 0.44f && relX >= 0.18f && relX <= 0.82f);
            const bool inMouth = (relY >= 0.67f && relY <= 0.82f && relX >= 0.30f && relX <= 0.70f);

            uint8_t* dst = outRgba64x64 + idx * 4;
            if (ovalDist >= 1.0f || inEyes || inMouth || cY < 0.12f) {
                dst[0] = 128;
                dst[1] = 128;
                dst[2] = 128;
                dst[3] = 0;
                continue;
            }

            float wSum = 1.0f;
            float avgR = cR, avgG = cG, avgB = cB;
            for (int k = 0; k < 16; k += 2) {
                const int nx = std::clamp(x + kRingOffsets[k], 0, kRoiSize - 1);
                const int ny = std::clamp(y + kRingOffsets[k + 1], 0, kRoiSize - 1);
                const int nIdx = ny * kRoiSize + nx;
                const float nR = patchR_[nIdx];
                const float nG = patchG_[nIdx];
                const float nB = patchB_[nIdx];
                const float dR = nR - cR;
                const float dG = nG - cG;
                const float dB = nB - cB;
                const float dist2 = dR * dR + dG * dG + dB * dB;
                const float w = std::exp(-dist2 * inv2Sigma2);
                wSum += w;
                avgR += nR * w;
                avgG += nG * w;
                avgB += nB * w;
            }
            avgR /= wSum;
            avgG /= wSum;
            avgB /= wSum;
            const float avgY = 0.299f * avgR + 0.587f * avgG + 0.114f * avgB;

            const float dY   = detail::clampf(avgY - cY, -0.035f, 0.065f) * (0.48f * str);
            const float dCrR = detail::clampf((avgR - avgY) - (cR - cY), -0.07f, 0.07f) * (0.72f * str);
            const float dCrG = detail::clampf((avgG - avgY) - (cG - cY), -0.06f, 0.06f) * (0.65f * str);
            const float dCrB = detail::clampf((avgB - avgY) - (cB - cY), -0.07f, 0.07f) * (0.72f * str);

            const float outDr = detail::clampf(dY + dCrR, -0.22f, 0.22f);
            const float outDg = detail::clampf(dY + dCrG, -0.22f, 0.22f);
            const float outDb = detail::clampf(dY + dCrB, -0.22f, 0.22f);

            const float skinLike = detail::clampf((cR - cB) - 0.02f, 0.0f, 0.14f) / 0.14f;
            const float ovalFade = detail::clampf((1.0f - ovalDist) / 0.28f, 0.0f, 1.0f);
            const int conf = std::clamp(static_cast<int>(std::lround(skinLike * ovalFade * 255.0f)), 0, 255);

            dst[0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + outDr * 2.0f) * 255.0f)), 0, 255));
            dst[1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + outDg * 2.0f) * 255.0f)), 0, 255));
            dst[2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround((0.5f + outDb * 2.0f) * 255.0f)), 0, 255));
            dst[3] = static_cast<uint8_t>(conf);
        }
    }
    return true;
}

}  // namespace livebeauty::math
