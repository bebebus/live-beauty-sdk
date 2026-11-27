#include "livebeauty/math/beauty_tuning.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace livebeauty::math {

float BeautyQualityTuning::macroEvenness(float smoothing) {
    if (std::isnan(smoothing) || smoothing <= 0.0f) return 0.0f;
    return 0.42f * std::sqrt(detail::clampf(smoothing, 0.0f, 1.0f));
}

float BeautyQualityTuning::guidedEps(float smoothing) {
    const float t = std::isnan(smoothing) ? 0.0f : detail::clampf(smoothing, 0.0f, 1.0f);
    const float sigma = kEnergyScale * std::sqrt(3.0f) * (kSigma0Lo + (kSigma0Hi - kSigma0Lo) * t) / 255.0f;
    return sigma * sigma;
}

float BeautyQualityTuning::coreThreshold(float lightLuma, float pressure) {
    const float dark = std::isnan(lightLuma) ? 0.0f : detail::clampf((0.30f - lightLuma) / (0.30f - 0.10f), 0.0f, 1.0f);
    const float load = std::isnan(pressure) ? 0.0f : detail::clampf(pressure, 0.0f, 1.0f);
    return kCoreBase + kCoreDark * dark + kCorePressure * load;
}

float BeautyQualityTuning::ambientSceneLift(float lightLuma) {
    if (!std::isfinite(lightLuma)) return 0.0f;
    // v3.9.4 (A1/tone-1): Cap low-light ambient compensation strictly to genuinely dark scenes (L < 0.30)
    // with <= 0.18 ceiling so AE-normalized indoor rooms (L ~ 0.45..0.55) never receive unrequested lift.
    return 0.18f * (1.0f - detail::smoothstep(0.12f, 0.30f, lightLuma));
}

float BeautyQualityTuning::indoorTemporalDenoise(float lightLuma) {
    if (!std::isfinite(lightLuma)) return 0.0f;
    // v3.9.4 (A5/tone-2/gpuperf-4): Zero out sub-LSB indoor range (>= 0.12) where 8-bit temporal
    // denoise was a bit-exact no-op; only engage >= 0.15 in extreme darkness (<= 0.08 full 0.18).
    const float w = 0.18f * (1.0f - detail::smoothstep(0.08f, 0.14f, lightLuma));
    return (w >= 0.15f) ? w : 0.0f;
}

float BeautyQualityTuning::twoStageSmoothMix(float smoothing) {
    if (!std::isfinite(smoothing) || smoothing <= 0.0f) return 0.0f;
    const float s = detail::clampf(smoothing, 0.0f, 1.0f);
    // v3.9.4 (A4/skin-2): Recalibrated smoothing curve leaving headroom above default 0.55
    // (0.55 -> k ~ 0.58 instead of 0.87) to prevent waxy/plastic skin while keeping 0.96 ceiling at 1.0.
    const float stage1 = detail::clampf(s / 0.85f, 0.0f, 1.0f);
    return detail::clampf(stage1 * (0.80f + 0.20f * stage1) * 0.96f, 0.0f, 0.96f);
}

float BeautyQualityTuning::twoStageDetailKeep(float smoothing) {
    if (!std::isfinite(smoothing) || smoothing <= 0.0f) return 1.0f;
    const float s = detail::clampf(smoothing, 0.0f, 1.0f);
    // Keep full natural skin texture across 0..0.60, taper smoothly to 0.45 toward 1.0 for porcelain smoothness.
    return 1.0f - 0.55f * detail::smoothstep(0.60f, 1.0f, s);
}

float BeautyQualityTuning::mouthOpenGate(float openRatio) {
    if (!std::isfinite(openRatio)) return 0.0f;
    return detail::smoothstep(0.055f, 0.135f, openRatio);
}

float BeautyQualityTuning::lowerFaceMakeupKeep(float lowerFaceOccRatio) {
    if (!std::isfinite(lowerFaceOccRatio)) return 1.0f;
    // Smoothly mute lower-face makeup (lipstick/blush/contour/highlight/teeth) when a hand, cup, or mic
    // covers >= 45%..78% of the lower/mid face, while leaving eye/brow makeup unaffected.
    return 1.0f - detail::smoothstep(0.45f, 0.78f, lowerFaceOccRatio);
}

void BeautyQualityTuning::cat02WhiteBalanceMatrix(float colorTemp, float colorTint, float outRowMajor3x3[9]) {
    if (!outRowMajor3x3) return;
    const float t = std::isfinite(colorTemp) ? detail::clampf(colorTemp, -1.0f, 1.0f) : 0.0f;
    const float g = std::isfinite(colorTint) ? detail::clampf(colorTint, -1.0f, 1.0f) : 0.0f;
    if (std::fabs(t) <= 1e-5f && std::fabs(g) <= 1e-5f) {
        outRowMajor3x3[0] = 1.0f; outRowMajor3x3[1] = 0.0f; outRowMajor3x3[2] = 0.0f;
        outRowMajor3x3[3] = 0.0f; outRowMajor3x3[4] = 1.0f; outRowMajor3x3[5] = 0.0f;
        outRowMajor3x3[6] = 0.0f; outRowMajor3x3[7] = 0.0f; outRowMajor3x3[8] = 1.0f;
        return;
    }

    // Textbook CIE 1931 sRGB (D65) <-> XYZ and CIECAM02 CAT02 von Kries chromatic adaptation
    static constexpr double kSrgbToXyz[9] = {
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041,
    };
    static constexpr double kXyzToSrgb[9] = {
         3.2404542, -1.5371385, -0.4985314,
        -0.9692660,  1.8760108,  0.0415560,
         0.0556434, -0.2040259,  1.0572252,
    };
    static constexpr double kCat02[9] = {
         0.7328,  0.4296, -0.1624,
        -0.7036,  1.6975,  0.0061,
         0.0030,  0.0136,  0.9834,
    };
    static constexpr double kCat02Inv[9] = {
         1.0961238208, -0.2788690002,  0.1827451794,
         0.4543690419,  0.4735331543,  0.0720978038,
        -0.0096276087, -0.0056980312,  1.0153256399,
    };

    // Shift source illuminant along the CIE daylight/Planckian locus around D65 (0.31271, 0.32902).
    // Positive colorTemp -> cooler assumed source white -> warming adaptation; negative -> cooling adaptation.
    const double x0 = 0.31271;
    const double x = x0 - static_cast<double>(t) * (t < 0.0f ? 0.065 : 0.045);
    const double y = std::clamp(2.87 * x - 3.0 * x * x - 0.2750951 + static_cast<double>(g) * 0.045, 0.20, 0.48);
    const double srcX = x / y;
    const double srcY = 1.0;
    const double srcZ = (1.0 - x - y) / y;
    const double d65[3] = {0.95047, 1.0, 1.08883};
    const double src[3] = {srcX, srcY, srcZ};

    double lmsD65[3]{}, lmsSrc[3]{};
    for (int r = 0; r < 3; ++r) {
        lmsD65[r] = kCat02[r * 3 + 0] * d65[0] + kCat02[r * 3 + 1] * d65[1] + kCat02[r * 3 + 2] * d65[2];
        lmsSrc[r] = kCat02[r * 3 + 0] * src[0] + kCat02[r * 3 + 1] * src[1] + kCat02[r * 3 + 2] * src[2];
    }
    const double scale[3] = {
        lmsD65[0] / std::max(lmsSrc[0], 0.05),
        lmsD65[1] / std::max(lmsSrc[1], 0.05),
        lmsD65[2] / std::max(lmsSrc[2], 0.05),
    };

    auto mul3 = [](const double a[9], const double b[9], double out[9]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                                 a[r * 3 + 1] * b[1 * 3 + c] +
                                 a[r * 3 + 2] * b[2 * 3 + c];
            }
        }
    };

    double scaledCat[9]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            scaledCat[r * 3 + c] = scale[r] * kCat02[r * 3 + c];
        }
    }
    double m1[9]{}, m2[9]{}, m3[9]{};
    mul3(kCat02Inv, scaledCat, m1);
    mul3(m1, kSrgbToXyz, m2);
    mul3(kXyzToSrgb, m2, m3);
    for (int i = 0; i < 9; ++i) {
        outRowMajor3x3[i] = static_cast<float>(m3[i]);
    }
}

void BeautyQualityTuning::estimateAutoWhiteBalance(
    const Vec3& ambientRgb, float strength, float& outTemp, float& outTint) {
    outTemp = 0.0f;
    outTint = 0.0f;
    const float str = std::isfinite(strength) ? detail::clampf(strength, 0.0f, 1.0f) : 0.0f;
    if (str <= 1e-4f) return;

    const float r = std::isfinite(ambientRgb.x) ? detail::clampf(ambientRgb.x, 0.65f, 1.35f) : 1.0f;
    const float g = std::isfinite(ambientRgb.y) ? detail::clampf(ambientRgb.y, 0.65f, 1.35f) : 1.0f;
    const float b = std::isfinite(ambientRgb.z) ? detail::clampf(ambientRgb.z, 0.65f, 1.35f) : 1.0f;

    // Warmth cast: positive under tungsten/yellow light (r > b) -> cool down (negative CAT02 temp);
    // negative under cool blue LED (b > r) -> warm up (positive CAT02 temp).
    const float warmCast = (r - b) / std::max(r + b, 1e-4f);
    // Green/magenta cast: positive under fluorescent green cast (g > 0.5*(r+b)) -> positive CAT02 tint (magenta shift).
    const float greenCast = (g - 0.5f * (r + b)) / std::max(r + g + b, 1e-4f);

    auto softDeadzone = [](float v, float dz) {
        const float av = std::fabs(v);
        if (av <= dz) return 0.0f;
        return (v > 0.0f ? 1.0f : -1.0f) * (av - dz);
    };

    const float dzWarm = softDeadzone(warmCast, 0.025f);
    const float dzGreen = softDeadzone(greenCast, 0.014f);
    outTemp = detail::clampf(-dzWarm * 1.95f * str, -0.42f, 0.42f);
    outTint = detail::clampf(dzGreen * 4.60f * str, -0.26f, 0.26f);
}

void BeautyQualityTuning::resolveStyleAtmosphere(
    const std::string& /*lutUrl*/, float /*lutIntensity*/,
    int /*filterStyleId*/, float /*filterIntensity*/,
    float& outBloom, float& outGrain) {
    // v3.9.4 (D4/tone-5/gpuperf-v1/tone-v2): Do not implicitly attach un-disableable 4-pass Bloom
    // (which duplicated LUT soft_glow for only 2-6 LSB flat lift) or sub-LSB no-op auto grain.
    // Users retain full explicit control via config.bloom and config.filmGrain sliders.
    outBloom = 0.0f;
    outGrain = 0.0f;
}

void GuidedSpread::reset() {
    value_ = kDefault;
    smoothed_ = kDefault;
}

float GuidedSpread::update(float faceWidthPx) {
    if (!std::isfinite(faceWidthPx) || faceWidthPx <= 0.0f) return value_;
    const float target = detail::clampf(faceWidthPx * kFaceFraction / kPxPerSpread, kMin, kMax);
    smoothed_ += (target - smoothed_) * kAlpha;
    const float gap = smoothed_ - value_;
    if (gap >= kStep - kSlack) {
        value_ = std::min(value_ + kStep, kMax);
    } else if (gap <= -(kStep - kSlack)) {
        value_ = std::max(value_ - kStep, kMin);
    }
    return value_;
}

float BitrateEffectsTuning::pressure(float ratio) {
    return std::isfinite(ratio) ? detail::clampf((0.75f - ratio) / 0.5f, 0.0f, 1.0f) : 0.0f;
}

float BitrateEffectsTuning::sharpness(float base, float pressure) {
    return base * (1.0f - 0.6f * detail::clampf(pressure, 0.0f, 1.0f));
}

float BitrateEffectsTuning::texture(float base, float pressure) {
    return base * (1.0f - 0.5f * detail::clampf(pressure, 0.0f, 1.0f));
}

float BitrateEffectsTuning::denoise(float pressure) {
    return 0.16f * detail::clampf(pressure, 0.0f, 1.0f);
}

void EffectsLoadGovernor::reset() {
    level_ = 0;
    overStreak_ = 0;
    overMinP95_ = std::numeric_limits<int64_t>::max();
    calmStreak_ = 0;
    settle_ = 0;
    recover_ = kRecoverStart;
    lastUpAt_ = kTimeInvalid;
    lastDownAt_ = kTimeInvalid;
    idleSince_ = kTimeInvalid;
    saved_ = {0, kSaveCapNs, kSaveCapNs, kSaveCapNs};
    pendingLevel_ = 0;
    postCount_ = 0;
    restartWindow();
}

void EffectsLoadGovernor::restartWindow() {
    workCount_ = 0;
    workSum_ = 0;
    deltaCount_ = 0;
    lastStamp_ = 0;
    windowStart_ = kTimeInvalid;
}

std::optional<int> EffectsLoadGovernor::onFrame(int64_t stampNs, int64_t workNs, bool processing, int64_t nowMs) {
    if (!processing || stampNs == 0) {
        overStreak_ = 0;
        calmStreak_ = 0;
        restartWindow();
        if (idleSince_ == kTimeInvalid) idleSince_ = nowMs;
        if (level_ > 0 && nowMs - idleSince_ >= kIdleResetMs) {
            reset();
            return 0;
        }
        return std::nullopt;
    }
    idleSince_ = kTimeInvalid;
    if (stampNs == lastStamp_) return std::nullopt;
    if (lastStamp_ != 0) {
        const int64_t delta = stampNs - lastStamp_;
        if (delta <= 0 || delta > 500000000LL) {
            restartWindow();
            lastStamp_ = stampNs;
            windowStart_ = nowMs;
            return std::nullopt;
        }
        deltas_[deltaCount_ % kWorkSamples] = delta;
        ++deltaCount_;
    }
    lastStamp_ = stampNs;
    const int64_t clampedWork = std::max<int64_t>(workNs, 0);
    works_[workCount_ % kWorkSamples] = clampedWork;
    ++workCount_;
    workSum_ += clampedWork;
    if (windowStart_ == kTimeInvalid) windowStart_ = nowMs;
    const int64_t span = nowMs - windowStart_;
    if (span < kWindowMs) return std::nullopt;

    const bool enough = (workCount_ >= kMinFrames);
    const int64_t p95 = percentile(works_.data(), std::min(workCount_, kWorkSamples), 0.95);
    const double busy = static_cast<double>(workSum_) / (static_cast<double>(span) * 1e6);
    const DropStats drop = computeDropStats();
    workCount_ = 0;
    workSum_ = 0;
    deltaCount_ = 0;
    windowStart_ = nowMs;
    return enough ? evaluateWindow(nowMs, p95, busy, drop) : std::nullopt;
}

std::optional<int> EffectsLoadGovernor::evaluateWindow(int64_t nowMs, int64_t p95, double busy, DropStats drop) {
    if (settle_ > 0) {
        --settle_;
        return std::nullopt;
    }
    if (pendingLevel_ > 0) {
        post_[postCount_++] = p95;
        if (postCount_ == 3) {
            std::sort(post_.begin(), post_.end());
            saved_[pendingLevel_] = std::clamp<int64_t>(pendingPre_ - post_[1], 0, kSaveCapNs);
            pendingLevel_ = 0;
            postCount_ = 0;
        }
    }
    const bool over = (p95 > kOverP95Ns) || (busy > kOverBusy) || (drop.ratio > kOverDrop && drop.fps < kOverFps);
    if (over) {
        ++overStreak_;
        overMinP95_ = std::min(overMinP95_, p95);
    } else {
        overStreak_ = 0;
        overMinP95_ = std::numeric_limits<int64_t>::max();
    }
    if (lastUpAt_ != kTimeInvalid && nowMs - lastUpAt_ > kBackoffResetMs) recover_ = kRecoverStart;
    if (overStreak_ >= kUpStrikes && level_ < kMaxLevel) {
        if (lastDownAt_ != kTimeInvalid && nowMs - lastDownAt_ < kProbeFailMs) {
            recover_ = std::min(recover_ * 2, kRecoverMax);
        }
        ++level_;
        pendingLevel_ = level_;
        pendingPre_ = overMinP95_;
        postCount_ = 0;
        overStreak_ = 0;
        overMinP95_ = std::numeric_limits<int64_t>::max();
        calmStreak_ = 0;
        settle_ = kSettleWindows;
        lastUpAt_ = nowMs;
        return level_;
    }
    if (level_ > 0) {
        const bool calm = (p95 + saved_[level_] < kOverP95Ns - kHysteresisNs) &&
                          (busy < kCalmBusy) && (drop.ratio < kCalmDrop);
        calmStreak_ = calm ? (calmStreak_ + 1) : 0;
        if (calmStreak_ >= recover_) {
            --level_;
            if (pendingLevel_ > level_) {
                pendingLevel_ = 0;
                postCount_ = 0;
            }
            calmStreak_ = 0;
            overStreak_ = 0;
            overMinP95_ = std::numeric_limits<int64_t>::max();
            settle_ = kSettleWindows;
            lastDownAt_ = nowMs;
            return level_;
        }
    }
    return std::nullopt;
}

EffectsLoadGovernor::DropStats EffectsLoadGovernor::computeDropStats() const {
    const int n = std::min(deltaCount_, kWorkSamples);
    if (n < 8) return {0.0, std::numeric_limits<double>::max()};
    std::vector<int64_t> sorted(deltas_.begin(), deltas_.begin() + n);
    std::sort(sorted.begin(), sorted.end());
    const int64_t period = std::max<int64_t>(sorted[static_cast<size_t>(n * 0.2)], 1);
    int expected = 0;
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        expected += std::max(1, static_cast<int>(std::floor(static_cast<double>(deltas_[i]) / static_cast<double>(period) + 0.25)));
        sum += deltas_[i];
    }
    return {static_cast<double>(expected - n) / static_cast<double>(expected),
            static_cast<double>(n) * 1e9 / static_cast<double>(sum)};
}

int64_t EffectsLoadGovernor::percentile(const int64_t* values, int count, double q) {
    if (count == 0) return 0;
    std::vector<int64_t> sorted(values, values + count);
    std::sort(sorted.begin(), sorted.end());
    const int idx = std::min(static_cast<int>((count - 1) * q + 0.5), count - 1);
    return sorted[idx];
}

void LoadFades::reset() {
    body_ = 1.0f;
    face_ = 1.0f;
    smooth_ = 1.0f;
    last_ = kTimeInvalid;
}

void LoadFades::update(int level, int64_t nowMs) {
    const float dt = (last_ == kTimeInvalid) ? 0.0f : static_cast<float>(std::clamp<int64_t>(nowMs - last_, 0, kMaxStepMs));
    last_ = nowMs;
    const float step = dt / kFadeMs;
    auto toward = [&](float v, float target) {
        return (v < target) ? std::min(v + step, target) : std::max(v - step, target);
    };
    body_ = toward(body_, (level < 1) ? 1.0f : 0.0f);
    face_ = toward(face_, (level < 2) ? 1.0f : 0.0f);
    smooth_ = toward(smooth_, (level < 3) ? 1.0f : 0.0f);
}

}  // namespace livebeauty::math
