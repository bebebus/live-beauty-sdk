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
