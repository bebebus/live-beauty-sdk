#include "livebeauty/math/hair_mask_tuning.h"

#include <algorithm>
#include <cmath>
#include "livebeauty/types.h"

namespace livebeauty::math {

namespace {

inline float smoothstep(float e0, float e1, float x) {
    if (e1 <= e0) return (x >= e1) ? 1.0f : 0.0f;
    const float t = detail::clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

}  // namespace

void HairMaskTuning::identity(float out8[kAffineSize]) {
    for (int i = 0; i < kAffineSize; ++i) out8[i] = 0.0f;
    out8[0] = 1.0f;
    out8[5] = 1.0f;
}

bool HairMaskTuning::affine(
    const float* poseMask6, const float* poseDisplay6, float aspect, float out8[kAffineSize]) {
    identity(out8);
    if (!poseMask6 || !poseDisplay6 || !(aspect > 0.0f) || !std::isfinite(aspect)) {
        return false;
    }
    const float mx = poseMask6[0];
    const float my = poseMask6[1];
    const float mr = poseMask6[2];
    const float dx = poseDisplay6[0];
    const float dy = poseDisplay6[1];
    const float dr = poseDisplay6[2];
    const float nm = std::hypot(poseMask6[4], poseMask6[5]);
    const float nd = std::hypot(poseDisplay6[4], poseDisplay6[5]);
    if (!(mr > 0.0f && dr > 0.0f && nm > 1e-6f && nd > 1e-6f)) return false;

    const float shift = std::hypot(mx - dx, (my - dy) / aspect);
    const float scale = mr / dr;
    if (!(shift <= kAffineMaxShift) || !(scale >= kAffineMinScale && scale <= kAffineMaxScale)) {
        return false;
    }
    const float ux = poseDisplay6[4] / nd;
    const float uy = poseDisplay6[5] / nd;
    const float wx = poseMask6[4] / nm;
    const float wy = poseMask6[5] / nm;
    const float c = scale * (ux * wx + uy * wy);
    const float s = scale * (ux * wy - uy * wx);

    out8[0] = c;
    out8[1] = -s / aspect;
    out8[2] = mx - c * dx + s * dy / aspect;
    out8[3] = 0.0f;
    out8[4] = s * aspect;
    out8[5] = c;
    out8[6] = my - s * aspect * dx - c * dy;
    out8[7] = 0.0f;
    return true;
}

float HairMaskTuning::fresh(float lagMs, float fullMs, float zeroMs) {
    if (!std::isfinite(lagMs)) return 0.0f;
    return 1.0f - smoothstep(fullMs, zeroMs, std::fabs(lagMs));
}

float HairMaskTuning::shown(float presence, float freshness, float bodyFade, float neckWeight) {
    if (!(neckWeight < 1.0f)) return 0.0f;
    const float nw = std::max(0.0f, neckWeight);
    return detail::clampf(presence * freshness * bodyFade * (1.0f - nw), 0.0f, 1.0f);
}

float HairMaskTuning::faceKeep(
    float previous, int64_t latestFaceMs, int64_t missSinceMs,
    int64_t nowMs, int64_t dtMs, bool faceInFlight) {
    if (latestFaceMs == kTimeInvalid) return 0.0f;
    const float prev = std::isfinite(previous) ? detail::clampf(previous, 0.0f, 1.0f) : 0.0f;
    const int64_t dt = std::clamp<int64_t>(dtMs, 0, 100);
    const int64_t age = std::max<int64_t>(0, nowMs - latestFaceMs);
    const float cap = faceInFlight
        ? 1.0f
        : detail::clampf(static_cast<float>(kFaceZeroMs - age) / kFaceStallFadeMs, 0.0f, 1.0f);
    const bool lost = (missSinceMs != kTimeInvalid && missSinceMs > latestFaceMs);
    if (!lost || age <= kFaceHoldMs) {
        return std::min(cap, prev + static_cast<float>(dt) / kFaceFadeInMs);
    }
    const int64_t from = std::max<int64_t>(age - dt, kFaceHoldMs);
    const float step = static_cast<float>(age - from);
    const float left = static_cast<float>(kFaceZeroMs - from);
    const float next = (step >= left)
        ? 0.0f
        : prev - step * std::max(1.0f / kFaceFadeOutMs, prev / left);
    return std::min(cap, std::max(0.0f, next));
}

int64_t HairMaskTuning::adaptiveInterval(float p50Ms, bool backgroundActive) {
    const int64_t floorMs = backgroundActive ? kBackgroundIntervalMs : kIntervalMs;
    if (!std::isfinite(p50Ms) || p50Ms <= 0.0f) return floorMs;
    const int64_t scaled = static_cast<int64_t>(std::ceil(p50Ms * kIntervalP50Scale));
    return std::max(floorMs, scaled);
}

void HairMaskGate::reset() {
    presence_ = 0.0f;
    lastPresentMs_ = kTimeInvalid;
    lastUpdateMs_ = kTimeInvalid;
    faceKeep_ = 0.0f;
    level_ = 0.0f;
}

float HairMaskGate::update(
    bool hasValidMask, int64_t maskStampMs, int64_t displayMs,
    float bodyFade, int64_t latestFaceMs, int64_t missSinceMs, int64_t nowMs) {
    const int64_t dt = (lastUpdateMs_ == kTimeInvalid)
        ? 0
        : std::clamp<int64_t>(nowMs - lastUpdateMs_, 0, 100);
    lastUpdateMs_ = nowMs;

    if (hasValidMask) {
        lastPresentMs_ = nowMs;
        presence_ = std::min(1.0f, presence_ + static_cast<float>(dt) / HairMaskTuning::kPresenceFadeInMs);
    } else if (lastPresentMs_ != kTimeInvalid &&
               (nowMs - lastPresentMs_) > static_cast<int64_t>(HairMaskTuning::kPresenceHoldMs)) {
        presence_ = std::max(0.0f, presence_ - static_cast<float>(dt) / HairMaskTuning::kPresenceFadeOutMs);
    } else if (lastPresentMs_ == kTimeInvalid) {
        presence_ = 0.0f;
    }

    faceKeep_ = HairMaskTuning::faceKeep(faceKeep_, latestFaceMs, missSinceMs, nowMs, dt, false);
    const float freshness = (hasValidMask && maskStampMs != kTimeInvalid)
        ? HairMaskTuning::fresh(static_cast<float>(maskStampMs - displayMs))
        : 0.0f;
    level_ = HairMaskTuning::shown(presence_, freshness, bodyFade * faceKeep_, 0.0f);
    return level_;
}

}  // namespace livebeauty::math
