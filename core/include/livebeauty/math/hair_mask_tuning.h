#pragma once

#include <array>
#include <cstdint>

namespace livebeauty::math {

/**
 * Hair semantic mask scheduling, freshness, temporal presence and rigid head-motion compensation.
 * Formula-for-formula aligned with Android HairMaskTuning.kt.
 */
struct HairMaskTuning {
    static constexpr int64_t kIntervalMs = 100;
    static constexpr int64_t kBackgroundIntervalMs = 200;
    static constexpr float kIntervalP50Scale = 2.0f;
    static constexpr int kP50Window = 10;
    static constexpr int kMinSamples = 8;
    static constexpr float kMaxP50Ms = 125.0f;

    static constexpr int64_t kFreshMs = 650;
    static constexpr float kFreshFullMs = 420.0f;
    static constexpr float kPresenceFadeInMs = 120.0f;
    static constexpr float kPresenceHoldMs = 300.0f;
    static constexpr float kPresenceFadeOutMs = 400.0f;

    static constexpr float kAffineMaxShift = 0.18f;
    static constexpr float kAffineMinScale = 0.80f;
    static constexpr float kAffineMaxScale = 1.25f;
    static constexpr int kAffineSize = 8;
    static constexpr int kPoseSize = 6;  // (cx, cy, rx, ry, axisX, axisY)

    static constexpr int64_t kFaceHoldMs = 200;
    static constexpr float kFaceFadeOutMs = 250.0f;
    static constexpr int64_t kFaceZeroMs = 450;
    static constexpr float kFaceStallFadeMs = 100.0f;
    static constexpr float kFaceFadeInMs = 300.0f;

    static void identity(float out8[kAffineSize]);

    /**
     * Compute 2x3 metric-space rigid motion compensation affine (display UV -> mask UV):
     *   u' = out[0]*u + out[1]*v + out[2]
     *   v' = out[4]*u + out[5]*v + out[6]
     * Returns false and writes identity when either pose is invalid or motion exceeds safety bounds.
     */
    static bool affine(const float* poseMask6, const float* poseDisplay6, float aspect, float out8[kAffineSize]);

    static float fresh(float lagMs, float fullMs = kFreshFullMs, float zeroMs = static_cast<float>(kFreshMs));
    static float shown(float presence, float freshness, float bodyFade, float neckWeight);
    static float faceKeep(float previous, int64_t latestFaceMs, int64_t missSinceMs,
                          int64_t nowMs, int64_t dtMs, bool faceInFlight = false);
    static int64_t adaptiveInterval(float p50Ms, bool backgroundActive);
};

/** Per-session hair mask temporal gate (presence + freshness + faceKeep). */
class HairMaskGate {
public:
    void reset();

    /**
     * Advance temporal gate for current frame.
     * @param hasValidMask true if hairMaskTex is bound and valid
     * @param maskStampMs timestamp (ms) of the bound hair mask
     * @param displayMs current display frame timestamp (ms)
     * @param bodyFade loadFadeBody [0..1]
     * @param latestFaceMs latest valid face sample timestamp (kTimeInvalid if none)
     * @param missSinceMs timestamp of latest face miss (kTimeInvalid if none)
     * @param nowMs current monotonic clock (ms)
     * @return level [0..1] before neck-lift attenuation
     */
    float update(bool hasValidMask, int64_t maskStampMs, int64_t displayMs,
                 float bodyFade, int64_t latestFaceMs, int64_t missSinceMs, int64_t nowMs);

    float level() const { return level_; }
    float faceKeep() const { return faceKeep_; }
    float shown(float neckWeight) const { return HairMaskTuning::shown(level_, 1.0f, 1.0f, neckWeight); }

private:
    float presence_ = 0.0f;
    int64_t lastPresentMs_ = -1;
    int64_t lastUpdateMs_ = -1;
    float faceKeep_ = 0.0f;
    float level_ = 0.0f;
};

}  // namespace livebeauty::math
