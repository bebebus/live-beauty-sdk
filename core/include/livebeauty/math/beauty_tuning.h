#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include "livebeauty/types.h"

namespace livebeauty::math {

struct BeautyQualityTuning {
    static constexpr float kEnergyScale = 2.5f;
    static constexpr float kDownTapScale = 0.6666667f;
    static constexpr float kSigma0Lo = 8.0f;
    static constexpr float kSigma0Hi = 16.0f;
    static constexpr float kCoreBase = 0.005f;
    static constexpr float kCoreDark = 0.008f;
    static constexpr float kCorePressure = 0.004f;

    static float macroEvenness(float smoothing);
    static float guidedEps(float smoothing);
    static float coreThreshold(float lightLuma, float pressure);
    static float ambientSceneLift(float lightLuma);
    static float indoorTemporalDenoise(float lightLuma);
    static float twoStageSmoothMix(float smoothing);
    static float twoStageDetailKeep(float smoothing);
    static float mouthOpenGate(float openRatio);
};

class GuidedSpread {
public:
    static constexpr float kDefault = 1.25f;
    static constexpr float kMin = 0.9f;
    static constexpr float kMax = 1.6f;

    float value() const { return value_; }
    void reset();
    float update(float faceWidthPx);

private:
    static constexpr float kFaceFraction = 0.02f;
    static constexpr float kAlpha = 0.15f;
    static constexpr float kStep = 0.05f;
    static constexpr float kPxPerSpread = 6.0f;
    static constexpr float kSlack = 0.001f;

    float value_ = kDefault;
    float smoothed_ = kDefault;
};

struct BitrateEffectsTuning {
    static float pressure(float ratio);
    static float sharpness(float base, float pressure);
    static float texture(float base, float pressure);
    static float denoise(float pressure);
};

class EffectsLoadGovernor {
public:
    static constexpr int64_t kWindowMs = 1000;
    static constexpr int kMinFrames = 10;
    static constexpr int kMaxLevel = 3;
    static constexpr int64_t kOverP95Ns = 28000000LL;
    static constexpr int64_t kHysteresisNs = 4000000LL;
    static constexpr double kOverBusy = 0.80;
    static constexpr double kCalmBusy = 0.60;
    static constexpr double kOverDrop = 0.10;
    static constexpr double kOverFps = 27.0;
    static constexpr double kCalmDrop = 0.03;
    static constexpr int kUpStrikes = 3;
    static constexpr int kSettleWindows = 2;
    static constexpr int kRecoverStart = 10;
    static constexpr int kRecoverMax = 240;
    static constexpr int64_t kProbeFailMs = 60000;
    static constexpr int64_t kBackoffResetMs = 600000;
    static constexpr int64_t kSaveCapNs = 10000000LL;
    static constexpr int64_t kIdleResetMs = 30000;

    int level() const { return level_; }
    void reset();
    std::optional<int> onFrame(int64_t stampNs, int64_t workNs, bool processing, int64_t nowMs);

private:
    static constexpr int kWorkSamples = 64;
    struct DropStats { double ratio; double fps; };

    void restartWindow();
    std::optional<int> evaluateWindow(int64_t nowMs, int64_t p95, double busy, DropStats drop);
    DropStats computeDropStats() const;
    static int64_t percentile(const int64_t* values, int count, double q);

    int level_ = 0;
    std::array<int64_t, kWorkSamples> works_{};
    int workCount_ = 0;
    int64_t workSum_ = 0;
    std::array<int64_t, kWorkSamples> deltas_{};
    int deltaCount_ = 0;
    int64_t lastStamp_ = 0;
    int64_t windowStart_ = kTimeInvalid;
    int overStreak_ = 0;
    int64_t overMinP95_ = std::numeric_limits<int64_t>::max();
    int calmStreak_ = 0;
    int settle_ = 0;
    int recover_ = kRecoverStart;
    int64_t lastUpAt_ = kTimeInvalid;
    int64_t lastDownAt_ = kTimeInvalid;
    int64_t idleSince_ = kTimeInvalid;
    std::array<int64_t, 4> saved_ = {0, kSaveCapNs, kSaveCapNs, kSaveCapNs};
    int pendingLevel_ = 0;
    int64_t pendingPre_ = 0;
    std::array<int64_t, 3> post_{};
    int postCount_ = 0;
};

class LoadFades {
public:
    static constexpr float kFadeMs = 400.0f;
    static constexpr int64_t kMaxStepMs = 100;

    float body() const { return body_; }
    float face() const { return face_; }
    float smooth() const { return smooth_; }

    void reset();
    void update(int level, int64_t nowMs);

private:
    float body_ = 1.0f;
    float face_ = 1.0f;
    float smooth_ = 1.0f;
    int64_t last_ = kTimeInvalid;
};

}  // namespace livebeauty::math
