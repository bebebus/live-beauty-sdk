#pragma once

#include <array>
#include <cstdint>
#include "livebeauty/types.h"

namespace livebeauty::math {

struct PoseLandmark {
    float x = 0.0f;
    float y = 0.0f;
    float visibility = 0.0f;
};

struct BodyGeometry {
    static constexpr int kStateSize = 7;
    static constexpr float kVisibilityAcquire = 0.6f;
    static constexpr float kVisibilityKeep = 0.4f;

    float hipX = 0.0f;
    float hipY = 0.0f;
    float axisX = 0.0f;
    float axisY = -1.0f;
    float torso = 0.0f;
    float half = 0.0f;
    float legs = 0.0f;
    float visibility = 1.0f;

    void toState(float outState[kStateSize]) const {
        outState[0] = hipX;
        outState[1] = hipY;
        outState[2] = axisX;
        outState[3] = axisY;
        outState[4] = torso;
        outState[5] = half;
        outState[6] = legs;
    }

    static bool fromPose(const PoseLandmark* points, int count, float aspect, BodyGeometry& out);
};

class BodyGeometrySmoother {
public:
    static constexpr float kHoldMs = 67.0f;
    static constexpr float kFadeOutMs = 100.0f;

    float presence() const { return presence_; }
    bool hasState() const { return hasState_; }
    const float* state() const { return hasState_ ? state_.data() : nullptr; }

    void reset();
    const float* update(const float* targetState, int64_t sampleTimestampMs, int64_t nowMs);

private:
    static constexpr float kAlpha = 0.45f;
    static constexpr float kFadeInMs = 200.0f;

    float presence_ = 0.0f;
    bool hasState_ = false;
    std::array<float, BodyGeometry::kStateSize> state_{};
    int64_t ts_ = kTimeInvalid;
    float missing_ = 0.0f;
    int64_t last_ = kTimeInvalid;
};

struct NeckLift {
    static void applyFaceState(const float* face41, const float* body7, float neck, float aspect, float* out41);
    static void applyLandmarks(
        const float* landmarks, int coordCount,
        float faceCenterX, float faceCenterY,
        const float* body7, float neck, float aspect,
        float* outLandmarks);
};

}  // namespace livebeauty::math
