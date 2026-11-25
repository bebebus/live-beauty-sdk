#pragma once

#include <array>
#include <cstdint>
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

class FaceTimeline {
public:
    static constexpr int kPose6Size = 6;
    static constexpr int kStateSize = FaceGeometryConstants::kStateSize;
    static constexpr int64_t kMaxGapMs = 220;
    static constexpr float kExtrapolateIntervals = 1.25f;
    static constexpr float kExtrapolateMinMs = 40.0f;
    static constexpr float kExtrapolateMaxMs = 100.0f;
    static constexpr float kExtrapolateDamp = 0.90f;
    static constexpr float kTInMs = 50.0f;
    static constexpr float kMissDebounceMinMs = 33.0f;
    static constexpr float kMissDebounceIntervals = 1.2f;
    static constexpr float kExitRampMs = 40.0f;
    static constexpr float kBusyHoldMinMs = 100.0f;
    static constexpr float kBusyHoldMaxMs = 250.0f;
    static constexpr int64_t kClearAfterMs = 500;
    static constexpr float kDefaultIntervalMs = 1000.0f / 30.0f;

    struct SmoothedMeshView {
        const float* landmarks = nullptr;  // 478 * 2
        const float* meta = nullptr;       // 15
        bool hasIris = false;

        float centerX() const { return meta[FaceMeshFrame::kMetaCenterX]; }
        float centerY() const { return meta[FaceMeshFrame::kMetaCenterY]; }
        float axisX() const { return meta[FaceMeshFrame::kMetaAxisX]; }
        float axisY() const { return meta[FaceMeshFrame::kMetaAxisY]; }
        float faceWidth() const { return meta[FaceMeshFrame::kMetaFaceWidth]; }
        float faceHeight() const { return meta[FaceMeshFrame::kMetaFaceHeight]; }
        float signedYaw() const { return meta[FaceMeshFrame::kMetaSignedYaw]; }
        float leftWeight() const { return meta[FaceMeshFrame::kMetaLeftWeight]; }
        float rightWeight() const { return meta[FaceMeshFrame::kMetaRightWeight]; }
        float yawWeight() const { return meta[FaceMeshFrame::kMetaYawWeight]; }
        float rightEyeOpen() const { return meta[FaceMeshFrame::kMetaRightEyeOpen]; }
        float leftEyeOpen() const { return meta[FaceMeshFrame::kMetaLeftEyeOpen]; }
        float mouthOpen() const { return meta[FaceMeshFrame::kMetaMouthOpen]; }
        float rightIrisGate() const { return meta[FaceMeshFrame::kMetaRightIrisGate]; }
        float leftIrisGate() const { return meta[FaceMeshFrame::kMetaLeftIrisGate]; }
    };

    FaceTimeline();

    float aspect() const { return aspect_; }
    void setAspect(float a) { aspect_ = a; }

    float presence() const { return presence_; }
    float offsetX() const { return offsetX_; }
    float offsetY() const { return offsetY_; }
    float speed() const { return speed_; }
    float steadyEye() const { return steadyEye_; }
    float steadyFace() const { return steadyFace_; }
    float eyeMakeupKeep() const { return eyeMakeupKeep_; }
    float denoiseKeep() const { return denoiseKeep_; }
    int runId() const { return runId_; }
    float sampleIntervalMs() const { return sampleIntervalMs_; }
    int64_t latest() const;
    int64_t missSince() const;
    float horizonMs() const;
    float missDebounceMs() const;

    void reset();
    void add(int64_t sampleTs, const float* targetState41, const FaceMeshFrame* mesh = nullptr);
    void addMiss(int64_t sampleTs);
    void update(int64_t nowMs);

    const float* at(int64_t t);
    bool poseAt(int64_t t, float outPose6[kPose6Size]) const;
    bool meshAt(int64_t t, SmoothedMeshView& outView);
    const float* dilate(const float* geometry41, float* out41) const;

private:
    static constexpr int kHistory = 16;
    static constexpr int kEvents = 16;

    void clearHistory();
    float presenceAt(int64_t t);
    float busyHoldMs() const;
    int64_t missRunStart(int fromNewest) const;
    void startRun(int64_t sampleTs, float level, bool hard);
    void redetect(int64_t sampleTs);
    void pushEvent(int64_t ts, bool face);
    void dropSamples();
    void updateVelocity(int64_t sampleTs, int slot);
    void settle();
    static void normalizeAxis(float* state41);
    static void normalizeMetaAxis(float* meta15);
    static float approach(float current, float target, float dt, float outMs = 80.0f, float inMs = 300.0f);

    float aspect_ = 1.0f;
    std::array<int64_t, kHistory> times_{};
    std::array<std::array<float, kStateSize>, kHistory> states_{};
    std::array<std::array<float, FaceMeshFrame::kCoordCount>, kHistory> meshCoords_{};
    std::array<std::array<float, FaceMeshFrame::kMetaSize>, kHistory> meshMeta_{};
    std::array<std::array<float, FaceMeshFilter::kPoseSize>, kHistory> meshPose_{};
    std::array<bool, kHistory> meshPresent_{};
    std::array<bool, kHistory> meshHasIris_{};
    std::array<float, FaceMeshFrame::kCoordCount> meshOutCoords_{};
    std::array<float, FaceMeshFrame::kMetaSize> meshOutMeta_{};
    FaceMeshFilter meshFilter_;

    int head_ = 0;
    int count_ = 0;
    std::array<float, kHistory> rawX_{};
    std::array<float, kHistory> rawY_{};
    float edxX_ = 0.0f;
    float edxY_ = 0.0f;
    float holdX_ = 0.0f;
    float holdY_ = 0.0f;
    std::array<float, kStateSize> output_{};
    float velocityX_ = 0.0f;
    float velocityY_ = 0.0f;
    std::array<float, FaceMeshFilter::kPoseSize> poseVel_{};
    bool poseVelValid_ = false;
    bool poseMoving_ = false;
    float poseVelGain_ = 0.0f;
    int64_t last_ = kTimeInvalid;
    float sampleSpeed_ = 0.0f;
    float sampleMotion_ = 0.0f;
    float pendingDt_ = 0.0f;
    float steadyEyeTarget_ = 1.0f;
    float steadyFaceTarget_ = 1.0f;
    float eyeMakeupTarget_ = 1.0f;

    std::array<int64_t, kEvents> eventTimes_{};
    std::array<bool, kEvents> eventFace_{};
    int eventHead_ = 0;
    int eventCount_ = 0;
    bool intervalKnown_ = false;
    int64_t runFrom_ = kTimeInvalid;
    int64_t runStart_ = kTimeInvalid;
    int64_t prevRunStart_ = kTimeInvalid;
    int64_t shownMax_ = kTimeInvalid;

    float presence_ = 0.0f;
    float offsetX_ = 0.0f;
    float offsetY_ = 0.0f;
    float speed_ = 0.0f;
    float steadyEye_ = 1.0f;
    float steadyFace_ = 1.0f;
    float eyeMakeupKeep_ = 1.0f;
    float denoiseKeep_ = 1.0f;
    int runId_ = 0;
    float sampleIntervalMs_ = kDefaultIntervalMs;

    std::array<float, 32> cosTable_{};
    std::array<float, 32> sinTable_{};
};

}  // namespace livebeauty::math
