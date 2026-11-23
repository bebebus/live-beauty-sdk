#pragma once

#include <array>
#include <cstdint>
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

class FaceMeshFilter {
public:
    static constexpr int kContinue = 0;
    static constexpr int kNewRun = 1;
    static constexpr int kPoseSize = 8;
    static constexpr int64_t kMaxGapMs = 300;

    static constexpr float kPoseFcMin = 2.0f;
    static constexpr float kPoseBeta = 60.0f;
    static constexpr float kPoseDcut = 5.0f;
    static constexpr float kKPose = 1.8f;
    static constexpr float kCPose = 0.25f;
    static constexpr float kResDcut = 1.5f;
    static constexpr float kKRes = 1.4f;
    static constexpr float kCRes = 0.20f;
    static constexpr float kResTeExp = 0.85f;
    static constexpr float kNfTauMs = 250.0f;
    static constexpr float kNfRise = 0.3f;
    static constexpr float kJumpT = 0.5f;
    static constexpr float kJumpLogS = 0.35f;
    static constexpr float kDetMin = 0.2f;
    static constexpr float kIrisFc = 1.0f;
    static constexpr float kIrisBeta = 10.0f;
    static constexpr float kIrisDcut = 2.0f;
    static constexpr float kIrisKDead = 2.0f;
    static constexpr float kIrisKFc = 0.3f;
    static constexpr float kIrisKMin = 0.16f;
    static constexpr float kIrisKMax = 0.28f;
    static constexpr float kGateFc = 1.2f;
    static constexpr float kGateBeta = 3.0f;
    static constexpr float kGateDcut = 2.0f;
    static constexpr float kGateClose = 0.40f;
    static constexpr float kGateOpen = 0.55f;
    static constexpr float kGateRampMs = 120.0f;
    static constexpr float kBaseUpS = 1.0f;
    static constexpr float kBaseDownS = 15.0f;
    static constexpr float kBaseMin = 0.15f;
    static constexpr float kBaseMax = 0.45f;

    FaceMeshFilter() = default;

    bool jumped() const { return jumped_; }
    float poseSpeed() const { return poseSpeed_; }
    float poseDead() const { return poseDead_; }
    float noiseFloor() const { return noiseFloor_; }
    float motionSpeed() const { return motionSpeed_; }
    float poseScale() const { return poseScale_; }
    float rightGate() const { return gateValue_[0]; }
    float leftGate() const { return gateValue_[1]; }
    float rightOpen() const { return gateOpen_[0]; }
    float leftOpen() const { return gateOpen_[1]; }

    void reset(bool keepBaseline = false);

    int filter(int64_t ts, const float* raw478x2, float aspect, float* out478x2, float* poseOut8);

    static float canonicalFaceWidth();
    static void fitPose(const float* mx, const float* my, float* poseOut8);
    static float detOf(const float* pose8);
    static float scaleOf(const float* pose8);
    static float viewScale2(const float* pose8);
    static void extrapolate(const float* src478x2, const float* pose8, const float* poseVel8,
                            float horizonMs, float aspect, float* out478x2);
    static void centerShift(const float* src478x2, const float* pose8, const float* poseVel8,
                            float horizonMs, float aspect, float outShift[2]);

private:
    static constexpr int kFace = FaceMeshFrame::kFaceCount;
    static constexpr int kRegions = 3;
    static constexpr int kIrisRight = 468;
    static constexpr int kIrisLeft = 473;
    static constexpr int kEyeRegion = 1;
    static constexpr int kLowRegion = 2;
    static constexpr float kTeRefMs = 1000.0f / 30.0f;

    void startRun();
    void step(float teMs);
    void residual(const float* pose8, float* outRx, float* outRy) const;
    void updateIris(int eye, int base, int a, int b, float teMs, bool restart);
    void updateGate(int eye, const int* lids, int lidCount, int a, int b, float teMs, bool restart);

    std::array<float, FaceMeshFrame::kLandmarkCount> mX_{};
    std::array<float, FaceMeshFrame::kLandmarkCount> mY_{};
    std::array<float, FaceMeshFrame::kLandmarkCount> outX_{};
    std::array<float, FaceMeshFrame::kLandmarkCount> outY_{};
    std::array<float, kFace> rRawX_{};
    std::array<float, kFace> rRawY_{};
    std::array<float, kFace> rPrevX_{};
    std::array<float, kFace> rPrevY_{};
    std::array<float, kFace> rX_{};
    std::array<float, kFace> rY_{};
    std::array<float, kFace> drX_{};
    std::array<float, kFace> drY_{};
    std::array<float, kPoseSize> q_{};
    std::array<float, kPoseSize> qRaw_{};
    std::array<float, kPoseSize> qRawPrev_{};
    std::array<float, kRegions> regionRms_{};
    std::array<float, kRegions> regionAlpha_{};

    bool started_ = false;
    int64_t lastTs_ = kTimeInvalid;
    float vp_ = 0.0f;
    float noisePower_ = -1.0f;
    float noiseMin_ = 0.0f;

    std::array<float, 2> irisQx_{};
    std::array<float, 2> irisQy_{};
    std::array<float, 2> irisRawX_{};
    std::array<float, 2> irisRawY_{};
    std::array<float, 2> irisDx_{};
    std::array<float, 2> irisDy_{};
    std::array<float, 2> irisK_{};
    std::array<float, 2> irisPower_ = {-1.0f, -1.0f};
    std::array<float, 2> irisMin_{};

    std::array<float, 2> gateBase_ = {-1.0f, -1.0f};
    std::array<float, 2> gateX_{};
    std::array<float, 2> gatePrev_{};
    std::array<float, 2> gateDx_{};
    std::array<bool, 2> gateClosed_ = {false, false};
    std::array<float, 2> gateValue_ = {1.0f, 1.0f};
    std::array<float, 2> gateOpen_{};

    bool jumped_ = false;
    float poseSpeed_ = 0.0f;
    float poseDead_ = std::numeric_limits<float>::max();
    float noiseFloor_ = 0.0f;
    float motionSpeed_ = 0.0f;
    float poseScale_ = 0.0f;
};

}  // namespace livebeauty::math
