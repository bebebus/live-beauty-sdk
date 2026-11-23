#pragma once

#include <array>
#include <cstdint>
#include "livebeauty/types.h"

namespace livebeauty::math {

struct FaceGeometryConstants {
    static constexpr int kStateSize = 41;
    static constexpr int kOvalIndex = 22;
    static constexpr int kRxIndex = 24;
    static constexpr int kAxisIndex = 26;
};

struct FaceMeshFrame {
    static constexpr int kFaceCount = 468;
    static constexpr int kLandmarkCount = 478;
    static constexpr int kCoordCount = kLandmarkCount * 2;
    static constexpr double kMaxYawRatio = 0.72;

    static constexpr int kMetaCenterX = 0;
    static constexpr int kMetaCenterY = 1;
    static constexpr int kMetaAxisX = 2;
    static constexpr int kMetaAxisY = 3;
    static constexpr int kMetaFaceWidth = 4;
    static constexpr int kMetaFaceHeight = 5;
    static constexpr int kMetaSignedYaw = 6;
    static constexpr int kMetaLeftWeight = 7;
    static constexpr int kMetaRightWeight = 8;
    static constexpr int kMetaYawWeight = 9;
    static constexpr int kMetaRightEyeOpen = 10;
    static constexpr int kMetaLeftEyeOpen = 11;
    static constexpr int kMetaMouthOpen = 12;
    static constexpr int kMetaRightIrisGate = 13;
    static constexpr int kMetaLeftIrisGate = 14;
    static constexpr int kMetaSize = 15;

    std::array<float, kCoordCount> landmarks{};
    std::array<float, kMetaSize> meta{};
    bool hasIris = false;

    float centerX() const { return meta[kMetaCenterX]; }
    float centerY() const { return meta[kMetaCenterY]; }
    float axisX() const { return meta[kMetaAxisX]; }
    float axisY() const { return meta[kMetaAxisY]; }
    float faceWidth() const { return meta[kMetaFaceWidth]; }
    float faceHeight() const { return meta[kMetaFaceHeight]; }
    float signedYaw() const { return meta[kMetaSignedYaw]; }
    float leftWeight() const { return meta[kMetaLeftWeight]; }
    float rightWeight() const { return meta[kMetaRightWeight]; }
    float yawWeight() const { return meta[kMetaYawWeight]; }
    float rightEyeOpen() const { return meta[kMetaRightEyeOpen]; }
    float leftEyeOpen() const { return meta[kMetaLeftEyeOpen]; }
    float mouthOpen() const { return meta[kMetaMouthOpen]; }
    float rightIrisGate() const { return meta[kMetaRightIrisGate]; }
    float leftIrisGate() const { return meta[kMetaLeftIrisGate]; }

    /** Extract 478-pt frame from MediaPipe image-normalized coordinates (y downward, pointCount >= 468). */
    static bool fromMediaPipe(const float* mpXy, int pointCount, float aspect, FaceMeshFrame& out);

    /** Compute scalar metadata from 468-pt GL texture coordinates (y upward). */
    static bool fillMeta(const float* landmarks, int coordCount, float aspect, float* outMeta);

    /** Check whether 468-pt GL texture coordinates have CW front winding (horizontally mirrored). */
    static bool isFrontFaceCw(const float* landmarks, int coordCount);

    /** Compute 41-D FaceGeometry state vector from 468-pt GL texture coordinates. */
    static bool fillGeometryState(
        const float* landmarks, int coordCount,
        float axisX, float axisY, float aspect, float warpWeight,
        float* outState);

    /** Compute 41-D FaceGeometry state directly from MediaPipe normalized coordinates (y downward). */
    static bool faceGeometryFromMediaPipe(const float* mpXy, int pointCount, float aspect, float* outState);

    /** Synthesize 5 iris points when input only has 468 points. */
    static void synthesizeIris(float* landmarks, int baseIdx, int outer, int inner, int top, int bottom, float aspect);
};

}  // namespace livebeauty::math
