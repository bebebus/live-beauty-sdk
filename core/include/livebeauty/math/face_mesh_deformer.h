#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include "livebeauty/config.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

class FaceMeshDeformer {
public:
    struct Params {
        float faceSlim = 0.0f;
        float vFace = 0.0f;
        float faceNarrow = 0.0f;
        float faceSmall = 0.0f;
        float cheekboneSlim = 0.0f;
        float chinShape = 0.0f;
        float foreheadShape = 0.0f;
        float eyeEnlarge = 0.0f;
        float eyeDistance = 0.0f;
        float eyeCorner = 0.0f;
        float noseSlim = 0.0f;
        float noseLength = 0.0f;
        float mouthSize = 0.0f;
        float mouthSmile = 0.0f;
        float cranialTop = 0.0f;

        bool isActive() const {
            constexpr float kEps = 1e-4f;
            return faceSlim > kEps || vFace > kEps || faceNarrow > kEps || faceSmall > kEps ||
                   cheekboneSlim > kEps || std::fabs(chinShape) > kEps || std::fabs(foreheadShape) > kEps ||
                   cranialTop > kEps || eyeEnlarge > kEps || std::fabs(eyeDistance) > kEps ||
                   std::fabs(eyeCorner) > kEps || noseSlim > kEps || std::fabs(noseLength) > kEps ||
                   std::fabs(mouthSize) > kEps || mouthSmile > kEps;
        }

        static Params fromConfig(const BeautyConfig& c) {
            return Params{
                c.faceSlim, c.vFace, c.faceNarrow, c.faceSmall, c.cheekboneSlim,
                c.chinShape, c.foreheadShape, c.eyeEnlarge, c.eyeDistance, c.eyeCorner,
                c.noseSlim, c.noseLength, c.mouthSize, c.mouthSmile, c.cranialTop,
            };
        }
    };

    static constexpr float kRhoFade = 0.15f;
    static constexpr float kMidInteriorCarry = 0.55f;
    static constexpr float kCranialOuterPad = 0.185f;
    static constexpr float kCanonFhOverFw = 1.153f;
    static constexpr float kVisTauDownMs = 55.0f;
    static constexpr float kVisTauUpMs = 65.0f;
    static constexpr int kRelaxIter = 8;
    static constexpr float kRelaxHFloor = 0.65f;
    static constexpr float kRelaxR0 = 0.08f;
    static constexpr float kRelaxR1 = 0.30f;
    static constexpr float kFrontWShape = 0.008f;
    static constexpr float kSafetyW0 = 0.3f;
    static constexpr float kSafetyW1 = 0.7f;
    static constexpr float kSafetyR0 = 0.0f;
    static constexpr float kSafetyR1 = 0.04f;
    static constexpr float kSafetyDepth = 0.5f;
    static constexpr float kRelaxTauUpMs = 120.0f;
    static constexpr float kRelaxTauDownMs = 90.0f;
    static constexpr float kDispTauSteadyMs = 95.0f;
    static constexpr float kDispTauInnerMs = 28.0f;
    static constexpr float kDispTauFastMs = 18.0f;
    static constexpr int64_t kTemporalResetMs = 150;

    FaceMeshDeformer() = default;

    const float* vertexData() const { return vertexData_.data(); }
    int vertexDataFloatCount() const { return FaceMeshTopology::kTotalVertexCount * 4; }
    const float* warpedLandmarks() const { return warpedLandmarks_.data(); }
    const float* vertexVisibility() const { return vis_.data(); }
    bool frontFaceCw() const { return frontFaceCw_; }

    void resetTemporal();

    bool deform(
        const float* landmarks478x2,
        float aspect,
        float axisX, float axisY,
        float faceWidth, float faceHeight,
        float leftWeight, float rightWeight, float yawWeight,
        float presence, float steadyFace, float steadyEye,
        const Params& params,
        int64_t nowMs = kTimeInvalid);

private:
    float areaRatio(int t) const;
    void limitVertices(int t, float f);
    void applyDisplacement();
    float softMaxSpan(const uint16_t* candidates, int count, float signDir,
                      float midX, float midY, float ux, float uy, float beta) const;
    float softMaxOvalExcess(int ovalSlot, float ox, float oy, float rx, float ry, float beta) const;
    void transformIris(const float* src, int baseIdx, int outer, int inner, int top, int bottom, float aspect);
    static Vec2 eyeEnlargeDelta(
        float px, float py, float eyeX, float eyeY, float eyeW, float eyeH,
        float browDistV, float ux, float uy, float vx, float vy, float strength);

    std::array<float, FaceMeshTopology::kTotalVertexCount * 4> vertexData_{};
    std::array<float, FaceMeshFrame::kCoordCount> warpedLandmarks_{};
    bool frontFaceCw_ = false;

    std::array<float, FaceMeshTopology::kTotalVertexCount> srcMx_{};
    std::array<float, FaceMeshTopology::kTotalVertexCount> srcMy_{};
    std::array<float, FaceMeshTopology::kTotalVertexCount> dstMx_{};
    std::array<float, FaceMeshTopology::kTotalVertexCount> dstMy_{};
    std::array<float, FaceMeshTopology::kBufferOffset> dispMx_{};
    std::array<float, FaceMeshTopology::kBufferOffset> dispMy_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> interiorMx_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> interiorMy_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> smoothMx_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> smoothMy_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> vis_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> visScratch_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> relax_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> relaxScratch_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> relaxFactor_{};
    std::array<float, FaceMeshFilter::kPoseSize> pose_{};
    std::array<float, FaceMeshTopology::kCanonicalTriangleCount> triVis_{};
    std::array<float, FaceMeshTopology::kDeformableTriangleCount> triFront_{};
    std::array<float, FaceMeshTopology::kDeformableTriangleCount> triOrig_{};

    std::array<float, FaceMeshTopology::kFaceVertexCount> visState_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> relaxState_{};
    std::array<float, FaceMeshTopology::kOvalVertexCount> silExcessRatioState_{};
    std::array<float, FaceMeshTopology::kBufferOffset> dispUState_{};
    std::array<float, FaceMeshTopology::kBufferOffset> dispVState_{};
    Params prevParams_{};
    float wLeftRatioState_ = 0.0f;
    float wRightRatioState_ = 0.0f;
    float eyeDuLeftRatioState_ = 0.0f;
    float eyeDuRightRatioState_ = 0.0f;
    float rEyeWRatioState_ = 0.0f;
    float lEyeWRatioState_ = 0.0f;
    float rEyeHRatioState_ = 0.0f;
    float lEyeHRatioState_ = 0.0f;
    float signedYawState_ = 0.0f;
    float bisectLambdaState_ = 1.0f;
    float prevEffYawW_ = 1.0f;
    float prevEyeMod_ = 1.0f;
    float prevFaceMod_ = 1.0f;
    bool temporalValid_ = false;
    bool dispTemporalValid_ = false;
    int64_t lastNowMs_ = kTimeInvalid;
};

}  // namespace livebeauty::math
