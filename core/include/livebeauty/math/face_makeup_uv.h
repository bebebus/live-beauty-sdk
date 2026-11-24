#pragma once

#include <cstdint>
#include <vector>
#include "livebeauty/config.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

struct FaceMakeupUv {
    static constexpr int kVertexStrideFloats = 6;  // dstNdcX, dstNdcY, screenU, screenV, makeupU, makeupV
    static constexpr int kVertexDataSize = FaceMeshTopology::kFaceVertexCount * kVertexStrideFloats;

    static const std::vector<uint16_t>& skinFaceIndices();
    static const std::vector<uint16_t>& innerMouthIndices();
    static const std::vector<uint16_t>& eyeOpeningIndices();

    static void packVertices(const float* landmarks, int coordCount, float* outVertices);
    static bool computeIrisCircle(const float* landmarks, int coordCount, int baseIdx, float aspect, float outCircle[4]);

    static Vec3 lipstickRgb(LipstickColor color);
    static Vec3 blushRgb(BlushColor color);
    static Vec4 blushChannel(BlushStyle style);
    static Vec3 eyebrowRgb(EyebrowColor color);
    static Vec4 eyebrowChannel(EyebrowStyle style);
    static Vec3 eyeshadowMainRgb(EyeshadowColor color);
    static Vec3 eyeshadowDeepRgb(EyeshadowColor color);
    static Vec2 irisTile(ContactLensStyle style);
    static Vec3 hairDyeRgb(HairDyeColor color);
};

/** 468-pt anatomical sub-mesh vertex weights & packing for dark circles, nasolabial folds, bright eyes & teeth. */
struct FaceRetouchSubMesh {
    static constexpr int kStrideFloats = 8;  // dstNdcX, dstNdcY, srcU, srcV, wTear, wFold, offU, offV
    static constexpr int kVertexDataSize = FaceMeshTopology::kFaceVertexCount * kStrideFloats;

    static const std::array<float, FaceMeshTopology::kFaceVertexCount>& staticTearWeights();
    static const std::array<float, FaceMeshTopology::kFaceVertexCount>& staticFoldWeights();
    static const std::array<float, FaceMeshTopology::kFaceVertexCount * 2>& staticFoldNormals();
    static const std::vector<uint16_t>& skinRetouchIndices();

    static void packVertices(
        const float* landmarks,
        const float* meshVertices4,  // optional 548*4 vertexData from FaceMeshDeformer
        float ux, float uy, float vx, float vy,
        float faceWidth, float faceHeight, float aspect,
        float mirrorSign, float dc, float nf,
        float* outVertices);
};

}  // namespace livebeauty::math
