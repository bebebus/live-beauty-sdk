#pragma once

#include <array>
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

enum class RelightStyle : int {
    kNatural = 0,        // Balanced front-top studio fill (neutralizes harsh overhead shadows)
    kStudioSoftbox = 1,  // Soft studio key + warm cheek fill (TikTok/Douyin live room look)
    kButterfly = 2,      // Classic beauty butterfly light from upper center
    kRembrandt = 3,      // Sculpted 3/4 editorial key light with cheek triangle highlight
};

struct HeadPose3D {
    // Orthonormal 3x3 rotation matrix (row-major) mapping canonical face space (+X left, +Y up, +Z toward camera)
    // into camera view space.
    std::array<float, 9> rot = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f};
    float yawRad = 0.0f;
    float pitchRad = 0.0f;
    float rollRad = 0.0f;
    float centerX = 0.5f;   // Normalized UV [0..1]
    float centerY = 0.5f;   // Normalized UV [0..1] (bottom-up GL convention)
    float scale = 0.25f;    // Face width in normalized isotropic space
    bool valid = false;
};

/**
 * Real-time 3D facial vertex normal estimator & physically-motivated 3-point studio relighting solver.
 *
 * Precomputes smooth area-weighted 3D outward vertex normals for all 468 canonical MediaPipe landmarks
 * from FaceMeshTopology::kCanonicalXyz & kCanonicalIndices. Per frame, estimates orthonormal 3D head
 * pose (yaw/pitch/roll) from tracked landmarks, rotates normals into camera space, and computes a
 * pose-reactive studio relighting weight per vertex (zero GPU vertex-normal overhead, <0.08 ms/frame CPU).
 */
class FaceRelightSolver {
public:
    FaceRelightSolver();

    /** Access precomputed unit outward canonical vertex normals (468 x 3). */
    const std::array<float, FaceMeshTopology::kFaceVertexCount * 3>& canonicalNormals() const {
        return canonicalNormals_;
    }

    /**
     * Estimate 6-DoF head orientation & center/scale from 478-pt bottom-up GL landmarks.
     * `landmarks478x2` are normalized [0..1] bottom-up; `aspect = width / height`.
     */
    static HeadPose3D estimateHeadPose(
        const float* landmarks478x2,
        float aspect,
        float axisX,
        float axisY,
        float yawWeight);

    /**
     * Compute per-vertex 3D studio relighting lobe into `out468` in [0..1] (neutral baseline ~0.30,
     * highlights up to 1.0, shadow-lifted contours down to 0.05) for rasterization into the skin mask B channel.
     */
    void computeVertexRelight(
        const HeadPose3D& pose,
        RelightStyle style,
        float relightStrength,
        float* out468,
        bool mirrored = false) const;

    /** Rotate a canonical 3D vector by `pose.rot`. */
    static Vec3 rotateVec(const HeadPose3D& pose, const Vec3& v);

private:
    std::array<float, FaceMeshTopology::kFaceVertexCount * 3> canonicalNormals_{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> anatomicalAccent_{};
};

}  // namespace livebeauty::math
