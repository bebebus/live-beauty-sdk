#include "livebeauty/math/face_relight.h"

#include <algorithm>
#include <cmath>

namespace livebeauty::math {

namespace {

inline float gauss2d(float u, float v, float cu, float cv, float su, float sv) {
    const float du = (u - cu) / su;
    const float dv = (v - cv) / sv;
    return std::exp(-0.5f * (du * du + dv * dv));
}

inline Vec3 normalize3(float x, float y, float z) {
    const float len = std::sqrt(x * x + y * y + z * z);
    if (len <= 1e-6f) return {0.0f, 0.0f, 1.0f};
    const float inv = 1.0f / len;
    return {x * inv, y * inv, z * inv};
}

inline float dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

}  // namespace

FaceRelightSolver::FaceRelightSolver() {
    canonicalNormals_.fill(0.0f);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& idx = FaceMeshTopology::kCanonicalIndices;
    constexpr int kN = FaceMeshTopology::kFaceVertexCount;

    // Accumulate area-weighted face normals across all 898 canonical triangles.
    for (int t = 0; t < FaceMeshTopology::kCanonicalTriangleCount; ++t) {
        const int i0 = idx[t * 3 + 0];
        const int i1 = idx[t * 3 + 1];
        const int i2 = idx[t * 3 + 2];
        if (i0 >= kN || i1 >= kN || i2 >= kN) continue;

        const float ax = xyz[i0 * 3 + 0], ay = xyz[i0 * 3 + 1], az = xyz[i0 * 3 + 2];
        const float bx = xyz[i1 * 3 + 0], by = xyz[i1 * 3 + 1], bz = xyz[i1 * 3 + 2];
        const float cx = xyz[i2 * 3 + 0], cy = xyz[i2 * 3 + 1], cz = xyz[i2 * 3 + 2];

        const float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;

        // Canonical face points toward +Z; orient every triangle normal consistently outward.
        if (nz < 0.0f) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        canonicalNormals_[i0 * 3 + 0] += nx;
        canonicalNormals_[i0 * 3 + 1] += ny;
        canonicalNormals_[i0 * 3 + 2] += nz;
        canonicalNormals_[i1 * 3 + 0] += nx;
        canonicalNormals_[i1 * 3 + 1] += ny;
        canonicalNormals_[i1 * 3 + 2] += nz;
        canonicalNormals_[i2 * 3 + 0] += nx;
        canonicalNormals_[i2 * 3 + 1] += ny;
        canonicalNormals_[i2 * 3 + 2] += nz;
    }

    // Normalize vertex normals and precompute subtle anatomical bone-plane accent (nose bridge, cheekbones, forehead, chin).
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    for (int i = 0; i < kN; ++i) {
        const Vec3 n = normalize3(
            canonicalNormals_[i * 3 + 0],
            canonicalNormals_[i * 3 + 1],
            canonicalNormals_[i * 3 + 2]);
        canonicalNormals_[i * 3 + 0] = n.x;
        canonicalNormals_[i * 3 + 1] = n.y;
        canonicalNormals_[i * 3 + 2] = n.z;

        const float u = uv[i * 2];
        const float v = uv[i * 2 + 1];
        const float bridge = gauss2d(u, v, 0.50f, 0.44f, 0.034f, 0.11f);
        const float forehead = gauss2d(u, v, 0.50f, 0.22f, 0.11f, 0.065f) * 0.80f;
        const float cheekR = gauss2d(u, v, 0.36f, 0.48f, 0.068f, 0.055f) * 0.85f;
        const float cheekL = gauss2d(u, v, 0.64f, 0.48f, 0.068f, 0.055f) * 0.85f;
        const float chin = gauss2d(u, v, 0.50f, 0.82f, 0.055f, 0.042f) * 0.65f;
        anatomicalAccent_[i] = std::max({bridge, forehead, cheekR, cheekL, chin});
    }
}

HeadPose3D FaceRelightSolver::estimateHeadPose(
    const float* landmarks478x2,
    float aspect,
    float axisX,
    float axisY,
    float yawWeight) {
    HeadPose3D pose{};
    if (!landmarks478x2 || !(aspect > 0.0f) || !std::isfinite(aspect)) return pose;

    const float m234x = landmarks478x2[234 * 2], m234y = landmarks478x2[234 * 2 + 1] / aspect;
    const float m454x = landmarks478x2[454 * 2], m454y = landmarks478x2[454 * 2 + 1] / aspect;
    const float m10x = landmarks478x2[10 * 2], m10y = landmarks478x2[10 * 2 + 1] / aspect;
    const float m152x = landmarks478x2[152 * 2], m152y = landmarks478x2[152 * 2 + 1] / aspect;
    const float noseX = landmarks478x2[1 * 2], noseY = landmarks478x2[1 * 2 + 1] / aspect;

    const float fw = std::hypot(m454x - m234x, m454y - m234y);
    const float fh = std::hypot(m10x - m152x, m10y - m152y);
    if (!(fw > 0.01f && fh > 0.01f)) return pose;

    const float cx = 0.25f * (m234x + m454x + m10x + m152x);
    const float cy = 0.25f * (m234y + m454y + m10y + m152y);

    // 1. Roll from eye/face horizontal axis (ux, uy)
    const float axisNorm = std::hypot(axisX, axisY);
    const float ux = (axisNorm > 1e-4f) ? (axisX / axisNorm) : 1.0f;
    const float uy = (axisNorm > 1e-4f) ? (axisY / axisNorm) : 0.0f;
    const float vx = -uy;
    const float vy = ux;
    const float roll = std::atan2(uy, ux);

    // 2. Yaw from nose-tip horizontal offset relative to cheek midpoint (signed, ~[-0.85, +0.85] rad)
    const float cheekMidX = 0.5f * (m234x + m454x);
    const float cheekMidY = 0.5f * (m234y + m454y);
    const float noseDu = ((noseX - cheekMidX) * ux + (noseY - cheekMidY) * uy) / fw;
    const float rawYaw = std::asin(detail::clampf(noseDu * 2.15f, -0.92f, 0.92f));
    const float yawCap = detail::clampf(yawWeight, 0.2f, 1.0f);
    const float yaw = rawYaw * (0.65f + 0.35f * yawCap);

    // 3. Pitch from nose vertical offset relative to eye-chin midpoint (neutral nose is ~0.04*fh below mid)
    const float vertMidX = 0.5f * (m10x + m152x);
    const float vertMidY = 0.5f * (m10y + m152y);
    const float noseDv = ((noseX - vertMidX) * vx + (noseY - vertMidY) * vy) / fh;
    const float pitch = std::asin(detail::clampf((noseDv + 0.04f) * 2.4f, -0.75f, 0.75f));

    // Compose R = Rz(roll) * Ry(yaw) * Rx(pitch)
    const float cr = std::cos(roll), sr = std::sin(roll);
    const float cyaw = std::cos(yaw), syaw = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);

    // Ry * Rx
    const float m00 = cyaw;
    const float m01 = syaw * sp;
    const float m02 = syaw * cp;
    const float m10 = 0.0f;
    const float m11 = cp;
    const float m12 = -sp;
    const float m20 = -syaw;
    const float m21 = cyaw * sp;
    const float m22 = cyaw * cp;

    // Rz * (Ry * Rx)
    pose.rot[0] = cr * m00 - sr * m10;
    pose.rot[1] = cr * m01 - sr * m11;
    pose.rot[2] = cr * m02 - sr * m12;
    pose.rot[3] = sr * m00 + cr * m10;
    pose.rot[4] = sr * m01 + cr * m11;
    pose.rot[5] = sr * m02 + cr * m12;
    pose.rot[6] = m20;
    pose.rot[7] = m21;
    pose.rot[8] = m22;

    pose.yawRad = yaw;
    pose.pitchRad = pitch;
    pose.rollRad = roll;
    pose.centerX = cx;
    pose.centerY = cy * aspect;
    pose.scale = fw;
    pose.valid = true;
    return pose;
}

Vec3 FaceRelightSolver::rotateVec(const HeadPose3D& pose, const Vec3& v) {
    return {
        pose.rot[0] * v.x + pose.rot[1] * v.y + pose.rot[2] * v.z,
        pose.rot[3] * v.x + pose.rot[4] * v.y + pose.rot[5] * v.z,
        pose.rot[6] * v.x + pose.rot[7] * v.y + pose.rot[8] * v.z,
    };
}

void FaceRelightSolver::computeVertexRelight(
    const HeadPose3D& pose,
    RelightStyle style,
    float relightStrength,
    float* out468) const {
    if (!out468) return;

    // Key light, soft bounce fill, and shadow-neutralization direction in camera view space (+Z toward camera, +Y up)
    Vec3 keyDir = normalize3(0.08f, 0.24f, 0.97f);
    Vec3 fillDir = normalize3(-0.22f, -0.35f, 0.91f);
    float keyWeight = 0.62f;
    float fillWeight = 0.26f;
    float accentWeight = 0.35f;

    switch (style) {
        case RelightStyle::kStudioSoftbox:
            keyDir = normalize3(0.14f, 0.28f, 0.95f);
            fillDir = normalize3(-0.25f, -0.32f, 0.91f);
            keyWeight = 0.68f;
            fillWeight = 0.28f;
            accentWeight = 0.32f;
            break;
        case RelightStyle::kButterfly:
            keyDir = normalize3(0.0f, 0.42f, 0.91f);
            fillDir = normalize3(0.0f, -0.45f, 0.89f);
            keyWeight = 0.72f;
            fillWeight = 0.24f;
            accentWeight = 0.38f;
            break;
        case RelightStyle::kRembrandt:
            keyDir = normalize3(0.52f, 0.34f, 0.78f);
            fillDir = normalize3(-0.35f, -0.18f, 0.92f);
            keyWeight = 0.76f;
            fillWeight = 0.16f;
            accentWeight = 0.28f;
            break;
        case RelightStyle::kNatural:
        default:
            break;
    }

    const float rl = detail::clampf(relightStrength, 0.0f, 1.0f);
    // Blend from subtle anatomical bone-plane highlight (when relight==0, preserving legacy radiance semantics)
    // to full physically-rotated 3D studio normal shading (when relight>0).
    const float threeDMix = 0.55f + 0.45f * rl;

    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const Vec3 nLocal{
            canonicalNormals_[i * 3 + 0],
            canonicalNormals_[i * 3 + 1],
            canonicalNormals_[i * 3 + 2],
        };
        const Vec3 nWorld = pose.valid ? normalize3(
            pose.rot[0] * nLocal.x + pose.rot[1] * nLocal.y + pose.rot[2] * nLocal.z,
            pose.rot[3] * nLocal.x + pose.rot[4] * nLocal.y + pose.rot[5] * nLocal.z,
            pose.rot[6] * nLocal.x + pose.rot[7] * nLocal.y + pose.rot[8] * nLocal.z) : nLocal;

        // Half-Lambert wrapped diffuse (smooth skin subsurface scattering feel, never harsh black terminator)
        const float ndlKey = std::max(0.0f, dot3(nWorld, keyDir));
        const float wrapKey = (ndlKey * 0.65f + 0.35f);
        const float keyLobe = wrapKey * wrapKey;

        // Upward bounce fill neutralizes harsh downward indoor overhead shadows under brows/eyes/nose
        const float ndlFill = std::max(0.0f, dot3(nWorld, fillDir));
        const float overheadShadowLift = detail::clampf(-nWorld.y * 0.5f + 0.5f, 0.0f, 1.0f) * 0.18f * rl;

        // Rim / side cheek contour depth (natural 3D face slimming through light falloff near grazing angles)
        const float facing = detail::clampf(nWorld.z, 0.0f, 1.0f);
        const float shading3d = detail::clampf(
            (keyLobe * keyWeight + ndlFill * fillWeight + overheadShadowLift) * (0.45f + 0.55f * facing),
            0.0f, 1.0f);

        const float accent = anatomicalAccent_[i];
        const float combined = detail::clampf(
            shading3d * threeDMix + accent * (1.0f - threeDMix * (1.0f - accentWeight)),
            0.0f, 1.0f);
        out468[i] = combined;
    }
}

}  // namespace livebeauty::math
