#include "livebeauty/math/ar_3d_accessory.h"

#include <algorithm>
#include <cmath>

namespace livebeauty::math {

namespace {

constexpr float kDepthCenter = 0.50f;
constexpr float kDepthScale = 0.032f;  // Maps canonical Z (~[-8, +8]) safely into [0.24, 0.76]

inline float normalizeDepth(float zCanonicalScaled) {
    return detail::clampf(kDepthCenter + zCanonicalScaled * kDepthScale, 0.02f, 0.98f);
}

inline Vec3 rotateLocalEuler(const Vec3& v, float yaw, float pitch, float roll) {
    const float cr = std::cos(roll), sr = std::sin(roll);
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);

    // Rx(pitch)
    const float x1 = v.x;
    const float y1 = cp * v.y - sp * v.z;
    const float z1 = sp * v.y + cp * v.z;
    // Ry(yaw)
    const float x2 = cy * x1 + sy * z1;
    const float y2 = y1;
    const float z2 = -sy * x1 + cy * z1;
    // Rz(roll)
    return {cr * x2 - sr * y2, sr * x2 + cr * y2, z2};
}

}  // namespace

void SpringPendulum::reset() {
    initialized_ = false;
    lastMs_ = kTimeInvalid;
    prevAnchorX_ = 0.0f;
    prevAnchorY_ = 0.0f;
    prevVelX_ = 0.0f;
    prevVelY_ = 0.0f;
    prevRoll_ = 0.0f;
    angle_ = 0.0f;
    angleVel_ = 0.0f;
    offsetX_ = 0.0f;
    offsetY_ = 0.0f;
}

void SpringPendulum::update(
    float anchorX, float anchorY, float headRollRad,
    float stiffness, float damping, int64_t nowMs) {
    if (stiffness <= 1e-3f) {
        angle_ = 0.0f;
        angleVel_ = 0.0f;
        offsetX_ = 0.0f;
        offsetY_ = 0.0f;
        initialized_ = false;
        lastMs_ = nowMs;
        return;
    }
    if (!initialized_ || lastMs_ == kTimeInvalid) {
        initialized_ = true;
        lastMs_ = nowMs;
        prevAnchorX_ = anchorX;
        prevAnchorY_ = anchorY;
        prevVelX_ = 0.0f;
        prevVelY_ = 0.0f;
        prevRoll_ = headRollRad;
        angle_ = 0.0f;
        angleVel_ = 0.0f;
        offsetX_ = 0.0f;
        offsetY_ = 0.0f;
        return;
    }

    const float dt = detail::clampf(static_cast<float>(nowMs - lastMs_) * 0.001f, 0.001f, 0.05f);
    lastMs_ = nowMs;

    const float vx = (anchorX - prevAnchorX_) / dt;
    const float vy = (anchorY - prevAnchorY_) / dt;
    const float ax = detail::clampf((vx - prevVelX_) / dt, -25.0f, 25.0f);
    const float rollVel = detail::clampf((headRollRad - prevRoll_) / dt, -12.0f, 12.0f);

    prevAnchorX_ = anchorX;
    prevAnchorY_ = anchorY;
    prevVelX_ = vx;
    prevVelY_ = vy;
    prevRoll_ = headRollRad;

    // Damped harmonic oscillator: theta'' = -k * (theta - target) - c * theta' - drive
    const float k = detail::clampf(stiffness, 2.0f, 80.0f);
    const float c = detail::clampf(damping, 1.0f, 20.0f);
    const float drive = ax * 0.14f + rollVel * 0.22f;
    const float acc = -k * angle_ - c * angleVel_ - drive;
    angleVel_ = detail::clampf(angleVel_ + acc * dt, -10.0f, 10.0f);
    angle_ = detail::clampf(angle_ + angleVel_ * dt, -0.58f, 0.58f);

    // Subtle secondary translation lag coupled to pendulum swing
    offsetX_ = std::sin(angle_) * 0.018f;
    offsetY_ = (1.0f - std::cos(angle_)) * 0.008f;
}

void Ar3dSolver::buildHeadDepthShell(
    const float* landmarks478x2,
    const HeadPose3D& pose,
    float* outShell468x5) {
    if (!landmarks478x2 || !outShell468x5) return;
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const Vec3 pLocal{xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]};
        const Vec3 pWorld = pose.valid ? FaceRelightSolver::rotateVec(pose, pLocal) : pLocal;
        const float depth01 = normalizeDepth(pWorld.z);
        const int k = i * kFloatsPerVertex;
        outShell468x5[k + 0] = landmarks478x2[i * 2 + 0] * 2.0f - 1.0f;
        outShell468x5[k + 1] = landmarks478x2[i * 2 + 1] * 2.0f - 1.0f;
        outShell468x5[k + 2] = 0.0f;
        outShell468x5[k + 3] = 0.0f;
        outShell468x5[k + 4] = depth01;
    }
}

bool Ar3dSolver::buildAccessoryQuad(
    const Ar3dAccessoryConfig& cfg,
    const float* landmarks478x2,
    const HeadPose3D& pose,
    float aspect,
    const SpringPendulum* spring,
    float* outQuad20,
    float& outAnchorX,
    float& outAnchorY) {
    if (!cfg.enabled || cfg.texture == rhi::kInvalidTexture || cfg.opacity <= 1e-4f ||
        !landmarks478x2 || !pose.valid || !outQuad20 || !(aspect > 0.0f)) {
        return false;
    }

    const float fw = std::max(pose.scale, 0.02f);
    // Canonical face width in kCanonicalXyz is ~15.328 units (lm[454].x - lm[234].x)
    constexpr float kCanonicalFw = 15.328f;
    const float pxPerCanonical = fw / kCanonicalFw;

    // Pick screen anchor & base canonical Z depth per anatomical slot
    float ax = pose.centerX;
    float ay = pose.centerY;
    float baseZ = 4.5f;
    float defaultWidth = 0.85f * fw;

    auto lm = [&](int idx) -> std::pair<float, float> {
        return {landmarks478x2[idx * 2], landmarks478x2[idx * 2 + 1]};
    };

    switch (cfg.anchor) {
        case Ar3dAnchor::kHeadCrown: {
            const auto p10 = lm(10);
            const auto p152 = lm(152);
            ax = p10.first + (p10.first - p152.first) * 0.24f;
            ay = p10.second + (p10.second - p152.second) * 0.24f;
            baseZ = 4.6f;
            defaultWidth = 1.25f * fw;
            break;
        }
        case Ar3dAnchor::kGlassesBridge: {
            const auto p168 = lm(168);
            ax = p168.first;
            ay = p168.second;
            baseZ = 6.2f;
            defaultWidth = 1.12f * fw;
            break;
        }
        case Ar3dAnchor::kLeftEar: {
            const auto p454 = lm(454);
            const auto p361 = lm(361);
            ax = 0.5f * (p454.first + p361.first) + 0.06f * fw;
            ay = 0.5f * (p454.second + p361.second) - 0.14f * fw * aspect;
            baseZ = -2.2f;  // Sits at ear plane: naturally occluded behind cheek when turning head!
            defaultWidth = 0.34f * fw;
            break;
        }
        case Ar3dAnchor::kRightEar: {
            const auto p234 = lm(234);
            const auto p132 = lm(132);
            ax = 0.5f * (p234.first + p132.first) - 0.06f * fw;
            ay = 0.5f * (p234.second + p132.second) - 0.14f * fw * aspect;
            baseZ = -2.2f;
            defaultWidth = 0.34f * fw;
            break;
        }
        case Ar3dAnchor::kNoseTip: {
            const auto p1 = lm(1);
            ax = p1.first;
            ay = p1.second;
            baseZ = 8.2f;
            defaultWidth = 0.36f * fw;
            break;
        }
        case Ar3dAnchor::kChin: {
            const auto p152 = lm(152);
            ax = p152.first;
            ay = p152.second - 0.08f * fw * aspect;
            baseZ = 4.4f;
            defaultWidth = 0.80f * fw;
            break;
        }
    }

    outAnchorX = ax;
    outAnchorY = ay;

    // Apply local user 3D offset rotated into camera space
    const Vec3 offCanonical{
        cfg.offsetX * kCanonicalFw,
        cfg.offsetY * kCanonicalFw,
        cfg.offsetZ * kCanonicalFw,
    };
    const Vec3 offWorld = FaceRelightSolver::rotateVec(pose, offCanonical);
    const float swayDx = spring ? spring->swayOffsetX() : 0.0f;
    const float swayDy = spring ? spring->swayOffsetY() : 0.0f;
    const float swayAngle = spring ? spring->swayAngleRad() : 0.0f;

    const float cxIso = ax + offWorld.x * pxPerCanonical + swayDx;
    const float cyIso = ay / aspect + offWorld.y * pxPerCanonical + swayDy;
    const float czWorld = baseZ + offWorld.z;

    const float halfW = 0.5f * defaultWidth * detail::clampf(cfg.scale, 0.1f, 3.0f);
    const float texAsp = (cfg.textureAspect > 0.05f) ? cfg.textureAspect : 1.0f;
    const float halfH = halfW / texAsp;

    // 4 local quad corners in canonical units, rotated by user Euler + spring sway + head pose 3D rotation
    const float halfWCan = halfW / std::max(pxPerCanonical, 1e-5f);
    const float halfHCan = halfH / std::max(pxPerCanonical, 1e-5f);
    const float corners[4][4] = {
        {-halfWCan, -halfHCan, 0.0f, 1.0f},
        { halfWCan, -halfHCan, 1.0f, 1.0f},
        {-halfWCan,  halfHCan, 0.0f, 0.0f},
        { halfWCan,  halfHCan, 1.0f, 0.0f},
    };

    for (int i = 0; i < 4; ++i) {
        const Vec3 cLocal = rotateLocalEuler(
            {corners[i][0], corners[i][1], 0.0f},
            cfg.yawOffsetRad, cfg.pitchOffsetRad, cfg.rollOffsetRad + swayAngle);
        const Vec3 cWorld = FaceRelightSolver::rotateVec(pose, cLocal);
        const float vx = cxIso + cWorld.x * pxPerCanonical;
        const float vy = (cyIso + cWorld.y * pxPerCanonical) * aspect;
        const float vz = czWorld + cWorld.z;
        const int k = i * kFloatsPerVertex;
        outQuad20[k + 0] = vx * 2.0f - 1.0f;
        outQuad20[k + 1] = vy * 2.0f - 1.0f;
        outQuad20[k + 2] = corners[i][2];
        outQuad20[k + 3] = corners[i][3];
        outQuad20[k + 4] = normalizeDepth(vz);
    }
    return true;
}

}  // namespace livebeauty::math
