#include "livebeauty/math/face_mesh_filter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "livebeauty/math/face_mesh_topology.h"

namespace livebeauty::math {

namespace {

constexpr int kLidRing[32] = {
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
};
constexpr int kLipRing[40] = {
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
};
constexpr int kRightLids[6] = {160, 144, 159, 145, 158, 153};
constexpr int kLeftLids[6] = {387, 373, 386, 374, 385, 380};
constexpr int kCenterPoints[4] = {234, 454, 10, 152};
constexpr float kResFcMin[4] = {0.65f, 1.20f, 1.25f, 0.52f};
constexpr float kResBeta[4] = {24.0f, 68.0f, 72.0f, 10.0f};

struct FilterStaticData {
    float canonicalFaceWidth = 1.0f;
    std::array<float, FaceMeshFrame::kFaceCount> lx{};
    std::array<float, FaceMeshFrame::kFaceCount> ly{};
    std::array<float, FaceMeshFrame::kFaceCount> z{};
    std::array<float, FaceMeshFrame::kLandmarkCount> zExt{};
    std::vector<int> basis;
    std::vector<float> basisW;
    std::vector<float> kMat;  // 8 x (2*n) row-major
    std::array<float, FaceMeshFrame::kFaceCount * 4> reg{};
    std::array<float, 4> regSum{};

    FilterStaticData() {
        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        canonicalFaceWidth = std::hypot(
            xyz[454 * 3] - xyz[234 * 3],
            std::hypot(xyz[454 * 3 + 1] - xyz[234 * 3 + 1], xyz[454 * 3 + 2] - xyz[234 * 3 + 2]));

        const auto& uv = FaceMeshTopology::kCanonicalUv;
        // Strictly rigid craniofacial skull basis (~117 landmarks):
        // nose bridge/tip/sides, upper forehead, temples (deep -Z), upper zygomatic cheekbone,
        // and bony orbital corners (33/133/263/362). Excludes eyelids, eyebrows/brow-band,
        // mouth/lips, chin, lower jaw, and smile cheeks so expressions never pull rigid pose,
        // while inverse-variance weighting cuts Z-column (w1, w2) noise variance by ~4.6x with
        // exact 1.000 3D yaw gain and zero phase lag (eliminating deep-Z cheek twitching).
        std::array<bool, FaceMeshFrame::kFaceCount> isExcluded{};
        isExcluded.fill(false);
        for (int id : kLidRing) isExcluded[id] = true;
        isExcluded[33] = false;
        isExcluded[133] = false;
        isExcluded[263] = false;
        isExcluded[362] = false;
        constexpr int kBrowIndices[20] = {
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        };
        for (int id : kBrowIndices) isExcluded[id] = true;

        std::vector<double> weights;
        auto eyeInnerDist = [&](int a, int b, double u, double v) {
            const double cu = 0.5 * (uv[a * 2] + uv[b * 2]);
            const double ew = std::fabs(uv[b * 2] - uv[a * 2]);
            const double cv = 0.5 * (uv[a * 2 + 1] + uv[b * 2 + 1]) - 0.04 * ew;
            return std::hypot((u - cu) / (ew * 0.62), (v - cv) / (ew * 0.60));
        };
        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            if (isExcluded[i]) continue;
            const double u = uv[i * 2];
            const double v = uv[i * 2 + 1];
            const double dMouth = std::hypot((u - 0.50) / 0.25, (v - 0.695) / 0.16);
            if (dMouth < 1.15 || v > 0.62) continue;
            const bool isOrbitalCorner = (i == 33 || i == 133 || i == 263 || i == 362);
            if (!isOrbitalCorner &&
                (eyeInnerDist(33, 133, u, v) < 1.0 || eyeInnerDist(362, 263, u, v) < 1.0)) {
                continue;
            }
            if (v >= 0.30 && v <= 0.39 && std::fabs(u - 0.50) >= 0.05 && std::fabs(u - 0.50) <= 0.34) {
                continue;
            }
            const double rUv = std::hypot((u - 0.50) / 0.36, (v - 0.50) / 0.43);
            const double isContour = static_cast<double>(detail::smoothstep(0.62f, 0.90f, static_cast<float>(rUv)));
            const double sigma = 0.85 * (1.0 - isContour) + 2.5 * isContour;
            basis.push_back(i);
            weights.push_back(1.0 / (sigma * sigma));
        }
        double wSum = 0.0;
        for (double w : weights) wSum += w;
        basisW.resize(basis.size());
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (size_t j = 0; j < basis.size(); ++j) {
            const double nw = weights[j] / wSum;
            basisW[j] = static_cast<float>(nw);
            const int b = basis[j];
            cx += nw * xyz[b * 3];
            cy += nw * xyz[b * 3 + 1];
            cz += nw * xyz[b * 3 + 2];
        }
        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            lx[i] = static_cast<float>((xyz[i * 3] - cx) / canonicalFaceWidth);
            ly[i] = static_cast<float>((xyz[i * 3 + 1] - cy) / canonicalFaceWidth);
            z[i] = static_cast<float>((xyz[i * 3 + 2] - cz) / canonicalFaceWidth);
            zExt[i] = z[i];
        }
        const float rightEyeZ = (z[33] + z[133] + z[159] + z[145]) * 0.25f;
        const float leftEyeZ = (z[362] + z[263] + z[386] + z[374]) * 0.25f;
        for (int k = 0; k < 5; ++k) {
            zExt[468 + k] = rightEyeZ;
            zExt[473 + k] = leftEyeZ;
        }

        const int n = static_cast<int>(basis.size());
        const int rows = 2 * n;
        std::vector<std::array<double, 8>> d(rows);
        std::vector<double> w(rows, 0.0);
        for (int j = 0; j < n; ++j) {
            const int b = basis[j];
            d[j].fill(0.0);
            d[n + j].fill(0.0);
            d[j][0] = 1.0; d[j][2] = lx[b]; d[j][3] = ly[b]; d[j][6] = z[b];
            d[n + j][1] = 1.0; d[n + j][4] = lx[b]; d[n + j][5] = ly[b]; d[n + j][7] = z[b];
            w[j] = basisW[j];
            w[n + j] = basisW[j];
        }
        double normal[8][8] = {};
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                double acc = 0.0;
                for (int k = 0; k < rows; ++k) acc += d[k][r] * w[k] * d[k][c];
                normal[r][c] = acc;
            }
        }
        // Gauss-Jordan 8x8 invert
        double aug[8][16] = {};
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                aug[r][c] = normal[r][c];
                aug[r][8 + c] = (r == c) ? 1.0 : 0.0;
            }
        }
        for (int col = 0; col < 8; ++col) {
            int pivot = col;
            for (int r = col + 1; r < 8; ++r) {
                if (std::fabs(aug[r][col]) > std::fabs(aug[pivot][col])) pivot = r;
            }
            if (pivot != col) {
                for (int c = 0; c < 16; ++c) std::swap(aug[col][c], aug[pivot][c]);
            }
            const double p = aug[col][col];
            for (int c = 0; c < 16; ++c) aug[col][c] /= p;
            for (int r = 0; r < 8; ++r) {
                if (r == col) continue;
                const double f = aug[r][col];
                if (f != 0.0) {
                    for (int c = 0; c < 16; ++c) aug[r][c] -= f * aug[col][c];
                }
            }
        }
        kMat.resize(8 * rows);
        for (int r = 0; r < 8; ++r) {
            for (int k = 0; k < rows; ++k) {
                double v = 0.0;
                for (int c = 0; c < 8; ++c) v += aug[r][8 + c] * d[k][c] * w[k];
                kMat[r * rows + k] = static_cast<float>(v);
            }
        }

        std::array<bool, FaceMeshFrame::kFaceCount> isOval{};
        isOval.fill(false);
        for (uint16_t oi : FaceMeshTopology::kFaceOval) {
            if (oi < FaceMeshFrame::kFaceCount) isOval[oi] = true;
        }
        auto eyeMember = [&](int i, int a, int b) {
            const float cu = 0.5f * (uv[a * 2] + uv[b * 2]);
            const float ew = std::fabs(uv[b * 2] - uv[a * 2]);
            const float cv = 0.5f * (uv[a * 2 + 1] + uv[b * 2 + 1]) - 0.14f * ew;
            const float dd = std::hypot((uv[i * 2] - cu) / ew, (uv[i * 2 + 1] - cv) / (0.95f * ew));
            return 1.0f - detail::smoothstep(0.65f, 1.05f, dd);
        };
        auto mouthMember = [&](int i) {
            const float dd = std::hypot((uv[i * 2] - 0.50f) / 0.19f, (uv[i * 2 + 1] - 0.695f) / 0.115f);
            return 1.0f - detail::smoothstep(0.62f, 1.02f, dd);
        };
        auto contourMember = [&](int i) {
            const float rUv = std::hypot((uv[i * 2] - 0.50f) / 0.36f, (uv[i * 2 + 1] - 0.50f) / 0.43f);
            return detail::smoothstep(0.66f, 0.92f, rUv);
        };
        regSum.fill(0.0f);
        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            const float eye = std::max(eyeMember(i, 33, 133), eyeMember(i, 362, 263));
            const float mouth = isOval[i] ? 0.0f : std::min(mouthMember(i), 1.0f - eye);
            const float rem = std::max(0.0f, 1.0f - eye - mouth);
            const float contour = rem * (isOval[i] ? 1.0f : contourMember(i));
            const float mid = rem - contour;
            reg[i * 4 + 0] = mid;
            reg[i * 4 + 1] = eye;
            reg[i * 4 + 2] = mouth;
            reg[i * 4 + 3] = contour;
            for (int g = 0; g < 4; ++g) regSum[g] += reg[i * 4 + g];
        }
    }
};

const FilterStaticData& staticData() {
    static const FilterStaticData kData;
    return kData;
}

}  // namespace

float FaceMeshFilter::canonicalFaceWidth() { return staticData().canonicalFaceWidth; }

float FaceMeshFilter::detOf(const float* pose8) {
    return pose8[2] * pose8[5] - pose8[3] * pose8[4];
}

float FaceMeshFilter::scaleOf(const float* pose8) {
    return std::sqrt(std::fabs(detOf(pose8)));
}

float FaceMeshFilter::viewScale2(const float* pose8) {
    const float cx = pose8[3] * pose8[7] - pose8[6] * pose8[5];
    const float cy = pose8[6] * pose8[4] - pose8[2] * pose8[7];
    const float cz = pose8[2] * pose8[5] - pose8[3] * pose8[4];
    return std::sqrt(cx * cx + cy * cy + cz * cz);
}

void FaceMeshFilter::fitPose(const float* mx, const float* my, float* poseOut8) {
    const auto& sd = staticData();
    const int n = static_cast<int>(sd.basis.size());
    const int rows = 2 * n;
    for (int r = 0; r < kPoseSize; ++r) {
        float v = 0.0f;
        const int base = r * rows;
        for (int j = 0; j < n; ++j) {
            const int b = sd.basis[j];
            v += sd.kMat[base + j] * mx[b] + sd.kMat[base + n + j] * my[b];
        }
        poseOut8[r] = v;
    }
}

float FaceMeshFilter::angularVelOf(const float* pose8, const float* poseVel8) {
    const float det = detOf(pose8);
    if (std::fabs(det) < 1e-10f) return 0.0f;
    const float s2 = std::max(std::fabs(det), 1e-8f);
    // Exact mirror- and roll-angle-invariant in-plane angular velocity omega from M = A * diag(detSign, 1) = s * R(theta):
    // works identically for both rear camera (det > 0) and mirrored front selfie camera (det < 0).
    const float detSign = (det >= 0.0f) ? 1.0f : -1.0f;
    const float m11 = pose8[2] * detSign;
    const float m12 = pose8[3];
    const float m21 = pose8[4] * detSign;
    const float m22 = pose8[5];
    const float dm11 = poseVel8[2] * detSign;
    const float dm12 = poseVel8[3];
    const float dm21 = poseVel8[4] * detSign;
    const float dm22 = poseVel8[5];
    const float omega = 0.5f * ((m11 * dm21 - m21 * dm11) + (m12 * dm22 - m22 * dm12)) / s2;
    return detail::clampf(omega, -0.0040f, 0.0040f);
}

void FaceMeshFilter::extrapolate(
    const float* src478x2, const float* pose8, const float* poseVel8,
    float horizonMs, float aspect, float* out478x2) {
    const float det = detOf(pose8);
    if (std::fabs(det) < 1e-10f || !(aspect > 0.0f)) {
        if (out478x2 != src478x2) {
            std::memcpy(out478x2, src478x2, sizeof(float) * FaceMeshFrame::kCoordCount);
        }
        return;
    }
    // Pure 2D Euclidean SE(2) rigid isometry (curvature-compensated translation + sign-invariant in-plane roll):
    // preserves all 2D facial widths identically and never extrapolates noisy Z-perspective derivatives!
    const float omega = angularVelOf(pose8, poseVel8);
    const float cx = pose8[0];
    const float cy = pose8[1];
    const float h = horizonMs;
    const float dTheta = h * omega;
    const float c = std::cos(dTheta);
    const float sn = std::sin(dTheta);
    for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
        const float rx = src478x2[i * 2] - cx;
        const float ry = src478x2[i * 2 + 1] / aspect - cy;
        const float mx = h * poseVel8[0] + (c - 1.0f) * rx - sn * ry;
        const float my = h * poseVel8[1] + sn * rx + (c - 1.0f) * ry;
        out478x2[i * 2] = src478x2[i * 2] + mx;
        out478x2[i * 2 + 1] = src478x2[i * 2 + 1] + my * aspect;
    }
}

void FaceMeshFilter::centerShift(
    const float* src478x2, const float* pose8, const float* poseVel8,
    float horizonMs, float aspect, float outShift[2]) {
    outShift[0] = 0.0f;
    outShift[1] = 0.0f;
    const float det = detOf(pose8);
    if (std::fabs(det) < 1e-10f || !(aspect > 0.0f)) return;
    const float omega = angularVelOf(pose8, poseVel8);
    float rx = 0.0f, ry = 0.0f;
    for (int idx : kCenterPoints) {
        rx += src478x2[idx * 2] - pose8[0];
        ry += src478x2[idx * 2 + 1] / aspect - pose8[1];
    }
    rx *= 0.25f;
    ry *= 0.25f;
    const float dTheta = horizonMs * omega;
    const float c = std::cos(dTheta);
    const float sn = std::sin(dTheta);
    outShift[0] = horizonMs * poseVel8[0] + (c - 1.0f) * rx - sn * ry;
    outShift[1] = horizonMs * poseVel8[1] + sn * rx + (c - 1.0f) * ry;
}

void FaceMeshFilter::reset(bool keepBaseline) {
    started_ = false;
    lastTs_ = kTimeInvalid;
    vp_ = 0.0f;
    dq_.fill(0.0f);
    dLogScale_ = 0.0f;
    noisePower_ = -1.0f;
    noiseMin_ = 0.0f;
    jumped_ = false;
    poseSpeed_ = 0.0f;
    poseDead_ = std::numeric_limits<float>::max();
    noiseFloor_ = 0.0f;
    motionSpeed_ = 0.0f;
    poseScale_ = 0.0f;
    if (!keepBaseline) gateBase_.fill(-1.0f);
    gateValue_.fill(1.0f);
}

int FaceMeshFilter::filter(int64_t ts, const float* raw478x2, float aspect, float* out478x2, float* poseOut8) {
    for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
        mX_[i] = raw478x2[i * 2];
        mY_[i] = raw478x2[i * 2 + 1] / aspect;
    }
    fitPose(mX_.data(), mY_.data(), qRaw_.data());
    jumped_ = false;
    bool restart = (!started_ || ts <= lastTs_ || ts - lastTs_ > kMaxGapMs);
    if (!restart) {
        const float s = scaleOf(q_.data());
        const float sRaw = scaleOf(qRaw_.data());
        if (detOf(qRaw_.data()) * detOf(q_.data()) <= 0.0f ||
            std::hypot(qRaw_[0] - q_[0], qRaw_[1] - q_[1]) > kJumpT * s ||
            !(sRaw > 0.0f) || std::fabs(std::log(sRaw / s)) > kJumpLogS) {
            restart = true;
            jumped_ = true;
        }
    }
    if (std::fabs(detOf(qRaw_.data())) < 1e-10f) {
        started_ = false;
        std::memcpy(out478x2, raw478x2, sizeof(float) * FaceMeshFrame::kCoordCount);
        std::memcpy(poseOut8, qRaw_.data(), sizeof(float) * kPoseSize);
        return kNewRun;
    }
    const float teMs = restart ? 0.0f : static_cast<float>(ts - lastTs_);
    if (restart) startRun();
    else step(teMs);
    lastTs_ = ts;

    const auto& sd = staticData();
    const float tx = q_[0], ty = q_[1], a11 = q_[2], a12 = q_[3], a21 = q_[4], a22 = q_[5], w1 = q_[6], w2 = q_[7];
    for (int i = 0; i < kFace; ++i) {
        const float lx = sd.lx[i] + rX_[i];
        const float ly = sd.ly[i] + rY_[i];
        const float z = sd.z[i];
        outX_[i] = tx + a11 * lx + a12 * ly + z * w1;
        outY_[i] = ty + a21 * lx + a22 * ly + z * w2;
    }
    updateIris(0, kIrisRight, 33, 133, teMs, restart);
    updateIris(1, kIrisLeft, 362, 263, teMs, restart);
    updateGate(0, kRightLids, 6, 33, 133, teMs, restart);
    updateGate(1, kLeftLids, 6, 362, 263, teMs, restart);
    for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
        out478x2[i * 2] = outX_[i];
        out478x2[i * 2 + 1] = outY_[i] * aspect;
    }
    std::memcpy(poseOut8, q_.data(), sizeof(float) * kPoseSize);
    poseScale_ = scaleOf(q_.data());
    return restart ? kNewRun : kContinue;
}

void FaceMeshFilter::startRun() {
    started_ = true;
    q_ = qRaw_;
    qRawPrev_ = qRaw_;
    dq_.fill(0.0f);
    const float s3d = std::sqrt(std::max(viewScale2(qRaw_.data()), 1e-10f));
    logScale_ = std::log(s3d);
    logScaleRawPrev_ = logScale_;
    dLogScale_ = 0.0f;
    const float detSign = (detOf(qRaw_.data()) >= 0.0f) ? 1.0f : -1.0f;
    const float m0 = (qRaw_[2] / s3d) * detSign;
    const float m1 = qRaw_[3] / s3d;
    const float m2 = (qRaw_[4] / s3d) * detSign;
    const float m3 = qRaw_[5] / s3d;
    const float rn = std::hypot(m0 + m3, m2 - m1);
    cRot_ = (rn > 1e-6f) ? ((m0 + m3) / rn) : 1.0f;
    sRot_ = (rn > 1e-6f) ? ((m2 - m1) / rn) : 0.0f;
    p00_ = cRot_ * m0 + sRot_ * m2;
    p01_ = cRot_ * m1 + sRot_ * m3;
    p11_ = -sRot_ * m1 + cRot_ * m3;
    hw1_ = qRaw_[6] / s3d;
    hw2_ = qRaw_[7] / s3d;
    hw1RawPrev_ = hw1_;
    hw2RawPrev_ = hw2_;
    dHw1_ = 0.0f;
    dHw2_ = 0.0f;
    residual(qRaw_.data(), rRawX_.data(), rRawY_.data());
    rX_ = rRawX_;
    rY_ = rRawY_;
    rPrevX_ = rRawX_;
    rPrevY_ = rRawY_;
    drX_.fill(0.0f);
    drY_.fill(0.0f);
    vp_ = 0.0f;
    noisePower_ = -1.0f;
    noiseMin_ = 0.0f;
    poseSpeed_ = 0.0f;
    poseDead_ = std::numeric_limits<float>::max();
    noiseFloor_ = 0.0f;
    motionSpeed_ = 0.0f;
}

void FaceMeshFilter::step(float teMs) {
    const auto& sd = staticData();
    const float teSec = teMs / 1000.0f;
    const float comp = std::min(1.0f, std::sqrt(kTeRefMs / teMs));
    const float s = std::max(scaleOf(q_.data()), 1e-4f);
    const float s3dPrev = std::exp(logScale_);
    const float s3dRaw = std::sqrt(std::max(viewScale2(qRaw_.data()), 1e-10f));
    const bool illPosed = (std::fabs(detOf(qRaw_.data())) < kDetMin * s * s);
    const float detSign = (detOf(qRaw_.data()) >= 0.0f) ? 1.0f : -1.0f;

    const float m0Raw = (qRaw_[2] / s3dRaw) * detSign;
    const float m1Raw = qRaw_[3] / s3dRaw;
    const float m2Raw = (qRaw_[4] / s3dRaw) * detSign;
    const float m3Raw = qRaw_[5] / s3dRaw;
    const float rnRaw = std::hypot(m0Raw + m3Raw, m2Raw - m1Raw);
    const float cRotRaw = (rnRaw > 1e-6f) ? ((m0Raw + m3Raw) / rnRaw) : cRot_;
    const float sRotRaw = (rnRaw > 1e-6f) ? ((m2Raw - m1Raw) / rnRaw) : sRot_;
    const float p00Raw = cRotRaw * m0Raw + sRotRaw * m2Raw;
    const float p01Raw = cRotRaw * m1Raw + sRotRaw * m3Raw;
    const float p11Raw = -sRotRaw * m1Raw + cRotRaw * m3Raw;
    const float hw1Raw = qRaw_[6] / s3dRaw;
    const float hw2Raw = qRaw_[7] / s3dRaw;

    const float adPoseUp = detail::oneEuroAlpha(kPoseDcut, teMs);
    const float adPoseDown = detail::oneEuroAlpha(kPoseDcutDown, teMs);
    for (int k = 0; k < kPoseSize; ++k) {
        const float instVq = (qRaw_[k] - qRawPrev_[k]) / teSec;
        // Signed velocity decays fast when motion stops (|instVq| < 0.55*|dq_|), while keeping symmetric adPoseUp on alternating noise!
        const float adk = (std::fabs(instVq) < 0.55f * std::fabs(dq_[k])) ? adPoseDown : adPoseUp;
        dq_[k] += adk * (instVq - dq_[k]);
    }

    float rawRigidAcc = 0.0f;
    float cohAcc = 0.0f;
    float lag2dAcc = 0.0f;
    // Scale- & 3D-noise-decoupled 2D rigid pose target using current smoothed (P, hw, s3dPrev) + raw (tx, ty, cRotRaw, sRotRaw)
    const float r2dA11 = (cRotRaw * p00_ - sRotRaw * p01_) * detSign * s3dPrev;
    const float r2dA12 = (cRotRaw * p01_ - sRotRaw * p11_) * s3dPrev;
    const float r2dA21 = (sRotRaw * p00_ + cRotRaw * p01_) * detSign * s3dPrev;
    const float r2dA22 = (sRotRaw * p01_ + cRotRaw * p11_) * s3dPrev;
    for (size_t j = 0; j < sd.basis.size(); ++j) {
        const int b = sd.basis[j];
        const float mx = (qRaw_[0] - qRawPrev_[0]) +
                         (qRaw_[2] - qRawPrev_[2]) * sd.lx[b] +
                         (qRaw_[3] - qRawPrev_[3]) * sd.ly[b];
        const float my = (qRaw_[1] - qRawPrev_[1]) +
                         (qRaw_[4] - qRawPrev_[4]) * sd.lx[b] +
                         (qRaw_[5] - qRawPrev_[5]) * sd.ly[b];
        rawRigidAcc += sd.basisW[j] * (mx * mx + my * my);

        const float vx = dq_[0] + dq_[2] * sd.lx[b] + dq_[3] * sd.ly[b];
        const float vy = dq_[1] + dq_[4] * sd.lx[b] + dq_[5] * sd.ly[b];
        cohAcc += sd.basisW[j] * (vx * vx + vy * vy);

        const float lx = (qRaw_[0] - q_[0]) + (r2dA11 - q_[2]) * sd.lx[b] + (r2dA12 - q_[3]) * sd.ly[b];
        const float ly = (qRaw_[1] - q_[1]) + (r2dA21 - q_[4]) * sd.lx[b] + (r2dA22 - q_[5]) * sd.ly[b];
        lag2dAcc += sd.basisW[j] * (lx * lx + ly * ly);
    }
    const float spRaw = std::sqrt(rawRigidAcc * 0.5f) / s / teSec;
    vp_ = std::sqrt(cohAcc * 0.5f) / s;

    if (!illPosed) {
        residual(qRaw_.data(), rRawX_.data(), rRawY_.data());
        float sum = 0.0f;
        for (int i = 0; i < kFace; ++i) {
            const float dx = rRawX_[i] - rPrevX_[i];
            const float dy = rRawY_[i] - rPrevY_[i];
            sum += dx * dx + dy * dy;
        }
        const float rmsAll = std::sqrt(sum / static_cast<float>(kFace)) / teSec;
        const float power = rmsAll * rmsAll;
        if (noisePower_ < 0.0f) {
            noisePower_ = power;
            noiseMin_ = power;
        }
        noisePower_ += (1.0f - std::exp(-teMs / kNfTauMs)) * (power - noisePower_);
        noiseMin_ = std::max(std::min(noiseMin_ * (1.0f + kNfRise * teSec), noisePower_), 1e-8f);
    }
    const float floorVal = (noisePower_ < 0.0f) ? 0.0f : std::sqrt(noiseMin_);
    noiseFloor_ = floorVal;
    const float dead = kKPose * kCPose * floorVal;
    const float basisLag = std::sqrt(lag2dAcc * 0.5f) / s;
    const float effVp = std::max(vp_, spRaw * detail::smoothstep(0.015f, 0.060f, basisLag));
    // Dimensionally consistent lag threshold: dead (1/s) * teSec (s) = faceWidth displacement
    const float lagCatchup = 45.0f * std::max(0.0f, basisLag - std::max(0.008f, 2.0f * dead * teSec));
    const float aPose = detail::oneEuroAlpha(
        kPoseFcMin * comp + kPoseBeta * (std::max(0.0f, effVp - dead) + lagCatchup), teMs);

    if (!illPosed) {
        // 1. Isotropic 3D scale (invariant to 2D roll + 3D yaw/pitch)
        const float logSRaw = std::log(s3dRaw);
        const float instDLogS = (logSRaw - logScaleRawPrev_) / teSec;
        logScaleRawPrev_ = logSRaw;
        dLogScale_ += detail::oneEuroAlpha(kScaleDcut, teMs) * (instDLogS - dLogScale_);
        const float scaleDead = std::max(0.16f, 1.2f * dead);
        const float aScale = detail::oneEuroAlpha(
            kScaleFcMin * comp + kScaleBeta * std::max(0.0f, std::fabs(dLogScale_) - scaleDead), teMs);
        logScale_ += aScale * (logSRaw - logScale_);
        const float sFilt = std::exp(logScale_);

        // 2. 3D out-of-plane perspective & foreshortening (smooth monotonic One-Euro response on the
        //    conditioned 117-pt rigid craniofacial basis: zero hold-and-snap twitch when still, and
        //    activity-coupled tracking during genuine 3D head turns or camera shake!)
        const float ad3d = detail::oneEuroAlpha(kPose3dDcut, teMs);
        dHw1_ += ad3d * ((hw1Raw - hw1RawPrev_) / teSec - dHw1_);
        dHw2_ += ad3d * ((hw2Raw - hw2RawPrev_) / teSec - dHw2_);
        hw1RawPrev_ = hw1Raw;
        hw2RawPrev_ = hw2Raw;
        const float yawSpeed = std::hypot(dHw1_, dHw2_);
        const float yawLag = std::hypot(hw1Raw - hw1_, hw2Raw - hw2_);
        const float yawDead = std::max(0.30f, 1.8f * dead);
        const float pose3dActBoost = 25.0f * std::max(0.0f, effVp - 1.5f * dead);
        const float a3d = detail::oneEuroAlpha(
            kPose3dFcMin * comp + kPose3dBeta * std::max(0.0f, yawSpeed - yawDead) +
            160.0f * std::max(0.0f, yawLag - 0.085f) + pose3dActBoost, teMs);
        p00_ += a3d * (p00Raw - p00_);
        p01_ += a3d * (p01Raw - p01_);
        p11_ += a3d * (p11Raw - p11_);
        hw1_ += a3d * (hw1Raw - hw1_);
        hw2_ += a3d * (hw2Raw - hw2_);

        // 3. Unified fast 2D SE(2) translation + camera roll update (exact 0.0px rotational shear!)
        q_[0] += aPose * (qRaw_[0] - q_[0]);
        q_[1] += aPose * (qRaw_[1] - q_[1]);
        cRot_ += aPose * (cRotRaw - cRot_);
        sRot_ += aPose * (sRotRaw - sRot_);
        const float rotNorm = std::hypot(cRot_, sRot_);
        if (rotNorm > 1e-6f) {
            cRot_ /= rotNorm;
            sRot_ /= rotNorm;
        }

        // 4. Recompose 2x3 Stiefel matrix R(theta) * P + [hw1; hw2], project to unit viewScale2, and scale by sFilt
        float u0 = (cRot_ * p00_ - sRot_ * p01_) * detSign;
        float u1 = cRot_ * p01_ - sRot_ * p11_;
        float u2 = (sRot_ * p00_ + cRot_ * p01_) * detSign;
        float u3 = sRot_ * p01_ + cRot_ * p11_;
        float u4 = hw1_;
        float u5 = hw2_;
        const float cx = u1 * u5 - u4 * u3;
        const float cy = u4 * u2 - u0 * u5;
        const float cz = u0 * u3 - u1 * u2;
        const float normM = std::sqrt(std::max(std::sqrt(cx * cx + cy * cy + cz * cz), 1e-8f));
        const float kScale = sFilt / normM;
        q_[2] = u0 * kScale;
        q_[3] = u1 * kScale;
        q_[4] = u2 * kScale;
        q_[5] = u3 * kScale;
        q_[6] = u4 * kScale;
        q_[7] = u5 * kScale;
    } else {
        for (int k = 0; k < kPoseSize; ++k) q_[k] += aPose * (qRaw_[k] - q_[k]);
        const float s3dCur = std::sqrt(std::max(viewScale2(q_.data()), 1e-10f));
        logScale_ = std::log(s3dCur);
        logScaleRawPrev_ = logScale_;
    }
    qRawPrev_ = qRaw_;
    poseSpeed_ = effVp;
    poseDead_ = dead;

    if (!illPosed) {
        const float ad = detail::oneEuroAlpha(kResDcut, teMs);
        for (int i = 0; i < kFace; ++i) {
            drX_[i] += ad * ((rRawX_[i] - rPrevX_[i]) / teSec - drX_[i]);
            drY_[i] += ad * ((rRawY_[i] - rPrevY_[i]) / teSec - drY_[i]);
        }
        regionRms_.fill(0.0f);
        for (int i = 0; i < kFace; ++i) {
            const float p = drX_[i] * drX_[i] + drY_[i] * drY_[i];
            for (int g = 0; g < kRegions; ++g) regionRms_[g] += sd.reg[i * kRegions + g] * p;
        }
        float lidPower = 0.0f;
        for (int idx : kLidRing) lidPower += drX_[idx] * drX_[idx] + drY_[idx] * drY_[idx];
        float lipPower = 0.0f;
        for (int idx : kLipRing) lipPower += drX_[idx] * drX_[idx] + drY_[idx] * drY_[idx];

        const float resDead = kKRes * kCRes * std::pow(teMs / kTeRefMs, kResTeExp) * floorVal;
        for (int g = 0; g < kRegions; ++g) {
            regionRms_[g] = std::sqrt(regionRms_[g] / std::max(sd.regSum[g], 1e-4f));
            if (g == kEyeRegion) regionRms_[g] = std::max(regionRms_[g], std::sqrt(lidPower / 32.0f));
            if (g == kLowRegion) regionRms_[g] = std::max(regionRms_[g], std::sqrt(lipPower / 40.0f));
            const float deadG = (g == kContourRegion) ? (1.6f * resDead) : resDead;
            regionAlpha_[g] = detail::oneEuroAlpha(
                kResFcMin[g] * comp + kResBeta[g] * std::max(0.0f, regionRms_[g] - deadG), teMs);
        }
        for (int i = 0; i < kFace; ++i) {
            float ai = 0.0f;
            for (int g = 0; g < kRegions; ++g) ai += sd.reg[i * kRegions + g] * regionAlpha_[g];
            rX_[i] += ai * (rRawX_[i] - rX_[i]);
            rY_[i] += ai * (rRawY_[i] - rY_[i]);
        }
        rPrevX_ = rRawX_;
        rPrevY_ = rRawY_;
    }
    // Whole-face rigid motion speed only (blinking/speaking residuals must NEVER trigger shake/steadyFace gates)
    motionSpeed_ = s * std::max(0.0f, effVp - dead);
}

void FaceMeshFilter::residual(const float* pose8, float* outRx, float* outRy) const {
    const auto& sd = staticData();
    const float det = detOf(pose8);
    const float i11 = pose8[5] / det;
    const float i12 = -pose8[3] / det;
    const float i21 = -pose8[4] / det;
    const float i22 = pose8[2] / det;
    for (int i = 0; i < kFace; ++i) {
        const float dx = mX_[i] - pose8[0] - sd.z[i] * pose8[6];
        const float dy = mY_[i] - pose8[1] - sd.z[i] * pose8[7];
        outRx[i] = i11 * dx + i12 * dy - sd.lx[i];
        outRy[i] = i21 * dx + i22 * dy - sd.ly[i];
    }
}

void FaceMeshFilter::updateIris(int eye, int base, int a, int b, float teMs, bool restart) {
    const auto& sd = staticData();
    float cx = 0.0f, cy = 0.0f;
    for (int k = 0; k < 5; ++k) {
        cx += mX_[base + k];
        cy += mY_[base + k];
    }
    cx *= 0.2f;
    cy *= 0.2f;
    float radius = 0.0f;
    for (int k = 1; k < 5; ++k) radius += std::hypot(mX_[base + k] - cx, mY_[base + k] - cy);
    radius *= 0.25f;

    const float ox = 0.5f * (outX_[a] + outX_[b]);
    const float oy = 0.5f * (outY_[a] + outY_[b]);
    const float ew = std::max(std::hypot(outX_[b] - outX_[a], outY_[b] - outY_[a]), 1e-4f);
    const float ex = (outX_[b] - outX_[a]) / ew;
    const float ey = (outY_[b] - outY_[a]) / ew;
    const float qx = ((cx - ox) * ex + (cy - oy) * ey) / ew;
    const float qy = (-(cx - ox) * ey + (cy - oy) * ex) / ew;
    const float dlx = sd.lx[b] - sd.lx[a];
    const float dly = sd.ly[b] - sd.ly[a];
    const float dz = sd.z[b] - sd.z[a];
    const float rigidEw = std::max(
        std::hypot(q_[2] * dlx + q_[3] * dly + q_[6] * dz, q_[4] * dlx + q_[5] * dly + q_[7] * dz), 1e-4f);
    const float kRaw = radius / rigidEw;
    if (restart || !(teMs > 0.0f)) {
        irisQx_[eye] = qx; irisQy_[eye] = qy;
        irisRawX_[eye] = qx; irisRawY_[eye] = qy;
        irisDx_[eye] = 0.0f; irisDy_[eye] = 0.0f;
        irisPower_[eye] = -1.0f; irisMin_[eye] = 0.0f;
        irisK_[eye] = detail::clampf(kRaw, kIrisKMin, kIrisKMax);
    } else {
        const float teSec = teMs / 1000.0f;
        const float vx = (qx - irisRawX_[eye]) / teSec;
        const float vy = (qy - irisRawY_[eye]) / teSec;
        irisRawX_[eye] = qx; irisRawY_[eye] = qy;
        const float power = vx * vx + vy * vy;
        if (irisPower_[eye] < 0.0f) {
            irisPower_[eye] = power;
            irisMin_[eye] = power;
        }
        irisPower_[eye] += (1.0f - std::exp(-teMs / kNfTauMs)) * (power - irisPower_[eye]);
        irisMin_[eye] = std::max(std::min(irisMin_[eye] * (1.0f + kNfRise * teSec), irisPower_[eye]), 1e-8f);
        const float ad = detail::oneEuroAlpha(kIrisDcut, teMs);
        irisDx_[eye] += ad * (vx - irisDx_[eye]);
        irisDy_[eye] += ad * (vy - irisDy_[eye]);
        const float dead = kIrisKDead * std::sqrt(irisMin_[eye]) * std::sqrt(ad / (2.0f - ad));
        const float comp = std::min(1.0f, std::sqrt(kTeRefMs / teMs));
        const float fc = kIrisFc * comp + kIrisBeta * std::max(0.0f, std::hypot(irisDx_[eye], irisDy_[eye]) - dead);
        const float aq = detail::oneEuroAlpha(fc, teMs);
        irisQx_[eye] += aq * (qx - irisQx_[eye]);
        irisQy_[eye] += aq * (qy - irisQy_[eye]);
        irisK_[eye] += detail::oneEuroAlpha(kIrisKFc, teMs) * (kRaw - irisK_[eye]);
    }
    const float k = detail::clampf(irisK_[eye], kIrisKMin, kIrisKMax);
    const float px = ox + ew * (irisQx_[eye] * ex - irisQy_[eye] * ey);
    const float py = oy + ew * (irisQx_[eye] * ey + irisQy_[eye] * ex);
    const float r = k * rigidEw;
    outX_[base] = px;     outY_[base] = py;
    outX_[base + 1] = px + r; outY_[base + 1] = py;
    outX_[base + 2] = px; outY_[base + 2] = py + r;
    outX_[base + 3] = px - r; outY_[base + 3] = py;
    outX_[base + 4] = px; outY_[base + 4] = py - r;
}

void FaceMeshFilter::updateGate(int eye, const int* lids, int lidCount, int a, int b, float teMs, bool restart) {
    const float ew = std::max(std::hypot(outX_[b] - outX_[a], outY_[b] - outY_[a]), 1e-4f);
    float gap = 0.0f;
    for (int k = 0; k < lidCount; k += 2) {
        gap += std::hypot(outX_[lids[k]] - outX_[lids[k + 1]], outY_[lids[k]] - outY_[lids[k + 1]]);
    }
    const float open = gap / static_cast<float>(lidCount / 2) / ew;
    gateOpen_[eye] = open;
    if (gateBase_[eye] < 0.0f) {
        gateBase_[eye] = detail::clampf(open, kBaseMin, kBaseMax);
    } else if (!restart && teMs > 0.0f) {
        const float tau = (open > gateBase_[eye]) ? kBaseUpS : kBaseDownS;
        gateBase_[eye] = detail::clampf(
            gateBase_[eye] + (open - gateBase_[eye]) * (1.0f - std::exp(-teMs / 1000.0f / tau)),
            kBaseMin, kBaseMax);
    }
    const float n = open / gateBase_[eye];
    if (restart || !(teMs > 0.0f)) {
        gateX_[eye] = n;
        gatePrev_[eye] = n;
        gateDx_[eye] = 0.0f;
        gateClosed_[eye] = (n < kGateClose);
        gateValue_[eye] = gateClosed_[eye] ? 0.0f : 1.0f;
        return;
    }
    gateDx_[eye] += detail::oneEuroAlpha(kGateDcut, teMs) * ((n - gatePrev_[eye]) / (teMs / 1000.0f) - gateDx_[eye]);
    gatePrev_[eye] = n;
    gateX_[eye] += detail::oneEuroAlpha(kGateFc + kGateBeta * std::fabs(gateDx_[eye]), teMs) * (n - gateX_[eye]);
    if (!gateClosed_[eye] && gateX_[eye] < kGateClose) gateClosed_[eye] = true;
    else if (gateClosed_[eye] && gateX_[eye] > kGateOpen) gateClosed_[eye] = false;
    gateValue_[eye] = gateClosed_[eye] ? 0.0f : std::min(1.0f, gateValue_[eye] + teMs / kGateRampMs);
}

}  // namespace livebeauty::math
