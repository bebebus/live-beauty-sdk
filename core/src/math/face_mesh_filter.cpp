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
constexpr float kResFcMin[3] = {2.4f, 3.2f, 3.4f};
constexpr float kResBeta[3] = {70.0f, 105.0f, 110.0f};

struct FilterStaticData {
    float canonicalFaceWidth = 1.0f;
    std::array<float, FaceMeshFrame::kFaceCount> lx{};
    std::array<float, FaceMeshFrame::kFaceCount> ly{};
    std::array<float, FaceMeshFrame::kFaceCount> z{};
    std::array<float, FaceMeshFrame::kLandmarkCount> zExt{};
    std::vector<int> basis;
    std::vector<float> basisW;
    std::vector<float> kMat;  // 8 x (2*n) row-major
    std::array<float, FaceMeshFrame::kFaceCount * 3> reg{};
    std::array<float, 3> regSum{};

    FilterStaticData() {
        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        canonicalFaceWidth = std::hypot(
            xyz[454 * 3] - xyz[234 * 3],
            std::hypot(xyz[454 * 3 + 1] - xyz[234 * 3 + 1], xyz[454 * 3 + 2] - xyz[234 * 3 + 2]));

        std::vector<double> weights;
        for (int j = 0; j < FaceMeshTopology::kProcrustesCount; ++j) {
            const int idx = FaceMeshTopology::kProcrustesIndices[j];
            if (idx == 136 || idx == 365) continue;
            basis.push_back(idx);
            weights.push_back(static_cast<double>(FaceMeshTopology::kProcrustesWeights[j]));
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

        const auto& uv = FaceMeshTopology::kCanonicalUv;
        auto eyeMember = [&](int i, int a, int b) {
            const float cu = 0.5f * (uv[a * 2] + uv[b * 2]);
            const float ew = std::fabs(uv[b * 2] - uv[a * 2]);
            const float cv = 0.5f * (uv[a * 2 + 1] + uv[b * 2 + 1]) - 0.14f * ew;
            const float dd = std::hypot((uv[i * 2] - cu) / ew, (uv[i * 2 + 1] - cv) / (0.95f * ew));
            return 1.0f - detail::smoothstep(0.65f, 1.05f, dd);
        };
        const float vSub = uv[2 * 2 + 1];
        regSum.fill(0.0f);
        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            const float eye = std::max(eyeMember(i, 33, 133), eyeMember(i, 362, 263));
            const float low = std::min(detail::smoothstep(vSub - 0.01f, vSub + 0.06f, uv[i * 2 + 1]), 1.0f - eye);
            reg[i * 3 + 0] = 1.0f - eye - low;
            reg[i * 3 + 1] = eye;
            reg[i * 3 + 2] = low;
            for (int g = 0; g < 3; ++g) regSum[g] += reg[i * 3 + g];
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
    const auto& sd = staticData();
    const float i11 = pose8[5] / det;
    const float i12 = -pose8[3] / det;
    const float i21 = -pose8[4] / det;
    const float i22 = pose8[2] / det;
    const float h = horizonMs;
    for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
        const float z = sd.zExt[i];
        const float dx = src478x2[i * 2] - pose8[0] - z * pose8[6];
        const float dy = src478x2[i * 2 + 1] / aspect - pose8[1] - z * pose8[7];
        const float ux = i11 * dx + i12 * dy;
        const float uy = i21 * dx + i22 * dy;
        const float mx = h * (poseVel8[0] + poseVel8[2] * ux + poseVel8[3] * uy + z * poseVel8[6]);
        const float my = h * (poseVel8[1] + poseVel8[4] * ux + poseVel8[5] * uy + z * poseVel8[7]);
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
    const auto& sd = staticData();
    const float i11 = pose8[5] / det;
    const float i12 = -pose8[3] / det;
    const float i21 = -pose8[4] / det;
    const float i22 = pose8[2] / det;
    float ux = 0.0f, uy = 0.0f, z = 0.0f;
    for (int idx : kCenterPoints) {
        const float zi = sd.zExt[idx];
        const float dx = src478x2[idx * 2] - pose8[0] - zi * pose8[6];
        const float dy = src478x2[idx * 2 + 1] / aspect - pose8[1] - zi * pose8[7];
        ux += i11 * dx + i12 * dy;
        uy += i21 * dx + i22 * dy;
        z += zi;
    }
    ux *= 0.25f;
    uy *= 0.25f;
    z *= 0.25f;
    outShift[0] = horizonMs * (poseVel8[0] + poseVel8[2] * ux + poseVel8[3] * uy + z * poseVel8[6]);
    outShift[1] = horizonMs * (poseVel8[1] + poseVel8[4] * ux + poseVel8[5] * uy + z * poseVel8[7]);
}

void FaceMeshFilter::reset(bool keepBaseline) {
    started_ = false;
    lastTs_ = kTimeInvalid;
    vp_ = 0.0f;
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
        if (std::hypot(qRaw_[0] - q_[0], qRaw_[1] - q_[1]) > kJumpT * s ||
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
    const float s = scaleOf(q_.data());

    float acc = 0.0f;
    float lagAcc = 0.0f;
    for (size_t j = 0; j < sd.basis.size(); ++j) {
        const int b = sd.basis[j];
        const float mx = (qRaw_[0] - qRawPrev_[0]) + (qRaw_[2] - qRawPrev_[2]) * sd.lx[b] +
                         (qRaw_[3] - qRawPrev_[3]) * sd.ly[b] + (qRaw_[6] - qRawPrev_[6]) * sd.z[b];
        const float my = (qRaw_[1] - qRawPrev_[1]) + (qRaw_[4] - qRawPrev_[4]) * sd.lx[b] +
                         (qRaw_[5] - qRawPrev_[5]) * sd.ly[b] + (qRaw_[7] - qRawPrev_[7]) * sd.z[b];
        acc += sd.basisW[j] * (mx * mx + my * my);

        const float lx = (qRaw_[0] - q_[0]) + (qRaw_[2] - q_[2]) * sd.lx[b] +
                         (qRaw_[3] - q_[3]) * sd.ly[b] + (qRaw_[6] - q_[6]) * sd.z[b];
        const float ly = (qRaw_[1] - q_[1]) + (qRaw_[4] - q_[4]) * sd.lx[b] +
                         (qRaw_[5] - q_[5]) * sd.ly[b] + (qRaw_[7] - q_[7]) * sd.z[b];
        lagAcc += sd.basisW[j] * (lx * lx + ly * ly);
    }
    const float spRaw = std::sqrt(acc * 0.5f) / s / teSec;
    const float poseDcut = (spRaw < vp_) ? kPoseDcutDown : kPoseDcut;
    vp_ += detail::oneEuroAlpha(poseDcut, teMs) * (spRaw - vp_);

    const bool illPosed = (std::fabs(detOf(qRaw_.data())) < kDetMin * s * s);
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
    const float basisLag = std::sqrt(lagAcc * 0.5f) / std::max(s, 1e-4f);
    const float effVp = std::max(vp_, spRaw * detail::smoothstep(0.025f, 0.095f, basisLag));
    const float lagCatchup = 18.0f * std::max(0.0f, basisLag - std::max(0.012f, 2.2f * dead));
    const float a = detail::oneEuroAlpha(
        kPoseFcMin * comp + kPoseBeta * (std::max(0.0f, effVp - dead) + lagCatchup), teMs);
    for (int k = 0; k < kPoseSize; ++k) q_[k] += a * (qRaw_[k] - q_[k]);
    qRawPrev_ = qRaw_;
    poseSpeed_ = effVp;
    poseDead_ = dead;

    float resExcess = 0.0f;
    if (!illPosed) {
        const float adUp = detail::oneEuroAlpha(kResDcut, teMs);
        const float adDown = detail::oneEuroAlpha(kResDcutDown, teMs);
        std::array<float, kRegions> regionLagSq{};
        regionLagSq.fill(0.0f);
        for (int i = 0; i < kFace; ++i) {
            const float instVx = (rRawX_[i] - rPrevX_[i]) / teSec;
            const float instVy = (rRawY_[i] - rPrevY_[i]) / teSec;
            const float instMag2 = instVx * instVx + instVy * instVy;
            const float curMag2 = drX_[i] * drX_[i] + drY_[i] * drY_[i];
            const float ad = (instMag2 < curMag2) ? adDown : adUp;
            drX_[i] += ad * (instVx - drX_[i]);
            drY_[i] += ad * (instVy - drY_[i]);
            const float lx = rRawX_[i] - rX_[i];
            const float ly = rRawY_[i] - rY_[i];
            const float lp = lx * lx + ly * ly;
            for (int g = 0; g < kRegions; ++g) regionLagSq[g] += sd.reg[i * kRegions + g] * lp;
        }
        regionRms_.fill(0.0f);
        for (int i = 0; i < kFace; ++i) {
            const float p = drX_[i] * drX_[i] + drY_[i] * drY_[i];
            for (int g = 0; g < kRegions; ++g) regionRms_[g] += sd.reg[i * kRegions + g] * p;
        }
        float lidPower = 0.0f, lidLagSq = 0.0f;
        for (int idx : kLidRing) {
            lidPower += drX_[idx] * drX_[idx] + drY_[idx] * drY_[idx];
            const float lx = rRawX_[idx] - rX_[idx], ly = rRawY_[idx] - rY_[idx];
            lidLagSq += lx * lx + ly * ly;
        }
        float lipPower = 0.0f, lipLagSq = 0.0f;
        for (int idx : kLipRing) {
            lipPower += drX_[idx] * drX_[idx] + drY_[idx] * drY_[idx];
            const float lx = rRawX_[idx] - rX_[idx], ly = rRawY_[idx] - rY_[idx];
            lipLagSq += lx * lx + ly * ly;
        }

        const float resDead = kKRes * kCRes * std::pow(teMs / kTeRefMs, kResTeExp) * floorVal;
        for (int g = 0; g < kRegions; ++g) {
            regionRms_[g] = std::sqrt(regionRms_[g] / sd.regSum[g]);
            float regLag = std::sqrt(regionLagSq[g] / sd.regSum[g]);
            if (g == kEyeRegion) {
                regionRms_[g] = std::max(regionRms_[g], std::sqrt(lidPower / 32.0f));
                regLag = std::max(regLag, std::sqrt(lidLagSq / 32.0f));
            }
            if (g == kLowRegion) {
                regionRms_[g] = std::max(regionRms_[g], std::sqrt(lipPower / 40.0f));
                regLag = std::max(regLag, std::sqrt(lipLagSq / 40.0f));
            }
            const float excess = std::max(0.0f, regionRms_[g] - resDead);
            resExcess = std::max(resExcess, excess);
            const float resLagBoost = 22.0f * std::max(0.0f, regLag - std::max(0.012f, 2.2f * resDead));
            regionAlpha_[g] = detail::oneEuroAlpha(
                kResFcMin[g] * comp + kResBeta[g] * (excess + resLagBoost), teMs);
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
    motionSpeed_ = s * std::max(std::max(0.0f, effVp - dead), resExcess);
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
