#include "livebeauty/math/face_mesh_deformer.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace livebeauty::math {

namespace {

inline float gaussian(float x, float center, float sigma) {
    const float z = (x - center) / sigma;
    return std::exp(-0.5f * z * z);
}

inline float radialFalloff(float px, float py, float cx, float cy, float radius) {
    if (radius <= 1e-4f) return 0.0f;
    const float r = std::hypot(px - cx, py - cy) / radius;
    if (r >= 1.0f) return 0.0f;
    const float t = 1.0f - r * r;
    return t * t;
}

struct DeformerStaticTables {
    std::array<int, FaceMeshTopology::kFaceVertexCount + 1> neighborOffsets{};
    std::vector<uint16_t> neighborIndices;
    std::array<int, FaceMeshTopology::kOvalVertexCount + 1> ovalNb2Offsets{};
    std::vector<uint16_t> ovalNb2Indices;
    std::vector<uint16_t> cheekAnatRight;
    std::vector<uint16_t> cheekAnatLeft;
    std::array<int, FaceMeshTopology::kFaceVertexCount + 1> vertexTriOffsets{};
    std::vector<uint16_t> vertexTriIndices;
    std::array<float, FaceMeshTopology::kCanonicalTriangleCount> triArea3d{};
    std::array<float, FaceMeshTopology::kBufferOffset> innerFeatureW{};

    DeformerStaticTables() {
        std::array<std::set<int>, FaceMeshTopology::kFaceVertexCount> adj;
        const auto& tris = FaceMeshTopology::kCanonicalIndices;
        for (size_t i = 0; i + 2 < tris.size(); i += 3) {
            const int a = tris[i], b = tris[i + 1], c = tris[i + 2];
            adj[a].insert(b); adj[a].insert(c);
            adj[b].insert(a); adj[b].insert(c);
            adj[c].insert(a); adj[c].insert(b);
        }
        int total = 0;
        for (int v = 0; v < FaceMeshTopology::kFaceVertexCount; ++v) {
            neighborOffsets[v] = total;
            total += static_cast<int>(adj[v].size());
        }
        neighborOffsets[FaceMeshTopology::kFaceVertexCount] = total;
        neighborIndices.reserve(total);
        for (int v = 0; v < FaceMeshTopology::kFaceVertexCount; ++v) {
            for (int nb : adj[v]) neighborIndices.push_back(static_cast<uint16_t>(nb));
        }

        const auto& oval = FaceMeshTopology::kFaceOval;
        int nb2Total = 0;
        for (int k = 0; k < FaceMeshTopology::kOvalVertexCount; ++k) {
            const int oi = oval[k];
            std::set<int> s;
            s.insert(oi);
            for (int n1 : adj[oi]) {
                s.insert(n1);
                for (int n2 : adj[n1]) s.insert(n2);
            }
            ovalNb2Offsets[k] = nb2Total;
            nb2Total += static_cast<int>(s.size());
            for (int v : s) ovalNb2Indices.push_back(static_cast<uint16_t>(v));
        }
        ovalNb2Offsets[FaceMeshTopology::kOvalVertexCount] = nb2Total;

        const auto& canonUv = FaceMeshTopology::kCanonicalUv;
        innerFeatureW.fill(0.0f);
        auto eyeProx = [&](int i, int a, int b) {
            const float cu = 0.5f * (canonUv[a * 2] + canonUv[b * 2]);
            const float ew = std::fabs(canonUv[b * 2] - canonUv[a * 2]);
            const float cv = 0.5f * (canonUv[a * 2 + 1] + canonUv[b * 2 + 1]) - 0.10f * ew;
            const float dd = std::hypot((canonUv[i * 2] - cu) / (ew * 0.95f),
                                        (canonUv[i * 2 + 1] - cv) / (ew * 0.90f));
            return 1.0f - detail::smoothstep(0.65f, 1.05f, dd);
        };
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const float u = canonUv[i * 2];
            const float v = canonUv[i * 2 + 1];
            const float nu = (u - 0.5f) * 2.0f;
            const float nv = (0.5f - v) * 2.0f;
            if (nv >= -0.72f && nv <= -0.04f) {
                if (nu < -0.35f) cheekAnatRight.push_back(static_cast<uint16_t>(i));
                else if (nu > 0.35f) cheekAnatLeft.push_back(static_cast<uint16_t>(i));
            }
            const float wEye = std::max(eyeProx(i, 33, 133), eyeProx(i, 362, 263));
            const float wNose = 1.0f - detail::smoothstep(
                0.65f, 1.05f, std::hypot((u - 0.50f) / 0.11f, (v - 0.51f) / 0.15f));
            const float wMouth = 1.0f - detail::smoothstep(
                0.65f, 1.05f, std::hypot((u - 0.50f) / 0.18f, (v - 0.695f) / 0.11f));
            innerFeatureW[i] = std::max({wEye, wNose, wMouth});
        }

        const auto& xyz = FaceMeshTopology::kCanonicalXyz;
        const float fw = FaceMeshFilter::canonicalFaceWidth();
        for (int t = 0; t < FaceMeshTopology::kCanonicalTriangleCount; ++t) {
            const int a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
            const float e1x = (xyz[b * 3] - xyz[a * 3]) / fw;
            const float e1y = (xyz[b * 3 + 1] - xyz[a * 3 + 1]) / fw;
            const float e1z = (xyz[b * 3 + 2] - xyz[a * 3 + 2]) / fw;
            const float e2x = (xyz[c * 3] - xyz[a * 3]) / fw;
            const float e2y = (xyz[c * 3 + 1] - xyz[a * 3 + 1]) / fw;
            const float e2z = (xyz[c * 3 + 2] - xyz[a * 3 + 2]) / fw;
            const float cx = e1y * e2z - e1z * e2y;
            const float cy = e1z * e2x - e1x * e2z;
            const float cz = e1x * e2y - e1y * e2x;
            triArea3d[t] = std::max(0.5f * std::sqrt(cx * cx + cy * cy + cz * cz), 1e-9f);
        }

        vertexTriOffsets.fill(0);
        for (uint16_t v : tris) vertexTriOffsets[v + 1]++;
        for (int v = 0; v < FaceMeshTopology::kFaceVertexCount; ++v) {
            vertexTriOffsets[v + 1] += vertexTriOffsets[v];
        }
        auto fill = vertexTriOffsets;
        vertexTriIndices.resize(tris.size());
        for (size_t i = 0; i < tris.size(); ++i) {
            vertexTriIndices[fill[tris[i]]++] = static_cast<uint16_t>(i / 3);
        }
    }
};

const DeformerStaticTables& tables() {
    static const DeformerStaticTables kTables;
    return kTables;
}

}  // namespace

void FaceMeshDeformer::resetTemporal() {
    temporalValid_ = false;
    dispTemporalValid_ = false;
    rawSignedYawPrev_ = 0.0f;
    yawVelEma_ = 0.0f;
    bisectLambdaState_ = 1.0f;
    prevEyeMod_ = 1.0f;
    prevFaceMod_ = 1.0f;
    lastNowMs_ = kTimeInvalid;
}

bool FaceMeshDeformer::deform(
    const float* landmarks478x2,
    float aspect,
    float axisX, float axisY,
    float faceWidth, float faceHeight,
    float leftWeight, float rightWeight, float yawWeight,
    float presence, float steadyFace, float steadyEye,
    const Params& params,
    int64_t nowMs) {
    if (!landmarks478x2 || !(aspect > 0.0f) || faceWidth < 0.03f || faceHeight < 0.03f) return false;

    const bool temporal = (nowMs != kTimeInvalid);
    float dtMs = 0.0f;
    bool temporalFresh = true;
    if (temporal) {
        if (temporalValid_ && lastNowMs_ != kTimeInvalid && nowMs >= lastNowMs_ &&
            nowMs - lastNowMs_ <= kTemporalResetMs) {
            dtMs = static_cast<float>(nowMs - lastNowMs_);
            temporalFresh = false;
        }
        lastNowMs_ = nowMs;
    }

    const float axisNorm = std::hypot(axisX, axisY);
    if (axisNorm < 1e-4f) return false;
    const float ux = axisX / axisNorm;
    const float uy = axisY / axisNorm;
    const float vx = -uy;
    const float vy = ux;

    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        srcMx_[i] = landmarks478x2[i * 2];
        srcMy_[i] = landmarks478x2[i * 2 + 1] / aspect;
        dispMx_[i] = 0.0f;
        dispMy_[i] = 0.0f;
        interiorMx_[i] = 0.0f;
        interiorMy_[i] = 0.0f;
    }
    for (int i = FaceMeshTopology::kFaceVertexCount; i < FaceMeshTopology::kBufferOffset; ++i) {
        dispMx_[i] = 0.0f;
        dispMy_[i] = 0.0f;
    }

    const float mirrorSign = ((srcMx_[454] - srcMx_[234]) * ux + (srcMy_[454] - srcMy_[234]) * uy >= 0.0f) ? 1.0f : -1.0f;
    frontFaceCw_ = (mirrorSign < 0.0f);

    const auto& tbl = tables();
    const auto& indices = FaceMeshTopology::kFullIndices;
    FaceMeshFilter::fitPose(srcMx_.data(), srcMy_.data(), pose_.data());
    float scale2 = FaceMeshFilter::viewScale2(pose_.data());
    if (!(scale2 > 1e-8f) || !std::isfinite(scale2)) {
        scale2 = faceWidth * faceHeight / kCanonFhOverFw;
    }
    for (int t = 0; t < FaceMeshTopology::kCanonicalTriangleCount; ++t) {
        const int a = indices[t * 3];
        const int b = indices[t * 3 + 1];
        const int c = indices[t * 3 + 2];
        const float projected = 0.5f * ((srcMx_[b] - srcMx_[a]) * (srcMy_[c] - srcMy_[a]) -
                                        (srcMy_[b] - srcMy_[a]) * (srcMx_[c] - srcMx_[a])) * mirrorSign;
        triVis_[t] = detail::smoothstep(0.0f, kRhoFade, projected / (scale2 * tbl.triArea3d[t]));
    }
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        float sumW = 0.0f, sumV = 0.0f;
        for (int p = tbl.vertexTriOffsets[i]; p < tbl.vertexTriOffsets[i + 1]; ++p) {
            const int t = tbl.vertexTriIndices[p];
            sumW += tbl.triArea3d[t];
            sumV += tbl.triArea3d[t] * triVis_[t];
        }
        const float v = (sumW > 0.0f) ? (sumV / sumW) : 0.0f;
        vis_[i] = v * v;
    }
    for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const int start = tbl.neighborOffsets[i];
            const int end = tbl.neighborOffsets[i + 1];
            if (end > start) {
                float sum = 0.0f;
                for (int p = start; p < end; ++p) sum += vis_[tbl.neighborIndices[p]];
                visScratch_[i] = 0.5f * vis_[i] + 0.5f * (sum / static_cast<float>(end - start));
            } else {
                visScratch_[i] = vis_[i];
            }
        }
        vis_ = visScratch_;
    }
    if (temporal) {
        if (temporalFresh) {
            visState_ = vis_;
        } else {
            const float down = 1.0f - std::exp(-dtMs / kVisTauDownMs);
            const float up = 1.0f - std::exp(-dtMs / kVisTauUpMs);
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
                const float target = vis_[i];
                visState_[i] += ((target < visState_[i]) ? down : up) * (target - visState_[i]);
                vis_[i] = visState_[i];
            }
        }
    }

    const float cx = 0.25f * (srcMx_[234] + srcMx_[454] + srcMx_[10] + srcMx_[152]);
    const float cy = 0.25f * (srcMy_[234] + srcMy_[454] + srcMy_[10] + srcMy_[152]);
    const float fw = faceWidth;
    const float fh = faceHeight;

    const float rEyeX = 0.35f * (srcMx_[33] + srcMx_[133]) + 0.15f * (srcMx_[159] + srcMx_[145]);
    const float rEyeY = 0.35f * (srcMy_[33] + srcMy_[133]) + 0.15f * (srcMy_[159] + srcMy_[145]);
    const float rawREyeW = std::max(std::hypot(srcMx_[33] - srcMx_[133], srcMy_[33] - srcMy_[133]), 0.01f);
    const float rawREyeH = std::max(std::hypot(srcMx_[159] - srcMx_[145], srcMy_[159] - srcMy_[145]), 0.004f);
    const float rBrowV = (srcMx_[105] - rEyeX) * vx + (srcMy_[105] - rEyeY) * vy;

    const float lEyeX = 0.35f * (srcMx_[362] + srcMx_[263]) + 0.15f * (srcMx_[386] + srcMx_[374]);
    const float lEyeY = 0.35f * (srcMy_[362] + srcMy_[263]) + 0.15f * (srcMy_[386] + srcMy_[374]);
    const float rawLEyeW = std::max(std::hypot(srcMx_[362] - srcMx_[263], srcMy_[362] - srcMy_[263]), 0.01f);
    const float rawLEyeH = std::max(std::hypot(srcMx_[386] - srcMx_[374], srcMy_[386] - srcMy_[374]), 0.004f);
    const float lBrowV = (srcMx_[334] - lEyeX) * vx + (srcMy_[334] - lEyeY) * vy;

    const float noseTipX = srcMx_[1], noseTipY = srcMy_[1];
    const float noseCenterX = 0.5f * (srcMx_[129] + srcMx_[358]);
    const float noseCenterY = 0.5f * (srcMy_[129] + srcMy_[358]);
    const float noseHalfW = std::max(0.5f * std::hypot(srcMx_[358] - srcMx_[129], srcMy_[358] - srcMy_[129]), 0.01f);
    const float noseLenBase = std::max(std::hypot(srcMx_[168] - noseTipX, srcMy_[168] - noseTipY), 0.02f);

    const float mouthX = 0.5f * (srcMx_[61] + srcMx_[291]);
    const float mouthY = 0.5f * (srcMy_[61] + srcMy_[291]);
    const float mouthHalfW = std::max(0.5f * std::hypot(srcMx_[291] - srcMx_[61], srcMy_[291] - srcMy_[61]), 0.01f);
    const float mouthHalfH = std::max(0.5f * std::hypot(srcMx_[0] - srcMx_[17], srcMy_[0] - srcMy_[17]), 0.35f * mouthHalfW);

    const float span132To361 = std::max(std::hypot(srcMx_[361] - srcMx_[132], srcMy_[361] - srcMy_[132]), 0.05f);
    const float cheekMidX = 0.5f * (srcMx_[132] + srcMx_[361]);
    const float cheekMidY = 0.5f * (srcMy_[132] + srcMy_[361]);
    const float rawSignedYaw = ((srcMx_[1] - cheekMidX) * ux + (srcMy_[1] - cheekMidY) * uy) / span132To361;

    const float lineMidX = 0.5f * (srcMx_[10] + srcMx_[152]);
    const float lineMidY = 0.5f * (srcMy_[10] + srcMy_[152]);
    const float midBoneX = 0.35f * srcMx_[168] + 0.35f * srcMx_[2] + 0.30f * srcMx_[152];
    const float midBoneY = 0.35f * srcMy_[168] + 0.35f * srcMy_[2] + 0.30f * srcMy_[152];
    const float halfH = std::max(0.5f * fh, 0.025f);
    const float spanBeta = 24.0f / std::max(fw, 0.05f);
    const float excessBeta = 20.0f / std::max(fw, 0.05f);
    const auto& leftCandidates = (mirrorSign > 0.0f) ? tbl.cheekAnatRight : tbl.cheekAnatLeft;
    const auto& rightCandidates = (mirrorSign > 0.0f) ? tbl.cheekAnatLeft : tbl.cheekAnatRight;
    const float rawWLeft = std::max(softMaxSpan(leftCandidates.data(), static_cast<int>(leftCandidates.size()),
                                                -1.0f, midBoneX, midBoneY, ux, uy, spanBeta), 0.10f * fw);
    const float rawWRight = std::max(softMaxSpan(rightCandidates.data(), static_cast<int>(rightCandidates.size()),
                                                 1.0f, midBoneX, midBoneY, ux, uy, spanBeta), 0.10f * fw);

    // v3.9.4 (C4/shape-3): Compute per-side outer eye corner lateral offsets (landmarks 33 & 263),
    // normalized against 31-pt Procrustes sRef and smoothed with envAlpha in temporal mode so
    // single-landmark outer-canthus jitter never modulates mid-cheek innerGate frame-to-frame.
    const float du33 = (srcMx_[33] - midBoneX) * ux + (srcMy_[33] - midBoneY) * uy;
    const float du263 = (srcMx_[263] - midBoneX) * ux + (srcMy_[263] - midBoneY) * uy;
    const float rawEyeDuLeft = (du33 < du263) ? std::fabs(du33) : std::fabs(du263);
    const float rawEyeDuRight = (du33 < du263) ? std::fabs(du263) : std::fabs(du33);

    float signedYaw = rawSignedYaw;
    float wLeft = rawWLeft;
    float wRight = rawWRight;
    float eyeDuLeft = rawEyeDuLeft;
    float eyeDuRight = rawEyeDuRight;
    float rEyeW = rawREyeW;
    float lEyeW = rawLEyeW;
    float rEyeH = rawREyeH;
    float lEyeH = rawLEyeH;
    float envAlpha = 1.0f;
    float yawJump = 0.0f;
    const float sRef = std::max(std::sqrt(scale2), 0.03f);
    if (temporal) {
        // Normalize contour widths against 31-pt Procrustes sRef (not noisy 2-pt ear-to-ear fw)
        // so single-landmark ear jitter never modulates mid-cheek slimming innerGate!
        const float invRef = 1.0f / sRef;
        const float rawL = rawWLeft * invRef;
        const float rawR = rawWRight * invRef;
        const float rawEyeDuL = rawEyeDuLeft * invRef;
        const float rawEyeDuR = rawEyeDuRight * invRef;
        const float rawRew = rawREyeW * invRef;
        const float rawLew = rawLEyeW * invRef;
        const float rawReh = rawREyeH * invRef;
        const float rawLeh = rawLEyeH * invRef;
        if (temporalFresh) {
            signedYawState_ = rawSignedYaw;
            rawSignedYawPrev_ = rawSignedYaw;
            yawVelEma_ = 0.0f;
            wLeftRatioState_ = rawL;
            wRightRatioState_ = rawR;
            eyeDuLeftRatioState_ = rawEyeDuL;
            eyeDuRightRatioState_ = rawEyeDuR;
            rEyeWRatioState_ = rawRew;
            lEyeWRatioState_ = rawLew;
            rEyeHRatioState_ = rawReh;
            lEyeHRatioState_ = rawLeh;
        } else {
            yawJump = std::fabs(rawSignedYaw - signedYawState_);
            const float envTau = 150.0f - (150.0f - 26.0f) * detail::smoothstep(0.05f, 0.10f, yawJump);
            // v3.9.7 (shape-k2/shape-e1): Symmetric continuous smoothed-velocity gate eliminates both
            // asymmetric noise rectification at stationary 40 deg and binary tau28/tau150 limit-cycle during
            // slow 3/4 oscillation while staying fast (~28ms) on genuine rapid turns.
            const float dtSec = std::max(dtMs * 0.001f, 0.008f);
            const float instYawVel = (rawSignedYaw - rawSignedYawPrev_) / dtSec;
            rawSignedYawPrev_ = rawSignedYaw;
            const float velAlpha = 1.0f - std::exp(-dtMs / 65.0f);
            yawVelEma_ += velAlpha * (instYawVel - yawVelEma_);
            const float turnSpeed = detail::smoothstep(0.65f, 1.50f, std::fabs(yawVelEma_)) *
                                    detail::smoothstep(0.35f, 0.52f, std::fabs(signedYawState_));
            const float yawTau = std::min(envTau, 150.0f - (150.0f - 28.0f) * turnSpeed);
            const float eyeEnvTau = 45.0f - (45.0f - 20.0f) * detail::smoothstep(0.05f, 0.10f, yawJump);
            envAlpha = 1.0f - std::exp(-dtMs / envTau);
            const float yawAlpha = 1.0f - std::exp(-dtMs / yawTau);
            const float eyeEnvAlpha = 1.0f - std::exp(-dtMs / eyeEnvTau);
            signedYawState_ += yawAlpha * (rawSignedYaw - signedYawState_);
            wLeftRatioState_ += envAlpha * (rawL - wLeftRatioState_);
            wRightRatioState_ += envAlpha * (rawR - wRightRatioState_);
            eyeDuLeftRatioState_ += envAlpha * (rawEyeDuL - eyeDuLeftRatioState_);
            eyeDuRightRatioState_ += envAlpha * (rawEyeDuR - eyeDuRightRatioState_);
            rEyeWRatioState_ += eyeEnvAlpha * (rawRew - rEyeWRatioState_);
            lEyeWRatioState_ += eyeEnvAlpha * (rawLew - lEyeWRatioState_);
            rEyeHRatioState_ += eyeEnvAlpha * (rawReh - rEyeHRatioState_);
            lEyeHRatioState_ += eyeEnvAlpha * (rawLeh - lEyeHRatioState_);
        }
        signedYaw = signedYawState_;
        wLeft = std::max(wLeftRatioState_ * sRef, 0.10f * fw);
        wRight = std::max(wRightRatioState_ * sRef, 0.10f * fw);
        eyeDuLeft = std::max(eyeDuLeftRatioState_ * sRef, 0.05f * fw);
        eyeDuRight = std::max(eyeDuRightRatioState_ * sRef, 0.05f * fw);
        rEyeW = std::max(rEyeWRatioState_ * sRef, 0.01f);
        lEyeW = std::max(lEyeWRatioState_ * sRef, 0.01f);
        rEyeH = std::max(rEyeHRatioState_ * sRef, 0.004f);
        lEyeH = std::max(lEyeHRatioState_ * sRef, 0.004f);
    }

    const float smoothYawWeight = 1.0f - detail::smoothstep(0.45f, FaceMeshFrame::kMaxYawRatio, std::fabs(signedYaw));
    const float effYawW = temporal ? smoothYawWeight : yawWeight;
    // v3.9.5 (shape-9): Fast-tracking signedYaw above 0.45 (tau <= 28ms) with (0.50..0.68) taper
    // fades steep-yaw warp cleanly before kMaxYawRatio (0.72) without injecting raw detector noise steps.
    const float yawKeep = 1.0f - detail::smoothstep(0.50f, 0.68f, std::fabs(signedYaw));
    const float faceMod = presence * steadyFace * yawKeep;
    const float eyeMod = presence * steadyEye * yawKeep;
    const float symFaceMod = faceMod * effYawW;
    const float symEyeMod = eyeMod * effYawW;

    const float sFaceSlim = detail::clampf(params.faceSlim, 0.0f, 1.0f) * faceMod;
    const float sVFace = detail::clampf(params.vFace, 0.0f, 1.0f) * faceMod;
    const float sFaceNarrow = detail::clampf(params.faceNarrow, 0.0f, 1.0f) * faceMod;
    const float sFaceSmall = detail::clampf(params.faceSmall, 0.0f, 1.0f) * symFaceMod;
    const float sCheekbone = detail::clampf(params.cheekboneSlim, 0.0f, 1.0f) * faceMod;
    // v3.9.4 (A6/shape-2): Reduced chin-only anti-elongation compensation (0.20*slim + 0.15*narrow)
    // so slimming genuinely narrows jaw/height ratio without shrinking the whole head or lowering hairline.
    const float antiLong = detail::clampf(sFaceSlim * 0.20f + sFaceNarrow * 0.15f, 0.0f, 1.0f) * effYawW;
    const float sChin = detail::clampf(params.chinShape, -1.0f, 1.0f) * symFaceMod;
    const float sForehead = detail::clampf(params.foreheadShape, -1.0f, 1.0f) * symFaceMod;
    const float sCranial = detail::clampf(params.cranialTop, 0.0f, 1.0f) * symFaceMod;
    const float sEyeEnlarge = detail::clampf(params.eyeEnlarge, 0.0f, 1.0f) * eyeMod;
    const float sEyeDist = detail::clampf(params.eyeDistance, -1.0f, 1.0f) * symEyeMod;
    const float sEyeCorner = detail::clampf(params.eyeCorner, -1.0f, 1.0f) * eyeMod;
    const float sNoseSlim = detail::clampf(params.noseSlim, 0.0f, 1.0f) * symEyeMod;
    const float sNoseLen = detail::clampf(params.noseLength, -1.0f, 1.0f) * symEyeMod;
    const float sMouthSize = detail::clampf(params.mouthSize, -1.0f, 1.0f) * symEyeMod;
    const float sMouthSmile = detail::clampf(params.mouthSmile, 0.0f, 1.0f) * symEyeMod;

    const float poseRightFade = 1.0f - 0.82f * detail::smoothstep(0.36f, 0.66f, signedYaw);
    const float poseLeftFade = 1.0f - 0.82f * detail::smoothstep(0.36f, 0.66f, -signedYaw);
    const float smoothRightW = 1.0f - detail::smoothstep(0.36f, 0.66f, signedYaw);
    const float smoothLeftW = 1.0f - detail::smoothstep(0.36f, 0.66f, -signedYaw);
    // v3.7.0 Extreme side-profile asymmetric narrow-side contour guard (ratio 1.95..2.55x):
    // prevents far narrow cheek silhouette from collapsing inward on steep 3/4 turns.
    const float wRatioL = wRight / std::max(wLeft, 1e-4f);
    const float wRatioR = wLeft / std::max(wRight, 1e-4f);
    const float narrowGuardL = 1.0f - 0.85f * detail::smoothstep(1.95f, 2.55f, wRatioL);
    const float narrowGuardR = 1.0f - 0.85f * detail::smoothstep(1.95f, 2.55f, wRatioR);
    const float effRightW = std::min(temporal ? smoothRightW : rightWeight, poseRightFade) * narrowGuardR;
    const float effLeftW = std::min(temporal ? smoothLeftW : leftWeight, poseLeftFade) * narrowGuardL;
    const float rSideW = (mirrorSign > 0.0f) ? effLeftW : effRightW;
    const float lSideW = (mirrorSign > 0.0f) ? effRightW : effLeftW;
    const float maxDisp = 0.135f * std::max(fw, fh);
    const auto& canonicalUv = FaceMeshTopology::kCanonicalUv;

    const float wMean = std::max(0.5f * (wLeft + wRight), 0.16f * fw);
    const float frontGate = 1.0f - detail::smoothstep(0.08f, 0.34f, std::fabs(signedYaw));
    const float symExp = 0.42f * frontGate;
    const float lSym = detail::clampf(std::pow(wLeft / wMean, symExp), 0.88f, 1.14f);
    const float rSym = detail::clampf(std::pow(wRight / wMean, symExp), 0.88f, 1.14f);
    const bool contourActive = (sFaceSlim > 1e-4f || sVFace > 1e-4f || sFaceNarrow > 1e-4f || sCheekbone > 1e-4f);

    auto evalContour2D = [&](float px, float py) -> Vec2 {
        if (!contourActive) return {0.0f, 0.0f};
        const float du = (px - midBoneX) * ux + (py - midBoneY) * uy;
        const float dv = ((px - lineMidX) * vx + (py - lineMidY) * vy) / halfH;
        const float sideBlend = detail::smoothstep(-0.14f * wMean, 0.14f * wMean, du);
        const float wSide = wLeft + (wRight - wLeft) * sideBlend;
        const float sideW = effLeftW + (effRightW - effLeftW) * sideBlend;
        const float symW = lSym + (rSym - lSym) * sideBlend;
        const float jawTaper = 1.0f - 0.36f * detail::smoothstep(-0.35f, -0.92f, dv);
        const float wLocal = std::max(wSide * jawTaper, 0.065f * fw);
        const float xi = std::fabs(du) / wLocal;
        if (xi >= 2.15f) return {0.0f, 0.0f};

        const float eyeDuSide = eyeDuLeft + (eyeDuRight - eyeDuLeft) * sideBlend;
        // v3.9.6 (shape-n1/shape-m2): Eye-band lateral exclusion interpolates innerGate threshold smoothly
        // over a wide transition (-0.72..0.12) so nasal-alar vertical shear stays within v3.9.1 + 2 deg,
        // while relaxing xi0 below the mouth (-0.62..-0.95) eliminates the chin-tip kink at v=148/377
        // and preserves <1% eye/iris horizontal deformation inside the eye band.
        const float denomEye = std::max(wSide, 0.065f * fw);
        const float xiE = std::fabs(du) / denomEye;
        const float xi0E = detail::clampf(eyeDuSide / denomEye + 0.05f, 0.60f, 0.85f);
        const float outerGate = 1.0f - detail::smoothstep(1.35f, 2.10f, xi);
        const float vertGuard = 1.0f - detail::smoothstep(0.95f, 1.32f, -dv);

        // v3.9.7 (shape-k1): Decouple lower-face (faceSlim/vFace) and upper-face (faceNarrow/cheekboneSlim)
        // lateral inner gates. Only faceNarrow/cheekboneSlim act at eye height (dv > 0.15), so giving them
        // a dedicated eyeBand-excluded etaUpper preserves 0% eye/iris horizontal warp and <= 8.6 deg nasal-alar
        // shear, while giving faceSlim/vFace a smooth chin-only etaLower (no eyeBand discontinuity) eliminates
        // the mouth-corner/chin vertical shear spike (beating v3.9.1 full-face shear & v3.9.5 min singular value)
        // and keeps jaw-contour chin-tip kink < 1 deg.
        const float chinSmooth = detail::smoothstep(-0.45f, -0.95f, dv);
        const float xi0Lower = 0.36f - 0.36f * chinSmooth;
        const float spanLower = 0.36f + 0.14f * chinSmooth;
        const float etaLower = detail::smoothstep(xi0Lower, xi0Lower + spanLower, xi) * outerGate;

        const float eyeBand = detail::smoothstep(-0.78f, 0.16f, dv) * (1.0f - detail::smoothstep(0.70f, 0.95f, dv));
        const float xi0Upper = 0.26f + eyeBand * (xi0E - 0.26f);
        const float spanUpper = 0.34f + eyeBand * (0.25f - 0.34f);
        const float xiUpper = xi + eyeBand * (xiE - xi);
        const float etaUpper = detail::smoothstep(xi0Upper, xi0Upper + spanUpper, xiUpper) * outerGate;

        float frac = 0.0f;
        float liftV = 0.0f;
        if (etaLower > 1e-5f || etaUpper > 1e-5f) {
            if (sFaceSlim > 0.0f && etaLower > 1e-5f) {
                const float slimVert = detail::smoothstep(-1.18f, -0.76f, dv) * (1.0f - detail::smoothstep(0.15f, 0.42f, dv));
                frac += sFaceSlim * 0.068f * slimVert * etaLower;
            }
            if (sVFace > 0.0f && etaLower > 1e-5f) {
                const float vVert = gaussian(dv, -0.58f, 0.48f) * vertGuard;
                frac += sVFace * 0.068f * vVert * etaLower;
                liftV += sVFace * 0.012f * fh * sideW * etaLower * vVert;
            }
            if (sFaceNarrow > 0.0f && etaUpper > 1e-5f) {
                frac += sFaceNarrow * 0.065f * gaussian(dv, -0.05f, 0.68f) * etaUpper;
            }
            if (sCheekbone > 0.0f && etaUpper > 1e-5f) {
                frac += sCheekbone * 0.070f * gaussian(dv, 0.06f, 0.36f) * detail::smoothstep(-0.55f, -0.20f, dv) * vertGuard * etaUpper;
            }
            frac *= symW * sideW;
            const float maxFrac = 0.115f * symW;
            if (frac > 0.0f && maxFrac > 1e-5f) {
                frac = (frac * maxFrac) / std::hypot(frac, maxFrac);
            }
        }
        if (antiLong > 1e-4f && outerGate > 1e-5f) {
            // Chin-only gentle anti-elongation lift (hairline/forehead left strictly to foreheadShape/faceSmall).
            const float lowerW = detail::smoothstep(-0.25f, -0.95f, dv) * vertGuard;
            liftV += antiLong * outerGate * fh * 0.036f * lowerW;
        }
        if (frac <= 1e-6f && std::fabs(liftV) <= 1e-6f) return {0.0f, 0.0f};
        const float sgn = 2.0f * sideBlend - 1.0f;
        const float cDu = -sgn * wSide * frac;
        return {cDu * ux + liftV * vx, cDu * uy + liftV * vy};
    };

    if (faceMod > 0.0f || eyeMod > 0.0f) {
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const float px = srcMx_[i], py = srcMy_[i];
            const float vVis = vis_[i];
            float intDx = 0.0f, intDy = 0.0f;
            if (vVis > 1e-4f) {
                const float rx = px - cx, ry = py - cy;
                const float du = rx * ux + ry * uy;
                const float dv = rx * vx + ry * vy;
                const float nu = (canonicalUv[i * 2] - 0.5f) * 2.0f;
                const float screenNu = nu * mirrorSign;
                const float nv = (0.5f - canonicalUv[i * 2 + 1]) * 2.0f;
                const float sideT = detail::clampf((screenNu + 0.15f) / 0.30f, 0.0f, 1.0f);
                const float sideW = effLeftW + (effRightW - effLeftW) * sideT;
                const float absNu = std::fabs(screenNu);
                float dU = 0.0f, dV = 0.0f;

                if (sFaceSmall > 0.0f) {
                    const float sc = sFaceSmall * 0.060f;
                    dU -= du * sc * sideW;
                    dV -= dv * sc;
                }
                if (std::fabs(sChin) > 1e-4f) {
                    const float w = radialFalloff(px, py, srcMx_[152], srcMy_[152], 0.34f * fh);
                    if (w > 0.0f) {
                        dV -= sChin * 0.095f * fh * w;
                        if (sChin > 0.0f) dU -= du * (sChin * 0.065f * w);
                    }
                }
                if (std::fabs(sForehead) > 1e-4f && nv > 0.32f) {
                    const float vert = detail::smoothstep(0.32f, 0.78f, nv);
                    const float lat = 1.0f - detail::smoothstep(0.70f, 1.05f, absNu) * 0.35f;
                    dV += sForehead * 0.092f * fh * vert * lat;
                }
                if (sCranial > 1e-4f && nv > 0.58f) {
                    const float tuck = detail::smoothstep(0.58f, 0.92f, nv) * (1.0f - detail::smoothstep(0.65f, 1.0f, absNu) * 0.5f);
                    dV -= sCranial * 0.006f * fh * tuck;
                }
                if (sEyeEnlarge > 0.0f) {
                    const Vec2 re = eyeEnlargeDelta(px, py, rEyeX, rEyeY, rEyeW, rEyeH, rBrowV, ux, uy, vx, vy, sEyeEnlarge * rSideW);
                    dU += re.x; dV += re.y;
                    const Vec2 le = eyeEnlargeDelta(px, py, lEyeX, lEyeY, lEyeW, lEyeH, lBrowV, ux, uy, vx, vy, sEyeEnlarge * lSideW);
                    dU += le.x; dV += le.y;
                }
                if (std::fabs(sEyeDist) > 1e-4f) {
                    const float wr = radialFalloff(px, py, rEyeX, rEyeY, 1.25f * rEyeW);
                    const float wl = radialFalloff(px, py, lEyeX, lEyeY, 1.25f * lEyeW);
                    if (wr > 0.0f) dU -= mirrorSign * sEyeDist * 0.18f * rEyeW * wr;
                    if (wl > 0.0f) dU += mirrorSign * sEyeDist * 0.18f * lEyeW * wl;
                }
                if (std::fabs(sEyeCorner) > 1e-4f) {
                    const float wr = radialFalloff(px, py, srcMx_[33], srcMy_[33], 0.85f * rEyeW);
                    const float wl = radialFalloff(px, py, srcMx_[263], srcMy_[263], 0.85f * lEyeW);
                    if (wr > 0.0f) dV += sEyeCorner * 0.18f * rEyeW * wr * rSideW;
                    if (wl > 0.0f) dV += sEyeCorner * 0.18f * lEyeW * wl * lSideW;
                }
                if (sNoseSlim > 0.0f) {
                    const float dnx = px - noseCenterX, dny = py - noseCenterY;
                    const float nDu = dnx * ux + dny * uy;
                    const float nDv = dnx * vx + dny * vy;
                    const float nr = std::hypot(nDu / (1.85f * noseHalfW), nDv / (1.45f * noseHalfW));
                    if (nr < 1.0f) {
                        const float t = 1.0f - nr * nr;
                        dU -= nDu * (sNoseSlim * 0.28f * t * t);
                    }
                }
                if (std::fabs(sNoseLen) > 1e-4f) {
                    const float dnx = px - noseTipX, dny = py - noseTipY;
                    const float nDu = dnx * ux + dny * uy;
                    const float nDv = dnx * vx + dny * vy;
                    const float nr = std::hypot(nDu / (1.65f * noseHalfW), nDv / (0.85f * noseLenBase));
                    if (nr < 1.0f) {
                        const float t = 1.0f - nr * nr;
                        dV -= sNoseLen * 0.15f * noseLenBase * t * t;
                    }
                }
                if (std::fabs(sMouthSize) > 1e-4f) {
                    const float dmx = px - mouthX, dmy = py - mouthY;
                    const float mDu = dmx * ux + dmy * uy;
                    const float mDv = dmx * vx + dmy * vy;
                    const float mr = std::hypot(mDu / (1.55f * mouthHalfW), mDv / (1.95f * mouthHalfH));
                    if (mr < 1.0f) {
                        const float t = 1.0f - mr * mr;
                        const float w = t * t;
                        dU += mDu * (sMouthSize * 0.22f * w);
                        dV += mDv * (sMouthSize * 0.18f * w);
                    }
                }
                if (sMouthSmile > 0.0f) {
                    const float wr = radialFalloff(px, py, srcMx_[61], srcMy_[61], 0.72f * mouthHalfW);
                    const float wl = radialFalloff(px, py, srcMx_[291], srcMy_[291], 0.72f * mouthHalfW);
                    if (wr > 0.0f) {
                        dV += sMouthSmile * 0.16f * mouthHalfW * wr;
                        dU -= mirrorSign * sMouthSmile * 0.045f * mouthHalfW * wr;
                    }
                    if (wl > 0.0f) {
                        dV += sMouthSmile * 0.16f * mouthHalfW * wl;
                        dU += mirrorSign * sMouthSmile * 0.045f * mouthHalfW * wl;
                    }
                }
                dU *= vVis;
                dV *= vVis;
                intDx = dU * ux + dV * vx;
                intDy = dU * uy + dV * vy;
            }
            interiorMx_[i] = intDx;
            interiorMy_[i] = intDy;
        }

        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const int start = tbl.neighborOffsets[i];
            const int end = tbl.neighborOffsets[i + 1];
            if (end > start) {
                float sx = 0.0f, sy = 0.0f;
                for (int p = start; p < end; ++p) {
                    const int nb = tbl.neighborIndices[p];
                    sx += interiorMx_[nb];
                    sy += interiorMy_[nb];
                }
                const float inv = 1.0f / static_cast<float>(end - start);
                smoothMx_[i] = 0.80f * interiorMx_[i] + 0.20f * (sx * inv);
                smoothMy_[i] = 0.80f * interiorMy_[i] + 0.20f * (sy * inv);
            } else {
                smoothMx_[i] = interiorMx_[i];
                smoothMy_[i] = interiorMy_[i];
            }
        }
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            interiorMx_[i] = smoothMx_[i];
            interiorMy_[i] = smoothMy_[i];
            const Vec2 con = evalContour2D(srcMx_[i], srcMy_[i]);
            float totDx = interiorMx_[i] + con.x;
            float totDy = interiorMy_[i] + con.y;
            const float mag = std::hypot(totDx, totDy);
            if (mag > 1e-6f && maxDisp > 1e-5f) {
                const float capScale = maxDisp / std::hypot(mag, maxDisp);
                totDx *= capScale;
                totDy *= capScale;
            }
            dispMx_[i] = totDx;
            dispMy_[i] = totDy;
        }
    }

    const auto& oval = FaceMeshTopology::kFaceOval;
    const float padMidBase = std::max(0.055f * fw, 0.048f * fh);
    const float padOuterBase = std::max(0.185f * fw, 0.165f * fh);
    for (int k = 0; k < FaceMeshTopology::kOvalVertexCount; ++k) {
        const int oi = oval[k];
        const float ox = srcMx_[oi], oy = srcMy_[oi];
        const float cNu = (canonicalUv[oi * 2] - 0.5f) * 2.0f * mirrorSign;
        const float cNv = (0.5f - canonicalUv[oi * 2 + 1]) * 2.0f;
        const float cLen = std::max(std::hypot(cNu, cNv), 1e-4f);
        const float cxDir = (cNu * ux + cNv * vx) / cLen;
        const float cyDir = (cNu * uy + cNv * vy) / cLen;
        const float dx = ox - cx, dy = oy - cy;
        const float len = std::hypot(dx, dy);
        const float radX = (len > 1e-4f) ? (dx / len) : cxDir;
        const float radY = (len > 1e-4f) ? (dy / len) : cyDir;
        const float align = detail::clampf(radX * cxDir + radY * cyDir, 0.0f, 1.0f);
        const float wRad = 0.65f * align;
        const float mx = cxDir * (1.0f - wRad) + radX * wRad;
        const float my = cyDir * (1.0f - wRad) + radY * wRad;
        const float mNorm = std::max(std::hypot(mx, my), 1e-4f);
        const float rx = mx / mNorm, ry = my / mNorm;
        const float rawSilExcess = std::min(softMaxOvalExcess(k, ox, oy, rx, ry, excessBeta), 0.070f * fw);
        float silExcess = rawSilExcess;
        if (temporal) {
            const float rawSilRatio = rawSilExcess / std::max(fw, 1e-4f);
            if (temporalFresh) {
                silExcessRatioState_[k] = rawSilRatio;
            } else {
                silExcessRatioState_[k] += envAlpha * (rawSilRatio - silExcessRatioState_[k]);
            }
            silExcess = silExcessRatioState_[k] * fw;
        }
        const float outwardDisp = std::max(0.0f, dispMx_[oi] * rx + dispMy_[oi] * ry);
        const float crownW = (sCranial > 1e-4f) ? detail::smoothstep(-0.10f, 0.68f, cNv) : 0.0f;
        const float templeW = (sCranial > 1e-4f)
            ? detail::smoothstep(0.02f, 0.44f, cNv) * (1.0f - detail::smoothstep(0.72f, 0.98f, cNv))
            : 0.0f;
        const float cranialSpan = sCranial * crownW * 0.095f * fh;
        const float padMid = padMidBase + silExcess + outwardDisp * 0.70f + cranialSpan;
        const float midX = ox + rx * padMid;
        const float midY = oy + ry * padMid;
        const int mi = FaceMeshTopology::kMidBufferOffset + k;
        srcMx_[mi] = midX; srcMy_[mi] = midY;
        dstMx_[mi] = midX; dstMy_[mi] = midY;
        if (faceMod > 0.0f || eyeMod > 0.0f) {
            const Vec2 mc = evalContour2D(midX, midY);
            float mdx = mc.x + interiorMx_[oi] * kMidInteriorCarry;
            float mdy = mc.y + interiorMy_[oi] * kMidInteriorCarry;
            if (sCranial > 1e-4f && (crownW > 0.0f || templeW > 0.0f)) {
                const float liftV = sCranial * crownW * 0.072f * fh;
                const float sideSgn = detail::smoothstep(-0.18f, 0.18f, cNu) * 2.0f - 1.0f;
                const float fluffU = sCranial * templeW * 0.026f * fw * sideSgn;
                mdx += fluffU * ux + liftV * vx;
                mdy += fluffU * uy + liftV * vy;
            }
            dispMx_[mi] = mdx;
            dispMy_[mi] = mdy;
        }

        const float padOuter = std::max(padOuterBase + silExcess * 0.85f + outwardDisp * 1.50f,
                                        padMid - cranialSpan + 0.115f * fw) +
                               sCranial * crownW * kCranialOuterPad * fh;
        const float bx = ox + rx * padOuter;
        const float by = oy + ry * padOuter;
        const int bi = FaceMeshTopology::kBufferOffset + k;
        srcMx_[bi] = bx; srcMy_[bi] = by;
        dstMx_[bi] = bx; dstMy_[bi] = by;
    }

    const float outerR = std::max(3.2f, std::max(fw, fh) * 5.0f);
    const auto& outerDirs = FaceMeshTopology::kOuterDirections;
    for (int j = 0; j < FaceMeshTopology::kOuterVertexCount; ++j) {
        const float du = outerDirs[j * 2] * mirrorSign * outerR;
        const float dv = outerDirs[j * 2 + 1] * outerR;
        const int fi = FaceMeshTopology::kOuterOffset + j;
        srcMx_[fi] = cx + du * ux + dv * vx;
        srcMy_[fi] = cy + du * uy + dv * vy;
        dstMx_[fi] = srcMx_[fi];
        dstMy_[fi] = srcMy_[fi];
    }

    for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
        const int a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
        const float ex1 = srcMx_[b] - srcMx_[a], ey1 = srcMy_[b] - srcMy_[a];
        const float ex2 = srcMx_[c] - srcMx_[a], ey2 = srcMy_[c] - srcMy_[a];
        const float ex3 = srcMx_[c] - srcMx_[b], ey3 = srcMy_[c] - srcMy_[b];
        const float origArea = ex1 * ey2 - ey1 * ex2;
        const float maxEdge2 = std::max({ex1 * ex1 + ey1 * ey1, ex2 * ex2 + ey2 * ey2, ex3 * ex3 + ey3 * ey3});
        triOrig_[t] = origArea;
        triFront_[t] = (origArea * mirrorSign > 0.0f && maxEdge2 > 0.0f)
            ? detail::smoothstep(0.0f, kFrontWShape, origArea * mirrorSign / maxEdge2)
            : 0.0f;
    }

    if (temporal && !temporalFresh) {
        const float up = 1.0f - std::exp(-dtMs / kRelaxTauUpMs);
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            relax_[i] = relaxState_[i] + up * (1.0f - relaxState_[i]);
        }
    } else {
        relax_.fill(1.0f);
    }
    applyDisplacement();

    for (int iter = 0; iter < kRelaxIter; ++iter) {
        relaxFactor_.fill(1.0f);
        bool anyLimited = false;
        for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
            const float w = triFront_[t];
            if (w <= 0.0f) continue;
            const float r = areaRatio(t);
            const float h = kRelaxHFloor + (1.0f - kRelaxHFloor) * detail::smoothstep(kRelaxR0, kRelaxR1, r);
            const float f = 1.0f - w * (1.0f - h);
            if (f >= 1.0f) continue;
            limitVertices(t, f);
            anyLimited = true;
        }
        if (!anyLimited) {
            bool allUnity = true;
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
                if (relax_[i] < 1.0f - 1e-5f) {
                    allUnity = false;
                    break;
                }
            }
            if (allUnity) break;
        }
        bool relaxChanged = anyLimited;
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) relax_[i] *= relaxFactor_[i];
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const int start = tbl.neighborOffsets[i];
            const int end = tbl.neighborOffsets[i + 1];
            if (end > start) {
                float minNb = relax_[i], sumNb = 0.0f;
                for (int p = start; p < end; ++p) {
                    const float rv = relax_[tbl.neighborIndices[p]];
                    if (rv < minNb) minNb = rv;
                    sumNb += rv;
                }
                const float avgNb = sumNb / static_cast<float>(end - start);
                const float nextVal = std::min(relax_[i], 0.5f * (relax_[i] + std::min(avgNb, minNb + 0.25f)));
                if (std::fabs(nextVal - relax_[i]) > 1e-5f) relaxChanged = true;
                relaxScratch_[i] = nextVal;
            } else {
                relaxScratch_[i] = relax_[i];
            }
        }
        relax_ = relaxScratch_;
        applyDisplacement();
        if (!relaxChanged) break;
    }

    relaxFactor_.fill(1.0f);
    for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
        const float w = triFront_[t];
        if (w <= 0.0f) continue;
        const float cap = 1.0f - kSafetyDepth * detail::smoothstep(kSafetyW0, kSafetyW1, w) *
                          (1.0f - detail::smoothstep(kSafetyR0, kSafetyR1, areaRatio(t)));
        if (cap < 1.0f) limitVertices(t, cap);
    }
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) relax_[i] *= relaxFactor_[i];
    if (temporal) {
        if (!temporalFresh) {
            const float down = 1.0f - std::exp(-dtMs / kRelaxTauDownMs);
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
                if (relax_[i] < relaxState_[i] && relaxFactor_[i] > 0.75f) {
                    relax_[i] = relaxState_[i] + down * (relax_[i] - relaxState_[i]);
                }
            }
        }
        relaxState_ = relax_;
        temporalValid_ = true;
    }
    applyDisplacement();

    // v3.9.7 (shape-d1/shape-k3): Protect all visible front-facing triangles (>= ~3.9px^2, triFront > 0.05)
    // including slender mouth-corner/lip-seam triangles on close-up big faces (fw capped at 0.45).
    const float fwCap = std::min(fw, 0.45f);
    const float minVerifyTriArea2 = 7.2e-5f * fwCap * fwCap;
    for (int k = 0; k < 4; ++k) {
        bool anyBad = false;
        for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
            if (triFront_[t] <= 0.05f || std::fabs(triOrig_[t]) < minVerifyTriArea2) continue;
            if (areaRatio(t) < 0.05f) {
                const int a = indices[t * 3];
                const int b = indices[t * 3 + 1];
                const int c = indices[t * 3 + 2];
                float m = 1.0f;
                if (a < FaceMeshTopology::kFaceVertexCount) m = std::min(m, relax_[a]);
                if (b < FaceMeshTopology::kFaceVertexCount) m = std::min(m, relax_[b]);
                if (c < FaceMeshTopology::kFaceVertexCount) m = std::min(m, relax_[c]);
                const float clamped = m * 0.7f;
                if (a < FaceMeshTopology::kFaceVertexCount) relax_[a] = clamped;
                if (b < FaceMeshTopology::kFaceVertexCount) relax_[b] = clamped;
                if (c < FaceMeshTopology::kFaceVertexCount) relax_[c] = clamped;
                anyBad = true;
            }
        }
        if (!anyBad) break;
        applyDisplacement();
    }
    {
        bool stillBad = false;
        for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
            if (triFront_[t] > 0.05f && std::fabs(triOrig_[t]) >= minVerifyTriArea2 && areaRatio(t) < 0.05f) {
                stillBad = true;
                break;
            }
        }
        float targetLambda = 1.0f;
        const auto savedRelax = relax_;
        if (stillBad) {
            float lo = 0.0f, hi = 1.0f;
            for (int step = 0; step < 6; ++step) {
                const float mid = 0.5f * (lo + hi);
                for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) relax_[i] = savedRelax[i] * mid;
                applyDisplacement();
                bool ok = true;
                for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
                    if (triFront_[t] > 0.05f && std::fabs(triOrig_[t]) >= minVerifyTriArea2 && areaRatio(t) < 0.05f) {
                        ok = false;
                        break;
                    }
                }
                if (ok) lo = mid; else hi = mid;
            }
            targetLambda = lo;
        }
        if (temporal) {
            if (temporalFresh) {
                bisectLambdaState_ = targetLambda;
            } else if (targetLambda < bisectLambdaState_) {
                bisectLambdaState_ = targetLambda;
            } else {
                const float up = 1.0f - std::exp(-dtMs / kRelaxTauUpMs);
                bisectLambdaState_ += up * (targetLambda - bisectLambdaState_);
            }
            targetLambda = bisectLambdaState_;
        }
        if (targetLambda < 0.9999f || stillBad) {
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) relax_[i] = savedRelax[i] * targetLambda;
            applyDisplacement();
        }
    }

    // Rigid-motion-invariant face-local displacement smoothing (zero lag under 2D translation/roll/scale!):
    // suppresses high-frequency cheek slimming/contour innerGate & grazing-triangle relax jitter while keeping
    // instant slider response and fast 3D yaw turn adaptation.
    if (temporal) {
        const float invRef = 1.0f / sRef;
        const bool paramsChanged = !dispTemporalValid_ ||
            std::fabs(params.faceSlim - prevParams_.faceSlim) > 0.004f ||
            std::fabs(params.vFace - prevParams_.vFace) > 0.004f ||
            std::fabs(params.faceNarrow - prevParams_.faceNarrow) > 0.004f ||
            std::fabs(params.faceSmall - prevParams_.faceSmall) > 0.004f ||
            std::fabs(params.cheekboneSlim - prevParams_.cheekboneSlim) > 0.004f ||
            std::fabs(params.chinShape - prevParams_.chinShape) > 0.004f ||
            std::fabs(params.foreheadShape - prevParams_.foreheadShape) > 0.004f ||
            std::fabs(params.eyeEnlarge - prevParams_.eyeEnlarge) > 0.004f ||
            std::fabs(params.eyeDistance - prevParams_.eyeDistance) > 0.004f ||
            std::fabs(params.eyeCorner - prevParams_.eyeCorner) > 0.004f ||
            std::fabs(params.noseSlim - prevParams_.noseSlim) > 0.004f ||
            std::fabs(params.noseLength - prevParams_.noseLength) > 0.004f ||
            std::fabs(params.mouthSize - prevParams_.mouthSize) > 0.004f ||
            std::fabs(params.mouthSmile - prevParams_.mouthSmile) > 0.004f ||
            std::fabs(params.cranialTop - prevParams_.cranialTop) > 0.004f;
        const bool initDisp = temporalFresh || !dispTemporalValid_;
        const float dtRef = std::max(dtMs, 16.0f);
        const bool yawWeightChanging = !initDisp && (std::fabs(effYawW - prevEffYawW_) > 0.0014f * dtRef);
        const bool modChanging = !initDisp &&
            (std::fabs(eyeMod - prevEyeMod_) > 0.0011f * dtRef || std::fabs(faceMod - prevFaceMod_) > 0.0011f * dtRef);
        const bool fastGlobal = (paramsChanged || yawJump > 0.10f || yawWeightChanging || modChanging);
        const float outerDispAlpha = initDisp ? 1.0f : (1.0f - std::exp(-dtMs / (fastGlobal ? kDispTauFastMs : kDispTauSteadyMs)));
        const float innerDispAlpha = initDisp ? 1.0f : (1.0f - std::exp(-dtMs / (fastGlobal ? kDispTauFastMs : kDispTauInnerMs)));

        std::array<float, FaceMeshTopology::kBufferOffset> rawUBuf{};
        std::array<float, FaceMeshTopology::kBufferOffset> rawVBuf{};
        for (int i = 0; i < FaceMeshTopology::kBufferOffset; ++i) {
            const float effDx = dstMx_[i] - srcMx_[i];
            const float effDy = dstMy_[i] - srcMy_[i];
            const float rawU = (effDx * ux + effDy * uy) * invRef;
            const float rawV = (effDx * vx + effDy * vy) * invRef;
            rawUBuf[i] = rawU;
            rawVBuf[i] = rawV;
            if (initDisp) {
                dispUState_[i] = rawU;
                dispVState_[i] = rawV;
            } else {
                const float wi = tbl.innerFeatureW[i];
                const float ai = outerDispAlpha + wi * (innerDispAlpha - outerDispAlpha);
                dispUState_[i] += ai * (rawU - dispUState_[i]);
                dispVState_[i] += ai * (rawV - dispVState_[i]);
            }
            const float smU = dispUState_[i] * sRef;
            const float smV = dispVState_[i] * sRef;
            dstMx_[i] = srcMx_[i] + smU * ux + smV * vx;
            dstMy_[i] = srcMy_[i] + smU * uy + smV * vy;
        }
        prevParams_ = params;
        prevEffYawW_ = effYawW;
        prevEyeMod_ = eyeMod;
        prevFaceMod_ = faceMod;
        dispTemporalValid_ = true;

        // Post-smoothing fold guard: if a visible front-facing triangle approaches inversion, blend gently
        // toward this frame's already fold-safe (rawU, rawV) rather than snapping displacement to zero!
        for (int pass = 0; pass < 2; ++pass) {
            bool fixedAny = false;
            for (int t = 0; t < FaceMeshTopology::kDeformableTriangleCount; ++t) {
                if (triFront_[t] <= 0.05f || std::fabs(triOrig_[t]) < minVerifyTriArea2) continue;
                const float r = areaRatio(t);
                if (r >= 0.04f) continue;
                const float backToSafe = (pass == 0) ? (1.0f - detail::smoothstep(0.0f, 0.04f, r)) : 1.0f;
                for (int k = 0; k < 3; ++k) {
                    const int v = indices[t * 3 + k];
                    if (v < FaceMeshTopology::kBufferOffset) {
                        dispUState_[v] += backToSafe * (rawUBuf[v] - dispUState_[v]);
                        dispVState_[v] += backToSafe * (rawVBuf[v] - dispVState_[v]);
                        const float smU = dispUState_[v] * sRef;
                        const float smV = dispVState_[v] * sRef;
                        dstMx_[v] = srcMx_[v] + smU * ux + smV * vx;
                        dstMy_[v] = srcMy_[v] + smU * uy + smV * vy;
                    }
                }
                fixedAny = true;
            }
            if (!fixedAny) break;
        }
    }

    for (int i = 0; i < FaceMeshTopology::kTotalVertexCount; ++i) {
        const float sx = srcMx_[i];
        const float sy = srcMy_[i] * aspect;
        const float dx = dstMx_[i];
        const float dy = dstMy_[i] * aspect;
        const int k = i * 4;
        vertexData_[k] = dx * 2.0f - 1.0f;
        vertexData_[k + 1] = dy * 2.0f - 1.0f;
        vertexData_[k + 2] = sx;
        vertexData_[k + 3] = sy;
        if (i < FaceMeshTopology::kFaceVertexCount) {
            warpedLandmarks_[i * 2] = dx;
            warpedLandmarks_[i * 2 + 1] = dy;
        }
    }

    transformIris(landmarks478x2, 468, 33, 133, 159, 145, aspect);
    transformIris(landmarks478x2, 473, 362, 263, 386, 374, aspect);
    return true;
}

float FaceMeshDeformer::areaRatio(int t) const {
    const auto& indices = FaceMeshTopology::kFullIndices;
    const int a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const float newArea = (dstMx_[b] - dstMx_[a]) * (dstMy_[c] - dstMy_[a]) -
                          (dstMy_[b] - dstMy_[a]) * (dstMx_[c] - dstMx_[a]);
    return newArea / triOrig_[t];
}

void FaceMeshDeformer::limitVertices(int t, float f) {
    const auto& indices = FaceMeshTopology::kFullIndices;
    float minR = 1.0f;
    for (int k = 0; k < 3; ++k) {
        const int v = indices[t * 3 + k];
        const int target = (v < FaceMeshTopology::kFaceVertexCount) ? v
            : (v < FaceMeshTopology::kBufferOffset) ? FaceMeshTopology::kFaceOval[v - FaceMeshTopology::kMidBufferOffset]
            : -1;
        if (target >= 0 && relax_[target] < minR) minR = relax_[target];
    }
    const float eq = detail::clampf((1.0f - f) * 2.5f, 0.0f, 1.0f);
    for (int k = 0; k < 3; ++k) {
        const int v = indices[t * 3 + k];
        const int target = (v < FaceMeshTopology::kFaceVertexCount) ? v
            : (v < FaceMeshTopology::kBufferOffset) ? FaceMeshTopology::kFaceOval[v - FaceMeshTopology::kMidBufferOffset]
            : -1;
        if (target >= 0) {
            const float cur = relax_[target];
            const float effF = (cur > 1e-6f) ? (f * (cur + eq * (minR - cur)) / cur) : f;
            if (effF < relaxFactor_[target]) relaxFactor_[target] = effF;
        }
    }
}

void FaceMeshDeformer::applyDisplacement() {
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float r = relax_[i];
        dstMx_[i] = srcMx_[i] + dispMx_[i] * r;
        dstMy_[i] = srcMy_[i] + dispMy_[i] * r;
    }
    for (int k = 0; k < FaceMeshTopology::kOvalVertexCount; ++k) {
        const int oi = FaceMeshTopology::kFaceOval[k];
        const int mi = FaceMeshTopology::kMidBufferOffset + k;
        const float r = relax_[oi];
        dstMx_[mi] = srcMx_[mi] + dispMx_[mi] * r;
        dstMy_[mi] = srcMy_[mi] + dispMy_[mi] * r;
    }
}

float FaceMeshDeformer::softMaxSpan(
    const uint16_t* candidates, int count, float signDir,
    float midX, float midY, float ux, float uy, float beta) const {
    if (count <= 0) return 0.0f;
    float maxVal = 0.0f;
    for (int idx = 0; idx < count; ++idx) {
        const int i = candidates[idx];
        const float v = std::max(0.0f, signDir * ((srcMx_[i] - midX) * ux + (srcMy_[i] - midY) * uy));
        if (v > maxVal) maxVal = v;
    }
    float sumWeighted = 0.0f, sumExp = 0.0f;
    for (int idx = 0; idx < count; ++idx) {
        const int i = candidates[idx];
        const float v = std::max(0.0f, signDir * ((srcMx_[i] - midX) * ux + (srcMy_[i] - midY) * uy));
        const float w = std::exp(beta * (v - maxVal));
        sumWeighted += v * w;
        sumExp += w;
    }
    return (sumExp > 0.0f) ? (sumWeighted / sumExp) : 0.0f;
}

float FaceMeshDeformer::softMaxOvalExcess(
    int ovalSlot, float ox, float oy, float rx, float ry, float beta) const {
    const auto& tbl = tables();
    const int start = tbl.ovalNb2Offsets[ovalSlot];
    const int end = tbl.ovalNb2Offsets[ovalSlot + 1];
    if (end <= start) return 0.0f;
    float maxVal = 0.0f;
    for (int p = start; p < end; ++p) {
        const int v = tbl.ovalNb2Indices[p];
        const float ex = std::max(0.0f, (srcMx_[v] - ox) * rx + (srcMy_[v] - oy) * ry);
        if (ex > maxVal) maxVal = ex;
    }
    float sumWeighted = 0.0f, sumExp = 0.0f;
    for (int p = start; p < end; ++p) {
        const int v = tbl.ovalNb2Indices[p];
        const float ex = std::max(0.0f, (srcMx_[v] - ox) * rx + (srcMy_[v] - oy) * ry);
        const float w = std::exp(beta * (ex - maxVal));
        sumWeighted += ex * w;
        sumExp += w;
    }
    return (sumExp > 0.0f) ? (sumWeighted / sumExp) : 0.0f;
}

void FaceMeshDeformer::transformIris(
    const float* src, int baseIdx, int outer, int inner, int top, int bottom, float aspect) {
    const float oldCx = 0.25f * (src[outer * 2] + src[inner * 2] + src[top * 2] + src[bottom * 2]);
    const float oldCy = 0.25f * (src[outer * 2 + 1] + src[inner * 2 + 1] + src[top * 2 + 1] + src[bottom * 2 + 1]);
    const float newCx = 0.25f * (warpedLandmarks_[outer * 2] + warpedLandmarks_[inner * 2] +
                                 warpedLandmarks_[top * 2] + warpedLandmarks_[bottom * 2]);
    const float newCy = 0.25f * (warpedLandmarks_[outer * 2 + 1] + warpedLandmarks_[inner * 2 + 1] +
                                 warpedLandmarks_[top * 2 + 1] + warpedLandmarks_[bottom * 2 + 1]);
    const float oldW = std::max(std::hypot(src[outer * 2] - src[inner * 2], (src[outer * 2 + 1] - src[inner * 2 + 1]) / aspect), 1e-4f);
    const float newW = std::max(std::hypot(warpedLandmarks_[outer * 2] - warpedLandmarks_[inner * 2],
                                           (warpedLandmarks_[outer * 2 + 1] - warpedLandmarks_[inner * 2 + 1]) / aspect), 1e-4f);
    const float scale = detail::clampf(newW / oldW, 0.75f, 1.35f);
    for (int k = 0; k < 5; ++k) {
        const int idx = (baseIdx + k) * 2;
        warpedLandmarks_[idx] = newCx + (src[idx] - oldCx) * scale;
        warpedLandmarks_[idx + 1] = newCy + (src[idx + 1] - oldCy) * scale;
    }
}

Vec2 FaceMeshDeformer::eyeEnlargeDelta(
    float px, float py, float eyeX, float eyeY, float eyeW, float eyeH,
    float browDistV, float ux, float uy, float vx, float vy, float strength) {
    if (strength <= 0.0f) return {0.0f, 0.0f};
    const float dx = px - eyeX, dy = py - eyeY;
    const float eDu = dx * ux + dy * uy;
    const float eDv = dx * vx + dy * vy;
    const float rx = 1.05f * eyeW;
    const float ry = std::max(0.85f * eyeW, 1.6f * eyeH);
    const float r = std::hypot(eDu / rx, eDv / ry);
    if (r >= 1.0f) return {0.0f, 0.0f};
    const float t = 1.0f - r * r;
    float w = t * t;
    if (eDv > 0.0f && browDistV > 0.005f) {
        w *= (1.0f - detail::smoothstep(0.50f * browDistV, 0.92f * browDistV, eDv));
    }
    return {eDu * (strength * 0.28f * w), eDv * (strength * 0.36f * w)};
}

}  // namespace livebeauty::math
