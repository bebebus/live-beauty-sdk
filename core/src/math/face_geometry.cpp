#include "livebeauty/math/face_geometry.h"

#include <algorithm>
#include <cmath>

namespace livebeauty::math {

namespace {

constexpr int kRequiredCore[] = {
    1, 132, 361, 172, 397, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14,
};
constexpr int kRequiredExt[] = {234, 454, 10, 152, 0, 17, 129, 358};
constexpr int kWarpPoints[] = {132, 361, 172, 397, 33, 133, 362, 263, 1};
constexpr int kMaskPoints[] = {234, 454, 10, 152, 61, 291, 0, 17};

inline bool validPoint(const float* mpXy, int idx, float lo, float hi) {
    const float x = mpXy[idx * 2];
    const float y = mpXy[idx * 2 + 1];
    return std::isfinite(x) && std::isfinite(y) && x >= lo && x <= hi && y >= lo && y <= hi;
}

}  // namespace

void FaceMeshFrame::synthesizeIris(
    float* landmarks, int baseIdx, int outer, int inner, int top, int bottom, float aspect) {
    const float cx = 0.25f * (landmarks[outer * 2] + landmarks[inner * 2] +
                              landmarks[top * 2] + landmarks[bottom * 2]);
    const float cy = 0.25f * (landmarks[outer * 2 + 1] + landmarks[inner * 2 + 1] +
                              landmarks[top * 2 + 1] + landmarks[bottom * 2 + 1]);
    const float r = std::hypot(landmarks[outer * 2] - landmarks[inner * 2],
                               (landmarks[outer * 2 + 1] - landmarks[inner * 2 + 1]) / aspect) * 0.22f;
    landmarks[baseIdx * 2] = cx;
    landmarks[baseIdx * 2 + 1] = cy;
    landmarks[(baseIdx + 1) * 2] = cx + r;
    landmarks[(baseIdx + 1) * 2 + 1] = cy;
    landmarks[(baseIdx + 2) * 2] = cx;
    landmarks[(baseIdx + 2) * 2 + 1] = cy + r * aspect;
    landmarks[(baseIdx + 3) * 2] = cx - r;
    landmarks[(baseIdx + 3) * 2 + 1] = cy;
    landmarks[(baseIdx + 4) * 2] = cx;
    landmarks[(baseIdx + 4) * 2 + 1] = cy - r * aspect;
}

bool FaceMeshFrame::fillMeta(const float* landmarks, int coordCount, float aspect, float* outMeta) {
    if (!landmarks || !outMeta || coordCount < kFaceCount * 2 || !std::isfinite(aspect) || aspect <= 0.0f) {
        return false;
    }
    const double ratio = static_cast<double>(aspect);
    auto mx = [&](int i) { return static_cast<double>(landmarks[i * 2]); };
    auto my = [&](int i) { return static_cast<double>(landmarks[i * 2 + 1]) / ratio; };
    auto dist = [&](int a, int b) { return std::hypot(mx(a) - mx(b), my(a) - my(b)); };

    const double span = dist(132, 361);
    if (span < 0.08 || span > 0.90) return false;

    const double acrossX = mx(454) - mx(234);
    const double acrossY = my(454) - my(234);
    const double faceWidth = std::hypot(acrossX, acrossY);
    if (faceWidth < 0.05) return false;
    double ux = acrossX / faceWidth;
    double uy = acrossY / faceWidth;
    const double f10x = mx(10) - mx(152);
    const double f10y = my(10) - my(152);
    if (-uy * f10x + ux * f10y < 0.0) {
        ux = -ux;
        uy = -uy;
    }
    if (!std::isfinite(ux) || !std::isfinite(uy) || std::fabs(std::hypot(ux, uy) - 1.0) >= 1e-5) {
        return false;
    }
    const double vx = -uy;
    const double vy = ux;
    const double faceHeight = std::fabs(f10x * vx + f10y * vy);
    if (faceHeight < 0.05) return false;

    const double cheekMidX = 0.5 * (mx(132) + mx(361));
    const double cheekMidY = 0.5 * (my(132) + my(361));
    const double signedYaw = ((mx(1) - cheekMidX) * ux + (my(1) - cheekMidY) * uy) / span;
    if (std::fabs(signedYaw) > kMaxYawRatio) return false;

    const double rightWeight = 1.0 - detail::smoothstepd(0.36, 0.66, signedYaw);
    const double leftWeight = 1.0 - detail::smoothstepd(0.36, 0.66, -signedYaw);
    const double yawWeight = 1.0 - detail::smoothstepd(0.45, kMaxYawRatio, std::fabs(signedYaw));

    const double rightEyeW = dist(33, 133);
    const double leftEyeW = dist(362, 263);
    if (rightEyeW < 0.012 || leftEyeW < 0.012 || rightEyeW > span * 0.55 || leftEyeW > span * 0.55) {
        return false;
    }
    const double rightEyeOpen = detail::clampd(dist(159, 145) / rightEyeW, 0.0, 1.0);
    const double leftEyeOpen = detail::clampd(dist(386, 374) / leftEyeW, 0.0, 1.0);
    const double mouthW = std::max(dist(61, 291), 1e-4);
    const double mouthOpen = detail::clampd(dist(13, 14) / mouthW, 0.0, 1.5);

    const double sideMidX = 0.5 * (mx(234) + mx(454));
    const double sideMidY = 0.5 * (my(234) + my(454));
    const double lineMidX = 0.5 * (mx(10) + mx(152));
    const double lineMidY = 0.5 * (my(10) + my(152));
    const double cu = 0.5 * (sideMidX * ux + sideMidY * uy + lineMidX * ux + lineMidY * uy);
    const double cv = lineMidX * vx + lineMidY * vy;

    outMeta[kMetaCenterX] = static_cast<float>(cu * ux + cv * vx);
    outMeta[kMetaCenterY] = static_cast<float>((cu * uy + cv * vy) * ratio);
    outMeta[kMetaAxisX] = static_cast<float>(ux);
    outMeta[kMetaAxisY] = static_cast<float>(uy);
    outMeta[kMetaFaceWidth] = static_cast<float>(faceWidth);
    outMeta[kMetaFaceHeight] = static_cast<float>(faceHeight);
    outMeta[kMetaSignedYaw] = static_cast<float>(signedYaw);
    outMeta[kMetaLeftWeight] = static_cast<float>(leftWeight);
    outMeta[kMetaRightWeight] = static_cast<float>(rightWeight);
    outMeta[kMetaYawWeight] = static_cast<float>(yawWeight);
    outMeta[kMetaRightEyeOpen] = static_cast<float>(rightEyeOpen);
    outMeta[kMetaLeftEyeOpen] = static_cast<float>(leftEyeOpen);
    outMeta[kMetaMouthOpen] = static_cast<float>(mouthOpen);
    outMeta[kMetaRightIrisGate] = 1.0f;
    outMeta[kMetaLeftIrisGate] = 1.0f;
    return true;
}

bool FaceMeshFrame::fromMediaPipe(const float* mpXy, int pointCount, float aspect, FaceMeshFrame& out) {
    if (!mpXy || pointCount < kFaceCount || !std::isfinite(aspect) || aspect <= 0.0f) return false;
    for (int idx : kRequiredCore) {
        if (!validPoint(mpXy, idx, -0.05f, 1.05f)) return false;
    }
    for (int idx : kRequiredExt) {
        if (!validPoint(mpXy, idx, -0.30f, 1.30f)) return false;
    }
    for (int i = 0; i < kFaceCount; ++i) {
        if (!validPoint(mpXy, i, -0.5f, 1.5f)) return false;
    }
    for (int i = 0; i < kFaceCount; ++i) {
        out.landmarks[i * 2] = mpXy[i * 2];
        out.landmarks[i * 2 + 1] = 1.0f - mpXy[i * 2 + 1];
    }
    if (!fillMeta(out.landmarks.data(), kCoordCount, aspect, out.meta.data())) return false;

    bool hasIris = (pointCount >= kLandmarkCount);
    if (hasIris) {
        for (int i = kFaceCount; i < kLandmarkCount; ++i) {
            if (!validPoint(mpXy, i, -0.25f, 1.25f)) {
                hasIris = false;
                break;
            }
            out.landmarks[i * 2] = mpXy[i * 2];
            out.landmarks[i * 2 + 1] = 1.0f - mpXy[i * 2 + 1];
        }
    }
    if (!hasIris) {
        synthesizeIris(out.landmarks.data(), 468, 33, 133, 159, 145, aspect);
        synthesizeIris(out.landmarks.data(), 473, 362, 263, 386, 374, aspect);
    }
    out.hasIris = hasIris;
    return true;
}

bool FaceMeshFrame::isFrontFaceCw(const float* landmarks, int coordCount) {
    if (!landmarks || coordCount < kFaceCount * 2) return false;
    const float ax = landmarks[454 * 2] - landmarks[234 * 2];
    const float ay = landmarks[454 * 2 + 1] - landmarks[234 * 2 + 1];
    const float bx = landmarks[10 * 2] - landmarks[152 * 2];
    const float by = landmarks[10 * 2 + 1] - landmarks[152 * 2 + 1];
    return (ax * by - ay * bx) < 0.0f;
}

bool FaceMeshFrame::fillGeometryState(
    const float* landmarks, int coordCount,
    float axisX, float axisY, float aspect, float warpWeight,
    float* outState) {
    if (!landmarks || !outState || coordCount < kFaceCount * 2 || !(aspect > 0.0f)) return false;
    const float n = std::hypot(axisX, axisY);
    if (n < 1e-4f) return false;
    const float ux = axisX / n;
    const float uy = axisY / n;
    const float vx = -uy;
    const float vy = ux;

    auto lx = [&](int i) { return landmarks[i * 2]; };
    auto ly = [&](int i) { return landmarks[i * 2 + 1]; };
    auto mx = [&](int i) { return lx(i); };
    auto my = [&](int i) { return ly(i) / aspect; };

    constexpr int pairs[4] = {132, 361, 172, 397};
    for (int p = 0; p < 2; ++p) {
        const int a = pairs[p * 2];
        const int b = pairs[p * 2 + 1];
        const float midX = 0.5f * (lx(a) + lx(b));
        const float midY = 0.5f * (ly(a) + ly(b));
        const int k0 = p * 8;
        const int k1 = k0 + 4;
        outState[k0] = lx(a);
        outState[k0 + 1] = ly(a);
        outState[k0 + 2] = midX;
        outState[k0 + 3] = midY;
        outState[k1] = lx(b);
        outState[k1 + 1] = ly(b);
        outState[k1 + 2] = midX;
        outState[k1 + 3] = midY;
    }

    constexpr int eyePairs[4] = {33, 133, 362, 263};
    for (int i = 0; i < 2; ++i) {
        const int a = eyePairs[i * 2];
        const int b = eyePairs[i * 2 + 1];
        const float ew = std::hypot(mx(a) - mx(b), my(a) - my(b));
        if (ew < 0.005f) return false;
        outState[16 + i * 3] = 0.5f * (lx(a) + lx(b));
        outState[17 + i * 3] = 0.5f * (ly(a) + ly(b));
        outState[18 + i * 3] = ew * 0.9f;
    }

    auto centre = [&](int a, int b, int c, int d, int dest) {
        const float sx = 0.5f * (mx(a) + mx(b));
        const float sy = 0.5f * (my(a) + my(b));
        const float tx = 0.5f * (mx(c) + mx(d));
        const float ty = 0.5f * (my(c) + my(d));
        const float cu = 0.5f * (sx * ux + sy * uy + tx * ux + ty * uy);
        const float cv = tx * vx + ty * vy;
        outState[dest] = cu * ux + cv * vx;
        outState[dest + 1] = (cu * uy + cv * vy) * aspect;
    };

    const float fw = std::hypot(mx(454) - mx(234), my(454) - my(234));
    const float fh = std::fabs((mx(10) - mx(152)) * vx + (my(10) - my(152)) * vy);
    if (fw < 0.03f || fh < 0.03f) return false;
    centre(234, 454, 10, 152, 22);
    outState[24] = 0.5f * fw * 1.12f;
    outState[25] = 0.5f * fh * 1.15f;
    outState[26] = ux;
    outState[27] = uy;

    centre(61, 291, 0, 17, 28);
    const float mouthRx = 0.5f * std::fabs((mx(61) - mx(291)) * ux + (my(61) - my(291)) * uy) * 1.20f;
    if (mouthRx < 0.004f) return false;
    outState[30] = mouthRx;
    outState[31] = std::max(0.5f * std::fabs((mx(0) - mx(17)) * vx + (my(0) - my(17)) * vy) * 1.25f, 0.25f * mouthRx);

    for (int i = 0; i < 2; ++i) {
        const int a = eyePairs[i * 2];
        const int b = eyePairs[i * 2 + 1];
        const float ew = std::hypot(mx(a) - mx(b), my(a) - my(b));
        const float cxm = 0.5f * (mx(a) + mx(b)) + 0.16f * ew * vx;
        const float cym = 0.5f * (my(a) + my(b)) + 0.16f * ew * vy;
        outState[32 + i * 4] = cxm;
        outState[33 + i * 4] = cym * aspect;
        outState[34 + i * 4] = 0.78f * ew;
        outState[35 + i * 4] = 0.52f * ew;
    }
    outState[40] = warpWeight;
    return true;
}

bool FaceMeshFrame::faceGeometryFromMediaPipe(const float* mpXy, int pointCount, float aspect, float* outState) {
    if (!mpXy || !outState || pointCount < kFaceCount || !std::isfinite(aspect) || aspect <= 0.0f) return false;
    for (int idx : kWarpPoints) {
        if (!validPoint(mpXy, idx, 0.0f, 1.0f)) return false;
    }
    for (int idx : kMaskPoints) {
        if (!validPoint(mpXy, idx, -0.25f, 1.25f)) return false;
    }
    const double ratio = static_cast<double>(aspect);
    auto px = [&](int i) { return static_cast<double>(mpXy[i * 2]); };
    auto py = [&](int i) { return 1.0 - static_cast<double>(mpXy[i * 2 + 1]); };
    auto dist = [&](int a, int b) { return std::hypot(px(a) - px(b), (py(a) - py(b)) / ratio); };

    const double span = dist(132, 361);
    if (span < 0.1 || span > 0.85) return false;

    const double centreX = 0.5 * (px(132) + px(361));
    const double centreY = 0.5 * (py(132) + py(361));
    const double yaw = std::fabs((px(1) - centreX) * (px(361) - px(132)) +
                                 (py(1) - centreY) * (py(361) - py(132)) / (ratio * ratio)) / span;
    const double r = yaw / span;
    if (r > 0.45) return false;
    const double warpWeight = 1.0 - detail::smoothstepd(0.24, 0.30, r);

    constexpr int pairs[4] = {132, 361, 172, 397};
    for (int p = 0; p < 2; ++p) {
        const int a = pairs[p * 2];
        const int b = pairs[p * 2 + 1];
        const double midX = 0.5 * (px(a) + px(b));
        const double midY = 0.5 * (py(a) + py(b));
        const int k0 = p * 8;
        const int k1 = k0 + 4;
        outState[k0] = static_cast<float>(px(a));
        outState[k0 + 1] = static_cast<float>(py(a));
        outState[k0 + 2] = static_cast<float>(midX);
        outState[k0 + 3] = static_cast<float>(midY);
        outState[k1] = static_cast<float>(px(b));
        outState[k1 + 1] = static_cast<float>(py(b));
        outState[k1 + 2] = static_cast<float>(midX);
        outState[k1 + 3] = static_cast<float>(midY);
    }

    constexpr int eyePairs[4] = {33, 133, 362, 263};
    for (int i = 0; i < 2; ++i) {
        const int a = eyePairs[i * 2];
        const int b = eyePairs[i * 2 + 1];
        const double w = dist(a, b);
        if (w < 0.015 || w > span * 0.5) return false;
        outState[16 + i * 3] = static_cast<float>(0.5 * (px(a) + px(b)));
        outState[17 + i * 3] = static_cast<float>(0.5 * (py(a) + py(b)));
        outState[18 + i * 3] = static_cast<float>(w * 0.9);
    }

    auto mx = [&](int i) { return px(i); };
    auto my = [&](int i) { return py(i) / ratio; };
    const double acrossX = mx(454) - mx(234);
    const double acrossY = my(454) - my(234);
    const double fw = std::hypot(acrossX, acrossY);
    if (fw < 0.05) return false;
    double ux = acrossX / fw;
    double uy = acrossY / fw;
    const double f10x = mx(10) - mx(152);
    const double f10y = my(10) - my(152);
    if (-uy * f10x + ux * f10y < 0.0) {
        ux = -ux;
        uy = -uy;
    }
    if (!std::isfinite(ux) || !std::isfinite(uy) || std::fabs(std::hypot(ux, uy) - 1.0) >= 1e-6) return false;
    const double vx = -uy;
    const double vy = ux;

    auto centre = [&](int a, int b, int c, int d, int dest) {
        const double sx = 0.5 * (mx(a) + mx(b));
        const double sy = 0.5 * (my(a) + my(b));
        const double tx = 0.5 * (mx(c) + mx(d));
        const double ty = 0.5 * (my(c) + my(d));
        const double cu = 0.5 * (sx * ux + sy * uy + tx * ux + ty * uy);
        const double cv = tx * vx + ty * vy;
        outState[dest] = static_cast<float>(cu * ux + cv * vx);
        outState[dest + 1] = static_cast<float>((cu * uy + cv * vy) * ratio);
    };

    const double ovalRx = 0.5 * fw * 1.12;
    const double ovalRy = 0.5 * std::fabs(f10x * vx + f10y * vy) * 1.15;
    if (ovalRx < 0.03 || ovalRy < 0.03) return false;
    centre(234, 454, 10, 152, 22);
    outState[24] = static_cast<float>(ovalRx);
    outState[25] = static_cast<float>(ovalRy);
    outState[26] = static_cast<float>(ux);
    outState[27] = static_cast<float>(uy);

    centre(61, 291, 0, 17, 28);
    const double mouthRx = 0.5 * std::fabs((mx(61) - mx(291)) * ux + (my(61) - my(291)) * uy) * 1.20;
    if (mouthRx < 0.005) return false;
    outState[30] = static_cast<float>(mouthRx);
    outState[31] = static_cast<float>(std::max(0.5 * std::fabs((mx(0) - mx(17)) * vx + (my(0) - my(17)) * vy) * 1.25, 0.25 * mouthRx));

    for (int i = 0; i < 2; ++i) {
        const int a = eyePairs[i * 2];
        const int b = eyePairs[i * 2 + 1];
        const double w = std::hypot(mx(a) - mx(b), my(a) - my(b));
        const double cx = 0.5 * (mx(a) + mx(b));
        const double cy = 0.5 * (my(a) + my(b));
        outState[32 + i * 4] = static_cast<float>(cx + 0.04 * w * vx);
        outState[33 + i * 4] = static_cast<float>((cy + 0.04 * w * vy) * ratio);
        outState[34 + i * 4] = static_cast<float>(0.72 * w);
        outState[35 + i * 4] = static_cast<float>(0.36 * w);
    }
    outState[40] = static_cast<float>(warpWeight);
    return true;
}

}  // namespace livebeauty::math
