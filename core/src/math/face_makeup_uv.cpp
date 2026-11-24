#include "livebeauty/math/face_makeup_uv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

#include "livebeauty/math/face_geometry.h"

namespace livebeauty::math {

namespace {

bool inRightEye(int v) {
    static const std::unordered_set<int> kSet = {
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    };
    return kSet.count(v) > 0;
}

bool inLeftEye(int v) {
    static const std::unordered_set<int> kSet = {
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    };
    return kSet.count(v) > 0;
}

bool inInnerLip(int v) {
    static const std::unordered_set<int> kSet = {
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    };
    return kSet.count(v) > 0;
}

bool inAegyoSalProtected(int v) {
    static const std::unordered_set<int> kSet = {
        110, 24, 23, 22, 26, 112, 243,
        339, 254, 253, 252, 256, 341, 463,
    };
    return kSet.count(v) > 0;
}

bool inNoseTipProtected(int v) {
    static const std::unordered_set<int> kSet = {1, 2, 4, 5, 19, 94, 168, 6, 197, 195};
    return kSet.count(v) > 0;
}

bool inLipProtected(int v) {
    static const std::unordered_set<int> kSet = [] {
        std::unordered_set<int> s = {
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        };
        for (uint16_t idx : FaceMeshTopology::kLipBandIndices) s.insert(static_cast<int>(idx));
        return s;
    }();
    return kSet.count(v) > 0;
}

struct DistNorm {
    float dist;
    float nx;
    float ny;
};

DistNorm polylineDistAndNormal(float u, float v, const Vec2* arc, int count) {
    float bestDist = std::numeric_limits<float>::max();
    float bestNx = 1.0f;
    float bestNy = 0.0f;
    for (int s = 0; s < count - 1; ++s) {
        const float ax = arc[s].x;
        const float ay = arc[s].y;
        const float abx = arc[s + 1].x - ax;
        const float aby = arc[s + 1].y - ay;
        const float len2 = std::max(abx * abx + aby * aby, 1e-8f);
        const float t = detail::clampf(((u - ax) * abx + (v - ay) * aby) / len2, 0.0f, 1.0f);
        const float px = ax + abx * t;
        const float py = ay + aby * t;
        const float d = std::hypot(u - px, v - py);
        if (d < bestDist) {
            bestDist = d;
            const float segLen = std::sqrt(len2);
            bestNx = -aby / segLen;
            bestNy = abx / segLen;
        }
    }
    return DistNorm{bestDist, bestNx, bestNy};
}

struct RetouchStaticTables {
    std::array<float, FaceMeshTopology::kFaceVertexCount> tearWeights{};
    std::array<float, FaceMeshTopology::kFaceVertexCount> foldWeights{};
    std::array<float, FaceMeshTopology::kFaceVertexCount * 2> foldNormals{};
    std::vector<uint16_t> skinRetouchIndices;

    RetouchStaticTables() {
        constexpr Vec2 kRightTearArc[5] = {
            {0.262f, 0.422f}, {0.298f, 0.440f}, {0.335f, 0.446f}, {0.372f, 0.436f}, {0.404f, 0.422f},
        };
        Vec2 leftTearArc[5];
        for (int i = 0; i < 5; ++i) leftTearArc[i] = {1.0f - kRightTearArc[i].x, kRightTearArc[i].y};

        constexpr Vec2 kRightFoldArc[5] = {
            {0.395f, 0.548f}, {0.378f, 0.574f}, {0.356f, 0.605f}, {0.337f, 0.646f}, {0.328f, 0.702f},
        };
        Vec2 leftFoldArc[5];
        for (int i = 0; i < 5; ++i) leftFoldArc[i] = {1.0f - kRightFoldArc[i].x, kRightFoldArc[i].y};

        const auto& uv = FaceMeshTopology::kCanonicalUv;
        for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
            const float u = uv[i * 2];
            const float v = uv[i * 2 + 1];
            if (!inRightEye(i) && !inLeftEye(i) && !inAegyoSalProtected(i)) {
                const float dr = polylineDistAndNormal(u, v, kRightTearArc, 5).dist;
                const float dl = polylineDistAndNormal(u, v, leftTearArc, 5).dist;
                const float d = std::min(dr, dl);
                constexpr float kRadius = 0.032f;
                if (d < kRadius && v >= 0.414f) {
                    const float t = 1.0f - (d / kRadius) * (d / kRadius);
                    const float lidGuard = detail::smoothstep(0.414f, 0.426f, v);
                    tearWeights[i] = t * t * lidGuard;
                }
            }
            if (!inLipProtected(i) && !inNoseTipProtected(i)) {
                const DistNorm dr = polylineDistAndNormal(u, v, kRightFoldArc, 5);
                const DistNorm dl = polylineDistAndNormal(u, v, leftFoldArc, 5);
                const DistNorm& best = (dr.dist <= dl.dist) ? dr : dl;
                constexpr float kRadius = 0.028f;
                if (best.dist < kRadius) {
                    const float t = 1.0f - (best.dist / kRadius) * (best.dist / kRadius);
                    foldWeights[i] = t * t;
                    foldNormals[i * 2] = best.nx;
                    foldNormals[i * 2 + 1] = best.ny;
                }
            }
        }

        const auto& src = FaceMakeupUv::skinFaceIndices();
        skinRetouchIndices.reserve(420);
        for (size_t t = 0; t + 2 < src.size(); t += 3) {
            const int a = src[t];
            const int b = src[t + 1];
            const int c = src[t + 2];
            if (std::max({tearWeights[a], tearWeights[b], tearWeights[c]}) > 0.002f ||
                std::max({foldWeights[a], foldWeights[b], foldWeights[c]}) > 0.002f) {
                skinRetouchIndices.push_back(src[t]);
                skinRetouchIndices.push_back(src[t + 1]);
                skinRetouchIndices.push_back(src[t + 2]);
            }
        }
    }
};

const RetouchStaticTables& retouchTables() {
    static const RetouchStaticTables kTables;
    return kTables;
}

}  // namespace

const std::vector<uint16_t>& FaceMakeupUv::skinFaceIndices() {
    static const std::vector<uint16_t> kIndices = [] {
        std::vector<uint16_t> kept;
        kept.reserve(852 * 3);
        const auto& src = FaceMeshTopology::kCanonicalIndices;
        for (size_t i = 0; i + 2 < src.size(); i += 3) {
            const int a = src[i];
            const int b = src[i + 1];
            const int c = src[i + 2];
            const bool inRe = inRightEye(a) && inRightEye(b) && inRightEye(c);
            const bool inLe = inLeftEye(a) && inLeftEye(b) && inLeftEye(c);
            const bool inM = inInnerLip(a) && inInnerLip(b) && inInnerLip(c);
            if (!inRe && !inLe && !inM) {
                kept.push_back(src[i]);
                kept.push_back(src[i + 1]);
                kept.push_back(src[i + 2]);
            }
        }
        return kept;
    }();
    return kIndices;
}

const std::vector<uint16_t>& FaceMakeupUv::innerMouthIndices() {
    static const std::vector<uint16_t> kIndices = [] {
        std::vector<uint16_t> kept;
        kept.reserve(18 * 3);
        const auto& src = FaceMeshTopology::kCanonicalIndices;
        for (size_t i = 0; i + 2 < src.size(); i += 3) {
            const int a = src[i];
            const int b = src[i + 1];
            const int c = src[i + 2];
            if (inInnerLip(a) && inInnerLip(b) && inInnerLip(c)) {
                kept.push_back(src[i]);
                kept.push_back(src[i + 1]);
                kept.push_back(src[i + 2]);
            }
        }
        return kept;
    }();
    return kIndices;
}

const std::vector<uint16_t>& FaceMakeupUv::eyeOpeningIndices() {
    static const std::vector<uint16_t> kIndices = [] {
        std::vector<uint16_t> out;
        out.reserve(FaceMeshTopology::kRightEyeIndexCount + FaceMeshTopology::kLeftEyeIndexCount);
        out.insert(out.end(), FaceMeshTopology::kRightEyeIndices.begin(), FaceMeshTopology::kRightEyeIndices.end());
        out.insert(out.end(), FaceMeshTopology::kLeftEyeIndices.begin(), FaceMeshTopology::kLeftEyeIndices.end());
        return out;
    }();
    return kIndices;
}

void FaceMakeupUv::packVertices(const float* landmarks, int coordCount, float* outVertices) {
    if (!landmarks || !outVertices || coordCount < FaceMeshTopology::kFaceVertexCount * 2) return;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float sx = landmarks[i * 2];
        const float sy = landmarks[i * 2 + 1];
        const int k = i * kVertexStrideFloats;
        outVertices[k] = sx * 2.0f - 1.0f;
        outVertices[k + 1] = sy * 2.0f - 1.0f;
        outVertices[k + 2] = sx;
        outVertices[k + 3] = sy;
        outVertices[k + 4] = uv[i * 2];
        outVertices[k + 5] = uv[i * 2 + 1];
    }
}

bool FaceMakeupUv::computeIrisCircle(
    const float* landmarks, int coordCount, int baseIdx, float aspect, float outCircle[4]) {
    if (!landmarks || !outCircle || coordCount < FaceMeshFrame::kCoordCount || !(aspect > 0.0f)) return false;
    const float cx = landmarks[baseIdx * 2];
    const float cy = landmarks[baseIdx * 2 + 1];
    float sumR = 0.0f;
    for (int k = 1; k <= 4; ++k) {
        const float px = landmarks[(baseIdx + k) * 2];
        const float py = landmarks[(baseIdx + k) * 2 + 1];
        sumR += std::hypot(px - cx, (py - cy) / aspect);
    }
    const float r = (sumR * 0.25f) * 1.08f;
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(r) || r < 0.002f || r > 0.25f) return false;
    outCircle[0] = cx;
    outCircle[1] = cy;
    outCircle[2] = r;
    outCircle[3] = aspect;
    return true;
}

Vec3 FaceMakeupUv::lipstickRgb(LipstickColor color) {
    switch (color) {
        case LipstickColor::kCoralPeach: return {0.76f, 0.27f, 0.17f};
        case LipstickColor::kCherryRed:  return {0.58f, 0.06f, 0.05f};
        case LipstickColor::kMilkTea:    return {0.56f, 0.29f, 0.18f};
        case LipstickColor::kBerryPlum:  return {0.46f, 0.08f, 0.16f};
        case LipstickColor::kOrangeSoda: return {0.74f, 0.21f, 0.05f};
        case LipstickColor::kRoseBean:
        default:                         return {0.60f, 0.23f, 0.19f};
    }
}

Vec3 FaceMakeupUv::blushRgb(BlushColor color) {
    switch (color) {
        case BlushColor::kRose:    return {0.88f, 0.44f, 0.54f};
        case BlushColor::kApricot: return {0.95f, 0.56f, 0.40f};
        case BlushColor::kBerry:   return {0.80f, 0.36f, 0.52f};
        case BlushColor::kPeach:
        default:                   return {0.96f, 0.54f, 0.58f};
    }
}

Vec4 FaceMakeupUv::blushChannel(BlushStyle style) {
    switch (style) {
        case BlushStyle::kDrunk:    return {0.0f, 1.0f, 0.0f, 0.0f};
        case BlushStyle::kUnderEye: return {0.0f, 0.0f, 1.0f, 0.0f};
        case BlushStyle::kApple:
        default:                    return {1.0f, 0.0f, 0.0f, 0.0f};
    }
}

Vec3 FaceMakeupUv::eyebrowRgb(EyebrowColor color) {
    switch (color) {
        case EyebrowColor::kSoftBrown: return {0.42f, 0.30f, 0.22f};
        case EyebrowColor::kGreyBlack: return {0.24f, 0.24f, 0.26f};
        case EyebrowColor::kDarkBrown:
        default:                       return {0.30f, 0.22f, 0.18f};
    }
}

Vec4 FaceMakeupUv::eyebrowChannel(EyebrowStyle style) {
    return (style == EyebrowStyle::kArched) ? Vec4{0.0f, 1.0f, 0.0f, 0.0f} : Vec4{1.0f, 0.0f, 0.0f, 0.0f};
}

Vec3 FaceMakeupUv::eyeshadowMainRgb(EyeshadowColor color) {
    switch (color) {
        case EyeshadowColor::kPeachPink: return {0.86f, 0.58f, 0.56f};
        case EyeshadowColor::kRoseWine:  return {0.74f, 0.42f, 0.48f};
        case EyeshadowColor::kChampagne: return {0.82f, 0.66f, 0.52f};
        case EyeshadowColor::kEarth:
        default:                         return {0.68f, 0.52f, 0.42f};
    }
}

Vec3 FaceMakeupUv::eyeshadowDeepRgb(EyeshadowColor color) {
    switch (color) {
        case EyeshadowColor::kPeachPink: return {0.66f, 0.36f, 0.38f};
        case EyeshadowColor::kRoseWine:  return {0.50f, 0.22f, 0.28f};
        case EyeshadowColor::kChampagne: return {0.58f, 0.42f, 0.30f};
        case EyeshadowColor::kEarth:
        default:                         return {0.44f, 0.32f, 0.25f};
    }
}

Vec2 FaceMakeupUv::irisTile(ContactLensStyle style) {
    switch (style) {
        case ContactLensStyle::kChocolate: return {0.5f, 0.0f};
        case ContactLensStyle::kMistGrey:  return {0.0f, 0.5f};
        case ContactLensStyle::kSakuraPink:return {0.5f, 0.5f};
        case ContactLensStyle::kHazel:
        default:                           return {0.0f, 0.0f};
    }
}

Vec3 FaceMakeupUv::hairDyeRgb(HairDyeColor color) {
    switch (color) {
        case HairDyeColor::kHazeBlue:       return {0.28f, 0.42f, 0.66f};
        case HairDyeColor::kAshPurple:      return {0.54f, 0.40f, 0.65f};
        case HairDyeColor::kRoseGold:       return {0.78f, 0.46f, 0.44f};
        case HairDyeColor::kCoolBlack:      return {0.20f, 0.23f, 0.30f};
        case HairDyeColor::kCaramelCopper:  return {0.74f, 0.38f, 0.20f};
        case HairDyeColor::kMilkTeaBrown:
        default:                            return {0.62f, 0.44f, 0.31f};
    }
}

const std::array<float, FaceMeshTopology::kFaceVertexCount>& FaceRetouchSubMesh::staticTearWeights() {
    return retouchTables().tearWeights;
}

const std::array<float, FaceMeshTopology::kFaceVertexCount>& FaceRetouchSubMesh::staticFoldWeights() {
    return retouchTables().foldWeights;
}

const std::array<float, FaceMeshTopology::kFaceVertexCount * 2>& FaceRetouchSubMesh::staticFoldNormals() {
    return retouchTables().foldNormals;
}

const std::vector<uint16_t>& FaceRetouchSubMesh::skinRetouchIndices() {
    return retouchTables().skinRetouchIndices;
}

void FaceRetouchSubMesh::packVertices(
    const float* landmarks,
    const float* meshVertices4,
    float ux, float uy, float vx, float vy,
    float faceWidth, float faceHeight, float aspect,
    float mirrorSign, float dc, float nf,
    float* outVertices) {
    if (!landmarks || !outVertices) return;
    const bool needSkinOffsets = (dc > 1e-4f || nf > 1e-4f);
    const float tearStepM = 0.042f * faceHeight;
    const float tearOffU = -vx * tearStepM;
    const float tearOffV = (-vy * tearStepM) * aspect;
    const float foldSpanM = 0.028f * faceWidth;

    const auto& tearW = staticTearWeights();
    const auto& foldW = staticFoldWeights();
    const auto& foldN = staticFoldNormals();

    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const int k = i * kStrideFloats;
        if (meshVertices4 != nullptr) {
            const int mk = i * 4;
            outVertices[k] = meshVertices4[mk];
            outVertices[k + 1] = meshVertices4[mk + 1];
            outVertices[k + 2] = meshVertices4[mk + 2];
            outVertices[k + 3] = meshVertices4[mk + 3];
        } else {
            const float sx = landmarks[i * 2];
            const float sy = landmarks[i * 2 + 1];
            outVertices[k] = sx * 2.0f - 1.0f;
            outVertices[k + 1] = sy * 2.0f - 1.0f;
            outVertices[k + 2] = sx;
            outVertices[k + 3] = sy;
        }
        const float wTear = tearW[i];
        const float wFold = foldW[i];
        outVertices[k + 4] = wTear;
        outVertices[k + 5] = wFold;
        if (!needSkinOffsets || (wTear * dc <= 0.0f && wFold * nf <= 0.0f)) {
            outVertices[k + 6] = 0.0f;
            outVertices[k + 7] = 0.0f;
        } else if (wTear * dc >= wFold * nf) {
            outVertices[k + 6] = tearOffU;
            outVertices[k + 7] = tearOffV;
        } else {
            const float cNu = foldN[i * 2] * mirrorSign;
            const float cNv = -foldN[i * 2 + 1];
            const float dmX = (cNu * ux + cNv * vx) * foldSpanM;
            const float dmY = (cNu * uy + cNv * vy) * foldSpanM;
            outVertices[k + 6] = dmX;
            outVertices[k + 7] = dmY * aspect;
        }
    }
}

}  // namespace livebeauty::math
