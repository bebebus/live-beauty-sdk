#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "livebeauty/bundle/effect_bundle.h"
#include "livebeauty/config.h"
#include "livebeauty/math/ar_3d_accessory.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/device_tier.h"
#include "livebeauty/math/face_aux_vision.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_parsing.h"
#include "livebeauty/math/face_makeup_uv.h"
#include "livebeauty/math/face_mesh_deformer.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/math/face_relight.h"
#include "livebeauty/math/face_roi_selector.h"
#include "livebeauty/math/face_timeline.h"
#include "livebeauty/math/hair_mask_tuning.h"
#include "livebeauty/math/multi_face_tracker.h"
#include "livebeauty/math/sticker_transform.h"

using namespace livebeauty;
using namespace livebeauty::math;

namespace {

int gFailures = 0;

#define EXPECT_TRUE(cond)                                                                      \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " (" << #cond << ")\n";    \
            ++gFailures;                                                                       \
        }                                                                                      \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                                                 \
    do {                                                                                       \
        const double _va = static_cast<double>(a);                                             \
        const double _vb = static_cast<double>(b);                                             \
        if (std::fabs(_va - _vb) > (eps)) {                                                    \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " (" << #a << "=" << _va    \
                      << " vs " << #b << "=" << _vb << ", eps=" << (eps) << ")\n";             \
            ++gFailures;                                                                       \
        }                                                                                      \
    } while (0)

// Build synthetic frontal or yaw-rotated 478-pt GL landmarks from canonical 3D mesh.
void makeCanonicalFace(float aspect, float cx, float cy, float faceWidth, float yawRad,
                       float shiftX, float shiftY, std::array<float, FaceMeshFrame::kCoordCount>& out) {
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float canonFw = FaceMeshFilter::canonicalFaceWidth();
    const float scale = faceWidth / canonFw;
    const float c = std::cos(yawRad);
    const float s = std::sin(yawRad);
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float x = xyz[i * 3];
        const float y = xyz[i * 3 + 1];
        const float z = xyz[i * 3 + 2];
        const float rx = x * c + z * s;
        const float mx = cx + shiftX + rx * scale;
        const float my = cy / aspect + shiftY + y * scale;
        out[i * 2] = mx;
        out[i * 2 + 1] = my * aspect;
    }
    FaceMeshFrame::synthesizeIris(out.data(), 468, 33, 133, 159, 145, aspect);
    FaceMeshFrame::synthesizeIris(out.data(), 473, 362, 263, 386, 374, aspect);
}

void testTopologyAndSubMeshes() {
    EXPECT_TRUE(FaceMeshTopology::kFaceVertexCount == 468);
    EXPECT_TRUE(FaceMeshTopology::kTotalVertexCount == 548);
    EXPECT_TRUE(FaceMeshTopology::kTotalTriangleCount == 1086);
    EXPECT_TRUE(FaceMeshTopology::kFullIndices.size() == 1086 * 3);
    EXPECT_TRUE(FaceMakeupUv::skinFaceIndices().size() == 852 * 3);
    EXPECT_TRUE(FaceMakeupUv::innerMouthIndices().size() == 18 * 3);
    EXPECT_TRUE(FaceMakeupUv::eyeOpeningIndices().size() == 28 * 3);
    EXPECT_TRUE(!FaceRetouchSubMesh::skinRetouchIndices().empty());

    // Aegyo-sal protected vertices must have zero tear-trough weight.
    const auto& tearW = FaceRetouchSubMesh::staticTearWeights();
    for (int v : {110, 24, 23, 22, 26, 112, 243, 339, 254, 253, 252, 256, 341, 463}) {
        EXPECT_NEAR(tearW[v], 0.0f, 1e-6f);
    }

    // Calibrated lipstick hues (cherry_red pure red G >= B, rose_bean warm R > G > B).
    const Vec3 cherry = FaceMakeupUv::lipstickRgb(LipstickColor::kCherryRed);
    EXPECT_TRUE(cherry.x > 0.50f && cherry.y >= cherry.z);
    const Vec3 rose = FaceMakeupUv::lipstickRgb(LipstickColor::kRoseBean);
    EXPECT_TRUE(rose.x > rose.y && rose.y > rose.z);
}

void testDeformerInvariantsAndCranialTop() {
    constexpr float kAspect = 9.0f / 16.0f;
    std::array<float, FaceMeshFrame::kCoordCount> lm{};
    makeCanonicalFace(kAspect, 0.5f, 0.5f, 0.42f, 0.0f, 0.0f, 0.0f, lm);

    std::array<float, FaceMeshFrame::kMetaSize> meta{};
    EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, meta.data()));

    // 1. Compare cranialTop = 0 vs cranialTop = 1: jaw outer ring stays identical, crown puffs up, zero triangle flips.
    FaceMeshDeformer def0;
    FaceMeshDeformer::Params p0;
    p0.faceSlim = 0.8f;
    p0.vFace = 0.7f;
    p0.eyeEnlarge = 0.6f;
    EXPECT_TRUE(def0.deform(
        lm.data(), kAspect, meta[FaceMeshFrame::kMetaAxisX], meta[FaceMeshFrame::kMetaAxisY],
        meta[FaceMeshFrame::kMetaFaceWidth], meta[FaceMeshFrame::kMetaFaceHeight],
        meta[FaceMeshFrame::kMetaLeftWeight], meta[FaceMeshFrame::kMetaRightWeight],
        meta[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, p0));

    FaceMeshDeformer def1;
    FaceMeshDeformer::Params p1 = p0;
    p1.cranialTop = 1.0f;
    EXPECT_TRUE(def1.deform(
        lm.data(), kAspect, meta[FaceMeshFrame::kMetaAxisX], meta[FaceMeshFrame::kMetaAxisY],
        meta[FaceMeshFrame::kMetaFaceWidth], meta[FaceMeshFrame::kMetaFaceHeight],
        meta[FaceMeshFrame::kMetaLeftWeight], meta[FaceMeshFrame::kMetaRightWeight],
        meta[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, p1));

    const float* v0 = def0.vertexData();
    const float* v1 = def1.vertexData();

    // Outer anchor ring (504..547) must have strictly zero displacement (dst == src).
    for (int i = FaceMeshTopology::kBufferOffset; i < FaceMeshTopology::kTotalVertexCount; ++i) {
        const float dstX = (v1[i * 4] + 1.0f) * 0.5f;
        const float dstY = (v1[i * 4 + 1] + 1.0f) * 0.5f;
        EXPECT_NEAR(dstX, v1[i * 4 + 2], 1e-5f);
        EXPECT_NEAR(dstY, v1[i * 4 + 3], 1e-5f);
    }

    // Jaw segment of outer anchor ring (oval slots 12..24) must be bit-identical when cranialTop toggles 0 -> 1.
    for (int k = 12; k <= 24; ++k) {
        const int bi = FaceMeshTopology::kBufferOffset + k;
        EXPECT_NEAR(v0[bi * 4 + 2], v1[bi * 4 + 2], 1e-6f);
        EXPECT_NEAR(v0[bi * 4 + 3], v1[bi * 4 + 3], 1e-6f);
    }

    // Crown mid-ring vertex (slot 0, forehead top 10) must lift upward with cranialTop = 1.
    const int crownMid = FaceMeshTopology::kMidBufferOffset + 0;
    const float crownLiftY = ((v1[crownMid * 4 + 1] + 1.0f) * 0.5f) - v1[crownMid * 4 + 3];
    EXPECT_TRUE(crownLiftY > 0.01f);

    // Pure faceSlim must gently shorten vertical face span (10..152) proportionally to jaw narrowing
    // so slimming never creates an elongated-head illusion.
    FaceMeshDeformer defSlim;
    FaceMeshDeformer::Params pSlim;
    pSlim.faceSlim = 1.0f;
    EXPECT_TRUE(defSlim.deform(
        lm.data(), kAspect, meta[FaceMeshFrame::kMetaAxisX], meta[FaceMeshFrame::kMetaAxisY],
        meta[FaceMeshFrame::kMetaFaceWidth], meta[FaceMeshFrame::kMetaFaceHeight],
        meta[FaceMeshFrame::kMetaLeftWeight], meta[FaceMeshFrame::kMetaRightWeight],
        meta[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, pSlim));
    const float* wl = defSlim.warpedLandmarks();
    const float origH = std::fabs(lm[10 * 2 + 1] - lm[152 * 2 + 1]) / kAspect;
    const float newH = std::fabs(wl[10 * 2 + 1] - wl[152 * 2 + 1]) / kAspect;
    const float origJaw = std::fabs(lm[397 * 2] - lm[172 * 2]);
    const float newJaw = std::fabs(wl[397 * 2] - wl[172 * 2]);
    EXPECT_TRUE(newJaw < origJaw - 0.01f);
    // v3.9.4 (A6/shape-2): faceSlim narrows jaw cleanly without uniformly shrinking the face or pulling hairline down.
    EXPECT_TRUE(newH < origH);
    EXPECT_TRUE(newH > origH * 0.94f);
    EXPECT_TRUE(std::fabs(wl[10 * 2 + 1] - lm[10 * 2 + 1]) < 1e-4f);

    // Forehead bare hairline vertex 10 must NOT be pulled upward by cranialTop.
    EXPECT_TRUE(def1.warpedLandmarks()[10 * 2 + 1] <= def0.warpedLandmarks()[10 * 2 + 1] + 1e-5f);

    // All 1086 triangles must preserve positive signed area under full 15-slider load and 24 deg yaw.
    for (float yawDeg : {0.0f, 24.0f, -24.0f}) {
        std::array<float, FaceMeshFrame::kCoordCount> lmYaw{};
        makeCanonicalFace(kAspect, 0.5f, 0.5f, 0.42f, yawDeg * kPi / 180.0f, 0.0f, 0.0f, lmYaw);
        std::array<float, FaceMeshFrame::kMetaSize> mYaw{};
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lmYaw.data(), FaceMeshFrame::kCoordCount, kAspect, mYaw.data()));

        FaceMeshDeformer defMax;
        FaceMeshDeformer::Params pMax{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        EXPECT_TRUE(defMax.deform(
            lmYaw.data(), kAspect, mYaw[FaceMeshFrame::kMetaAxisX], mYaw[FaceMeshFrame::kMetaAxisY],
            mYaw[FaceMeshFrame::kMetaFaceWidth], mYaw[FaceMeshFrame::kMetaFaceHeight],
            mYaw[FaceMeshFrame::kMetaLeftWeight], mYaw[FaceMeshFrame::kMetaRightWeight],
            mYaw[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, pMax));

        const float* vm = defMax.vertexData();
        const auto& idx = FaceMeshTopology::kFullIndices;
        int flippedFront = 0;
        for (int t = 0; t < FaceMeshTopology::kTotalTriangleCount; ++t) {
            const int a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
            const float sx1 = vm[b * 4 + 2] - vm[a * 4 + 2], sy1 = vm[b * 4 + 3] - vm[a * 4 + 3];
            const float sx2 = vm[c * 4 + 2] - vm[a * 4 + 2], sy2 = vm[c * 4 + 3] - vm[a * 4 + 3];
            const float dx1 = vm[b * 4] - vm[a * 4], dy1 = vm[b * 4 + 1] - vm[a * 4 + 1];
            const float dx2 = vm[c * 4] - vm[a * 4], dy2 = vm[c * 4 + 1] - vm[a * 4 + 1];
            const float sArea = sx1 * sy2 - sy1 * sx2;
            const float dArea = dx1 * dy2 - dy1 * dx2;
            if (sArea > 1e-6f && dArea <= 0.0f) ++flippedFront;
        }
        EXPECT_TRUE(flippedFront == 0);
    }
}

void testFaceTimelineAndEyeMakeupKeep() {
    constexpr float kAspect = 9.0f / 16.0f;
    FaceTimeline tl;
    tl.setAspect(kAspect);

    std::array<float, FaceMeshFrame::kCoordCount> lm{};
    std::array<float, FaceGeometryConstants::kStateSize> state{};
    FaceMeshFrame frame{};

    // Still phase: eyeMakeupKeep must stay at 1.0.
    for (int f = 0; f < 10; ++f) {
        const int64_t ts = 1000 + f * 33;
        makeCanonicalFace(kAspect, 0.5f, 0.5f, 0.40f, 0.0f, 0.0f, 0.0f, lm);
        frame.landmarks = lm;
        frame.hasIris = true;
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, state.data()));
        tl.add(ts, state.data(), &frame);
        tl.update(ts);
        EXPECT_TRUE(tl.at(ts) != nullptr);
    }
    EXPECT_NEAR(tl.eyeMakeupKeep(), 1.0f, 1e-3f);
    EXPECT_TRUE(tl.presence() > 0.99f);

    // Fast pan: eyeMakeupKeep must fade down rapidly to prevent eyeshadow ghosting.
    for (int f = 1; f <= 6; ++f) {
        const int64_t ts = 1330 + f * 33;
        const float dx = 0.025f * static_cast<float>(f);  // ~0.75 screenWidth/sec
        makeCanonicalFace(kAspect, 0.5f, 0.5f, 0.40f, 0.0f, dx, 0.0f, lm);
        frame.landmarks = lm;
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, state.data()));
        tl.add(ts, state.data(), &frame);
        tl.update(ts);
        EXPECT_TRUE(tl.at(ts) != nullptr);
    }
    EXPECT_TRUE(tl.eyeMakeupKeep() < 0.35f);

    // Stop moving: eyeMakeupKeep smoothly recovers toward 1.0.
    const float finalDx = 0.025f * 6.0f;
    for (int f = 1; f <= 15; ++f) {
        const int64_t ts = 1528 + f * 33;
        makeCanonicalFace(kAspect, 0.5f, 0.5f, 0.40f, 0.0f, finalDx, 0.0f, lm);
        frame.landmarks = lm;
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, state.data()));
        tl.add(ts, state.data(), &frame);
        tl.update(ts);
        EXPECT_TRUE(tl.at(ts) != nullptr);
    }
    EXPECT_TRUE(tl.eyeMakeupKeep() > 0.85f);
}

void testRoiSelectorAndTuning() {
    FaceRoiSelector roi;
    EXPECT_TRUE(!roi.isActive());

    // Enter ROI on far-field face (width 0.22 <= 0.30).
    roi.onFace(0.48f, 0.60f, 0.22f, 1000, 1020);
    EXPECT_TRUE(roi.isActive());
    const CropRect rect = roi.select(1080, 1920, 1050);
    EXPECT_TRUE(!rect.isFullFrame());
    // Isotropic square in 9:16 portrait => cropH / cropW == aspect.
    EXPECT_NEAR(rect.height / rect.width, 1080.0f / 1920.0f, 1e-4f);
    // Unproject center back to MP coordinates (0.48, 1 - 0.60 = 0.40).
    EXPECT_NEAR(rect.unprojectX(0.5f), 0.48f, 1e-3f);
    EXPECT_NEAR(rect.unprojectY(0.5f), 0.40f, 1e-3f);

    // Hysteresis: faceWidth 0.33 stays active (< 0.36).
    for (int i = 0; i < 6; ++i) roi.onFace(0.48f, 0.60f, 0.33f, 1100 + i * 33, 1120 + i * 33);
    EXPECT_TRUE(roi.isActive());

    // Miss immediately exits ROI.
    roi.onMiss();
    EXPECT_TRUE(!roi.isActive());
    EXPECT_TRUE(roi.select(1080, 1920, 1400).isFullFrame());

    // ES3 version parser.
    EXPECT_TRUE(isEs3VersionString("OpenGL ES 3.2 V@0502.0"));
    EXPECT_TRUE(!isEs3VersionString("OpenGL ES 2.0"));
    EXPECT_TRUE(!isEs3VersionString(nullptr));

    // Guided filter tuning & macro evenness.
    EXPECT_NEAR(BeautyQualityTuning::macroEvenness(0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(BeautyQualityTuning::macroEvenness(1.0f), 0.42f, 1e-6f);
    EXPECT_NEAR(BeautyQualityTuning::guidedEps(0.0f), 0.01845f, 1e-4f);
    EXPECT_NEAR(BeautyQualityTuning::guidedEps(1.0f), 0.07381f, 1e-4f);
}

void testBodyAndLoadGovernor() {
    std::array<PoseLandmark, 33> pts{};
    pts[11] = {0.40f, 0.30f, 0.9f};
    pts[12] = {0.60f, 0.30f, 0.9f};
    pts[23] = {0.42f, 0.60f, 0.9f};
    pts[24] = {0.58f, 0.60f, 0.9f};
    pts[25] = {0.43f, 0.85f, 0.9f};
    pts[26] = {0.57f, 0.85f, 0.9f};
    BodyGeometry bg{};
    EXPECT_TRUE(BodyGeometry::fromPose(pts.data(), 33, 9.0f / 16.0f, bg));
    EXPECT_NEAR(bg.legs, 1.0f, 1e-6f);
    EXPECT_TRUE(bg.torso > 0.4f);

    EffectsLoadGovernor gov;
    int64_t tsNs = 33000000LL;
    int64_t nowMs = 0;
    int upgraded = 0;
    for (int w = 0; w < 4; ++w) {
        for (int f = 0; f < 32; ++f) {
            tsNs += 33333333LL;
            nowMs += 33;
            auto res = gov.onFrame(tsNs, 35000000LL, true, nowMs);
            if (res.has_value()) upgraded = *res;
        }
    }
    EXPECT_TRUE(upgraded >= 1 && gov.level() >= 1);
}

void testHairMaskTuningAndMotionComp() {
    float aff[HairMaskTuning::kAffineSize];
    const float poseMask[6] = {0.48f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};
    const float poseDisp[6] = {0.52f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};
    // Small 0.04 shift -> valid rigid compensation mapping display center (0.52, 0.50) -> mask center (0.48, 0.50)
    EXPECT_TRUE(HairMaskTuning::affine(poseMask, poseDisp, 0.5625f, aff));
    const float mappedU = aff[0] * 0.52f + aff[1] * 0.50f + aff[2];
    const float mappedV = aff[4] * 0.52f + aff[5] * 0.50f + aff[6];
    EXPECT_NEAR(mappedU, 0.48f, 1e-5f);
    EXPECT_NEAR(mappedV, 0.50f, 1e-5f);

    // Excessive shift (> 0.10) falls back to identity for safety
    const float poseFar[6] = {0.75f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};
    EXPECT_TRUE(!HairMaskTuning::affine(poseMask, poseFar, 0.5625f, aff));
    EXPECT_NEAR(aff[0], 1.0f, 1e-6f);
    EXPECT_NEAR(aff[5], 1.0f, 1e-6f);
    EXPECT_NEAR(aff[2], 0.0f, 1e-6f);

    // Freshness window: <= 420ms is 1.0, >= 650ms is 0.0
    EXPECT_NEAR(HairMaskTuning::fresh(200.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(HairMaskTuning::fresh(420.0f), 1.0f, 1e-6f);
    EXPECT_TRUE(HairMaskTuning::fresh(535.0f) > 0.3f && HairMaskTuning::fresh(535.0f) < 0.7f);
    EXPECT_NEAR(HairMaskTuning::fresh(700.0f), 0.0f, 1e-6f);

    // faceKeep fades out cleanly by 450ms after confirmed miss (t reaches 1462 >= 1450)
    float fk = 1.0f;
    for (int64_t t = 1000; t <= 1470; t += 33) {
        fk = HairMaskTuning::faceKeep(fk, 1000, 1050, t, 33, false);
    }
    EXPECT_NEAR(fk, 0.0f, 1e-5f);
}

void testStickerTransformAnchorsAndRotation() {
    const float aspect = 9.0f / 16.0f;
    std::array<float, FaceMeshFrame::kCoordCount> glPts{};
    makeCanonicalFace(aspect, 0.50f, 0.50f, 0.42f, 0.0f, 0.0f, 0.0f, glPts);
    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    for (int i = 0; i < 478; ++i) {
        mpXy[i * 2] = glPts[i * 2];
        mpXy[i * 2 + 1] = 1.0f - glPts[i * 2 + 1];
    }
    FaceMeshFrame frame{};
    EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mpXy.data(), 478, aspect, frame));

    StickerLayerConfig cfg{};
    cfg.enabled = true;
    cfg.texture = 99;
    cfg.anchor = StickerAnchor::kForeheadCrown;
    cfg.scale = 1.0f;
    cfg.opacity = 0.9f;
    float quad[StickerTransform::kQuadFloats];
    EXPECT_TRUE(StickerTransform::buildQuadVertices(cfg, frame.landmarks.data(), frame.meta.data(), aspect, quad));
    // Crown should sit above nose tip (NDC y > 0) and stay finite
    const float crownCenterY = 0.25f * (quad[1] + quad[5] + quad[9] + quad[13]);
    EXPECT_TRUE(std::isfinite(crownCenterY) && crownCenterY > 0.05f);

    // Nose tip anchor should center directly on landmark 1
    cfg.anchor = StickerAnchor::kNoseTip;
    EXPECT_TRUE(StickerTransform::buildQuadVertices(cfg, frame.landmarks.data(), frame.meta.data(), aspect, quad));
    const float noseCenterX = 0.25f * (quad[0] + quad[4] + quad[8] + quad[12]);
    const float noseCenterY = 0.25f * (quad[1] + quad[5] + quad[9] + quad[13]);
    EXPECT_NEAR(noseCenterX, frame.landmarks[2] * 2.0f - 1.0f, 1e-4f);
    EXPECT_NEAR(noseCenterY, frame.landmarks[3] * 2.0f - 1.0f, 1e-4f);

    // Disabled layer returns false
    cfg.enabled = false;
    EXPECT_TRUE(!StickerTransform::buildQuadVertices(cfg, frame.landmarks.data(), frame.meta.data(), aspect, quad));
}

void testMultiFaceTrackerStableAssociationAndCrossing() {
    const float aspect = 9.0f / 16.0f;
    auto makeMp = [&](float cx, float cy, float fw) {
        std::array<float, FaceMeshFrame::kCoordCount> glPts{}, mp{};
        makeCanonicalFace(aspect, cx, cy, fw, 0.0f, 0.0f, 0.0f, glPts);
        for (int i = 0; i < 478; ++i) {
            mp[i * 2] = glPts[i * 2];
            mp[i * 2 + 1] = 1.0f - glPts[i * 2 + 1];
        }
        return mp;
    };

    MultiFaceTracker tracker;
    tracker.setAspect(aspect);

    // Frame 0: Face A (left, larger fw=0.34) & Face B (right, smaller fw=0.28)
    auto fA0 = makeMp(0.30f, 0.50f, 0.34f);
    auto fB0 = makeMp(0.70f, 0.50f, 0.28f);
    MultiFaceTracker::RawFaceInput pair0[2] = {{fB0.data(), 478}, {fA0.data(), 478}};
    EXPECT_TRUE(tracker.submitFaces(1000, pair0, 2, 720, 1280, 1000) == 2);
    tracker.update(1000);
    // Larger face (A at x=0.30) must enter primary slot 0, smaller face (B at x=0.70) enters slot 1.
    FaceTimeline::SmoothedMeshView mv0{}, mv1{};
    EXPECT_TRUE(tracker.slot(0).meshAt(1000, mv0));
    EXPECT_TRUE(tracker.slot(1).meshAt(1000, mv1));
    EXPECT_NEAR(mv0.centerX(), 0.30f, 0.02f);
    EXPECT_NEAR(mv1.centerX(), 0.70f, 0.02f);

    // Frame 1: Detector returns faces in reversed array order after slight motion; slot IDs must NOT swap!
    auto fA1 = makeMp(0.32f, 0.50f, 0.34f);
    auto fB1 = makeMp(0.68f, 0.50f, 0.28f);
    MultiFaceTracker::RawFaceInput pair1[2] = {{fB1.data(), 478}, {fA1.data(), 478}};
    EXPECT_TRUE(tracker.submitFaces(1033, pair1, 2, 720, 1280, 1033) == 2);
    tracker.update(1033);
    EXPECT_TRUE(tracker.slot(0).meshAt(1033, mv0));
    EXPECT_TRUE(tracker.slot(1).meshAt(1033, mv1));
    EXPECT_TRUE(mv0.centerX() < 0.40f);  // Still Face A
    EXPECT_TRUE(mv1.centerX() > 0.60f);  // Still Face B

    // Ramp presence over 6 frames so both faces are active
    for (int k = 2; k < 8; ++k) {
        const int64_t ts = 1000 + k * 33;
        EXPECT_TRUE(tracker.submitFaces(ts, pair1, 2, 720, 1280, ts) == 2);
        tracker.update(ts);
    }
    EXPECT_TRUE(tracker.activeFaceCount() == 2);
}

void testFaceRelight3dNormalsAndStyles() {
    FaceRelightSolver solver;
    const auto& normals = solver.canonicalNormals();
    // Nose tip (1) and forehead (10) must point outward (+Z) with unit norm
    const auto checkUnit = [&](int idx) {
        const float nx = normals[idx * 3 + 0];
        const float ny = normals[idx * 3 + 1];
        const float nz = normals[idx * 3 + 2];
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        EXPECT_NEAR(len, 1.0f, 1e-3f);
    };
    checkUnit(1);
    checkUnit(10);
    checkUnit(234);
    checkUnit(454);
    EXPECT_TRUE(normals[1 * 3 + 2] > 0.55f);
    EXPECT_TRUE(normals[234 * 3 + 0] < -0.40f);  // right cheek points -X
    EXPECT_TRUE(normals[454 * 3 + 0] > 0.40f);   // left cheek points +X

    const float aspect = 9.0f / 16.0f;
    std::array<float, FaceMeshFrame::kCoordCount> glPts{};
    makeCanonicalFace(aspect, 0.5f, 0.5f, 0.35f, 0.0f, 0.0f, 0.0f, glPts);
    const HeadPose3D poseFront = FaceRelightSolver::estimateHeadPose(glPts.data(), aspect, 1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(poseFront.valid);
    // Rotation matrix must be orthonormal (row0 . row0 == 1, row0 . row1 == 0)
    const float r0Norm = poseFront.rot[0] * poseFront.rot[0] + poseFront.rot[1] * poseFront.rot[1] + poseFront.rot[2] * poseFront.rot[2];
    const float r01Dot = poseFront.rot[0] * poseFront.rot[3] + poseFront.rot[1] * poseFront.rot[4] + poseFront.rot[2] * poseFront.rot[5];
    EXPECT_NEAR(r0Norm, 1.0f, 1e-4f);
    EXPECT_NEAR(r01Dot, 0.0f, 1e-4f);

    std::array<float, FaceMeshTopology::kFaceVertexCount> rembrandt{};
    solver.computeVertexRelight(poseFront, RelightStyle::kRembrandt, 1.0f, rembrandt.data());
    // Rembrandt key light comes from +X (left side): left cheek (454) brighter than right cheek (234)
    EXPECT_TRUE(rembrandt[454] > rembrandt[234] + 0.05f);
    for (float v : rembrandt) {
        EXPECT_TRUE(std::isfinite(v) && v >= 0.0f && v <= 1.0f);
    }
}

void testDeviceTierPolicyClassificationAndDownshift() {
    EXPECT_TRUE(DeviceTierPolicy::classify("Adreno (TM) 750", 8, 12288) == PerformanceTier::kUltra);
    EXPECT_TRUE(DeviceTierPolicy::classify("Apple M3 Pro", 12, 18432) == PerformanceTier::kUltra);
    EXPECT_TRUE(DeviceTierPolicy::classify("Immortalis-G720 MC12", 8, 12288) == PerformanceTier::kUltra);
    EXPECT_TRUE(DeviceTierPolicy::classify("Adreno (TM) 642L", 8, 8192) == PerformanceTier::kStandard);
    EXPECT_TRUE(DeviceTierPolicy::classify("PowerVR Rogue GE8320", 4, 3072) == PerformanceTier::kLite);
    EXPECT_TRUE(DeviceTierPolicy::classify("Mali-G52", 4, 2048) == PerformanceTier::kLite);

    const auto ultraNormal = DeviceTierPolicy::resolve(PerformanceTier::kAuto, PerformanceTier::kUltra, 0);
    EXPECT_TRUE(ultraNormal.resolvedTier == PerformanceTier::kUltra);
    EXPECT_TRUE(ultraNormal.enableAr3d && ultraNormal.enableBlemishHeal && ultraNormal.enableMatteRefine);

    // Runtime load governor level 2 downshifts Ultra -> Standard smoothly
    const auto ultraThrottled = DeviceTierPolicy::resolve(PerformanceTier::kAuto, PerformanceTier::kUltra, 2);
    EXPECT_TRUE(ultraThrottled.resolvedTier == PerformanceTier::kStandard);
    EXPECT_TRUE(ultraThrottled.enableRelight3d);

    // Explicit Lite forces lightweight path
    const auto lite = DeviceTierPolicy::resolve(PerformanceTier::kLite, PerformanceTier::kUltra, 0);
    EXPECT_TRUE(lite.resolvedTier == PerformanceTier::kLite);
    EXPECT_TRUE(!lite.enableAr3d && !lite.enableBlemishHeal && !lite.enableMatteRefine && lite.maxFacesCap == 1);
}

void testAr3dSolverAndSpringPendulum() {
    SpringPendulum spring;
    spring.update(0.50f, 0.50f, 0.0f, 22.0f, 4.5f, 1000);
    // Sudden lateral head acceleration must excite bounded sway
    spring.update(0.56f, 0.50f, 0.0f, 22.0f, 4.5f, 1033);
    spring.update(0.64f, 0.50f, 0.05f, 22.0f, 4.5f, 1066);
    const float peakAngle = std::fabs(spring.swayAngleRad());
    EXPECT_TRUE(peakAngle > 1e-4f && peakAngle <= 0.581f);
    // Hold stationary for ~2.5 seconds: damped spring must settle smoothly toward 0
    for (int i = 0; i < 80; ++i) {
        spring.update(0.64f, 0.50f, 0.05f, 22.0f, 4.5f, 1099 + i * 33);
    }
    EXPECT_TRUE(std::fabs(spring.swayAngleRad()) < peakAngle);
    EXPECT_TRUE(std::fabs(spring.swayAngleRad()) < 0.05f);

    const float aspect = 9.0f / 16.0f;
    std::array<float, FaceMeshFrame::kCoordCount> glPts{};
    makeCanonicalFace(aspect, 0.5f, 0.5f, 0.35f, 0.0f, 0.0f, 0.0f, glPts);
    const HeadPose3D pose = FaceRelightSolver::estimateHeadPose(glPts.data(), aspect, 1.0f, 0.0f, 0.0f);
    std::array<float, Ar3dSolver::kHeadShellFloats> shell{};
    Ar3dSolver::buildHeadDepthShell(glPts.data(), pose, shell.data());
    // Nose tip (lm[1]) must be closer (higher depth01) than ear edge (lm[234])
    EXPECT_TRUE(shell[1 * 5 + 4] > shell[234 * 5 + 4] + 0.15f);

    Ar3dAccessoryConfig earCfg{};
    earCfg.enabled = true;
    earCfg.anchor = Ar3dAnchor::kLeftEar;
    earCfg.texture = 99;
    float quad[Ar3dSolver::kQuadFloats]{};
    float ax = 0.0f, ay = 0.0f;
    EXPECT_TRUE(Ar3dSolver::buildAccessoryQuad(earCfg, glPts.data(), pose, aspect, &spring, quad, ax, ay));
    // Left ear quad sits behind nose tip plane (depth < nose depth)
    EXPECT_TRUE(quad[4] < shell[1 * 5 + 4]);
}

void testFaceAuxVisionNeonAndExpressionBus() {
    // 1. FaceExpressionBus
    FaceExpressionBus bus;
    BlendshapeInput neutral{};
    auto r0 = bus.update(&neutral);
    EXPECT_TRUE(r0.trigger == ExpressionTrigger::kNone);
    EXPECT_NEAR(r0.scaleMultiplier, 1.0f, 1e-4f);

    BlendshapeInput smile{};
    smile.smile = 0.86f;
    ExpressionReaction rs{};
    for (int i = 0; i < 6; ++i) rs = bus.update(&smile);
    EXPECT_TRUE(rs.trigger == ExpressionTrigger::kSmile);
    EXPECT_TRUE(rs.scaleMultiplier > 1.05f);
    EXPECT_TRUE(rs.offsetYBoost < -0.005f);

    // 2. FaceOcclusionSolver (clean skin vs foreign occluder + eye dark pupil exemption)
    const int w = 128, h = 128;
    std::vector<uint8_t> rgba(w * h * 4, 255);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int off = (y * w + x) * 4;
            if (x >= 70 && x <= 92 && y >= 58 && y <= 88) {
                rgba[off] = 22; rgba[off + 1] = 140; rgba[off + 2] = 180;
            } else if (x >= 46 && x <= 52 && y >= 46 && y <= 50) {
                rgba[off] = 40; rgba[off + 1] = 36; rgba[off + 2] = 34;
            } else {
                rgba[off] = 215; rgba[off + 1] = 175; rgba[off + 2] = 152;
            }
        }
    }
    FaceOcclusionSolver occ;
    CropRect bounds{0.15f, 0.15f, 0.70f, 0.70f};
    std::array<uint8_t, FaceOcclusionSolver::kMaskPixels> mask{};
    occ.compute(rgba.data(), w, h, &bounds, mask.data());
    EXPECT_TRUE(mask[34 * 64 + 22] >= 240);  // clean cheek stays visible
    EXPECT_TRUE(mask[36 * 64 + 40] <= 80);   // cyan cup occluded
    EXPECT_TRUE(mask[24 * 64 + 24] >= 220);  // dark pupil exempted

    // 3. FaceRoiResidualSolver (heals red blemish & zeroes eyes)
    std::fill(rgba.begin(), rgba.end(), 255);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int off = (y * w + x) * 4;
            if (x >= 44 && x <= 48 && y >= 66 && y <= 70) {
                rgba[off] = 195; rgba[off + 1] = 125; rgba[off + 2] = 115;
            } else {
                rgba[off] = 215; rgba[off + 1] = 175; rgba[off + 2] = 152;
            }
        }
    }
    FaceRoiResidualSolver resSolver;
    CropRect roi{0.10f, 0.10f, 0.80f, 0.80f};
    std::array<uint8_t, FaceRoiResidualSolver::kRoiPixels * 4> resRgba{};
    EXPECT_TRUE(resSolver.compute(rgba.data(), w, h, roi, 0.85f, resRgba.data()));
    const int eyeOff = (22 * 64 + 22) * 4;
    EXPECT_TRUE(resRgba[eyeOff + 3] == 0 && resRgba[eyeOff] == 128);
    const int bx = static_cast<int>(((46.0f / 128.0f) - roi.left) / roi.width * 64.0f);
    const int by = static_cast<int>(((68.0f / 128.0f) - roi.top) / roi.height * 64.0f);
    const int bOff = (by * 64 + bx) * 4;
    EXPECT_TRUE(resRgba[bOff + 3] > 100);
    EXPECT_TRUE((static_cast<int>(resRgba[bOff + 1]) - 128) > (static_cast<int>(resRgba[bOff]) - 128));
}

void testFaceParsing19ClassAndNeuralResidualAdapter() {
    // 1. 19-class FaceParsingMapper: skin(1)/lip(12) visible, eyeglasses(6)/teeth(11)/hair(17) occluded!
    const int w = 64, h = 64;
    std::vector<uint8_t> clsMap(w * h, static_cast<uint8_t>(FaceParsingClass::kSkin));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y < 14) clsMap[y * w + x] = static_cast<uint8_t>(FaceParsingClass::kHair);
            else if (y >= 20 && y <= 28 && x >= 14 && x <= 50) clsMap[y * w + x] = static_cast<uint8_t>(FaceParsingClass::kEyeglasses);
            else if (y >= 44 && y <= 50 && x >= 24 && x <= 40) clsMap[y * w + x] = static_cast<uint8_t>(FaceParsingClass::kInnerMouth);
        }
    }
    FaceParsingMapper mapper;
    std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
    EXPECT_TRUE(mapper.mapClassIds(clsMap.data(), w, h, occ.data(), hair.data(), person.data()));

    // Clean cheek (y=72, x=64 in 128x128) -> occ=255, hair=0, person=255
    EXPECT_TRUE(occ[72 * 128 + 64] >= 250);
    EXPECT_TRUE(hair[72 * 128 + 64] <= 5);
    EXPECT_TRUE(person[72 * 128 + 64] >= 250);
    // Eyeglasses center (y=48, x=64 in 128x128) -> occ=0!
    EXPECT_TRUE(occ[48 * 128 + 64] <= 10);
    // Inner-mouth teeth center (y=94, x=64 in 128x128) -> occ=0!
    EXPECT_TRUE(occ[94 * 128 + 64] <= 10);
    // Hair crown (y=12, x=64 in 128x128) -> hair=255, occ=0!
    EXPECT_TRUE(hair[12 * 128 + 64] >= 245);
    EXPECT_TRUE(occ[12 * 128 + 64] <= 10);

    // 2. NeuralSkinResidualAdapter external NCHW tensor + built-in structural shadow micro-kernel
    std::vector<float> nchw(4 * 32 * 32, 0.0f);
    for (int i = 0; i < 32 * 32; ++i) {
        nchw[0 * 1024 + i] = 0.10f;   // +0.10 R delta -> ~179
        nchw[3 * 1024 + i] = 0.80f;   // 0.80 confidence -> ~204
    }
    std::array<uint8_t, NeuralSkinResidualAdapter::kOutPixels * 4> packed{};
    EXPECT_TRUE(NeuralSkinResidualAdapter::packExternalTensorNchw(nchw.data(), 4, 32, 32, packed.data()));
    EXPECT_TRUE(packed[0] >= 175 && packed[0] <= 183);
    EXPECT_TRUE(packed[1] == 128);
    EXPECT_TRUE(packed[3] >= 200 && packed[3] <= 208);

    // Built-in micro-kernel lifts localized tear-trough / nasolabial shadow dip
    std::vector<uint8_t> faceImg(128 * 128 * 4, 255);
    for (int y = 0; y < 128; ++y) {
        for (int x = 0; x < 128; ++x) {
            const int off = (y * 128 + x) * 4;
            const bool hollowShadow = (y >= 72 && y <= 76 && x >= 40 && x <= 56);
            faceImg[off]     = hollowShadow ? 182 : 212;
            faceImg[off + 1] = hollowShadow ? 142 : 172;
            faceImg[off + 2] = hollowShadow ? 122 : 150;
        }
    }
    NeuralSkinResidualAdapter adapter;
    std::array<uint8_t, NeuralSkinResidualAdapter::kOutPixels * 4> microOut{};
    CropRect roi{0.10f, 0.10f, 0.80f, 0.80f};
    EXPECT_TRUE(adapter.evaluateMicroKernel(faceImg.data(), 128, 128, roi, 0.90f, microOut.data()));
    const int sx = static_cast<int>(((48.0f / 128.0f) - roi.left) / roi.width * 64.0f);
    const int sy = static_cast<int>(((74.0f / 128.0f) - roi.top) / roi.height * 64.0f);
    const int sOff = (sy * 64 + sx) * 4;
    EXPECT_TRUE(microOut[sOff] > 132 && microOut[sOff + 1] > 132);
}

void testV35Curved3dMeshesAndEffectBundleParser() {
    // 1. Declarative .lbfx Effect Bundle manifest JSON parser
    const std::string lbfxJson = R"JSON({
        "schemaVersion": 1,
        "id": "rose_crown_glam",
        "name": "Rose Crown Glam",
        "author": "Studio",
        "beauty": {
            "preset": "korean_water_glow",
            "cranialTop": 0.62,
            "hairDye": 0.75,
            "hairDyeColor": "rose_gold",
            "hairShine": 0.58,
            "lipstick": 0.65,
            "lipstickFinish": "glossy"
        },
        "stickers": [
            {
                "slot": 0,
                "anchor": "cheeks",
                "assetPath": "stickers/sparkle_heart.png",
                "scale": 1.15,
                "opacity": 0.90,
                "trigger": "smile"
            }
        ],
        "ar3d": [
            {
                "slot": 0,
                "anchor": "glasses_bridge",
                "meshPreset": "curved_glasses",
                "assetPath": "ar3d/cyber_glasses.png",
                "scale": 1.05,
                "headOcclusion": true,
                "trigger": "always"
            },
            {
                "slot": 1,
                "anchor": "head_crown",
                "meshPreset": "curved_tiara",
                "assetPath": "ar3d/star_tiara.png",
                "springStiffness": 22.0,
                "trigger": "brow_up"
            }
        ]
    })JSON";

    bundle::EffectBundleManifest manifest{};
    std::string err;
    EXPECT_TRUE(bundle::EffectBundleParser::parseJson(lbfxJson, manifest, &err));
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(manifest.id == "rose_crown_glam");
    EXPECT_TRUE(manifest.hasBeautySection);
    EXPECT_TRUE(std::fabs(manifest.beauty.cranialTop - 0.62f) < 1e-5f);
    EXPECT_TRUE(std::fabs(manifest.beauty.hairDye - 0.75f) < 1e-5f);
    EXPECT_TRUE(manifest.beauty.hairDyeColor == HairDyeColor::kRoseGold);
    EXPECT_TRUE(std::fabs(manifest.beauty.hairShine - 0.58f) < 1e-5f);
    EXPECT_TRUE(manifest.stickers.size() == 1);
    EXPECT_TRUE(manifest.stickers[0].trigger == bundle::BundleTriggerRule::kSmile);
    EXPECT_TRUE(manifest.ar3d.size() == 2);
    EXPECT_TRUE(manifest.ar3d[0].layer.meshPreset == Ar3dMeshPreset::kCurvedGlasses);
    EXPECT_TRUE(manifest.ar3d[1].layer.meshPreset == Ar3dMeshPreset::kCurvedTiara);
    EXPECT_TRUE(manifest.ar3d[1].trigger == bundle::BundleTriggerRule::kBrowUp);

    // Invalid JSON cleanly rejected with diagnostic message
    bundle::EffectBundleManifest bad{};
    std::string badErr;
    EXPECT_TRUE(!bundle::EffectBundleParser::parseJson("{ invalid json ", bad, &badErr));
    EXPECT_TRUE(!badErr.empty());

    // Trigger gain evaluation
    ExpressionReaction neutralRx{};
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(bundle::BundleTriggerRule::kAlways, neutralRx) == 1.0f);
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(bundle::BundleTriggerRule::kSmile, neutralRx) == 0.0f);
    ExpressionReaction smileRx{};
    smileRx.smile = 0.75f;
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(bundle::BundleTriggerRule::kSmile, smileRx) > 0.95f);

    // 2. Curved 3D Glasses & Tiara indexed mesh projection with receding -Z temples
    std::array<float, FaceMeshFrame::kCoordCount> glLm{};
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float aspect = 9.0f / 16.0f;
    const float scale = 0.40f / FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        glLm[i * 2] = 0.5f + xyz[i * 3] * scale;
        glLm[i * 2 + 1] = (0.5f / aspect + xyz[i * 3 + 1] * scale) * aspect;
    }
    const auto pose = FaceRelightSolver::estimateHeadPose(glLm.data(), aspect, 1.0f, 0.0f, 1.0f);
    EXPECT_TRUE(pose.valid);

    Ar3dAccessoryConfig glassesCfg = manifest.ar3d[0].layer;
    glassesCfg.texture = 7;
    std::array<float, Ar3dSolver::kMaxMeshFloats> meshVerts{};
    int vertCount = 0, idxCount = 0;
    const uint16_t* indices = nullptr;
    float ax = 0.0f, ay = 0.0f;
    EXPECT_TRUE(Ar3dSolver::buildAccessoryMesh(
        glassesCfg, glLm.data(), pose, aspect, nullptr,
        meshVerts.data(), Ar3dSolver::kMaxMeshVertices, vertCount, indices, idxCount, ax, ay));
    EXPECT_TRUE(vertCount == 18 && idxCount == 36 && indices != nullptr);
    // Front bridge vertex (idx 4) is significantly closer to camera (higher depth01) than rear temple tip (idx 12)
    const float bridgeDepth = meshVerts[4 * 5 + 4];
    const float templeRearDepth = meshVerts[12 * 5 + 4];
    EXPECT_TRUE(bridgeDepth > templeRearDepth + 0.12f);
}

void testCodeReviewRegressionGuards() {
    // 1. NaN clamp guard: neural residual NaN maps to exact 0 delta (128 byte), [0,1] maps to lo
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(livebeauty::detail::clampf(nanVal, -0.24f, 0.24f) == 0.0f);
    EXPECT_TRUE(livebeauty::detail::clampf(nanVal, 0.20f, 1.0f) == 0.20f);
    std::vector<float> nanTensor(4 * 16 * 16, nanVal);
    std::array<uint8_t, NeuralSkinResidualAdapter::kOutPixels * 4> resRgba{};
    EXPECT_TRUE(NeuralSkinResidualAdapter::packExternalTensorNchw(nanTensor.data(), 4, 16, 16, resRgba.data()));
    EXPECT_TRUE(resRgba[0] == 128 && resRgba[1] == 128 && resRgba[2] == 128 && resRgba[3] == 0);

    // 2. FaceParsingMapper ignore_index = 255 stays visible (255), NOT false-positive occluder (0)
    FaceParsingMapper mapper;
    std::vector<uint8_t> ignoreMap(32 * 32, 255);
    std::array<uint8_t, FaceParsingMapper::kOutPixels> occOut{}, hairOut{};
    EXPECT_TRUE(mapper.mapClassIds(ignoreMap.data(), 32, 32, occOut.data(), hairOut.data(), nullptr));
    EXPECT_TRUE(occOut[FaceParsingMapper::kOutPixels / 2] == 255);
    EXPECT_TRUE(hairOut[FaceParsingMapper::kOutPixels / 2] == 0);

    // 3. Ar3dSolver custom mesh validation rejects OOB indices, non-triangles, and NaN coords
    const Ar3dMeshVertex goodVerts[3] = {{-0.5f, -0.5f, 0.0f, 0.0f, 0.0f}, {0.5f, -0.5f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.5f, 0.0f, 0.5f, 1.0f}};
    const uint16_t goodIdx[3] = {0, 1, 2};
    const uint16_t badOobIdx[3] = {0, 1, 3};
    const uint16_t badCountIdx[4] = {0, 1, 2, 0};
    EXPECT_TRUE(Ar3dSolver::validateCustomMesh(goodVerts, 3, goodIdx, 3));
    EXPECT_TRUE(!Ar3dSolver::validateCustomMesh(goodVerts, 3, badOobIdx, 3));
    EXPECT_TRUE(!Ar3dSolver::validateCustomMesh(goodVerts, 3, badCountIdx, 4));

    // 4. Mirrored front-camera HeadDepthShell keeps nose closer than cheek
    std::array<float, FaceMeshFrame::kCoordCount> mirrorLm{};
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float aspect = 9.0f / 16.0f;
    const float scale = 0.40f / FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        mirrorLm[i * 2] = 0.5f - xyz[i * 3] * scale;  // mirrored X (CW front face)
        mirrorLm[i * 2 + 1] = (0.5f / aspect + xyz[i * 3 + 1] * scale) * aspect;
    }
    const auto mirrorPose = FaceRelightSolver::estimateHeadPose(mirrorLm.data(), aspect, -1.0f, 0.0f, 1.0f);
    EXPECT_TRUE(mirrorPose.valid);
    std::array<float, Ar3dSolver::kHeadShellFloats> shellMirrored{};
    Ar3dSolver::buildHeadDepthShell(mirrorLm.data(), mirrorPose, shellMirrored.data(), /*mirrored=*/true);
    EXPECT_TRUE(shellMirrored[4 * 5 + 4] > shellMirrored[234 * 5 + 4]);

    // 5. EffectBundleParser: out-of-order slots, UTF-16 surrogate pairs, body/bg/pipeline overrides, hex reject
    const std::string bundleJson = R"JSON({
        "schemaVersion": 1,
        "id": "review_guard",
        "name": "\uD83C\uDF38 Blossom",
        "beauty": { "waistSlim": 0.42, "neckTone": 0.35, "oilControl": 0.50, "matteTemporal": 0.72 },
        "background": { "mode": "virtual_blur", "strength": 0.85 },
        "pipelineOptions": { "maxFaces": 2, "frameDelayFrames": 2 },
        "stickers": [
            { "slot": 3, "anchor": "forehead", "assetPath": "s3.png" },
            { "slot": 0, "anchor": "cheeks", "assetPath": "s0.png", "trigger": "tongue_out" }
        ]
    })JSON";
    bundle::EffectBundleManifest mf{};
    EXPECT_TRUE(bundle::EffectBundleParser::parseJson(bundleJson, mf, nullptr));
    EXPECT_TRUE(mf.name == "\xF0\x9F\x8C\xB8 Blossom");
    EXPECT_TRUE(mf.beauty.needsFaceTracking());
    EXPECT_TRUE(std::fabs(mf.beauty.waistSlim - 0.42f) < 1e-5f);
    EXPECT_TRUE(mf.overridesBackground && mf.beauty.backgroundMode == BackgroundMode::kVirtualBlur);
    EXPECT_TRUE(mf.overridesPipelineOptions && mf.beauty.maxFaces == 2 && mf.beauty.frameDelayFrames == 2);
    EXPECT_TRUE(mf.stickers.size() == 2);
    EXPECT_TRUE(mf.stickers[1].slot == 0 && mf.stickers[1].trigger == bundle::BundleTriggerRule::kTongueOut);

    bundle::EffectBundleManifest hexBad{};
    EXPECT_TRUE(!bundle::EffectBundleParser::parseJson(R"({"beauty":{"smoothing":0x1.0p0}})", hexBad, nullptr));

    // 6. FaceExpressionBus: blink does not dim permanent kAlways opacityBoost, kTongueOut activates cleanly,
    //    and sub-threshold smile ramps scaleMultiplier smoothly from 1.0f without step jump at trigger boundary
    FaceExpressionBus bus;
    const BlendshapeInput subSmile{0.62f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ExpressionReaction rxSub{};
    for (int i = 0; i < 2; ++i) rxSub = bus.update(&subSmile);
    EXPECT_TRUE(rxSub.trigger == ExpressionTrigger::kNone);
    EXPECT_TRUE(rxSub.scaleMultiplier > 1.0f && rxSub.scaleMultiplier < 1.042f);

    bus.reset();
    const BlendshapeInput bsSample{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.90f};
    bus.update(&bsSample);
    const auto rxBlink = bus.update(&bsSample);
    EXPECT_TRUE(rxBlink.opacityBoost == 1.0f);
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(bundle::BundleTriggerRule::kTongueOut, rxBlink) > 0.8f);

    // 7. Head-roll-compensated eye exemption in FaceRoiResidualSolver & NeuralSkinResidualAdapter (+45 deg roll)
    const float roll45 = 0.78539816f;
    // Canonical left eye center (relX=0.30, relY=0.35) rotated by +45 deg around (0.5, 0.5):
    // dx = -0.20, dy = -0.15 -> rotX = 0.5 + (-0.20*0.7071 - (-0.15)*0.7071) = 0.4646, rotY = 0.5 + (-0.35*0.7071) = 0.2525
    const int rotEyeX = static_cast<int>(0.4646f * 64.0f);
    const int rotEyeY = static_cast<int>(0.2525f * 64.0f);
    std::vector<uint8_t> rollFace(64 * 64 * 4, 200);
    CropRect fullRoi{0.0f, 0.0f, 1.0f, 1.0f};
    FaceRoiResidualSolver rollResSolver;
    std::array<uint8_t, FaceRoiResidualSolver::kRoiPixels * 4> rollBi{};
    EXPECT_TRUE(rollResSolver.compute(rollFace.data(), 64, 64, fullRoi, 0.85f, rollBi.data(), roll45));
    EXPECT_TRUE(rollBi[(rotEyeY * 64 + rotEyeX) * 4 + 3] == 0);

    NeuralSkinResidualAdapter rollAdapter;
    std::array<uint8_t, NeuralSkinResidualAdapter::kOutPixels * 4> rollMicro{};
    EXPECT_TRUE(rollAdapter.evaluateMicroKernel(rollFace.data(), 64, 64, fullRoi, 0.85f, rollMicro.data(), roll45));
    EXPECT_TRUE(rollMicro[(rotEyeY * 64 + rotEyeX) * 4 + 3] == 0);
}

void testV36MulticlassParsingParticlesAndBundleGestureTriggers() {
    // 1. 6-class MediaPipe selfie_multiclass NHWC softmax mapper
    constexpr int kS = 32;
    std::vector<float> logits(static_cast<size_t>(kS * kS * 6), -4.0f);
    for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
            float* c = logits.data() + static_cast<size_t>((y * kS + x) * 6);
            if (y < 8) {
                c[1] = 5.0f;  // class 1: hair
            } else if (y < 16 && x >= 10 && x < 22) {
                c[5] = 5.0f;  // class 5: glasses/accessories/hands -> occludes face makeup!
            } else if (y < 24) {
                c[3] = 5.0f;  // class 3: face_skin -> visible skin!
            } else {
                c[0] = 5.0f;  // class 0: background
            }
        }
    }
    FaceParsingMapper mapper;
    std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
    EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwc(logits.data(), kS, kS, occ.data(), hair.data(), person.data()));
    // Top hair band (outY = 8) -> hair > 220, occ < 20, person > 220
    EXPECT_TRUE(hair[8 * 128 + 64] > 220 && occ[8 * 128 + 64] < 20 && person[8 * 128 + 64] > 220);
    // Glasses/accessory band (outY = 48, outX = 64) -> occ < 20 (shielded!), person > 220
    EXPECT_TRUE(occ[48 * 128 + 64] < 20 && person[48 * 128 + 64] > 220);
    // Bare face skin (outY = 80, outX = 64) -> occ > 220, hair < 20, person > 220
    EXPECT_TRUE(occ[80 * 128 + 64] > 220 && hair[80 * 128 + 64] < 20 && person[80 * 128 + 64] > 220);
    // Background (outY = 116, outX = 64) -> person < 20
    EXPECT_TRUE(person[116 * 128 + 64] < 20);

    // v3.6.5: Ambiguous dynamic-INT8 forehead/temple boundary (c[1]=1.6 hair spill vs c[3]=1.2 face_skin)
    // must NOT misclassify forehead skin as hair or occluder!
    {
        std::vector<float> ambig(16 * 16 * 6, -3.0f);
        for (int i = 0; i < 16 * 16; ++i) {
            ambig[static_cast<size_t>(i * 6 + 1)] = 1.6f;
            ambig[static_cast<size_t>(i * 6 + 3)] = 1.2f;
        }
        FaceParsingMapper ambigMapper;
        std::array<uint8_t, FaceParsingMapper::kOutPixels> aOcc{}, aHair{}, aPerson{};
        EXPECT_TRUE(ambigMapper.mapMulticlass6LogitsNhwc(ambig.data(), 16, 16, aOcc.data(), aHair.data(), aPerson.data()));
        EXPECT_TRUE(aHair[64 * 128 + 64] < 10 && aOcc[64 * 128 + 64] > 240 && aPerson[64 * 128 + 64] > 230);
    }

    // 2. ParticleBurstSolver physics, packing & expiration
    ParticleBurstSolver solver;
    EXPECT_TRUE(!solver.hasActiveParticles(1000));
    solver.triggerBurst(GestureType::kHeart, 0.50f, 0.55f, 1000);
    EXPECT_TRUE(solver.hasActiveParticles(1100));
    std::array<float, ParticleBurstSolver::kMaxVertexFloats> verts{};
    const int nQuads = solver.buildBatch(1200, 720.0f / 1280.0f, verts.data());
    EXPECT_TRUE(nQuads >= 12 && nQuads <= ParticleBurstSolver::kMaxParticles);
    // Verify packed shapeId*2 + alpha stays well-formed
    const float packedW = verts[7];
    const float shapeId = std::floor(packedW * 0.5f + 0.001f);
    const float alpha = packedW - shapeId * 2.0f;
    EXPECT_TRUE((shapeId == 0.0f || shapeId == 3.0f) && alpha > 0.05f && alpha < 1.0f);
    // After 3 seconds all particles expire cleanly
    EXPECT_TRUE(solver.buildBatch(4500, 720.0f / 1280.0f, verts.data()) == 0);

    // 3. .lbfx schema v2 chroma_key + gesture trigger parsing
    const char* v36Json = R"JSON({
        "schemaVersion": 2,
        "id": "chroma_heart_v2",
        "background": {
            "mode": "chroma_key",
            "keyColor": "blue",
            "similarity": 0.42,
            "smoothness": 0.15,
            "despill": 0.80,
            "strength": 0.95
        },
        "stickers": [
            { "slot": 0, "anchor": "forehead_crown", "assetPath": "heart.png", "trigger": "gesture_heart" }
        ]
    })JSON";
    bundle::EffectBundleManifest mf{};
    EXPECT_TRUE(bundle::EffectBundleParser::parseJson(v36Json, mf, nullptr));
    EXPECT_TRUE(mf.beauty.backgroundMode == BackgroundMode::kChromaKey);
    EXPECT_TRUE(mf.beauty.chromaKeyColor == ChromaKeyColor::kBlue);
    EXPECT_TRUE(std::fabs(mf.beauty.chromaKeySimilarity - 0.42f) < 1e-5f);
    EXPECT_TRUE(mf.stickers.size() == 1 && mf.stickers[0].trigger == bundle::BundleTriggerRule::kGestureHeart);
    ExpressionReaction neutralRx{};
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(
        bundle::BundleTriggerRule::kGestureHeart, neutralRx, GestureType::kNone) == 0.0f);
    EXPECT_TRUE(bundle::EffectBundleParser::evaluateTriggerGain(
        bundle::BundleTriggerRule::kGestureHeart, neutralRx, GestureType::kHeart) == 1.0f);
}

void testV366FastReacquireAndZeroLagLipNoseEye() {
    constexpr float kAspect = 9.0f / 16.0f;
    FaceTimeline tl;
    tl.setAspect(kAspect);

    std::array<float, FaceMeshFrame::kCoordCount> lmCenter{};
    makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.40f, 0.0f, 0.0f, 0.0f, lmCenter);

    auto feedAt = [&](int64_t ts, float cx, float cy, float yawRad) {
        std::array<float, FaceMeshFrame::kCoordCount> lm{};
        std::array<float, FaceGeometryConstants::kStateSize> st{};
        FaceMeshFrame frame{};
        makeCanonicalFace(kAspect, cx, cy, 0.40f, yawRad, 0.0f, 0.0f, lm);
        frame.landmarks = lm;
        frame.hasIris = true;
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, st.data()));
        tl.add(ts, st.data(), &frame);
        tl.update(ts);
        return tl.at(ts);
    };

    // 1. Move face rapidly toward frame edge, exit for 140ms (< old 300ms gap), then re-center at (0.50, 0.50):
    //    Frame 0 & Frame 1 of re-entry must snap nose/lips/eye-sockets immediately with zero stale pre-exit drag.
    for (int f = 0; f < 5; ++f) feedAt(1000 + f * 33, 0.50f, 0.50f, 0.0f);
    for (int f = 1; f <= 4; ++f) {
        const int64_t ts = 1132 + f * 33;
        feedAt(ts, 0.50f + 0.045f * static_cast<float>(f), 0.50f, 0.0f);
        // Simulate extrapolation offset before leaving frame
        tl.update(ts + 25);
        (void)tl.at(ts + 25);
    }
    tl.addMiss(1300);
    tl.update(1300);
    (void)tl.at(1300);
    tl.addMiss(1350);
    tl.update(1350);
    (void)tl.at(1350);
    tl.addMiss(1400);
    tl.update(1400);
    (void)tl.at(1400);

    // Re-enter centered at t = 1433ms (169ms since last face at 1264ms)
    const float* stRe0 = feedAt(1433, 0.50f, 0.50f, 0.0f);
    EXPECT_TRUE(stRe0 != nullptr);
    FaceTimeline::SmoothedMeshView mvRe0{};
    EXPECT_TRUE(tl.meshAt(1433, mvRe0));
    EXPECT_TRUE(tl.presence() >= 0.45f);
    EXPECT_TRUE(tl.steadyEye() > 0.95f);
    EXPECT_TRUE(tl.eyeMakeupKeep() > 0.95f);
    for (int idx : {1, 13, 14, 61, 291, 33, 133, 263, 362}) {
        EXPECT_NEAR(mvRe0.landmarks[idx * 2], lmCenter[idx * 2], 0.002f);
        EXPECT_NEAR(mvRe0.landmarks[idx * 2 + 1], lmCenter[idx * 2 + 1], 0.002f);
    }
    // uMouth center (state[36..37]) must match canonical mouth center with zero secondary 1Hz One-Euro lag
    std::array<float, FaceGeometryConstants::kStateSize> stCanon{};
    FaceMeshFrame canonFrame{};
    canonFrame.landmarks = lmCenter;
    EXPECT_TRUE(FaceMeshFrame::fillMeta(lmCenter.data(), FaceMeshFrame::kCoordCount, kAspect, canonFrame.meta.data()));
    EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
        lmCenter.data(), FaceMeshFrame::kCoordCount, canonFrame.axisX(), canonFrame.axisY(), kAspect, 1.0f, stCanon.data()));
    EXPECT_NEAR(stRe0[36], stCanon[36], 0.002f);
    EXPECT_NEAR(stRe0[37], stCanon[37], 0.002f);

    // Next frame (t = 1466ms, +33ms): full presence recovered
    (void)feedAt(1466, 0.50f, 0.50f, 0.0f);
    EXPECT_TRUE(tl.presence() > 0.98f);

    // 2. Continuous in-frame shake / pan & snap-back without missing:
    //    Within 2 still frames (66ms), nose/lips/eye-sockets & steadyEye/eyeMakeupKeep must catch up cleanly.
    feedAt(1500, 0.34f, 0.48f, 0.0f);
    feedAt(1533, 0.28f, 0.46f, 0.0f);
    feedAt(1566, 0.42f, 0.49f, 0.0f);
    feedAt(1599, 0.50f, 0.50f, 0.0f);  // stopped at center
    feedAt(1632, 0.50f, 0.50f, 0.0f);  // +33ms still
    const float* stStill = feedAt(1665, 0.50f, 0.50f, 0.0f);  // +66ms still
    EXPECT_TRUE(stStill != nullptr);
    FaceTimeline::SmoothedMeshView mvStill{};
    EXPECT_TRUE(tl.meshAt(1665, mvStill));
    EXPECT_TRUE(tl.steadyEye() > 0.88f);
    EXPECT_TRUE(tl.eyeMakeupKeep() > 0.85f);
    for (int idx : {1, 13, 14, 61, 291, 33, 133, 263, 362}) {
        EXPECT_NEAR(mvStill.landmarks[idx * 2], lmCenter[idx * 2], 0.004f);
        EXPECT_NEAR(mvStill.landmarks[idx * 2 + 1], lmCenter[idx * 2 + 1], 0.004f);
    }
    EXPECT_NEAR(stStill[36], stCanon[36], 0.003f);
    EXPECT_NEAR(stStill[37], stCanon[37], 0.003f);

    // 3. 3D Head Yaw turn (0.40 rad) -> frontal snap-back (0.0 rad):
    //    Procrustes 8-DoF basisLag + residual lag catch-up + fast FaceMeshDeformer vis/relax recovery in <= 2 frames.
    FaceMeshDeformer def;
    FaceMeshDeformer::Params params{};
    params.eyeEnlarge = 0.70f;
    params.noseSlim = 0.75f;
    params.mouthSize = -0.50f;
    params.mouthSmile = 0.65f;

    FaceMeshDeformer defRef;
    EXPECT_TRUE(defRef.deform(
        lmCenter.data(), kAspect, canonFrame.axisX(), canonFrame.axisY(),
        canonFrame.faceWidth(), canonFrame.faceHeight(),
        canonFrame.leftWeight(), canonFrame.rightWeight(), canonFrame.yawWeight(),
        1.0f, 1.0f, 1.0f, params, 1000));
    const float refNoseSlimDx = std::fabs(defRef.warpedLandmarks()[48 * 2] - defRef.warpedLandmarks()[278 * 2]);

    for (int f = 1; f <= 4; ++f) {
        const int64_t ts = 1700 + f * 33;
        feedAt(ts, 0.50f, 0.50f, 0.42f);
        FaceTimeline::SmoothedMeshView mvYaw{};
        EXPECT_TRUE(tl.meshAt(ts, mvYaw));
        EXPECT_TRUE(def.deform(
            mvYaw.landmarks, kAspect, mvYaw.axisX(), mvYaw.axisY(),
            mvYaw.faceWidth(), mvYaw.faceHeight(),
            mvYaw.leftWeight(), mvYaw.rightWeight(), mvYaw.yawWeight(),
            tl.presence(), tl.steadyFace(), tl.steadyEye(), params, ts));
    }
    // Snap back to frontal and hold for 2 frames (66ms)
    for (int f = 1; f <= 3; ++f) {
        const int64_t ts = 1832 + f * 33;
        feedAt(ts, 0.50f, 0.50f, 0.0f);
        FaceTimeline::SmoothedMeshView mvFront{};
        EXPECT_TRUE(tl.meshAt(ts, mvFront));
        EXPECT_TRUE(def.deform(
            mvFront.landmarks, kAspect, mvFront.axisX(), mvFront.axisY(),
            mvFront.faceWidth(), mvFront.faceHeight(),
            mvFront.leftWeight(), mvFront.rightWeight(), mvFront.yawWeight(),
            tl.presence(), tl.steadyFace(), tl.steadyEye(), params, ts));
        if (f == 3) {
            for (int idx : {1, 13, 14, 61, 291, 33, 133, 263, 362}) {
                EXPECT_NEAR(mvFront.landmarks[idx * 2], lmCenter[idx * 2], 0.004f);
                EXPECT_NEAR(mvFront.landmarks[idx * 2 + 1], lmCenter[idx * 2 + 1], 0.004f);
            }
            const float gotNoseSlimDx = std::fabs(def.warpedLandmarks()[48 * 2] - def.warpedLandmarks()[278 * 2]);
            EXPECT_NEAR(gotNoseSlimDx, refNoseSlimDx, 0.003f);
        }
    }
}

void testV367BeardAndNasalShadowNotMisclassifiedAsForeignOccluder() {
    const int w = 128, h = 128;
    std::vector<uint8_t> rgba(w * h * 4, 255);
    CropRect bounds{0.15f, 0.15f, 0.70f, 0.70f};
    for (int y = 0; y < h; ++y) {
        const float relY = ((static_cast<float>(y) + 0.5f) / h - bounds.top) / bounds.height;
        for (int x = 0; x < w; ++x) {
            const float relX = ((static_cast<float>(x) + 0.5f) / w - bounds.left) / bounds.width;
            const int off = (y * w + x) * 4;
            if (relX >= 0.62f && relX <= 0.86f && relY >= 0.42f && relY <= 0.68f) {
                // Foreign cyan teacup -> must be occluded (<= 80)
                rgba[off] = 22; rgba[off + 1] = 140; rgba[off + 2] = 180;
            } else if (relX >= 0.42f && relX <= 0.58f && relY >= 0.52f && relY <= 0.62f) {
                // Dark nostril / nose-wing shadow outside eyes & mouth -> must stay visible (>= 240)
                rgba[off] = 34; rgba[off + 1] = 28; rgba[off + 2] = 26;
            } else if (relX >= 0.30f && relX <= 0.70f && relY >= 0.85f && relY <= 0.96f) {
                // Dark chin / jawline beard outside mouth -> must stay visible (>= 240)
                rgba[off] = 30; rgba[off + 1] = 27; rgba[off + 2] = 26;
            } else {
                // Indoor warm skin
                rgba[off] = 185; rgba[off + 1] = 148; rgba[off + 2] = 128;
            }
        }
    }
    FaceOcclusionSolver occ;
    std::array<uint8_t, FaceOcclusionSolver::kMaskPixels> mask{};
    occ.compute(rgba.data(), w, h, &bounds, mask.data());

    const int noseMx = static_cast<int>((bounds.left + 0.50f * bounds.width) * 64.0f);
    const int noseMy = static_cast<int>((bounds.top + 0.57f * bounds.height) * 64.0f);
    const int beardMx = static_cast<int>((bounds.left + 0.50f * bounds.width) * 64.0f);
    const int beardMy = static_cast<int>((bounds.top + 0.90f * bounds.height) * 64.0f);
    const int cupMx = static_cast<int>((bounds.left + 0.74f * bounds.width) * 64.0f);
    const int cupMy = static_cast<int>((bounds.top + 0.55f * bounds.height) * 64.0f);

    EXPECT_TRUE(mask[noseMy * 64 + noseMx] >= 240);
    EXPECT_TRUE(mask[beardMy * 64 + beardMx] >= 240);
    EXPECT_TRUE(mask[cupMy * 64 + cupMx] <= 80);
}

void testV368FacialContourStabilityNoJumping() {
    constexpr float kAspect = 9.0f / 16.0f;
    FaceTimeline tl;
    tl.setAspect(kAspect);
    FaceMeshDeformer def;
    FaceMeshDeformer::Params params{};
    params.faceSlim = 0.78f;
    params.vFace = 0.68f;
    params.faceNarrow = 0.50f;
    params.faceSmall = 0.42f;
    params.cheekboneSlim = 0.58f;
    params.chinShape = 0.45f;

    // 1. Talking / animated lip motion + realistic MediaPipe 256px contour noise around 5-deg yaw (signedYaw ~ 0.09):
    //    steadyFace() must stay >= 0.99 (never pumped down by lip motion!), and warped jaw/cheek width must NOT jump!
    float prevJawW = -1.0f;
    float prevCheekW = -1.0f;
    float maxJawStep = 0.0f;
    float maxCheekStep = 0.0f;
    float minSteadyFace = 1.0f;

    for (int f = 0; f < 45; ++f) {
        const int64_t ts = 1000 + f * 33;
        const float jitterSign = (f % 2 == 0) ? 1.0f : -1.0f;
        const float scaleJitter = 1.0f + 0.008f * jitterSign;
        const float yawJitter = 0.09f + 0.022f * jitterSign;  // oscillates right across old [0.05, 0.14] frontGate cliff!
        std::array<float, FaceMeshFrame::kCoordCount> lm{};
        makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.36f * scaleJitter, yawJitter, 0.0f, 0.0f, lm);

        // Inject realistic thumbnail contour jitter on kFaceOval + speaking lip motion on lower lip/mouth
        for (uint16_t oi : FaceMeshTopology::kFaceOval) {
            const float side = (lm[oi * 2] >= 0.50f) ? 1.0f : -1.0f;
            lm[oi * 2] += side * 0.0032f * jitterSign;
        }
        const float talkOpen = 0.016f * (0.5f + 0.5f * std::sin(static_cast<float>(f) * 1.15f));
        for (int idx : {14, 17, 87, 317, 84, 314, 178, 402}) {
            lm[idx * 2 + 1] -= talkOpen;  // bottom-up GL Y: lower lip moves down when speaking
        }

        FaceMeshFrame frame{};
        frame.landmarks = lm;
        frame.hasIris = true;
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
        std::array<float, FaceGeometryConstants::kStateSize> st{};
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, st.data()));

        tl.add(ts, st.data(), &frame);
        tl.update(ts);
        EXPECT_TRUE(tl.at(ts) != nullptr);
        FaceTimeline::SmoothedMeshView mv{};
        EXPECT_TRUE(tl.meshAt(ts, mv));
        EXPECT_TRUE(def.deform(
            mv.landmarks, kAspect, mv.axisX(), mv.axisY(),
            mv.faceWidth(), mv.faceHeight(),
            mv.leftWeight(), mv.rightWeight(), mv.yawWeight(),
            tl.presence(), tl.steadyFace(), tl.steadyEye(), params, ts));

        if (f >= 8) {
            minSteadyFace = std::min(minSteadyFace, tl.steadyFace());
            const float* wl = def.warpedLandmarks();
            const float jawW = std::hypot(wl[397 * 2] - wl[172 * 2], (wl[397 * 2 + 1] - wl[172 * 2 + 1]) / kAspect);
            const float cheekW = std::hypot(wl[454 * 2] - wl[234 * 2], (wl[454 * 2 + 1] - wl[234 * 2 + 1]) / kAspect);
            if (prevJawW > 0.0f) {
                maxJawStep = std::max(maxJawStep, std::fabs(jawW - prevJawW));
                maxCheekStep = std::max(maxCheekStep, std::fabs(cheekW - prevCheekW));
            }
            prevJawW = jawW;
            prevCheekW = cheekW;
        }
    }
    EXPECT_TRUE(minSteadyFace >= 0.99f);
    EXPECT_TRUE(maxJawStep <= 0.0018f);
    EXPECT_TRUE(maxCheekStep <= 0.0020f);

    // 2. Async 15Hz inference + 30Hz render extrapolation during gentle head sway:
    //    Inter-sample extrapolated frames must NOT extrapolate scale breathing or pump steadyFace!
    tl.reset();
    def.resetTemporal();
    float maxExtrapWidthSawtooth = 0.0f;
    float minSwaySteadyFace = 1.0f;
    float lastRenderJawW = -1.0f;
    for (int r = 0; r < 40; ++r) {
        const int64_t renderTs = 3000 + r * 33;
        if (r % 2 == 0) {
            const float swayX = 0.50f + 0.045f * std::sin(static_cast<float>(r) * 0.22f);
            const float scaleNoise = (r % 4 == 0) ? 1.010f : 0.990f;
            std::array<float, FaceMeshFrame::kCoordCount> lm{};
            makeCanonicalFace(kAspect, swayX, 0.50f, 0.36f * scaleNoise, 0.0f, 0.0f, 0.0f, lm);
            FaceMeshFrame frame{};
            frame.landmarks = lm;
            frame.hasIris = true;
            EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, frame.meta.data()));
            std::array<float, FaceGeometryConstants::kStateSize> st{};
            EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
                lm.data(), FaceMeshFrame::kCoordCount, frame.axisX(), frame.axisY(), kAspect, 1.0f, st.data()));
            tl.add(renderTs, st.data(), &frame);
        }
        tl.update(renderTs);
        EXPECT_TRUE(tl.at(renderTs) != nullptr);
        FaceTimeline::SmoothedMeshView mv{};
        EXPECT_TRUE(tl.meshAt(renderTs, mv));
        EXPECT_TRUE(def.deform(
            mv.landmarks, kAspect, mv.axisX(), mv.axisY(),
            mv.faceWidth(), mv.faceHeight(),
            mv.leftWeight(), mv.rightWeight(), mv.yawWeight(),
            tl.presence(), tl.steadyFace(), tl.steadyEye(), params, renderTs));
        if (r >= 10) {
            minSwaySteadyFace = std::min(minSwaySteadyFace, tl.steadyFace());
            const float* wl = def.warpedLandmarks();
            const float jawW = std::hypot(wl[397 * 2] - wl[172 * 2], (wl[397 * 2 + 1] - wl[172 * 2 + 1]) / kAspect);
            if (lastRenderJawW > 0.0f) {
                maxExtrapWidthSawtooth = std::max(maxExtrapWidthSawtooth, std::fabs(jawW - lastRenderJawW));
            }
            lastRenderJawW = jawW;
        }
    }
    EXPECT_TRUE(minSwaySteadyFace >= 0.98f);
    EXPECT_TRUE(maxExtrapWidthSawtooth <= 0.0020f);
}

void testV369SlightCameraShakeZeroWaterRippleAndZeroMeshShear() {
    constexpr int kW = 360;
    constexpr int kH = 640;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    const float baseScale = 0.44f / FaceMeshFilter::canonicalFaceWidth();

    auto buildShakenMp = [&](int64_t tsMs, float noiseAmp, int frameIdx, float outMp[478 * 2]) {
        const double t = static_cast<double>(tsMs) * 0.001;
        const float tx = static_cast<float>(0.028 * std::sin(2.0 * 3.1415926535 * 2.7 * t) +
                                            0.011 * std::sin(2.0 * 3.1415926535 * 4.3 * t + 0.7));
        const float ty = static_cast<float>(0.022 * std::cos(2.0 * 3.1415926535 * 2.3 * t + 0.4) +
                                            0.009 * std::sin(2.0 * 3.1415926535 * 3.9 * t));
        const float roll = static_cast<float>(0.038 * std::sin(2.0 * 3.1415926535 * 2.5 * t + 0.3));
        const float yaw = static_cast<float>(0.062 * std::sin(2.0 * 3.1415926535 * 2.1 * t + 0.9));
        const float pitch = static_cast<float>(0.035 * std::cos(2.0 * 3.1415926535 * 2.4 * t + 0.2));
        const float sc = static_cast<float>(1.0 + 0.012 * std::sin(2.0 * 3.1415926535 * 1.9 * t));
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cr = std::cos(roll), sr = std::sin(roll);

        for (int i = 0; i < 468; ++i) {
            const float lx = xyz[i * 3 + 0] * baseScale;
            const float ly = -xyz[i * 3 + 1] * baseScale;
            const float lz = xyz[i * 3 + 2] * baseScale;
            const float x1 = lx, y1 = cp * ly - sp * lz, z1 = sp * ly + cp * lz;
            const float x2 = cy * x1 + sy * z1, y2 = y1;
            float u = 0.5f + tx + (cr * x2 - sr * y2) * sc;
            float vTop = ((0.48f / kAspect) + ty + (sr * x2 + cr * y2) * sc) * kAspect;
            if (noiseAmp > 0.0f) {
                const float rUv = std::hypot(uv[i * 2] - 0.5f, uv[i * 2 + 1] - 0.5f);
                const float mul = (rUv > 0.22f) ? 1.8f : 1.0f;
                const float n1 = std::sin(static_cast<float>(i * 12.9898f + frameIdx * 78.233f));
                const float n2 = std::cos(static_cast<float>(i * 39.3467f + frameIdx * 41.135f));
                u += n1 * noiseAmp * mul;
                vTop += n2 * noiseAmp * mul * kAspect;
            }
            outMp[i * 2 + 0] = u;
            outMp[i * 2 + 1] = vTop;
        }
        for (int i = 468; i < 478; ++i) {
            outMp[i * 2 + 0] = 0.5f + tx;
            outMp[i * 2 + 1] = 0.45f + ty * kAspect;
        }
    };

    auto runShakeCheck = [&](int64_t latencyMs, float noiseAmp, double& outMaxLagPx, double& outMaxShearPx) {
        FaceTimeline tl;
        tl.setAspect(kAspect);
        outMaxLagPx = 0.0;
        outMaxShearPx = 0.0;
        for (int f = 0; f < 55; ++f) {
            const int64_t frameTs = 1000 + f * 33;
            const int64_t lmTs = frameTs - latencyMs;
            float mpXy[478 * 2];
            buildShakenMp(lmTs, noiseAmp, f, mpXy);
            FaceMeshFrame mf{};
            float st41[41]{};
            EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mpXy, 478, kAspect, mf));
            EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
                mf.landmarks.data(), FaceMeshFrame::kCoordCount,
                mf.meta[FaceMeshFrame::kMetaAxisX], mf.meta[FaceMeshFrame::kMetaAxisY],
                kAspect, 1.0f, st41));
            tl.add(lmTs, st41, &mf);
            tl.update(frameTs);
            EXPECT_TRUE(tl.at(frameTs) != nullptr);
            FaceTimeline::SmoothedMeshView mv{};
            EXPECT_TRUE(tl.meshAt(frameTs, mv));
            if (f >= 15) {
                float trueMp[478 * 2];
                buildShakenMp(frameTs, 0.0f, f, trueMp);
                const float bx = (mv.landmarks[168 * 2] - trueMp[168 * 2]) * kW;
                const float by = ((1.0f - mv.landmarks[168 * 2 + 1]) - trueMp[168 * 2 + 1]) * kH;
                outMaxLagPx = std::max(outMaxLagPx, static_cast<double>(std::hypot(bx, by)));
                for (int i = 0; i < 468; ++i) {
                    const float ex = (mv.landmarks[i * 2] - trueMp[i * 2]) * kW;
                    const float ey = ((1.0f - mv.landmarks[i * 2 + 1]) - trueMp[i * 2 + 1]) * kH;
                    outMaxShearPx = std::max(outMaxShearPx, static_cast<double>(std::hypot(ex - bx, ey - by)));
                }
            }
        }
    };

    // 1. Aligned FrameDelay (0ms) with tangential MediaPipe skin noise during 3D slight camera shake:
    //    Old v3.6.8 had lagPx=2.84 and shearPx=9.83 (29.5px@1080p jello shear!). Must keep lag <= 1.20px and shear <= 2.80px!
    double lag0 = 0.0, shear0 = 0.0;
    runShakeCheck(0, 0.0025f, lag0, shear0);
    EXPECT_TRUE(lag0 <= 1.20);
    EXPECT_TRUE(shear0 <= 2.80);

    // 2. Unaligned 1-frame (33ms) SE(2) sub-frame extrapolation fallback under 3D camera shake:
    //    Old v3.6.8 had lagPx=8.93 (26.8px@1080p!) and shearPx=10.36 (31.1px@1080p!).
    //    Curvature-braked SE(2) extrapolation cuts both nearly in half (respecting exact 3D parallax lower bound)!
    double lag33 = 0.0, shear33 = 0.0;
    runShakeCheck(33, 0.0025f, lag33, shear33);
    EXPECT_TRUE(lag33 <= 5.60);
    EXPECT_TRUE(shear33 <= 5.80);
}

void testV3610CheekBeautyStabilityMirroredNoTwitch() {
    constexpr int kW = 360;
    constexpr int kH = 640;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    const float canonW = FaceMeshFilter::canonicalFaceWidth();
    float mx0 = 0.0f, my0 = 0.0f, mz0 = 0.0f;
    for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
        mx0 += xyz[i * 3 + 0];
        my0 += xyz[i * 3 + 1];
        mz0 += xyz[i * 3 + 2];
    }
    mx0 /= FaceMeshFrame::kFaceCount;
    my0 /= FaceMeshFrame::kFaceCount;
    mz0 /= FaceMeshFrame::kFaceCount;

    const std::vector<int> cheekIdx = {
        50, 101, 118, 123, 147, 187, 205, 207, 213, 132, 234, 93, 58, 172,
        280, 330, 347, 352, 376, 411, 425, 427, 433, 361, 454, 323, 288, 397,
    };

    auto makeMirroredMp = [&](float cx, float cyTop, float faceW, float yawRad, float rollRad,
                              int seed, float innerPx, float contourPx) {
        std::array<float, FaceMeshFrame::kCoordCount> mp{};
        auto hnoise = [](int a, int b) {
            uint32_t x = static_cast<uint32_t>(a) * 0x9E3779B9u ^
                         (static_cast<uint32_t>(b) * 0x85EBCA6Bu + 0xC2B2AE35u);
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;
            return static_cast<float>(x & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f;
        };
        const float corrL = (contourPx > 0.0f) ? hnoise(seed, 901) * contourPx * 0.70f / kW : 0.0f;
        const float corrR = (contourPx > 0.0f) ? hnoise(seed, 902) * contourPx * 0.70f / kW : 0.0f;
        const float effYaw = yawRad + ((innerPx > 0.0f) ? hnoise(seed, 903) * 0.012f : 0.0f);
        const float effS = faceW * (1.0f + ((innerPx > 0.0f) ? hnoise(seed, 904) * 0.006f : 0.0f));
        const float cyE = std::cos(effYaw), syE = std::sin(effYaw);
        const float cr = std::cos(rollRad), sr = std::sin(rollRad);

        for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
            float lx = (xyz[i * 3 + 0] - mx0) / canonW;
            const float ly = (xyz[i * 3 + 1] - my0) / canonW;
            const float lz = (xyz[i * 3 + 2] - mz0) / canonW;
            const float nu = (uv[i * 2] - 0.5f) * 2.0f;
            const float nv = (0.5f - uv[i * 2 + 1]) * 2.0f;
            if (nv < 0.15f && nv > -0.85f && std::fabs(nu) > 0.35f) {
                const float w = std::exp(-0.5f * std::pow((nv + 0.32f) / 0.34f, 2.0f)) *
                                detail::smoothstep(0.35f, 0.85f, std::fabs(nu));
                lx *= (1.0f + 0.18f * w);
            }
            const float x1 = -(lx * cyE + lz * syE);  // Mirrored front camera (det < 0)
            const float xr = cr * x1 - sr * ly;
            const float yr = sr * x1 + cr * ly;
            float sx = cx + xr * effS;
            float syTop = 1.0f - ((1.0f - cyTop) + yr * effS * kAspect);
            if (innerPx > 0.0f || contourPx > 0.0f) {
                const float rUv = std::hypot((uv[i * 2] - 0.5f) / 0.36f, (uv[i * 2 + 1] - 0.5f) / 0.43f);
                const float isC = detail::smoothstep(0.62f, 0.90f, rUv);
                const float sig = innerPx * (1.0f - isC) + contourPx * isC;
                sx += hnoise(seed * 503 + i, 1) * sig * 1.4f / kW;
                syTop += hnoise(seed * 503 + i, 2) * sig * 1.4f / kH;
                if (isC > 0.1f) sx += isC * ((-nu < 0.0f) ? corrL : corrR);
            }
            mp[i * 2 + 0] = sx;
            mp[i * 2 + 1] = syTop;
        }
        auto synthIris = [&](int b, int o, int in, int t, int bo) {
            const float icx = 0.25f * (mp[o * 2] + mp[in * 2] + mp[t * 2] + mp[bo * 2]);
            const float icy = 0.25f * (mp[o * 2 + 1] + mp[in * 2 + 1] + mp[t * 2 + 1] + mp[bo * 2 + 1]);
            const float ir = std::hypot(mp[o * 2] - mp[in * 2], (mp[o * 2 + 1] - mp[in * 2 + 1]) / kAspect) * 0.22f;
            mp[b * 2] = icx; mp[b * 2 + 1] = icy;
            mp[(b + 1) * 2] = icx + ir; mp[(b + 1) * 2 + 1] = icy;
            mp[(b + 2) * 2] = icx; mp[(b + 2) * 2 + 1] = icy + ir * kAspect;
            mp[(b + 3) * 2] = icx - ir; mp[(b + 3) * 2 + 1] = icy;
            mp[(b + 4) * 2] = icx; mp[(b + 4) * 2 + 1] = icy - ir * kAspect;
        };
        synthIris(468, 33, 133, 159, 145);
        synthIris(473, 362, 263, 386, 374);
        return mp;
    };

    // 1. Mirrored front-camera (det < 0) in-plane roll extrapolation must recover true omega (not cancel it!).
    {
        const auto mp0 = makeMirroredMp(0.50f, 0.46f, 0.42f, 0.0f, 0.00f, 1, 0.0f, 0.0f);
        const auto mp1 = makeMirroredMp(0.50f, 0.46f, 0.42f, 0.0f, 0.06f, 2, 0.0f, 0.0f);
        const auto mp2 = makeMirroredMp(0.50f, 0.46f, 0.42f, 0.0f, 0.12f, 3, 0.0f, 0.0f);
        FaceMeshFrame f0{}, f1{}, f2{};
        EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mp0.data(), 478, kAspect, f0));
        EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mp1.data(), 478, kAspect, f1));
        EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mp2.data(), 478, kAspect, f2));
        std::array<float, 478> mx0{}, my0{}, mx1{}, my1{};
        for (int i = 0; i < 478; ++i) {
            mx0[i] = f0.landmarks[i * 2]; my0[i] = f0.landmarks[i * 2 + 1] / kAspect;
            mx1[i] = f1.landmarks[i * 2]; my1[i] = f1.landmarks[i * 2 + 1] / kAspect;
        }
        float p0[8], p1[8], vel[8];
        FaceMeshFilter::fitPose(mx0.data(), my0.data(), p0);
        FaceMeshFilter::fitPose(mx1.data(), my1.data(), p1);
        EXPECT_TRUE(FaceMeshFilter::detOf(p1) < 0.0f);
        for (int k = 0; k < 8; ++k) vel[k] = (p1[k] - p0[k]) / 33.333f;
        std::array<float, FaceMeshFrame::kCoordCount> ext{};
        FaceMeshFilter::extrapolate(f1.landmarks.data(), p1, vel, 33.333f, kAspect, ext.data());
        float errMean = 0.0f;
        for (int i : cheekIdx) {
            errMean += std::hypot((ext[i * 2] - f2.landmarks[i * 2]) * kW,
                                  (ext[i * 2 + 1] - f2.landmarks[i * 2 + 1]) * kH);
        }
        errMean /= static_cast<float>(cheekIdx.size());
        EXPECT_TRUE(errMean <= 0.80f);
    }

    // 2. Stationary mirrored human face with realistic MediaPipe jitter:
    //    - FaceMeshFilter 3D Z-column (poseW) must NOT explode into 13px spikes that yank z=-0.43 cheek points;
    //    - FaceMeshDeformer cheek slimming displacement (v205 mid-cheek + v361/v288/v397 outer cheek) must NOT twitch!
    {
        FaceMeshFilter flt;
        FaceMeshDeformer def;
        FaceMeshDeformer::Params dp{};
        dp.faceSlim = 0.65f;
        dp.vFace = 0.50f;
        dp.cheekboneSlim = 0.45f;
        dp.eyeEnlarge = 0.20f;

        std::array<float, FaceMeshFrame::kCoordCount> out{}, prevOut{}, prevDisp{};
        float pose[8]{}, prevPose[8]{};
        float maxFiltCheekPx = 0.0f;
        float maxPoseWPx = 0.0f;
        float maxDeformDispJumpPx = 0.0f;

        for (int f = 0; f < 140; ++f) {
            const int64_t ts = 2000 + f * 33;
            const auto mp = makeMirroredMp(0.50f, 0.46f, 0.44f, 0.20f, 0.0f, 100 + f, 0.85f, 2.5f);
            FaceMeshFrame mf{};
            EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mp.data(), 478, kAspect, mf));
            flt.filter(ts, mf.landmarks.data(), kAspect, out.data(), pose);
            float meta[FaceMeshFrame::kMetaSize]{};
            EXPECT_TRUE(FaceMeshFrame::fillMeta(out.data(), FaceMeshFrame::kCoordCount, kAspect, meta));
            EXPECT_TRUE(def.deform(
                out.data(), kAspect, meta[2], meta[3], meta[4], meta[5],
                meta[7], meta[8], meta[9], 1.0f, 1.0f, 1.0f, dp, ts));
            const float* wl = def.warpedLandmarks();
            if (f >= 15) {
                const float dW = std::hypot(pose[6] - prevPose[6], pose[7] - prevPose[7]) * kW;
                maxPoseWPx = std::max(maxPoseWPx, dW);
                for (int i : cheekIdx) {
                    const float fs = std::hypot((out[i * 2] - prevOut[i * 2]) * kW,
                                                (out[i * 2 + 1] - prevOut[i * 2 + 1]) * kH);
                    maxFiltCheekPx = std::max(maxFiltCheekPx, fs);
                    const float cdx = (wl[i * 2] - out[i * 2]) * kW;
                    const float cdy = (wl[i * 2 + 1] - out[i * 2 + 1]) * kH;
                    const float dj = std::hypot(cdx - prevDisp[i * 2], cdy - prevDisp[i * 2 + 1]);
                    maxDeformDispJumpPx = std::max(maxDeformDispJumpPx, dj);
                }
            }
            prevOut = out;
            for (int i = 0; i < FaceMeshFrame::kFaceCount; ++i) {
                prevDisp[i * 2] = (wl[i * 2] - out[i * 2]) * kW;
                prevDisp[i * 2 + 1] = (wl[i * 2 + 1] - out[i * 2 + 1]) * kH;
            }
            std::memcpy(prevPose, pose, sizeof(pose));
        }
        EXPECT_TRUE(maxPoseWPx <= 2.20f);
        EXPECT_TRUE(maxFiltCheekPx <= 1.35f);
        EXPECT_TRUE(maxDeformDispJumpPx <= 0.45f);
    }

    // 3. FaceRoiSelector deadzone lock: slight handheld sway at mid-distance (fw=0.26) must NOT drift
    //    the 256x256 crop rect every frame into MediaPipe VIDEO tracker, yet large moves must still recenter.
    {
        FaceRoiSelector roi;
        int cropMoves = 0;
        CropRect prev = CropRect::full();
        for (int f = 0; f < 150; ++f) {
            const int64_t ts = 5000 + f * 33;
            const float tSec = static_cast<float>(f * 33) * 0.001f;
            const float cx = 0.50f + 0.022f * std::sin(2.0f * 3.14159f * 0.8f * tSec);
            const float cy = 0.52f + 0.010f * std::cos(2.0f * 3.14159f * 0.7f * tSec);
            const float fw = 0.26f + 0.018f * std::sin(2.0f * 3.14159f * 0.5f * tSec);
            roi.onFace(cx, cy, fw, ts, ts + 20);
            const CropRect c = roi.select(720, 1280, ts + 25);
            EXPECT_TRUE(!c.isFullFrame());
            if (f > 5 && (std::fabs(c.left - prev.left) > 1e-4f ||
                          std::fabs(c.top - prev.top) > 1e-4f ||
                          std::fabs(c.width - prev.width) > 1e-4f)) {
                ++cropMoves;
            }
            prev = c;
        }
        EXPECT_TRUE(cropMoves <= 12);

        // Large walk to left side (cx=0.20) must recenter promptly
        for (int f = 0; f < 25; ++f) {
            const int64_t ts = 10000 + f * 33;
            roi.onFace(0.20f, 0.52f, 0.26f, ts, ts + 20);
        }
        const CropRect recentered = roi.select(720, 1280, 10850);
        EXPECT_TRUE(!recentered.isFullFrame());
        EXPECT_TRUE(recentered.left <= 0.05f);
    }
}

void testV3612PhoneShakeZeroFeatureLag() {
    constexpr int kW = 720;
    constexpr int kH = 1280;
    constexpr float kAspect = static_cast<float>(kW) / static_cast<float>(kH);
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const auto& uv = FaceMeshTopology::kCanonicalUv;
    const float baseScale = 0.44f / FaceMeshFilter::canonicalFaceWidth();
    constexpr int64_t kStopAtMs = 3000;

    auto buildPhoneShakeMp = [&](int64_t tsMs, float noisePx, int frameIdx, float outMp[478 * 2]) {
        const double tRaw = static_cast<double>(std::max<int64_t>(0, tsMs - 1000)) * 0.001;
        const double tStop = static_cast<double>(kStopAtMs - 1000) * 0.001;
        double amp = 1.0;
        if (tRaw < 0.30) amp = 0.0;
        else if (tRaw < 0.45) amp = (tRaw - 0.30) / 0.15;
        else if (tRaw >= tStop) amp = 0.0;
        const double t = (tRaw >= tStop) ? tStop : tRaw;

        const float tx = static_cast<float>(amp * (0.095 * std::sin(2.0 * 3.1415926535 * 2.1 * t) +
                                                   0.032 * std::sin(2.0 * 3.1415926535 * 3.6 * t + 0.5)));
        const float ty = static_cast<float>(amp * (0.055 * std::cos(2.0 * 3.1415926535 * 1.8 * t + 0.3)));
        const float roll = static_cast<float>(amp * 0.11 * std::sin(2.0 * 3.1415926535 * 2.0 * t + 0.4));
        const float yaw = static_cast<float>(amp * 0.24 * std::sin(2.0 * 3.1415926535 * 1.9 * t + 0.7));
        const float pitch = static_cast<float>(amp * 0.12 * std::cos(2.0 * 3.1415926535 * 2.2 * t + 0.2));
        const float sc = static_cast<float>(1.0 + amp * 0.08 * std::sin(2.0 * 3.1415926535 * 1.6 * t));
        const float rsSkew = static_cast<float>(amp * 0.045 * std::cos(2.0 * 3.1415926535 * 2.1 * t));
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cr = std::cos(roll), sr = std::sin(roll);
        auto hnoise = [](int a, int b) {
            uint32_t x = static_cast<uint32_t>(a) * 0x9E3779B9u ^
                         (static_cast<uint32_t>(b) * 0x85EBCA6Bu + 0xC2B2AE35u);
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;
            return static_cast<float>(x & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f;
        };

        for (int i = 0; i < 468; ++i) {
            const float lx = xyz[i * 3 + 0] * baseScale;
            const float ly = -xyz[i * 3 + 1] * baseScale;
            const float lz = xyz[i * 3 + 2] * baseScale;
            const float y1 = cp * ly - sp * lz;
            const float z1 = sp * ly + cp * lz;
            const float x2 = -(cy * lx + sy * z1);
            const float z2 = -sy * lx + cy * z1;
            const float persp = 1.0f / std::max(0.5f, 1.0f - 0.32f * z2 * sc);
            const float xr = (cr * x2 - sr * y1) * sc * persp;
            const float yr = (sr * x2 + cr * y1) * sc * persp;
            float u = 0.5f + tx + xr + rsSkew * yr;
            float vTop = ((0.48f / kAspect) + ty + yr) * kAspect;
            if (noisePx > 0.0f) {
                const float rUv = std::hypot((uv[i * 2] - 0.5f) / 0.36f, (uv[i * 2 + 1] - 0.5f) / 0.43f);
                const float isC = detail::smoothstep(0.62f, 0.90f, rUv);
                const float sig = noisePx * (1.0f + 1.5f * isC);
                u += hnoise(frameIdx * 503 + i, 1) * sig / kW;
                vTop += hnoise(frameIdx * 503 + i, 2) * sig / kH;
            }
            outMp[i * 2 + 0] = u;
            outMp[i * 2 + 1] = vTop;
        }
        auto synthIris = [&](int b, int o, int in, int tIdx, int bo) {
            const float icx = 0.25f * (outMp[o * 2] + outMp[in * 2] + outMp[tIdx * 2] + outMp[bo * 2]);
            const float icy = 0.25f * (outMp[o * 2 + 1] + outMp[in * 2 + 1] + outMp[tIdx * 2 + 1] + outMp[bo * 2 + 1]);
            const float ir = std::hypot(outMp[o * 2] - outMp[in * 2], (outMp[o * 2 + 1] - outMp[in * 2 + 1]) / kAspect) * 0.22f;
            outMp[b * 2] = icx; outMp[b * 2 + 1] = icy;
            outMp[(b + 1) * 2] = icx + ir; outMp[(b + 1) * 2 + 1] = icy;
            outMp[(b + 2) * 2] = icx; outMp[(b + 2) * 2 + 1] = icy + ir * kAspect;
            outMp[(b + 3) * 2] = icx - ir; outMp[(b + 3) * 2 + 1] = icy;
            outMp[(b + 4) * 2] = icx; outMp[(b + 4) * 2 + 1] = icy - ir * kAspect;
        };
        synthIris(468, 33, 133, 159, 145);
        synthIris(473, 362, 263, 386, 374);
    };

    // 1. Handheld 2D+3D+perspective+zoom phone shake on mirrored front camera:
    //    nose/lips/eyes must track within a few pixels, steadyEye must stay locked (no shrink/reinflate pumping),
    //    and abrupt stop must settle landmarks in <= 1 frame (33ms) and deformer features in <= 3 frames (99ms).
    FaceTimeline tl;
    tl.setAspect(kAspect);
    FaceMeshDeformer def, defRef;
    FaceMeshDeformer::Params dp{};
    dp.faceSlim = 0.65f;
    dp.eyeEnlarge = 0.65f;
    dp.noseSlim = 0.65f;
    dp.mouthSize = -0.40f;
    dp.mouthSmile = 0.55f;

    float refMp[478 * 2];
    buildPhoneShakeMp(kStopAtMs + 500, 0.0f, 0, refMp);
    FaceMeshFrame refMf{};
    EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(refMp, 478, kAspect, refMf));
    EXPECT_TRUE(defRef.deform(
        refMf.landmarks.data(), kAspect, refMf.axisX(), refMf.axisY(),
        refMf.faceWidth(), refMf.faceHeight(), refMf.leftWeight(), refMf.rightWeight(),
        refMf.yawWeight(), 1.0f, 1.0f, 1.0f, dp, 1000));
    const float refNoseW = std::fabs(defRef.warpedLandmarks()[48 * 2] - defRef.warpedLandmarks()[278 * 2]) * kW;
    const float refEyeH = std::fabs(defRef.warpedLandmarks()[159 * 2 + 1] - defRef.warpedLandmarks()[145 * 2 + 1]) * kH;
    const float refMouthW = std::fabs(defRef.warpedLandmarks()[291 * 2] - defRef.warpedLandmarks()[61 * 2]) * kW;

    float maxNose = 0.0f, sumNose = 0.0f;
    float maxLip = 0.0f, sumLip = 0.0f;
    float maxEye = 0.0f, sumEye = 0.0f;
    float minSteadyEye = 1.0f;
    int shakeFrames = 0;
    int settleLmFrame = -1;
    int settleDefFrame = -1;

    for (int f = 0; f < 95; ++f) {
        const int64_t ts = 1000 + f * 33;
        float mp[478 * 2];
        buildPhoneShakeMp(ts, 1.1f, f, mp);
        FaceMeshFrame mf{};
        float st[41]{};
        EXPECT_TRUE(FaceMeshFrame::fromMediaPipe(mp, 478, kAspect, mf));
        EXPECT_TRUE(FaceMeshFrame::fillGeometryState(
            mf.landmarks.data(), FaceMeshFrame::kCoordCount, mf.axisX(), mf.axisY(), kAspect, 1.0f, st));
        tl.add(ts, st, &mf);
        tl.update(ts);
        EXPECT_TRUE(tl.at(ts) != nullptr);
        FaceTimeline::SmoothedMeshView mv{};
        EXPECT_TRUE(tl.meshAt(ts, mv));
        EXPECT_TRUE(def.deform(
            mv.landmarks, kAspect, mv.axisX(), mv.axisY(),
            mv.faceWidth(), mv.faceHeight(), mv.leftWeight(), mv.rightWeight(),
            mv.yawWeight(), tl.presence(), tl.steadyFace(), tl.steadyEye(), dp, ts));

        float trueMp[478 * 2];
        buildPhoneShakeMp(ts, 0.0f, f, trueMp);
        auto errPx = [&](int idx) {
            const float dx = (mv.landmarks[idx * 2] - trueMp[idx * 2]) * kW;
            const float dy = ((1.0f - mv.landmarks[idx * 2 + 1]) - trueMp[idx * 2 + 1]) * kH;
            return std::hypot(dx, dy);
        };
        const float eNose = std::max({errPx(1), errPx(4), errPx(48), errPx(278)});
        const float eLip = std::max({errPx(13), errPx(14), errPx(61), errPx(291), errPx(0), errPx(17)});
        const float eEye = std::max({errPx(33), errPx(133), errPx(159), errPx(145), errPx(263), errPx(362)});

        if (ts >= 1500 && ts < kStopAtMs) {
            maxNose = std::max(maxNose, eNose); sumNose += eNose;
            maxLip = std::max(maxLip, eLip); sumLip += eLip;
            maxEye = std::max(maxEye, eEye); sumEye += eEye;
            minSteadyEye = std::min(minSteadyEye, tl.steadyEye());
            ++shakeFrames;
        }
        if (ts >= kStopAtMs) {
            const int postF = static_cast<int>((ts - kStopAtMs) / 33);
            if (settleLmFrame < 0 && std::max({eNose, eLip, eEye}) <= 2.5f) settleLmFrame = postF;
            const float gotNoseW = std::fabs(def.warpedLandmarks()[48 * 2] - def.warpedLandmarks()[278 * 2]) * kW;
            const float gotEyeH = std::fabs(def.warpedLandmarks()[159 * 2 + 1] - def.warpedLandmarks()[145 * 2 + 1]) * kH;
            const float gotMouthW = std::fabs(def.warpedLandmarks()[291 * 2] - def.warpedLandmarks()[61 * 2]) * kW;
            if (settleDefFrame < 0 &&
                std::fabs(gotNoseW - refNoseW) <= 1.2f &&
                std::fabs(gotEyeH - refEyeH) <= 1.2f &&
                std::fabs(gotMouthW - refMouthW) <= 1.2f &&
                tl.steadyEye() >= 0.95f && tl.eyeMakeupKeep() >= 0.85f) {
                settleDefFrame = postF;
            }
        }
    }
    EXPECT_TRUE(shakeFrames > 30);
    EXPECT_TRUE(maxNose <= 2.5f && (sumNose / shakeFrames) <= 1.2f);
    EXPECT_TRUE(maxLip <= 5.0f && (sumLip / shakeFrames) <= 2.0f);
    EXPECT_TRUE(maxEye <= 3.8f && (sumEye / shakeFrames) <= 1.6f);
    EXPECT_TRUE(minSteadyEye >= 0.92f);
    EXPECT_TRUE(settleLmFrame >= 0 && settleLmFrame <= 1);
    EXPECT_TRUE(settleDefFrame >= 0 && settleDefFrame <= 3);

    // 2. FaceMeshDeformer steadyEye recovery must switch to kDispTauFastMs (18ms) instead of lingering at 95ms:
    //    within 2 frames (66ms) after steadyEye recovers 0.25 -> 1.0, warped eye/nose/mouth must reach >= 95% target.
    FaceMeshDeformer defRec;
    for (int f = 0; f < 6; ++f) {
        EXPECT_TRUE(defRec.deform(
            refMf.landmarks.data(), kAspect, refMf.axisX(), refMf.axisY(),
            refMf.faceWidth(), refMf.faceHeight(), refMf.leftWeight(), refMf.rightWeight(),
            refMf.yawWeight(), 1.0f, 1.0f, 0.25f, dp, 5000 + f * 33));
    }
    for (int f = 1; f <= 2; ++f) {
        EXPECT_TRUE(defRec.deform(
            refMf.landmarks.data(), kAspect, refMf.axisX(), refMf.axisY(),
            refMf.faceWidth(), refMf.faceHeight(), refMf.leftWeight(), refMf.rightWeight(),
            refMf.yawWeight(), 1.0f, 1.0f, 1.0f, dp, 5165 + f * 33));
    }
    const float recNoseW = std::fabs(defRec.warpedLandmarks()[48 * 2] - defRec.warpedLandmarks()[278 * 2]) * kW;
    const float recEyeH = std::fabs(defRec.warpedLandmarks()[159 * 2 + 1] - defRec.warpedLandmarks()[145 * 2 + 1]) * kH;
    EXPECT_NEAR(recNoseW, refNoseW, 0.8f);
    EXPECT_NEAR(recEyeH, refEyeH, 0.8f);
}

void testV3613HairCoverageGhostFreeEmaAndAffineMargin() {
    // 1. Moderate real-world hair confidence (c[1]=2.0 vs c[0]=1.4 -> p[1]≈0.646, pSkin≈0)
    //    MUST yield strong hair coverage (>= 200/255) instead of being crushed by high 0.55..0.88 threshold!
    constexpr int kS = 16;
    std::vector<float> logits(kS * kS * 6, -3.5f);
    for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
            float* c = logits.data() + static_cast<size_t>((y * kS + x) * 6);
            if (y < kS / 2) {
                c[1] = 2.0f;  // moderate hair confidence (~0.646)
                c[0] = 1.4f;
            } else {
                c[3] = 2.5f;  // clear face skin
                c[1] = 0.2f;
            }
        }
    }
    FaceParsingMapper mapper;
    std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
    EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwc(logits.data(), kS, kS, occ.data(), hair.data(), person.data()));
    EXPECT_TRUE(hair[24 * 128 + 64] >= 200);
    EXPECT_TRUE(hair[100 * 128 + 64] < 5);

    // 2. Responsive hair EMA snaps immediately when hair moves to a new region (zero temporal ghost smear!)
    std::fill(logits.begin(), logits.end(), -3.5f);
    for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
            float* c = logits.data() + static_cast<size_t>((y * kS + x) * 6);
            if (y >= kS / 2) {
                c[1] = 4.0f;  // moved to bottom half
            } else {
                c[0] = 4.0f;  // top half is now background
            }
        }
    }
    EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwc(logits.data(), kS, kS, occ.data(), hair.data(), person.data()));
    EXPECT_TRUE(hair[24 * 128 + 64] <= 15);
    EXPECT_TRUE(hair[100 * 128 + 64] >= 240);

    // 3. HairMaskTuning::affine tolerates realistic 14% head displacement between 50ms segmentation ticks
    //    while still rejecting stale 25% jumps (> kAffineMaxShift = 0.18).
    float aff[HairMaskTuning::kAffineSize];
    const float poseMask[6] = {0.43f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};
    const float poseMid[6] = {0.57f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};   // shift = 0.14 <= 0.18 -> valid!
    const float poseJump[6] = {0.68f, 0.50f, 0.20f, 0.25f, 1.0f, 0.0f};  // shift = 0.25 > 0.18 -> rejected!
    EXPECT_TRUE(HairMaskTuning::affine(poseMask, poseMid, 0.5625f, aff));
    EXPECT_TRUE(!HairMaskTuning::affine(poseMask, poseJump, 0.5625f, aff));
}

void testV3615DouyinAmbientSceneLiftCurve() {
    // Non-finite (unmeasured) stays 0.0
    EXPECT_NEAR(BeautyQualityTuning::ambientSceneLift(std::numeric_limits<float>::quiet_NaN()), 0.0f, 1e-6f);
    // v3.9.4 (A1/tone-1): Strictly capped to <= 0.18 in genuine darkness (L < 0.30);
    // normal AE-normalized indoor room (0.42) and sunny daylight (0.78) receive 0.0 lift.
    const float veryDark = BeautyQualityTuning::ambientSceneLift(0.10f);
    const float dim = BeautyQualityTuning::ambientSceneLift(0.18f);
    const float indoor = BeautyQualityTuning::ambientSceneLift(0.42f);
    const float sunny = BeautyQualityTuning::ambientSceneLift(0.78f);
    EXPECT_NEAR(veryDark, 0.18f, 1e-5f);
    EXPECT_TRUE(dim > 0.05f && dim < veryDark);
    EXPECT_NEAR(indoor, 0.0f, 1e-5f);
    EXPECT_NEAR(sunny, 0.0f, 1e-5f);
}

void testV370TwoStageSmoothIndoorDenoiseMouthGateAndProfileGuard() {
    // 1. Adaptive extreme-dark temporal stabilization curve (v3.9.4 A5/tone-2: zero in normal indoor >= 0.14)
    EXPECT_NEAR(BeautyQualityTuning::indoorTemporalDenoise(std::numeric_limits<float>::quiet_NaN()), 0.0f, 1e-6f);
    const float dDim = BeautyQualityTuning::indoorTemporalDenoise(0.08f);
    const float dIndoor = BeautyQualityTuning::indoorTemporalDenoise(0.24f);
    const float dBright = BeautyQualityTuning::indoorTemporalDenoise(0.40f);
    EXPECT_NEAR(dDim, 0.18f, 1e-4f);
    EXPECT_NEAR(dIndoor, 0.0f, 1e-5f);
    EXPECT_NEAR(dBright, 0.0f, 1e-5f);

    // 2. Two-stage commercial smoothing curve (v3.9.4 A4/skin-2): headroom above 0.55..0.60, ceiling 0.96 at 0.85..1.0
    EXPECT_NEAR(BeautyQualityTuning::twoStageSmoothMix(0.0f), 0.0f, 1e-6f);
    EXPECT_TRUE(BeautyQualityTuning::twoStageSmoothMix(0.55f) > 0.54f &&
                BeautyQualityTuning::twoStageSmoothMix(0.55f) < 0.62f);
    EXPECT_TRUE(BeautyQualityTuning::twoStageSmoothMix(0.60f) > 0.60f &&
                BeautyQualityTuning::twoStageSmoothMix(0.60f) < 0.70f);
    EXPECT_NEAR(BeautyQualityTuning::twoStageSmoothMix(0.85f), 0.96f, 1e-4f);
    EXPECT_NEAR(BeautyQualityTuning::twoStageSmoothMix(1.0f), 0.96f, 1e-4f);
    EXPECT_TRUE(BeautyQualityTuning::twoStageSmoothMix(0.30f) > 0.25f &&
                BeautyQualityTuning::twoStageSmoothMix(0.30f) < 0.36f);
    EXPECT_NEAR(BeautyQualityTuning::twoStageDetailKeep(0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(BeautyQualityTuning::twoStageDetailKeep(0.60f), 1.0f, 1e-6f);
    EXPECT_TRUE(BeautyQualityTuning::twoStageDetailKeep(0.80f) > 0.65f &&
                BeautyQualityTuning::twoStageDetailKeep(0.80f) < 0.80f);
    EXPECT_NEAR(BeautyQualityTuning::twoStageDetailKeep(1.0f), 0.45f, 1e-4f);

    // 3. Open-mouth teeth whitening gate
    EXPECT_NEAR(BeautyQualityTuning::mouthOpenGate(std::numeric_limits<float>::quiet_NaN()), 0.0f, 1e-6f);
    EXPECT_NEAR(BeautyQualityTuning::mouthOpenGate(0.04f), 0.0f, 1e-6f);
    const float halfOpen = BeautyQualityTuning::mouthOpenGate(0.095f);
    EXPECT_TRUE(halfOpen > 0.35f && halfOpen < 0.65f);
    EXPECT_NEAR(BeautyQualityTuning::mouthOpenGate(0.16f), 1.0f, 1e-5f);

    // 4. Extreme 3/4 side-profile asymmetric narrow-side contour guard (width ratio > 2.55x):
    //    Narrow far-side cheek contour MUST be protected against inward silhouette collapse!
    constexpr float kAspect = 9.0f / 16.0f;
    std::array<float, FaceMeshFrame::kCoordCount> yawGl{};
    makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.42f, 0.48f, 0.0f, 0.0f, yawGl);
    float meta[FaceMeshFrame::kMetaSize]{};
    EXPECT_TRUE(FaceMeshFrame::fillMeta(yawGl.data(), FaceMeshFrame::kCoordCount, kAspect, meta));

    // Confirm steep asymmetric ratio (> 2.2x between wide and narrow cheek halves)
    const float noseX = yawGl[1 * 2];
    const float dist234 = std::fabs(yawGl[234 * 2] - noseX);
    const float dist454 = std::fabs(yawGl[454 * 2] - noseX);
    const bool leftIsNarrow = (dist234 < dist454);
    EXPECT_TRUE(std::max(dist234, dist454) / std::max(std::min(dist234, dist454), 1e-4f) > 2.2f);

    FaceMeshDeformer deformer;
    FaceMeshDeformer::Params dp{};
    dp.faceSlim = 0.85f;
    dp.vFace = 0.85f;
    dp.cheekboneSlim = 0.65f;
    EXPECT_TRUE(deformer.deform(
        yawGl.data(), kAspect,
        meta[FaceMeshFrame::kMetaAxisX], meta[FaceMeshFrame::kMetaAxisY],
        meta[FaceMeshFrame::kMetaFaceWidth], meta[FaceMeshFrame::kMetaFaceHeight],
        meta[FaceMeshFrame::kMetaLeftWeight], meta[FaceMeshFrame::kMetaRightWeight],
        meta[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, dp, 1000));

    const float* warped = deformer.warpedLandmarks();
    const int narrowCheek = leftIsNarrow ? 132 : 361;
    const int wideCheek = leftIsNarrow ? 361 : 132;
    const float narrowLatDisp = std::fabs(warped[narrowCheek * 2] - yawGl[narrowCheek * 2]);
    const float wideLatDisp = std::fabs(warped[wideCheek * 2] - yawGl[wideCheek * 2]);
    // Narrow far-side cheek lateral slimming stays strictly guarded (< 0.0025 screenW and < 25% of wide near-side)
    EXPECT_TRUE(wideLatDisp > 0.003f);
    EXPECT_TRUE(narrowLatDisp < 0.0025f && narrowLatDisp < wideLatDisp * 0.25f);
}

void testV380Cat02WhiteBalanceStyleAtmosphereAndLowerFaceOcclusion() {
    // 1. Exact CIE D65 CAT02 chromatic adaptation matrix
    float id[9]{};
    BeautyQualityTuning::cat02WhiteBalanceMatrix(0.0f, 0.0f, id);
    EXPECT_NEAR(id[0], 1.0f, 1e-6f); EXPECT_NEAR(id[4], 1.0f, 1e-6f); EXPECT_NEAR(id[8], 1.0f, 1e-6f);
    EXPECT_NEAR(id[1], 0.0f, 1e-6f); EXPECT_NEAR(id[2], 0.0f, 1e-6f); EXPECT_NEAR(id[3], 0.0f, 1e-6f);

    auto applyMat = [](const float m[9], float r, float g, float b, float& oR, float& oG, float& oB) {
        oR = m[0] * r + m[1] * g + m[2] * b;
        oG = m[3] * r + m[4] * g + m[5] * b;
        oB = m[6] * r + m[7] * g + m[8] * b;
    };
    float warmM[9]{}, coolM[9]{}, magM[9]{}, grnM[9]{};
    BeautyQualityTuning::cat02WhiteBalanceMatrix(0.60f, 0.0f, warmM);
    BeautyQualityTuning::cat02WhiteBalanceMatrix(-0.60f, 0.0f, coolM);
    BeautyQualityTuning::cat02WhiteBalanceMatrix(0.0f, 0.60f, magM);
    BeautyQualityTuning::cat02WhiteBalanceMatrix(0.0f, -0.60f, grnM);

    float r, g, b;
    applyMat(warmM, 0.5f, 0.5f, 0.5f, r, g, b);
    EXPECT_TRUE(r > b + 0.06f);
    applyMat(coolM, 0.5f, 0.5f, 0.5f, r, g, b);
    EXPECT_TRUE(b > r + 0.06f);
    applyMat(magM, 0.5f, 0.5f, 0.5f, r, g, b);
    EXPECT_TRUE(0.5f * (r + b) > g + 0.03f);
    applyMat(grnM, 0.5f, 0.5f, 0.5f, r, g, b);
    EXPECT_TRUE(g > 0.5f * (r + b) + 0.03f);

    // 2. Style-coupled atmosphere resolver (v3.9.4 D4/tone-5: automatic bloom & grain disabled by default)
    float bl = 0.0f, gr = 0.0f;
    BeautyQualityTuning::resolveStyleAtmosphere("asset://luts/night_vibe.png", 1.0f, 0, 0.0f, bl, gr);
    EXPECT_NEAR(bl, 0.0f, 1e-5f); EXPECT_NEAR(gr, 0.0f, 1e-5f);
    BeautyQualityTuning::resolveStyleAtmosphere("asset://luts/pure_pink.png", 0.8f, 0, 0.0f, bl, gr);
    EXPECT_NEAR(bl, 0.0f, 1e-5f); EXPECT_NEAR(gr, 0.0f, 1e-5f);
    BeautyQualityTuning::resolveStyleAtmosphere("asset://luts/natural_studio.png", 1.0f, 0, 0.0f, bl, gr);
    EXPECT_NEAR(bl, 0.0f, 1e-5f); EXPECT_NEAR(gr, 0.0f, 1e-5f);
    BeautyQualityTuning::resolveStyleAtmosphere("", 0.0f, 0, 0.0f, bl, gr);
    EXPECT_NEAR(bl, 0.0f, 1e-5f); EXPECT_NEAR(gr, 0.0f, 1e-5f);

    // 3. Lower-face makeup occlusion mute curve
    EXPECT_NEAR(BeautyQualityTuning::lowerFaceMakeupKeep(std::numeric_limits<float>::quiet_NaN()), 1.0f, 1e-5f);
    EXPECT_NEAR(BeautyQualityTuning::lowerFaceMakeupKeep(0.30f), 1.0f, 1e-5f);
    const float midKeep = BeautyQualityTuning::lowerFaceMakeupKeep(0.615f);
    EXPECT_TRUE(midKeep > 0.35f && midKeep < 0.65f);
    EXPECT_NEAR(BeautyQualityTuning::lowerFaceMakeupKeep(0.82f), 0.0f, 1e-5f);

    // 4. FaceOcclusionSolver lowerFaceOcclusion() distinguishes clean skin vs large cup covering lower face
    constexpr int kS = 128;
    std::vector<uint8_t> rgba(kS * kS * 4, 255);
    for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
            const int o = (y * kS + x) * 4;
            rgba[o] = 212; rgba[o + 1] = 172; rgba[o + 2] = 150;
        }
    }
    CropRect fb{0.15f, 0.15f, 0.70f, 0.70f};
    std::array<uint8_t, FaceOcclusionSolver::kMaskPixels> mask{};
    FaceOcclusionSolver occ;
    occ.compute(rgba.data(), kS, kS, &fb, mask.data());
    EXPECT_TRUE(occ.lowerFaceOcclusion() < 0.05f);

    // Cover lower half of face with a large blue/cyan bottle
    for (int y = 58; y <= 112; ++y) {
        for (int x = 22; x <= 106; ++x) {
            const int o = (y * kS + x) * 4;
            rgba[o] = 20; rgba[o + 1] = 145; rgba[o + 2] = 210;
        }
    }
    occ.reset();
    occ.compute(rgba.data(), kS, kS, &fb, mask.data());
    EXPECT_TRUE(occ.lowerFaceOcclusion() > 0.72f);
    EXPECT_TRUE(BeautyQualityTuning::lowerFaceMakeupKeep(occ.lowerFaceOcclusion()) < 0.15f);

    // 5. FaceParsingMapper::estimateLowerFaceOcclusion()
    std::array<uint8_t, FaceParsingMapper::kOutPixels> pOcc{};
    pOcc.fill(255);
    EXPECT_NEAR(FaceParsingMapper::estimateLowerFaceOcclusion(pOcc.data(), FaceParsingMapper::kOutSize, fb), 0.0f, 1e-5f);
    for (int y = 60; y < FaceParsingMapper::kOutSize; ++y) {
        for (int x = 0; x < FaceParsingMapper::kOutSize; ++x) {
            pOcc[y * FaceParsingMapper::kOutSize + x] = 0;
        }
    }
    EXPECT_TRUE(FaceParsingMapper::estimateLowerFaceOcclusion(pOcc.data(), FaceParsingMapper::kOutSize, fb) > 0.90f);

    // 6. EffectBundleParser parses bloom/filmGrain/colorTemp/colorTint
    bundle::EffectBundleManifest mf{};
    EXPECT_TRUE(bundle::EffectBundleParser::parseJson(
        R"({"schemaVersion":1,"id":"atm","beauty":{"bloom":0.35,"filmGrain":0.12,"colorTemp":0.25,"colorTint":-0.15}})",
        mf, nullptr));
    EXPECT_NEAR(mf.beauty.bloom, 0.35f, 1e-5f);
    EXPECT_NEAR(mf.beauty.filmGrain, 0.12f, 1e-5f);
    EXPECT_NEAR(mf.beauty.colorTemp, 0.25f, 1e-5f);
    EXPECT_NEAR(mf.beauty.colorTint, -0.15f, 1e-5f);
    EXPECT_TRUE(mf.beauty.isProcessing());
}

void testV390AutoWhiteBalanceAndMulticlass6PlanarProbabilities() {
    // 1. Neutral D65 ambientRgb=(1,1,1) produces exact (0,0) correction
    float t = 99.0f, g = 99.0f;
    BeautyQualityTuning::estimateAutoWhiteBalance({1.0f, 1.0f, 1.0f}, 1.0f, t, g);
    EXPECT_NEAR(t, 0.0f, 1e-6f);
    EXPECT_NEAR(g, 0.0f, 1e-6f);

    // 2. Warm tungsten room (r=1.24, g=1.00, b=0.76) -> negative CAT02 temp (cooling)
    BeautyQualityTuning::estimateAutoWhiteBalance({1.24f, 1.00f, 0.76f}, 0.85f, t, g);
    EXPECT_TRUE(t < -0.25f && t >= -0.42f);
    EXPECT_TRUE(std::fabs(g) < 0.05f);

    // 3. Cool greenish fluorescent room (r=0.84, g=1.14, b=1.16) -> warming temp > 0 & magenta tint > 0
    BeautyQualityTuning::estimateAutoWhiteBalance({0.84f, 1.14f, 1.16f}, 0.90f, t, g);
    EXPECT_TRUE(t > 0.15f && t <= 0.42f);
    EXPECT_TRUE(g > 0.08f && g <= 0.26f);

    // 4. Zero strength & NaN safety
    BeautyQualityTuning::estimateAutoWhiteBalance({1.3f, 0.9f, 0.7f}, 0.0f, t, g);
    EXPECT_NEAR(t, 0.0f, 1e-6f);
    BeautyQualityTuning::estimateAutoWhiteBalance({std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f}, 1.0f, t, g);
    EXPECT_TRUE(std::isfinite(t) && std::isfinite(g));

    // 5. FaceParsingMapper::mapMulticlass6PlanarProbabilities (iOS MPPImageSegmenter 6-plane direct path)
    FaceParsingMapper mapper;
    constexpr int kW = 32, kH = 32;
    std::vector<float> p0(kW * kH, 0.0f), p1(kW * kH, 0.0f), p2(kW * kH, 0.0f);
    std::vector<float> p3(kW * kH, 0.0f), p4(kW * kH, 0.0f), p5(kW * kH, 0.0f);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int i = y * kW + x;
            if (y < 8) {
                p1[i] = 0.96f; p0[i] = 0.04f;       // Top band: hair
            } else if (y < 20) {
                p3[i] = 0.95f; p0[i] = 0.05f;       // Middle band: bare face skin
            } else {
                p5[i] = 0.94f; p0[i] = 0.06f;       // Lower band: mask/hand occluder (accessories)
            }
        }
    }
    const float* planes[6] = {p0.data(), p1.data(), p2.data(), p3.data(), p4.data(), p5.data()};
    std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
    EXPECT_TRUE(mapper.mapMulticlass6PlanarProbabilities(planes, kW, kH, occ.data(), hair.data(), person.data()));
    // Top hair row -> high hair mask, occluded for makeup; middle face row -> visible (255), hair=0; bottom row -> occluded (0)
    const int kS = FaceParsingMapper::kOutSize;
    EXPECT_TRUE(hair[16 * kS + 64] > 200);
    EXPECT_TRUE(hair[56 * kS + 64] < 20);
    EXPECT_TRUE(occ[56 * kS + 64] > 235);
    EXPECT_TRUE(occ[104 * kS + 64] < 30);
    EXPECT_TRUE(person[56 * kS + 64] > 220);

    // Lower-face occlusion estimator reports > 0.70 on lower-masked face box
    const CropRect fb{0.10f, 0.20f, 0.80f, 0.75f};
    EXPECT_TRUE(FaceParsingMapper::estimateLowerFaceOcclusion(occ.data(), kS, fb) > 0.65f);

    // Null plane guard
    const float* badPlanes[6] = {p0.data(), nullptr, p2.data(), p3.data(), p4.data(), p5.data()};
    EXPECT_TRUE(!mapper.mapMulticlass6PlanarProbabilities(badPlanes, kW, kH, occ.data()));

    // 6. EffectBundleParser parses autoWhiteBalance
    bundle::EffectBundleManifest mf{};
    EXPECT_TRUE(bundle::EffectBundleParser::parseJson(
        R"({"schemaVersion":1,"id":"awb","beauty":{"autoWhiteBalance":0.75}})",
        mf, nullptr));
    EXPECT_NEAR(mf.beauty.autoWhiteBalance, 0.75f, 1e-5f);
    EXPECT_TRUE(mf.beauty.isProcessing());
}

void testV394AuditFixesMathAndTracking() {
    constexpr float kAspect = 9.0f / 16.0f;

    // 1. C1 (makeup-1/seg-1): Hand (class 2 body_skin) inside inner face ellipse occludes makeup & triggers lowerFaceOcc,
    //    while body_skin outside face ellipse (neck/chest) remains 255 (visible).
    {
        constexpr int kW = 64, kH = 64;
        std::vector<float> logits(kW * kH * 6, -5.0f);
        const CropRect faceBounds{0.25f, 0.20f, 0.50f, 0.50f};
        for (int y = 0; y < kH; ++y) {
            const float v = (y + 0.5f) / kH;
            for (int x = 0; x < kW; ++x) {
                const float u = (x + 0.5f) / kW;
                const int base = (y * kW + x) * 6;
                if (u >= 0.35f && u <= 0.65f && v >= 0.45f && v <= 0.68f) {
                    // Hand (body_skin = class 2) resting on lower face/mouth
                    logits[base + 2] = 6.0f;
                } else if (v >= 0.72f && v <= 0.95f && u >= 0.30f && u <= 0.70f) {
                    // Neck/chest (body_skin = class 2) outside face bounds
                    logits[base + 2] = 6.0f;
                } else if (u >= 0.25f && u <= 0.75f && v >= 0.20f && v <= 0.70f) {
                    // Visible face_skin (class 3)
                    logits[base + 3] = 6.0f;
                } else {
                    logits[base + 0] = 6.0f;
                }
            }
        }
        FaceParsingMapper mapper;
        std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
        EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwc(
            logits.data(), kW, kH, occ.data(), hair.data(), person.data(), &faceBounds, nullptr));
        const int S = FaceParsingMapper::kOutSize;
        const int handIdx = static_cast<int>(0.55f * S) * S + static_cast<int>(0.50f * S);
        const int foreheadIdx = static_cast<int>(0.30f * S) * S + static_cast<int>(0.50f * S);
        const int neckIdx = static_cast<int>(0.85f * S) * S + static_cast<int>(0.50f * S);
        EXPECT_TRUE(occ[handIdx] < 40);
        EXPECT_TRUE(occ[foreheadIdx] >= 240);
        EXPECT_TRUE(occ[neckIdx] >= 240);
        EXPECT_TRUE(FaceParsingMapper::estimateLowerFaceOcclusion(occ.data(), S, faceBounds) > 0.35f);

        // E2: Letterbox-aware mapMulticlass6LogitsNhwcRegion produces identical hand occlusion with padding
        std::vector<float> lbLogits(128 * 128 * 6, -5.0f);
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                for (int c = 0; c < 6; ++c) {
                    lbLogits[((y + 32) * 128 + (x + 32)) * 6 + c] = logits[(y * kW + x) * 6 + c];
                }
            }
        }
        mapper.reset();
        std::array<uint8_t, FaceParsingMapper::kOutPixels> occLb{};
        EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwcRegion(
            lbLogits.data(), 128, 128, 32, 32, kW, kH, occLb.data(), nullptr, nullptr, &faceBounds, nullptr));
        EXPECT_TRUE(occLb[handIdx] < 40);
        EXPECT_TRUE(occLb[neckIdx] >= 240);
    }

    // 2. C4 (shape-3): faceNarrow + cheekboneSlim must NOT squash eye width horizontally (< 2.5% drift).
    {
        std::array<float, FaceMeshFrame::kCoordCount> lm{};
        makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.40f, 0.0f, 0.0f, 0.0f, lm);
        std::array<float, FaceMeshFrame::kMetaSize> meta{};
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, meta.data()));

        FaceMeshDeformer def;
        FaceMeshDeformer::Params p{};
        p.faceNarrow = 1.0f;
        p.cheekboneSlim = 1.0f;
        EXPECT_TRUE(def.deform(
            lm.data(), kAspect, meta[FaceMeshFrame::kMetaAxisX], meta[FaceMeshFrame::kMetaAxisY],
            meta[FaceMeshFrame::kMetaFaceWidth], meta[FaceMeshFrame::kMetaFaceHeight],
            meta[FaceMeshFrame::kMetaLeftWeight], meta[FaceMeshFrame::kMetaRightWeight],
            meta[FaceMeshFrame::kMetaYawWeight], 1.0f, 1.0f, 1.0f, p));
        const float* wl = def.warpedLandmarks();
        const float origREye = std::hypot(lm[133 * 2] - lm[33 * 2], (lm[133 * 2 + 1] - lm[33 * 2 + 1]) / kAspect);
        const float newREye = std::hypot(wl[133 * 2] - wl[33 * 2], (wl[133 * 2 + 1] - wl[33 * 2 + 1]) / kAspect);
        const float origLEye = std::hypot(lm[263 * 2] - lm[362 * 2], (lm[263 * 2 + 1] - lm[362 * 2 + 1]) / kAspect);
        const float newLEye = std::hypot(wl[263 * 2] - wl[362 * 2], (wl[263 * 2 + 1] - wl[362 * 2 + 1]) / kAspect);
        EXPECT_TRUE(std::fabs(newREye - origREye) / origREye < 0.025f);
        EXPECT_TRUE(std::fabs(newLEye - origLEye) / origLEye < 0.025f);
    }

    // 3. C7 (shape-v2): BodyGeometrySmoother knee visibility hysteresis smoothly ramps legsWeight
    {
        BodyGeometrySmoother sm;
        const float st[7] = {0.5f, 0.5f, 0.0f, -1.0f, 0.35f, 0.25f, 1.0f};
        const float* out = nullptr;
        for (int i = 0; i < 6; ++i) {
            out = sm.update(st, 1000 + i * 33, 1000 + i * 33, 0.9f, 0.8f);
        }
        EXPECT_TRUE(out != nullptr && out[6] > 0.40f);
        // Drop knee visibility below 0.35 for >= 3 consecutive samples -> legsWeight ramps smoothly down toward 0
        for (int i = 0; i < 15; ++i) {
            out = sm.update(st, 1200 + i * 33, 1200 + i * 33, 0.9f, 0.2f);
        }
        EXPECT_TRUE(out != nullptr && out[6] < 0.05f);
    }

    // 4. B4 (tracking-4): MultiFaceTracker duplicate-face NMS + slot 1 -> slot 0 promotion
    {
        MultiFaceTracker tracker;
        tracker.setAspect(kAspect);
        std::array<float, FaceMeshFrame::kCoordCount> glA{}, glB{}, mpA{}, mpB{};
        makeCanonicalFace(kAspect, 0.48f, 0.50f, 0.34f, 0.0f, 0.0f, 0.0f, glA);
        makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.32f, 0.0f, 0.0f, 0.0f, glB);  // Overlapping duplicate (< 0.45 * fw)
        for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
            mpA[i * 2] = glA[i * 2]; mpA[i * 2 + 1] = 1.0f - glA[i * 2 + 1];
            mpB[i * 2] = glB[i * 2]; mpB[i * 2 + 1] = 1.0f - glB[i * 2 + 1];
        }
        MultiFaceTracker::RawFaceInput dupInputs[2] = {{mpA.data(), 478}, {mpB.data(), 478}};
        const int accepted = tracker.submitFaces(1000, dupInputs, 2, 720, 1280, 1000);
        EXPECT_TRUE(accepted == 1);
        EXPECT_TRUE(tracker.slot(1).latest() == kTimeInvalid);

        // Two distinct faces, then primary (left) disappears and expires -> right face promotes to slot 0
        tracker.reset();
        makeCanonicalFace(kAspect, 0.25f, 0.50f, 0.28f, 0.0f, 0.0f, 0.0f, glA);
        makeCanonicalFace(kAspect, 0.75f, 0.50f, 0.24f, 0.0f, 0.0f, 0.0f, glB);
        for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
            mpA[i * 2] = glA[i * 2]; mpA[i * 2 + 1] = 1.0f - glA[i * 2 + 1];
            mpB[i * 2] = glB[i * 2]; mpB[i * 2 + 1] = 1.0f - glB[i * 2 + 1];
        }
        MultiFaceTracker::RawFaceInput twoFaces[2] = {{mpA.data(), 478}, {mpB.data(), 478}};
        EXPECT_TRUE(tracker.submitFaces(2000, twoFaces, 2, 720, 1280, 2000) == 2);
        // Feed only face B for > 550ms so slot 0 clears -> slot 1 auto-promotes to slot 0
        MultiFaceTracker::RawFaceInput onlyB[1] = {{mpB.data(), 478}};
        for (int f = 1; f <= 20; ++f) {
            const int64_t ts = 2000 + f * 33;
            tracker.submitFaces(ts, onlyB, 1, 720, 1280, ts);
            tracker.update(ts);
        }
        EXPECT_TRUE(tracker.slot(0).latest() != kTimeInvalid);
        EXPECT_TRUE(tracker.slot(1).latest() == kTimeInvalid);
    }

    // 5. B5 (tracking-1): 2nd-order acceleration rigid noise floor catches pure coherent rigid jitter
    {
        FaceMeshFilter flt;
        std::array<float, FaceMeshFrame::kCoordCount> lm{}, out{};
        float pose[8]{};
        for (int f = 0; f < 30; ++f) {
            const float j = (f % 2 == 0) ? 0.004f : -0.004f;  // Pure coherent rigid jitter (0 residual!)
            makeCanonicalFace(kAspect, 0.50f + j, 0.50f, 0.36f, 0.0f, 0.0f, 0.0f, lm);
            flt.filter(1000 + f * 33, lm.data(), kAspect, out.data(), pose);
        }
        EXPECT_TRUE(flt.poseDead() > 0.03f);
    }

    // 6. E4 (cpuperf-14): Modern Mali-G1 & PowerVR D-Series/DXT classification
    EXPECT_TRUE(DeviceTierPolicy::classify("Mali-G1-Ultra MC12", 8, 12288) == PerformanceTier::kUltra);
    EXPECT_TRUE(DeviceTierPolicy::classify("Mali-G1 MC6", 8, 8192) == PerformanceTier::kStandard);
    EXPECT_TRUE(DeviceTierPolicy::classify("PowerVR DXT-48-1536", 8, 12288) == PerformanceTier::kUltra);
}

void testV395ReverificationFixesMath() {
    constexpr float kAspect = 720.0f / 1280.0f;

    // 1. shape-n1: Eye-band lateral exclusion restores chin/jaw slimming (>5% narrowing, jaw kink <4 deg)
    //    while keeping narrow+cheekbone eye/iris horizontal deformation <1%.
    {
        std::array<float, FaceMeshFrame::kCoordCount> lm{};
        makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.36f, 0.0f, 0.0f, 0.0f, lm);
        float meta[FaceMeshFrame::kMetaSize]{};
        EXPECT_TRUE(FaceMeshFrame::fillMeta(lm.data(), FaceMeshFrame::kCoordCount, kAspect, meta));

        FaceMeshDeformer def;
        FaceMeshDeformer::Params pSlim{};
        pSlim.faceSlim = 1.0f;
        pSlim.vFace = 1.0f;
        EXPECT_TRUE(def.deform(lm.data(), kAspect, meta[2], meta[3], meta[4], meta[5],
                               meta[7], meta[8], meta[9], 1.0f, 1.0f, 1.0f, pSlim));
        const float* wl = def.warpedLandmarks();
        const float origJaw = std::hypot(lm[378 * 2] - lm[149 * 2], (lm[378 * 2 + 1] - lm[149 * 2 + 1]) / kAspect);
        const float warpJaw = std::hypot(wl[378 * 2] - wl[149 * 2], (wl[378 * 2 + 1] - wl[149 * 2 + 1]) / kAspect);
        EXPECT_TRUE(warpJaw < origJaw * 0.95f);  // >5% chin/jaw narrowing restored

        // Eye horizontal protection under faceNarrow=0.5 + cheekboneSlim=0.5
        FaceMeshDeformer defEye;
        FaceMeshDeformer::Params pEye{};
        pEye.faceNarrow = 0.5f;
        pEye.cheekboneSlim = 0.5f;
        EXPECT_TRUE(defEye.deform(lm.data(), kAspect, meta[2], meta[3], meta[4], meta[5],
                                  meta[7], meta[8], meta[9], 1.0f, 1.0f, 1.0f, pEye));
        const float* we = defEye.warpedLandmarks();
        const float origEyeW = std::hypot(lm[133 * 2] - lm[33 * 2], (lm[133 * 2 + 1] - lm[33 * 2 + 1]) / kAspect);
        const float warpEyeW = std::hypot(we[133 * 2] - we[33 * 2], (we[133 * 2 + 1] - we[33 * 2 + 1]) / kAspect);
        EXPECT_TRUE(std::fabs(warpEyeW - origEyeW) / origEyeW < 0.01f);

        // Yaw 15 deg + 0.3px detector noise over 120 frames: maxJump < 0.6px and 0 visible front-facing flips
        FaceMeshDeformer defYaw;
        FaceMeshDeformer::Params pYaw{};
        pYaw.faceSlim = 0.8f;
        pYaw.vFace = 0.6f;
        pYaw.cheekboneSlim = 0.5f;
        std::array<float, FaceMeshFrame::kCoordCount> prevWl{};
        float maxJumpPx = 0.0f;
        uint32_t rng = 0x3950A1u;
        auto nextGauss = [&]() -> float {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                rng = rng * 1664525u + 1013904223u;
                s += (static_cast<float>((rng >> 8) & 0xFFFFu) / 65535.0f) - 0.5f;
            }
            return s * 1.732f;
        };
        for (int f = 0; f < 120; ++f) {
            std::array<float, FaceMeshFrame::kCoordCount> noisy{};
            makeCanonicalFace(kAspect, 0.50f, 0.50f, 0.36f, 0.0f, 0.26f, 0.0f, noisy);  // ~15 deg yaw
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
                noisy[i * 2] += (nextGauss() * 0.3f) / 720.0f;
                noisy[i * 2 + 1] += (nextGauss() * 0.3f) / 1280.0f;
            }
            float m[FaceMeshFrame::kMetaSize]{};
            EXPECT_TRUE(FaceMeshFrame::fillMeta(noisy.data(), FaceMeshFrame::kCoordCount, kAspect, m));
            EXPECT_TRUE(defYaw.deform(noisy.data(), kAspect, m[2], m[3], m[4], m[5],
                                      m[7], m[8], m[9], 1.0f, 1.0f, 1.0f, pYaw, 1000 + f * 33));
            const float* curWl = defYaw.warpedLandmarks();
            if (f >= 15) {
                for (int i : {149, 172, 176, 378, 397, 400}) {
                    const float dx = (curWl[i * 2] - prevWl[i * 2]) * 720.0f;
                    const float dy = (curWl[i * 2 + 1] - prevWl[i * 2 + 1]) * 1280.0f;
                    const float ndx = (noisy[i * 2] - prevWl[i * 2]) * 720.0f;
                    (void)ndx;
                    const float dispCurX = (curWl[i * 2] - noisy[i * 2]) * 720.0f;
                    const float dispCurY = (curWl[i * 2 + 1] - noisy[i * 2 + 1]) * 1280.0f;
                    const float dispPrevX = prevWl[i * 2];
                    const float dispPrevY = prevWl[i * 2 + 1];
                    maxJumpPx = std::max(maxJumpPx, std::hypot(dispCurX - dispPrevX, dispCurY - dispPrevY));
                    (void)dx; (void)dy;
                }
            }
            for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
                prevWl[i * 2] = (curWl[i * 2] - noisy[i * 2]) * 720.0f;
                prevWl[i * 2 + 1] = (curWl[i * 2 + 1] - noisy[i * 2 + 1]) * 1280.0f;
            }
        }
        EXPECT_TRUE(maxJumpPx < 0.6f);
    }

    // 2. makeup-1 + segmentation-n4 + seg-8: uncertain hand (p2=0.60, p3=0.30) occludes makeup cleanly,
    //    normal face skin has 0 false positive, planar 256->128 uses 2x2 box averaging.
    {
        FaceParsingMapper mapper;
        constexpr int kS = 256;
        std::vector<float> logits(kS * kS * 6, -4.0f);
        for (int y = 0; y < kS; ++y) {
            for (int x = 0; x < kS; ++x) {
                float* p = &logits[(y * kS + x) * 6];
                // Default face skin (p3 ~ 0.96)
                p[3] = 3.0f;
                // Uncertain hand patch in lower face center: p2=0.60, p3=0.30, p0=0.10
                if (x >= 112 && x <= 144 && y >= 136 && y <= 168) {
                    p[0] = std::log(0.10f);
                    p[1] = -8.0f;
                    p[2] = std::log(0.60f);
                    p[3] = std::log(0.30f);
                    p[4] = -8.0f;
                    p[5] = -8.0f;
                }
            }
        }
        CropRect face{0.25f, 0.25f, 0.50f, 0.50f};
        std::array<uint8_t, FaceParsingMapper::kOutPixels> occ{}, hair{}, person{};
        EXPECT_TRUE(mapper.mapMulticlass6LogitsNhwcRegion(
            logits.data(), kS, kS, 0, 0, kS, kS, occ.data(), hair.data(), person.data(), &face, nullptr));
        const int handIdx = (76 * FaceParsingMapper::kOutSize) + 64;
        const int cheekIdx = (56 * FaceParsingMapper::kOutSize) + 52;
        EXPECT_TRUE(occ[handIdx] < 128);   // Uncertain finger occludes makeup
        EXPECT_TRUE(occ[cheekIdx] > 245);  // Clean cheek stays unoccluded

        // Planar 2x2 box averaging (seg-8): checkerboard alternating class 1 (hair) and class 3 (face) averages cleanly
        mapper.reset();
        std::vector<float> pl[6];
        for (int c = 0; c < 6; ++c) pl[c].assign(kS * kS, 0.0f);
        for (int y = 0; y < kS; ++y) {
            for (int x = 0; x < kS; ++x) {
                const bool alt = ((x + y) & 1) == 0;
                pl[alt ? 1 : 0][y * kS + x] = 1.0f;
            }
        }
        const float* ptrs[6] = {pl[0].data(), pl[1].data(), pl[2].data(), pl[3].data(), pl[4].data(), pl[5].data()};
        EXPECT_TRUE(mapper.mapMulticlass6PlanarProbabilities(ptrs, kS, kS, occ.data(), hair.data(), person.data()));
        const uint8_t centerHair = hair[64 * FaceParsingMapper::kOutSize + 64];
        EXPECT_TRUE(centerHair > 40 && centerHair < 215);  // 2x2 box averages 0.5 (not aliased 0 or 255)
    }

    // 3. tracking-4 + tracking-n3: 0.8*fw association gate & promotionCount increment
    {
        MultiFaceTracker tracker;
        std::array<float, FaceMeshFrame::kCoordCount> lmA{}, lmB{}, mpA{}, mpB{};
        makeCanonicalFace(kAspect, 0.25f, 0.50f, 0.24f, 0.0f, 0.0f, 0.0f, lmA);
        makeCanonicalFace(kAspect, 0.78f, 0.50f, 0.24f, 0.0f, 0.0f, 0.0f, lmB);
        for (int i = 0; i < 478; ++i) {
            mpA[i * 2] = lmA[i * 2]; mpA[i * 2 + 1] = 1.0f - lmA[i * 2 + 1];
            mpB[i * 2] = lmB[i * 2]; mpB[i * 2 + 1] = 1.0f - lmB[i * 2 + 1];
        }
        MultiFaceTracker::RawFaceInput onlyA[1] = {{mpA.data(), 478}};
        MultiFaceTracker::RawFaceInput onlyB[1] = {{mpB.data(), 478}};
        EXPECT_TRUE(tracker.submitFaces(1000, onlyA, 1, 720, 1280, 1000) == 1);
        tracker.update(1000);
        // Single far-away face B (>0.8*fw) must route to slot 1 instead of hijacking slot 0 immediately
        EXPECT_TRUE(tracker.submitFaces(1033, onlyB, 1, 720, 1280, 1033) == 1);
        EXPECT_TRUE(tracker.slot(1).latest() == 1033);
        const uint64_t promoBefore = tracker.promotionCount();
        for (int f = 2; f <= 22; ++f) {
            const int64_t ts = 1000 + f * 33;
            tracker.submitFaces(ts, onlyB, 1, 720, 1280, ts);
            tracker.update(ts);
        }
        EXPECT_TRUE(tracker.promotionCount() > promoBefore);
        EXPECT_TRUE(tracker.slot(0).latest() != kTimeInvalid);
    }

    // 4. skin-7 / cpuperf-v1: loadLevel == 1 keeps enableBlemishHeal & enableMatteRefine governed by tier
    {
        const auto capL1 = DeviceTierPolicy::resolve(PerformanceTier::kAuto, PerformanceTier::kStandard, 1);
        EXPECT_TRUE(capL1.enableBlemishHeal);
        EXPECT_TRUE(capL1.enableMatteRefine);
    }
}

}  // namespace

int main() {
    testTopologyAndSubMeshes();
    testDeformerInvariantsAndCranialTop();
    testFaceTimelineAndEyeMakeupKeep();
    testRoiSelectorAndTuning();
    testBodyAndLoadGovernor();
    testHairMaskTuningAndMotionComp();
    testStickerTransformAnchorsAndRotation();
    testMultiFaceTrackerStableAssociationAndCrossing();
    testFaceRelight3dNormalsAndStyles();
    testDeviceTierPolicyClassificationAndDownshift();
    testAr3dSolverAndSpringPendulum();
    testFaceAuxVisionNeonAndExpressionBus();
    testFaceParsing19ClassAndNeuralResidualAdapter();
    testV35Curved3dMeshesAndEffectBundleParser();
    testCodeReviewRegressionGuards();
    testV36MulticlassParsingParticlesAndBundleGestureTriggers();
    testV366FastReacquireAndZeroLagLipNoseEye();
    testV367BeardAndNasalShadowNotMisclassifiedAsForeignOccluder();
    testV368FacialContourStabilityNoJumping();
    testV369SlightCameraShakeZeroWaterRippleAndZeroMeshShear();
    testV3610CheekBeautyStabilityMirroredNoTwitch();
    testV3612PhoneShakeZeroFeatureLag();
    testV3613HairCoverageGhostFreeEmaAndAffineMargin();
    testV3615DouyinAmbientSceneLiftCurve();
    testV370TwoStageSmoothIndoorDenoiseMouthGateAndProfileGuard();
    testV380Cat02WhiteBalanceStyleAtmosphereAndLowerFaceOcclusion();
    testV390AutoWhiteBalanceAndMulticlass6PlanarProbabilities();
    testV394AuditFixesMathAndTracking();
    testV395ReverificationFixesMath();

    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty C++17 math core tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
