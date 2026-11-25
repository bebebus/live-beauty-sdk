#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
    EXPECT_TRUE(newH < origH - 0.01f);
    EXPECT_TRUE(newH > origH * 0.90f);

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

    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty C++17 math core tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
