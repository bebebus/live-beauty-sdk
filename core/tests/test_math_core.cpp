#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_makeup_uv.h"
#include "livebeauty/math/face_mesh_deformer.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
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

    if (gFailures == 0) {
        std::cout << "PASS  all livebeauty C++17 math core tests passed!\n";
        return 0;
    }
    std::cerr << "FAIL  " << gFailures << " assertion(s) failed.\n";
    return 1;
}
