#include <array>
#include <chrono>
#include <iostream>

#include "livebeauty/config.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_mesh_deformer.h"
#include "livebeauty/math/face_mesh_filter.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/math/face_timeline.h"
#include "livebeauty/math/sticker_transform.h"

using namespace livebeauty;
using namespace livebeauty::math;

int main() {
    constexpr int kWarmup = 200;
    constexpr int kIters = 2000;
    const float aspect = 9.0f / 16.0f;

    std::array<float, FaceMeshFrame::kCoordCount> mpXy{};
    const auto& xyz = FaceMeshTopology::kCanonicalXyz;
    const float scale = 0.42f / FaceMeshFilter::canonicalFaceWidth();
    for (int i = 0; i < FaceMeshTopology::kFaceVertexCount; ++i) {
        const float mx = 0.5f + xyz[i * 3] * scale;
        const float my = (0.5f / aspect) + xyz[i * 3 + 1] * scale;
        mpXy[i * 2] = mx;
        mpXy[i * 2 + 1] = 1.0f - (my * aspect);
    }
    for (int i = 468; i < 478; ++i) {
        mpXy[i * 2] = 0.5f;
        mpXy[i * 2 + 1] = 0.45f;
    }

    FaceMeshFrame frame{};
    FaceMeshFrame::fromMediaPipe(mpXy.data(), 478, aspect, frame);
    float state41[FaceGeometryConstants::kStateSize];
    FaceMeshFrame::faceGeometryFromMediaPipe(mpXy.data(), 478, aspect, state41);

    FaceTimeline timeline;
    timeline.setAspect(aspect);
    FaceMeshDeformer deformer;
    auto cfg = BeautyConfig::koreanWaterGlow();
    cfg.cranialTop = 0.65f;
    cfg.vFace = 0.55f;
    auto params = FaceMeshDeformer::Params::fromConfig(cfg);
    StickerLayerConfig sticker{};
    sticker.enabled = true;
    sticker.texture = 1;
    sticker.anchor = StickerAnchor::kForeheadCrown;
    float quad16[StickerTransform::kQuadFloats];

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        const int64_t ts = 1000 + i * 33;
        timeline.add(ts, state41, &frame);
        timeline.update(ts);
        FaceTimeline::SmoothedMeshView mv{};
        timeline.meshAt(ts, mv);
        deformer.deform(mv.landmarks, aspect, mv.axisX(), mv.axisY(), mv.faceWidth(), mv.faceHeight(),
                        mv.leftWeight(), mv.rightWeight(), mv.yawWeight(), 1.0f, 1.0f, 1.0f, params, ts);
        StickerTransform::buildQuadVertices(sticker, deformer.warpedLandmarks(), mv.meta, aspect, quad16);
    }

    const auto t0 = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int i = 0; i < kIters; ++i) {
        const int64_t ts = 100000 + i * 33;
        timeline.add(ts, state41, &frame);
        timeline.update(ts);
        FaceTimeline::SmoothedMeshView mv{};
        timeline.meshAt(ts, mv);
        deformer.deform(mv.landmarks, aspect, mv.axisX(), mv.axisY(), mv.faceWidth(), mv.faceHeight(),
                        mv.leftWeight(), mv.rightWeight(), mv.yawWeight(), 1.0f, 1.0f, 1.0f, params, ts);
        StickerTransform::buildQuadVertices(sticker, deformer.warpedLandmarks(), mv.meta, aspect, quad16);
        sink += deformer.vertexData()[0] + quad16[0];
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double usPerFrame =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / (1000.0 * kIters);
    std::cout << "BENCH livebeauty C++17 full CPU pipeline (Timeline + 478pt Filter + 548v Deformer + Sticker): "
              << usPerFrame << " us/frame (" << (1e6 / usPerFrame) << " fps theoretical CPU ceiling, sink="
              << sink << ")\n";
    return 0;
}
