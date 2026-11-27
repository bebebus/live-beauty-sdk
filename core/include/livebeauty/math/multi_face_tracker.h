#pragma once

#include <array>
#include <cstdint>

#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_timeline.h"

namespace livebeauty::math {

/**
 * Multi-face slot tracker (up to kMaxFaces = 2 simultaneous faces) with optimal minimum-cost
 * 2D center association so two people moving or crossing in frame never swap timeline IDs.
 */
class MultiFaceTracker {
public:
    static constexpr int kMaxFaces = 2;

    struct RawFaceInput {
        const float* mpXy = nullptr;
        int pointCount = 0;
    };

    void setAspect(float aspect);
    void reset();

    /** Submit 0..kMaxFaces detected faces for the same camera sample timestamp. */
    int submitFaces(int64_t sampleTsMs, const RawFaceInput* faces, int faceCount,
                    int frameWidth, int frameHeight, int64_t receivedMs,
                    float* outPrimaryCenterX = nullptr, float* outPrimaryCenterY = nullptr,
                    float* outPrimaryFaceWidth = nullptr);

    void submitAllMiss(int64_t sampleTsMs);
    void update(int64_t nowMs);
    /** If primary slot 0 is empty while secondary slot 1 is active, promote slot 1 -> slot 0 cleanly. */
    bool promoteSecondaryIfPrimaryEmpty();
    uint64_t promotionCount() const { return promotionCount_; }

    FaceTimeline& slot(int idx) { return timelines_[idx]; }
    const FaceTimeline& slot(int idx) const { return timelines_[idx]; }

    int activeFaceCount() const;

private:
    struct ParsedFace {
        FaceMeshFrame mesh{};
        std::array<float, FaceGeometryConstants::kStateSize> state41{};
        float cx = 0.5f;
        float cy = 0.5f;
        float fw = 0.0f;
        bool valid = false;
    };

    bool parseOne(const RawFaceInput& raw, float aspect, ParsedFace& out) const;
    static float distSq(float x0, float y0, float x1, float y1, float aspect);

    std::array<FaceTimeline, kMaxFaces> timelines_{};
    std::array<float, kMaxFaces> lastCx_{0.5f, 0.5f};
    std::array<float, kMaxFaces> lastCy_{0.5f, 0.5f};
    std::array<float, kMaxFaces> lastFw_{0.0f, 0.0f};
    std::array<bool, kMaxFaces> hasLastCenter_{false, false};
    uint64_t promotionCount_ = 0;
    float aspect_ = 1.0f;
};

}  // namespace livebeauty::math
