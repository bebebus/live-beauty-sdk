#pragma once

#include <cstdint>
#include <memory>

#include "livebeauty/config.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_roi_selector.h"
#include "livebeauty/math/face_timeline.h"
#include "livebeauty/rhi/render_pipeline.h"

namespace livebeauty {

/**
 * Unified cross-platform C++17 beauty engine facade.
 * Combines timeline smoothing, ROI selection, load governor, body/face geometry, and RHI RenderPipeline.
 */
class LiveBeautyEngine {
public:
    explicit LiveBeautyEngine(rhi::IRhiDevice& device);
    ~LiveBeautyEngine() = default;

    void setConfig(const BeautyConfig& config);
    const BeautyConfig& config() const { return config_; }

    void setAssets(const rhi::PipelineAssets& assets) { pipeline_.setAssets(assets); }
    void resetTracking();
    void releaseGpuResources() { pipeline_.releaseGpuResources(); }

    /** Feed asynchronous MediaPipe 468/478-pt face landmarks (y-down normalized). */
    bool submitFaceLandmarks(int64_t sampleTsMs, const float* mpXy, int pointCount,
                             int frameWidth, int frameHeight, int64_t receivedMs);
    void submitFaceMiss(int64_t sampleTsMs);

    /** Feed asynchronous MediaPipe 33-pt pose landmarks. */
    bool submitBodyPose(int64_t sampleTsMs, const math::PoseLandmark* pose33, int count,
                        int frameWidth, int frameHeight);
    void submitBodyMiss(int64_t sampleTsMs);

    /** Compute adaptive far-field face ROI crop rect for next vision sampling pass. */
    CropRect selectFaceRoi(int frameWidth, int frameHeight, int64_t nowMs);

    /** Process one frame and return output texture handle (or inputTexture when bypassed). */
    rhi::TextureId processFrame(
        rhi::TextureId inputTexture,
        int width, int height,
        int64_t frameTimestampMs,
        int64_t nowMs,
        rhi::TextureId personMaskTex = rhi::kInvalidTexture,
        float lightLuma = 0.30f,
        float bitrateRatio = 1.0f);

    int loadLevel() const { return governor_.level(); }
    float facePresence() const { return timeline_.presence(); }

private:
    BeautyConfig config_;
    rhi::RenderPipeline pipeline_;
    math::FaceTimeline timeline_;
    math::FaceRoiSelector roiSelector_;
    math::BodyGeometrySmoother bodySmoother_;
    math::EffectsLoadGovernor governor_;
    math::LoadFades loadFades_;
    math::GuidedSpread guidedSpread_;

    int lastWidth_ = 0;
    int lastHeight_ = 0;
    bool hasPendingBody_ = false;
    std::array<float, math::BodyGeometry::kStateSize> pendingBodyState_{};
    int64_t pendingBodyTs_ = kTimeInvalid;
};

}  // namespace livebeauty
