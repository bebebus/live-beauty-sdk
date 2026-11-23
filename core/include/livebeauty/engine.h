#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "livebeauty/config.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_roi_selector.h"
#include "livebeauty/math/face_timeline.h"
#include "livebeauty/math/hair_mask_tuning.h"
#include "livebeauty/math/multi_face_tracker.h"
#include "livebeauty/math/sticker_transform.h"
#include "livebeauty/rhi/render_pipeline.h"

namespace livebeauty {

/** Real-time runtime telemetry snapshot exposed for host HUD / QA profiling. */
struct PerfStats {
    float lastFrameMs = 0.0f;
    float emaFrameMs = 0.0f;
    float fps = 0.0f;
    int loadLevel = 0;
    float facePresence = 0.0f;
    float hairShown = 0.0f;
    uint32_t activePassesMask = 0;
    int64_t frameCount = 0;
    int activeFaces = 0;
    int64_t displayedTimestampMs = kTimeInvalid;
};

/**
 * Unified cross-platform C++17 beauty engine facade.
 * Combines multi-face timeline smoothing (up to 2 simultaneous faces), ROI selection,
 * hair motion compensation + hair-weighted guided filter, FrameDelayRing zero-lag alignment,
 * 2D AR stickers, virtual background replacement, load governor, telemetry and RHI RenderPipeline.
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

    /** Configure or clear one of up to 4 face-anchored 2D AR sticker layers. */
    void setSticker(int slot, const math::StickerLayerConfig& layer);
    void clearStickers();

    /** Feed asynchronous MediaPipe 468/478-pt single primary face landmarks (y-down normalized). */
    bool submitFaceLandmarks(int64_t sampleTsMs, const float* mpXy, int pointCount,
                             int frameWidth, int frameHeight, int64_t receivedMs);
    /** Feed 0..2 simultaneous MediaPipe 468/478-pt faces (minimum-cost ID-preserving association). */
    int submitMultiFaceLandmarks(int64_t sampleTsMs, const math::MultiFaceTracker::RawFaceInput* faces,
                                 int faceCount, int frameWidth, int frameHeight, int64_t receivedMs);
    void submitFaceMiss(int64_t sampleTsMs);

    /** Feed asynchronous MediaPipe 33-pt pose landmarks. */
    bool submitBodyPose(int64_t sampleTsMs, const math::PoseLandmark* pose33, int count,
                        int frameWidth, int frameHeight);
    void submitBodyMiss(int64_t sampleTsMs);

    /** Compute adaptive far-field face ROI crop rect for next vision sampling pass. */
    CropRect selectFaceRoi(int frameWidth, int frameHeight, int64_t nowMs);

    /**
     * Process one frame and return output texture handle (or inputTexture when bypassed).
     * Optional hairMaskTex + hairMaskStampMs enable hair semantic protection & flyaway bokeh/virtual-bg merge.
     */
    rhi::TextureId processFrame(
        rhi::TextureId inputTexture,
        int width, int height,
        int64_t frameTimestampMs,
        int64_t nowMs,
        rhi::TextureId personMaskTex = rhi::kInvalidTexture,
        float lightLuma = 0.30f,
        float bitrateRatio = 1.0f,
        rhi::TextureId hairMaskTex = rhi::kInvalidTexture,
        int64_t hairMaskStampMs = kTimeInvalid);

    int loadLevel() const { return governor_.level(); }
    float facePresence() const { return faces_.slot(0).presence(); }
    int activeFaceCount() const { return faces_.activeFaceCount(); }
    const PerfStats& stats() const { return stats_; }

private:
    BeautyConfig config_;
    rhi::RenderPipeline pipeline_;
    math::MultiFaceTracker faces_;
    math::FaceRoiSelector roiSelector_;
    math::BodyGeometrySmoother bodySmoother_;
    math::EffectsLoadGovernor governor_;
    math::LoadFades loadFades_;
    math::GuidedSpread guidedSpread_;
    math::HairMaskGate hairGate_;

    std::array<math::StickerLayerConfig, math::kMaxStickerLayers> stickers_{};
    int stickerCount_ = 0;

    int lastWidth_ = 0;
    int lastHeight_ = 0;
    bool hasPendingBody_ = false;
    std::array<float, math::BodyGeometry::kStateSize> pendingBodyState_{};
    int64_t pendingBodyTs_ = kTimeInvalid;

    PerfStats stats_{};
    int64_t lastStatsNowMs_ = kTimeInvalid;
};

}  // namespace livebeauty
