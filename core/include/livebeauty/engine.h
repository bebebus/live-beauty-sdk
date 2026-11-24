#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "livebeauty/bundle/effect_bundle.h"
#include "livebeauty/config.h"
#include "livebeauty/math/ar_3d_accessory.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/device_tier.h"
#include "livebeauty/math/face_aux_vision.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_parsing.h"
#include "livebeauty/math/face_roi_selector.h"
#include "livebeauty/math/face_timeline.h"
#include "livebeauty/math/hair_mask_tuning.h"
#include "livebeauty/math/multi_face_tracker.h"
#include "livebeauty/math/particle_system.h"
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
    int resolvedTier = static_cast<int>(math::PerformanceTier::kStandard);
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
    ~LiveBeautyEngine();

    void setConfig(const BeautyConfig& config);
    const BeautyConfig& config() const { return config_; }

    void setAssets(const rhi::PipelineAssets& assets) { pipeline_.setAssets(assets); }
    void resetTracking();
    void releaseGpuResources();

    /** Configure or clear one of up to 4 face-anchored 2D AR sticker layers. */
    void setSticker(int slot, const math::StickerLayerConfig& layer,
                    bundle::BundleTriggerRule trigger = bundle::BundleTriggerRule::kAlways);
    void clearStickers();

    /** P4: Configure or clear one of up to 4 head-occluded 3D AR accessories with spring physics. */
    void setAr3dAccessory(int slot, const math::Ar3dAccessoryConfig& layer,
                          bundle::BundleTriggerRule trigger = bundle::BundleTriggerRule::kAlways);
    void clearAr3dAccessories();

    /**
     * v3.5: Load and apply a declarative `.lbfx` Effect Bundle manifest (or raw JSON text),
     * automatically configuring beauty look overrides, 2D stickers, 3D meshes, hair dye, and
     * expression-reactive trigger rules (`smile`, `mouth_open`, `brow_up`, `blink`, etc.).
     */
    bool applyEffectBundle(
        const bundle::EffectBundleManifest& manifest,
        const bundle::BundleTextureResolver& textureResolver = nullptr);
    bool applyEffectBundleJson(
        const std::string& manifestJson,
        const bundle::BundleTextureResolver& textureResolver = nullptr,
        std::string* outError = nullptr);

    /** P4: Declare device hardware profile once so DeviceTierPolicy can auto-schedule Lite/Standard/Ultra. */
    void setHardwareProfile(const std::string& gpuRenderer, int cpuCores, int totalRamMb);

    /** P1: Update smoothed ambient illuminant white-balance RGB (normalized ~[0.6..1.4]) for natural makeup tint. */
    void setAmbientRgb(float r, float g, float b);

    /** P2: Attach optional external Face-Parsing / hand occlusion mask texture (R8, 255=visible, 0=occluded). */
    void setOcclusionMask(rhi::TextureId occlusionMaskTex);

    /** P3: Attach optional external Face-ROI AI retouching residual texture (RGBA8, 0.5=zero delta). */
    void setAiResidual(rhi::TextureId residualTex, const CropRect& roiRect, float strength);

    /** v3.2: Submit 52-channel distilled Face Blendshapes for reactive 2D/3D AR triggers (thread-safe). */
    void submitFaceBlendshapes(const math::BlendshapeInput& blendshapes);
    void setExpressionReactive(bool enabled) { expressionReactive_ = enabled; }
    math::ExpressionReaction expressionReaction() const;

    /**
     * v3.2: Submit a top-down RGBA vision sample on any worker thread to run C++17 + ARM NEON
     * foreground occlusion detection (`64x64` R8) and high-frequency-preserving Face-ROI skin residual
     * (`64x64` RGBA8). Automatically uploaded and bound on the GPU thread during `processFrame()`
     * whenever external textures are not explicitly overriding them.
     */
    bool submitAuxRgbaSample(
        const uint8_t* rgbaTopDown,
        int srcWidth,
        int srcHeight,
        float retouchStrength);

    /**
     * v3.3: Submit 19-class CelebAMask-HQ / BiSeNetV2 / SegFormer semantic face-parsing class IDs (`0..19`)
     * or NCHW float logits on any worker thread. Automatically generates `128x128` occlusion, hair, and
     * person masks and uploads/binds them on the GPU thread in `processFrame()`.
     */
    bool submitFaceParsingClassMap(const uint8_t* classIds, int srcWidth, int srcHeight, int64_t sampleTsMs);
    bool submitFaceParsingLogits(const float* logitsNchw, int numClasses, int srcWidth, int srcHeight, int64_t sampleTsMs);

    /**
     * v3.6: Submit 6-class MediaPipe selfie_multiclass NHWC float logits
     * (`0=bg, 1=hair, 2=body_skin, 3=face_skin, 4=clothes, 5=accessories/hands`)
     * from any worker thread. Softmax-maps anti-aliased `128x128` occlusion, hair, and person masks.
     */
    bool submitMulticlass6Logits(const float* logitsNhwc6, int srcWidth, int srcHeight, int64_t sampleTsMs);

    /**
     * v3.6: Trigger an interactive 2D SDF particle burst (`heart`, `thumbs_up`, `victory`, `open_palm`)
     * at normalized top-down `(normX, normY)` in `[0..1]` and latch gesture-reactive `.lbfx` layers.
     */
    void triggerGesture(math::GestureType gesture, float normX, float normY, int64_t nowMs);

    /**
     * v3.6: Update active hand palm centers in normalized top-down `[0..1]` (`< 0` when inactive)
     * so `ShaderId::kBodyWarp` shields hands/fingers from waist-slimming lateral pull.
     */
    void setHandShieldCoords(float hand0X, float hand0Y, float hand1X, float hand1Y);

    /**
     * v3.6.2: Notify engine from host IMU / gyroscope monitor when rapid camera shake is occurring
     * (thread-safe). Immediately suppresses temporal frame/matte history and stale async ROI masks
     * to prevent motion trails ("ghosting") during device shaking.
     */
    void setCameraShaking(bool shaking) { cameraShaking_.store(shaking, std::memory_order_relaxed); }
    bool isCameraShaking() const { return cameraShaking_.load(std::memory_order_relaxed); }

    /**
     * v3.3: Submit external Face-ROI neural skin retouching tensor (`3xHxW` or `4xHxW` float NCHW in `[-0.24, +0.24]`)
     * from any ONNX/CoreML/TFLite worker thread.
     */
    bool submitNeuralResidualTensor(
        const float* tensorNchw,
        int channels,
        int srcWidth,
        int srcHeight,
        const CropRect& roiRect,
        float strength);

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
    rhi::IRhiDevice& device_;
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
    std::array<bundle::BundleTriggerRule, math::kMaxStickerLayers> stickerTriggers_{};
    int stickerCount_ = 0;
    std::array<math::Ar3dAccessoryConfig, math::kMaxAr3dLayers> ar3dAccessories_{};
    std::array<bundle::BundleTriggerRule, math::kMaxAr3dLayers> ar3dTriggers_{};
    int ar3dCount_ = 0;

    math::PerformanceTier hardwareTier_ = math::PerformanceTier::kStandard;
    Vec3 ambientRgb_ = {1.0f, 1.0f, 1.0f};
    rhi::TextureId occlusionMaskTex_ = rhi::kInvalidTexture;
    rhi::TextureId aiResidualTex_ = rhi::kInvalidTexture;
    CropRect aiResidualRect_ = CropRect::full();
    float aiResidualStrength_ = 0.0f;

    // v3.2 built-in C++17 + NEON occlusion, ROI residual & blendshape expression bus
    mutable std::mutex auxMutex_;
    math::FaceOcclusionSolver occlusionSolver_;
    math::FaceRoiResidualSolver residualSolver_;
    math::FaceParsingMapper faceParsingMapper_;
    math::NeuralSkinResidualAdapter neuralResidualAdapter_;
    math::FaceExpressionBus expressionBus_;
    bool expressionReactive_ = true;
    bool hasLatestFaceBounds_ = false;
    CropRect latestFaceBounds_ = CropRect::full();
    float latestFaceRollRad_ = 0.0f;
    bool semanticOccActive_ = false;
    bool stagedOccDirty_ = false;
    int stagedOccSize_ = math::FaceOcclusionSolver::kMaskSize;
    int64_t stagedOccTsMs_ = kTimeInvalid;
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> stagedOccMask_{};
    bool stagedParseHairDirty_ = false;
    int64_t stagedParseHairTsMs_ = kTimeInvalid;
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> stagedParseHair_{};
    bool stagedResidualDirty_ = false;
    bool stagedResidualActive_ = false;
    CropRect stagedResidualRect_ = CropRect::full();
    float stagedResidualStrength_ = 0.0f;
    std::array<uint8_t, math::FaceRoiResidualSolver::kRoiPixels * 4> stagedResidualRgba_{};
    rhi::TextureId autoOccTex_ = rhi::kInvalidTexture;
    int autoOccTexSize_ = 0;
    int64_t autoOccTsMs_ = kTimeInvalid;
    std::atomic<bool> cameraShaking_{false};
    rhi::TextureId autoParseHairTex_ = rhi::kInvalidTexture;
    bool autoParseHairValid_ = false;
    int64_t autoParseHairTsMs_ = kTimeInvalid;
    bool stagedParsePersonDirty_ = false;
    std::array<uint8_t, math::FaceParsingMapper::kOutPixels> stagedParsePerson_{};
    rhi::TextureId autoParsePersonTex_ = rhi::kInvalidTexture;
    bool autoParsePersonValid_ = false;
    math::ParticleBurstSolver particleSolver_;
    math::GestureType activeGesture_ = math::GestureType::kNone;
    int64_t activeGestureUntilMs_ = kTimeInvalid;
    Vec4 handShieldCoords_ = {-10.0f, -10.0f, -10.0f, -10.0f};
    std::array<float, math::ParticleBurstSolver::kMaxVertexFloats> particleVerts_{};
    uint64_t nextAr3dMeshGen_ = 0;
    rhi::TextureId autoResidualTex_ = rhi::kInvalidTexture;
    CropRect autoResidualRect_ = CropRect::full();
    float autoResidualStrength_ = 0.0f;

    int lastWidth_ = 0;
    int lastHeight_ = 0;
    int deformerRunId_ = -1;
    bool hasPendingBody_ = false;
    std::array<float, math::BodyGeometry::kStateSize> pendingBodyState_{};
    std::array<float, math::FaceGeometryConstants::kStateSize> dilatedState_{};
    int64_t pendingBodyTs_ = kTimeInvalid;

    PerfStats stats_{};
    int64_t lastStatsNowMs_ = kTimeInvalid;
};

}  // namespace livebeauty
