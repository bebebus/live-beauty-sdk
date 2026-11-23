#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/math/beauty_tuning.h"
#include "livebeauty/math/body_geometry.h"
#include "livebeauty/math/face_geometry.h"
#include "livebeauty/math/face_makeup_uv.h"
#include "livebeauty/math/face_mesh_deformer.h"
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/math/hair_mask_tuning.h"
#include "livebeauty/math/sticker_transform.h"
#include "livebeauty/rhi/rhi_device.h"

namespace livebeauty::rhi {

/** Bitmask of passes executed during the last RenderPipeline::render() call (for PerfStats). */
enum PassBit : uint32_t {
    kPassBg        = 1u << 0,
    kPassBody      = 1u << 1,
    kPassFaceWarp  = 1u << 2,
    kPassRetouch   = 1u << 3,
    kPassSkinMask  = 1u << 4,
    kPassHairAlpha = 1u << 5,
    kPassGuided    = 1u << 6,
    kPassComposite = 1u << 7,
    kPassMakeup    = 1u << 8,
    kPassSticker   = 1u << 9,
    kPassDenoise   = 1u << 10,
    kPassDualFace  = 1u << 11,
    kPassFrameDelay = 1u << 12,
};

/** Static makeup atlases, virtual background & 3D LUT texture handles uploaded once by platform host. */
struct PipelineAssets {
    TextureId lutTex = kInvalidTexture;
    TextureId blushAtlas = kInvalidTexture;
    TextureId eyebrowAtlas = kInvalidTexture;
    TextureId eyeMakeupAtlas = kInvalidTexture;
    TextureId contourHighlightAtlas = kInvalidTexture;
    TextureId irisAtlas = kInvalidTexture;
    TextureId whiteAlphaFallbackTex = kInvalidTexture;  // 1x1 RGBA(0,0,0,255) for nonHair=1
    TextureId virtualBgTex = kInvalidTexture;           // Optional virtual background image (RGBA)
    float virtualBgAspect = 0.0f;                       // <=0 means match frame aspect
};

struct FrameInputs {
    TextureId inputTexture = kInvalidTexture;
    int width = 0;
    int height = 0;
    int64_t nowMs = 0;

    // Optional tracked face mesh & 41-D state
    bool hasFaceState = false;
    const float* faceState41 = nullptr;
    const float* dilatedFaceState41 = nullptr;  // Motion-dilated fallback ellipses when extrapolating
    bool hasFaceMesh = false;
    const float* landmarks478x2 = nullptr;
    const float* faceMeta15 = nullptr;
    bool hasIris = false;

    float facePresence = 0.0f;
    float steadyFace = 1.0f;
    float steadyEye = 1.0f;
    float eyeMakeupKeep = 1.0f;
    float denoiseKeep = 1.0f;

    // Optional body geometry (7-D state)
    bool hasBody = false;
    const float* bodyState7 = nullptr;
    float bodyPresence = 0.0f;

    // Optional portrait & hair segmentation masks (R8, top-down)
    TextureId personMaskTex = kInvalidTexture;
    TextureId hairMaskTex = kInvalidTexture;
    float hairShown = 0.0f;     // Semantic alpha strength [0..1] (post neck-lift attenuation)
    float hairBgShown = 0.0f;   // Background hair-merge strength [0..1] (pre-warp coordinates)
    std::array<float, math::HairMaskTuning::kAffineSize> hairAffine = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    // Optional face-anchored 2D AR sticker overlays (up to 4 layers)
    std::array<math::StickerLayerConfig, math::kMaxStickerLayers> stickers{};
    int stickerCount = 0;

    // Optional secondary tracked face (when config.maxFaces == 2)
    struct SecondaryFace {
        bool hasState = false;
        const float* state41 = nullptr;
        bool hasMesh = false;
        const float* landmarks478x2 = nullptr;
        const float* meta15 = nullptr;
        bool hasIris = false;
        float presence = 0.0f;
        float steadyFace = 1.0f;
        float steadyEye = 1.0f;
        float eyeMakeupKeep = 1.0f;
    } secondaryFace{};

    // Load fades & tuning
    float loadFadeBody = 1.0f;
    float loadFadeFace = 1.0f;
    float loadFadeSmooth = 1.0f;
    float guidedSpread = 1.25f;
    float lightLuma = 0.30f;
    float bitratePressure = 0.0f;
};

/**
 * Cross-backend unified beauty RenderGraph orchestrator.
 * Owns all FBO pools, vertex/index buffers, uniform packing and pass sequencing.
 */
class RenderPipeline {
public:
    explicit RenderPipeline(IRhiDevice& device);
    ~RenderPipeline();

    RenderPipeline(const RenderPipeline&) = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;

    void setAssets(const PipelineAssets& assets) { assets_ = assets; }
    void resetTemporal();
    void resetDeformerTemporal();
    void releaseGpuResources();

    /**
     * Cross-backend FrameDelayRing (0..4 frames): copies inputTex into a ring buffer and
     * returns the texture & timestamp from `delayFrames` frames ago so rendering aligns strictly
     * with asynchronous MediaPipe face inference (GPU 1-2 frames, CPU 2-4 frames). Zero allocation when delayFrames == 0.
     */
    void pushFrameDelay(
        TextureId inputTex, int64_t inputTsMs, int delayFrames, int width, int height,
        TextureId& outTex, int64_t& outTsMs);

    /** Execute full beauty RenderGraph and return output RGBA texture handle. */
    TextureId render(const BeautyConfig& config, const FrameInputs& in);
    uint32_t lastActivePassesMask() const { return lastActivePassesMask_; }

private:
    void ensureResources(int width, int height);
    void ensureStaticBuffers();
    void renderGuidedFilter(TextureId srcTex, int width, int height, float eps, float evenness, float spread,
                            TextureId hairWeightTex = kInvalidTexture);
    void renderSkinMask(const float* landmarks478x2, float axisX, float axisY,
                        bool frontFaceCw, int width, int height, bool clearFirst = true);
    void blurSkinMask();
    void renderHairAlpha(TextureId hairMaskTex, float hairShown, const float* hairAffine8,
                         const float* ovalState41, float aspect, bool clearRgbToZero,
                         bool meshWarped, bool frontFaceCw);
    void renderMakeup(
        RenderTargetId targetRt, TextureId baseTex, int width, int height,
        const BeautyConfig& cfg, const float* landmarks478x2, const float* meta15,
        bool hasIris, bool frontFaceCw, float presence, float steadyEye, float eyeMakeupKeep,
        bool copyBaseFirst = true);
    void renderStickers(RenderTargetId targetRt, TextureId baseTex, bool copyBaseFirst,
                        int width, int height, const FrameInputs& in,
                        const float* landmarks478x2, const float* meta15, float presence);

    IRhiDevice& device_;
    PipelineAssets assets_{};

    int width_ = 0;
    int height_ = 0;
    int guidedW_ = 0;
    int guidedH_ = 0;
    int macroW_ = 0;
    int macroH_ = 0;
    int skinW_ = 0;
    int skinH_ = 0;
    int bgW_ = 0;
    int bgH_ = 0;

    // Full-res ping-pong render targets
    RenderTargetId rtBody_ = kInvalidRenderTarget;
    RenderTargetId rtWarp_ = kInvalidRenderTarget;
    RenderTargetId rtComposite_ = kInvalidRenderTarget;
    RenderTargetId rtMakeup_ = kInvalidRenderTarget;
    RenderTargetId rtDenoise_[2] = {kInvalidRenderTarget, kInvalidRenderTarget};
    int denoiseParity_ = 0;
    bool denoiseHistoryValid_ = false;

    // Guided filter (1/3 + 1/9 dual-band, feedback-loop isolated)
    RenderTargetId rtGuided_[3] = {kInvalidRenderTarget, kInvalidRenderTarget, kInvalidRenderTarget};
    RenderTargetId rtMacro_[2] = {kInvalidRenderTarget, kInvalidRenderTarget};

    // Polygon skin mask (1/4 res + ping-pong feather blur target)
    RenderTargetId rtSkinMask_ = kInvalidRenderTarget;
    RenderTargetId rtSkinMaskTmp_ = kInvalidRenderTarget;

    // Background bokeh / virtual background (1/4 res) & merged person+hair mask
    RenderTargetId rtBg_[2] = {kInvalidRenderTarget, kInvalidRenderTarget};
    RenderTargetId rtBgHair_ = kInvalidRenderTarget;
    RenderTargetId rtBgOut_ = kInvalidRenderTarget;
    uint32_t lastActivePassesMask_ = 0;

    // Owned 1x1 fallback white-alpha texture if host didn't provide one
    TextureId ownedFallbackTex_ = kInvalidTexture;

    // Static index buffers & dynamic vertex buffers
    bool staticBuffersReady_ = false;
    BufferId vbMeshWarp_ = kInvalidBuffer;
    BufferId vbSkinMask_ = kInvalidBuffer;
    BufferId vbRetouch_ = kInvalidBuffer;
    BufferId vbMakeup_ = kInvalidBuffer;
    BufferId vbSticker_ = kInvalidBuffer;
    BufferId ibFullMesh_ = kInvalidBuffer;
    BufferId ibSkinRetouch_ = kInvalidBuffer;
    BufferId ibEyeOpening_ = kInvalidBuffer;
    BufferId ibInnerMouth_ = kInvalidBuffer;
    BufferId ibSkinFace_ = kInvalidBuffer;
    BufferId ibLipBand_ = kInvalidBuffer;
    BufferId ibRightEye_ = kInvalidBuffer;
    BufferId ibLeftEye_ = kInvalidBuffer;
    BufferId ibSticker_ = kInvalidBuffer;

    // Cross-backend FrameDelayRing (allocated lazily only when frameDelayFrames > 0; supports 0..4 frames)
    static constexpr int kMaxDelaySlots = 5;
    RenderTargetId rtDelay_[kMaxDelaySlots] = {
        kInvalidRenderTarget, kInvalidRenderTarget, kInvalidRenderTarget,
        kInvalidRenderTarget, kInvalidRenderTarget,
    };
    int64_t delayStamps_[kMaxDelaySlots] = {
        kTimeInvalid, kTimeInvalid, kTimeInvalid, kTimeInvalid, kTimeInvalid,
    };
    int delayW_ = 0;
    int delayH_ = 0;
    int delayWriteIdx_ = 0;
    int delayCount_ = 0;

    // Reusable CPU staging buffers (zero per-frame heap allocation)
    math::FaceMeshDeformer deformer_;
    math::FaceMeshDeformer deformer2_;
    std::array<float, math::FaceMeshFrame::kCoordCount> liftedLandmarks_{};
    std::array<float, math::FaceGeometryConstants::kStateSize> liftedState_{};
    std::array<float, math::FaceGeometryConstants::kStateSize> warpedState_{};
    std::array<float, (math::FaceMeshTopology::kFaceVertexCount + math::FaceMeshTopology::kOvalVertexCount) * 5> skinMaskVerts_{};
    std::array<float, math::FaceRetouchSubMesh::kVertexDataSize> retouchVerts_{};
    std::array<float, math::FaceMakeupUv::kVertexDataSize> makeupVerts_{};
    std::array<float, (math::FaceMeshTopology::kFaceVertexCount + math::FaceMeshTopology::kOvalVertexCount) * 3> staticSkinWeights_{};
    bool staticSkinWeightsReady_ = false;
};

}  // namespace livebeauty::rhi
