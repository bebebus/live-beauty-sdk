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
#include "livebeauty/rhi/rhi_device.h"

namespace livebeauty::rhi {

/** Static makeup atlases & 3D LUT texture handles uploaded once by platform host. */
struct PipelineAssets {
    TextureId lutTex = kInvalidTexture;
    TextureId blushAtlas = kInvalidTexture;
    TextureId eyebrowAtlas = kInvalidTexture;
    TextureId eyeMakeupAtlas = kInvalidTexture;
    TextureId contourHighlightAtlas = kInvalidTexture;
    TextureId irisAtlas = kInvalidTexture;
    TextureId whiteAlphaFallbackTex = kInvalidTexture;  // 1x1 RGBA(0,0,0,255) for nonHair=1
};

struct FrameInputs {
    TextureId inputTexture = kInvalidTexture;
    int width = 0;
    int height = 0;
    int64_t nowMs = 0;

    // Optional tracked face mesh & 41-D state
    bool hasFaceState = false;
    const float* faceState41 = nullptr;
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

    // Optional portrait/hair segmentation mask (R8, top-down)
    TextureId personMaskTex = kInvalidTexture;

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
    void releaseGpuResources();

    /** Execute full beauty RenderGraph and return output RGBA texture handle. */
    TextureId render(const BeautyConfig& config, const FrameInputs& in);

private:
    void ensureResources(int width, int height);
    void ensureStaticBuffers();
    void renderGuidedFilter(TextureId srcTex, int width, int height, float eps, float evenness, float spread);
    void renderSkinMask(const float* landmarks478x2, bool frontFaceCw, int width, int height);
    void renderMakeup(
        RenderTargetId targetRt, TextureId baseTex, int width, int height,
        const BeautyConfig& cfg, const float* landmarks478x2, const float* meta15,
        bool hasIris, bool frontFaceCw, float presence, float steadyEye, float eyeMakeupKeep);

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

    // Polygon skin mask (1/4 res)
    RenderTargetId rtSkinMask_ = kInvalidRenderTarget;

    // Background bokeh (1/4 res)
    RenderTargetId rtBg_[2] = {kInvalidRenderTarget, kInvalidRenderTarget};
    RenderTargetId rtBgOut_ = kInvalidRenderTarget;

    // Owned 1x1 fallback white-alpha texture if host didn't provide one
    TextureId ownedFallbackTex_ = kInvalidTexture;

    // Static index buffers & dynamic vertex buffers
    bool staticBuffersReady_ = false;
    BufferId vbMeshWarp_ = kInvalidBuffer;
    BufferId vbSkinMask_ = kInvalidBuffer;
    BufferId vbRetouch_ = kInvalidBuffer;
    BufferId vbMakeup_ = kInvalidBuffer;
    BufferId ibFullMesh_ = kInvalidBuffer;
    BufferId ibSkinRetouch_ = kInvalidBuffer;
    BufferId ibEyeOpening_ = kInvalidBuffer;
    BufferId ibInnerMouth_ = kInvalidBuffer;
    BufferId ibSkinFace_ = kInvalidBuffer;
    BufferId ibLipBand_ = kInvalidBuffer;
    BufferId ibRightEye_ = kInvalidBuffer;
    BufferId ibLeftEye_ = kInvalidBuffer;

    // Reusable CPU staging buffers (zero per-frame heap allocation)
    math::FaceMeshDeformer deformer_;
    std::array<float, math::FaceMeshFrame::kCoordCount> liftedLandmarks_{};
    std::array<float, math::FaceGeometryConstants::kStateSize> liftedState_{};
    std::array<float, (math::FaceMeshTopology::kFaceVertexCount + math::FaceMeshTopology::kOvalVertexCount) * 5> skinMaskVerts_{};
    std::array<float, math::FaceRetouchSubMesh::kVertexDataSize> retouchVerts_{};
    std::array<float, math::FaceMakeupUv::kVertexDataSize> makeupVerts_{};
    std::array<float, (math::FaceMeshTopology::kFaceVertexCount + math::FaceMeshTopology::kOvalVertexCount) * 3> staticSkinWeights_{};
    bool staticSkinWeightsReady_ = false;
};

}  // namespace livebeauty::rhi
