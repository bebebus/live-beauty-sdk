#pragma once

#include <array>
#include <unordered_map>
#include "livebeauty/rhi/rhi_device.h"

namespace livebeauty::rhi::gles {

/** OpenGL ES 2.0/3.0 RHI device implementation for Android NDK. */
class GlesDevice final : public IRhiDevice {
public:
    GlesDevice();
    ~GlesDevice() override;

    void releaseAll(bool contextAlive);

    /** Wrap an external GL_TEXTURE_2D owned by camera/RTC host into a RHI TextureId. */
    TextureId wrapExternalTexture(uint32_t glTexId);
    uint32_t unwrapGlTexture(TextureId tex) const;

    /**
     * One-step conversion of Android SurfaceTexture GL_TEXTURE_EXTERNAL_OES into an internal
     * RGBA8 GL_TEXTURE_2D owned by GlesDevice (with 4x4 SurfaceTexture transform matrix).
     */
    TextureId convertOesTo2D(uint32_t oesTexId, int width, int height, const float* stMatrix16);

    RenderTargetId createRenderTarget(int width, int height, TextureFormat format) override;
    void destroyRenderTarget(RenderTargetId rt) override;
    TextureId textureOf(RenderTargetId rt) const override;

    TextureId createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) override;
    void updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) override;
    void destroyTexture(TextureId tex) override;

    BufferId createDynamicVertexBuffer(size_t byteSize) override;
    void updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) override;
    BufferId createStaticIndexBuffer(const uint16_t* indices, size_t count) override;
    void destroyBuffer(BufferId buf) override;

    void beginPass(RenderTargetId rt, int viewportW, int viewportH, bool clear, const Vec4& clearColor) override;
    void drawFullScreenQuad(
        ShaderId shader, const RenderState& state,
        const TextureBindings& textures, const UniformBlock& uniforms) override;
    void drawIndexed(
        ShaderId shader, const RenderState& state,
        BufferId vertexBuffer, BufferId indexBuffer,
        int indexOffset, int indexCount, VertexLayout layout,
        const TextureBindings& textures, const UniformBlock& uniforms) override;
    void endPass() override;

private:
    struct ProgramCache {
        uint32_t program = 0;
        int attr0 = -1;
        int attr1 = -1;
        int attr2 = -1;
        std::array<int, 4> texLoc = {-1, -1, -1, -1};
        std::array<int, 16> uniLoc{};
    };
    struct RtRecord {
        uint32_t fbo = 0;
        TextureId texId = kInvalidTexture;
        int width = 0;
        int height = 0;
    };
    struct TexRecord {
        uint32_t glTex = 0;
        bool external = false;
    };
    struct BufRecord {
        uint32_t glBuf = 0;
        bool isIndex = false;
    };

    const ProgramCache& ensureProgram(ShaderId id);
    void ensureQuadVbo();
    void applyStateAndUniforms(const ProgramCache& pc, const RenderState& state,
                               const TextureBindings& textures, const UniformBlock& uniforms);
    void restoreSharedState();

    uint32_t nextId_ = 1;
    uint32_t quadVbo_ = 0;
    uint32_t oesProgram_ = 0;
    int oesAttrPos_ = -1;
    int oesAttrUv_ = -1;
    int oesUniTex_ = -1;
    int oesUniMat_ = -1;
    RenderTargetId oesRt_ = kInvalidRenderTarget;
    int oesW_ = 0;
    int oesH_ = 0;
    // 0 = unprobed, 1 = GLES3 RGBA16F + HALF_FLOAT, 2 = GLES2 RGBA + HALF_FLOAT_OES + linear, 3 = RGBA8 fallback
    int halfFloatMode_ = 0;
    std::array<ProgramCache, static_cast<size_t>(ShaderId::kCount)> programs_{};
    std::unordered_map<RenderTargetId, RtRecord> rts_;
    std::unordered_map<TextureId, TexRecord> textures_;
    std::unordered_map<BufferId, BufRecord> buffers_;
};

}  // namespace livebeauty::rhi::gles
