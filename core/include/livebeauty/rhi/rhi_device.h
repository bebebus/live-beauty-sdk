#pragma once

#include <cstddef>
#include <cstdint>
#include "livebeauty/rhi/rhi_types.h"

namespace livebeauty::rhi {

/**
 * Minimal cross-backend GPU device abstraction (GLES 2.0/3.0, Metal, Mock).
 * Designed for zero per-frame heap allocation and explicit shared-context safety.
 */
class IRhiDevice {
public:
    virtual ~IRhiDevice() = default;

    virtual RenderTargetId createRenderTarget(int width, int height, TextureFormat format = TextureFormat::kRgba8) = 0;
    virtual void destroyRenderTarget(RenderTargetId rt) = 0;
    virtual TextureId textureOf(RenderTargetId rt) const = 0;

    virtual TextureId createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) = 0;
    virtual void updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) = 0;
    virtual void destroyTexture(TextureId tex) = 0;

    virtual BufferId createDynamicVertexBuffer(size_t byteSize) = 0;
    virtual void updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) = 0;
    virtual BufferId createStaticIndexBuffer(const uint16_t* indices, size_t count) = 0;
    virtual void destroyBuffer(BufferId buf) = 0;

    virtual void beginPass(RenderTargetId rt, int viewportW, int viewportH, LoadOp loadOp, const Vec4& clearColor = {}) = 0;
    void beginPass(RenderTargetId rt, int viewportW, int viewportH, bool clear, const Vec4& clearColor = {}) {
        beginPass(rt, viewportW, viewportH, clear ? LoadOp::kClear : LoadOp::kDontCare, clearColor);
    }
    virtual void drawFullScreenQuad(
        ShaderId shader,
        const RenderState& state,
        const TextureBindings& textures,
        const UniformBlock& uniforms) = 0;
    virtual void drawIndexed(
        ShaderId shader,
        const RenderState& state,
        BufferId vertexBuffer,
        BufferId indexBuffer,
        int indexOffset,
        int indexCount,
        VertexLayout layout,
        const TextureBindings& textures,
        const UniformBlock& uniforms) = 0;
    virtual void endPass() = 0;

    virtual void prewarmShader(ShaderId /*shader*/) {}
    virtual void beginFrameGpuTimer() {}
    virtual void endFrameGpuTimer() {}
    virtual int64_t lastGpuFrameTimeNs() const { return 0; }
    virtual void flush() {}
};

}  // namespace livebeauty::rhi
