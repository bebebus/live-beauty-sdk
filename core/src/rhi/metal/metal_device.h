#pragma once

#include <cstdint>
#include <memory>
#include "livebeauty/rhi/rhi_device.h"

#ifdef __OBJC__
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#endif

namespace livebeauty::rhi::metal {

/** Apple Metal RHI device backend for iOS 15+ and macOS 12+. */
class MetalDevice final : public IRhiDevice {
public:
    MetalDevice();
    ~MetalDevice() override;

    bool isReady() const;
    size_t trackedTextureCount() const;

#ifdef __OBJC__
    id<MTLDevice> nativeDevice() const;
    id<MTLCommandQueue> nativeQueue() const;
    /** Wrap a persistent external MTLTexture (caller owns lifetime and calls destroyTexture when replaced). */
    TextureId wrapMetalTexture(id<MTLTexture> tex);
    /** Wrap a per-frame transient MTLTexture (automatically evicted from RHI table at the end of flush()). */
    TextureId wrapTransientMetalTexture(id<MTLTexture> tex);
    id<MTLTexture> unwrapMetalTexture(TextureId tex) const;
    /** Zero-copy GPU pass-through blit (supports RGBA8 -> BGRA8 IOSurface conversion in active command buffer). */
    bool renderToExternalTexture(TextureId srcTex, id<MTLTexture> dstTex);
    bool readbackTextureRgba8(TextureId tex, uint8_t* outRgba, int width, int height);
#endif

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
    void flush() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace livebeauty::rhi::metal
