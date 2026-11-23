#import "metal_device.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <unordered_map>

#include "metal_shaders.h"

namespace livebeauty::rhi::metal {

struct MetalDevice::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLBuffer> quadVb = nil;

    // Active command buffer & render pass state (one command buffer per frame or flush)
    id<MTLCommandBuffer> cmdBuf = nil;
    id<MTLRenderCommandEncoder> encoder = nil;

    struct RtRec {
        TextureId texId = kInvalidTexture;
        int width = 0;
        int height = 0;
        TextureFormat format = TextureFormat::kRgba8;
    };
    struct TexRec {
        id<MTLTexture> tex = nil;
        bool external = false;
    };

    uint32_t nextId = 1;
    std::unordered_map<RenderTargetId, RtRec> rts;
    std::unordered_map<TextureId, TexRec> textures;
    std::unordered_map<BufferId, id<MTLBuffer>> buffers;
    // Key = (shaderId << 8) | (layout << 4) | (blend << 1) | colorMaskAlphaOnly
    std::unordered_map<uint32_t, id<MTLRenderPipelineState>> pipelines;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        queue = [device newCommandQueue];
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalBeautyShadersSource];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        library = [device newLibraryWithSource:src options:opts error:&err];
        if (!library) {
            NSLog(@"[LiveBeauty Metal] Shader compilation error: %@", err);
            return;
        }
        static constexpr float kQuad[16] = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
        };
        quadVb = [device newBufferWithBytes:kQuad length:sizeof(kQuad) options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> ensureCmdBuf() {
        if (!cmdBuf) {
            cmdBuf = [queue commandBuffer];
        }
        return cmdBuf;
    }

    MTLVertexDescriptor* vertexDescFor(VertexLayout layout) const {
        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        switch (layout) {
            case VertexLayout::kQuadPos2Uv2:
            case VertexLayout::kMeshPos2Uv2:
                vd.attributes[0].format = MTLVertexFormatFloat2;
                vd.attributes[0].offset = 0;
                vd.attributes[0].bufferIndex = 0;
                vd.attributes[1].format = MTLVertexFormatFloat2;
                vd.attributes[1].offset = 8;
                vd.attributes[1].bufferIndex = 0;
                vd.layouts[0].stride = 16;
                break;
            case VertexLayout::kSkinMaskPos2W3:
                vd.attributes[0].format = MTLVertexFormatFloat2;
                vd.attributes[0].offset = 0;
                vd.attributes[0].bufferIndex = 0;
                vd.attributes[1].format = MTLVertexFormatFloat3;
                vd.attributes[1].offset = 8;
                vd.attributes[1].bufferIndex = 0;
                vd.layouts[0].stride = 20;
                break;
            case VertexLayout::kRetouchPos2Uv2A4:
                vd.attributes[0].format = MTLVertexFormatFloat2;
                vd.attributes[0].offset = 0;
                vd.attributes[0].bufferIndex = 0;
                vd.attributes[1].format = MTLVertexFormatFloat2;
                vd.attributes[1].offset = 8;
                vd.attributes[1].bufferIndex = 0;
                vd.attributes[2].format = MTLVertexFormatFloat4;
                vd.attributes[2].offset = 16;
                vd.attributes[2].bufferIndex = 0;
                vd.layouts[0].stride = 32;
                break;
            case VertexLayout::kMakeupPos2Uv2Uv2:
                vd.attributes[0].format = MTLVertexFormatFloat2;
                vd.attributes[0].offset = 0;
                vd.attributes[0].bufferIndex = 0;
                vd.attributes[1].format = MTLVertexFormatFloat2;
                vd.attributes[1].offset = 8;
                vd.attributes[1].bufferIndex = 0;
                vd.attributes[2].format = MTLVertexFormatFloat2;
                vd.attributes[2].offset = 16;
                vd.attributes[2].bufferIndex = 0;
                vd.layouts[0].stride = 24;
                break;
        }
        return vd;
    }

    void shaderNames(ShaderId shaderId, NSString* __strong& vsName, NSString* __strong& fsName) const {
        vsName = @"vs_quad";
        switch (shaderId) {
            case ShaderId::kDownsample:      vsName = @"vs_down";      fsName = @"fs_down"; break;
            case ShaderId::kBlurRgb:         vsName = @"vs_blur";      fsName = @"fs_blur_rgb"; break;
            case ShaderId::kBlurAlpha:       vsName = @"vs_blur";      fsName = @"fs_blur_alpha"; break;
            case ShaderId::kGuidedDiff:      vsName = @"vs_blur";      fsName = @"fs_guided_diff"; break;
            case ShaderId::kGuidedCoeff:     vsName = @"vs_blur";      fsName = @"fs_guided_coeff"; break;
            case ShaderId::kMeshWarp:        vsName = @"vs_quad";      fsName = @"fs_pass_through"; break;
            case ShaderId::kSkinMask:        vsName = @"vs_skin_mask"; fsName = @"fs_skin_mask"; break;
            case ShaderId::kRetouchSkin:     vsName = @"vs_retouch";   fsName = @"fs_retouch_skin"; break;
            case ShaderId::kRetouchEyeTeeth: vsName = @"vs_retouch";   fsName = @"fs_retouch_eye_teeth"; break;
            case ShaderId::kMakeupLayer:     vsName = @"vs_makeup";    fsName = @"fs_makeup_layer"; break;
            case ShaderId::kMakeupLip:       vsName = @"vs_makeup";    fsName = @"fs_makeup_lip"; break;
            case ShaderId::kMakeupIris:      vsName = @"vs_makeup";    fsName = @"fs_makeup_iris"; break;
            case ShaderId::kBodyWarp:        vsName = @"vs_quad";      fsName = @"fs_body_warp"; break;
            case ShaderId::kCompositePro:    vsName = @"vs_quad";      fsName = @"fs_composite_pro"; break;
            case ShaderId::kTemporalDenoise: vsName = @"vs_quad";      fsName = @"fs_temporal_denoise"; break;
            case ShaderId::kBgPremask:       vsName = @"vs_quad";      fsName = @"fs_bg_premask"; break;
            case ShaderId::kBgBlur:          vsName = @"vs_quad";      fsName = @"fs_bg_blur"; break;
            case ShaderId::kBgMix:           vsName = @"vs_quad";      fsName = @"fs_bg_mix"; break;
            case ShaderId::kHairAlpha:       vsName = @"vs_quad";      fsName = @"fs_hair_alpha"; break;
            case ShaderId::kBgHairMerge:     vsName = @"vs_quad";      fsName = @"fs_bg_hair_merge"; break;
            case ShaderId::kBgVirtualMix:    vsName = @"vs_quad";      fsName = @"fs_bg_virtual_mix"; break;
            case ShaderId::kStickerQuad:     vsName = @"vs_quad";      fsName = @"fs_sticker_quad"; break;
            case ShaderId::kPassThrough:
            default:                         vsName = @"vs_quad";      fsName = @"fs_pass_through"; break;
        }
    }

    id<MTLRenderPipelineState> getPipeline(ShaderId shaderId, VertexLayout layout, const RenderState& st) {
        const uint32_t key = (static_cast<uint32_t>(shaderId) << 8) |
                             (static_cast<uint32_t>(layout) << 4) |
                             (st.blendEnable ? 2u : 0u) |
                             (st.colorMaskAlphaOnly ? 1u : 0u);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;

        NSString* vsName = nil;
        NSString* fsName = nil;
        shaderNames(shaderId, vsName, fsName);
        id<MTLFunction> vf = [library newFunctionWithName:vsName];
        id<MTLFunction> ff = [library newFunctionWithName:fsName];
        if (!vf || !ff) return nil;

        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vf;
        desc.fragmentFunction = ff;
        desc.vertexDescriptor = vertexDescFor(layout);
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        if (st.blendEnable) {
            desc.colorAttachments[0].blendingEnabled = YES;
            desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
            desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
            desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        }
        if (st.colorMaskAlphaOnly) {
            desc.colorAttachments[0].writeMask = MTLColorWriteMaskAlpha;
        }

        NSError* err = nil;
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&err];
        if (!pso) {
            NSLog(@"[LiveBeauty Metal] Pipeline creation error (%@ / %@): %@", vsName, fsName, err);
            return nil;
        }
        pipelines[key] = pso;
        return pso;
    }

    void bindCommon(const RenderState& st, const TextureBindings& textures, const UniformBlock& uniforms) {
        if (st.cullBack) {
            [encoder setFrontFacingWinding:st.frontFaceCw ? MTLWindingClockwise : MTLWindingCounterClockwise];
            [encoder setCullMode:MTLCullModeBack];
        } else {
            [encoder setCullMode:MTLCullModeNone];
        }
        for (int i = 0; i < textures.count; ++i) {
            if (textures.slots[i] != kInvalidTexture) {
                auto it = this->textures.find(textures.slots[i]);
                if (it != this->textures.end() && it->second.tex) {
                    [encoder setFragmentTexture:it->second.tex atIndex:static_cast<NSUInteger>(i)];
                }
            }
        }
        if (uniforms.count > 0) {
            [encoder setVertexBytes:uniforms.v.data() length:sizeof(uniforms.v) atIndex:1];
            [encoder setFragmentBytes:uniforms.v.data() length:sizeof(uniforms.v) atIndex:1];
        }
    }
};

MetalDevice::MetalDevice() : impl_(std::make_unique<Impl>()) {}
MetalDevice::~MetalDevice() {
    flush();
}

bool MetalDevice::isReady() const {
    return impl_ && impl_->device != nil && impl_->queue != nil && impl_->library != nil;
}

id<MTLDevice> MetalDevice::nativeDevice() const { return impl_->device; }
id<MTLCommandQueue> MetalDevice::nativeQueue() const { return impl_->queue; }

TextureId MetalDevice::wrapMetalTexture(id<MTLTexture> tex) {
    if (!tex) return kInvalidTexture;
    for (const auto& [id, rec] : impl_->textures) {
        if (rec.external && rec.tex == tex) return id;
    }
    const TextureId id = impl_->nextId++;
    impl_->textures[id] = Impl::TexRec{tex, true};
    return id;
}

id<MTLTexture> MetalDevice::unwrapMetalTexture(TextureId tex) const {
    auto it = impl_->textures.find(tex);
    return (it != impl_->textures.end()) ? it->second.tex : nil;
}

bool MetalDevice::readbackTextureRgba8(TextureId tex, uint8_t* outRgba, int width, int height) {
    flush();
    id<MTLTexture> mtlTex = unwrapMetalTexture(tex);
    if (!mtlTex || !outRgba || width <= 0 || height <= 0) return false;
    const MTLRegion region = MTLRegionMake2D(0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
    [mtlTex getBytes:outRgba bytesPerRow:static_cast<NSUInteger>(width * 4) fromRegion:region mipmapLevel:0];
    return true;
}

RenderTargetId MetalDevice::createRenderTarget(int width, int height, TextureFormat format) {
    if (!isReady()) return kInvalidRenderTarget;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(format == TextureFormat::kR8 ? MTLPixelFormatR8Unorm : MTLPixelFormatRGBA8Unorm)
                                     width:static_cast<NSUInteger>(width)
                                    height:static_cast<NSUInteger>(height)
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [impl_->device newTextureWithDescriptor:desc];
    const TextureId texId = impl_->nextId++;
    impl_->textures[texId] = Impl::TexRec{tex, false};
    const RenderTargetId rtId = impl_->nextId++;
    impl_->rts[rtId] = Impl::RtRec{texId, width, height, format};
    return rtId;
}

void MetalDevice::destroyRenderTarget(RenderTargetId rt) {
    auto it = impl_->rts.find(rt);
    if (it == impl_->rts.end()) return;
    destroyTexture(it->second.texId);
    impl_->rts.erase(it);
}

TextureId MetalDevice::textureOf(RenderTargetId rt) const {
    auto it = impl_->rts.find(rt);
    return (it != impl_->rts.end()) ? it->second.texId : kInvalidTexture;
}

TextureId MetalDevice::createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) {
    if (!isReady()) return kInvalidTexture;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(format == TextureFormat::kR8 ? MTLPixelFormatR8Unorm : MTLPixelFormatRGBA8Unorm)
                                     width:static_cast<NSUInteger>(width)
                                    height:static_cast<NSUInteger>(height)
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [impl_->device newTextureWithDescriptor:desc];
    if (pixels) {
        const NSUInteger bpp = (format == TextureFormat::kR8) ? 1 : 4;
        [tex replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height))
               mipmapLevel:0
                 withBytes:pixels
               bytesPerRow:static_cast<NSUInteger>(width) * bpp];
    }
    const TextureId id = impl_->nextId++;
    impl_->textures[id] = Impl::TexRec{tex, false};
    return id;
}

void MetalDevice::updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) {
    id<MTLTexture> mtlTex = unwrapMetalTexture(tex);
    if (!mtlTex || !pixels) return;
    const NSUInteger bpp = (format == TextureFormat::kR8) ? 1 : 4;
    [mtlTex replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height))
              mipmapLevel:0
                withBytes:pixels
              bytesPerRow:static_cast<NSUInteger>(width) * bpp];
}

void MetalDevice::destroyTexture(TextureId tex) {
    impl_->textures.erase(tex);
}

BufferId MetalDevice::createDynamicVertexBuffer(size_t byteSize) {
    if (!isReady()) return kInvalidBuffer;
    id<MTLBuffer> buf = [impl_->device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
    const BufferId id = impl_->nextId++;
    impl_->buffers[id] = buf;
    return id;
}

void MetalDevice::updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) {
    auto it = impl_->buffers.find(buf);
    if (it == impl_->buffers.end() || !data) return;
    if (it->second.length < byteSize) {
        it->second = [impl_->device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
    }
    std::memcpy([it->second contents], data, byteSize);
}

BufferId MetalDevice::createStaticIndexBuffer(const uint16_t* indices, size_t count) {
    if (!isReady() || !indices || count == 0) return kInvalidBuffer;
    id<MTLBuffer> buf = [impl_->device newBufferWithBytes:indices
                                                   length:count * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
    const BufferId id = impl_->nextId++;
    impl_->buffers[id] = buf;
    return id;
}

void MetalDevice::destroyBuffer(BufferId buf) {
    impl_->buffers.erase(buf);
}

void MetalDevice::beginPass(RenderTargetId rt, int viewportW, int viewportH, bool clear, const Vec4& clearColor) {
    if (!isReady()) return;
    auto it = impl_->rts.find(rt);
    if (it == impl_->rts.end()) return;
    id<MTLTexture> targetTex = unwrapMetalTexture(it->second.texId);
    if (!targetTex) return;

    id<MTLCommandBuffer> cb = impl_->ensureCmdBuf();
    MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = targetTex;
    passDesc.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (clear) {
        passDesc.colorAttachments[0].clearColor = MTLClearColorMake(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
    }
    impl_->encoder = [cb renderCommandEncoderWithDescriptor:passDesc];
    MTLViewport vp{0.0, 0.0, static_cast<double>(viewportW), static_cast<double>(viewportH), 0.0, 1.0};
    [impl_->encoder setViewport:vp];
}

void MetalDevice::drawFullScreenQuad(
    ShaderId shader, const RenderState& state,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (!impl_->encoder) return;
    id<MTLRenderPipelineState> pso = impl_->getPipeline(shader, VertexLayout::kQuadPos2Uv2, state);
    if (!pso) return;
    [impl_->encoder setRenderPipelineState:pso];
    [impl_->encoder setVertexBuffer:impl_->quadVb offset:0 atIndex:0];
    impl_->bindCommon(state, textures, uniforms);
    [impl_->encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

void MetalDevice::drawIndexed(
    ShaderId shader, const RenderState& state,
    BufferId vertexBuffer, BufferId indexBuffer,
    int indexOffset, int indexCount, VertexLayout layout,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (!impl_->encoder) return;
    auto vbIt = impl_->buffers.find(vertexBuffer);
    auto ibIt = impl_->buffers.find(indexBuffer);
    if (vbIt == impl_->buffers.end() || ibIt == impl_->buffers.end()) return;
    id<MTLRenderPipelineState> pso = impl_->getPipeline(shader, layout, state);
    if (!pso) return;

    [impl_->encoder setRenderPipelineState:pso];
    [impl_->encoder setVertexBuffer:vbIt->second offset:0 atIndex:0];
    impl_->bindCommon(state, textures, uniforms);
    [impl_->encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                               indexCount:static_cast<NSUInteger>(indexCount)
                                indexType:MTLIndexTypeUInt16
                              indexBuffer:ibIt->second
                        indexBufferOffset:static_cast<NSUInteger>(indexOffset) * sizeof(uint16_t)];
}

void MetalDevice::endPass() {
    if (impl_->encoder) {
        [impl_->encoder endEncoding];
        impl_->encoder = nil;
    }
}

void MetalDevice::flush() {
    endPass();
    if (impl_->cmdBuf) {
        [impl_->cmdBuf commit];
        [impl_->cmdBuf waitUntilCompleted];
        impl_->cmdBuf = nil;
    }
}

}  // namespace livebeauty::rhi::metal
