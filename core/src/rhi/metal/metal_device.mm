#import "metal_device.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "metal_shaders.h"

namespace livebeauty::rhi::metal {

struct MetalDevice::Impl {
    static constexpr NSUInteger kSlabAlign = 256;
    static constexpr NSUInteger kSlabDefaultSize = 256 * 1024;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLBuffer> quadVb = nil;

    // Active command buffer & render pass state (one command buffer per frame or flush)
    id<MTLCommandBuffer> cmdBuf = nil;
    id<MTLRenderCommandEncoder> encoder = nil;
    MTLPixelFormat activeMtlFormat = MTLPixelFormatRGBA8Unorm;

    // Per-frame dynamic vertex buffer slab arena: prevents intra-frame VBO overwrites across
    // multi-pass / multi-slot / dual-face draws sharing one uncommitted MTLCommandBuffer.
    std::vector<id<MTLBuffer>> dynamicSlabs;
    size_t activeSlab = 0;
    NSUInteger slabHead = 0;

    static MTLPixelFormat mtlFormatFor(TextureFormat format) {
        switch (format) {
            case TextureFormat::kR8:      return MTLPixelFormatR8Unorm;
            case TextureFormat::kRgba16f: return MTLPixelFormatRGBA16Float;
            case TextureFormat::kRgba8:
            default:                      return MTLPixelFormatRGBA8Unorm;
        }
    }

    struct RtRec {
        TextureId texId = kInvalidTexture;
        int width = 0;
        int height = 0;
        TextureFormat format = TextureFormat::kRgba8;
    };
    struct TexRec {
        id<MTLTexture> tex = nil;
        bool external = false;
        bool transient = false;
    };
    struct BufRec {
        id<MTLBuffer> buf = nil;
        NSUInteger offset = 0;
        bool dynamic = false;
    };

    uint32_t nextId = 1;
    std::unordered_map<RenderTargetId, RtRec> rts;
    std::unordered_map<TextureId, TexRec> textures;
    std::unordered_map<BufferId, BufRec> buffers;
    // Key = (activeMtlFormat << 24) | (shaderId << 8) | (layout << 4) | (blend << 1) | colorMaskAlphaOnly
    std::unordered_map<uint64_t, id<MTLRenderPipelineState>> pipelines;

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
            case VertexLayout::kAr3dPos2Uv2Z1:
                vd.attributes[0].format = MTLVertexFormatFloat2;
                vd.attributes[0].offset = 0;
                vd.attributes[0].bufferIndex = 0;
                vd.attributes[1].format = MTLVertexFormatFloat2;
                vd.attributes[1].offset = 8;
                vd.attributes[1].bufferIndex = 0;
                vd.attributes[2].format = MTLVertexFormatFloat;
                vd.attributes[2].offset = 16;
                vd.attributes[2].bufferIndex = 0;
                vd.layouts[0].stride = 20;
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
            case ShaderId::kGuidedDiff:         vsName = @"vs_blur";      fsName = @"fs_guided_diff"; break;
            case ShaderId::kGuidedCoeff:        vsName = @"vs_blur";      fsName = @"fs_guided_coeff"; break;
            case ShaderId::kGuidedWeightBlur:   vsName = @"vs_blur";      fsName = @"fs_guided_weight_blur"; break;
            case ShaderId::kGuidedWeightDiff:   vsName = @"vs_blur";      fsName = @"fs_guided_weight_diff"; break;
            case ShaderId::kGuidedWeightCoeff:  vsName = @"vs_blur";      fsName = @"fs_guided_weight_coeff"; break;
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
            case ShaderId::kHairAlpha:       vsName = @"vs_semantic";  fsName = @"fs_hair_alpha"; break;
            case ShaderId::kBgHairMerge:     vsName = @"vs_quad";      fsName = @"fs_bg_hair_merge"; break;
            case ShaderId::kBgVirtualMix:    vsName = @"vs_quad";      fsName = @"fs_bg_virtual_mix"; break;
            case ShaderId::kStickerQuad:     vsName = @"vs_quad";      fsName = @"fs_sticker_quad"; break;
            case ShaderId::kMatteRefine:     vsName = @"vs_quad";      fsName = @"fs_matte_refine"; break;
            case ShaderId::kBlemishHeal:     vsName = @"vs_quad";      fsName = @"fs_blemish_heal"; break;
            case ShaderId::kAiResidualBlend: vsName = @"vs_quad";      fsName = @"fs_ai_residual_blend"; break;
            case ShaderId::kHeadDepthShell:  vsName = @"vs_ar3d";      fsName = @"fs_head_depth_shell"; break;
            case ShaderId::kAr3dMesh:        vsName = @"vs_ar3d";      fsName = @"fs_ar3d_mesh"; break;
            case ShaderId::kHairDye:         vsName = @"vs_quad";      fsName = @"fs_hair_dye"; break;
            case ShaderId::kChromaKey:        vsName = @"vs_quad";      fsName = @"fs_chroma_key"; break;
            case ShaderId::kParticleBurst:    vsName = @"vs_retouch";   fsName = @"fs_particle_burst"; break;
            case ShaderId::kPostBloomExtract: vsName = @"vs_quad";      fsName = @"fs_post_bloom_extract"; break;
            case ShaderId::kPostAtmosphere:   vsName = @"vs_quad";      fsName = @"fs_post_atmosphere"; break;
            case ShaderId::kPassThrough:
            default:                         vsName = @"vs_quad";      fsName = @"fs_pass_through"; break;
        }
    }

    id<MTLRenderPipelineState> getPipeline(ShaderId shaderId, VertexLayout layout, const RenderState& st) {
        const uint64_t key = (static_cast<uint64_t>(activeMtlFormat) << 24) |
                             (static_cast<uint64_t>(shaderId) << 8) |
                             (static_cast<uint64_t>(layout) << 4) |
                             (st.blendEnable ? 2ULL : 0ULL) |
                             (st.colorMaskAlphaOnly ? 1ULL : 0ULL);
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
        desc.colorAttachments[0].pixelFormat = activeMtlFormat;
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

    bool allocDynamicSlice(size_t byteSize, id<MTLBuffer>* outBuf, NSUInteger* outOffset) {
        if (!device || byteSize == 0) return false;
        const NSUInteger reqBytes = static_cast<NSUInteger>(byteSize);
        NSUInteger alignedHead = (slabHead + kSlabAlign - 1) & ~(kSlabAlign - 1);
        if (activeSlab < dynamicSlabs.size() &&
            alignedHead + reqBytes <= dynamicSlabs[activeSlab].length) {
            *outBuf = dynamicSlabs[activeSlab];
            *outOffset = alignedHead;
            slabHead = alignedHead + reqBytes;
            return true;
        }
        if (!dynamicSlabs.empty() && activeSlab < dynamicSlabs.size() && slabHead > 0) {
            ++activeSlab;
            slabHead = 0;
        }
        while (activeSlab < dynamicSlabs.size() && dynamicSlabs[activeSlab].length < reqBytes) {
            ++activeSlab;
        }
        if (activeSlab >= dynamicSlabs.size()) {
            const NSUInteger alignedReq = (reqBytes + kSlabAlign - 1) & ~(kSlabAlign - 1);
            const NSUInteger allocLen = std::max(kSlabDefaultSize, alignedReq);
            id<MTLBuffer> slab = [device newBufferWithLength:allocLen options:MTLResourceStorageModeShared];
            if (!slab) return false;
            dynamicSlabs.push_back(slab);
            activeSlab = dynamicSlabs.size() - 1;
        }
        *outBuf = dynamicSlabs[activeSlab];
        *outOffset = 0;
        slabHead = reqBytes;
        return true;
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

size_t MetalDevice::trackedTextureCount() const {
    return impl_ ? impl_->textures.size() : 0;
}

id<MTLDevice> MetalDevice::nativeDevice() const { return impl_->device; }
id<MTLCommandQueue> MetalDevice::nativeQueue() const { return impl_->queue; }

TextureId MetalDevice::wrapMetalTexture(id<MTLTexture> tex) {
    if (!tex) return kInvalidTexture;
    for (auto& [id, rec] : impl_->textures) {
        if (rec.external && rec.tex == tex) {
            rec.transient = false;
            return id;
        }
    }
    const TextureId id = impl_->nextId++;
    impl_->textures[id] = Impl::TexRec{tex, true, false};
    return id;
}

TextureId MetalDevice::wrapTransientMetalTexture(id<MTLTexture> tex) {
    if (!tex) return kInvalidTexture;
    for (const auto& [id, rec] : impl_->textures) {
        if (rec.external && rec.tex == tex) return id;
    }
    const TextureId id = impl_->nextId++;
    impl_->textures[id] = Impl::TexRec{tex, true, true};
    return id;
}

id<MTLTexture> MetalDevice::unwrapMetalTexture(TextureId tex) const {
    auto it = impl_->textures.find(tex);
    return (it != impl_->textures.end()) ? it->second.tex : nil;
}

bool MetalDevice::renderToExternalTexture(TextureId srcTex, id<MTLTexture> dstTex) {
    if (!isReady() || srcTex == kInvalidTexture || !dstTex) return false;
    if ((dstTex.usage & MTLTextureUsageRenderTarget) == 0) return false;
    if (!unwrapMetalTexture(srcTex)) return false;

    endPass();
    impl_->activeMtlFormat = dstTex.pixelFormat;
    id<MTLCommandBuffer> cb = impl_->ensureCmdBuf();
    MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = dstTex;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    impl_->encoder = [cb renderCommandEncoderWithDescriptor:passDesc];
    if (!impl_->encoder) {
        impl_->activeMtlFormat = MTLPixelFormatRGBA8Unorm;
        return false;
    }
    MTLViewport vp{
        0.0, 0.0,
        static_cast<double>(dstTex.width),
        static_cast<double>(dstTex.height),
        0.0, 1.0,
    };
    [impl_->encoder setViewport:vp];

    RenderState st{};
    TextureBindings tb{};
    UniformBlock ub{};
    tb.bind(0, srcTex);
    drawFullScreenQuad(ShaderId::kPassThrough, st, tb, ub);
    endPass();
    return true;
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
    if (!isReady() || width <= 0 || height <= 0) return kInvalidRenderTarget;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:Impl::mtlFormatFor(format)
                                     width:static_cast<NSUInteger>(width)
                                    height:static_cast<NSUInteger>(height)
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [impl_->device newTextureWithDescriptor:desc];
    if (!tex) return kInvalidRenderTarget;
    const TextureId texId = impl_->nextId++;
    impl_->textures[texId] = Impl::TexRec{tex, false, false};
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
    if (!isReady() || width <= 0 || height <= 0) return kInvalidTexture;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:Impl::mtlFormatFor(format)
                                     width:static_cast<NSUInteger>(width)
                                    height:static_cast<NSUInteger>(height)
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [impl_->device newTextureWithDescriptor:desc];
    if (!tex) return kInvalidTexture;
    if (pixels) {
        const NSUInteger bpp = (format == TextureFormat::kR8) ? 1 : (format == TextureFormat::kRgba16f ? 8 : 4);
        [tex replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height))
               mipmapLevel:0
                 withBytes:pixels
               bytesPerRow:static_cast<NSUInteger>(width) * bpp];
    }
    const TextureId id = impl_->nextId++;
    impl_->textures[id] = Impl::TexRec{tex, false, false};
    return id;
}

void MetalDevice::updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) {
    id<MTLTexture> mtlTex = unwrapMetalTexture(tex);
    if (!mtlTex || !pixels || width <= 0 || height <= 0) return;
    const NSUInteger bpp = (format == TextureFormat::kR8) ? 1 : (format == TextureFormat::kRgba16f ? 8 : 4);
    [mtlTex replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height))
              mipmapLevel:0
                withBytes:pixels
              bytesPerRow:static_cast<NSUInteger>(width) * bpp];
}

void MetalDevice::destroyTexture(TextureId tex) {
    impl_->textures.erase(tex);
}

BufferId MetalDevice::createDynamicVertexBuffer(size_t byteSize) {
    if (!isReady() || byteSize == 0) return kInvalidBuffer;
    const BufferId id = impl_->nextId++;
    impl_->buffers[id] = Impl::BufRec{nil, 0, true};
    return id;
}

void MetalDevice::updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) {
    auto it = impl_->buffers.find(buf);
    if (it == impl_->buffers.end() || !data || byteSize == 0) return;
    if (it->second.dynamic) {
        id<MTLBuffer> slab = nil;
        NSUInteger offset = 0;
        if (impl_->allocDynamicSlice(byteSize, &slab, &offset)) {
            std::memcpy(static_cast<uint8_t*>([slab contents]) + offset, data, byteSize);
            it->second.buf = slab;
            it->second.offset = offset;
        }
        return;
    }
    if (!it->second.buf || it->second.buf.length < byteSize) {
        it->second.buf = [impl_->device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
        it->second.offset = 0;
    }
    if (it->second.buf) {
        std::memcpy([it->second.buf contents], data, byteSize);
    }
}

BufferId MetalDevice::createStaticIndexBuffer(const uint16_t* indices, size_t count) {
    if (!isReady() || !indices || count == 0) return kInvalidBuffer;
    id<MTLBuffer> buf = [impl_->device newBufferWithBytes:indices
                                                   length:count * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
    if (!buf) return kInvalidBuffer;
    const BufferId id = impl_->nextId++;
    impl_->buffers[id] = Impl::BufRec{buf, 0, false};
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
    impl_->activeMtlFormat = Impl::mtlFormatFor(it->second.format);

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
    if (!impl_->encoder || indexCount <= 0 || indexOffset < 0) return;
    auto vbIt = impl_->buffers.find(vertexBuffer);
    auto ibIt = impl_->buffers.find(indexBuffer);
    if (vbIt == impl_->buffers.end() || ibIt == impl_->buffers.end()) return;
    if (!vbIt->second.buf || !ibIt->second.buf) return;
    id<MTLRenderPipelineState> pso = impl_->getPipeline(shader, layout, state);
    if (!pso) return;

    [impl_->encoder setRenderPipelineState:pso];
    [impl_->encoder setVertexBuffer:vbIt->second.buf offset:vbIt->second.offset atIndex:0];
    impl_->bindCommon(state, textures, uniforms);
    [impl_->encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                               indexCount:static_cast<NSUInteger>(indexCount)
                                indexType:MTLIndexTypeUInt16
                              indexBuffer:ibIt->second.buf
                        indexBufferOffset:static_cast<NSUInteger>(indexOffset) * sizeof(uint16_t)];
}

void MetalDevice::endPass() {
    if (impl_->encoder) {
        [impl_->encoder endEncoding];
        impl_->encoder = nil;
    }
    impl_->activeMtlFormat = MTLPixelFormatRGBA8Unorm;
}

void MetalDevice::flush() {
    if (!impl_) return;
    endPass();
    if (impl_->cmdBuf) {
        [impl_->cmdBuf commit];
        [impl_->cmdBuf waitUntilCompleted];
        impl_->cmdBuf = nil;
    }
    impl_->activeSlab = 0;
    impl_->slabHead = 0;
    for (auto it = impl_->textures.begin(); it != impl_->textures.end(); ) {
        if (it->second.transient) {
            it = impl_->textures.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace livebeauty::rhi::metal
