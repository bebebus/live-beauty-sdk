#import "FLTLiveBeautyMetalEngine.h"

#import <CoreGraphics/CoreGraphics.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/engine.h"
#include "metal_device.h"

namespace {

float dictFloat(NSDictionary* d, NSString* key, float fallback) {
    id val = d[key];
    if ([val isKindOfClass:[NSNumber class]]) return [(NSNumber*)val floatValue];
    return fallback;
}

bool dictBool(NSDictionary* d, NSString* key, bool fallback) {
    id val = d[key];
    if ([val isKindOfClass:[NSNumber class]]) return [(NSNumber*)val boolValue];
    return fallback;
}

std::string dictString(NSDictionary* d, NSString* key, const char* fallback) {
    id val = d[key];
    if ([val isKindOfClass:[NSString class]]) {
        const char* s = [(NSString*)val UTF8String];
        return s ? std::string(s) : std::string(fallback);
    }
    return std::string(fallback);
}

livebeauty::BeautyConfig parseObjcConfig(NSDictionary<NSString*, id>* d) {
    livebeauty::BeautyConfig c{};
    c.enabled = dictBool(d, @"enabled", true);
    c.smoothing = dictFloat(d, @"smoothing", 0.0f);
    c.brightness = dictFloat(d, @"brightness", 0.0f);
    c.filter = livebeauty::parseFilterStyle(dictString(d, @"filter", "none"));
    c.skinOnly = dictBool(d, @"skinOnly", false);
    c.sharpness = dictFloat(d, @"sharpness", 0.0f);
    c.saturation = dictFloat(d, @"saturation", 1.0f);
    c.filterIntensity = dictFloat(d, @"filterIntensity", 1.0f);
    c.compareOriginal = dictBool(d, @"compareOriginal", false);
    c.faceSlim = dictFloat(d, @"faceSlim", 0.0f);
    c.eyeEnlarge = dictFloat(d, @"eyeEnlarge", 0.0f);
    c.whiten = dictFloat(d, @"whiten", 0.0f);
    c.rosy = dictFloat(d, @"rosy", 0.0f);
    c.waistSlim = dictFloat(d, @"waistSlim", 0.0f);
    c.bustEnhance = dictFloat(d, @"bustEnhance", 0.0f);
    c.legLengthen = dictFloat(d, @"legLengthen", 0.0f);
    c.shoulderSlim = dictFloat(d, @"shoulderSlim", 0.0f);
    c.neckLengthen = dictFloat(d, @"neckLengthen", 0.0f);
    c.chinShape = dictFloat(d, @"chinShape", 0.0f);
    c.foreheadShape = dictFloat(d, @"foreheadShape", 0.0f);
    c.noseSlim = dictFloat(d, @"noseSlim", 0.0f);
    c.vFace = dictFloat(d, @"vFace", 0.0f);
    c.faceNarrow = dictFloat(d, @"faceNarrow", 0.0f);
    c.faceSmall = dictFloat(d, @"faceSmall", 0.0f);
    c.cheekboneSlim = dictFloat(d, @"cheekboneSlim", 0.0f);
    c.eyeDistance = dictFloat(d, @"eyeDistance", 0.0f);
    c.eyeCorner = dictFloat(d, @"eyeCorner", 0.0f);
    c.noseLength = dictFloat(d, @"noseLength", 0.0f);
    c.mouthSize = dictFloat(d, @"mouthSize", 0.0f);
    c.mouthSmile = dictFloat(d, @"mouthSmile", 0.0f);
    c.cranialTop = dictFloat(d, @"cranialTop", 0.0f);
    c.darkCircles = dictFloat(d, @"darkCircles", 0.0f);
    c.brightEyes = dictFloat(d, @"brightEyes", 0.0f);
    c.teethWhiten = dictFloat(d, @"teethWhiten", 0.0f);
    c.nasolabial = dictFloat(d, @"nasolabial", 0.0f);
    c.backgroundBlur = dictFloat(d, @"backgroundBlur", 0.0f);
    c.lutUrl = dictString(d, @"lutUrl", "");
    c.lutIntensity = dictFloat(d, @"lutIntensity", 1.0f);
    c.skinTexture = dictFloat(d, @"skinTexture", 0.0f);
    c.radiance = dictFloat(d, @"radiance", 0.0f);
    c.lipstick = dictFloat(d, @"lipstick", 0.0f);
    c.lipstickColor = livebeauty::parseLipstickColor(dictString(d, @"lipstickColor", "rose_bean"));
    c.lipstickFinish = livebeauty::parseLipstickFinish(dictString(d, @"lipstickFinish", "velvet"));
    c.blush = dictFloat(d, @"blush", 0.0f);
    c.blushStyle = livebeauty::parseBlushStyle(dictString(d, @"blushStyle", "apple"));
    c.blushColor = livebeauty::parseBlushColor(dictString(d, @"blushColor", "peach"));
    c.eyebrow = dictFloat(d, @"eyebrow", 0.0f);
    c.eyebrowStyle = livebeauty::parseEyebrowStyle(dictString(d, @"eyebrowStyle", "natural"));
    c.eyebrowColor = livebeauty::parseEyebrowColor(dictString(d, @"eyebrowColor", "dark_brown"));
    c.eyeshadow = dictFloat(d, @"eyeshadow", 0.0f);
    c.eyeshadowColor = livebeauty::parseEyeshadowColor(dictString(d, @"eyeshadowColor", "earth"));
    c.eyeliner = dictFloat(d, @"eyeliner", 0.0f);
    c.aegyoSal = dictFloat(d, @"aegyoSal", 0.0f);
    c.contour = dictFloat(d, @"contour", 0.0f);
    c.highlight = dictFloat(d, @"highlight", 0.0f);
    c.contactLens = dictFloat(d, @"contactLens", 0.0f);
    c.contactLensStyle = livebeauty::parseContactLensStyle(dictString(d, @"contactLensStyle", "hazel"));
    c.catchlight = dictFloat(d, @"catchlight", 0.0f);
    c.relight = dictFloat(d, @"relight", 0.0f);
    c.relightStyle = livebeauty::parseRelightStyle(dictString(d, @"relightStyle", "natural"));
    c.oilControl = dictFloat(d, @"oilControl", 0.0f);
    c.neckTone = dictFloat(d, @"neckTone", 0.0f);
    c.ambientMakeupBlend = dictFloat(d, @"ambientMakeupBlend", 0.5f);
    c.matteTemporal = dictFloat(d, @"matteTemporal", 0.6f);
    c.blemishRemove = dictFloat(d, @"blemishRemove", 0.0f);
    c.performanceTier = livebeauty::parsePerformanceTier(dictString(d, @"performanceTier", "auto"));
    c.maxFaces = static_cast<int>(dictFloat(d, @"maxFaces", 1.0f));
    c.frameDelayFrames = static_cast<int>(dictFloat(d, @"frameDelayFrames", 0.0f));
    return c;
}

}  // namespace

@implementation FLTLiveBeautyMetalEngine {
    std::mutex _mutex;
    std::unique_ptr<livebeauty::rhi::metal::MetalDevice> _device;
    std::unique_ptr<livebeauty::LiveBeautyEngine> _engine;
    livebeauty::rhi::PipelineAssets _assets;
    CVMetalTextureCacheRef _textureCache;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = std::make_unique<livebeauty::rhi::metal::MetalDevice>();
        if (_device->isReady()) {
            _engine = std::make_unique<livebeauty::LiveBeautyEngine>(*_device);
            NSProcessInfo* pi = [NSProcessInfo processInfo];
            const char* gpuName = _device->nativeDevice().name.UTF8String;
            const int cores = static_cast<int>(pi.processorCount);
            const int ramMb = static_cast<int>(pi.physicalMemory / (1024ULL * 1024ULL));
            _engine->setHardwareProfile(gpuName ? std::string(gpuName) : std::string("Apple GPU"), cores, ramMb);
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, _device->nativeDevice(), nullptr, &_textureCache);
        }
    }
    return self;
}

- (void)dealloc {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->releaseGpuResources();
    if (_textureCache) {
        CFRelease(_textureCache);
        _textureCache = nullptr;
    }
}

- (BOOL)isReady {
    return _device && _device->isReady() && _engine != nullptr;
}

- (void)setConfiguration:(NSDictionary<NSString*, id>*)config {
    if (!config) return;
    const auto parsed = parseObjcConfig(config);
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->setConfig(parsed);
}

- (BOOL)setAssetImage:(nullable CGImageRef)image forSlot:(NSString*)slot {
    if (!self.isReady || !slot) return NO;
    std::lock_guard<std::mutex> lock(_mutex);
    livebeauty::rhi::TextureId* targetSlot = nullptr;
    if ([slot isEqualToString:@"lut"]) targetSlot = &_assets.lutTex;
    else if ([slot isEqualToString:@"blush"]) targetSlot = &_assets.blushAtlas;
    else if ([slot isEqualToString:@"eyebrow"]) targetSlot = &_assets.eyebrowAtlas;
    else if ([slot isEqualToString:@"eye_makeup"]) targetSlot = &_assets.eyeMakeupAtlas;
    else if ([slot isEqualToString:@"contour_highlight"]) targetSlot = &_assets.contourHighlightAtlas;
    else if ([slot isEqualToString:@"iris"]) targetSlot = &_assets.irisAtlas;
    if (!targetSlot) return NO;

    if (*targetSlot != livebeauty::rhi::kInvalidTexture) {
        _device->destroyTexture(*targetSlot);
        *targetSlot = livebeauty::rhi::kInvalidTexture;
    }
    if (image != nullptr) {
        const size_t w = CGImageGetWidth(image);
        const size_t h = CGImageGetHeight(image);
        if (w == 0 || h == 0) return NO;
        std::vector<uint8_t> rgba(w * h * 4, 0);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(
            rgba.data(), w, h, 8, w * 4, cs,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);
        if (!ctx) return NO;
        CGContextDrawImage(ctx, CGRectMake(0, 0, static_cast<CGFloat>(w), static_cast<CGFloat>(h)), image);
        CGContextRelease(ctx);
        *targetSlot = _device->createTexture2D(
            static_cast<int>(w), static_cast<int>(h),
            livebeauty::rhi::TextureFormat::kRgba8, rgba.data());
    }
    _engine->setAssets(_assets);
    return YES;
}

- (BOOL)submitFaceLandmarks:(const float*)mpXy
                 pointCount:(int)pointCount
                      width:(int)width
                     height:(int)height
                 sampleTsMs:(int64_t)sampleTsMs
                 receivedMs:(int64_t)receivedMs {
    if (!self.isReady || !mpXy) return NO;
    std::lock_guard<std::mutex> lock(_mutex);
    return _engine->submitFaceLandmarks(sampleTsMs, mpXy, pointCount, width, height, receivedMs) ? YES : NO;
}

- (void)submitFaceMissAtMs:(int64_t)sampleTsMs {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->submitFaceMiss(sampleTsMs);
}

- (void)resetTracking {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->resetTracking();
}

- (void)setAmbientR:(float)r g:(float)g b:(float)b {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->setAmbientRgb(r, g, b);
}

- (void)setOcclusionMaskTexture:(nullable id<MTLTexture>)occlusionMask {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    const auto tex = occlusionMask ? _device->wrapMetalTexture(occlusionMask) : livebeauty::rhi::kInvalidTexture;
    _engine->setOcclusionMask(tex);
}

- (void)setAiResidualTexture:(nullable id<MTLTexture>)residualTex
                     roiLeft:(float)roiLeft
                      roiTop:(float)roiTop
                    roiWidth:(float)roiWidth
                   roiHeight:(float)roiHeight
                    strength:(float)strength {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    const auto tex = residualTex ? _device->wrapMetalTexture(residualTex) : livebeauty::rhi::kInvalidTexture;
    _engine->setAiResidual(tex, {roiLeft, roiTop, roiWidth, roiHeight}, strength);
}

- (void)setAr3dAccessoryAtSlot:(int)slot
                       enabled:(BOOL)enabled
                        anchor:(int)anchor
                       texture:(nullable id<MTLTexture>)texture
                         scale:(float)scale
                       offsetX:(float)offsetX
                       offsetY:(float)offsetY
                       offsetZ:(float)offsetZ
                        yawRad:(float)yawRad
                      pitchRad:(float)pitchRad
                       rollRad:(float)rollRad
                       opacity:(float)opacity
                 textureAspect:(float)textureAspect
               springStiffness:(float)springStiffness
                 springDamping:(float)springDamping
                 headOcclusion:(BOOL)headOcclusion {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    livebeauty::math::Ar3dAccessoryConfig cfg{};
    cfg.enabled = (enabled == YES);
    cfg.anchor = static_cast<livebeauty::math::Ar3dAnchor>(anchor);
    cfg.texture = texture ? _device->wrapMetalTexture(texture) : livebeauty::rhi::kInvalidTexture;
    cfg.scale = scale;
    cfg.offsetX = offsetX;
    cfg.offsetY = offsetY;
    cfg.offsetZ = offsetZ;
    cfg.yawOffsetRad = yawRad;
    cfg.pitchOffsetRad = pitchRad;
    cfg.rollOffsetRad = rollRad;
    cfg.opacity = opacity;
    cfg.textureAspect = textureAspect;
    cfg.springStiffness = springStiffness;
    cfg.springDamping = springDamping;
    cfg.headOcclusion = (headOcclusion == YES);
    _engine->setAr3dAccessory(slot, cfg);
}

- (void)clearAr3dAccessories {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->clearAr3dAccessories();
}

- (NSDictionary<NSString*, NSNumber*>*)statsDictionary {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_engine) return @{};
    const auto& st = _engine->stats();
    return @{
        @"lastFrameMs": @(st.lastFrameMs),
        @"emaFrameMs": @(st.emaFrameMs),
        @"fps": @(st.fps),
        @"loadLevel": @(st.loadLevel),
        @"resolvedTier": @(st.resolvedTier),
        @"facePresence": @(st.facePresence),
        @"hairShown": @(st.hairShown),
        @"activePassesMask": @(st.activePassesMask),
        @"frameCount": @(st.frameCount),
        @"activeFaces": @(st.activeFaces),
    };
}

- (nullable id<MTLTexture>)processTexture:(id<MTLTexture>)inputTexture
                         frameTimestampMs:(int64_t)frameTsMs
                                    nowMs:(int64_t)nowMs {
    return [self processTexture:inputTexture
               frameTimestampMs:frameTsMs
                          nowMs:nowMs
                     personMask:nil
                       hairMask:nil
                hairMaskStampMs:livebeauty::kTimeInvalid
                      lightLuma:0.30f
                   bitrateRatio:1.0f];
}

- (nullable id<MTLTexture>)processTexture:(id<MTLTexture>)inputTexture
                         frameTimestampMs:(int64_t)frameTsMs
                                    nowMs:(int64_t)nowMs
                               personMask:(nullable id<MTLTexture>)personMask
                                 hairMask:(nullable id<MTLTexture>)hairMask
                          hairMaskStampMs:(int64_t)hairMaskStampMs
                                lightLuma:(float)lightLuma
                             bitrateRatio:(float)bitrateRatio {
    if (!self.isReady || !inputTexture) return inputTexture;
    std::lock_guard<std::mutex> lock(_mutex);
    const auto inId = _device->wrapMetalTexture(inputTexture);
    const auto maskId = personMask ? _device->wrapMetalTexture(personMask) : livebeauty::rhi::kInvalidTexture;
    const auto hairId = hairMask ? _device->wrapMetalTexture(hairMask) : livebeauty::rhi::kInvalidTexture;
    const auto outId = _engine->processFrame(
        inId,
        static_cast<int>(inputTexture.width),
        static_cast<int>(inputTexture.height),
        frameTsMs, nowMs, maskId, lightLuma, bitrateRatio, hairId, hairMaskStampMs);
    _device->flush();
    return _device->unwrapMetalTexture(outId);
}

- (nullable CVPixelBufferRef)copyProcessedPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                     frameTimestampMs:(int64_t)frameTsMs
                                                nowMs:(int64_t)nowMs {
    if (!self.isReady || !pixelBuffer || !_textureCache) return nullptr;
    const size_t w = CVPixelBufferGetWidth(pixelBuffer);
    const size_t h = CVPixelBufferGetHeight(pixelBuffer);
    if (w == 0 || h == 0) return nullptr;

    CVMetalTextureRef cvTex = nullptr;
    const CVReturn rc = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, _textureCache, pixelBuffer, nullptr,
        MTLPixelFormatBGRA8Unorm, w, h, 0, &cvTex);
    if (rc != kCVReturnSuccess || !cvTex) return nullptr;

    id<MTLTexture> inMtl = CVMetalTextureGetTexture(cvTex);
    id<MTLTexture> outMtl = [self processTexture:inMtl frameTimestampMs:frameTsMs nowMs:nowMs];
    CFRelease(cvTex);
    if (!outMtl) return nullptr;

    CVPixelBufferRef outPb = nullptr;
    NSDictionary* attrs = @{
        (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
        (NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
    };
    if (CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32RGBA,
                            (__bridge CFDictionaryRef)attrs, &outPb) != kCVReturnSuccess || !outPb) {
        return nullptr;
    }
    CVPixelBufferLockBaseAddress(outPb, 0);
    uint8_t* dst = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(outPb));
    const size_t stride = CVPixelBufferGetBytesPerRow(outPb);
    [outMtl getBytes:dst bytesPerRow:stride fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    CVPixelBufferUnlockBaseAddress(outPb, 0);
    return outPb;
}

@end
