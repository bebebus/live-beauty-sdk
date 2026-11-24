#import "FLTLiveBeautyMetalEngine.h"
#import "FLTVisionTracker.h"

#import <CoreGraphics/CoreGraphics.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/engine.h"
#include "metal_device.h"

namespace {

/**
 * Decode a CGImageRef into straight (unpremultiplied) top-down RGBA8 bytes.
 * Critical for packed channel atlases such as `eye_makeup.png` where A=0 stores the 4th channel
 * while R/G/B store eyeshadow/eyeliner masks (CoreGraphics `kCGImageAlphaPremultipliedLast` would
 * multiply RGB by A=0 and wipe 100% of the atlas).
 */
bool decodeStraightRgba(CGImageRef image, std::vector<uint8_t>& outRgba, size_t& outW, size_t& outH) {
    if (!image) return false;
    const size_t w = CGImageGetWidth(image);
    const size_t h = CGImageGetHeight(image);
    if (w == 0 || h == 0 || w > 8192 || h > 8192) return false;
    outW = w;
    outH = h;
    outRgba.assign(w * h * 4, 0);

    const size_t bpc = CGImageGetBitsPerComponent(image);
    const size_t bpp = CGImageGetBitsPerPixel(image);
    const size_t bpr = CGImageGetBytesPerRow(image);
    const CGBitmapInfo bmpInfo = CGImageGetBitmapInfo(image);
    const CGImageAlphaInfo alphaInfo = static_cast<CGImageAlphaInfo>(bmpInfo & kCGBitmapAlphaInfoMask);
    const uint32_t byteOrder = bmpInfo & kCGBitmapByteOrderMask;

    if (bpc == 8 && (bpp == 32 || bpp == 24) && (bmpInfo & kCGBitmapFloatComponents) == 0) {
        CGDataProviderRef provider = CGImageGetDataProvider(image);
        CFDataRef rawData = provider ? CGDataProviderCopyData(provider) : nullptr;
        if (rawData) {
            const size_t bytesPerPx = bpp / 8;
            const size_t minNeeded = (h - 1) * bpr + w * bytesPerPx;
            const size_t rawLen = static_cast<size_t>(CFDataGetLength(rawData));
            const uint8_t* src = CFDataGetBytePtr(rawData);
            if (src && rawLen >= minNeeded) {
                int rIdx = 0, gIdx = 1, bIdx = 2, aIdx = 3;
                bool hasAlpha = (bpp == 32);
                bool isPremul = false;
                if (bpp == 32) {
                    const bool little = (byteOrder == kCGBitmapByteOrder32Little);
                    const bool alphaFirst = (alphaInfo == kCGImageAlphaPremultipliedFirst ||
                                             alphaInfo == kCGImageAlphaFirst ||
                                             alphaInfo == kCGImageAlphaNoneSkipFirst);
                    hasAlpha = (alphaInfo != kCGImageAlphaNone &&
                                alphaInfo != kCGImageAlphaNoneSkipLast &&
                                alphaInfo != kCGImageAlphaNoneSkipFirst);
                    isPremul = (alphaInfo == kCGImageAlphaPremultipliedLast ||
                                alphaInfo == kCGImageAlphaPremultipliedFirst);
                    if (little) {
                        if (alphaFirst) { rIdx = 2; gIdx = 1; bIdx = 0; aIdx = 3; }  // BGRA in memory
                        else            { aIdx = 0; bIdx = 1; gIdx = 2; rIdx = 3; }  // ABGR in memory
                    } else {
                        if (alphaFirst) { aIdx = 0; rIdx = 1; gIdx = 2; bIdx = 3; }  // ARGB in memory
                        else            { rIdx = 0; gIdx = 1; bIdx = 2; aIdx = 3; }  // RGBA in memory
                    }
                }
                for (size_t y = 0; y < h; ++y) {
                    const uint8_t* row = src + y * bpr;
                    uint8_t* dst = outRgba.data() + y * w * 4;
                    for (size_t x = 0; x < w; ++x) {
                        const uint8_t* p = row + x * bytesPerPx;
                        int r = p[rIdx];
                        int g = p[gIdx];
                        int b = p[bIdx];
                        const int a = hasAlpha ? p[aIdx] : 255;
                        if (isPremul && a > 0 && a < 255) {
                            r = std::min(255, (r * 255 + a / 2) / a);
                            g = std::min(255, (g * 255 + a / 2) / a);
                            b = std::min(255, (b * 255 + a / 2) / a);
                        }
                        dst[x * 4 + 0] = static_cast<uint8_t>(r);
                        dst[x * 4 + 1] = static_cast<uint8_t>(g);
                        dst[x * 4 + 2] = static_cast<uint8_t>(b);
                        dst[x * 4 + 3] = static_cast<uint8_t>(a);
                    }
                }
                CFRelease(rawData);
                return true;
            }
            CFRelease(rawData);
        }
    }

    // Fallback for indexed/16-bpc/non-RGB color spaces: draw via CoreGraphics and unpremultiply
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        outRgba.data(), w, h, 8, w * 4, cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctx) return false;
    CGContextDrawImage(ctx, CGRectMake(0, 0, static_cast<CGFloat>(w), static_cast<CGFloat>(h)), image);
    CGContextRelease(ctx);
    for (size_t i = 0; i < w * h * 4; i += 4) {
        const int a = outRgba[i + 3];
        if (a > 0 && a < 255) {
            outRgba[i + 0] = static_cast<uint8_t>(std::min(255, (outRgba[i + 0] * 255 + a / 2) / a));
            outRgba[i + 1] = static_cast<uint8_t>(std::min(255, (outRgba[i + 1] * 255 + a / 2) / a));
            outRgba[i + 2] = static_cast<uint8_t>(std::min(255, (outRgba[i + 2] * 255 + a / 2) / a));
        }
    }
    return true;
}

livebeauty::math::BlendshapeInput distillBlendshapesFromLandmarks(const float* mpXy, int pointCount) {
    livebeauty::math::BlendshapeInput bs{};
    if (!mpXy || pointCount < 468) return bs;
    auto pt = [&](int idx, float& x, float& y) {
        x = mpXy[idx * 2 + 0];
        y = mpXy[idx * 2 + 1];
    };
    float x234, y234, x454, y454, x10, y10, x152, y152;
    pt(234, x234, y234); pt(454, x454, y454);
    pt(10, x10, y10);    pt(152, x152, y152);
    const float fw = std::max(std::hypot(x454 - x234, y454 - y234), 1e-3f);
    const float fh = std::max(std::hypot(x152 - x10, y152 - y10), 1e-3f);

    float x13, y13, x14, y14, x61, y61, x291, y291;
    pt(13, x13, y13); pt(14, x14, y14);
    pt(61, x61, y61); pt(291, x291, y291);
    const float lipOpen = std::hypot(x14 - x13, y14 - y13) / fh;
    bs.jawOpen = livebeauty::detail::clampf((lipOpen - 0.025f) / 0.135f, 0.0f, 1.0f);

    const float mouthW = std::hypot(x291 - x61, y291 - y61) / fw;
    const float cornerLift = ((0.5f * (y13 + y14)) - (0.5f * (y61 + y291))) / fh;
    bs.smile = livebeauty::detail::clampf((mouthW - 0.37f) * 4.2f + cornerLift * 5.5f, 0.0f, 1.0f);

    float x33, y33, x133, y133, x159, y159, x145, y145;
    float x263, y263, x362, y362, x386, y386, x374, y374;
    pt(33, x33, y33);     pt(133, x133, y133); pt(159, x159, y159); pt(145, x145, y145);
    pt(263, x263, y263);  pt(362, x362, y362); pt(386, x386, y386); pt(374, x374, y374);
    const float earL = std::hypot(x159 - x145, y159 - y145) / std::max(std::hypot(x133 - x33, y133 - y33), 1e-4f);
    const float earR = std::hypot(x386 - x374, y386 - y374) / std::max(std::hypot(x263 - x362, y263 - y362), 1e-4f);
    bs.blink = livebeauty::detail::clampf((0.19f - std::min(earL, earR)) / 0.11f, 0.0f, 1.0f);

    float x105, y105, x334, y334;
    pt(105, x105, y105); pt(334, x334, y334);
    const float browLift = (0.5f * ((y159 - y105) + (y386 - y334))) / fh;
    bs.browUp = livebeauty::detail::clampf((browLift - 0.088f) / 0.045f, 0.0f, 1.0f);
    return bs;
}

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
    c.hairDye = dictFloat(d, @"hairDye", 0.0f);
    c.hairDyeColor = livebeauty::parseHairDyeColor(dictString(d, @"hairDyeColor", "milk_tea_brown"));
    c.hairShine = dictFloat(d, @"hairShine", 0.0f);
    c.preferHalfFloat = dictBool(d, @"preferHalfFloat", true);
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
    livebeauty::rhi::TextureId _occWrappedTex;
    livebeauty::rhi::TextureId _aiResidualWrappedTex;
    std::array<livebeauty::rhi::TextureId, livebeauty::math::kMaxAr3dLayers> _ar3dWrappedTex;
    CVMetalTextureCacheRef _textureCache;
    CVPixelBufferPoolRef _outputPool;
    size_t _poolW;
    size_t _poolH;
    OSType _poolFormat;
    uint64_t _frameCounter;
    int64_t _lastLightSampleMs;
    float _lightLuma;
    bool _explicitBlendshapes;
    FLTVisionTracker* _visionTracker;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _occWrappedTex = livebeauty::rhi::kInvalidTexture;
        _aiResidualWrappedTex = livebeauty::rhi::kInvalidTexture;
        _ar3dWrappedTex.fill(livebeauty::rhi::kInvalidTexture);
        _outputPool = nullptr;
        _poolW = 0;
        _poolH = 0;
        _poolFormat = 0;
        _frameCounter = 0;
        _lastLightSampleMs = INT64_MIN;
        _lightLuma = 0.30f;
        _explicitBlendshapes = false;
        _device = std::make_unique<livebeauty::rhi::metal::MetalDevice>();
        if (_device->isReady()) {
            _engine = std::make_unique<livebeauty::LiveBeautyEngine>(*_device);
            NSProcessInfo* pi = [NSProcessInfo processInfo];
            const char* gpuName = _device->nativeDevice().name.UTF8String;
            const int cores = static_cast<int>(pi.processorCount);
            const int ramMb = static_cast<int>(pi.physicalMemory / (1024ULL * 1024ULL));
            _engine->setHardwareProfile(gpuName ? std::string(gpuName) : std::string("Apple GPU"), cores, ramMb);
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, _device->nativeDevice(), nullptr, &_textureCache);
            _visionTracker = [[FLTVisionTracker alloc] init];
        }
    }
    return self;
}

- (void)dealloc {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->releaseGpuResources();
    if (_outputPool) {
        CVPixelBufferPoolRelease(_outputPool);
        _outputPool = nullptr;
    }
    if (_textureCache) {
        CVMetalTextureCacheFlush(_textureCache, 0);
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
        std::vector<uint8_t> rgba;
        size_t w = 0, h = 0;
        if (!decodeStraightRgba(image, rgba, w, h)) return NO;
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
    const bool ok = _engine->submitFaceLandmarks(sampleTsMs, mpXy, pointCount, width, height, receivedMs);
    if (ok && !_explicitBlendshapes && pointCount >= 468) {
        _engine->submitFaceBlendshapes(distillBlendshapesFromLandmarks(mpXy, pointCount));
    }
    return ok ? YES : NO;
}

- (void)submitFaceMissAtMs:(int64_t)sampleTsMs {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->submitFaceMiss(sampleTsMs);
}

- (BOOL)submitBodyPose:(const float*)poseXyv99
            pointCount:(int)pointCount
                 width:(int)width
                height:(int)height
            sampleTsMs:(int64_t)sampleTsMs {
    if (!self.isReady || !poseXyv99 || pointCount < 33) return NO;
    livebeauty::math::PoseLandmark landmarks[33];
    for (int i = 0; i < 33; ++i) {
        landmarks[i].x = poseXyv99[i * 3 + 0];
        landmarks[i].y = poseXyv99[i * 3 + 1];
        landmarks[i].visibility = poseXyv99[i * 3 + 2];
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _engine->submitBodyPose(sampleTsMs, landmarks, 33, width, height) ? YES : NO;
}

- (void)submitBodyMissAtMs:(int64_t)sampleTsMs {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->submitBodyMiss(sampleTsMs);
}

- (void)submitFaceBlendshapesWithSmile:(float)smile
                               jawOpen:(float)jawOpen
                                browUp:(float)browUp
                                 blink:(float)blink
                             cheekPuff:(float)cheekPuff
                             tongueOut:(float)tongueOut {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    _explicitBlendshapes = true;
    livebeauty::math::BlendshapeInput bs{smile, jawOpen, browUp, blink, cheekPuff, tongueOut};
    _engine->submitFaceBlendshapes(bs);
}

- (BOOL)submitAuxRgbaSample:(const uint8_t*)rgbaTopDown
                      width:(int)width
                     height:(int)height
            retouchStrength:(float)retouchStrength {
    if (!self.isReady || !rgbaTopDown || width <= 0 || height <= 0) return NO;
    std::lock_guard<std::mutex> lock(_mutex);
    return _engine->submitAuxRgbaSample(rgbaTopDown, width, height, retouchStrength) ? YES : NO;
}

- (BOOL)setVirtualBackgroundImage:(nullable CGImageRef)image
                             mode:(NSString*)mode
                         strength:(float)strength {
    if (!self.isReady) return NO;
    std::lock_guard<std::mutex> lock(_mutex);
    if (_assets.virtualBgTex != livebeauty::rhi::kInvalidTexture) {
        _device->destroyTexture(_assets.virtualBgTex);
        _assets.virtualBgTex = livebeauty::rhi::kInvalidTexture;
        _assets.virtualBgAspect = 0.0f;
    }
    if (image != nullptr) {
        std::vector<uint8_t> rgba;
        size_t w = 0, h = 0;
        if (!decodeStraightRgba(image, rgba, w, h)) return NO;
        _assets.virtualBgTex = _device->createTexture2D(
            static_cast<int>(w), static_cast<int>(h),
            livebeauty::rhi::TextureFormat::kRgba8, rgba.data());
        _assets.virtualBgAspect = static_cast<float>(w) / static_cast<float>(h);
    }
    _engine->setAssets(_assets);
    auto cfg = _engine->config();
    cfg.backgroundMode = livebeauty::parseBackgroundMode(mode ? std::string(mode.UTF8String) : std::string("none"));
    cfg.virtualBgStrength = livebeauty::detail::clampf(strength, 0.0f, 1.0f);
    _engine->setConfig(cfg);
    return YES;
}

- (void)submitVisionPixelBuffer:(CVPixelBufferRef)pixelBuffer
               frameTimestampMs:(int64_t)frameTsMs {
    if (!self.isReady || !pixelBuffer || !_visionTracker) return;
    BOOL wantSeg = NO;
    BOOL wantPose = NO;
    BOOL wantAux = NO;
    float retouchStr = 0.0f;
    int quality = 1;
    const int pbW = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
    const int pbH = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto& cfg = _engine->config();
        if (!cfg.enabled || cfg.compareOriginal) return;
        const bool wantVbg = (_assets.virtualBgTex != livebeauty::rhi::kInvalidTexture &&
                              (cfg.backgroundMode == livebeauty::BackgroundMode::kVirtualImage ||
                               cfg.backgroundMode == livebeauty::BackgroundMode::kVirtualBlur) &&
                              cfg.virtualBgStrength > 1e-4f);
        wantSeg = (cfg.backgroundBlur > 1e-4f || wantVbg) ? YES : NO;
        wantPose = cfg.hasBodyReshape() ? YES : NO;
        wantAux = (_engine->facePresence() > 0.05f) ? YES : NO;
        retouchStr = livebeauty::detail::clampf(
            cfg.smoothing * 0.42f + cfg.blemishRemove * 0.52f + cfg.oilControl * 0.30f, 0.0f, 0.72f);
        quality = (_engine->stats().resolvedTier <= 0 || _engine->loadLevel() >= 1) ? 0 : 1;
    }
    _visionTracker.qualityLevel = quality;

    FLTLiveBeautyMetalEngine* __weak weakSelf = self;
    [_visionTracker submitPixelBuffer:pixelBuffer
                      sampleTimestamp:frameTsMs
                     wantSegmentation:wantSeg
                         wantBodyPose:wantPose
                          wantFaceAux:wantAux
                      onAuxRgbaSample:^(const uint8_t* rgba, int w, int h) {
        [weakSelf submitAuxRgbaSample:rgba width:w height:h retouchStrength:retouchStr];
    }];

    float poseXyv[33 * 3];
    int64_t poseTs = -1;
    if ([_visionTracker pollLatestBodyPose:poseXyv sampleTimestamp:&poseTs]) {
        [self submitBodyPose:poseXyv pointCount:33 width:pbW height:pbH sampleTsMs:poseTs];
    }
}

- (BOOL)applyEffectBundleJson:(NSString*)manifestJson {
    if (!self.isReady || !manifestJson) return NO;
    const char* utf8 = [manifestJson UTF8String];
    if (!utf8) return NO;
    std::lock_guard<std::mutex> lock(_mutex);
    return _engine->applyEffectBundleJson(std::string(utf8), nullptr, nullptr) ? YES : NO;
}

- (void)resetTracking {
    std::lock_guard<std::mutex> lock(_mutex);
    _explicitBlendshapes = false;
    if (_engine) _engine->resetTracking();
    if (_textureCache) CVMetalTextureCacheFlush(_textureCache, 0);
    [_visionTracker reset];
}

- (void)setAmbientR:(float)r g:(float)g b:(float)b {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->setAmbientRgb(r, g, b);
}

- (void)setOcclusionMaskTexture:(nullable id<MTLTexture>)occlusionMask {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    if (_occWrappedTex != livebeauty::rhi::kInvalidTexture) {
        _device->destroyTexture(_occWrappedTex);
        _occWrappedTex = livebeauty::rhi::kInvalidTexture;
    }
    _occWrappedTex = occlusionMask ? _device->wrapMetalTexture(occlusionMask) : livebeauty::rhi::kInvalidTexture;
    _engine->setOcclusionMask(_occWrappedTex);
}

- (void)setAiResidualTexture:(nullable id<MTLTexture>)residualTex
                     roiLeft:(float)roiLeft
                      roiTop:(float)roiTop
                    roiWidth:(float)roiWidth
                   roiHeight:(float)roiHeight
                    strength:(float)strength {
    if (!self.isReady) return;
    std::lock_guard<std::mutex> lock(_mutex);
    if (_aiResidualWrappedTex != livebeauty::rhi::kInvalidTexture) {
        _device->destroyTexture(_aiResidualWrappedTex);
        _aiResidualWrappedTex = livebeauty::rhi::kInvalidTexture;
    }
    _aiResidualWrappedTex = residualTex ? _device->wrapMetalTexture(residualTex) : livebeauty::rhi::kInvalidTexture;
    _engine->setAiResidual(_aiResidualWrappedTex, {roiLeft, roiTop, roiWidth, roiHeight}, strength);
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
    [self setAr3dAccessoryAtSlot:slot
                         enabled:enabled
                          anchor:anchor
                         texture:texture
                           scale:scale
                         offsetX:offsetX
                         offsetY:offsetY
                         offsetZ:offsetZ
                          yawRad:yawRad
                        pitchRad:pitchRad
                         rollRad:rollRad
                         opacity:opacity
                   textureAspect:textureAspect
                 springStiffness:springStiffness
                   springDamping:springDamping
                   headOcclusion:headOcclusion
                      meshPreset:0
                     triggerRule:0];
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
                 headOcclusion:(BOOL)headOcclusion
                    meshPreset:(int)meshPreset
                   triggerRule:(int)triggerRule {
    if (!self.isReady || slot < 0 || slot >= livebeauty::math::kMaxAr3dLayers) return;
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ar3dWrappedTex[static_cast<size_t>(slot)] != livebeauty::rhi::kInvalidTexture) {
        _device->destroyTexture(_ar3dWrappedTex[static_cast<size_t>(slot)]);
        _ar3dWrappedTex[static_cast<size_t>(slot)] = livebeauty::rhi::kInvalidTexture;
    }
    livebeauty::math::Ar3dAccessoryConfig cfg{};
    cfg.enabled = (enabled == YES);
    cfg.anchor = static_cast<livebeauty::math::Ar3dAnchor>(std::clamp(anchor, 0, 5));
    cfg.meshPreset = static_cast<livebeauty::math::Ar3dMeshPreset>(std::clamp(meshPreset, 0, 2));
    if (texture) {
        _ar3dWrappedTex[static_cast<size_t>(slot)] = _device->wrapMetalTexture(texture);
        cfg.texture = _ar3dWrappedTex[static_cast<size_t>(slot)];
    }
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
    const auto trig = static_cast<livebeauty::bundle::BundleTriggerRule>(std::clamp(triggerRule, 0, 6));
    _engine->setAr3dAccessory(slot, cfg, trig);
}

- (void)clearAr3dAccessories {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_engine) _engine->clearAr3dAccessories();
    for (auto& t : _ar3dWrappedTex) {
        if (t != livebeauty::rhi::kInvalidTexture) {
            _device->destroyTexture(t);
            t = livebeauty::rhi::kInvalidTexture;
        }
    }
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
    CVPixelBufferRef maskPb = [_visionTracker copyLatestPersonMaskBuffer:nullptr];
    CVMetalTextureRef maskCvTex = nullptr;
    id<MTLTexture> autoMaskTex = nil;
    if (maskPb && _textureCache) {
        const size_t mw = CVPixelBufferGetWidth(maskPb);
        const size_t mh = CVPixelBufferGetHeight(maskPb);
        if (CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, _textureCache, maskPb, nullptr,
                MTLPixelFormatR8Unorm, mw, mh, 0, &maskCvTex) == kCVReturnSuccess && maskCvTex) {
            autoMaskTex = CVMetalTextureGetTexture(maskCvTex);
        }
    }
    float luma = 0.30f;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        luma = _lightLuma;
    }
    id<MTLTexture> out = [self processTexture:inputTexture
                             frameTimestampMs:frameTsMs
                                        nowMs:nowMs
                                   personMask:autoMaskTex
                                     hairMask:nil
                              hairMaskStampMs:livebeauty::kTimeInvalid
                                    lightLuma:luma
                                 bitrateRatio:1.0f];
    if (maskCvTex) CFRelease(maskCvTex);
    if (maskPb) CFRelease(maskPb);
    return out;
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
    const auto inId = _device->wrapTransientMetalTexture(inputTexture);
    const auto maskId = personMask ? _device->wrapTransientMetalTexture(personMask) : livebeauty::rhi::kInvalidTexture;
    const auto hairId = hairMask ? _device->wrapTransientMetalTexture(hairMask) : livebeauty::rhi::kInvalidTexture;
    const auto outId = _engine->processFrame(
        inId,
        static_cast<int>(inputTexture.width),
        static_cast<int>(inputTexture.height),
        frameTsMs, nowMs, maskId, lightLuma, bitrateRatio, hairId, hairMaskStampMs);
    id<MTLTexture> outMtl = (outId == inId) ? inputTexture : _device->unwrapMetalTexture(outId);
    _device->flush();
    if (_textureCache && (++_frameCounter % 120ULL) == 0ULL) {
        CVMetalTextureCacheFlush(_textureCache, 0);
    }
    return outMtl ? outMtl : inputTexture;
}

- (void)sampleLightAndAmbientFromPixelBuffer:(CVPixelBufferRef)pb nowMs:(int64_t)nowMs {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_lastLightSampleMs != INT64_MIN && nowMs - _lastLightSampleMs < 1000) return;
        _lastLightSampleMs = nowMs;
    }
    const OSType fmt = CVPixelBufferGetPixelFormatType(pb);
    if (fmt != kCVPixelFormatType_32BGRA && fmt != kCVPixelFormatType_32RGBA) return;
    if (CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return;

    const size_t w = CVPixelBufferGetWidth(pb);
    const size_t h = CVPixelBufferGetHeight(pb);
    const size_t stride = CVPixelBufferGetBytesPerRow(pb);
    const uint8_t* base = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const bool isBgra = (fmt == kCVPixelFormatType_32BGRA);
    double sumR = 0.0, sumG = 0.0, sumB = 0.0;
    size_t count = 0;
    if (base && w > 0 && h > 0) {
        const size_t stepX = std::max<size_t>(w / 16, 1);
        const size_t stepY = std::max<size_t>(h / 16, 1);
        for (size_t y = stepY / 2; y < h; y += stepY) {
            const uint8_t* row = base + y * stride;
            for (size_t x = stepX / 2; x < w; x += stepX) {
                const uint8_t* p = row + x * 4;
                sumR += isBgra ? p[2] : p[0];
                sumG += p[1];
                sumB += isBgra ? p[0] : p[2];
                ++count;
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    if (count == 0) return;

    const float meanR = static_cast<float>(sumR / static_cast<double>(count));
    const float meanG = static_cast<float>(sumG / static_cast<double>(count));
    const float meanB = static_cast<float>(sumB / static_cast<double>(count));
    const float luma = livebeauty::detail::clampf(
        (0.2126f * meanR + 0.7152f * meanG + 0.0722f * meanB) / 255.0f, 0.0f, 1.0f);
    const float avg = std::max((meanR + meanG + meanB) / 3.0f, 16.0f);
    const float ar = livebeauty::detail::clampf(meanR / avg, 0.75f, 1.25f);
    const float ag = livebeauty::detail::clampf(meanG / avg, 0.75f, 1.25f);
    const float ab = livebeauty::detail::clampf(meanB / avg, 0.75f, 1.25f);

    std::lock_guard<std::mutex> lock(_mutex);
    _lightLuma += 0.35f * (luma - _lightLuma);
    if (_engine) _engine->setAmbientRgb(ar, ag, ab);
}

- (nullable CVPixelBufferRef)dequeueOutputPixelBufferWithWidth:(size_t)w
                                                        height:(size_t)h
                                                        format:(OSType)fmt {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_outputPool || _poolW != w || _poolH != h || _poolFormat != fmt) {
        if (_outputPool) {
            CVPixelBufferPoolRelease(_outputPool);
            _outputPool = nullptr;
        }
        NSDictionary* poolAttrs = @{
            (NSString*)kCVPixelBufferPoolMinimumBufferCountKey : @3,
        };
        NSDictionary* pbAttrs = @{
            (NSString*)kCVPixelBufferPixelFormatTypeKey : @(fmt),
            (NSString*)kCVPixelBufferWidthKey : @(w),
            (NSString*)kCVPixelBufferHeightKey : @(h),
            (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
            (NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
        };
        if (CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                (__bridge CFDictionaryRef)poolAttrs,
                (__bridge CFDictionaryRef)pbAttrs,
                &_outputPool) != kCVReturnSuccess || !_outputPool) {
            _outputPool = nullptr;
            return nullptr;
        }
        _poolW = w;
        _poolH = h;
        _poolFormat = fmt;
    }
    CVPixelBufferRef outPb = nullptr;
    NSDictionary* aux = @{(NSString*)kCVPixelBufferPoolAllocationThresholdKey : @8};
    if (CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault, _outputPool, (__bridge CFDictionaryRef)aux, &outPb) != kCVReturnSuccess || !outPb) {
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _outputPool, &outPb);
    }
    return outPb;
}

- (nullable CVPixelBufferRef)copyProcessedPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                     frameTimestampMs:(int64_t)frameTsMs
                                                nowMs:(int64_t)nowMs {
    if (!self.isReady || !pixelBuffer || !_textureCache) return nullptr;
    const size_t w = CVPixelBufferGetWidth(pixelBuffer);
    const size_t h = CVPixelBufferGetHeight(pixelBuffer);
    if (w == 0 || h == 0) return nullptr;

    const OSType inFmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    const OSType ioFmt = (inFmt == kCVPixelFormatType_32RGBA)
        ? kCVPixelFormatType_32RGBA
        : kCVPixelFormatType_32BGRA;
    const MTLPixelFormat mtlFmt = (ioFmt == kCVPixelFormatType_32RGBA)
        ? MTLPixelFormatRGBA8Unorm
        : MTLPixelFormatBGRA8Unorm;

    CVMetalTextureRef inCvTex = nullptr;
    if (CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, _textureCache, pixelBuffer, nullptr,
            mtlFmt, w, h, 0, &inCvTex) != kCVReturnSuccess || !inCvTex) {
        return nullptr;
    }

    [self sampleLightAndAmbientFromPixelBuffer:pixelBuffer nowMs:nowMs];
    [self submitVisionPixelBuffer:pixelBuffer frameTimestampMs:frameTsMs];

    CVPixelBufferRef outPb = [self dequeueOutputPixelBufferWithWidth:w height:h format:ioFmt];
    if (!outPb) {
        CFRelease(inCvTex);
        return nullptr;
    }

    NSDictionary* dstTexAttrs = @{
        (NSString*)kCVMetalTextureUsage : @(MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead),
    };
    CVMetalTextureRef outCvTex = nullptr;
    if (CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, _textureCache, outPb, (__bridge CFDictionaryRef)dstTexAttrs,
            mtlFmt, w, h, 0, &outCvTex) != kCVReturnSuccess || !outCvTex) {
        CFRelease(inCvTex);
        CVPixelBufferRelease(outPb);
        return nullptr;
    }

    CVPixelBufferRef maskPb = [_visionTracker copyLatestPersonMaskBuffer:nullptr];
    CVMetalTextureRef maskCvTex = nullptr;
    id<MTLTexture> autoMaskTex = nil;
    if (maskPb) {
        const size_t mw = CVPixelBufferGetWidth(maskPb);
        const size_t mh = CVPixelBufferGetHeight(maskPb);
        if (CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, _textureCache, maskPb, nullptr,
                MTLPixelFormatR8Unorm, mw, mh, 0, &maskCvTex) == kCVReturnSuccess && maskCvTex) {
            autoMaskTex = CVMetalTextureGetTexture(maskCvTex);
        }
    }

    id<MTLTexture> inMtl = CVMetalTextureGetTexture(inCvTex);
    id<MTLTexture> dstMtl = CVMetalTextureGetTexture(outCvTex);
    bool rendered = false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto inId = _device->wrapTransientMetalTexture(inMtl);
        const auto maskId = autoMaskTex ? _device->wrapTransientMetalTexture(autoMaskTex) : livebeauty::rhi::kInvalidTexture;
        const auto outId = _engine->processFrame(
            inId, static_cast<int>(w), static_cast<int>(h),
            frameTsMs, nowMs, maskId, _lightLuma, 1.0f);
        rendered = _device->renderToExternalTexture(outId, dstMtl);
        _device->flush();
        if ((++_frameCounter % 120ULL) == 0ULL) {
            CVMetalTextureCacheFlush(_textureCache, 0);
        }
    }

    if (maskCvTex) CFRelease(maskCvTex);
    if (maskPb) CFRelease(maskPb);
    CFRelease(outCvTex);
    CFRelease(inCvTex);
    if (!rendered) {
        CVPixelBufferRelease(outPb);
        return nullptr;
    }
    return outPb;
}

@end
