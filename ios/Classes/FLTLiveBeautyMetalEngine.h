#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Full-featured iOS/macOS Metal beauty engine backed by the unified C++17 core (`livebeauty::LiveBeautyEngine`).
 * Provides full v3.0 parity with Android: 15-axis C∞ diffeomorphic mesh warp, 3D normal studio relighting,
 * T-zone oil control, seamless neck-face continuity, ambient WB & bangs/occlusion-gated UV makeup,
 * temporal portrait matting, isolated blemish healer + pluggable AI residual, and 3D head-occluded AR accessories.
 */
@interface FLTLiveBeautyMetalEngine : NSObject

@property(nonatomic, readonly) BOOL isReady;

- (instancetype)init;

/** Thread-safe configuration update using the unified 50+ key dictionary contract. */
- (void)setConfiguration:(NSDictionary<NSString *, id> *)config;

/** Load RGBA8 atlas or 512x512 3D LUT image from CGImageRef into Metal GPU texture. */
- (BOOL)setAssetImage:(nullable CGImageRef)image forSlot:(NSString *)slot;

/** Update ambient illuminant RGB white-balance ratios [0.65..1.35] for makeup color adaptation. */
- (void)setAmbientR:(float)r g:(float)g b:(float)b;

/** Bind optional R8 face-parsing / hand-occlusion mask texture (255=visible skin, 0=occluded). */
- (void)setOcclusionMaskTexture:(nullable id<MTLTexture>)occlusionMask;

/** Bind optional face-ROI neural retouching residual delta texture (RGBA8, 0.5=zero delta, A=confidence). */
- (void)setAiResidualTexture:(nullable id<MTLTexture>)residualTex
                     roiLeft:(float)roiLeft
                      roiTop:(float)roiTop
                    roiWidth:(float)roiWidth
                   roiHeight:(float)roiHeight
                    strength:(float)strength;

/** Configure a 3D head-occluded AR accessory slot (0..3) with optional spring pendulum physics. */
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
                 headOcclusion:(BOOL)headOcclusion;

- (void)clearAr3dAccessories;

/** Submit 468/478-pt MediaPipe normalized face landmarks (y-down, interleaved x0,y0,x1,y1,...). */
- (BOOL)submitFaceLandmarks:(const float *)mpXy
                 pointCount:(int)pointCount
                      width:(int)width
                     height:(int)height
                 sampleTsMs:(int64_t)sampleTsMs
                 receivedMs:(int64_t)receivedMs;

- (void)submitFaceMissAtMs:(int64_t)sampleTsMs;

/** v3.2: Submit 33-point top-down body pose (`33 * 3` floats: x, y, visibility). */
- (BOOL)submitBodyPose:(const float *)poseXyv99
            pointCount:(int)pointCount
                 width:(int)width
                height:(int)height
            sampleTsMs:(int64_t)sampleTsMs;
- (void)submitBodyMissAtMs:(int64_t)sampleTsMs;

/** v3.2: Submit distilled Face Blendshapes for reactive 3D AR / sticker animation. */
- (void)submitFaceBlendshapesWithSmile:(float)smile
                               jawOpen:(float)jawOpen
                                browUp:(float)browUp
                                 blink:(float)blink
                             cheekPuff:(float)cheekPuff
                             tongueOut:(float)tongueOut;

/** v3.2: Submit top-down RGBA sample for built-in C++17 + ARM NEON occlusion & Face-ROI skin residual. */
- (BOOL)submitAuxRgbaSample:(const uint8_t *)rgbaTopDown
                      width:(int)width
                     height:(int)height
            retouchStrength:(float)retouchStrength;

/** v3.2: Configure virtual background CGImage (`mode`: @"none" / @"blur" / @"virtual_image" / @"virtual_blur"). */
- (BOOL)setVirtualBackgroundImage:(nullable CGImageRef)image
                             mode:(NSString *)mode
                         strength:(float)strength;

/**
 * v3.2: Drive built-in asynchronous Apple Vision ANE tracker (`VNGeneratePersonSegmentationRequest` +
 * `VNDetectHumanBodyPoseRequest` + C++17 NEON face aux) directly from a camera `CVPixelBufferRef`.
 */
- (void)submitVisionPixelBuffer:(CVPixelBufferRef)pixelBuffer
               frameTimestampMs:(int64_t)frameTsMs;

/** v3.5: Parse and apply a declarative `.lbfx` Effect Bundle JSON manifest. */
- (BOOL)applyEffectBundleJson:(NSString *)manifestJson;

- (void)resetTracking;

/** Query live C++17 performance telemetry & resolved hardware tier. */
- (NSDictionary<NSString *, NSNumber *> *)statsDictionary;

/** Process an RGBA8/BGRA8 Metal texture directly (zero-copy GPU-to-GPU). */
- (nullable id<MTLTexture>)processTexture:(id<MTLTexture>)inputTexture
                         frameTimestampMs:(int64_t)frameTsMs
                                    nowMs:(int64_t)nowMs;

/** Extended Metal processing with optional person/hair masks, light luma & bitrate ratio. */
- (nullable id<MTLTexture>)processTexture:(id<MTLTexture>)inputTexture
                         frameTimestampMs:(int64_t)frameTsMs
                                    nowMs:(int64_t)nowMs
                               personMask:(nullable id<MTLTexture>)personMask
                                 hairMask:(nullable id<MTLTexture>)hairMask
                          hairMaskStampMs:(int64_t)hairMaskStampMs
                                lightLuma:(float)lightLuma
                             bitrateRatio:(float)bitrateRatio;

/** Process a BGRA/RGBA CVPixelBuffer via CVMetalTextureCache zero-copy bridge. Caller owns returned buffer. */
- (nullable CVPixelBufferRef)copyProcessedPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                     frameTimestampMs:(int64_t)frameTsMs
                                                nowMs:(int64_t)nowMs CF_RETURNS_RETAINED;

@end

NS_ASSUME_NONNULL_END
