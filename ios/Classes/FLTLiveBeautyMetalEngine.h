#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Full-featured iOS/macOS Metal beauty engine backed by the unified C++17 core (`livebeauty::LiveBeautyEngine`).
 * Provides full parity with Android v4.5.2+: 15-axis C∞ diffeomorphic mesh warp (including cranialTop),
 * dual-band creamy guided smoothing, 468-pt anatomical sub-mesh retouch, 6-layer physically-based UV makeup,
 * 64^3 3D LUT, 5-axis body reshape, and portrait bokeh.
 */
@interface FLTLiveBeautyMetalEngine : NSObject

@property(nonatomic, readonly) BOOL isReady;

- (instancetype)init;

/** Thread-safe configuration update using the unified 40+ key dictionary contract. */
- (void)setConfiguration:(NSDictionary<NSString *, id> *)config;

/** Load RGBA8 atlas or 512x512 3D LUT image from CGImageRef into Metal GPU texture. */
- (BOOL)setAssetImage:(nullable CGImageRef)image forSlot:(NSString *)slot;

/** Submit 468/478-pt MediaPipe normalized face landmarks (y-down, interleaved x0,y0,x1,y1,...). */
- (BOOL)submitFaceLandmarks:(const float *)mpXy
                 pointCount:(int)pointCount
                      width:(int)width
                     height:(int)height
                 sampleTsMs:(int64_t)sampleTsMs
                 receivedMs:(int64_t)receivedMs;

- (void)submitFaceMissAtMs:(int64_t)sampleTsMs;
- (void)resetTracking;

/** Process an RGBA8/BGRA8 Metal texture directly (zero-copy GPU-to-GPU). */
- (nullable id<MTLTexture>)processTexture:(id<MTLTexture>)inputTexture
                         frameTimestampMs:(int64_t)frameTsMs
                                    nowMs:(int64_t)nowMs;

/** Process a BGRA/RGBA CVPixelBuffer via CVMetalTextureCache zero-copy bridge. Caller owns returned buffer. */
- (nullable CVPixelBufferRef)copyProcessedPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                     frameTimestampMs:(int64_t)frameTsMs
                                                nowMs:(int64_t)nowMs CF_RETURNS_RETAINED;

@end

NS_ASSUME_NONNULL_END
