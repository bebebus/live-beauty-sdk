#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Asynchronous Apple Vision Neural Engine (ANE) tracker for iOS 15+ / macOS 12+:
 * - `VNGeneratePersonSegmentationRequest`: 0 MB bundle overhead, hardware-accelerated soft-alpha portrait matte (`OneComponent8` CVPixelBuffer);
 * - `VNDetectHumanBodyPoseRequest`: 19-joint body skeleton mapped into MediaPipe 33-landmark top-down layout (`x, y, confidence`);
 * - Optional lightweight 128x128 top-down RGBA downsample for C++17 + ARM NEON foreground occlusion & Face-ROI skin residual.
 */
@interface FLTVisionTracker : NSObject

/** Quality tier: 0 = Fast (Lite), 1 = Balanced (Standard/Ultra). */
@property(nonatomic, assign) int qualityLevel;

/** Submit a camera CVPixelBuffer asynchronously; drops immediately if a request is already in flight. */
- (void)submitPixelBuffer:(CVPixelBufferRef)pixelBuffer
          sampleTimestamp:(int64_t)sampleTsMs
         wantSegmentation:(BOOL)wantSeg
             wantBodyPose:(BOOL)wantPose
              wantFaceAux:(BOOL)wantAux
          onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock;

/** Copy latest OneComponent8 person segmentation CVPixelBuffer (caller must CFRelease if non-null). */
- (nullable CVPixelBufferRef)copyLatestPersonMaskBuffer:(nullable int64_t *)outSampleTsMs CF_RETURNS_RETAINED;

/** Poll latest 33-point MediaPipe-compatible top-down pose array (`33 * 3` floats: x, y, visibility) if updated since last poll. */
- (BOOL)pollLatestBodyPose:(float *)outXyv99 sampleTimestamp:(int64_t *)outSampleTsMs;

- (void)reset;

@end

NS_ASSUME_NONNULL_END
