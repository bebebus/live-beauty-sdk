#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <CoreGraphics/CoreGraphics.h>
#import "../Classes/FLTFaceGeometry.h"
#import "../Classes/FLTFaceGeometrySmoother.h"
#import "../Classes/FLTVideoEffects.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // 1. Verify configuration validator
        NSDictionary *valid = @{
            @"enabled": @YES,
            @"smoothing": @0.5,
            @"brightness": @0.15,
            @"sharpness": @0.2,
            @"saturation": @1.0,
            @"whiten": @0.3,
            @"rosy": @0.2,
            @"faceSlim": @0.4,
            @"eyeEnlarge": @0.3,
            @"skinOnly": @YES,
            @"compareOriginal": @NO,
            @"filter": @"warm",
            @"filterIntensity": @0.8
        };
        if (![FLTVideoEffects validConfiguration:valid]) {
            fprintf(stderr, "FAIL: validConfiguration rejected valid dictionary\n");
            return 1;
        }
        if ([FLTVideoEffects validConfiguration:@{@"smoothing": @1.5}]) {
            fprintf(stderr, "FAIL: validConfiguration accepted out-of-range smoothing\n");
            return 1;
        }

        // 2. Verify CoreImage kernel compilation & non-nil output image on synthetic 128x128 frame
        CIColor *skinColor = [CIColor colorWithRed:0.78 green:0.60 blue:0.52 alpha:1.0];
        CIImage *src = [[CIImage imageWithColor:skinColor] imageByCroppingToRect:CGRectMake(0, 0, 128, 128)];
        CIImage *outImg = [FLTVideoEffects beautyImage:src configuration:valid geometry:NULL presence:0.0f];
        if (!outImg || !CGRectEqualToRect(outImg.extent, CGRectMake(0, 0, 128, 128))) {
            fprintf(stderr, "FAIL: beautyImage returned nil or mismatched extent\n");
            return 1;
        }
        CIContext *ctx = [CIContext contextWithOptions:@{kCIContextUseSoftwareRenderer: @NO}];
        uint8_t px[4] = {0};
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        [ctx render:outImg toBitmap:px rowBytes:4 bounds:CGRectMake(64, 64, 1, 1) format:kCIFormatRGBA8 colorSpace:cs];
        CGColorSpaceRelease(cs);
        if (px[0] == 0 || px[3] != 255) {
            fprintf(stderr, "FAIL: rendered pixel invalid RGBA=(%d,%d,%d,%d)\n", px[0], px[1], px[2], px[3]);
            return 1;
        }

        // 3. Verify FLTFaceGeometrySmoother lifecycle & presence ramp
        FLTFaceGeometrySmoother *smoother = [[FLTFaceGeometrySmoother alloc] initWithAspect:0.5625f];
        float state[FLTFaceGeometryStateSize] = {0};
        BOOL hasGeom = [smoother updateTarget:NULL sampleTimestamp:0.0 now:100.0 state:state];
        if (hasGeom || smoother.presence != 0.0f) {
            fprintf(stderr, "FAIL: smoother without face should return NO and 0 presence\n");
            return 1;
        }
        float target[FLTFaceGeometryStateSize] = {
            0.50f, 0.45f, 0.20f, 0.26f, // face center + radii
            0.40f, 0.42f, 0.58f, 0.42f, // left/right eyes
            0.05f, 0.34f, 0.66f, 0.50f, // eye radius + cheek left/right + y
            0.09f                       // cheek radius
        };
        BOOL active1 = [smoother updateTarget:target sampleTimestamp:100.0 now:100.0 state:state];
        BOOL active2 = [smoother updateTarget:target sampleTimestamp:133.0 now:175.0 state:state];
        if (active1 || !active2 || fabsf(smoother.presence - 0.5f) > 1e-3f) {
            fprintf(stderr, "FAIL: smoother failed 150ms fade-in ramp (active1=%d, active2=%d, presence=%f)\n",
                    active1, active2, smoother.presence);
            return 1;
        }
        CIImage *warpedImg = [FLTVideoEffects beautyImage:src configuration:valid geometry:state presence:smoother.presence];
        if (!warpedImg || !CGRectEqualToRect(warpedImg.extent, CGRectMake(0, 0, 128, 128))) {
            fprintf(stderr, "FAIL: beautyImage with active face geometry returned nil or mismatched extent\n");
            return 1;
        }

        printf("PASS  iOS/macOS CoreImage 美颜内核编译、像素渲染与几何平滑器单测通过 (pixel RGBA=%d,%d,%d,%d, presence=%.2f)\n",
               px[0], px[1], px[2], px[3], smoother.presence);
        return 0;
    }
}
