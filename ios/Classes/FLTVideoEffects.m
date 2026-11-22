#import "FLTVideoEffects.h"
#import "FLTFaceGeometrySmoother.h"
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>
#import <math.h>

// CIKL 书写约束（规格 4.3）：不用函数外全局常量，不用 mat2/3/4；uniform 只用 float/vec2/vec3/vec4。
// 违反时 kernelWithString 仍可能返回非 nil，渲染却静默不写目标缓冲，所以首次使用前必须自检。
// 公式与 reference/beauty_ref.py 逐式一致；滤镜编号与公式以 Android 现有着色器为准。
NSString *const FLTVideoEffectsDemoSiteURL = @"https://dh.video";
NSString *const FLTVideoEffectsKernelCommon =
    @"float skinWeight(vec3 c) {"
     " float cb = 0.5 + dot(c, vec3(-0.168736, -0.331264, 0.5));"
     " float cr = 0.5 + dot(c, vec3(0.5, -0.418688, -0.081312));"
     " return smoothstep(0.26, 0.34, cb) * (1.0 - smoothstep(0.48, 0.56, cb)) *"
     "  smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.72, 0.80, cr)); }\n"
     "vec3 lift(vec3 x, float a) { return x * (1.0 + a) / (vec3(1.0) + a * x); }\n"
     "float ellipseE(vec2 p, vec4 c, vec2 axis, float rmin) {"
     " vec2 d = p - c.xy;"
     " float a = clamp(dot(d, axis) / max(c.z, rmin), -4.0, 4.0);"
     " float b = clamp(dot(d, vec2(-axis.y, axis.x)) / max(c.w, rmin), -4.0, 4.0);"
     " return a * a + b * b; }\n";

// Adapted from GPUPixel v1.3.1 FaceReshapeFilter, Copyright PixPark, Apache-2.0.
// MediaPipe indices, four bounded cheek warps, pixel-space metric and CI ROI are local changes.
NSString *const FLTVideoEffectsKernelWarp =
    @"vec2 cheek(vec2 uv, vec4 c, float amount) {"
     " float radius = max(distance(c.xy, c.zw), 0.001);"
     " float weight = clamp(1.0 - distance(uv, c.xy) / radius, 0.0, 1.0);"
     " return uv - (c.zw - c.xy) * amount * weight; }\n"
     "vec2 eye(vec2 uv, vec3 e, float amount) {"
     " float ratio = distance(uv, e.xy) / max(e.z, 0.001);"
     " float weight = clamp(1.0 - (1.0 - ratio * ratio) * amount, 0.0, 1.0);"
     " return e.xy + (uv - e.xy) * weight; }\n";

// 遮罩在采样坐标处求值：parts = (oval, eye0, eye1, mouth)，返回 (face, excl)。
NSString *const FLTVideoEffectsKernelMasks =
    @"vec4 maskParts(vec2 p, vec4 oval, vec2 axis, vec4 mouth, vec4 eyeMask0, vec4 eyeMask1, float rmin) {"
     " return vec4(1.0 - smoothstep(0.50, 1.0, ellipseE(p, oval, axis, rmin)),"
     "  1.0 - smoothstep(0.55, 1.0, ellipseE(p, eyeMask0, axis, rmin)),"
     "  1.0 - smoothstep(0.55, 1.0, ellipseE(p, eyeMask1, axis, rmin)),"
     "  1.0 - smoothstep(0.65, 1.0, ellipseE(p, mouth, axis, rmin))); }\n"
     "vec2 faceMasks(vec4 parts, float presence) {"
     " float keep = (1.0 - parts.y) * (1.0 - parts.z) * (1.0 - parts.w);"
     " return presence * parts.x * vec2(keep, 1.0 - keep); }\n";

// p0 = (smoothing, skinOnly, sharpness, saturation)，p1 = (whiten, rosy, brightness, filterIntensity)，
// style：none 0、warm 1、cool 2、monochrome 3、sepia 4、rose 5。调用方保证 processing 为真。
NSString *const FLTVideoEffectsKernelComposite =
    @"vec3 beautyColor(vec3 I, vec3 mean, float v, vec3 avg4, float face, float excl, vec4 p0, vec4 p1, float style) {"
     " vec3 lum = vec3(0.2126, 0.7152, 0.0722);"
     " float gate = skinWeight(I);"
     " float mSkin = face * mix(0.85, 1.0, gate) + (1.0 - face) * gate * (1.0 - 0.55 * excl);"
     " vec3 color = I;"
     " float k = 0.0;"
     " if (p0.x > 0.0) {"
     "  float r = v / 0.15;"
     "  k = p0.x * 0.9 * (p0.y > 0.5 ? mSkin : 1.0) / (1.0 + r * r);"
     "  color = mix(I, mean, k);"
     " }\n"
     " if (p0.z > 0.0) {"
     "  float detail = dot(I - avg4, lum);"
     "  detail *= smoothstep(0.008, 0.03, abs(detail));"
     "  color += vec3(clamp(detail, -0.1, 0.1) * p0.z * 0.8 * (1.0 - 0.6 * mSkin) * (1.0 - k));"
     " }\n"
     " color = clamp(color, 0.0, 1.0);"
     " if (p1.x > 0.0) {"
     "  float y = clamp(dot(color, lum), 0.0, 1.0);"
     "  vec3 lighter = min(color * (vec3(1.0) + 0.5 * p1.x * (vec3(1.0) - mix(color, vec3(y), 0.5))), vec3(1.0));"
     "  color = mix(color, lighter, mix(0.15, 1.0, mSkin));"
     " }\n"
     " if (p1.y > 0.0) { float d = 0.03 * p1.y * mSkin; color += vec3(1.402 * d, -0.714 * d, 0.0); }"
     " if (p1.z > 0.0) { color = lift(color, 0.5 * p1.z); }\n"
     " color = mix(vec3(dot(color, lum)), color, p0.w);"
     " vec3 toned = color;"
     " if (style > 4.5) { toned = color + vec3(0.045, -0.005, 0.025); }"
     " else if (style > 3.5) { toned = vec3(dot(color, vec3(0.393, 0.769, 0.189)),"
     "  dot(color, vec3(0.349, 0.686, 0.168)), dot(color, vec3(0.272, 0.534, 0.131))); }"
     " else if (style > 2.5) { toned = vec3(dot(color, vec3(0.299, 0.587, 0.114))); }"
     " else if (style > 1.5) { toned = color + vec3(-0.025, 0.0, 0.045); }"
     " else if (style > 0.5) { toned = color + vec3(0.045, 0.008, -0.025); }"
     " color = mix(color, clamp(toned, 0.0, 1.0), p1.w);"
     " return clamp(color, 0.0, 1.0); }\n";

// T0：每个 1/4 像素在 ±1 原始像素处取 4 个双线性抽头，即 4×4 盒式均值；不得用两次 0.5 倍仿射代替
// （iOS 默认 HighQualityDownsample=NO 时会退化成单个 2×2 抽头）。
static NSString *const FLTBoxKernelSource =
    @"kernel vec4 box4(sampler s) {"
     " vec2 d = destCoord() * 4.0;"
     " return 0.25 * (sample(s, samplerTransform(s, d + vec2(-1.0, -1.0))) + sample(s, samplerTransform(s, d + vec2(1.0, -1.0)))"
     " + sample(s, samplerTransform(s, d + vec2(-1.0, 1.0))) + sample(s, samplerTransform(s, d + vec2(1.0, 1.0)))); }";

static NSString *const FLTVarianceKernelSource =
    @"kernel vec4 variance(__sample t0, __sample mean) {"
     " float d = dot(t0.rgb - mean.rgb, vec3(0.2126, 0.7152, 0.0722)) * 7.07;"
     " float v = min(d * d, 1.0);"
     " return vec4(v, v, v, 1.0); }";

static NSString *const FLTWarpKernelSource =
    @"kernel vec2 face(vec4 c0, vec4 c1, vec4 c2, vec4 c3, vec3 e0, vec3 e1, vec2 size, float slim, float big) {"
     " vec2 uv = destCoord();"
     " if (slim > 0.0) { uv = cheek(uv, c0, slim); uv = cheek(uv, c1, slim); uv = cheek(uv, c2, slim); uv = cheek(uv, c3, slim); }"
     " if (big > 0.0) { uv = eye(uv, e0, big); uv = eye(uv, e1, big); }"
     " return clamp(uv, vec2(0.5), size - vec2(0.5)); }";

// 颜色内核由 destCoord() 用同一段 cheek()/eye() 重算采样坐标，遮罩与 Android 一样在采样坐标处求值；
// mean/var/avg 来自变形后的图像，与 Android 先采样后计算等价到重采样误差。
static NSString *const FLTBeautyKernelSource =
    @"kernel vec4 beauty(__sample src, __sample meanUp, __sample varUp, __sample avg,"
     "  vec4 oval, vec2 axis, vec4 mouth, vec4 eyeMask0, vec4 eyeMask1, vec3 eye0, vec3 eye1,"
     "  vec4 c0, vec4 c1, vec4 c2, vec4 c3, float presence, vec2 size, vec2 warp, vec4 p0, vec4 p1, float style) {"
     " vec2 masks = vec2(0.0);"
     " if (presence > 0.0) {"
     "  vec2 uv = destCoord();"
     "  if (warp.x > 0.0) { uv = cheek(uv, c0, warp.x); uv = cheek(uv, c1, warp.x); uv = cheek(uv, c2, warp.x); uv = cheek(uv, c3, warp.x); }"
     "  if (warp.y > 0.0) { uv = eye(uv, eye0, warp.y); uv = eye(uv, eye1, warp.y); }"
     "  uv = clamp(uv, vec2(0.5), size - vec2(0.5));"
     "  masks = faceMasks(maskParts(uv, oval, axis, mouth, eyeMask0, eyeMask1, 0.0001 * size.x), presence);"
     " }\n"
     " return vec4(beautyColor(src.rgb, meanUp.rgb, varUp.r, avg.rgb, masks.x, masks.y, p0, p1, style), src.a); }";

// 与 Android 9 抽头（sigma 2，±4 截断后归一化）完全一致，用作 CIConvolution9Horizontal/Vertical 的权重。
const CGFloat FLTVideoEffectsGaussianWeights[9] = {
    0.027631, 0.066282, 0.123832, 0.180174, 0.204164, 0.180174, 0.123832, 0.066282, 0.027631};
static const CGFloat FLTNeighbourWeights[9] = {0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0};
static const int FLTMaxConsecutiveFailures = 30;

static CIKernel *FLTBoxKernel;
static CIColorKernel *FLTVarianceKernel, *FLTBeautyKernel;
static CIWarpKernel *FLTWarpKernel;

// 内核与上下文无关，进程内只编译一次；任一为 nil 都按永久故障处理。
static BOOL FLTLoadKernels(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        FLTBoxKernel = [CIKernel kernelWithString:FLTBoxKernelSource];
        FLTVarianceKernel = [CIColorKernel kernelWithString:FLTVarianceKernelSource];
        FLTWarpKernel = [CIWarpKernel kernelWithString:[FLTVideoEffectsKernelWarp stringByAppendingString:FLTWarpKernelSource]];
        FLTBeautyKernel = [CIColorKernel kernelWithString:[@[FLTVideoEffectsKernelCommon, FLTVideoEffectsKernelWarp,
            FLTVideoEffectsKernelMasks, FLTVideoEffectsKernelComposite, FLTBeautyKernelSource] componentsJoinedByString:@""]];
#pragma clang diagnostic pop
    });
    return FLTBoxKernel && FLTVarianceKernel && FLTWarpKernel && FLTBeautyKernel;
}

static double FLTNumber(NSDictionary *config, NSString *key, double fallback) {
    id value = config[key];
    return value ? [value doubleValue] : fallback;
}
static BOOL FLTActive(NSDictionary *config) {
    return [config[@"enabled"] boolValue] && ![config[@"compareOriginal"] boolValue];
}
static BOOL FLTShaping(NSDictionary *config) {
    return FLTNumber(config, @"faceSlim", 0) > 0 || FLTNumber(config, @"eyeEnlarge", 0) > 0;
}
// 人脸跟踪启用谓词（与 Android 共用同一判定）：enabled && !compareOriginal && faceHealthy && healthy &&
// (塑形 || whiten || rosy || skinOnly 磨皮)。渲染故障会清空配置，因此 healthy 已含在 enabled 里。
static BOOL FLTFaceTrackingWanted(NSDictionary *config, BOOL faceHealthy) {
    return faceHealthy && FLTActive(config) && (FLTShaping(config) || FLTNumber(config, @"whiten", 0) > 0 ||
        FLTNumber(config, @"rosy", 0) > 0 || ([config[@"skinOnly"] boolValue] && FLTNumber(config, @"smoothing", 0) > 0));
}
// 中性参数：除塑形外不做任何处理；再加上「没有塑形几何」才严格直通。
static BOOL FLTNeutralParameters(NSDictionary *config) {
    return FLTNumber(config, @"smoothing", 0) == 0 && FLTNumber(config, @"sharpness", 0) == 0 &&
        FLTNumber(config, @"whiten", 0) == 0 && FLTNumber(config, @"rosy", 0) == 0 &&
        FLTNumber(config, @"brightness", 0) == 0 && FLTNumber(config, @"saturation", 1) == 1 &&
        ([config[@"filter"] isEqual:@"none"] || FLTNumber(config, @"filterIntensity", 1) == 0);
}
static CIImage *FLTGaussian(CIImage *image, CGRect rect) {
    NSDictionary *parameters = @{kCIInputWeightsKey: [CIVector vectorWithValues:FLTVideoEffectsGaussianWeights count:9], kCIInputBiasKey: @0};
    CIImage *horizontal = [[image imageByClampingToExtent] imageByApplyingFilter:@"CIConvolution9Horizontal"
        withInputParameters:parameters];
    return [[horizontal imageByApplyingFilter:@"CIConvolution9Vertical" withInputParameters:parameters] imageByCroppingToRect:rect];
}

@implementation FLTVideoEffects {
    CIContext *_context;
    BOOL _rendererChecked;
    CVPixelBufferPoolRef _pool;
    size_t _width, _height;
    // 以下只在 copyProcessedSample 内访问（整体 @synchronized(self)）。
    FLTFaceGeometrySmoother *_smoother;
    NSUInteger _generation;
    int _consecutiveFailures;
    NSDictionary *_countedConfiguration;
}
+ (BOOL)validConfiguration:(NSDictionary *)values {
    if (![values isKindOfClass:NSDictionary.class]) return NO;
    for (NSString *key in @[@"enabled", @"skinOnly", @"compareOriginal"]) {
        id value = values[key];
        if (!value && ![key isEqual:@"enabled"]) continue;
        if (![value isKindOfClass:NSNumber.class] ||
            CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID()) return NO;
    }
    for (NSString *key in @[@"smoothing", @"brightness", @"sharpness", @"saturation", @"filterIntensity", @"faceSlim", @"eyeEnlarge", @"whiten", @"rosy"]) {
        id value = values[key];
        if (!value && ![key isEqual:@"smoothing"] && ![key isEqual:@"brightness"]) continue;
        double maximum = [key isEqual:@"saturation"] ? 2 : 1;
        if (![value isKindOfClass:NSNumber.class] ||
            CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
            !isfinite([value doubleValue]) || [value doubleValue] < 0 || [value doubleValue] > maximum) return NO;
    }
    return [@[@"none", @"warm", @"cool", @"monochrome", @"sepia", @"rose"] containsObject:values[@"filter"]];
}
+ (BOOL)prepassForImage:(CIImage *)image mean:(CIImage **)mean variance:(CIImage **)variance {
    if (!FLTLoadKernels()) return NO;
    CGRect extent = image.extent;
    // 1/4 尺寸取 ceil，原点对齐的精确 4 倍映射，边缘按复制延拓。
    CGRect quarter = CGRectMake(0, 0, ceil(extent.size.width / 4), ceil(extent.size.height / 4));
    CIImage *t0 = [FLTBoxKernel applyWithExtent:quarter roiCallback:^CGRect(int index, CGRect rect) {
        return CGRectInset(CGRectMake(rect.origin.x * 4, rect.origin.y * 4, rect.size.width * 4, rect.size.height * 4), -2, -2);
    } arguments:@[[image imageByClampingToExtent]]];
    if (!t0) return NO;
    CIImage *blurred = FLTGaussian(t0, quarter);
    CIImage *seed = blurred ? [FLTVarianceKernel applyWithExtent:quarter arguments:@[t0, blurred]] : nil;
    CIImage *spread = seed ? FLTGaussian(seed, quarter) : nil;
    if (!spread) return NO;
    // 放大前必须再 clamp，否则边缘会与透明黑混合。
    CGAffineTransform up = CGAffineTransformMakeScale(4, 4);
    *mean = [[[blurred imageByClampingToExtent] imageByApplyingTransform:up] imageByCroppingToRect:extent];
    *variance = [[[spread imageByClampingToExtent] imageByApplyingTransform:up] imageByCroppingToRect:extent];
    return YES;
}
+ (CIImage *)beautyImage:(CIImage *)image configuration:(NSDictionary *)config
                geometry:(const float *)state presence:(float)presence {
    if (!FLTLoadKernels()) return nil;
    CGRect extent = image.extent;
    CGFloat w = extent.size.width, h = extent.size.height;
    double smoothing = FLTNumber(config, @"smoothing", 0), sharpness = FLTNumber(config, @"sharpness", 0);
    NSUInteger style = [@[@"none", @"warm", @"cool", @"monochrome", @"sepia", @"rose"] indexOfObject:config[@"filter"]];
    if (style == NSNotFound) style = 0;
    // 像素空间：中心 × (w, h)，半径与眼宽 × w，axis 不变。无几何时 presence 为 0，内核跳过遮罩与变形。
    CIVector *zero = [CIVector vectorWithX:0 Y:0 Z:0 W:0];
    NSMutableArray<CIVector *> *cheeks = [NSMutableArray arrayWithObjects:zero, zero, zero, zero, nil];
    NSMutableArray<CIVector *> *eyes = [NSMutableArray arrayWithObjects:[CIVector vectorWithX:0 Y:0 Z:0], [CIVector vectorWithX:0 Y:0 Z:0], nil];
    NSMutableArray<CIVector *> *ellipses = [NSMutableArray arrayWithObjects:zero, zero, zero, zero, nil]; // oval、mouth、eyeMask0、eyeMask1
    CIVector *axis = [CIVector vectorWithX:1 Y:0];
    CGFloat slim = 0, big = 0;
    if (state) {
        for (int i = 0; i < 4; i++)
            cheeks[i] = [CIVector vectorWithX:state[4 * i] * w Y:state[4 * i + 1] * h Z:state[4 * i + 2] * w W:state[4 * i + 3] * h];
        for (int i = 0; i < 2; i++)
            eyes[i] = [CIVector vectorWithX:state[16 + 3 * i] * w Y:state[17 + 3 * i] * h Z:state[18 + 3 * i] * w];
        const int offsets[4] = {22, 28, 32, 36};
        for (int i = 0; i < 4; i++) {
            const float *e = state + offsets[i];
            ellipses[i] = [CIVector vectorWithX:e[0] * w Y:e[1] * h Z:e[2] * w W:e[3] * w];
        }
        axis = [CIVector vectorWithX:state[26] Y:state[27]];
        slim = FLTNumber(config, @"faceSlim", 0) * 0.12 * presence * state[40];
        big = FLTNumber(config, @"eyeEnlarge", 0) * 0.25 * presence * state[40];
    } else {
        presence = 0;
    }
    CIVector *size = [CIVector vectorWithX:w Y:h];
    if (slim > 0 || big > 0) {
        image = [FLTWarpKernel applyWithExtent:extent roiCallback:^CGRect(int index, CGRect dest) {
            return extent;
        } inputImage:image arguments:@[cheeks[0], cheeks[1], cheeks[2], cheeks[3], eyes[0], eyes[1], size, @(slim), @(big)]];
        if (!image) return nil;
    }
    // 未用到的输入直接用源图占位，不产生额外渲染通道；对应分支由 uniform 关闭。
    CIImage *meanUp = image, *varUp = image, *avg = image;
    if (smoothing > 0 && ![self prepassForImage:image mean:&meanUp variance:&varUp]) return nil;
    if (sharpness > 0) {
        avg = [[[image imageByClampingToExtent] imageByApplyingFilter:@"CIConvolution3X3" withInputParameters:@{
            kCIInputWeightsKey: [CIVector vectorWithValues:FLTNeighbourWeights count:9], kCIInputBiasKey: @0}]
            imageByCroppingToRect:extent];
        if (!avg) return nil;
    }
    CIVector *p0 = [CIVector vectorWithX:smoothing Y:[config[@"skinOnly"] boolValue] ? 1 : 0 Z:sharpness
        W:FLTNumber(config, @"saturation", 1)];
    CIVector *p1 = [CIVector vectorWithX:FLTNumber(config, @"whiten", 0) Y:FLTNumber(config, @"rosy", 0)
        Z:FLTNumber(config, @"brightness", 0) W:FLTNumber(config, @"filterIntensity", 1)];
    return [FLTBeautyKernel applyWithExtent:extent arguments:@[image, meanUp, varUp, avg,
        ellipses[0], axis, ellipses[1], ellipses[2], ellipses[3], eyes[0], eyes[1],
        cheeks[0], cheeks[1], cheeks[2], cheeks[3], @(presence), size,
        [CIVector vectorWithX:slim Y:big], p0, p1, @(style)]];
}
static CVPixelBufferRef FLTSolidBuffer(int side, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha) {
    CVPixelBufferRef buffer = NULL;
    if (CVPixelBufferCreate(NULL, side, side, kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)@{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}}, &buffer) != kCVReturnSuccess) return NULL;
    CVPixelBufferLockBaseAddress(buffer, 0);
    uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
    size_t stride = CVPixelBufferGetBytesPerRow(buffer);
    for (int y = 0; y < side; y++) for (int x = 0; x < side; x++) {
        uint8_t *c = base + y * stride + x * 4;
        c[0] = blue; c[1] = green; c[2] = red; c[3] = alpha;
    }
    CVPixelBufferUnlockBaseAddress(buffer, 0);
    return buffer;
}
// 首次使用的内核自检：CIKL 后端编译失败时 render 既不抛异常也不报错，只会留下目标缓冲原值。
// 8×8 纯色 (170,115,85)，smoothing = sharpness = brightness = 1，渲染到预填哨兵值的缓冲；
// 纯色帧上只有 lift(x, 0.5) 可见，中心须为 (191,141,109) ±1 且 alpha = 255。
- (BOOL)selfTestWithContext:(CIContext *)context {
    CVPixelBufferRef input = FLTSolidBuffer(8, 85, 115, 170, 255), output = FLTSolidBuffer(8, 3, 3, 3, 3);
    BOOL passed = NO;
    if (input && output) {
        CIImage *image = [FLTVideoEffects beautyImage:[CIImage imageWithCVPixelBuffer:input options:@{kCIImageColorSpace: [NSNull null]}]
            configuration:@{@"enabled": @YES, @"smoothing": @1, @"sharpness": @1, @"brightness": @1, @"skinOnly": @NO,
                            @"filter": @"none"} geometry:NULL presence:0];
        if (image) {
            [context render:image toCVPixelBuffer:output bounds:CGRectMake(0, 0, 8, 8) colorSpace:NULL];
            CVPixelBufferLockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
            const uint8_t *c = (const uint8_t *)CVPixelBufferGetBaseAddress(output) + 4 * CVPixelBufferGetBytesPerRow(output) + 4 * 4;
            passed = abs(c[2] - 191) <= 1 && abs(c[1] - 141) <= 1 && abs(c[0] - 109) <= 1 && c[3] == 255;
            CVPixelBufferUnlockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
        }
    }
    if (input) CVPixelBufferRelease(input);
    if (output) CVPixelBufferRelease(output);
    return passed;
}
- (BOOL)prepareRenderer {
    if (_rendererChecked) return _context != nil;
    _rendererChecked = YES;
    // 不做色彩管理：所有计算在伽马编码值上进行，与 Android 一致；显式给出全部选项以消除 iOS/macOS 默认值差异。
    CIContext *context = [CIContext contextWithOptions:@{
        kCIContextWorkingColorSpace: [NSNull null], kCIContextOutputColorSpace: [NSNull null],
        kCIContextWorkingFormat: @(kCIFormatRGBAh), kCIContextHighQualityDownsample: @NO,
        kCIContextCacheIntermediates: @NO}];
    if (!context || !FLTLoadKernels() || ![self selfTestWithContext:context]) return NO;
    _context = context;
    return YES;
}
// 平滑器只在采集队列（本方法的锁内）更新；tracker 代际号变化、尺寸变化或停止跟踪时就地重置。
- (BOOL)smoothedGeometry:(float *)state presence:(float *)presence input:(CVPixelBufferRef)input config:(NSDictionary *)config {
    *presence = 0;
    BOOL faceEnabled = FLTFaceTrackingWanted(config, !self.faceTrackingFailed);
    NSUInteger generation = _generation;
    NSInteger sampleTimestamp = 0;
    FLTFaceGeometry *target = [self.faceTracker geometryForPixelBuffer:input enabled:faceEnabled
        generation:&generation sampleTimestamp:&sampleTimestamp];
    float aspect = input && CVPixelBufferGetHeight(input) ? (float)CVPixelBufferGetWidth(input) / CVPixelBufferGetHeight(input) : 1;
    if (!_smoother) _smoother = [[FLTFaceGeometrySmoother alloc] initWithAspect:aspect];
    if (!faceEnabled || generation != _generation || aspect != _smoother.aspect) {
        _smoother.aspect = aspect;
        [_smoother reset];
        _generation = generation;
    }
    if (!faceEnabled) return NO;
    float targetState[FLTFaceGeometryStateSize];
    if (target) [target getState:targetState];
    double now = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock())) * 1000; // 与 tracker 的 CACurrentMediaTime 同一时钟
    BOOL tracked = [_smoother updateTarget:target ? targetState : NULL sampleTimestamp:sampleTimestamp now:now state:state];
    *presence = _smoother.presence;
    return tracked;
}
- (void)fallBackToOriginal {
    self.configuration = nil;
    _consecutiveFailures = 0;
    if (self.onFailure) self.onFailure();
}
- (CMSampleBufferRef)copyProcessedSample:(CMSampleBufferRef)sample {
    // 开播时采集委托换队列，前后两帧可能短暂并发进入；平滑器与失败计数都只在锁内读写。
    @synchronized(self) {
    @autoreleasepool {
        NSDictionary *config = self.configuration;
        if (config != _countedConfiguration) { _consecutiveFailures = 0; _countedConfiguration = config; }
        CVPixelBufferRef input = CMSampleBufferGetImageBuffer(sample);
        float state[FLTFaceGeometryStateSize], presence = 0;
        BOOL tracked = [self smoothedGeometry:state presence:&presence input:input config:config];
        if (!FLTActive(config) || (FLTNeutralParameters(config) && !(tracked && FLTShaping(config)))) {
            _consecutiveFailures = 0;
            return (CMSampleBufferRef)CFRetain(sample);
        }
        CVPixelBufferRef output = NULL;
        CMVideoFormatDescriptionRef format = NULL;
        CMSampleBufferRef processed = NULL;
        BOOL permanent = NO;
        // 预期内的失败（池超阈值、格式/样本创建失败等）一律 break 出 do/while，不用 @throw 做控制流：
        // 插件未开 -fobjc-arc-exceptions，异常离开作用域时 ARC 不释放强引用局部变量（limits、image 及其持有的相机缓冲），
        // 而池超阈值是会连续重复的常规丢帧路径。@try 只兜底 Core Image 意外抛出的异常（按瞬时故障计）。
        @try {
            do {
                if (!input) break;
                if (![self prepareRenderer]) { permanent = YES; break; }
                size_t width = CVPixelBufferGetWidth(input), height = CVPixelBufferGetHeight(input);
                if (!_pool || width != _width || height != _height) {
                    if (_pool) { CVPixelBufferPoolRelease(_pool); _pool = NULL; }
                    NSDictionary *attributes = @{
                        (id)kCVPixelBufferWidthKey: @(width), (id)kCVPixelBufferHeightKey: @(height),
                        (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                        (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
                        (id)kCVPixelBufferMetalCompatibilityKey: @YES,
                    };
                    if (CVPixelBufferPoolCreate(NULL, NULL, (__bridge CFDictionaryRef)attributes, &_pool) != kCVReturnSuccess) {
                        permanent = YES;
                        break;
                    }
                    _width = width; _height = height;
                }
                // 编码器滞留帧时限制分配，禁止无界积压；超过阈值属瞬时故障，丢帧等缓冲回池。
                NSDictionary *limits = @{(id)kCVPixelBufferPoolAllocationThresholdKey: @6};
                if (CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(NULL, _pool, (__bridge CFDictionaryRef)limits, &output) != kCVReturnSuccess)
                    break;
                CIImage *image = [CIImage imageWithCVPixelBuffer:input options:@{kCIImageColorSpace: [NSNull null]}];
                CGRect extent = image.extent;
                image = [FLTVideoEffects beautyImage:image configuration:config geometry:tracked ? state : NULL presence:presence];
                if (!image) break;
                CVBufferPropagateAttachments(input, output);
                [_context render:image toCVPixelBuffer:output bounds:extent colorSpace:NULL];
                if (CMVideoFormatDescriptionCreateForImageBuffer(NULL, output, &format) != noErr) break;
                CMSampleTimingInfo timing;
                if (CMSampleBufferGetSampleTimingInfo(sample, 0, &timing) != noErr ||
                    CMSampleBufferCreateReadyWithImageBuffer(NULL, output, format, &timing, &processed) != noErr) {
                    if (processed) { CFRelease(processed); processed = NULL; }
                    break;
                }
            } while (0);
        } @catch (NSException *exception) {
            if (processed) { CFRelease(processed); processed = NULL; }
        } @finally {
            if (format) CFRelease(format);
            if (output) CVPixelBufferRelease(output);
        }
        if (processed) {
            _consecutiveFailures = 0;
            return processed;
        }
        // 永久故障或连续 30 帧瞬时故障：回原画并报告一次效果故障（不映射为断流）。
        if (permanent || ++_consecutiveFailures >= FLTMaxConsecutiveFailures) {
            [self fallBackToOriginal];
            return (CMSampleBufferRef)CFRetain(sample);
        }
        return NULL;
    }
    }
}
- (void)dealloc { if (_pool) CVPixelBufferPoolRelease(_pool); }
@end
