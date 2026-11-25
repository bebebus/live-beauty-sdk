#import "FLTFaceLandmarkTracker.h"
#import <QuartzCore/QuartzCore.h>
#if __has_include(<MediaPipeTasksVision/MediaPipeTasksVision.h>)
#import <MediaPipeTasksVision/MediaPipeTasksVision.h>
#define LB_HAS_MEDIAPIPE 1
#else
#define LB_HAS_MEDIAPIPE 0
#endif

@implementation FLTFaceLandmarkTracker {
    dispatch_queue_t _worker;
#if LB_HAS_MEDIAPIPE
    MPPFaceLandmarker *_detector; // 仅worker访问。
#endif
    CIContext *_context;
    BOOL _busy, _enabled;
    BOOL _failed; // 模型加载或推理失败后置位，本实例内不复位、不再提交（规格 1.3：本次预览内不重试）。
    NSUInteger _epoch;
    NSInteger _lastSample, _lastSubmitted, _resultTime;
    size_t _width, _height;
    FLTFaceGeometry *_geometry;
}
- (instancetype)init {
    if ((self=[super init])) {
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        _worker=dispatch_queue_create("io.bebebus.livebeauty.face-landmarks", attr);
        _lastSample=-1000; _lastSubmitted=-1;
    }
    return self;
}
- (void)reset {
    @synchronized(self) { _epoch++; _geometry=nil; _lastSample=-1000; }
}
- (FLTFaceGeometry *)geometryForPixelBuffer:(CVPixelBufferRef)buffer enabled:(BOOL)enabled
                                 generation:(NSUInteger *)generation sampleTimestamp:(NSInteger *)sampleTimestamp {
    if (!buffer) return nil;
    NSInteger now=(NSInteger)(CACurrentMediaTime()*1000);
    size_t width=CVPixelBufferGetWidth(buffer),height=CVPixelBufferGetHeight(buffer);
    __block NSUInteger epoch;
    FLTFaceGeometry *current;
    @synchronized(self) {
        if (_enabled != enabled || width != _width || height != _height) {
            [self reset]; _enabled=enabled; _width=width; _height=height;
        }
        // 几何、代际号与采样时间戳在同一把锁内取出；平滑器以时间戳变化识别新样本。
        current=_enabled && !_failed && now-_resultTime <= 250 ? _geometry : nil;
        if (generation) *generation=_epoch;
        if (sampleTimestamp) *sampleTimestamp=_resultTime;
        if (!enabled || _failed || _busy || now <= _lastSubmitted || now-_lastSample < 14) return current;
        _busy=YES; _lastSample=now; _lastSubmitted=now; epoch=_epoch;
    }
    CVPixelBufferRetain(buffer); // 最多一帧在后台；不阻塞采集等待推理。
    dispatch_async(_worker, ^{
        @autoreleasepool {
            BOOL failed=NO;
            FLTFaceGeometry *geometry=nil;
            CVPixelBufferRef source=buffer,small=NULL;
            // 预期内的失败用 failed + break，不用 @throw 做控制流：插件未开 -fobjc-arc-exceptions，
            // 异常离开作用域时 ARC 不释放强引用局部变量。@try 只兜底 MediaPipe/Core Image 意外抛出的异常。
            @try {
                do {
                    // 先缩小到 256 px 并立即把采集缓冲还回相机池，模型冷加载与推理期间只持有小图。
                    CGFloat scale=256.0/MAX(width,height);
                    size_t w=MAX(1,(size_t)(width*scale)),h=MAX(1,(size_t)(height*scale));
                    if (CVPixelBufferCreate(NULL,w,h,kCVPixelFormatType_32BGRA,
                        (__bridge CFDictionaryRef)@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}},&small) != kCVReturnSuccess) {
                        failed=YES; break;
                    }
                    @autoreleasepool { // CIImage 持有采集缓冲：出作用域即释放，之后才能真正还回相机池。
                        if (!self->_context) self->_context=[CIContext contextWithOptions:@{kCIContextCacheIntermediates:@NO}];
                        CIImage *image=[[CIImage imageWithCVPixelBuffer:source] imageByApplyingTransform:CGAffineTransformMakeScale((CGFloat)w/width,(CGFloat)h/height)];
                        CGColorSpaceRef space=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
                        [self->_context render:image toCVPixelBuffer:small bounds:CGRectMake(0,0,w,h) colorSpace:space];
                        CGColorSpaceRelease(space);
                    }
                    CVPixelBufferRelease(source); source=NULL;
#if LB_HAS_MEDIAPIPE
                    if (!self->_detector) {
                        NSBundle *owner=[NSBundle bundleForClass:self.class];
                        NSString *bundlePath=[owner pathForResource:@"LiveBeautyModels" ofType:@"bundle"] ?:
                            [NSBundle.mainBundle pathForResource:@"LiveBeautyModels" ofType:@"bundle"];
                        NSString *path=bundlePath ? [[NSBundle bundleWithPath:bundlePath] pathForResource:@"face_landmarker" ofType:@"task"] :
                            ([owner pathForResource:@"face_landmarker" ofType:@"task"] ?: [NSBundle.mainBundle pathForResource:@"face_landmarker" ofType:@"task"]);
                        if (!path) { failed=YES; break; }
                        MPPFaceLandmarkerOptions *options=[MPPFaceLandmarkerOptions new];
                        options.baseOptions.modelAssetPath=path;
                        options.runningMode=MPPRunningModeVideo;
                        options.numFaces=2;
                        options.minFaceDetectionConfidence=0.5; options.minFacePresenceConfidence=0.5;
                        options.minTrackingConfidence=0.35;
                        self->_detector=[[MPPFaceLandmarker alloc] initWithOptions:options error:nil];
                        if (!self->_detector) { failed=YES; break; }
                    }
                    MPPImage *input=[[MPPImage alloc] initWithPixelBuffer:small error:nil];
                    if (!input) { failed=YES; break; }
                    MPPFaceLandmarkerResult *result=[self->_detector detectVideoFrame:input timestampInMilliseconds:now error:nil];
                    if (!result) { failed=YES; break; }
                    if (result.faceLandmarks.count == 1) {
                        NSMutableArray *points=[NSMutableArray new];
                        for (MPPNormalizedLandmark *point in result.faceLandmarks.firstObject)
                            [points addObject:@[@(point.x),@(point.y)]];
                        geometry=[FLTFaceGeometry fromMesh:points aspect:(CGFloat)width/height];
                    }
#else
                    failed=YES;
#endif
                } while (0);
            } @catch (NSException *exception) {
                failed=YES;
            } @finally {
                if (small) CVPixelBufferRelease(small);
                if (source) CVPixelBufferRelease(source);
            }
            BOOL report=NO;
            @synchronized(self) {
                self->_busy=NO;
                if (failed) {
                    // 与 Android 一致：人脸故障与代际、启用状态无关（期间 reset/关闭跟踪也照样报告），
                    // 本实例停止跟踪且至多报告一次。
                    report=!self->_failed;
                    self->_failed=YES;
                    self->_geometry=nil;
                } else if (epoch == self->_epoch && self->_enabled) {
                    self->_geometry=CACurrentMediaTime()*1000-now <= 250 ? geometry : nil;
                    self->_resultTime=now;
                }
            }
            void (^failure)(void)=self.onFailure;
            if (report && failure) failure();
        }
    });
    return current;
}
@end
