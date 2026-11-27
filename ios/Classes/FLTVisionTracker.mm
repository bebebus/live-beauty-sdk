#import "FLTVisionTracker.h"

#import <CoreGraphics/CoreGraphics.h>
#import <TargetConditionals.h>
#import <Vision/Vision.h>
#if TARGET_OS_IOS && __has_include(<MediaPipeTasksVision/MediaPipeTasksVision.h>)
#import <MediaPipeTasksVision/MediaPipeTasksVision.h>
#define FLT_HAS_MPP_SEGMENTER 1
#else
#define FLT_HAS_MPP_SEGMENTER 0
#endif
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

@implementation FLTVisionTracker {
    dispatch_queue_t _queue;
    std::atomic<bool> _inFlight;
    std::mutex _lock;

    VNSequenceRequestHandler *_seqHandler API_AVAILABLE(ios(15.0), macos(12.0));
    VNGeneratePersonSegmentationRequest *_segReq API_AVAILABLE(ios(15.0), macos(12.0));
    VNDetectHumanBodyPoseRequest *_poseReq API_AVAILABLE(ios(14.0), macos(11.0));
#if FLT_HAS_MPP_SEGMENTER
    MPPImageSegmenter *_multiclassSegmenter;
    BOOL _multiclassFailed;
    int64_t _lastMulticlassInferTsMs;
    int64_t _multiclassIdleSinceMs;
    std::atomic<bool> _hasMulticlassSegmenter;
#endif

    CVPixelBufferRef _latestMask;
    int64_t _latestMaskTsMs;

    float _latestPoseXyv[33 * 3];
    int64_t _latestPoseTsMs;
    bool _poseDirty;

    std::atomic<int64_t> _lastSegMs;
    std::atomic<int64_t> _lastPoseMs;
    std::atomic<int64_t> _lastAuxMs;
    std::atomic<int64_t> _lastMulticlassMs;
    std::vector<uint8_t> _auxRgba;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _qualityLevel = 1;
        _inFlight.store(false);
        _latestMask = nullptr;
        _latestMaskTsMs = -1;
        _latestPoseTsMs = -1;
        _poseDirty = false;
        _lastSegMs.store(INT64_MIN, std::memory_order_relaxed);
        _lastPoseMs.store(INT64_MIN, std::memory_order_relaxed);
        _lastAuxMs.store(INT64_MIN, std::memory_order_relaxed);
        _lastMulticlassMs.store(INT64_MIN, std::memory_order_relaxed);
#if FLT_HAS_MPP_SEGMENTER
        _multiclassFailed = NO;
        _lastMulticlassInferTsMs = 0;
        _multiclassIdleSinceMs = INT64_MIN;
        _hasMulticlassSegmenter.store(false, std::memory_order_relaxed);
#endif
        std::memset(_latestPoseXyv, 0, sizeof(_latestPoseXyv));

        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
        _queue = dispatch_queue_create("com.bebebus.livebeauty.vision", attr);
    }
    return self;
}

- (void)dealloc {
    std::lock_guard<std::mutex> guard(_lock);
    if (_latestMask) {
        CFRelease(_latestMask);
        _latestMask = nullptr;
    }
}

- (void)reset {
    std::lock_guard<std::mutex> guard(_lock);
    if (_latestMask) {
        CFRelease(_latestMask);
        _latestMask = nullptr;
    }
    _latestMaskTsMs = -1;
    _latestPoseTsMs = -1;
    _poseDirty = false;
    _lastSegMs.store(INT64_MIN, std::memory_order_relaxed);
    _lastPoseMs.store(INT64_MIN, std::memory_order_relaxed);
    _lastAuxMs.store(INT64_MIN, std::memory_order_relaxed);
    _lastMulticlassMs.store(INT64_MIN, std::memory_order_relaxed);
}

- (void)submitPixelBuffer:(CVPixelBufferRef)pixelBuffer
          sampleTimestamp:(int64_t)sampleTsMs
         wantSegmentation:(BOOL)wantSeg
             wantBodyPose:(BOOL)wantPose
              wantFaceAux:(BOOL)wantAux
          onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock {
    [self submitPixelBuffer:pixelBuffer
            sampleTimestamp:sampleTsMs
           wantSegmentation:wantSeg
               wantBodyPose:wantPose
                wantFaceAux:wantAux
             wantMulticlass:NO
            onAuxRgbaSample:auxBlock
         onMulticlassPlanes:nil];
}

- (void)submitPixelBuffer:(CVPixelBufferRef)pixelBuffer
          sampleTimestamp:(int64_t)sampleTsMs
         wantSegmentation:(BOOL)wantSeg
             wantBodyPose:(BOOL)wantPose
              wantFaceAux:(BOOL)wantAux
           wantMulticlass:(BOOL)wantMulticlass
          onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock
       onMulticlassPlanes:(nullable void (^)(const float * const _Nonnull * _Nonnull planes, int width, int height, int64_t sampleTsMs))multiclassBlock {
    if (!pixelBuffer || (!wantSeg && !wantPose && !wantAux && !wantMulticlass)) return;

    const int64_t prevSeg = _lastSegMs.load(std::memory_order_relaxed);
    const int64_t prevPose = _lastPoseMs.load(std::memory_order_relaxed);
    const int64_t prevAux = _lastAuxMs.load(std::memory_order_relaxed);
    const int64_t prevMc = _lastMulticlassMs.load(std::memory_order_relaxed);
    const BOOL doSeg = wantSeg && (prevSeg == INT64_MIN || sampleTsMs - prevSeg >= 33);
    const BOOL doPose = wantPose && (prevPose == INT64_MIN || sampleTsMs - prevPose >= 66);
    const BOOL doAux = wantAux && auxBlock != nil &&
                       (prevAux == INT64_MIN || (doSeg && prevSeg == INT64_MIN) || sampleTsMs - prevAux >= 95);
    const BOOL doMc = wantMulticlass && multiclassBlock != nil &&
                      (prevMc == INT64_MIN || sampleTsMs - prevMc >= 95);
    if (!doSeg && !doPose && !doAux && !doMc) {
#if FLT_HAS_MPP_SEGMENTER
        // Idle 5s reclamation: only wake queue if segmenter is currently resident in RAM
        if (!wantMulticlass && _hasMulticlassSegmenter.load(std::memory_order_relaxed)) {
            bool expectedIdle = false;
            if (_inFlight.compare_exchange_strong(expectedIdle, true)) {
                dispatch_async(_queue, ^{
                    if (self->_multiclassSegmenter) {
                        if (self->_multiclassIdleSinceMs == INT64_MIN) {
                            self->_multiclassIdleSinceMs = sampleTsMs;
                        } else if (sampleTsMs - self->_multiclassIdleSinceMs >= 5000) {
                            self->_multiclassSegmenter = nil;
                            self->_multiclassIdleSinceMs = INT64_MIN;
                            self->_hasMulticlassSegmenter.store(false, std::memory_order_relaxed);
                        }
                    }
                    self->_inFlight.store(false);
                });
            }
        }
#endif
        return;
    }

    bool expected = false;
    if (!_inFlight.compare_exchange_strong(expected, true)) return;

    if (doSeg) _lastSegMs.store(sampleTsMs, std::memory_order_relaxed);
    if (doPose) _lastPoseMs.store(sampleTsMs, std::memory_order_relaxed);
    if (doAux) _lastAuxMs.store(sampleTsMs, std::memory_order_relaxed);
    if (doMc) _lastMulticlassMs.store(sampleTsMs, std::memory_order_relaxed);

    CVPixelBufferRetain(pixelBuffer);
    const int reqQuality = self.qualityLevel;
    void (^copiedAux)(const uint8_t *, int, int) = [auxBlock copy];
    void (^copiedMc)(const float * const *, int, int, int64_t) = [multiclassBlock copy];

    dispatch_async(_queue, ^{
        @autoreleasepool {
            [self runOnQueueWithBuffer:pixelBuffer
                       sampleTimestamp:sampleTsMs
                                 doSeg:doSeg
                                doPose:doPose
                                 doAux:doAux
                          doMulticlass:doMc
                          qualityLevel:reqQuality
                       onAuxRgbaSample:copiedAux
                    onMulticlassPlanes:copiedMc];
        }
        CVPixelBufferRelease(pixelBuffer);
        self->_inFlight.store(false);
    });
}

- (void)runOnQueueWithBuffer:(CVPixelBufferRef)pb
             sampleTimestamp:(int64_t)sampleTsMs
                       doSeg:(BOOL)doSeg
                      doPose:(BOOL)doPose
                       doAux:(BOOL)doAux
                doMulticlass:(BOOL)doMulticlass
                qualityLevel:(int)qualityLevel
             onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock
          onMulticlassPlanes:(nullable void (^)(const float * const _Nonnull * _Nonnull planes, int width, int height, int64_t sampleTsMs))multiclassBlock {
    if (@available(iOS 15.0, macOS 12.0, *)) {
        if (doSeg || doPose) {
            if (!_seqHandler) {
                _seqHandler = [[VNSequenceRequestHandler alloc] init];
            }
            NSMutableArray<VNRequest *> *requests = [NSMutableArray arrayWithCapacity:2];
            if (doSeg) {
                if (!_segReq) {
                    _segReq = [[VNGeneratePersonSegmentationRequest alloc] init];
                    _segReq.outputPixelFormat = kCVPixelFormatType_OneComponent8;
                }
                // Keep Balanced quality on Apple Neural Engine (<4ms) regardless of loadLevel:
                // QualityLevelFast drops arms, dark hair & shoulders, causing virtual background to occlude the person.
                (void)qualityLevel;
                _segReq.qualityLevel = VNGeneratePersonSegmentationRequestQualityLevelBalanced;
                [requests addObject:_segReq];
            }
            if (doPose) {
                if (!_poseReq) {
                    _poseReq = [[VNDetectHumanBodyPoseRequest alloc] init];
                }
                [requests addObject:_poseReq];
            }

            NSError *err = nil;
            const BOOL ok = [_seqHandler performRequests:requests onCVPixelBuffer:pb error:&err];
            if (ok && !err) {
                if (doSeg) {
                    VNPixelBufferObservation *obs = _segReq.results.firstObject;
                    CVPixelBufferRef maskBuf = obs.pixelBuffer;
                    if (maskBuf) {
                        CVPixelBufferRetain(maskBuf);
                        std::lock_guard<std::mutex> guard(_lock);
                        if (_latestMask) CFRelease(_latestMask);
                        _latestMask = maskBuf;
                        _latestMaskTsMs = sampleTsMs;
                    }
                }
                if (doPose) {
                    VNHumanBodyPoseObservation *poseObs = _poseReq.results.firstObject;
                    if (poseObs) {
                        float mapped[33 * 3];
                        if ([self mapBodyObservation:poseObs toMediaPipe33:mapped]) {
                            std::lock_guard<std::mutex> guard(_lock);
                            std::memcpy(_latestPoseXyv, mapped, sizeof(mapped));
                            _latestPoseTsMs = sampleTsMs;
                            _poseDirty = true;
                        }
                    }
                }
            }
        }
    }

    if (doAux && auxBlock) {
        constexpr int kAuxW = 128;
        constexpr int kAuxH = 128;
        if (_auxRgba.size() != kAuxW * kAuxH * 4) {
            _auxRgba.resize(kAuxW * kAuxH * 4, 0);
        }
        if ([self downsamplePixelBuffer:pb toTopDownRgba:_auxRgba.data() dstW:kAuxW dstH:kAuxH]) {
            auxBlock(_auxRgba.data(), kAuxW, kAuxH);
        }
    }

#if FLT_HAS_MPP_SEGMENTER
    if (doMulticlass && multiclassBlock && !_multiclassFailed) {
        _multiclassIdleSinceMs = INT64_MIN;
        if (!_multiclassSegmenter) {
            NSBundle *owner = [NSBundle bundleForClass:self.class];
            NSString *modelPath = [owner pathForResource:@"selfie_multiclass_256x256" ofType:@"tflite"] ?:
                [NSBundle.mainBundle pathForResource:@"selfie_multiclass_256x256" ofType:@"tflite"];
            if (modelPath) {
                MPPImageSegmenterOptions *opts = [MPPImageSegmenterOptions new];
                opts.baseOptions.modelAssetPath = modelPath;
                opts.runningMode = MPPRunningModeVideo;
                opts.shouldOutputConfidenceMasks = YES;
                opts.shouldOutputCategoryMask = NO;
                NSError *initErr = nil;
                _multiclassSegmenter = [[MPPImageSegmenter alloc] initWithOptions:opts error:&initErr];
                if (!_multiclassSegmenter || initErr) {
                    _multiclassSegmenter = nil;
                    _multiclassFailed = YES;
                } else {
                    _hasMulticlassSegmenter.store(true, std::memory_order_relaxed);
                }
            } else {
                _multiclassFailed = YES;
            }
        }
        if (_multiclassSegmenter && CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA) {
            NSError *imgErr = nil;
            MPPImage *mppImg = [[MPPImage alloc] initWithPixelBuffer:pb error:&imgErr];
            if (mppImg && !imgErr) {
                int64_t inferTs = (sampleTsMs > _lastMulticlassInferTsMs)
                    ? sampleTsMs
                    : (_lastMulticlassInferTsMs + 1);
                _lastMulticlassInferTsMs = inferTs;
                [_multiclassSegmenter segmentVideoFrame:mppImg
                                timestampInMilliseconds:(NSInteger)inferTs
                                  withCompletionHandler:^(MPPImageSegmenterResult * _Nullable result,
                                                          NSError * _Nullable segErr) {
                    if (segErr || !result || result.confidenceMasks.count < 6) return;
                    const float *planes[6];
                    const int mw = static_cast<int>(result.confidenceMasks[0].width);
                    const int mh = static_cast<int>(result.confidenceMasks[0].height);
                    if (mw <= 0 || mh <= 0) return;
                    for (int c = 0; c < 6; ++c) {
                        MPPMask *m = result.confidenceMasks[static_cast<NSUInteger>(c)];
                        const float *ptr = m.float32Data;
                        if (!ptr || static_cast<int>(m.width) != mw || static_cast<int>(m.height) != mh) return;
                        planes[c] = ptr;
                    }
                    multiclassBlock(planes, mw, mh, sampleTsMs);
                }];
            }
        }
    }
#else
    (void)doMulticlass;
    (void)multiclassBlock;
#endif
}

- (BOOL)mapBodyObservation:(VNHumanBodyPoseObservation *)obs
             toMediaPipe33:(float *)outXyv99 API_AVAILABLE(ios(14.0), macos(11.0)) {
    std::memset(outXyv99, 0, sizeof(float) * 33 * 3);
    NSError *err = nil;
    NSDictionary<VNHumanBodyPoseObservationJointName, VNRecognizedPoint *> *pts =
        [obs recognizedPointsForJointsGroupName:VNHumanBodyPoseObservationJointsGroupNameAll error:&err];
    if (err || pts.count == 0) return NO;

    auto putJoint = [&](int mpIdx, VNHumanBodyPoseObservationJointName key) {
        VNRecognizedPoint *pt = pts[key];
        if (!pt || pt.confidence < 0.1f) return;
        outXyv99[mpIdx * 3 + 0] = static_cast<float>(pt.location.x);
        outXyv99[mpIdx * 3 + 1] = static_cast<float>(1.0 - pt.location.y);  // Vision bottom-left -> top-down
        outXyv99[mpIdx * 3 + 2] = static_cast<float>(pt.confidence);
    };

    putJoint(0, VNHumanBodyPoseObservationJointNameNose);
    putJoint(2, VNHumanBodyPoseObservationJointNameLeftEye);
    putJoint(5, VNHumanBodyPoseObservationJointNameRightEye);
    putJoint(7, VNHumanBodyPoseObservationJointNameLeftEar);
    putJoint(8, VNHumanBodyPoseObservationJointNameRightEar);
    putJoint(11, VNHumanBodyPoseObservationJointNameLeftShoulder);
    putJoint(12, VNHumanBodyPoseObservationJointNameRightShoulder);
    putJoint(13, VNHumanBodyPoseObservationJointNameLeftElbow);
    putJoint(14, VNHumanBodyPoseObservationJointNameRightElbow);
    putJoint(15, VNHumanBodyPoseObservationJointNameLeftWrist);
    putJoint(16, VNHumanBodyPoseObservationJointNameRightWrist);
    putJoint(23, VNHumanBodyPoseObservationJointNameLeftHip);
    putJoint(24, VNHumanBodyPoseObservationJointNameRightHip);
    putJoint(25, VNHumanBodyPoseObservationJointNameLeftKnee);
    putJoint(26, VNHumanBodyPoseObservationJointNameRightKnee);
    putJoint(27, VNHumanBodyPoseObservationJointNameLeftAnkle);
    putJoint(28, VNHumanBodyPoseObservationJointNameRightAnkle);

    // Require at least shoulders to have valid confidence
    return (outXyv99[11 * 3 + 2] >= 0.2f && outXyv99[12 * 3 + 2] >= 0.2f) ? YES : NO;
}

- (BOOL)downsamplePixelBuffer:(CVPixelBufferRef)pb
                toTopDownRgba:(uint8_t *)dst
                         dstW:(int)dstW
                         dstH:(int)dstH {
    const OSType fmt = CVPixelBufferGetPixelFormatType(pb);
    if (fmt != kCVPixelFormatType_32BGRA && fmt != kCVPixelFormatType_32RGBA) return NO;
    if (CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return NO;

    const size_t srcW = CVPixelBufferGetWidth(pb);
    const size_t srcH = CVPixelBufferGetHeight(pb);
    const size_t stride = CVPixelBufferGetBytesPerRow(pb);
    const uint8_t *base = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(pb));
    const bool isBgra = (fmt == kCVPixelFormatType_32BGRA);

    if (!base || srcW == 0 || srcH == 0) {
        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        return NO;
    }

    for (int y = 0; y < dstH; ++y) {
        const size_t sy = ((static_cast<size_t>(y) * srcH) / static_cast<size_t>(dstH));
        const uint8_t *row = base + sy * stride;
        uint8_t *outRow = dst + y * dstW * 4;
        for (int x = 0; x < dstW; ++x) {
            const size_t sx = ((static_cast<size_t>(x) * srcW) / static_cast<size_t>(dstW));
            const uint8_t *p = row + sx * 4;
            if (isBgra) {
                outRow[x * 4 + 0] = p[2];
                outRow[x * 4 + 1] = p[1];
                outRow[x * 4 + 2] = p[0];
                outRow[x * 4 + 3] = p[3];
            } else {
                outRow[x * 4 + 0] = p[0];
                outRow[x * 4 + 1] = p[1];
                outRow[x * 4 + 2] = p[2];
                outRow[x * 4 + 3] = p[3];
            }
        }
    }

    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return YES;
}

- (nullable CVPixelBufferRef)copyLatestPersonMaskBuffer:(nullable int64_t *)outSampleTsMs {
    std::lock_guard<std::mutex> guard(_lock);
    if (!_latestMask) return nullptr;
    if (outSampleTsMs) *outSampleTsMs = _latestMaskTsMs;
    return CVPixelBufferRetain(_latestMask);
}

- (BOOL)pollLatestBodyPose:(float *)outXyv99 sampleTimestamp:(int64_t *)outSampleTsMs {
    if (!outXyv99) return NO;
    std::lock_guard<std::mutex> guard(_lock);
    if (!_poseDirty) return NO;
    _poseDirty = false;
    std::memcpy(outXyv99, _latestPoseXyv, sizeof(_latestPoseXyv));
    if (outSampleTsMs) *outSampleTsMs = _latestPoseTsMs;
    return YES;
}

@end
