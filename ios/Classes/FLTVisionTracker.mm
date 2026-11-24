#import "FLTVisionTracker.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Vision/Vision.h>
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

    CVPixelBufferRef _latestMask;
    int64_t _latestMaskTsMs;

    float _latestPoseXyv[33 * 3];
    int64_t _latestPoseTsMs;
    bool _poseDirty;

    int64_t _lastSegMs;
    int64_t _lastPoseMs;
    int64_t _lastAuxMs;
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
        _lastSegMs = INT64_MIN;
        _lastPoseMs = INT64_MIN;
        _lastAuxMs = INT64_MIN;
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
    _lastSegMs = INT64_MIN;
    _lastPoseMs = INT64_MIN;
    _lastAuxMs = INT64_MIN;
}

- (void)submitPixelBuffer:(CVPixelBufferRef)pixelBuffer
          sampleTimestamp:(int64_t)sampleTsMs
         wantSegmentation:(BOOL)wantSeg
             wantBodyPose:(BOOL)wantPose
              wantFaceAux:(BOOL)wantAux
          onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock {
    if (!pixelBuffer || (!wantSeg && !wantPose && !wantAux)) return;

    const BOOL doSeg = wantSeg && (_lastSegMs == INT64_MIN || sampleTsMs - _lastSegMs >= 45);
    const BOOL doPose = wantPose && (_lastPoseMs == INT64_MIN || sampleTsMs - _lastPoseMs >= 66);
    const BOOL doAux = wantAux && auxBlock != nil && (_lastAuxMs == INT64_MIN || sampleTsMs - _lastAuxMs >= 95);
    if (!doSeg && !doPose && !doAux) return;

    bool expected = false;
    if (!_inFlight.compare_exchange_strong(expected, true)) return;

    if (doSeg) _lastSegMs = sampleTsMs;
    if (doPose) _lastPoseMs = sampleTsMs;
    if (doAux) _lastAuxMs = sampleTsMs;

    CVPixelBufferRetain(pixelBuffer);
    const int reqQuality = self.qualityLevel;
    void (^copiedAux)(const uint8_t *, int, int) = [auxBlock copy];

    dispatch_async(_queue, ^{
        @autoreleasepool {
            [self runOnQueueWithBuffer:pixelBuffer
                       sampleTimestamp:sampleTsMs
                                 doSeg:doSeg
                                doPose:doPose
                                 doAux:doAux
                          qualityLevel:reqQuality
                       onAuxRgbaSample:copiedAux];
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
                qualityLevel:(int)qualityLevel
             onAuxRgbaSample:(nullable void (^)(const uint8_t *rgba, int width, int height))auxBlock {
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
                _segReq.qualityLevel = (qualityLevel <= 0)
                    ? VNGeneratePersonSegmentationRequestQualityLevelFast
                    : VNGeneratePersonSegmentationRequestQualityLevelBalanced;
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
