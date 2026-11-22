#import "FLTFaceGeometrySmoother.h"
#import <math.h>
#import <string.h>

// 常量与 beauty_ref.SMOOTHER 一致。
static const float FLTSmootherFcMin = 0.6f, FLTSmootherK = 60, FLTSmootherDead = 0.012f, FLTSmootherRxMin = 1e-3f;
static const double FLTSmootherInMs = 150, FLTSmootherHoldMs = 150, FLTSmootherOutMs = 250;
static const double FLTSmootherInterpMinMs = 66, FLTSmootherInterpMaxMs = 150, FLTSmootherInterpInitMs = 100;
static const double FLTSmootherDtClampMs = 100, FLTSmootherTeMinMs = 33, FLTSmootherTeMaxMs = 500;
static const int FLTSmootherRxIndex = 24, FLTSmootherAxisIndex = 26;

// 各分量的偏差类别（beauty_ref.STATE_KINDS）：x 类 ×1，纹理 y 类 ÷aspect，axis × oval.rx，n 不参与。
static const char FLTSmootherKinds[FLTFaceGeometryStateSize + 1] =
    "xyxyxyxyxyxyxyxy" "xyxxyx" "xyxx" "aa" "xyxx" "xyxxxyxx" "n";

@implementation FLTFaceGeometrySmoother {
    float _f[FLTFaceGeometryStateSize];     // 滤波后的样本
    float _prev[FLTFaceGeometryStateSize];  // 上个样本到达时的输出
    float _x[FLTFaceGeometryStateSize];     // 上次输出
    BOOL _tracking;                         // f、prev、x、ts 是否有效
    double _ts;                             // 上个已消费样本的采样时间戳
    double _arrival;                        // 该样本到达时的渲染时间
    double _interval;                       // 插值时长 T
    double _missing;
    double _last;
    BOOL _hasLast;
    float _presence;
}
- (instancetype)initWithAspect:(float)aspect {
    if ((self = [super init])) {
        _aspect = aspect;
        [self reset];
    }
    return self;
}
- (float)presence { return _presence; }
- (void)reset {
    _tracking = NO;
    _ts = 0;
    _arrival = 0;
    _interval = FLTSmootherInterpInitMs;
    _presence = 0;
    _missing = 0;
    _hasLast = NO;
}
- (BOOL)updateTarget:(const float *)target sampleTimestamp:(double)sampleTimestampMs
                 now:(double)nowMs state:(float *)state {
    double dt;
    if (!_hasLast) {
        dt = 0;
    } else {
        dt = fmin(fmax(nowMs - _last, 0), FLTSmootherDtClampMs);
        if (dt == 0) return [self output:state];
    }
    _last = nowMs;
    _hasLast = YES;
    if (target) {
        _missing = 0;
        if (!_tracking) {
            memcpy(_f, target, sizeof(_f));
            memcpy(_prev, target, sizeof(_prev));
            memcpy(_x, target, sizeof(_x));
            _ts = sampleTimestampMs; _arrival = nowMs; _interval = FLTSmootherInterpInitMs;
            _tracking = YES;
        } else if (sampleTimestampMs != _ts) {
            double te = fmin(fmax(sampleTimestampMs - _ts, FLTSmootherTeMinMs), FLTSmootherTeMaxMs);
            memcpy(_prev, _x, sizeof(_prev));
            float rx = fmaxf(_f[FLTSmootherRxIndex], FLTSmootherRxMin);
            float deviation = 0;
            for (int i = 0; i < FLTFaceGeometryStateSize; i++) {
                char kind = FLTSmootherKinds[i];
                float weight = kind == 'x' ? 1 : kind == 'y' ? 1 / _aspect : kind == 'a' ? rx : 0;
                deviation = fmaxf(deviation, fabsf(target[i] - _f[i]) * weight);
            }
            float s = deviation / rx;
            float fc = FLTSmootherFcMin + FLTSmootherK * fmaxf(0, s - FLTSmootherDead);
            float alpha = 1 - expf(-2 * (float)M_PI * fc * (float)te / 1000);
            for (int i = 0; i < FLTFaceGeometryStateSize; i++) _f[i] += alpha * (target[i] - _f[i]);
            _interval = fmin(fmax(0.5 * _interval + 0.5 * te, FLTSmootherInterpMinMs), FLTSmootherInterpMaxMs);
            _ts = sampleTimestampMs; _arrival = nowMs;
        }
        _presence = fminf(1, _presence + (float)(dt / FLTSmootherInMs));
    } else {
        _missing += dt;
        if (_missing > FLTSmootherHoldMs) _presence = fmaxf(0, _presence - (float)(dt / FLTSmootherOutMs));
        if (_presence <= 0) {
            _tracking = NO;
            _missing = 0;
        }
    }
    if (_tracking) {
        float g = (float)fmin(1, fmax(0, (nowMs - _arrival + dt) / _interval));
        for (int i = 0; i < FLTFaceGeometryStateSize; i++) _x[i] = _prev[i] + g * (_f[i] - _prev[i]);
        int i = FLTSmootherAxisIndex;
        float n = hypotf(_x[i], _x[i + 1]);
        if (n > 1e-6f) { _x[i] /= n; _x[i + 1] /= n; }
    }
    return [self output:state];
}
- (BOOL)output:(float *)state {
    if (!_tracking) return NO;
    memcpy(state, _x, sizeof(_x));
    return YES;
}
@end
