#import <Foundation/Foundation.h>
#import "FLTFaceGeometry.h"

/// 人脸几何时间平滑（规格第 3 节），逐行移植 reference/beauty_ref.py 的 FinalSmoother，状态为 float32。
/// - 自适应低通只在新样本（采样时间戳变化）到达时执行一次；每帧在上个输出与滤波值之间按实测样本间隔线性插值。
/// - presence：有目标时 150 ms 渐入；目标消失后先保持 150 ms，再 250 ms 渐出，到 0 清空几何。
/// 只在采集队列使用，非线程安全；其他线程通过 tracker 的代际号间接触发 reset。
@interface FLTFaceGeometrySmoother : NSObject
- (instancetype)initWithAspect:(float)aspect;
/// 宽 / 高；仅用于偏差中纹理 y 分量的单位换算。修改后应 reset。
@property(nonatomic) float aspect;
@property(nonatomic, readonly) float presence;
- (void)reset;
/// target 为 FLTFaceGeometryStateSize 个 float，NULL 表示本帧没有目标（sampleTimestampMs 此时不用）。
/// 返回 YES 时把平滑后的几何写入 state；返回 NO 表示当前没有几何。同一 nowMs 可重复调用，结果不变。
- (BOOL)updateTarget:(const float *)target sampleTimestamp:(double)sampleTimestampMs
                 now:(double)nowMs state:(float *)state;
@end
