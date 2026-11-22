#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <CoreVideo/CoreVideo.h>

/// 平滑器状态向量长度：cheeks[0..15]、eyes[16..21]、oval[22..25]、axis[26..27]、mouth[28..31]、eyeMasks[32..39]、warpWeight[40]。
#define FLTFaceGeometryStateSize 41

/// 坐标归一化到原始纹理，y向上；仅包含局部几何，不含图像/身份。
/// 中心为纹理坐标，半径与眼宽为度量值（p' = (x, y / aspect)，x 单位），axis 为度量空间单位向量 u，v = (-u.y, u.x) 指向额头。
/// 数学与 reference/beauty_ref.py 的 full_geometry() 逐行一致。
@interface FLTFaceGeometry : NSObject
@property(nonatomic, copy) NSArray<CIVector *> *cheeks;
@property(nonatomic, copy) NSArray<CIVector *> *eyes;
@property(nonatomic, strong) CIVector *oval;
@property(nonatomic, strong) CIVector *axis;
@property(nonatomic, strong) CIVector *mouth;
@property(nonatomic, copy) NSArray<CIVector *> *eyeMasks;
/// 1 − smoothstep(0.24, 0.30, 侧脸比)，只乘在塑形强度上。
@property(nonatomic) CGFloat warpWeight;
+ (instancetype)fromMesh:(NSArray<NSArray<NSNumber *> *> *)points aspect:(CGFloat)aspect;
/// 按平滑器布局写出 FLTFaceGeometryStateSize 个 float。
- (void)getState:(float *)state;
@end
@protocol FLTFaceTracking <NSObject>
/// 一次加锁取出当前几何（超过 250 ms 视为过期，返回 nil）、代际号与该几何的采样时间戳（毫秒，CACurrentMediaTime 时钟）。
/// 代际号在每次 reset（含启用状态或尺寸变化）后 +1；调用方见到变化即在自己的队列上重置平滑器，不得跨线程直接重置。
- (FLTFaceGeometry *)geometryForPixelBuffer:(CVPixelBufferRef)buffer enabled:(BOOL)enabled
                                 generation:(NSUInteger *)generation sampleTimestamp:(NSInteger *)sampleTimestamp;
- (void)reset;
@end
