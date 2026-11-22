#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <CoreMedia/CoreMedia.h>
#import "FLTFaceGeometry.h"

/// CIKL 片段（规格 4.2 代码组织）。生产内核由片段拼接，像素测试用同一片段拼调试内核。
/// COMMON：skinWeight、lift、ellipseE；WARP：cheek、eye；MASKS：maskParts、faceMasks → (face, excl)；
/// COMPOSITE：beautyColor(I, mean, var, avg4, face, excl, p0, p1, style)。
/// 几何一律用像素空间（度量空间 × w，各向同性，y 向上，与 destCoord() 一致）。
/// 官方在线体验与实机测试站点地址。
FOUNDATION_EXPORT NSString *const FLTVideoEffectsDemoSiteURL;
FOUNDATION_EXPORT NSString *const FLTVideoEffectsKernelCommon;
FOUNDATION_EXPORT NSString *const FLTVideoEffectsKernelWarp;
FOUNDATION_EXPORT NSString *const FLTVideoEffectsKernelMasks;
FOUNDATION_EXPORT NSString *const FLTVideoEffectsKernelComposite;
/// 预处理高斯的 CIConvolution9Horizontal/Vertical 权重（生产实际使用的数组），须与共享向量 kernel.iosConvolution9 逐项一致。
FOUNDATION_EXPORT const CGFloat FLTVideoEffectsGaussianWeights[9];

/// 由串行采集队列持有；输出仍使用输入帧的媒体时钟。
@interface FLTVideoEffects : NSObject
@property(atomic, copy) NSDictionary *configuration;
@property(nonatomic, strong) id<FLTFaceTracking> faceTracker;
/// 渲染故障（永久故障或连续 30 帧瞬时故障）回原画时调用一次；此时 configuration 已清空。
@property(nonatomic, copy) void (^onFailure)(void);
/// 人脸模型加载或推理失败后置位（任意线程写，采集队列读），本实例内不复位。
/// 置位后不再跟踪人脸：塑形与人脸遮罩停用，调色与磨皮照常。
@property(atomic) BOOL faceTrackingFailed;
+ (BOOL)validConfiguration:(NSDictionary *)values;
/// 返回 +1 引用的输出帧：
/// - 直通（未启用、对比原画、中性参数且无塑形几何）返回原 sample 本身。
/// - 返回 NULL 表示瞬时故障（输出池超过分配阈值、样本或格式描述创建失败、构图异常），该帧被丢弃：
///   调用方必须跳过这一帧（预览、推流、录制都不送），且不得对 NULL 调用 CFRelease。配置保留。
/// - 连续 30 帧瞬时故障，或永久故障（内核不可用、首次自检失败、缓冲池创建失败）：清空配置、调用 onFailure，并返回原 sample。
- (CMSampleBufferRef)copyProcessedSample:(CMSampleBufferRef)sample CF_RETURNS_RETAINED;
/// 构图入口（首次自检与 macOS 像素测试共用）。image 为未做色彩管理、原点在 (0, 0) 的源图；
/// state 为 FLTFaceGeometryStateSize 维平滑后几何（NULL 表示无人脸），presence 为平滑器输出。内核不可用时返回 nil。
+ (CIImage *)beautyImage:(CIImage *)image configuration:(NSDictionary *)config
                geometry:(const float *)state presence:(float)presence;
/// 1/4 分辨率预处理并放大回全幅（规格 4.1）：mean 为高斯均值，variance 的各通道为方差。内核不可用时返回 NO。
+ (BOOL)prepassForImage:(CIImage *)image mean:(CIImage **)mean variance:(CIImage **)variance;
@end
