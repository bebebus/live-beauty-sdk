#import "FLTFaceGeometry.h"
/// MediaPipe 人脸关键点跟踪：后台单任务、100 ms 节流；reset 与尺寸/启用状态变化都会使代际号 +1。
@interface FLTFaceLandmarkTracker : NSObject <FLTFaceTracking>
/// 模型加载或推理失败时在后台队列调用，本实例至多一次，与代际号和启用状态无关（与 Android 一致）；
/// 之后本实例不再提交推理、只返回 nil 几何（规格 1.3：本次预览内不重试）。
@property(atomic, copy) void (^onFailure)(void);
@end
