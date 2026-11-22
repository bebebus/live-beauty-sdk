#import <Flutter/Flutter.h>
#import "FLTVideoEffects.h"

NS_ASSUME_NONNULL_BEGIN

/// Exposes the shared `FLTVideoEffects` processor so host iOS camera / WebRTC / RTC
/// capture pipelines inside the same Flutter app can call `copyProcessedSample:`.
@interface LiveBeautySdkPlugin : NSObject <FlutterPlugin, FlutterStreamHandler>

@property(class, nonatomic, readonly, nullable) FLTVideoEffects *sharedEffects;

@end

NS_ASSUME_NONNULL_END
