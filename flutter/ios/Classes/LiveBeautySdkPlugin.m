#import "LiveBeautySdkPlugin.h"

static FLTVideoEffects *_sharedEffects = nil;

@implementation LiveBeautySdkPlugin {
    FlutterEventSink _eventSink;
}

+ (FLTVideoEffects *)sharedEffects {
    return _sharedEffects;
}

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar> *)registrar {
    LiveBeautySdkPlugin *instance = [[LiveBeautySdkPlugin alloc] init];
    FlutterMethodChannel *methodChannel =
        [FlutterMethodChannel methodChannelWithName:@"io.bebebus.livebeauty/controller"
                                    binaryMessenger:[registrar messenger]];
    [registrar addMethodCallDelegate:instance channel:methodChannel];

    FlutterEventChannel *eventChannel =
        [FlutterEventChannel eventChannelWithName:@"io.bebebus.livebeauty/events"
                                  binaryMessenger:[registrar messenger]];
    [eventChannel setStreamHandler:instance];
}

- (FlutterError *)onListenWithArguments:(id)arguments eventSink:(FlutterEventSink)events {
    _eventSink = events;
    return nil;
}

- (FlutterError *)onCancelWithArguments:(id)arguments {
    _eventSink = nil;
    return nil;
}

- (void)emitEvent:(NSDictionary *)payload {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (self->_eventSink) {
            self->_eventSink(payload);
        }
    });
}

- (FLTVideoEffects *)ensureEffects {
    if (!_sharedEffects) {
        __weak typeof(self) weakSelf = self;
        _sharedEffects = [[FLTVideoEffects alloc] initWithFailure:^{
            [weakSelf emitEvent:@{@"type": @"engineFailed"}];
        } faceFailure:^{
            [weakSelf emitEvent:@{@"type": @"faceTrackingFailed"}];
        }];
    }
    return _sharedEffects;
}

- (void)handleMethodCall:(FlutterMethodCall *)call result:(FlutterResult)result {
    if ([@"initialize" isEqualToString:call.method] || [@"setConfig" isEqualToString:call.method]) {
        if (![call.arguments isKindOfClass:[NSDictionary class]]) {
            result([FlutterError errorWithCode:@"invalid_args"
                                       message:@"Expected configuration dictionary."
                                       details:nil]);
            return;
        }
        NSDictionary *dict = (NSDictionary *)call.arguments;
        if (![FLTVideoEffects validConfiguration:dict]) {
            result([FlutterError errorWithCode:@"invalid_effects"
                                       message:@"Configuration violated native bounds."
                                       details:nil]);
            return;
        }
        [[self ensureEffects] setConfiguration:dict];
        result(nil);
    } else if ([@"resetTracking" isEqualToString:call.method]) {
        [_sharedEffects resetFaceTracking];
        result(nil);
    } else if ([@"dispose" isEqualToString:call.method]) {
        _sharedEffects = nil;
        result(nil);
    } else {
        result(FlutterMethodNotImplemented);
    }
}

@end
