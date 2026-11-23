import 'dart:async';

import 'package:flutter/services.dart';

import 'beauty_config.dart';
import 'beauty_extensions.dart';

/// Asynchronous runtime events emitted by the native Live Beauty engine.
sealed class LiveBeautyEvent {
  const LiveBeautyEvent();

  static LiveBeautyEvent? tryParse(Object? raw) {
    if (raw is! Map<Object?, Object?>) {
      return null;
    }
    final String? type = raw['type'] as String?;
    switch (type) {
      case 'engineFailed':
        return const LiveBeautyEngineFailed();
      case 'faceTrackingFailed':
        return const LiveBeautyFaceTrackingFailed();
      case 'loadReduced':
        final int level = (raw['level'] as num?)?.toInt() ?? 0;
        return LiveBeautyLoadReduced(level);
      case 'legsHiddenChanged':
        final bool hidden = raw['hidden'] as bool? ?? false;
        return LiveBeautyLegsHiddenChanged(hidden);
      case 'lowLightChanged':
        final bool lowLight = raw['lowLight'] as bool? ?? false;
        return LiveBeautyLowLightChanged(lowLight);
      default:
        return null;
    }
  }
}

/// Emitted when OpenGL/CoreImage shader compilation fails and the engine falls
/// back to zero-copy camera pass-through.
final class LiveBeautyEngineFailed extends LiveBeautyEvent {
  const LiveBeautyEngineFailed();
}

/// Emitted when MediaPipe model initialization fails; skin/color passes stay active.
final class LiveBeautyFaceTrackingFailed extends LiveBeautyEvent {
  const LiveBeautyFaceTrackingFailed();
}

/// Emitted when the thermal/FPS governor steps down rendering load (`level` 1..3).
final class LiveBeautyLoadReduced extends LiveBeautyEvent {
  const LiveBeautyLoadReduced(this.level);

  final int level;
}

/// Emitted when lower-body landmarks become occluded or re-enter the frame.
final class LiveBeautyLegsHiddenChanged extends LiveBeautyEvent {
  const LiveBeautyLegsHiddenChanged(this.hidden);

  final bool hidden;
}

/// Emitted when ambient camera luma enters or exits low-light compensation mode.
final class LiveBeautyLowLightChanged extends LiveBeautyEvent {
  const LiveBeautyLowLightChanged(this.lowLight);

  final bool lowLight;
}

/// Controls the native `LiveBeautyProcessor` (Android) / `FLTVideoEffects` (iOS)
/// shared instance registered in `LiveBeautyEngineRegistry`.
class LiveBeautyController {
  LiveBeautyController({
    MethodChannel? methodChannel,
    EventChannel? eventChannel,
  })  : _methodChannel =
            methodChannel ?? const MethodChannel(defaultMethodChannelName),
        _eventChannel =
            eventChannel ?? const EventChannel(defaultEventChannelName);

  static const String defaultMethodChannelName =
      'io.bebebus.livebeauty/controller';
  static const String defaultEventChannelName = 'io.bebebus.livebeauty/events';

  final MethodChannel _methodChannel;
  final EventChannel _eventChannel;

  LiveBeautyConfig _config = LiveBeautyConfig.original;
  Stream<LiveBeautyEvent>? _events;
  bool _disposed = false;

  /// Latest validated beauty configuration applied through this controller.
  LiveBeautyConfig get config => _config;

  /// Broadcast stream of native thermal/tracking/lighting telemetry events.
  Stream<LiveBeautyEvent> get events {
    return _events ??= _eventChannel
        .receiveBroadcastStream()
        .map(LiveBeautyEvent.tryParse)
        .where((LiveBeautyEvent? event) => event != null)
        .cast<LiveBeautyEvent>();
  }

  /// Initializes the native shared processor with [initialConfig].
  Future<void> initialize({
    LiveBeautyConfig initialConfig = LiveBeautyConfig.naturalStudio,
  }) async {
    _ensureNotDisposed();
    initialConfig.validate();
    await _methodChannel.invokeMethod<void>('initialize', initialConfig.toMap());
    _config = initialConfig;
  }

  /// Atomically updates all beauty, reshape, makeup, LUT, and body parameters.
  Future<void> setConfig(LiveBeautyConfig config) async {
    _ensureNotDisposed();
    config.validate();
    await _methodChannel.invokeMethod<void>('setConfig', config.toMap());
    _config = config;
  }

  /// Resets temporal landmark/mask smoothers (call after switching front/back camera).
  Future<void> resetTracking() async {
    _ensureNotDisposed();
    await _methodChannel.invokeMethod<void>('resetTracking');
  }

  /// Configures dual-face co-streaming (`maxFaces` 1..2) and zero-lag `FrameDelayRing` (0..2).
  Future<void> setPipelineOptions(LiveBeautyPipelineOptions options) async {
    _ensureNotDisposed();
    options.validate();
    await _methodChannel.invokeMethod<void>('setPipelineOptions', options.toMap());
  }

  /// Configures green-screen-free virtual background replacement using a host GL texture ID.
  Future<void> setVirtualBackgroundTexture({
    required int glTextureId,
    double aspect = 0.0,
    LiveBeautyVirtualBackgroundMode mode =
        LiveBeautyVirtualBackgroundMode.virtualImage,
    double strength = 1.0,
  }) async {
    _ensureNotDisposed();
    if (strength < 0.0 || strength > 1.0) {
      throw RangeError('strength must be in [0.0, 1.0], got $strength');
    }
    await _methodChannel.invokeMethod<void>(
      'setVirtualBackgroundTexture',
      <String, Object>{
        'glTextureId': glTextureId,
        'aspect': aspect,
        'mode': mode.wireName,
        'strength': strength,
      },
    );
  }

  /// Sets or clears one of up to 4 face-anchored 2D AR sticker slots (`slot` 0..3).
  Future<void> setSticker(int slot, LiveBeautySticker? sticker) async {
    _ensureNotDisposed();
    if (slot < 0 || slot > 3) {
      throw RangeError('slot must be in [0, 3], got $slot');
    }
    sticker?.validate();
    await _methodChannel.invokeMethod<void>(
      'setSticker',
      <String, Object?>{
        'slot': slot,
        'sticker': sticker?.toMap(),
      },
    );
  }

  /// Clears all active 2D AR sticker layers.
  Future<void> clearStickers() async {
    _ensureNotDisposed();
    await _methodChannel.invokeMethod<void>('clearStickers');
  }

  /// Reads real-time performance & tracking telemetry from the native C++17 core.
  Future<LiveBeautyPerfStats> getPerfStats() async {
    _ensureNotDisposed();
    final Map<Object?, Object?>? raw = await _methodChannel
        .invokeMethod<Map<Object?, Object?>>('getPerfStats');
    return LiveBeautyPerfStats.fromMap(raw);
  }

  /// Releases the controller instance on the Dart side and notifies native registry.
  Future<void> dispose() async {
    if (_disposed) {
      return;
    }
    _disposed = true;
    await _methodChannel.invokeMethod<void>('dispose');
  }

  void _ensureNotDisposed() {
    if (_disposed) {
      throw StateError('LiveBeautyController has already been disposed.');
    }
  }
}
