import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:live_beauty_sdk/live_beauty_sdk.dart';

class _FakeAssetBundle extends CachingAssetBundle {
  @override
  Future<ByteData> load(String key) => throw UnimplementedError();

  @override
  Future<String> loadString(String key, {bool cache = true}) async {
    return 'Apache License 2.0 ($key)';
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('LiveBeautyConfig', () {
    test('studio presets pass strict native validation and wire round-trip', () {
      const List<LiveBeautyConfig> presets = <LiveBeautyConfig>[
        LiveBeautyConfig.original,
        LiveBeautyConfig.naturalStudio,
        LiveBeautyConfig.clearColdWhite,
        LiveBeautyConfig.koreanWaterGlow,
        LiveBeautyConfig.mensClean,
      ];
      for (final LiveBeautyConfig preset in presets) {
        expect(() => preset.validate(), returnsNormally);
        final Map<String, Object> encoded = preset.toMap();
        final LiveBeautyConfig decoded = LiveBeautyConfig.fromMap(encoded);
        expect(decoded, equals(preset));
        expect(decoded.hashCode, equals(preset.hashCode));
      }
      expect(LiveBeautyConfig.original.enabled, isFalse);
      expect(LiveBeautyConfig.original.hasFaceReshape, isFalse);
      expect(LiveBeautyConfig.original.hasMakeup, isFalse);
      expect(LiveBeautyConfig.naturalStudio.hasFaceReshape, isTrue);
      expect(LiveBeautyConfig.koreanWaterGlow.hasMakeup, isTrue);
    });

    test('validates all 10 built-in 3D LUT asset URIs and rejects unsafe paths', () {
      expect(LiveBeautyConfig.builtInLutIds, hasLength(10));
      for (final String id in LiveBeautyConfig.builtInLutIds) {
        final String uri = LiveBeautyConfig.assetLutUrl(id);
        expect(LiveBeautyConfig.isValidLutUrl(uri), isTrue);
        final File bundled =
            File('../android/live-beauty/src/main/assets/luts/$id.png');
        expect(bundled.existsSync(), isTrue, reason: 'Missing bundled LUT $id');
      }
      expect(LiveBeautyConfig.isValidLutUrl(''), isTrue);
      expect(
        LiveBeautyConfig.isValidLutUrl('https://cdn.bebebus.io/luts/film.png'),
        isTrue,
      );
      for (final String invalid in <String>[
        'http://cdn.bebebus.io/luts/film.png',
        'asset://luts/../secret.png',
        'asset://luts/UPPERCASE.png',
        'file:///etc/passwd',
        'https://bad space.example/lut.png',
      ]) {
        expect(LiveBeautyConfig.isValidLutUrl(invalid), isFalse);
        expect(
          () => LiveBeautyConfig(lutUrl: invalid).validate(),
          throwsArgumentError,
        );
      }
    });

    test('supports bidirectional reshape [-1, 1] and rejects out-of-range or unknown enums', () {
      const LiveBeautyConfig custom = LiveBeautyConfig(
        chinShape: -0.45,
        foreheadShape: 0.35,
        eyeDistance: -0.25,
        eyeCorner: 0.30,
        noseLength: -0.20,
        mouthSize: 0.15,
        waistSlim: 0.40,
        lipstick: 0.55,
        lipstickColor: 'cherry_red',
        lipstickFinish: 'glossy',
      );
      expect(() => custom.validate(), returnsNormally);
      expect(custom.hasFaceReshape, isTrue);
      expect(custom.hasBodyReshape, isTrue);
      expect(custom.hasMakeup, isTrue);

      expect(
        () => custom.copyWith(smoothing: 1.01).validate(),
        throwsRangeError,
      );
      expect(
        () => custom.copyWith(chinShape: -1.05).validate(),
        throwsRangeError,
      );
      expect(
        () => custom.copyWith(lipstickColor: 'neon_green').validate(),
        throwsArgumentError,
      );
    });
  });

  group('LiveBeautyController & Licenses', () {
    const MethodChannel channel =
        MethodChannel(LiveBeautyController.defaultMethodChannelName);
    final List<MethodCall> calls = <MethodCall>[];

    setUp(() {
      calls.clear();
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        calls.add(call);
        return null;
      });
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('dispatches initialize, setConfig, resetTracking, and dispose over MethodChannel', () async {
      final LiveBeautyController controller = LiveBeautyController();
      await controller.initialize();
      expect(controller.config, equals(LiveBeautyConfig.naturalStudio));

      await controller.setConfig(LiveBeautyConfig.koreanWaterGlow);
      expect(controller.config, equals(LiveBeautyConfig.koreanWaterGlow));

      await controller.resetTracking();
      await controller.dispose();

      expect(
        calls.map((MethodCall c) => c.method).toList(),
        equals(<String>['initialize', 'setConfig', 'resetTracking', 'dispose']),
      );
      expect(() => controller.resetTracking(), throwsStateError);
    });

    test('parses native telemetry events cleanly', () {
      expect(
        LiveBeautyEvent.tryParse(<Object?, Object?>{'type': 'engineFailed'}),
        isA<LiveBeautyEngineFailed>(),
      );
      final LiveBeautyEvent? load = LiveBeautyEvent.tryParse(
        <Object?, Object?>{'type': 'loadReduced', 'level': 2},
      );
      expect(load, isA<LiveBeautyLoadReduced>());
      expect((load! as LiveBeautyLoadReduced).level, equals(2));
      expect(
        LiveBeautyEvent.tryParse(<Object?, Object?>{'type': 'lowLightChanged', 'lowLight': true}),
        isA<LiveBeautyLowLightChanged>(),
      );
    });

    test('registers MediaPipe and GPUPixel Apache-2.0 licenses once', () async {
      debugResetLiveBeautyLicensesForTest();
      registerLiveBeautyLicenses(bundle: _FakeAssetBundle());
      registerLiveBeautyLicenses(bundle: _FakeAssetBundle());
      final List<LicenseEntry> entries = await LicenseRegistry.licenses.toList();
      expect(
        entries.any((LicenseEntry e) => e.packages.contains('gpupixel')),
        isTrue,
      );
      expect(
        entries.any(
          (LicenseEntry e) => e.packages.contains('com.google.mediapipe'),
        ),
        isTrue,
      );
    });
  });
}
