import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

bool _licensesRegistered = false;

/// Registers third-party open-source attributions bundled with `live_beauty_sdk`
/// (Google MediaPipe & PixPark GPUPixel, both Apache License 2.0) into Flutter's
/// [LicenseRegistry]. Safe to call multiple times; registration happens once.
void registerLiveBeautyLicenses({AssetBundle? bundle}) {
  if (_licensesRegistered) {
    return;
  }
  _licensesRegistered = true;
  final AssetBundle effectiveBundle = bundle ?? rootBundle;
  LicenseRegistry.addLicense(() async* {
    final String mediaPipe = await effectiveBundle.loadString(
      'packages/live_beauty_sdk/licenses/MediaPipe-LICENSE.txt',
    );
    yield LicenseEntryWithLineBreaks(
      const <String>['com.google.mediapipe', 'MediaPipe Tasks Vision'],
      mediaPipe,
    );
    final String gpuPixel = await effectiveBundle.loadString(
      'packages/live_beauty_sdk/licenses/GPUPixel-LICENSE.txt',
    );
    yield LicenseEntryWithLineBreaks(
      const <String>['gpupixel'],
      gpuPixel,
    );
  });
}

@visibleForTesting
void debugResetLiveBeautyLicensesForTest() {
  _licensesRegistered = false;
}
