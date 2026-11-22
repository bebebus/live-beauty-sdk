# Bebebus Live Beauty SDK

> [中文](./README.md) | English

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Live Demo: dh.video](https://img.shields.io/badge/Live_Demo-dh.video-FF4D6D?logo=googlechrome&logoColor=white)](https://dh.video)
[![Android API](https://img.shields.io/badge/Android-API_24%2B-3DDC84?logo=android&logoColor=white)](./android)
[![iOS](https://img.shields.io/badge/iOS-15.0%2B-000000?logo=apple&logoColor=white)](./ios)
[![Flutter](https://img.shields.io/badge/Flutter-3.22%2B-02569B?logo=flutter&logoColor=white)](./flutter)

**Bebebus Live Beauty SDK** is a production-grade, royalty-free, real-time on-device GPU beauty and face/body retouching engine for **Android (OpenGL ES 2.0)**, **iOS (CoreImage + Metal)**, and **Flutter**.

Designed to be **100% transport- and RTC-agnostic**, it requires zero license servers or online activation, bundles all offline MediaPipe vision models, 10 studio `64³` 3D LUTs, and 5 high-precision UV makeup atlases out of the box, and plugs cleanly into **WebRTC, Agora, Zego, Tencent TRTC, CameraX, MediaCodec, or `AVCaptureSession`** pipelines.

### Live Demo & Experience Testbed

- **Official Live Demo Site**: [https://dh.video](https://dh.video)
- Powered in production by **Bebebus Live Beauty SDK** — visit to preview real-time live broadcast beauty rendering or download the client app for hands-on testing of **478-pt diffeomorphic reshape, 1080p guided skin smoothing, beard hair edge preservation, 6-layer UV makeup, and `64³` 3D LUTs**.

---

## Highlights

- **Zero Background Warp · Zero Contour Ghosting · Symmetric 478-pt $C^\infty$ Mesh Reshape**
  - Expands MediaPipe's 478 face landmarks with a dual-ring pinned outer boundary (`548` vertices / `1086` triangles) that locks background pixels to exact zero displacement.
  - Combines a **2D Pure-Shear Plateau contour field**, **Boltzmann-weighted `SoftMax` half-width estimation**, and **minimum-relax Jacobian safety equalization** to eliminate bent doorframes, yaw-induced left/right facial asymmetry, and grazing cheek double-contours.
  - Covers **15 facial & cranial micro-reshape controls**: slim face, V-face, narrow face, small face, cheekbone slim, **high cranial top (`cranialTop`)**, chin length, forehead height, eye enlarge, eye distance, eye corner, nose slim, nose length, mouth size, and smile lift.
- **Dual-Band Creamy Skin Guided Smoothing & 468-pt Anatomical Sub-Mesh Retouching**
  - Combines a `1/3` edge-preserving guided filter with a `1/9` wide-scale skin homogenizer and `2–4 px` pore micro-texture restoration — melting away tear-trough dullness and cheek blotches while keeping 3D bone shadows, eyebrows, and men's beard edges crisp.
  - **In-Place 468-pt Sub-Mesh Retouching (Zero Extra 1080p FBO)**: Tear-trough concealer that preserves natural aegyo-sal fullness, curved nasolabial fold softening, 28-triangle eye brightening, and 18-triangle inner-mouth teeth whitening that physically never bleaches glossy lips.
- **6-Layer UV Real-Time Makeup & `64³` 3D LUT Grading**
  - Velvet & water-glow specular lipsticks (6 shades), 3 blush styles, eyebrows, eyeshadow, eyeliner, aegyo-sal, 3D nose/cheek contour & highlight, iris-masked contact lenses, and dual catchlights.
  - Ships with **10 built-in `512×512` (`64³`) studio 3D LUTs** (`asset://luts/<id>.png`) plus disk-cached `https://` remote LUT loading.
- **5-Axis Body Reshape & Portrait Bokeh**
  - Pose-driven waist slimming, bust enhancement, leg lengthening, shoulder slimming, neck lengthening, and real-time background blur.
- **Live-Streaming Resilience**
  - **`FrameDelayRing` + Adaptive Far-Field Face ROI + GLES 3.0 Dual-PBO Readback**: Automatically zooms a `1:1` `256×256` ROI onto distant faces (`2.5–3.5×` higher effective DPI in half-body/dance streams) and uses asynchronous DMA `GL_PIXEL_PACK_BUFFER` readback on OpenGL ES 3.0+ contexts.
  - **`GpuDelegatePolicy`** detects native TFLite GPU driver crashes across process restarts and automatically falls back to CPU.
  - **`BeautyLoadGovernor` & `BitrateEffectsTuning`** step down GPU passes under thermal pressure and adaptively boost temporal denoising when encoder bitrate drops.

---

## Repository Layout

```text
live-beauty-sdk/
├── android/                  # Standalone Android Gradle library (io.bebebus:live-beauty, minSdk 24)
│   └── live-beauty/          # OpenGL ES 2.0 pipeline, async MediaPipe trackers, bundled assets & 34 JVM test suites
├── ios/                      # Standalone iOS CocoaPods library (LiveBeautySDK.podspec, iOS 15.0+)
│   ├── Classes/              # FLTVideoEffects / FLTFaceGeometry / FLTFaceGeometrySmoother / FLTFaceLandmarkTracker
│   ├── Resources/            # Bundled offline face_landmarker.task model
│   └── Tests/                # macOS / iOS CoreImage pixel & smoother regression suite
├── flutter/                  # Flutter plugin wrapper (live_beauty_sdk)
│   ├── lib/                  # LiveBeautyConfig (40+ params & 5 presets), LiveBeautyController, LicenseRegistry helper
│   └── test/                 # Dart unit tests for wire round-trip, whitelists & MethodChannel lifecycle
├── reference/                # Python mathematical reference (beauty_ref.py), golden vectors & asset generators
├── _tooling/                 # 41-shader headless WebGL1 gate, SHA256 model verifier, sanitizer & full test runner
└── ARCHITECTURE.md           # Deep-dive pipeline topology & diffeomorphic mesh documentation
```

---

## Quick Start

### 1. Android Native (WebRTC / Agora / Zego / TRTC / CameraX)

```kotlin
import io.bebebus.livebeauty.LiveBeautyPresets
import io.bebebus.livebeauty.LiveBeautyProcessor

// 1. Instantiate (thread-safe config updates from any thread)
val beautyProcessor = LiveBeautyProcessor(appContext = context.applicationContext)
beautyProcessor.updateConfig(
    LiveBeautyPresets.koreanWaterGlow.copy(
        faceSlim = 0.42f,
        vFace = 0.30f,
        skinTexture = 0.50f,
    )
)

// 2. Initialize on your OpenGL ES rendering thread
beautyProcessor.onGlCreated(context)

// 3. Process upright RGBA GL_TEXTURE_2D per frame -> returns processed GL_TEXTURE_2D
val outputTexId = beautyProcessor.process(
    inputTexture = cameraRgbaTex2D,
    width = frameWidth,
    height = frameHeight,
    timestampNs = surfaceTimestampNs,
)

// 4. Reset temporal smoothers when flipping front/rear camera
beautyProcessor.resetTracking()

// 5. Release before tearing down the GL context
beautyProcessor.onGlReleased(contextAlive = true)
```

### 2. iOS Native (CocoaPods / `AVCaptureSession` / Custom Video Capturer)

```ruby
pod 'LiveBeautySDK', :git => 'https://github.com/bebebus/live-beauty-sdk.git'
```

```objc
#import <LiveBeautySDK/FLTVideoEffects.h>

self.beautyEffects = [[FLTVideoEffects alloc] initWithFailure:^{
    NSLog(@"Beauty kernel fallback to pass-through");
} faceFailure:^{
    NSLog(@"Face tracker unavailable, skin/color still active");
}];

[self.beautyEffects setConfiguration:@{
    @"enabled": @YES,
    @"smoothing": @0.55,
    @"skinOnly": @YES,
    @"whiten": @0.32,
    @"rosy": @0.18,
    @"faceSlim": @0.38,
    @"eyeEnlarge": @0.28,
    @"filter": @"none",
    @"filterIntensity": @1.0,
}];

CMSampleBufferRef processed = [self.beautyEffects copyProcessedSample:sampleBuffer];
if (processed) {
    [rtcEngine pushExternalVideoFrame:processed];
    CFRelease(processed);
}
```

### 3. Flutter (`live_beauty_sdk`)

```yaml
dependencies:
  live_beauty_sdk:
    git:
      url: https://github.com/bebebus/live-beauty-sdk.git
      path: flutter
```

```dart
import 'package:live_beauty_sdk/live_beauty_sdk.dart';

registerLiveBeautyLicenses();

final controller = LiveBeautyController();
await controller.initialize(initialConfig: LiveBeautyConfig.naturalStudio);

await controller.setConfig(
  LiveBeautyConfig.clearColdWhite.copyWith(
    faceSlim: 0.40,
    lipstick: 0.50,
    lipstickColor: 'cherry_red',
    lipstickFinish: 'glossy',
    lutUrl: LiveBeautyConfig.assetLutUrl('japanese_airy'),
    lutIntensity: 0.45,
  ),
);
```

---

## Verification Suite

Run all 7 verification gates (open-source sanitization, SHA256 model integrity, 548-vertex topology check, headless WebGL1 41-shader + 27-mutant gate, CoreImage pixel test, Android JVM unit tests + Release AAR build, and Flutter static analysis + unit tests) with a single command:

```bash
bash _tooling/run-all-tests.sh
```

---

## License & Acknowledgments

Released under the **[Apache License 2.0](./LICENSE)**. Free for both commercial and open-source use without royalties or license keys.

See [NOTICE](./NOTICE) for third-party attributions:
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache License 2.0)
- [PixPark GPUPixel](https://github.com/pixpark/gpupixel) (Apache License 2.0)
