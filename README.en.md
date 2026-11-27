# Bebebus Live Beauty SDK

> [中文](./README.md) | English

[![CI](https://github.com/bebebus/live-beauty-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/bebebus/live-beauty-sdk/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Version-v3.9.4-1E88E5.svg)](./CHANGELOG.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Free Forever](https://img.shields.io/badge/Commercial_Use-Free_Forever-2E7D32.svg)](#license-free-forever-pledge--sponsorship)
[![Live Demo: dh.video](https://img.shields.io/badge/Live_Demo-dh.video-FF4D6D?logo=googlechrome&logoColor=white)](https://dh.video)
[![Android API](https://img.shields.io/badge/Android-API_24%2B-3DDC84?logo=android&logoColor=white)](./android)
[![iOS](https://img.shields.io/badge/iOS-15.0%2B-000000?logo=apple&logoColor=white)](./ios)
[![Flutter](https://img.shields.io/badge/Flutter-3.22%2B-02569B?logo=flutter&logoColor=white)](./flutter)

**Bebebus Live Beauty SDK** is a production-grade, royalty-free, real-time on-device GPU beauty and AR engine powered by a **unified zero-heavy-dependency C++17 math & RHI core (`core/`)**, targeting **Android (OpenGL ES 2.0/3.0 + JNI)**, **iOS / macOS (full-parity Apple Metal + CoreImage compatibility layer)**, **Web (WebGL 2.0 / Emscripten RHI)**, and **Flutter**.

Designed to be **100% transport- and RTC-agnostic**, it requires zero license servers or online activation, bundles offline MediaPipe vision models, 10 studio `64³` 3D LUTs, and 5 high-precision UV makeup atlases out of the box, and plugs cleanly into **WebRTC, Agora, Zego, Tencent TRTC, CameraX, MediaCodec, or `AVCaptureSession`** pipelines.

### Live Demo & Experience Testbed

- **Official Live Demo Site**: [https://dh.video](https://dh.video)
- Powered in production by **Bebebus Live Beauty SDK** — preview live broadcast beauty rendering online or download the client app for hands-on testing of **478-pt diffeomorphic reshape, dual-band creamy skin smoothing, 6-layer PBR makeup, salon hair dyeing, CAT02 auto white balance, and `64³` 3D LUTs**.

---

## Highlights

### 1. 478-pt $C^\infty$ Diffeomorphic Face Reshape & Dual-Face Zero-Twitch Tracking
- **Zero Background Warp · Zero Contour Ghosting · Symmetric Profile Stability**: Expands MediaPipe's 478 landmarks with a dual-ring pinned outer boundary (`548` vertices / `1086` triangles) locking background pixels to exact zero displacement, combining a **2D Pure-Shear Plateau contour field**, **Boltzmann-weighted `SoftMax` half-width symmetry**, **narrow-side extreme yaw guard**, and **Jacobian safety + triangle-flip bisection**.
- **15 Facial & Cranial Micro-Reshape Controls**: Face slim (chin-only anti-elongation compensation without hairline drop), V-face, narrow face, small face, cheekbone slim, **high cranial top (`cranialTop`)**, chin length, forehead height, eye enlarge (outer-canthus inner gate), eye distance, eye corner, nose slim, nose length, mouth size, and smile lift.
- **Zero-Lag Handheld Shake & Jello Elimination (`FaceMeshFilter` / `FaceTimeline`)**: Conditioned 117-pt rigid craniofacial Procrustes basis + exact 2D `SE(2)` polar-rotation decomposition + 4-region One-Euro filtering + regional lag catch-up, plus **simultaneous dual-face co-streaming** (`maxFaces = 2` via `MultiFaceTracker` with duplicate-face NMS and zero extra 1080p FBOs).

### 2. Dual-Band Creamy Skin Smoothing, Studio Optics & 468-pt Anatomical Retouching
- **Green-Dominant Perceptual Guided Filter & 8-Tap Hairline/Beard Preservation**: Combines a `1/3` edge-preserving guided filter, `1/9` wide-scale skin homogenizer, hair-weighted guided filtering (`kGuidedWeight*`), and automatic `RGBA16F` intermediate precision — removing blotches and tear-trough dullness while keeping `1 px` eyebrow/beard strands (`8-tap lineScore`) and `2–4 px` skin pores crisp.
- **Porcelain De-Yellowing Whitening & Studio Optical Realism**:
  - Luminance-normalized neck/body skin gating (`neckTone` + `portraitPrior` spatial guard) for seamless face-to-collarbone continuity without bleaching warm walls or leaving dark chest shadows;
  - **6-DoF 3D Vertex-Normal Studio Relighting (`relight`)** with 4 pose-reactive lighting rigs (`natural`, `studio_softbox`, `butterfly`, `rembrandt`);
  - **T-Zone Oil/Specular Control (`oilControl`)**, **multi-scale isolated blemish healer (`blemishRemove`)**, and **pluggable neural skin residual hook (`setAiResidualTexture`)**.
- **In-Place 468-pt Anatomical Sub-Mesh Retouching (Zero Extra 1080p FBO)**: YCbCr chroma-migrating tear-trough concealer (preserving aegyo-sal), curved nasolabial fold softening, mouth-open-gated 18-triangle inner-mouth teeth whitening (physically never bleaches glossy lips), and 28-triangle eye brightening.

### 3. 6-Layer PBR Real-Time Makeup & Strand-Preserving Salon Hair Dyeing
- **6-Layer UV Real-Time Makeup**: Velvet & water-glow specular lipsticks (6 shades with 6-DoF yaw specular glide, open-mouth inner-lip cutout, and non-darkening dark-lip concealer), 3 blush styles, eyebrows, eyeshadow, eyeliner, aegyo-sal, exact W3C Soft-Light 3D contour & highlight, blink-gated iris contact lenses, and 3D parallax dual catchlights.
- **Ambient WB Adaptation & Semantic Occlusion Decoupling**: Adapts pigments to live room lighting (`ambientMakeupBlend`); defers semantic occlusion masks (bangs, glasses, hands, cups, microphones) strictly after skin whitening/smoothing and before makeup so occluders never punch dark holes in skin while 100% blocking makeup bleed and auto-muting lower-face makeup under heavy occlusion.
- **Real-Time Salon Hair Dyeing & Angel-Ring Sheen (`hairDye` / `hairShine`)**: 6 shades (`milk_tea_brown`, `haze_blue`, `ash_purple`, `rose_gold`, `cool_black`, `caramel_copper`) with melanin S-curve lift, Foreground/Background color-projection boundary matting, dual-lobe Kajiya-Kay sheen, and `0/255` forehead skin bleed.

### 4. Cinematic Atmosphere Post-Processing, CIE D65 CAT02 Auto WB & `64³` 3D LUTs
- **10 Built-In `512×512` (`64³`) Studio 3D LUTs**: `natural_studio`, `clear_white`, `peach_milk`, `pure_pink`, `korean_glow`, `japanese_airy`, `soft_film`, `night_vibe`, `champagne_glam`, `sweet_berry`, plus disk-cached `https://` remote LUT hot-swapping.
- **Exact CIE D65 CAT02 3×3 White Balance (`colorTemp` / `colorTint` / `autoWhiteBalance`)**: Supports manual temperature/tint grading and live face-skin ROI auto white balance (neutralizing warm tungsten and green fluorescent rooms with zero shift in daylight), fused directly into `kCompositePro` with zero extra passes when Bloom is off.
- **Soft-Knee Pyramid Bloom & Non-Darkening Filmic Grain (`bloom` / `filmGrain`)**: Reuses existing `1/4` and `1/9` buffers (zero extra FBOs) for clean highlight bloom without lifting deep blacks.

### 5. 6-Class Semantic Parsing, Bokeh, Virtual Background, Chroma Key & 5-Axis Body Reshape
- **6-Class & 19-Class Semantic Parsing**: Ingests MediaPipe `selfie_multiclass_256x256` (background, hair, body skin, face skin, clothes, accessories/hands) and CelebAMask-HQ / BiSeNetV2 19-class parsing maps.
- **Full Background Suite**: Supports **Portrait Bokeh**, **Green-Screen-Free Virtual Background (`kVirtualImage` / `kVirtualBlur`)**, and **Studio YCbCr Green/Blue Chroma Key (`kChromaKey` + despill + light wrap)** with dual-scale RGB-guided temporal matte refinement (`kMatteRefine`) and dual-face foreground anchor shields.
- **5-Axis Body Reshape with Hand Shield**: Pose-driven waist slimming (with dual-hand palm shield preventing bent fingers), bust enhancement, C¹ integral leg lengthening, shoulder slimming, and chin-aware swan neck.

### 6. 3D Head-Occluded AR Accessories, 2D Stickers, Gesture SDF Particles & `.lbfx` Bundles
- **3D Head-Occluded AR Accessories (`Ar3dSolver`)**: 468-pt head depth shell occluding receding temples/earrings, curved glasses/tiara presets & custom glTF-baked meshes, 2nd-order damped spring pendulum physics, and 52-channel Face Blendshapes reactive triggers (`smile`, `mouth_open`, `brow_up`, `blink`, `cheek_puff`, `tongue_out`).
- **Face-Anchored 2D Stickers & Gesture SDF Particles**: Up to 4 concurrent 2D sticker slots plus zero-texture GPU-batched analytical SDF particle bursts (`heart`, `thumbs_up`, `victory`, `open_palm`).
- **Declarative `.lbfx` Effect Bundles (`EffectBundleParser`)**: Zero-dependency JSON manifests bundling beauty presets, hair dye, 2D stickers, 3D meshes, and expression/gesture rules.

### 7. Live-Streaming Performance, Auto-Tiering & Production Resilience
- **Ultra-Low Overhead**: Unified C++17 CPU core runs in **~55 μs/frame** (`< 0.2%` of a 30fps budget); supports `LoadOp::kDontCare` tile discard, zero-copy `outputRt` FBO output (`nativeProcessFrameToFbo`), one-step Android CameraX `GL_TEXTURE_EXTERNAL_OES` ingestion (`processOes`), zero-copy iOS `CVMetalTextureCache` `CVPixelBuffer` bridging, and Apple Vision ANE async tracking (`FLTVisionTracker`).
- **End-to-End Stability**: `DeviceTierPolicy` (`Lite` / `Standard` / `Ultra`) GPU auto-tiering, `FrameDelayRing` (0..4 frames), adaptive `256×256` far-field face ROI, GLES 3.0 dual-PBO DMA readback, IMU gyroscope shake anti-ghosting (`setCameraShaking`), `GpuDelegatePolicy` native crash watchdog, 3-stage thermal/FPS/bitrate load governor, and live `BeautyPerfStats` telemetry.

> See [docs/integration-recipes.md](./docs/integration-recipes.md) for complete **CameraX (OES) / WebRTC / Agora / Zego / Tencent TRTC** integration recipes.

---

## Repository Layout

```text
live-beauty-sdk/
├── core/                     # Unified C++17 math & RHI core (zero heavy third-party deps)
│   ├── include/livebeauty/   # Math core, RHI (GLES / Metal / WebGL2), LiveBeautyEngine, Ar3dSolver, EffectBundleParser
│   ├── src/                  # Pure C++17 implementation, RenderPipeline, 35 shared GPU shaders (GLES/JNI, Metal/MSL, WebGL2)
│   └── tests/                # CTest math tests, Mock RHI pipeline tests, CPU micro-benchmark & macOS Metal GPU pixel gates
├── android/                  # Standalone Android Gradle library (io.bebebus:live-beauty:3.9.4, minSdk 24, arm64-v8a/armeabi-v7a)
│   └── live-beauty/          # OpenGL ES 2.0/3.0 engine, JNI bridge, MediaPipe trackers, bundled assets & 37 JVM test suites
├── ios/                      # Standalone iOS CocoaPods library (LiveBeautySDK.podspec 3.9.4, iOS 15.0+)
│   ├── Classes/              # FLTLiveBeautyMetalEngine (full Metal) + FLTVisionTracker (ANE) + FLTVideoEffects (CoreImage)
│   ├── Resources/            # Bundled offline face_landmarker.task model
│   └── Tests/                # macOS / iOS CoreImage pixel & smoother regression suite
├── flutter/                  # Flutter plugin wrapper (live_beauty_sdk 3.9.4)
│   ├── lib/                  # LiveBeautyConfig (60+ params & 5 presets), LiveBeautyController, extensions & LicenseRegistry
│   └── test/                 # Dart unit tests for wire round-trip, whitelists & MethodChannel lifecycle
├── reference/                # Python mathematical reference (beauty_ref.py), golden vectors & deterministic asset generators
├── _tooling/                 # 44+35 shader headless WebGL1 gate, SHA256 model verifier, sanitizer & 8-gate test runner
├── docs/                     # Integration recipes (integration-recipes.md) & architecture plans
├── ARCHITECTURE.md           # Deep-dive pipeline topology, unified C++17 core & diffeomorphic mesh documentation
└── CHANGELOG.md              # Full release history (v1.0.0 → v3.9.4)
```

---

## Quick Start

### 1. Android Native (WebRTC / Agora / Zego / TRTC / CameraX)

Include `android/live-beauty` as a Gradle module (or consume `io.bebebus:live-beauty:3.9.4` / `live-beauty-release.aar`):

```kotlin
import io.bebebus.livebeauty.LiveBeautyPresets
import io.bebebus.livebeauty.LiveBeautyProcessor
import io.bebebus.livebeauty.LiveBeautySticker
import io.bebebus.livebeauty.StickerAnchor
import io.bebebus.livebeauty.VirtualBackgroundMode

// 1. Instantiate (thread-safe config updates from any thread)
val beautyProcessor = LiveBeautyProcessor(appContext = context.applicationContext)
beautyProcessor.setPipelineOptions(maxFaces = 2, frameDelayFrames = 0)
beautyProcessor.updateConfig(
    LiveBeautyPresets.koreanWaterGlow.copy(
        faceSlim = 0.42f,
        vFace = 0.30f,
        skinTexture = 0.50f,
        autoWhiteBalance = 0.55f,
    )
)

// 2. Initialize on your OpenGL ES rendering thread
beautyProcessor.onGlCreated(context)

// 3. Process upright RGBA GL_TEXTURE_2D per frame (or use processOes for CameraX/WebRTC OES textures)
val outputTexId = beautyProcessor.process(
    inputTexture = cameraRgbaTex2D,
    width = frameWidth,
    height = frameHeight,
    timestampNs = surfaceTimestampNs,
)

// Optional: virtual background, 2D face stickers, IMU shake anti-ghosting & live telemetry
beautyProcessor.setVirtualBackgroundTexture(bgTex2D, aspect = 9f / 16f, mode = VirtualBackgroundMode.VIRTUAL_IMAGE)
beautyProcessor.setSticker(0, LiveBeautySticker(anchor = StickerAnchor.FOREHEAD_CROWN, glTextureId = crownTex2D))
beautyProcessor.setCameraShaking(isGyroShaking)
val stats = beautyProcessor.perfStats() // emaFrameMs / fps / loadLevel / activePassesMask

// 4. Reset temporal smoothers when flipping front/rear camera
beautyProcessor.resetTracking()

// 5. Release before tearing down the GL context
beautyProcessor.onGlReleased(contextAlive = true)
```

### 2. iOS Native (CocoaPods / Full-Parity Metal Engine & CoreImage Layer)

```ruby
pod 'LiveBeautySDK', :git => 'https://github.com/bebebus/live-beauty-sdk.git', :tag => 'v3.9.4'
```

**Recommended: Unified C++17 + Apple Metal Engine (`FLTLiveBeautyMetalEngine`, full feature parity with Android & zero-copy `CVPixelBuffer` / `MTLTexture`):**

```objc
#import <LiveBeautySDK/FLTLiveBeautyMetalEngine.h>

// 1. Initialize Metal engine & update config (thread-safe 50+ key dictionary)
self.metalBeauty = [[FLTLiveBeautyMetalEngine alloc] init];
[self.metalBeauty setConfiguration:@{
    @"enabled": @YES,
    @"smoothing": @0.55,
    @"skinOnly": @YES,
    @"whiten": @0.32,
    @"rosy": @0.18,
    @"faceSlim": @0.38,
    @"eyeEnlarge": @0.28,
    @"autoWhiteBalance": @0.50,
    @"hairDye": @0.0,
}];

// 2. Drive Apple Vision ANE segmentation/pose & render zero-copy CVPixelBuffer
[self.metalBeauty submitVisionPixelBuffer:pixelBuffer frameTimestampMs:tsMs];
CVPixelBufferRef outBuffer = [self.metalBeauty copyProcessedPixelBuffer:pixelBuffer
                                                       frameTimestampMs:tsMs
                                                                  nowMs:nowMs];
if (outBuffer) {
    // Forward to RTC capturer or preview layer
    CVPixelBufferRelease(outBuffer);
}
```

> For lightweight CoreImage / `CMSampleBufferRef` pipelines, the backward-compatible `FLTVideoEffects` API (`copyProcessedSample:` / `beautyImage:configuration:geometry:presence:`) remains fully supported.

### 3. Flutter (`live_beauty_sdk`)

```yaml
dependencies:
  live_beauty_sdk:
    git:
      url: https://github.com/bebebus/live-beauty-sdk.git
      path: flutter
      ref: v3.9.4
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
    hairDye: 0.35,
    hairDyeColor: 'milk_tea_brown',
    lutUrl: LiveBeautyConfig.assetLutUrl('japanese_airy'),
    lutIntensity: 0.45,
  ),
);
```

---

## Built-In Studio Presets

| Preset | Best For | Signature Look |
| --- | --- | --- |
| `original` | A/B comparison / pass-through | Zero-copy skip of all GPU passes |
| `naturalStudio` | Daily live / co-streaming (default) | 45% natural micro-pore texture, subtle reshape, `natural_studio` 3D LUT |
| `clearColdWhite` | E-commerce / bright studio streams | Cool porcelain skin, glossy rose-bean lip, peach blush, dual catchlights |
| `koreanWaterGlow` | Korean dewy peach glam | Luminous 3D highlight, 40% aegyo-sal, peach eyeshadow, hazel lenses |
| `mensClean` | Male broadcasters | 65% skin & beard strand preservation, under-eye concealer, crisp bone contour |

---

## Verification Suite

Run all **8 automated quality gates** (open-source sanitization, SHA256 model integrity, 548-vertex topology verification, headless Chrome WebGL1 gate covering 44 Android GLSL shaders + 28 mutants + 35 C++ GLES shaders in both `highp`/`mediump` + 10 `64³` 3D LUTs, CoreImage pixel tests, 37 Android JVM test suites + Release AAR build, Flutter static analysis + unit tests, and C++17 math/RHI + Apple Metal real-GPU pixel & 7-scenario quality benchmarks) with a single command:

```bash
bash _tooling/run-all-tests.sh
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for deep-dive pipeline topology and [CHANGELOG.md](./CHANGELOG.md) for the complete version history.

---

## License, Free-Forever Pledge & Sponsorship

- **Commercial Use Allowed**: Released under the **[Apache License 2.0](./LICENSE)**. Free for personal, open-source, and commercial products with zero royalties, license keys, or online activation.
- **Free Forever — No Paid Tiers**: We pledge that this project is **permanently free and will never charge fees or release any paid/pro/paywalled advanced features** — every beauty, reshape, makeup, hair-dyeing, 3D AR, virtual background, and optimization feature remains 100% open-source and free for everyone.
- **Voluntary Sponsorship**: If this SDK helps your product or team, you can optionally support ongoing development via the EVM address or QR code below (ETH / USDT / USDC / BSC / Arbitrum / Polygon, etc.):
  - `0x70Ce069e59B3a5B0e3d83a8500094b6666666666`

  <img src="./docs/assets/sponsor-evm-qr.png" width="180" alt="EVM Sponsor Address QR Code" />

See [NOTICE](./NOTICE) for third-party attributions:
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache License 2.0)
- [PixPark GPUPixel](https://github.com/pixpark/gpupixel) (Apache License 2.0)
