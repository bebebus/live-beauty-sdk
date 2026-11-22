# Changelog

All notable changes to **Bebebus Live Beauty SDK** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-22

### Added
- **Android Native SDK (`io.bebebus:live-beauty:1.0.0`)**:
  - Push-stream/RTC-agnostic OpenGL ES 2.0 `GL_TEXTURE_2D` processor (`LiveBeautyProcessor`, `VideoEffectsFilter`, `BeautyEngine`) with zero `RootEncoder` / JitPack coupling.
  - **14-item 478-pt $C^\infty$ Diffeomorphic Face Mesh Deformer (`FaceMeshDeformer`)**: 2D pure-shear plateau contour field (`evalContour2D`), Boltzmann weighted `SoftMax` half-width symmetry, geometry-based yaw invariance, minimum-relax equalization across grazing slivers, and real-time background boundary pin ring (548 vertices / 1086 triangles).
  - **1080p Guided Skin Smoothing & Texture Restoration (`GuidedSkinSmoothPass`)**: Edge-preserving O(1) box-guided filter with pore micro-texture retention, beard/eyebrow high-pass hair protection (`HairSegmenterTracker` + `SelfieSegmenterTracker`), blemish suppression, nasolabial fold softening, dark-circle removal, eye brightening, and YCbCr teeth whitening.
  - **6-Layer UV Real-Time Makeup (`FaceMakeupPass`)**: Lipstick (velvet & water-glow specular), blush (`apple` / `drunk` / `under_eye`), eyebrow, eyeshadow, eyeliner, aegyo-sal, 3D nose/cheekbone contour & highlight, iris-masked contact lenses, and dual catchlights.
  - **Color Grading, Body Reshape & Portrait Bokeh**: 10 built-in `512x512` (`64^3`) 3D LUTs (`LutLoader`) + HTTPS LUT cache, 5-axis pose-driven body reshape (`PoseBodyTracker`), portrait background blur, temporal denoise (`TemporalDenoisePass`), and 3-stage thermal/FPS/bitrate load governor.
- **iOS Native SDK (`LiveBeautySDK.podspec`)**:
  - Pure `CoreImage` + `MediaPipeTasksVision` 478-pt face beauty processor (`FLTVideoEffects`, `FLTFaceGeometry`, `FLTFaceGeometrySmoother`, `FLTFaceLandmarkTracker`) supporting both `CMSampleBufferRef` capture pipelines and direct `CIImage` composition.
- **Flutter Plugin Wrapper (`live_beauty_sdk`)**:
  - Immutable 40+ parameter `LiveBeautyConfig`, 5 studio presets (`original`, `naturalStudio`, `clearColdWhite`, `koreanWaterGlow`, `mensClean`), `LiveBeautyController` (`MethodChannel` + `EventChannel`), `LiveBeautyEngineRegistry`, and `registerLiveBeautyLicenses`.
- **Deterministic Reference & Verification Suite (`reference/` & `_tooling/`)**:
  - Python reference mathematical model (`beauty_ref.py`), cross-platform golden vectors (`beauty_vectors.json`), deterministic topology/LUT/makeup generators, headless Chrome WebGL1 41-shader & 27-mutant verification gate (`glsl_gate.py`), offline SHA256 model integrity gate, and open-source sanitization gate.
