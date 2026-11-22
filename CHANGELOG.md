# Changelog

All notable changes to **Bebebus Live Beauty SDK** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-22

### Added
- **Dual-Band Creamy Skin Homogenization (`GuidedFilterPrePass` + `GUIDED_COEFF`)**:
  - Added a `1/9` resolution wide-scale (`~32 px` full-res equivalent) skin tone homogenizer folded directly into the guided filter's `(b, a)` linear coefficients (zero extra `FRAGMENT_PRO` uniforms, preserving the 15-vector budget). Smooths tear-trough dullness and cheek/perinasal blotches by `≥ 34%` while protecting 3D nose-bridge/jawline bone contrast (`boneKeep`) and `2–4 px` pore micro-texture.
- **468-pt Anatomical Sub-Mesh Retouching & Pass Merging (`FaceRetouchMeshPass`)**:
  - Upgraded dark-circle concealer, nasolabial fold softening, eye brightening, and teeth whitening from full-screen ellipses to in-place anatomical sub-mesh rasterization on `FaceMeshWarpPass`'s FBO (eliminating an entire `1080×1920` RGBA8 intermediate FBO):
    - **Aegyo-Sal Preservation (`R1`)**: Locks vertex weights to `0` across all 14 lower-lid aegyo-sal vertices (`AEGYO_SAL_PROTECTED_SET`), lifting only infraorbital tear troughs toward lower-cheek skin tone.
    - **Curved Smile Nasolabial Softening (`R2`)**: Tracks the 4-segment curved alar-to-commissure crease with per-vertex perpendicular skin-bridge normals.
    - **Zero-Lip-Bleach Inner-Mouth Teeth Whitening (`R3`)**: Rasterizes strictly on the 18 inner-mouth cavity triangles (`INNER_MOUTH_INDICES`), physically guaranteeing `0/255` modification on glossy or pale lipsticks.
- **Adaptive Far-Field Face ROI & OpenGL ES 3.0 Dual-PBO Async Readback (`FaceRoiSelector` + `FaceTextureSampler`)**:
  - Automatically crops an isotropic `1:1` `256×256` face ROI (`2.5–3.5×` higher effective resolution) with hysteresis (`0.30` / `0.36`) during half-body/dance streams, reprojecting coordinates seamlessly and falling back to full-frame search immediately on any missed frame.
  - Opportunistically enables dual `GL_PIXEL_PACK_BUFFER` (PBO) asynchronous DMA readback on OpenGL ES 3.0+ contexts with automatic fallback to `GLES20.glReadPixels`.
- **High Cranial Top / Fluffy Hair-Framing-Face (`cranialTop: 0..1`)**:
  - Added the 15th 478-pt diffeomorphic reshape parameter (`cranialTop`) across Android (`VideoEffectsConfig`, `FaceMeshDeformer`) and Flutter (`LiveBeautyConfig`), puffing the cranial crown and upper-temporal hair silhouette while keeping the bare forehead hairline natural and locking outer background vertices to exact zero displacement.
- **Official Live Demo & Testbed**: Linked `https://dh.video` across documentation and SDK metadata.

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
