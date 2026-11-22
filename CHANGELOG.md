# Changelog

All notable changes to **Bebebus Live Beauty SDK** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Published release tags are immutable; any post-release fix is published as a new patch version.

## [1.1.2] - 2026-11-23

### Improved
- **Physically-Based Soft-Blur Lip Realism (`MAKEUP_LIP_FRAGMENT`, `MakeupPass`, `contour_highlight.png`)**:
  - Replaced the fixed `0.745` brightness floor (`normY`) with proportional scene-illumination shading (`0 → 0` in shadows) + midtone-only dark-lip de-dulling + multiply/pigment fusion, preserving natural 3D upper/lower lip shading contrast (`ΔY ≥ 0.12`), closed-mouth parting seams (`ΔY ≈ +0.001`), oral cavity shadows (`ΔY = 0.000`), room white balance, and fine vertical lip creases.
  - Replaced the wide pink-tinted static gloss band (`+ vec3(0.42, 0.38, 0.40)`) with a wet-clearcoat optical model (deeper wet pigment + tight neutral warm-white Fresnel catchlight centered on the true lower-lip belly `v = 0.714` and cupid-bow crest `v = 0.673`, modulated by real surface brightness and micro-creases).
  - Added zero-extra-uniform anatomical pixel gates: `darkGate` (keeps parting seam / cavity / beard dark), `teethGate` (100% guards exposed bright teeth against pink staining), and `skinRimGate` (selectively attenuates yellowish perioral skin bleed in the outer feather zone by `> 5×`).
  - Widened `contour_highlight.png` lip feathering (`0.018` upper / `0.015` lower outer blur, `0.008..0.044` mouth-corner taper, inner-belly-to-rim gradient) to eliminate hard sticker/tape edges.
  - Recalibrated all 6 lipstick shades (`rose_bean`, `coral_peach`, `cherry_red`, `milk_tea`, `berry_plum`, `orange_soda`) for rich, distinct true-to-swatch hue separation across both velvet and glossy finishes.
- **Anti-Lag Speech Lip Tracking (`FaceMeshFilter`)**:
  - Added a 40-point `LIP_RING` speed boost to the lower-face filter region (matching `LID_RING` for blinks), keeping stationary jitter `≤ 0.8 px@720` while preventing lip makeup lag during rapid speech/smiling.
- **GPU Regression Probes `L1–L5` (`_tooling/glsl/`)**:
  - Added 5 WebGL shader probes locking dark-seam/cavity neutrality (`L1`), zero teeth staining (`L2`), outer-rim perioral skin rejection (`L3`), glossy multi-shade hue separation (`L4`), and upper/lower 3D lip shading contrast (`L5`).

## [1.1.1] - 2026-11-22

### Fixed
- **Cranial Top Outer Anchor Ring Isolation (`FaceMeshDeformer`)**: Restored the `0.115 * fw` minimum outer-buffer clearance invariant for `cranialTop = 0` and localized `cranialTop` outer expansion strictly to the crown/upper-temple arc (`crownW * CRANIAL_OUTER_PAD * fh`) so jaw/cheek background anchors remain untouched when slimming.
- **Even Dark-Skin Under-Eye Concealer Guard (`RETOUCH_SKIN_FRAGMENT` & `FaceRetouchMeshPass`)**: Gated the warm tear-trough neutralization offset by `liftMask = clamp(deficit * 3.8, 0.0, 0.75)` (guaranteed `0/255` change on uniform dark skin with zero under-eye shadow, enforced by GPU gate `R1`), selected vertex sample offsets using live `(dc, nf)` weights, eliminated the redundant 15 KB per-frame vertex buffer upload, and wrapped `renderInPlace` GL draws in `try/finally` to restore shared GL state on exceptions.
- **Guided Filter FBO Feedback Loop Elimination (`GuidedFilterPrePass`)**: Bound pre-allocated `macroTextures[0]` whenever `uEvenness == 0f` instead of leaving `GL_TEXTURE1` bound to `textures[0]` while rendering into `frames[0]`.
- **Shared-Context GLES 3.0 PACK State & Error Scoping (`FaceTextureSampler`)**: Paired `beginPack()` / `endPack()` to save and restore host `GL_PACK_ALIGNMENT`, `GL_PACK_ROW_LENGTH`, `GL_PACK_SKIP_PIXELS`, `GL_PACK_SKIP_ROWS`, and `GL_PIXEL_PACK_BUFFER_BINDING` across sync and PBO readbacks; preserved newly-created resource error checks prior to PBO enqueue; accumulated second-pass ROI readback latency into `lastStallNs` / `unreadStall`.
- **Slow-Device Far-Field ROI Stability (`FaceRoiSelector` & `VideoEffectsFilter`)**: Evaluated `STALE_MS` (`350 ms`) from inference arrival time (`receivedMs`) bounded by absolute capture age (`MAX_SAMPLE_AGE_MS = 650 ms`), eliminating ROI ↔ full-frame oscillation on slow/CPU-throttled devices while preserving immediate miss fallback; added same-frame clean fallback if `FaceRetouchMeshPass` throws.
- **CI & Compiler Cleanliness**: Upgraded GitHub Actions to `checkout@v5`, `setup-python@v6`, `setup-java@v5`, and `setup-chrome@v2`; removed redundant null checks in `VideoEffectsFilter`.

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
