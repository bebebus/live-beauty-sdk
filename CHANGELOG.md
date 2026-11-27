# Changelog

All notable changes to **Bebebus Live Beauty SDK** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Published release tags are immutable; any post-release fix is published as a new patch version.

## [3.9.4] - 2026-11-27

### Fixed
- **Default Image Quality (Group A)**:
  - Narrowed `ambientSceneLift` to true dim scenes (`L <= 0.30`, max `0.18`) and scaled by `portraitPrior` so normal well-lit backgrounds stay completely un-lifted; raised ambient light EMA damping to `tau = 4000ms` to suppress camera AE oscillation.
  - Added `portraitPrior` spatial guard on body/neck whitening, rosy, and smoothing so warm wooden/beige walls outside the portrait silhouette no longer bleed.
  - Replaced 4-tap with 8-tap directional `lineScore` (diagonal ±45° included) inside `kCompositePro` while preserving `guidedA` gating so 1px eyebrows/beard stubble retain high contrast without blocking blemish/blackhead smoothing.
  - Recalibrated `twoStageSmoothMix` (`stage1 = clamp(s/0.85)`), preserved 65% macro facial shading in `kSkinHighFreq`, clamped negative texture deltas (`>= -0.025`), and fixed negative `brightness` dimming, warm `kPorcelain`, luma-normalized `kRose`, and purple-only lip care.
  - Zeroed `indoorTemporalDenoise` for `L >= 0.14` (`0.18` only at `L <= 0.08`, gated `>= 0.15`) so normal lighting skips the wasted full-res temporal denoise pass.
  - Changed face-slim anti-elongation compensation to chin-only (`0.20*slim + 0.15*narrow`), removing hairline drop and whole-head shrinkage.
- **Makeup, Reshape, Body & Segmentation Bugs (Group C)**:
  - Hand `body_skin` inside face ellipse now masks makeup and triggers `lowerFaceOcc` (`FaceParsingMapper`).
  - Decoupled Porter-Duff alpha in `kMakeupIris` with blink gate `(0.06, 0.13)` and corneal specular preservation.
  - Added 250ms stale body timeout and config-transition reset in `LiveBeautyEngine` + `nativeSubmitBodyMiss`.
  - Added per-side outer-canthus `innerGate` protection (temporally smoothed against `sRef`), unified `yawKeep`, post-safety triangle flip bisection, C1 integral long legs, chin-aware swan neck, and 300ms knee-visibility hysteresis.
  - Fixed `kHairAlpha` independent `occAffine` and `hairMotionKeep`, monotonic `kBgBlur` steps (`1.0, 2.0, 3.5`), 4-tap `kBgPremask`, privacy-preserving blur simplification under load, and low-contrast release in `kMatteRefine`.
- **GPU Bandwidth & Performance (Group D) + Tracking & CPU Load Governance (Groups B & E)**:
  - Downgraded `rtBody_`/`rtWarp_` from `RGBA16F` to `RGBA8` (`~24MB/frame` savings), added `LoadOp::kDontCare` (`glInvalidateFramebuffer` / `glDiscardFramebufferEXT` / `MTLLoadActionDontCare` without `glClear`), added `outputRt` zero-copy FBO direct output (`nativeProcessFrameToFbo`), disabled automatic style Bloom (`resolveStyleAtmosphere -> 0,0`), upgraded `kCompositePro` to full 3×3 CAT02 WB, skipped redundant `kPassThrough` when mesh-warped, and locked FP16 eligibility to `hardwareTier` with format-only RT reuse and smooth fadeout.
  - Added pre-assignment duplicate-face NMS + secondary slot promotion (`MultiFaceTracker`), reviewer-corrected 2nd-order acceleration rigid noise floor (`FaceMeshFilter`), minimum exit ramp & debounce floors (`FaceTimeline`), unified host/engine load governor (`setHostLoadLevel` + `GL_EXT_disjoint_timer_query`), direct letterboxed 6-class logits ingestion (`submitMulticlass6LogitsRegion`), `Mali-G1`/`PowerVR D-Series` tier recognition, and `prewarmShaders`.

## [3.9.3] - 2026-11-27

### Fixed
- **Decouple Semantic Occlusion from Skin Whitening/Smoothing (`RenderPipeline`)**: Defers `renderHairAlpha` for `occlusionMaskTex` (6-class hair/clothes/accessories & chroma occluders) to Stage 6c (strictly after Stage 6 `kCompositePro` and right before Stage 7 `renderMakeup`) so in-face hairlines/bangs, eyeglasses lenses, high collars, facial shadows, and motion-skewed occlusion edges NEVER zero out `rtSkinMask_.a` (`nonHair`) in `kCompositePro` and punch un-whitened dark shadow patches — while preserving 100% makeup occlusion on blush, contour, lipstick, and eyeshadow.
- **Luminance-Normalized Body/Neck Skin Gating & Hairline Bleed Guard (`gles_shaders.cpp`, `metal_shaders.h`)**: Replaces the absolute `R - B` body skin cliff outside `faceM` with a luminance-normalized `bodyGate` (`gateNorm * bodyChroma * bodyLuma`) so mildly shaded or cool-LED collarbones, necks, chests, and arms whiten smoothly without dark shadow blotches while keeping neutral/warm-grey walls, black hair, and dark beard bristles at `<= 1/255` bleed; also adds `hairSkinGuard` inside `faceM` to prevent coarse 128×128 `hairMask` edge bleed from darkening lit forehead/temple skin.
- **Eliminate Zero-Weight Pure-Black Poisoning in `kGuidedWeightDiff` (`gles_shaders.cpp`, `metal_shaders.h`)**: Smoothly falls back from `sum.rgb / max(sum.a, 0.001)` to original pixel RGB as `sum.a -> 0` (`smoothstep(0.02, 0.12, sum.a)`), preventing zero-weight hair pixels from collapsing to pure black `(0,0,0)` and bleeding dark halos through the 1/9-res `rtMacro_` blur.
- **Pipeline & Metal GPU Pixel Regression Gates**: Adds `testV393OcclusionDeferredAfterCompositeBeforeMakeup` and `testV393NoDarkShadowPatchesBodyFaceHairGuided`, and updates `testV392MulticlassChestNoDarkShadowPatch`.

## [3.9.2] - 2026-11-27

### Fixed
- **Eliminate Dark Shadow Patch on Chest under 6-Class Semantic Parsing**: Restricts full-frame semantic occlusion visibility masks (`submitMulticlass6Logits` / `submitMulticlass6Planar` / `submitFaceParsing*`) strictly to active face ellipse(s) (`primary` + `secondary`) so `clothes` (class 4) / `accessories` (class 5) / `hair` (class 1) below the chin never zero out `rtSkinMask_.a` (`nonHair`) in `kCompositePro` and leave an un-whitened dark shadow across the chest or collar.
- **Metal GPU Pixel Regression Gate**: Adds `testV392MulticlassChestNoDarkShadowPatch` verifying chest pixels match baseline whitening (`<= 2/255` diff) even when 6-class logits mark the entire chest as clothes, while in-face occluders remain masked.

## [3.9.1] - 2026-11-27

### Added
- **iOS SDK Bridge Parity**: Exposes `submitMultiFaceLandmarks` and `submitMulticlass6Planar` on `FLTLiveBeautyMetalEngine` and `wantMulticlass` / `onMulticlassPlanes` with 5-second idle segmenter memory reclamation on `FLTVisionTracker`.

## [3.9.0] - 2026-11-27

### Added
- **Automatic Ambient Illuminant CAT02 White Balance (`autoWhiteBalance`)**: Computes damped CIE D65 CAT02 chromatic adaptation temperature/tint offsets directly from live face-skin ROI `ambientRgb` (`BeautyQualityTuning::estimateAutoWhiteBalance`), neutralizing indoor warm tungsten and greenish fluorescent casts with zero shift on neutral daylight.
- **Direct 6-Class Planar Probability Ingestion (`submitMulticlass6Planar`)**: Adds `FaceParsingMapper::mapMulticlass6PlanarProbabilities` and `LiveBeautyEngine::submitMulticlass6Planar` for zero-interleave iOS/Android MediaPipe `ImageSegmenter` (`selfie_multiclass_256x256.tflite`) 6-plane confidence mask ingestion.
- **Full JNI / Android / iOS Atmosphere & Auto-WB Config Bridging**: Extends `NativeBeautyBridge` flat scalar transport to 58 slots and updates `VideoEffectsConfig` & iOS `parseObjcConfig` to pass `bloom`, `filmGrain`, `colorTemp`, `colorTint`, and `autoWhiteBalance`.
- **Dual-Face Independent Lower-Face Occlusion Gating**: Tracks secondary face bounds independently in `submitMultiFaceLandmarks` and drives per-face `secondaryFace.lowerMakeupKeep` so masking one co-host never mutes the other.
- **7-Scenario Automated Portrait Quality Benchmark Gate**: Adds quantitative Apple Metal GPU regression benchmarks covering warm tungsten, cool green fluorescent, low-light ambient lift, open-mouth teeth-safe lipstick, lower-face mask occlusion, dark male beard preservation, and dark hair dyeing with skin guard.

### Changed
- **GPU Pass Fusion & Fast-Path Bloom**: Fuses CAT02 white balance and non-darkening filmic grain directly into Stage 6 `kCompositePro` when Bloom is inactive and no post-composite layers follow (saving 1 full-resolution render pass), and skips the 1/9-res wide halo pyramid on mild bloom or non-Ultra tiers (cutting Bloom from 7 passes to 4 passes).

## [3.8.0] - 2026-11-27

### Added & Improved
- **Cinematic Atmosphere Post-Processing (`BeautyQualityTuning`, `RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`)**:
  - Added Stage 8 zero-extra-FBO post-processing (`ShaderId::kPostBloomExtract` + `ShaderId::kPostAtmosphere`, reusing existing 1/4-res `rtBg_` and 1/9-res `rtMacro_` scratch buffers):
    - **Multi-Scale Soft-Knee Pyramid Bloom**: Anti-flicker 5-tap bright extraction (`thresh=0.62, knee=0.28`) with 1/4-res tight glow + 1/9-res wide halo composited via highlight-safe, strictly non-darkening addition (`out >= src`); automatically skipped on `kLite` tier or when inactive (`0` GPU cost).
    - **Non-Darkening Organic Filmic Grain**: Bounded modular phase hash avoiding `mediump` float16 time-freeze, strictly non-darkening lift curve (`exponent ∈ [0.72, 1.0]`, shadow/highlight tapered so blacks stay crisp).
    - **Exact CIE D65 CAT02 Chromatic Adaptation White Balance**: `BeautyQualityTuning::cat02WhiteBalanceMatrix(colorTemp, colorTint)` computes exact linear von Kries CAT02 adaptation in double precision (exact identity at `0, 0`), applied in 3 dot products per pixel.
  - Added `BeautyConfig` fields (`bloom`, `filmGrain`, `colorTemp`, `colorTint`, supported in `.lbfx` bundles) plus automatic style-coupled atmosphere synergy (`BeautyQualityTuning::resolveStyleAtmosphere`) for cinematic/glam/soft presets (`night_vibe`, `soft_film`, `champagne_glam`, `pure_pink`, `peach_milk`, `korean_glow`, `clear_white`, while keeping `natural` crisp at `0/0`).
- **Open-Mouth Geometric Inner-Lip Cutout for Lipstick (`RenderPipeline`, `kMakeupLip`, `fs_makeup_lip`)**:
  - Passed live post-warp inner-mouth opening ellipse (`lm[13]/14/78/308`) and `mouthOpenGate` into `kMakeupLip` (`uU3`/`uU4`), cleanly cutting out the open oral cavity and exposed teeth during smiling/speaking while keeping seamless full-lip coverage when closed.
- **Lower-Face Heavy Occlusion Makeup Auto-Mute (`FaceOcclusionSolver`, `FaceParsingMapper`, `LiveBeautyEngine`, `RenderPipeline`)**:
  - Computed smoothed lower/mid-face occlusion ratio (`lowerFaceOcclusion()` / `estimateLowerFaceOcclusion()`) and mapped it via `BeautyQualityTuning::lowerFaceMakeupKeep` (`smoothstep(0.45, 0.78)`) to smoothly mute lower-face makeup (`lipstick`, `blush`, `contour`, `highlight`, `teethWhiten`) when hands, cups, or microphones cover the lower face — while keeping `eyebrow`, `eyeshadow`, `eyeliner`, `aegyoSal`, and `contactLens` 100% active.

## [3.7.0] - 2026-11-27

### Added & Improved
- **Commercial-Grade Beauty Pipeline Upgrade (`BeautyQualityTuning`, `RenderPipeline`, `FaceMeshDeformer`, `gles_shaders.cpp`, `metal_shaders.h`)**:
  - **Green-Dominant Perceptual Skin Smoothing & Two-Stage Curve**: Upgraded `kGuidedDiff` / `kGuidedWeightDiff`, `kBlemishHeal`, and `kCompositePro` range kernels to a green-dominant perceptual distance (`sqrt(0.24*dr² + 0.58*dg² + 0.18*db²) * 1.73205`) resisting indoor R/B chroma sensor noise while keeping exact edge preservation on neutral gray contours; added zero-extra-pass mid-band isolated dark pore/blemish pre-lift inside skin (`spotDelta ∈ [0.018, 0.240]`, `yOrig > 0.14`, skips dark hair/lash/beard and flat skin) and a two-stage smoothing curve (`BeautyQualityTuning::twoStageSmoothMix` / `twoStageDetailKeep`: `0..0.60` ramps clean coverage with 100% natural micro-grain, `0.60..1.0` tapers grain smoothly to porcelain finish).
  - **Log-Compressed Halo-Free Clarity**: Upgraded `kCompositePro` / `fs_composite_pro` sharpening to green-dominant band-pass with logarithmic gradient compression (`sign(g) * log(1 + 42*|g|) / 42`), sub-noise floor gating (`0.35..0.90 * core`), and balanced skin shielding (`1 - 0.62 * skin * featureKeep * faceM`), lifting eyelashes, hair strands, and eye contour crispness without ringing halos or grain amplification.
  - **Adaptive Indoor Temporal Stabilization**: Added `BeautyQualityTuning::indoorTemporalDenoise(lightLuma)` to extend subtle temporal noise stabilization (`0.04..0.11`) into normal indoor rooms (`lightLuma ∈ [0.18, 0.38]`) while staying `0.0` in bright daylight and `0.0` during camera shake (`denoiseKeep = 0`).
  - **Implicit Natural Retouch Synergy & Upgraded Local Retouch (`kRetouchSkin`, `kRetouchEyeTeeth`)**: Enabled subtle automatic dark-circle (`smoothing * 0.35`), nasolabial fold (`smoothing * 0.25`), and bright-eye (`smoothing * 0.40`) synergy whenever skin smoothing is active; upgraded tear-trough retouch to neutralize cool/bluish under-eye cast alongside luminance deficit (`0/255` on uniform skin, strictly non-darkening), upgraded nasolabial fold retouch with crease-gated base lift (`0/255` on flat skin), upgraded bright-eyes to strictly brighten-only 4-tap local sparkle + clean sclera (never darkening pupils, irises, or eyeliner), and gated `teethWhiten` by landmark mouth-opening ratio (`BeautyQualityTuning::mouthOpenGate`).
  - **Exact W3C Soft-Light Makeup & Extreme Side-Profile Contour Guard**: Upgraded `softLightBlend` in `kMakeupLayer` / `fs_makeup_layer` from Pegtop approximation to exact W3C/Photoshop Soft-Light for more luminous 3D contour & highlight; added asymmetric narrow-side contour guard (`1.95..2.55x` width ratio) in `FaceMeshDeformer` to prevent far-cheek silhouette collapse during steep 3/4 turns.

## [3.6.15] - 2026-11-26

### Added & Fixed
- **Douyin-Style Adaptive Ambient Scene Brightness & Global Brightness Fix (`BeautyQualityTuning`, `RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`)**:
  - Added `BeautyQualityTuning::ambientSceneLift(lightLuma)` wired via `uGrade.w` (`uU10.w`, zero extra uniforms) into `kCompositePro` / `fs_composite_pro`, smoothly lifting dim and indoor room environments (`~0.20..0.56` filmic exposure gain) while tapering cleanly to `0.0` in bright daylight (`lightLuma >= 0.76`), anchoring deep blacks (`0.015..0.085`) against milky haze, and rolling off bright lamps/windows (`0.72..0.96`) against clipping.
  - Promoted `config.brightness` (`uParams.z` / `uU1.z`) from a skin-only mask multiplier (`* w`) to the global highlight-preserving filmic exposure lift so adjusting brightness brightens the entire room and subject together rather than creating a bright floating face inside a dark room.

## [3.6.14] - 2026-11-26

### Fixed
- **TikTok-Style Exposure-Coupled Non-Darkening Lips (`gles_shaders.cpp`, `metal_shaders.h`, `face_makeup_uv.cpp`)**:
  - Added exposure-coupled, hue-preserving bare-lip luma tracking (`faceExp = whiten + 0.50*brightness + 0.35*relight`) plus gentle dark-upper-lip shadow recovery and cyanosis neutralization in `kCompositePro` / `fs_composite_pro` (with lip-seam guard tightened to `0.125..0.205` so real upper-lip shadows lift cleanly while dark seams and teeth remain `0/255` untouched), eliminating "white face -> blackened lips" simultaneous-contrast darkening without adding fake Barbie pink.
  - Replaced multiplicative Beer-Lambert G/B darkening in `kMakeupLip` / `fs_makeup_lip` with a TikTok-style dark-lip concealer pre-lift + luma-preserving chroma transfer (`fitLipGamut` / `fitGamut`), calibrated the 6 lipstick shades (`face_makeup_uv.cpp`), and lowered `darkGate` to `0.125..0.20` so all 6 shades lift dark/indoor lips cleanly (`dY > +0.010..+0.043`) while preserving 3D upper/lower shading, lip-line micro-texture, and bright-camera anti-washout (`L6`).

## [3.6.13] - 2026-11-26

### Fixed
- **Stable, Pixel-Accurate Salon Hair Dyeing & Coverage (`LiveBeautyEngine`, `FaceParsingMapper`, `HairMaskTuning`, `RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`)**:
  - Decoupled active hair styling (`hairDye > 0 || hairShine > 0`) from `FaceTimeline::denoiseKeep_`'s 16ms micro-motion binary cutoff and `loadFades_.body()` thermal level-1 shutoff in `LiveBeautyEngine` (now gating on `affineValid * shakeGate * steadyFace` and `loadFades_.smooth()`), expanded `HairMaskTuning::kAffineMaxShift` (`0.10 -> 0.18`) and tightened `kPresenceFadeInMs` (`300ms -> 120ms`), eliminating hair-color flickering/flashing off during slight head movement or normal streaming load while preserving camera-shake and stale-jump suppression.
  - Replaced `FaceParsingMapper::mapMulticlass6LogitsNhwc`'s overly strict hair cutoff (`smoothstep(0.55, 0.88, p[1] - 0.85*pSkin)`) and sluggish 3x3 box-blur EMA (`alpha=0.46`) with a calibrated coverage ramp (`smoothstep(0.32, 0.72, p[1] - 0.75*pSkin)`) and center-weighted 3x3 Gaussian + responsive jump-snapping hair EMA (`alpha=0.74`, snap `>56 -> 0.96`), lifting real-portrait hair coverage from ~52–57% to ~80–87% with zero forehead skin bleed and zero motion ghost trails.
  - Upgraded `kHairDyeFrag` (GLES2/WebGL1) and `fs_hair_dye` (Metal) in lockstep with a 3-tier Local Foreground/Background RGB Color Projection boundary matting pass (passing mask matting radius in `uU5.zw`, snapping coarse 128×128 mask bleed on skin/walls to `0/255`, snapping under-segmented dark hairline edges up to full coverage, and filling dark interior mask dropouts while preserving scalp partings), single calibrated mask ramp (`smoothstep(0.16, 0.58)`), bangs-friendly forehead guard (unconditionally protecting `sem.g` eyes/brows/lips while dyeing confident bangs over forehead `sem.r`), and uniform dark-hair coloring (`strandLift = smoothstep(0.008, 0.08, y)`, mix `0.62 + 0.38 * strandLift`).

## [3.6.12] - 2026-11-26

### Fixed
- **Zero-Lag Facial Feature Tracking During Handheld Phone Shake (`LiveBeautyCppEngine`, `FrameDelay`, `FLTFaceLandmarkTracker`, `FaceMeshFilter`, `FaceTimeline`, `FaceMeshDeformer`)**:
  - Eliminated MediaPipe `FaceLandmarker`'s un-configurable internal `LandmarksSmoothingCalculator` (`OneEuroFilter: min_cutoff=0.05Hz, beta=80.0, derivate_cutoff=1.0Hz`, which activates exclusively in `RunningMode.VIDEO` with `num_faces == 1` and injected 120–250ms / 20–63px double-filtering lag on oscillatory phone shake) by configuring `numFaces >= 2` (`minTrackingConfidence = 0.35f`) with sticky primary-face selection (preferring the fresh BlazeFace crop on overlapping whip-pan duplicates, and locking onto the broadcaster's anchor when a bystander enters frame), running `setOutputFaceBlendshapes` strictly on demand (saving ~5–10ms/frame CPU when AR accessories/stickers are inactive), elevating the face worker thread to `THREAD_PRIORITY_DISPLAY` (Android) / `QOS_CLASS_USER_INTERACTIVE` (iOS), eliminating the deferred-PBO / CPU 45ms beat-frequency stall, and enabling preview-mode `PacingLock` upward adaptation when inference slows down after initial lock.
  - Added coherent regional mean-lag catch-up (`regionLag_` on nose/eye/mouth, canceling uncorrelated landmark jitter by ~sqrt(N) while snapping rolling-shutter and wide-angle perspective lag back in 1–2 frames), `logScale_` lag catch-up (`> 2.2%`), and full 3D `(hw, P)` perspective lag catch-up in `FaceMeshFilter`, preserving 100% of v3.6.8–v3.6.10 stationary cheek/contour stability and slight-shake zero-ripple gates.
  - Prevented `FaceTimeline::at()` from crushing `steadyEye` down to `0.25` during aligned handheld phone shake (`shakeEyeKeep` shifted to extreme motion blur `1.35..2.50`), tightened `eyeMakeupKeep` and `steadyEye` recovery (`36ms` / `32ms`), expanded `horizonMs()` to `66..85ms` with rate-scaled curvature lead, and updated `FaceMeshDeformer::deform()` with split inner-feature displacement smoothing (`kDispTauInnerMs = 28ms` vs outer cheek/contour `kDispTauSteadyMs = 95ms`), fast `steadyEye`/`presence` transition detection (`kDispTauFastMs = 18ms`), smooth `envTau` yaw ramp, and crisp `45ms` eye-envelope tracking.

## [3.6.11] - 2026-11-25

### Fixed
- **Realistic Natural Lip Color & 3D-Shaded Translucent Lipstick (`gles_shaders.cpp`, `metal_shaders.h`, `face_makeup_uv.cpp`, `render_pipeline.cpp`)**:
  - Removed the synthetic `healthyLip` pink RGB override (`B > G` magenta-pink blended up to `51%` even in default `natural` mode with `lipstick = 0`) from `kCompositePro` / `fs_composite_pro`, and tightened `mouthZone` (`exclM` `0.55..0.88`) to match `1 - toneKeep` cleanly with zero lip-border skin bleed. Bare-lip care now preserves real lip RGB, shading, and texture 100%, only neutralizing unnatural bluish-purple cyanosis (`B > G`) and applying gentle proportional exposure balance for genuinely dim lips (`yLip < 0.42`, fading cleanly to `0.0` on normal/bright lips).
  - Replaced `kMakeupLip` / `fs_makeup_lip`'s flat opaque tint (`illum ∈ [0.88, 0.99]`, ~89% flat paint) with physically-based 3D-shaded Beer-Lambert translucent lip staining (`shade ∈ [0.56, 1.00]` preserving real upper-lip shadow vs lower-lip fullness and `1.15x/0.85x` lip-line micro-texture), scaled glossy specular naturally with `uLipColor.a`, removed velvet white-haze lift, calibrated `kRoseBean` (`0.74, 0.33, 0.30`) and `kCoralPeach` (`0.82, 0.39, 0.31`) away from fluorescent pink, and refined `lipAlpha` (`0.75 * lipstick * faceGain`).

## [3.6.10] - 2026-11-25

### Fixed
- **Cheek Beauty Spasmodic Twitching Elimination (`FaceMeshFilter`, `FaceMeshDeformer`, `FaceTimeline`, `FaceRoiSelector`, `FaceOcclusionSolver`, `RenderPipeline`, `LiveBeautyEngine`)**:
  - Replaced `FaceMeshFilter`'s 31-pt nose-dominated Procrustes basis (where single-vertex nose-tip leverage `+1.39` amplified 3D Z-column `(w1, w2)` jitter into `13px` spikes on deep `z=-0.43` cheek/jaw vertices) with a conditioned ~117-pt strictly rigid craniofacial inverse-variance basis (excluding eyelids, eyebrows/brow-band, mouth, chin, and lower jaw), cutting Z-column noise variance by ~4.6x with exact `1.000` 3D yaw fidelity and fixing `lagCatchup`'s dimensional velocity-vs-displacement mismatch (`2.0 * dead * teSec`) for smooth, zero-snap cheek stability alongside sub-pixel camera-shake tracking.
  - Added exact mirror- (`det < 0`) and roll-angle-invariant `FaceMeshFilter::angularVelOf` wired into `extrapolate`, `centerShift`, and `FaceTimeline::meshAt` (fixing mirrored front-camera roll cancellation), and gated `FaceTimeline` second-derivative curvature lead velocity by `poseVelGain_` to prevent stationary noise differentiation.
  - Normalized `FaceMeshDeformer` contour/eye width ratios against Procrustes 3D `sRef` instead of noisy 2-pt `lm[454]-lm[234]` span, raised `kFrontWShape` (`0.002 -> 0.008`), and added rigid-motion-invariant face-local normalized displacement smoothing (`dispUState_`/`dispVState_`, `95ms` steady / `18ms` fast-turn/slider adaptation with fold-safe blending toward current-frame raw displacement), eliminating `v205` mid-cheek slimming threshold jumps and grazing-triangle relaxation snaps.
  - Decoupled cheek/face makeup (`blush`, `contour`, `highlight`, `eyebrow`, `lipstick`) from eye-blink/shake `steadyEye` onto `steadyFace`, decoupled 10Hz internal `autoOccTex_`/`autoResidualTex_` from binary 16ms `denoiseKeep_` resets and full-face `kGuidedWeight*` guided-filter shader toggling, softened `FaceOcclusionSolver` temporal EMA and cool-indoor cheek shadow chroma thresholds, and added a generous active deadzone (`0.18*fw` pos, `0.12*fw` width) in `FaceRoiSelector` to keep `MediaPipe RunningMode.VIDEO` input crops rock-steady during handheld sway.

## [3.6.9] - 2026-11-25

### Fixed
- **Slight Camera Shake Water-Ripple / Jello Distortion Elimination (`FaceMeshFilter`, `FaceTimeline`, `FaceMeshDeformer`, `FaceRelightSolver`, `RenderPipeline`, `LiveBeautyEngine`)**:
  - Replaced `FaceMeshFilter`'s split `q_[0..1]` vs slow `q_[2..7]` affine smoothing with an exact 2D `SE(2)` polar-rotation + 3D perspective/foreshortening + isotropic `viewScale2` log-scale decomposition: `(tx, ty)` and in-plane camera roll `(cRot, sRot)` update at identical unified fast `aPose` (`0.0 px` rotational shear during handheld camera tilt), while ill-conditioned 3D Z-columns `(w1, w2)` and symmetric foreshortening `(p00, p01, p11)` filter on their own 3D velocity/lag gate (`yawLag` fast-turn snap preserved), and signed 8-DoF velocity `dq_` eliminates rectified jitter bias.
  - Restored symmetric `kResDcut = 1.5 Hz` One-Euro derivative filtering (removing `kResDcutDown` peak-rectification and `resLagBoost` tangential skin-noise amplification) with matched calm regional cutoffs (`0.52..1.25 Hz`) and zero `kFaceOval` lip coupling, eliminating multi-rate triangle shear waves across cheeks during slight camera shake.
  - Added curvature-braked 2D `SE(2)` sub-frame extrapolation + `meshOutMeta_` axis rotation in `FaceTimeline`, canthi-anchored dimensionless eye-lens ratio smoothing (`rEyeWRatioState_`, `lEyeWRatioState_`, `rEyeHRatioState_`, `lEyeHRatioState_`) in `FaceMeshDeformer`, per-slot temporal 3D angle smoothing in `FaceRelightSolver`, and face-center motion compensation + speed gating for async 64x64 `aiResidualRect` in `LiveBeautyEngine`.

## [3.6.8] - 2026-11-25

### Fixed
- **Facial Contour & Face-Shape Stability Elimination of Frame-to-Frame Jumping (`FaceMeshFilter`, `FaceTimeline`, `FaceMeshDeformer`, `FaceRoiSelector`, `LiveBeautyEngine`)**:
  - Split `FaceMeshFilter` canonical residuals into 4 anatomical regions (`midface`, `eyes`, `mouth`, and dedicated `facial contour` covering `kFaceOval` + outer cheek/jaw/temple ring at `0.65Hz` One-Euro with `2.2x` deadband and zero `lipPower` coupling), decoupled Procrustes translation `q_[0..1]` from steady linear scale/3D-pose `q_[2..7]` (`shapeCatchup` preserved for fast head turns), and removed `resExcess` lip-motion contamination from `motionSpeed_`, stopping speech and thumbnail contour noise from shaking the jawline and cheeks.
  - Stabilized `FaceTimeline::steadyFace_` (`1.0` throughout active landmark tracking without speech/sway `shakeFaceKeep` pumping; smooth `160ms`/`120ms` miss ramps) and upgraded `FaceMeshFilter::extrapolate` + `poseVel_` to a pure 2D rigid Euclidean isometry (translation + damped roll only, zero scale/shear/3D-perspective extrapolation) with continuous `poseVelGain_`, eliminating 15Hz/30Hz inter-sample face-width sawtooth breathing.
  - Added dimensionless scale-invariant temporal smoothing (`wLeftRatioState_`, `wRightRatioState_`, `signedYawState_`, `silExcessRatioState_`, `tau = 150ms` with `26ms` fast-turn adaptation), softened `spanBeta`/`excessBeta` (`24/fw`, `20/fw`) and `frontGate` (`[0.08, 0.34]`), restored damped `relax_` transitions (`kRelaxTauUpMs = 120ms`, `kRelaxTauDownMs = 35ms`), and split `RenderPipeline::resetPixelTemporal()` so handheld `cameraShaking` flushes pixel denoise/matte ghosts without resetting `FaceMeshDeformer` temporal state or thrashing `FaceRoiSelector` on moderate head motion.

## [3.6.7] - 2026-11-25

### Fixed
- **Eye-Socket, Nose-Contour & Beard-Contour Dark Trench / Patch Elimination (`RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`, `FaceOcclusionSolver`)**:
  - Decoupled spatial guided-smoothing edge protection (`featureKeep * fineDark`, restricted to true dark-hair boundaries `yFine < 0.18` and fine dark eyebrow/beard hair fibers `hairStrand`) from per-pixel tone whitening weight (`w = skin * toneKeep`), eliminating razor-sharp `bleedGuard = 0` black trenches along nose-wing creases, eye corners, and beard rims.
  - Separated upper orbital / brow-bone (`kBrowCore`: `skin = 0.88, excl = 0.52`) and glabella / temple rim (`kBrowBorder`: `skin = 0.95, excl = 0.20`) from eyeball/inner-lip full exclusions (`excl = 1.0`), preserving crisp eyebrow fibers while restoring seamless skin-tone brightening across eye sockets and the nasal root without goggle-shaped dark clouds.
  - Replaced absolute YCbCr `polyGate` collapse in dim shadows with luminance-normalized skin chromaticity (`relRB`, `relRG`) extending cleanly across the outer `faceM` feather (`smoothstep(0.04, 0.22, faceM)`), clamped negative face sharpening (`negCap = -0.010`) and negative non-highlight `sculpt`, and scaled cool-ivory desaturation down in deep shadows (`y2 < 0.42`) so shaded contours and beard stubble never turn ashy black while keeping `0/255` bleed on neutral walls and jet-black scalp hair.
  - Added `mode.z = 2.0` skin & brightness gating (`smoothstep(0.28, 0.52, baseY)`) for `contour` makeup so Multiply contour never blackens dark eye sockets, nasal creases, or jawline beards, added lower-central-face `beardShield` in `kHairAlpha` / `fs_hair_alpha`, and removed `darkObject` false-positive occluder cutouts on nostrils and beards in `FaceOcclusionSolver::compute`.

## [3.6.6] - 2026-11-25

### Fixed
- **Post-Shake & Face Re-Centering Lip, Nose & Eye-Socket Zero-Lag Recovery (`FaceTimeline`, `FaceMeshFilter`, `FaceMeshDeformer`, `LiveBeautyEngine`)**:
  - Reset stale pre-exit extrapolation holds (`holdX_`, `holdY_`) and motion suppression states (`steadyEye_`, `steadyFace_`, `eyeMakeupKeep_`) across `clearHistory()`, `dropSamples()`, and post-exit re-entry (`exitedDuringMiss`), derived 41-D `states_` (`uMouth` bare-lip lift & `uEyeMask` socket shield) directly from filtered 478-pt `meshCoords_` with zero secondary One-Euro lag, and reduced fade-in/settle ramps (`kTInMs = 50ms`, `steadyEye` `inMs = 50ms`, `kEyeMakeupInMs = 65ms`).
  - Added full 8-DoF Procrustes `basisLag` + per-region (`nose`, `eye sockets`, `lips`) non-rigid residual lag catch-up and asymmetric deceleration cutoffs (`kPoseDcutDown = 16Hz`, `kResDcutDown = 16Hz`, `kJumpT = 0.28`) in `FaceMeshFilter`, plus faster `FaceMeshDeformer` visibility/relaxation recovery (`kVisTauUpMs = 38ms`, `kRelaxTauUpMs = 38ms`), snapping lip, nose, and eye-socket landmarks within `< 66ms` (`1–2` frames) after camera shake, head yaw turns, or face re-centering.
  - Decoupled live landmark-tracked `in.steadyEye`, `in.steadyFace`, and `in.eyeMakeupKeep` from IMU gyroscope `setCameraShaking(true)` while preserving full temporal anti-ghosting (`kPassDenoise`, `kPassMatteRefine`, `asyncMaskKeep`), and forced immediate full-frame `selectFaceRoi` fallback during camera shake or rapid face motion.

## [3.6.5] - 2026-11-25

### Fixed
- **Forehead & Face-Contour Shadow Jumping Elimination (`FaceParsingMapper`, `LiveBeautyEngine`, `kBgVirtualMix`, `kMatteRefine`)**:
  - Gated internal asynchronous `11Hz` dynamic-INT8 multiclass hair/occlusion & `10Hz` ROI residual textures strictly to `hairDye`/`hairShine`, `hasMakeup()`, and `blemishRemove` (without virtual background), and tightened `FaceParsingMapper::mapMulticlass6LogitsNhwc` skin-vs-hair/occluder margins (`p[1] - 0.85 * pSkin`, `rawOccluder - 0.90 * pSkin`, `kParseEmaAlpha = 0.46`), eliminating `11Hz` dark unwhitened shadow patches jumping across the forehead and temples.
  - Replaced the oversized `1.12x * 1.15x` bounding ellipse in `personAnchorShield` with an anatomically tapered inner face-core (`jawT` + `browT` taper) and narrow center-throat shield (`|a| <= 0.28..0.42`), reducing real-room background bleed outside the jawline/cheek contour from `698` pixels to `0`.
  - Upgraded `kMatteRefine` / `fs_matte_refine` to an 8-tap dual-scale (`1.6` + `4.2` texel) RGB-guided joint bilateral + contrast-gated dark-halo snap + spatial-gradient-normalized temporal filter (`effRgbDiff = max(rgbDiff - 0.36 * spatialGrad, 0)`), plus translucent boundary dark-halo despill in `kBgVirtualMix`, snapping bloated `256x256` segmentation halos cleanly to the real forehead and face contour without sub-pixel tremor drop-outs.

## [3.6.4] - 2026-11-24

### Fixed
- **Virtual Background Shadow-Flash Elimination & RGB-Verified Temporal Matting (`RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`)**:
  - Restored calibrated sigmoid foreground thresholds (`[0.42, 0.72]` in `kBgVirtualMix` and `[0.38, 0.68]` in `kBgPremask`/`kBgMix`) and removed wide-area `headHalo`/`torsoHalo` `weakPerson` probability boosters from `personAnchorShield`, eliminating dark real-room shadow patches flashing across walls/curtains while preserving tight interior face-oval (`rFace <= 1.0`) + throat anti-occlusion shields.
  - Replaced asymmetric peak-holding `expand * 0.90` temporal matting in `kMatteRefine` / `fs_matte_refine` with 3-point neighborhood RGB-change-verified temporal damping (`temporalWeight = 0.30 + 0.52 * matteTemporal`), suppressing 1-frame neural confidence spikes on static backgrounds to `0/255` shadow bleed while snapping moving silhouettes in 1 frame.
  - Restricted `kBgHairMerge` to soft bokeh (`BackgroundMode::kBlur`) so asynchronous `5–12Hz` hair masks never inject lagging dark shadow silhouettes onto crisp virtual background images.

## [3.6.3] - 2026-11-24

### Fixed
- **Virtual Background Anti-Occlusion Foreground Shield & Load-Governor Stability (`RenderPipeline`, `gles_shaders.cpp`, `metal_shaders.h`, `LiveBeautyEngine`)**:
  - Decoupled virtual background replacement (`BackgroundMode::kVirtualImage`, `kVirtualBlur`, `kChromaKey`) from `EffectsLoadGovernor` `loadFadeBody` so thermal/CPU load step-ups (`level >= 1`) never fade out the virtual background or expose the broadcaster's real room; under load, only optional 4-pass background-image bokeh blur is bypassed.
  - Tightened foreground opacity saturation threshold (`0.88 -> 0.56`) and added dual-face (`primary` + `secondaryFace`) anatomical face-oval + neck/collarbone + shoulder/torso foreground shields (`personAnchorShield`) in `kBgVirtualMix` / `fs_bg_virtual_mix` and face-core protection in `kBgMix` / `kChromaKey` (plus green/blue chroma dominance gating), preventing segmentation dropouts from ever occluding people's faces, necks, or clothes.
  - Upgraded `kMatteRefine` / `fs_matte_refine` to asymmetric foreground-first temporal matting (1-frame snap on foreground expansion, stationary RGB flicker absorption on minor shrink, clean release on motion/exit) and preserved internal `autoParsePersonTex_` binding across camera-shake resets.

## [3.6.2] - 2026-11-24

### Fixed
- **Camera Shake Anti-Ghosting & IMU Gyroscope Linkage (`LiveBeautyEngine::setCameraShaking`, `FaceMeshFilter`, `FaceTimeline`, `FaceParsingMapper`, `RenderPipeline`)**:
  - Added lock-free `setCameraShaking(bool)` (`nativeSetCameraShaking`) to flush temporal denoise/matte history and suppress stale async ROI/hair/occlusion masks on hardware gyroscope shake.
  - Enforced `HairMaskTuning::affine` displacement bounds (`kAffineMaxShift`) and post-jump pose validity, added 1-frame One-Euro lag snap, non-skin `warmPoly` whitening bleed guard, and 5-tap spatio-temporal `kTemporalDenoise` neighborhood diff.

## [3.6.1] - 2026-11-24

### Fixed
- **Cold-Porcelain De-Yellowing Whitening & Calibrated Non-Darkening Lip Rendering (`kCompositePro`, `kMakeupLip`, `FaceMakeupUv`, `FaceParsingMapper`)**:
  - Upgraded skin whitening with excessive-yellow green-channel suppression, ivory cool-porcelain balance, automatic neck-tone continuity, natural bare-lip complexion lift, true-sRGB lipstick palette calibration, and conservative multiclass skin-vs-occluder margin.

## [3.6.0] - 2026-11-24

### Added
- **6-Class Neural Multiclass Face/Body/Accessory Parsing (`FaceParsingMapper::mapMulticlass6LogitsNhwc`, `LiveBeautyEngine::submitMulticlass6Logits`)**:
  - Added numerically stable 6-way softmax probability mapping for MediaPipe `selfie_multiclass_256x256` NHWC float logits (`background`, `hair`, `body_skin`, `face_skin`, `clothes`, `accessories/hands`) with `α=0.72` temporal EMA, simultaneously feeding `128x128` anti-aliased makeup occlusion, hair protection, and fallback person segmentation masks.
- **Interactive 2D SDF Gesture Particle System & Gesture-Reactive `.lbfx` Bundles (`ParticleBurstSolver`, `ShaderId::kParticleBurst`, `LiveBeautyEngine::triggerGesture`)**:
  - Added zero-texture GPU-batched analytical SDF particle bursts (`heart`, `5-point star`, `diamond`, `4-ray sparkle`) driven by `heart`, `thumbs_up`, `victory`, and `open_palm` hand gestures (`<= 64` quads in 1 indexed draw call), plus `.lbfx` schema v2 gesture triggers (`gesture_heart`, `gesture_thumbs_up`, `gesture_victory`, `gesture_palm`).
- **Studio YCbCr Chroma Key, Hand-Waist Shield & Melanin Hair Dye Upgrade (`ShaderId::kChromaKey`, `ShaderId::kBodyWarp`, `ShaderId::kHairDye`)**:
  - Added green/blue screen YCbCr Chroma Key (`BackgroundMode::kChromaKey`) with luminance-preserving color spill suppression and 5-tap virtual background Light Wrap (`nativeSetChromaKey`).
  - Added dual-hand normalized palm shield (`setHandShieldCoords` / `nativeSetHandShield`) in `kBodyWarp` to prevent waist-slimming lateral pull from bending fingers/hands near the waist.
  - Upgraded `kHairDye` with melanin S-curve bleach lift on dark hair, 2-tap lateral micro-strand detail preservation, and dual-lobe (`R` + `TRT`) Kajiya-Kay Angel-Ring specular sheen.

## [3.5.2] - 2026-11-24

### Fixed
- **Continuous 3D AR Expression Modulation & Roll-Compensated Skin Residual (`LiveBeautyEngine`, `FaceRoiResidualSolver`, `NeuralSkinResidualAdapter`, `FLTLiveBeautyMetalEngine`, `FLTVisionTracker`)**:
  - Removed discrete `exprReaction.trigger != kNone` step gate on `scaleMultiplier` and `offsetYBoost` in `LiveBeautyEngine::processFrame`, ensuring `C¹`-continuous `smoothstep` AR scale and vertical bounce across trigger thresholds.
  - Propagated `headRollRad` into `FaceRoiResidualSolver::compute` and `NeuralSkinResidualAdapter::evaluateMicroKernel` so eye and mouth exemption windows stay aligned under `>45°` head roll.
  - Enforced strict `kCVPixelBufferPoolAllocationThresholdKey` backpressure and explicit `32BGRA`/`32RGBA` format validation in `FLTLiveBeautyMetalEngine::copyProcessedPixelBuffer:`, triggered immediate `doAux` alongside initial segmentation in `FLTVisionTracker`, and refreshed cross-platform KDoc comments in `VideoEffectsConfig.kt`.

## [3.5.1] - 2026-11-24

### Fixed
- **Apple Metal RHI Intra-Frame VBO Slab Arena & Zero-Copy `IOSurface` Lifecycle (`MetalDevice`, `FLTLiveBeautyMetalEngine`, `LiveBeautySDK.podspec`)**:
  - Replaced single-buffer offset-0 `updateVertexBuffer` overwrites with a reusable 256-byte-aligned per-frame dynamic VBO slab arena (`dynamicSlabs`), preventing vertex corruption across multi-slot 2D stickers, multi-slot 3D AR meshes, and dual-face passes within an uncommitted `MTLCommandBuffer`.
  - Added `wrapTransientMetalTexture` (auto-evicted on `flush()`) and `renderToExternalTexture` (`MTLPixelFormat`-keyed PSO cache) with a pooled `CVPixelBufferPool` in `FLTLiveBeautyMetalEngine`, eliminating per-frame `IOSurface` handle leaks and CPU `getBytes:` readback copies.
  - Replaced `kCGImageAlphaPremultipliedLast` asset decoding with raw straight-RGBA extraction (`decodeStraightRgba`) and disabled Xcode `pngcrush` premultiplication (`COMPRESS_PNG_FILES = NO`), preserving `A = 0` packed channel masks in `eye_makeup.png` and straight-alpha RGB in `iris_atlas.png`.
- **C++17 Math, `.lbfx` Bundle Parser & Engine State Hardening (`LiveBeautyEngine`, `RenderPipeline`, `Ar3dSolver`, `EffectBundleParser`)**:
  - Guarded `detail::clampf`/`clampd` against `NaN`, guarded `FaceParsingMapper` & `NeuralSkinResidualAdapter` with 64-bit `size_t` stride arithmetic (`<= 8192`) and neutral handling for `ignore_index = 255`, and corrected mirrored front-camera (`frontFaceCw`) X-axis sign in `FaceRelightSolver` and `Ar3dSolver`.
  - Deep-copied and bounds-validated custom `Ar3dMeshVertex` / index buffers (`validateCustomMesh` + generation-keyed IBO cache), fixed back-to-front painter's triangle order on `kCurvedGlasses` / `kCurvedTiara`, cleared `autoParseHairValid_` on `resetTracking()`, and prioritized 19-class semantic parsing over `64x64` YCbCr aux masks.
  - Upgraded `EffectBundleParser` with locale-independent RFC 8259 decimal parsing, UTF-16 surrogate pair decoding, non-sequential slot support, body/background/pipeline overrides, and `ExpressionTrigger::kTongueOut` without blink-induced `kAlways` opacity dimming.
- **Android JNI, GLES & WebGL2 Hardening (`live_beauty_jni.cpp`, `GlesDevice`, `WebGl2Device`, `LiveBeautyCppEngine`)**:
  - Added 64-bit overflow & array bounds checks, `ExceptionCheck()` UTF-8 guards, safe `nativeReleaseGpu` ordering, `nativeSetAr3dAccessoryEx`, `nativeApplyEffectBundleJsonWithAssets`, and background `RawAuxWorker` offloading on Android.
  - Checked `GL_OES_texture_half_float_linear` before enabling GLES2 OES half-float render targets and drained GL errors before `WebGl2Device` `RGBA8` fallback.

## [3.5.0] - 2026-11-24

### Added
- **Unified C++17 + ARM NEON Face Aux Vision & 52-Channel Expression Bus (`v3.2`, `FaceOcclusionSolver`, `FaceRoiResidualSolver`, `FaceExpressionBus`)**:
  - Ported YCbCr foreground object occlusion detection (`64x64` R8, shielding hands, microphones, cups, and dark frames while preserving dark irises/mouths) and bilateral high-frequency-preserving Face-ROI skin residual (`64x64` RGBA8) into cross-platform C++17 with ARM NEON SIMD acceleration (`submitAuxRgbaSample`).
  - Added thread-safe 52-channel distilled Face Blendshapes expression bus (`submitFaceBlendshapes`) driving reactive 2D/3D AR scale, vertical bounce, and visibility triggers identically across Android, iOS, and Web.
- **Native iOS/macOS Apple Vision Neural Engine (ANE) Tracker (`v3.2`, `FLTVisionTracker`, `FLTLiveBeautyMetalEngine`)**:
  - Added asynchronous `VNGeneratePersonSegmentationRequest` (zero-copy `OneComponent8` `CVPixelBuffer` → `MTLPixelFormatR8Unorm` via `CVMetalTextureCache`) + `VNDetectHumanBodyPoseRequest` (mapped to 33-point top-down body pose) + `128x128` RGBA downsample into C++17 NEON aux (`submitVisionPixelBuffer:frameTimestampMs:`).
- **19-Class Semantic Face-Parsing Mapper & Neural Skin Residual Adapter (`v3.3`, `FaceParsingMapper`, `NeuralSkinResidualAdapter`)**:
  - Added CelebAMask-HQ / BiSeNetV2 / SegFormer 19-class semantic mapper (`submitFaceParsingClassMap` / `submitFaceParsingLogits`) producing temporally smoothed `128x128` R8 occlusion, hair, and person masks.
  - Added NCHW (`3xHxW` / `4xHxW`) external neural skin residual tensor packer (`submitNeuralResidualTensor`) alongside a built-in 2-scale depthwise-separable structural tear-trough/nasolabial shadow-lift micro-kernel.
- **View-Dependent 6-DoF PBR Lip/Iris Glaze, Salon Hair Dyeing & `RGBA16F` Pipeline (`v3.4`, `ShaderId::kHairDye`, `TextureFormat::kRgba16f`)**:
  - Upgraded `kMakeupLip` and `kMakeupIris` with 6-DoF signed-yaw specular belly glide, grazing Fresnel rim sheen, and corneal catchlight 3D dome parallax while keeping frontal (`yaw = 0`) output bit-identical.
  - Added real-time strand-preserving salon hair dyeing (`hairDye`, 6 shades: `milk_tea_brown`, `haze_blue`, `ash_purple`, `rose_gold`, `cool_black`, `caramel_copper`) and anisotropic Kajiya-Kay-inspired crown Angel-Ring sheen (`hairShine`), strictly gated against bare facial skin (`0/255` skin bleed).
  - Promoted intermediate multi-pass accumulation & dual-band guided filter FBOs to `TextureFormat::kRgba16f` (`MTLPixelFormatRGBA16Float` / `GL_RGBA16F` with automatic FBO-completeness fallback) on non-`Lite` tiers while preserving `RGBA8` final output ABI.
- **True 3D Indexed Mesh AR, Declarative `.lbfx` Effect Bundles & WebGL2/Emscripten RHI Backend (`v3.5`, `Ar3dSolver::buildAccessoryMesh`, `EffectBundleParser`, `WebGl2Device`)**:
  - Extended `Ar3dSolver` and `RenderPipeline` to render full 3D indexed meshes (`kCurvedGlasses` with receding `-Z` ear temples, `kCurvedTiara` cylindrical cranium wrap, and custom glTF-baked `Ar3dMeshVertex` buffers).
  - Added zero-dependency C++17 `.lbfx` Effect Bundle JSON manifest loader (`EffectBundleParser`, `applyEffectBundleJson`) supporting preset/beauty overrides, hair dye, 2D stickers, 3D meshes, and expression-gated triggers (`smile`, `mouth_open`, `brow_up`, `blink`, `cheek_puff`, `tongue_out`).
  - Added `WebGl2Device` (`core/include/livebeauty/rhi/webgl/webgl2_device.h`) implementing `IRhiDevice` over WebGL 2.0 / Emscripten (`GL_R8`, `EXT_color_buffer_float` `GL_RGBA16F`, and all 31 shared GLSL passes) for WebRTC / WebCodecs browser pipelines.

## [3.0.0] - 2026-11-23

### Added
- **P1 Studio Portrait Rendering & Optical Realism (`FaceRelightSolver`, `kCompositePro`, `kMakeupLayer`, `kMakeupLip`)**:
  - **Real 3D Vertex Normals & 6-DoF Studio Relighting (`relight`, `relightStyle`)**: Precomputes smooth area-weighted outward 3D vertex normals for all 468 canonical landmarks, estimates orthonormal head pose per frame, and evaluates 4 pose-reactive 3-point studio lighting presets (`natural`, `studio_softbox`, `butterfly`, `rembrandt`) with gentle shadow-fill lift.
  - **T-Zone Oil & Specular Highlight Control (`oilControl`)**: Suppresses washed-out greasy glare on forehead/nose bridge while preserving natural matte skin tone and texture.
  - **Seamless Neck-Face Continuity (`neckTone`)**: Extends a 3-row feathered 13-vertex jawline apron into `rtSkinMask_` with strict skin-chroma gating in `kCompositePro` (0/255 collar bleed).
  - **Ambient Illuminant White-Balance Makeup & Micro-Pore Pass-Through (`ambientMakeupBlend`)**: Adapts blush/contour/eyeshadow/lipstick pigments to the live scene white balance (`setAmbientRgb`), preserves micro-pore skin texture, adds subtle eyelid crease micro-shadowing, and adds 3-band micro-pore preservation to `kCompositePro`.
- **P2 Temporal RGB-Guided Portrait Matting & Bangs/Occlusion Makeup Guard (`kMatteRefine`, `kHairAlpha`, `setOcclusionMask`)**:
  - Added 100% Apache-2.0 GPU RGB-guided 5-tap joint bilateral + motion-adaptive temporal EMA portrait matte refinement (`matteTemporal`) snapping segmentation edges cleanly to real RGB hair/shoulder contours without ghosting.
  - Wired unified `rtSkinMask_.a` (hair + pluggable face-parsing/hand `occlusionMaskTex`) into `kMakeupLayer` and `kMakeupLip` so bangs, hands, cups, and microphones never get painted with makeup.
- **P3 Multi-Scale Isolated Blemish Healer & Pluggable Neural Residual Hook (`kBlemishHeal`, `kAiResidualBlend`)**:
  - Added zero-weight 8-tap ring morphological/chroma isolated acne/dark-spot/redness healer (`blemishRemove`) strictly gated inside `faceM * (1 - exclM) * nonHair` (0/255 impact on clean skin, eyes, brows, lips, or hair).
  - Added pluggable Face-ROI neural residual delta fusion pass (`setAiResidual`) ready for host-supplied TFLite/CoreML portrait enhancement checkpoints.
- **P4 Lightweight 3D Head-Occluded AR Accessories & Hardware Auto-Tiering (`Ar3dSolver`, `SpringPendulum`, `DeviceTierPolicy`)**:
  - Added 468-pt 3D head-occluder depth shell (`kHeadDepthShell`) + perspective-rotated 3D accessory pass (`kAr3dMesh`) with 2nd-order damped spring pendulum physics for dangling earrings/headbands/glasses.
  - Added automatic SoC/GPU tier classifier (`DeviceTierPolicy`: `Lite` / `Standard` / `Ultra`) with runtime load-governor downshift, automatic Android `GL_RENDERER` & Apple `MTLDevice` profiling, and `PerfStats.resolvedTier` telemetry.
- **Expanded Verification Gates**:
  - All 30 C++ GLES shaders verified in both `highp` and forced `mediump` (`kCompositePro` strictly within 15 active uniforms).
  - Added Apple Metal real GPU pixel gates (`P1-P4`) verifying T-zone oil removal, ambient WB makeup, 0/255 bangs/occlusion makeup blocking, isolated acne healing, and 3D head-depth occlusion.

## [2.2.1] - 2026-11-23

### Fixed
- Upgraded `FrameDelayRing` to 0..4 frames, tuned fast-motion head tracking & low-light temporal denoise cutoff (`<= 8/255`, cap `0.18`), and aligned `kCompositePro` LUT/filter exclusion and 2-pass skin mask feathering with Kotlin reference.

## [2.2.0] - 2026-11-23

### Added
- **Hair-Weighted Dual-Band Guided Filter in C++17 Core (`kGuidedWeightBlur`, `kGuidedWeightDiff`, `kGuidedWeightCoeff`)**:
  - Completed hair-weighted guided smoothing in both GLES and Apple Metal shaders: premultiplies 1/3 box blur by `nonHair` alpha weight and blends guided coefficients toward identity `(a=1, b=0)` over hair pixels, eliminating dark hairline/bangs gray-bleed halo into adjacent fair skin.
- **Dual-Face Simultaneous Co-Streaming (`MultiFaceTracker`, `maxFaces = 2`)**:
  - Added optimal minimum-cost 2×2 Euclidean face-slot association (`MultiFaceTracker`) preventing timeline ID swaps when two hosts cross in frame.
  - Added zero-extra-1080p-FBO secondary face rendering across `RenderPipeline` (independent 15-axis diffeomorphic warp, 468-pt anatomical retouch, merged polygon skin mask, 6-layer UV makeup, and 2D AR stickers for both faces simultaneously) + dynamic `numFaces` MediaPipe inference & `nativeSubmitMultiFaceLandmarks`.
- **Cross-Backend Zero-Lag `FrameDelayRing` (`frameDelayFrames = 0..2`)**:
  - Added lazy-allocated RHI `FrameDelayRing` in `RenderPipeline::pushFrameDelay` so both GLES and Metal hosts can delay display frames by 1–2 frames to strictly match asynchronous MediaPipe inference timestamps with zero-allocation passthrough when disabled.
- **Flutter Plugin `v2.2.0` Feature Parity (`live_beauty_sdk`)**:
  - Exposed `LiveBeautyPipelineOptions`, `LiveBeautyVirtualBackgroundMode`, `LiveBeautyStickerAnchor`, `LiveBeautySticker`, `LiveBeautyPerfStats`, and corresponding `LiveBeautyController` + `LiveBeautySdkPlugin` MethodChannel endpoints (`setPipelineOptions`, `setVirtualBackgroundTexture`, `setSticker`, `clearStickers`, `getPerfStats`).
- **Integration Recipes & Expanded GPU Pixel Gates (`H2`, `M1`)**:
  - Added [docs/integration-recipes.md](./docs/integration-recipes.md) covering CameraX OES, WebRTC, Agora, Zego, and Tencent TRTC.
  - Added Apple Metal real GPU pixel probes `H2` (hair-weighted guided filter zero dark-hair bleed) and `M1` (simultaneous dual-face whitening & skin mask).

## [2.1.0] - 2026-11-23

### Added
- **Hair Semantic Protection & Flyaway Preservation in Unified C++17 Core (`HairMaskTuning`, `HairMaskGate`, `kHairAlpha`, `kBgHairMerge`)**:
  - Ported rigid head-motion compensation affine (`HairMaskTuning::affine`), freshness/presence/faceKeep temporal gating, alpha-only skin-mask hair protection (`kHairAlpha`), and person+hair mask flyaway merging (`kBgHairMerge`) into C++17 `core/` for both GLES and Metal backends.
- **Green-Screen-Free Virtual Background Replacement (`BackgroundMode::kVirtualImage / kVirtualBlur`, `kBgVirtualMix`)**:
  - Added centered aspect-fill virtual background replacement and soft-blurred virtual background modes reusing the merged person+hair mask, zero extra inference cost.
- **Face-Anchored 2D AR Sticker Pipeline (`StickerTransform`, `kStickerQuad`)**:
  - Supports up to 4 concurrent face-anchored 2D AR sticker layers (`FOREHEAD_CROWN`, `EYES_BRIDGE`, `NOSE_TIP`, `CHEEKS`, `FULL_FACE`) driven by post-warp 478-pt landmarks with rigid scale/rotation tracking and zero extra 1080p FBO allocation.
- **One-Step Android `GL_TEXTURE_EXTERNAL_OES` Direct Ingestion (`GlesDevice::convertOesTo2D`, `nativeProcessOesFrame`, `LiveBeautyProcessor.processOes`)**:
  - Allows CameraX / Camera2 / WebRTC `SurfaceTexture` OES textures + 4×4 transform matrices to enter the C++17 pipeline directly without host-side OES→2D conversion boilerplate.
- **Real-Time Telemetry (`PerfStats`, `nativeGetStats`, `BeautyPerfStats`) & Default C++17 Core in `LiveBeautyProcessor`**:
  - Exposes per-frame latency (`lastFrameMs`, `emaFrameMs`), `fps`, `loadLevel`, `facePresence`, `hairShown`, `activePassesMask`, and `frameCount`.
  - `LiveBeautyProcessor` now enables the unified C++17 kernel (`LiveBeautyCppEngine`) by default when `liblivebeauty_jni.so` is present, with seamless zero-crash fallback on x86_64 emulators / pure JVM environments.
- **Standard `maven-publish` Configuration & CPU Micro-Benchmark (`livebeauty_bench`)**:
  - Added Gradle `maven-publish` release publication (`io.bebebus:live-beauty:2.1.0` + sourcesJar + Apache-2.0 POM) for one-command local Maven / JitPack / Maven Central publishing.
  - Added `livebeauty_bench` verifying the full C++17 CPU pipeline (`FaceTimeline` + 478-pt `FaceMeshFilter` + 548-vertex 15-axis `FaceMeshDeformer` + `StickerTransform`) runs in **~55 μs/frame** on Apple Silicon (~18,000 fps theoretical CPU ceiling).
  - Expanded Apple Metal GPU pixel gate (`livebeauty_metal_pixel_gate`) with `V1/H1` (virtual background + hair flyaway preservation) and `S1` (2D AR sticker alpha overlay) probes.

## [2.0.0] - 2026-11-23

### Added
- **Unified C++17 Math & Render Core (`core/`)**:
  - **Pure C++17 Math & State-Machine Core (`livebeauty_math`, zero third-party runtime dependencies)**: Consolidated all core algorithms into a single cross-platform source of truth — `FaceMeshTopology` (deterministically co-generated and verified alongside Kotlin via `reference/gen_face_topology.py`), 15-axis $C^\infty$ diffeomorphic `FaceMeshDeformer` (including `cranialTop`, Boltzmann SoftMax half-width symmetry, and Jacobian barrier guard), `FaceMeshFilter` (Procrustes weak-perspective + 3-region One-Euro + `LID_RING`/`LIP_RING`/`EYE_REGION` boost), `FaceTimeline` (including 35ms velocity-aware `eyeMakeupKeep`), `FaceRoiSelector`, `FaceMakeupUv` (calibrated palettes + 468-pt anatomical sub-mesh packing), `BodyGeometry` (5-axis reshape + `NeckLift`), and `EffectsLoadGovernor`.
  - **Lightweight Cross-Backend RHI & RenderGraph Orchestrator (`livebeauty_engine`)**: Introduced `IRhiDevice`, `RenderPipeline`, and `LiveBeautyEngine` unifying all 8 beauty passes (portrait bokeh, 5-axis body warp, painter's-order 548-vertex diffeomorphic face warp, in-place 468-pt anatomical sub-mesh retouch, 504-vertex polygon skin mask, feedback-loop-isolated `1/3 + 1/9` dual-band guided smoothing, 15-uniform `CompositePro` + `64^3` 3D LUT, 6-layer physically-based UV makeup, and adaptive temporal denoise).
  - **Android OpenGL ES 2.0/3.0 Backend & NDK JNI Bridge (`livebeauty_gles` + `livebeauty_jni`)**: Integrated `core/CMakeLists.txt` into `:live-beauty` Gradle build producing dual-ABI (`arm64-v8a` / `armeabi-v7a`) `liblivebeauty_jni.so` inside the Release AAR, exposed via `NativeBeautyBridge`.
  - **Apple Metal RHI Backend, Equivalent Hand-Tuned MSL Shaders & Full-Parity iOS Engine (`livebeauty_metal` + `FLTLiveBeautyMetalEngine`)**: Implemented `MetalDevice`, complete hand-tuned MSL shader suite matching Android GLSL formulas, `FLTLiveBeautyMetalEngine` (`MTLTexture` & `CVMetalTextureCache` zero-copy `CVPixelBuffer` bridge), and CocoaPods integration in `LiveBeautySDK.podspec` while keeping existing `FLTVideoEffects` CoreImage APIs backward-compatible.
  - **Cross-Platform CTest & Apple Metal GPU Pixel Gates (`_tooling/run-all-tests.sh` `[8/8]` & GitHub Actions CI)**: Added `livebeauty_math_tests`, `livebeauty_rhi_tests` (Mock RHI pass ordering & zero-feedback-loop checks), and `livebeauty_metal_pixel_gate` running real Apple Metal GPU probes (`L1/L2/L6`, `E1–E4`, `R1`, and end-to-end `koreanWaterGlow` rendering).

## [1.1.4] - 2026-11-23

### Fixed
- **Subtractive True-to-Life Lip Color & Bright-Camera Anti-Pink Calibration (`MAKEUP_LIP_FRAGMENT`, `FaceMakeupUv`)**:
  - Replaced the `smoothY / 0.44` illumination boost and `normC <= 1.45` channel brightening with a calibrated Beer-Lambert subtractive stain (`stainTint <= 1.02`) + neutral `0.54` scene illumination with soft highlight compression (`<= 1.04`), stopping well-lit front cameras and beauty LUTs (`Y = 0.55..0.68`) from over-exposing rich lip shades into glowing pastel pink.
  - Raised the glossy `naturalHot` specular threshold from `0.48` to `0.78` and tightened lower-lip/cupid-bow catchlight bands (`145 / 165`), eliminating full-lip milky white wash on glossy shades so 90% of the lip body keeps juicy, wet-deepened pigment (`0.88×`) with crisp localized wet glints.
  - Recalibrated all 6 lipstick shades (`rose_bean`, `coral_peach`, `cherry_red`, `milk_tea`, `berry_plum`, `orange_soda`) to deeper, warm-neutral realistic pigment values (eliminating `B > G` cool-pink drift on `rose_bean` and `cherry_red`).
  - Added GPU regression probe `L6` (`_tooling/glsl/`) locking bright front-camera (`(210,152,150)`) glossy `rose_bean` and `cherry_red` against pastel/milky pink washout.

## [1.1.3] - 2026-11-23

### Fixed
- **Zero-Ghosting Eyeshadow Tracking & Eyelid-Skin Pixel Gate (`MAKEUP_LAYER_FRAGMENT`, `MakeupPass`, `FaceTimeline`, `FaceMeshFilter`, `eye_makeup.png`)**:
  - Added a zero-extra-uniform eyelid-skin protection gate (`uMode.z > 0.5`) to `MAKEUP_LAYER_FRAGMENT` combining `darkGate` (shields dark iris, lashes, and eyebrows) and `skinGate` (shields white/shadowed sclera and neutral background walls), alongside a soft-powder `Multiply + SoftLight` blend that replaces muddy double-`Multiply` darkening.
  - Replaced the 1-pixel binary eye-hole cutout in `eye_makeup.png` with smooth exterior eye-contour feathering (`0.0005..0.0075` UV), strict upper-eyelid containment (`upper_only`), browbone fade-out (`brow_guard`), and tighter crease-hugging Gaussian radii (`σ = 0.0165 / 0.010`).
  - Added a dedicated `FaceTimeline.eyeMakeupKeep` anti-ghosting gain for fine eye makeup (`eyeshadow`, `eyeliner`, `aegyoSal`) driven by both face speed (`0.14..0.42 sw/s`) and tight extrapolation distance (`0.12..0.35 * eye`), with 1-frame fast fade-out (`35 ms`) and smooth recovery (`160 ms`) while keeping the existing `steadyEye` contract intact.
  - Raised `FaceMeshFilter` `EYE_REGION` velocity gain (`RES_BETA[1] = 40f → 75f`) while keeping `RES_DCUT = 1.5f` calibrated so stationary jitter stays `≤ 0.8 px@720`, and shifted `eyeMember` upward to cover the upper-eyelid crease and browbone canvas as a single unified filter region.
  - Added WebGL shader regression probes `E1–E4` (`_tooling/glsl/`) verifying natural eyelid powder tinting (`E1`) and exact `0/255` staining on bright/shadowed sclera (`E2`), dark iris/lashes (`E3`), and neutral grey backgrounds (`E4`), plus `FaceTimelineTest.eyeMakeupKeepStaysFullWhenStillAndFadesFastOnPan`.

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
