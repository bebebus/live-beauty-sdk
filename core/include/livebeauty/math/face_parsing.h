#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "livebeauty/types.h"

namespace livebeauty::math {

/**
 * Standard 19-class CelebAMask-HQ / BiSeNetV2 / SegFormer semantic face-parsing taxonomy
 * (plus class 19 reserved for foreground hand/object occluders).
 */
enum class FaceParsingClass : uint8_t {
    kBackground = 0,
    kSkin       = 1,
    kLeftBrow   = 2,
    kRightBrow  = 3,
    kLeftEye    = 4,
    kRightEye   = 5,
    kEyeglasses = 6,
    kLeftEar    = 7,
    kRightEar   = 8,
    kEarring    = 9,
    kNose       = 10,
    kInnerMouth = 11,  // Oral cavity & teeth
    kUpperLip   = 12,
    kLowerLip   = 13,
    kNeck       = 14,
    kNecklace   = 15,
    kCloth      = 16,
    kHair       = 17,
    kHat        = 18,
    kHand       = 19,
    kNumClasses = 20,
};

/**
 * C++17 + ARM NEON 19-class semantic face-parsing mapper (`v3.3`).
 * Converts either a raw `W x H` `uint8_t` class-ID map or a `C x H x W` float logit tensor into
 * temporally smoothed, anti-aliased `128x128` R8 masks:
 * - `occlusionMaskR8` (`255` = paintable face/lips/brows/eyes, `0` = eyeglasses, inner-mouth teeth, hair bangs, hat, necklace, cloth, hand);
 * - `hairMaskR8` (`255` = hair strands, `0` = non-hair);
 * - `personMaskR8` (`255` = foreground person `1..18`, `0` = background).
 */
class FaceParsingMapper {
public:
    static constexpr int kOutSize = 128;
    static constexpr int kOutPixels = kOutSize * kOutSize;

    FaceParsingMapper();
    void reset();

    /** Convert `srcW x srcH` `uint8_t` class-ID map (`0..19`) into smoothed `128x128` semantic masks. */
    bool mapClassIds(
        const uint8_t* classIds,
        int srcW,
        int srcH,
        uint8_t* outOcclusionR8,
        uint8_t* outHairR8 = nullptr,
        uint8_t* outPersonR8 = nullptr);

    /** Argmax `numClasses x srcH x srcW` float NCHW logits (`numClasses >= 19`) and map into `128x128` masks. */
    bool mapLogitsNchw(
        const float* logitsNchw,
        int numClasses,
        int srcW,
        int srcH,
        uint8_t* outOcclusionR8,
        uint8_t* outHairR8 = nullptr,
        uint8_t* outPersonR8 = nullptr);

    /**
     * v3.6/v3.9.4: Softmax-map 6-class MediaPipe Selfie Multiclass NHWC logits (`srcH x srcW x 6`:
     * `[0=bg, 1=hair, 2=body_skin (incl. hands/arms), 3=face_skin, 4=clothes, 5=others/accessories]`)
     * into continuous anti-aliased `128x128` R8 occlusion, hair, and person masks.
     * Optional `primaryFaceBounds`/`secondaryFaceBounds` (top-down normalized) mark `body_skin`
     * dominant pixels inside the inner face ellipse as hand occluders for makeup (C1/makeup-1/seg-1).
     */
    bool mapMulticlass6LogitsNhwc(
        const float* logitsNhwc6,
        int srcW,
        int srcH,
        uint8_t* outOcclusionR8,
        uint8_t* outHairR8 = nullptr,
        uint8_t* outPersonR8 = nullptr,
        const CropRect* primaryFaceBounds = nullptr,
        const CropRect* secondaryFaceBounds = nullptr);

    /**
     * v3.9.4 (E2/cpuperf-6/seg-5/seg-8): Direct letterbox-region NHWC logits mapping with 2x2 box
     * sampling directly inside `[padX, padX+validW) x [padY, padY+validH)` (eliminates host unletterbox).
     */
    bool mapMulticlass6LogitsNhwcRegion(
        const float* logitsNhwc6,
        int srcW,
        int srcH,
        int padX,
        int padY,
        int validW,
        int validH,
        uint8_t* outOcclusionR8,
        uint8_t* outHairR8 = nullptr,
        uint8_t* outPersonR8 = nullptr,
        const CropRect* primaryFaceBounds = nullptr,
        const CropRect* secondaryFaceBounds = nullptr);

    /**
     * v3.9: Direct planar probability ingestion for iOS/Android MediaPipe `ImageSegmenter`
     * `selfie_multiclass_256x256.tflite` 6 confidence mask planes `[0=bg..5=accessories]` in `[0..1]`.
     */
    bool mapMulticlass6PlanarProbabilities(
        const float* const planes[6],
        int srcW,
        int srcH,
        uint8_t* outOcclusionR8,
        uint8_t* outHairR8 = nullptr,
        uint8_t* outPersonR8 = nullptr,
        const CropRect* primaryFaceBounds = nullptr,
        const CropRect* secondaryFaceBounds = nullptr);

    /** Estimate occluded fraction [0..1] inside the lower/mid face region of a top-down R8 visibility mask. */
    static float estimateLowerFaceOcclusion(
        const uint8_t* occMaskR8,
        int maskSize,
        const CropRect& faceBounds);

private:
    std::array<uint8_t, kOutPixels> rawOcc_{};
    std::array<uint8_t, kOutPixels> rawHair_{};
    std::array<uint8_t, kOutPixels> rawPerson_{};
    std::array<float, kOutPixels> emaOcc_{};
    std::array<float, kOutPixels> emaHair_{};
    std::array<float, kOutPixels> emaPerson_{};
    std::vector<uint8_t> argmaxScratch_;
    bool hasOccHistory_ = false;
    bool hasHairHistory_ = false;
    bool hasPersonHistory_ = false;
};

/**
 * v3.3 Micro-UNet / Depthwise-Separable Neural Residual Adapter & Built-in Multi-Scale Retouch Kernel.
 * - `packExternalTensorNchw`: Quantizes external ONNX/CoreML/TFLite `1 x 3 x H x W` float delta tensor
 *   (in `[-0.25, +0.25]`) + optional confidence channel into `64x64` RGBA8 (`128` = zero delta).
 * - `evaluateMicroKernel`: Runs a fast 2-scale depthwise-separable frequency-decomposed residual pass
 *   targeting structural tear-trough / nasolabial mid-band shadows (`4-10px`) while strictly preserving
 *   `1-2px` micro-pores (`0 KB` external weight dependency, Apache-2.0).
 */
class NeuralSkinResidualAdapter {
public:
    static constexpr int kOutSize = 64;
    static constexpr int kOutPixels = kOutSize * kOutSize;

    /** Pack external `3 x srcH x srcW` (or `4 x srcH x srcW` with alpha confidence) float NCHW tensor into `64x64` RGBA8. */
    static bool packExternalTensorNchw(
        const float* tensorNchw,
        int channels,
        int srcW,
        int srcH,
        uint8_t* outRgba64x64);

    /** Run built-in 2-scale depthwise-separable structural shadow & blemish residual micro-kernel on top-down RGBA frame. */
    bool evaluateMicroKernel(
        const uint8_t* rgbaTopDown,
        int srcWidth,
        int srcHeight,
        const CropRect& faceBounds,
        float strength,
        uint8_t* outRgba64x64,
        float headRollRad = 0.0f);

private:
    std::array<float, kOutPixels> lumaFine_{};
    std::array<float, kOutPixels> lumaMid_{};
    std::array<float, kOutPixels> lumaCoarse_{};
    std::array<float, kOutPixels> rPlane_{};
    std::array<float, kOutPixels> gPlane_{};
    std::array<float, kOutPixels> bPlane_{};
};

}  // namespace livebeauty::math
