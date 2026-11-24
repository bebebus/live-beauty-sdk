#pragma once

#include <array>
#include <cstdint>

#include "livebeauty/types.h"

namespace livebeauty::math {

/**
 * 52-channel MediaPipe Face Blendshapes distilled input for reactive 2D/3D AR effects.
 */
struct BlendshapeInput {
    float smile = 0.0f;
    float jawOpen = 0.0f;
    float browUp = 0.0f;
    float blink = 0.0f;
    float cheekPuff = 0.0f;
    float tongueOut = 0.0f;
};

enum class ExpressionTrigger : uint8_t {
    kNone = 0,
    kSmile = 1,
    kMouthOpen = 2,
    kBrowRaise = 3,
    kBlink = 4,
    kCheekPuff = 5,
};

struct ExpressionReaction {
    float scaleMultiplier = 1.0f;
    float offsetYBoost = 0.0f;
    float opacityBoost = 1.0f;
    ExpressionTrigger trigger = ExpressionTrigger::kNone;
    int spriteFrameOffset = 0;
    float smile = 0.0f;
    float mouthOpen = 0.0f;
    float browUp = 0.0f;
    float blink = 0.0f;
    float cheekPuff = 0.0f;
    float tongueOut = 0.0f;
};

/**
 * Temporal EMA state machine converting raw per-frame blendshapes into smooth
 * 3D AR spring/scale reactions and 2D animated sticker frame triggers.
 */
class FaceExpressionBus {
public:
    void reset();
    ExpressionReaction update(const BlendshapeInput* sample);
    const ExpressionReaction& latest() const { return latest_; }

private:
    float smSmile_ = 0.0f;
    float smJaw_ = 0.0f;
    float smBrow_ = 0.0f;
    float smBlink_ = 0.0f;
    float smPuff_ = 0.0f;
    float smTongue_ = 0.0f;
    int animTick_ = 0;
    ExpressionReaction latest_{};
};

/**
 * C++17 + ARM NEON foreground foreign-object occlusion detector (`64x64` top-down R8 mask).
 * `255` = visible face skin, `0` = occluded by microphone, cup, phone, or hand.
 * Explicitly exempts dark pupils/eyelashes and inner mouth cavities so natural facial features
 * never trigger false occlusion.
 */
class FaceOcclusionSolver {
public:
    static constexpr int kMaskSize = 64;
    static constexpr int kMaskPixels = kMaskSize * kMaskSize;

    FaceOcclusionSolver();
    void reset();

    /** Extract padded normalized [0..1] face bounding box from 468/478-pt interleaved landmarks. */
    static bool boundsFromLandmarks(const float* mpXy, int pointCount, CropRect& outBounds);

    /** Compute top-down 64x64 R8 visibility mask (`255` = visible, `0` = occluded). */
    void compute(
        const uint8_t* rgbaTopDown,
        int srcWidth,
        int srcHeight,
        const CropRect* faceBounds,
        uint8_t* outMaskR8);

private:
    void fillNeutral(uint8_t* outMaskR8);

    std::array<uint8_t, kMaskPixels> rawOcc_{};
    std::array<uint8_t, kMaskPixels> openedOcc_{};
    std::array<float, kMaskPixels> emaMask_{};
    bool hasHistory_ = false;
};

/**
 * C++17 + ARM NEON Face-ROI high-frequency-preserving skin retouch residual solver (`64x64` RGBA8).
 * Encodes `RGB` with `128` = zero delta (`delta = (rgb / 255 - 0.5) * 0.50` in `kAiResidualBlend`)
 * and `A` = skin confidence mask (`0` over eyes, brows, lips, and background).
 */
class FaceRoiResidualSolver {
public:
    static constexpr int kRoiSize = 64;
    static constexpr int kRoiPixels = kRoiSize * kRoiSize;

    bool compute(
        const uint8_t* rgbaTopDown,
        int srcWidth,
        int srcHeight,
        const CropRect& faceBounds,
        float retouchStrength,
        uint8_t* outRgba64x64);

private:
    std::array<float, kRoiPixels> patchR_{};
    std::array<float, kRoiPixels> patchG_{};
    std::array<float, kRoiPixels> patchB_{};
};

}  // namespace livebeauty::math
