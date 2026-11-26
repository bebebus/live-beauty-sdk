#pragma once

#include <string>
#include "livebeauty/math/device_tier.h"
#include "livebeauty/math/face_relight.h"

namespace livebeauty {

enum class ColorFilterStyle : int {
    kNone = 0,
    kWarm = 1,
    kCool = 2,
    kMonochrome = 3,
    kSepia = 4,
    kRose = 5,
};

enum class LipstickColor : int {
    kRoseBean = 0,
    kCoralPeach = 1,
    kCherryRed = 2,
    kMilkTea = 3,
    kBerryPlum = 4,
    kOrangeSoda = 5,
};

enum class LipstickFinish : int {
    kVelvet = 0,
    kGlossy = 1,
};

enum class BlushStyle : int {
    kApple = 0,
    kDrunk = 1,
    kUnderEye = 2,
};

enum class BlushColor : int {
    kPeach = 0,
    kRose = 1,
    kApricot = 2,
    kBerry = 3,
};

enum class EyebrowStyle : int {
    kNatural = 0,
    kArched = 1,
};

enum class EyebrowColor : int {
    kDarkBrown = 0,
    kSoftBrown = 1,
    kGreyBlack = 2,
};

enum class EyeshadowColor : int {
    kEarth = 0,
    kPeachPink = 1,
    kRoseWine = 2,
    kChampagne = 3,
};

enum class ContactLensStyle : int {
    kHazel = 0,
    kChocolate = 1,
    kMistGrey = 2,
    kSakuraPink = 3,
};

enum class HairDyeColor : int {
    kMilkTeaBrown = 0,
    kHazeBlue = 1,
    kAshPurple = 2,
    kRoseGold = 3,
    kCoolBlack = 4,
    kCaramelCopper = 5,
};

enum class BackgroundMode : int {
    kNone = 0,          // Follow backgroundBlur (>0 => bokeh blur)
    kBlur = 1,          // Portrait bokeh blur
    kVirtualImage = 2,  // Green-screen-free virtual background replacement
    kVirtualBlur = 3,   // Virtual background image with soft bokeh blur
    kChromaKey = 4,     // Studio green/blue screen chroma key + edge despill
};

enum class ChromaKeyColor : int {
    kGreen = 0,
    kBlue = 1,
};

struct BeautyConfig {
    bool enabled = true;
    float smoothing = 0.0f;
    float brightness = 0.0f;
    ColorFilterStyle filter = ColorFilterStyle::kNone;
    bool skinOnly = false;
    float sharpness = 0.0f;
    float saturation = 1.0f;
    float filterIntensity = 1.0f;
    bool compareOriginal = false;

    // Basic face & skin tone
    float faceSlim = 0.0f;
    float eyeEnlarge = 0.0f;
    float whiten = 0.0f;
    float rosy = 0.0f;

    // 5-axis body reshape
    float waistSlim = 0.0f;
    float bustEnhance = 0.0f;
    float legLengthen = 0.0f;
    float shoulderSlim = 0.0f;
    float neckLengthen = 0.0f;

    // 15-axis 478-pt face & cranial reshape
    float chinShape = 0.0f;
    float foreheadShape = 0.0f;
    float noseSlim = 0.0f;
    float vFace = 0.0f;
    float faceNarrow = 0.0f;
    float faceSmall = 0.0f;
    float cheekboneSlim = 0.0f;
    float eyeDistance = 0.0f;
    float eyeCorner = 0.0f;
    float noseLength = 0.0f;
    float mouthSize = 0.0f;
    float mouthSmile = 0.0f;
    float cranialTop = 0.0f;

    // 468-pt anatomical sub-mesh retouch
    float darkCircles = 0.0f;
    float brightEyes = 0.0f;
    float teethWhiten = 0.0f;
    float nasolabial = 0.0f;

    // Portrait bokeh, virtual background, studio chroma key & 3D LUT
    float backgroundBlur = 0.0f;
    BackgroundMode backgroundMode = BackgroundMode::kNone;
    float virtualBgStrength = 1.0f;
    ChromaKeyColor chromaKeyColor = ChromaKeyColor::kGreen;
    float chromaKeySimilarity = 0.38f;
    float chromaKeySmoothness = 0.12f;
    float chromaKeyDespill = 0.75f;
    int maxFaces = 1;           // 1 = single primary face (default), 2 = dual-face co-streaming
    int frameDelayFrames = 0;   // 0 = lowest latency, 1..4 = FrameDelayRing zero-lag alignment
    std::string lutUrl;
    float lutIntensity = 1.0f;

    // Texture preservation, radiance & P1-P4 studio enhancements
    float skinTexture = 0.0f;
    float radiance = 0.0f;
    float relight = 0.0f;                                             // P1: 3D normal studio relighting [0..1]
    math::RelightStyle relightStyle = math::RelightStyle::kStudioSoftbox;
    float oilControl = 0.0f;                                          // P1: T-zone oil/specular removal [0..1]
    float neckTone = 0.0f;                                            // P1: neck-face tone continuity [0..1] (auto-assisted when whiten/smoothing > 0)
    float ambientMakeupBlend = 0.55f;                                 // P1: ambient white-balance adaptation for makeup [0..1]
    float matteTemporal = 0.65f;                                      // P2: RGB-guided temporal portrait matting stabilization [0..1]
    float blemishRemove = 0.0f;                                       // P3: multi-scale isolated blemish/redness residual removal [0..1]
    math::PerformanceTier performanceTier = math::PerformanceTier::kAuto;  // P4: Lite/Standard/Ultra hardware auto-tiering

    // 6-layer real-time UV makeup
    float lipstick = 0.0f;
    LipstickColor lipstickColor = LipstickColor::kRoseBean;
    LipstickFinish lipstickFinish = LipstickFinish::kVelvet;
    float blush = 0.0f;
    BlushStyle blushStyle = BlushStyle::kApple;
    BlushColor blushColor = BlushColor::kPeach;
    float eyebrow = 0.0f;
    EyebrowStyle eyebrowStyle = EyebrowStyle::kNatural;
    EyebrowColor eyebrowColor = EyebrowColor::kDarkBrown;
    float eyeshadow = 0.0f;
    EyeshadowColor eyeshadowColor = EyeshadowColor::kEarth;
    float eyeliner = 0.0f;
    float aegyoSal = 0.0f;
    float contour = 0.0f;
    float highlight = 0.0f;
    float contactLens = 0.0f;
    ContactLensStyle contactLensStyle = ContactLensStyle::kHazel;
    float catchlight = 0.0f;

    // v3.4: Real-time strand-preserving hair dyeing, crown Angel-Ring sheen & RGBA16F intermediate pipeline
    float hairDye = 0.0f;
    HairDyeColor hairDyeColor = HairDyeColor::kMilkTeaBrown;
    float hairShine = 0.0f;
    bool preferHalfFloat = true;

    // v3.8: Cinematic atmosphere post-processing (soft-knee pyramid bloom, non-darkening filmic grain, CIE D65 CAT02 WB)
    float bloom = 0.0f;            // [0..1] soft-knee highlight glow
    float filmGrain = 0.0f;        // [0..1] organic non-darkening filmic grain
    float colorTemp = 0.0f;        // [-1..+1] CAT02 chromatic adaptation temperature
    float colorTint = 0.0f;        // [-1..+1] CAT02 green/magenta tint
    // v3.9: Automatic ambient illuminant CAT02 white-balance correction driven by live face-skin ROI
    float autoWhiteBalance = 0.0f; // [0..1] damped auto correction for tungsten/fluorescent room lighting

    bool hasFaceReshape() const;
    bool hasBodyReshape() const;
    bool hasRetouch() const;
    bool hasMakeup() const;
    bool needsFaceTracking() const;
    bool isProcessing() const;

    static BeautyConfig original();
    static BeautyConfig naturalStudio();
    static BeautyConfig clearColdWhite();
    static BeautyConfig koreanWaterGlow();
    static BeautyConfig mensClean();
};

ColorFilterStyle parseFilterStyle(const std::string& s);
LipstickColor parseLipstickColor(const std::string& s);
LipstickFinish parseLipstickFinish(const std::string& s);
BlushStyle parseBlushStyle(const std::string& s);
BlushColor parseBlushColor(const std::string& s);
EyebrowStyle parseEyebrowStyle(const std::string& s);
EyebrowColor parseEyebrowColor(const std::string& s);
EyeshadowColor parseEyeshadowColor(const std::string& s);
ContactLensStyle parseContactLensStyle(const std::string& s);
HairDyeColor parseHairDyeColor(const std::string& s);
BackgroundMode parseBackgroundMode(const std::string& s);
ChromaKeyColor parseChromaKeyColor(const std::string& s);
math::RelightStyle parseRelightStyle(const std::string& s);
math::PerformanceTier parsePerformanceTier(const std::string& s);

}  // namespace livebeauty
