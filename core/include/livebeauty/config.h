#pragma once

#include <string>

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

    // Portrait bokeh & 3D LUT
    float backgroundBlur = 0.0f;
    std::string lutUrl;
    float lutIntensity = 1.0f;

    // Texture preservation & radiance
    float skinTexture = 0.0f;
    float radiance = 0.0f;

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

}  // namespace livebeauty
