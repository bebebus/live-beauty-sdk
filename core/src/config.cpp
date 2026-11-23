#include "livebeauty/config.h"

#include <cmath>

namespace livebeauty {

bool BeautyConfig::hasFaceReshape() const {
    return faceSlim > 0.0f || eyeEnlarge > 0.0f || vFace > 0.0f || faceNarrow > 0.0f || faceSmall > 0.0f ||
           cheekboneSlim > 0.0f || chinShape != 0.0f || foreheadShape != 0.0f || cranialTop > 0.0f ||
           eyeDistance != 0.0f || eyeCorner != 0.0f || noseSlim > 0.0f ||
           noseLength != 0.0f || mouthSize != 0.0f || mouthSmile > 0.0f;
}

bool BeautyConfig::hasBodyReshape() const {
    return waistSlim > 0.0f || bustEnhance > 0.0f || legLengthen > 0.0f ||
           shoulderSlim > 0.0f || neckLengthen > 0.0f;
}

bool BeautyConfig::hasRetouch() const {
    return darkCircles > 0.0f || brightEyes > 0.0f || teethWhiten > 0.0f || nasolabial > 0.0f;
}

bool BeautyConfig::hasMakeup() const {
    return lipstick > 0.0f || blush > 0.0f || eyebrow > 0.0f || eyeshadow > 0.0f || eyeliner > 0.0f ||
           aegyoSal > 0.0f || contour > 0.0f || highlight > 0.0f || contactLens > 0.0f || catchlight > 0.0f;
}

bool BeautyConfig::needsFaceTracking() const {
    if (!enabled || compareOriginal) return false;
    return hasFaceReshape() || hasRetouch() || hasMakeup() ||
           (smoothing > 0.0f && skinOnly) || whiten > 0.0f || rosy > 0.0f ||
           radiance > 0.0f || relight > 0.0f || blemishRemove > 0.0f;
}

bool BeautyConfig::isProcessing() const {
    if (!enabled || compareOriginal) return false;
    return smoothing > 0.0f || brightness != 0.0f || filter != ColorFilterStyle::kNone ||
           sharpness > 0.0f || std::fabs(saturation - 1.0f) > 1e-4f || whiten > 0.0f ||
           rosy > 0.0f || radiance > 0.0f || relight > 0.0f || oilControl > 0.0f ||
           blemishRemove > 0.0f || backgroundBlur > 0.0f ||
           backgroundMode == BackgroundMode::kVirtualImage ||
           backgroundMode == BackgroundMode::kVirtualBlur ||
           !lutUrl.empty() ||
           hasFaceReshape() || hasBodyReshape() || hasRetouch() || hasMakeup();
}

BeautyConfig BeautyConfig::original() {
    BeautyConfig c;
    c.enabled = false;
    return c;
}

BeautyConfig BeautyConfig::naturalStudio() {
    BeautyConfig c;
    c.enabled = true;
    c.smoothing = 0.52f;
    c.brightness = 0.10f;
    c.filter = ColorFilterStyle::kNone;
    c.skinOnly = true;
    c.skinTexture = 0.45f;
    c.whiten = 0.32f;
    c.rosy = 0.16f;
    c.radiance = 0.25f;
    c.sharpness = 0.20f;
    c.faceSlim = 0.35f;
    c.vFace = 0.25f;
    c.eyeEnlarge = 0.28f;
    c.noseSlim = 0.20f;
    c.darkCircles = 0.35f;
    c.brightEyes = 0.30f;
    c.teethWhiten = 0.25f;
    c.lutUrl = "asset://luts/natural_studio.png";
    c.lutIntensity = 0.45f;
    return c;
}

BeautyConfig BeautyConfig::clearColdWhite() {
    BeautyConfig c = naturalStudio();
    c.smoothing = 0.68f;
    c.skinTexture = 0.38f;
    c.whiten = 0.58f;
    c.rosy = 0.22f;
    c.radiance = 0.38f;
    c.faceSlim = 0.52f;
    c.vFace = 0.42f;
    c.faceSmall = 0.20f;
    c.eyeEnlarge = 0.42f;
    c.lipstick = 0.45f;
    c.lipstickColor = LipstickColor::kRoseBean;
    c.lipstickFinish = LipstickFinish::kGlossy;
    c.blush = 0.30f;
    c.blushStyle = BlushStyle::kApple;
    c.blushColor = BlushColor::kPeach;
    c.catchlight = 0.35f;
    c.lutUrl = "asset://luts/clear_white.png";
    c.lutIntensity = 0.55f;
    return c;
}

BeautyConfig BeautyConfig::koreanWaterGlow() {
    BeautyConfig c = clearColdWhite();
    c.rosy = 0.30f;
    c.radiance = 0.52f;
    c.aegyoSal = 0.40f;
    c.eyeshadow = 0.32f;
    c.eyeshadowColor = EyeshadowColor::kPeachPink;
    c.eyeliner = 0.28f;
    c.contactLens = 0.38f;
    c.contactLensStyle = ContactLensStyle::kHazel;
    c.catchlight = 0.48f;
    c.lutUrl = "asset://luts/korean_glow.png";
    c.lutIntensity = 0.50f;
    return c;
}

BeautyConfig BeautyConfig::mensClean() {
    BeautyConfig c;
    c.enabled = true;
    c.smoothing = 0.38f;
    c.brightness = 0.08f;
    c.filter = ColorFilterStyle::kNone;
    c.skinOnly = true;
    c.skinTexture = 0.65f;
    c.whiten = 0.18f;
    c.sharpness = 0.25f;
    c.faceSlim = 0.20f;
    c.cheekboneSlim = 0.15f;
    c.noseSlim = 0.15f;
    c.darkCircles = 0.42f;
    c.brightEyes = 0.25f;
    c.lutUrl = "asset://luts/natural_studio.png";
    c.lutIntensity = 0.35f;
    return c;
}

ColorFilterStyle parseFilterStyle(const std::string& s) {
    if (s == "warm") return ColorFilterStyle::kWarm;
    if (s == "cool") return ColorFilterStyle::kCool;
    if (s == "monochrome") return ColorFilterStyle::kMonochrome;
    if (s == "sepia") return ColorFilterStyle::kSepia;
    if (s == "rose") return ColorFilterStyle::kRose;
    return ColorFilterStyle::kNone;
}

LipstickColor parseLipstickColor(const std::string& s) {
    if (s == "coral_peach") return LipstickColor::kCoralPeach;
    if (s == "cherry_red") return LipstickColor::kCherryRed;
    if (s == "milk_tea") return LipstickColor::kMilkTea;
    if (s == "berry_plum") return LipstickColor::kBerryPlum;
    if (s == "orange_soda") return LipstickColor::kOrangeSoda;
    return LipstickColor::kRoseBean;
}

LipstickFinish parseLipstickFinish(const std::string& s) {
    return (s == "glossy") ? LipstickFinish::kGlossy : LipstickFinish::kVelvet;
}

BlushStyle parseBlushStyle(const std::string& s) {
    if (s == "drunk") return BlushStyle::kDrunk;
    if (s == "under_eye") return BlushStyle::kUnderEye;
    return BlushStyle::kApple;
}

BlushColor parseBlushColor(const std::string& s) {
    if (s == "rose") return BlushColor::kRose;
    if (s == "apricot") return BlushColor::kApricot;
    if (s == "berry") return BlushColor::kBerry;
    return BlushColor::kPeach;
}

EyebrowStyle parseEyebrowStyle(const std::string& s) {
    return (s == "arched") ? EyebrowStyle::kArched : EyebrowStyle::kNatural;
}

EyebrowColor parseEyebrowColor(const std::string& s) {
    if (s == "soft_brown") return EyebrowColor::kSoftBrown;
    if (s == "grey_black") return EyebrowColor::kGreyBlack;
    return EyebrowColor::kDarkBrown;
}

EyeshadowColor parseEyeshadowColor(const std::string& s) {
    if (s == "peach_pink") return EyeshadowColor::kPeachPink;
    if (s == "rose_wine") return EyeshadowColor::kRoseWine;
    if (s == "champagne") return EyeshadowColor::kChampagne;
    return EyeshadowColor::kEarth;
}

ContactLensStyle parseContactLensStyle(const std::string& s) {
    if (s == "chocolate") return ContactLensStyle::kChocolate;
    if (s == "mist_grey") return ContactLensStyle::kMistGrey;
    if (s == "sakura_pink") return ContactLensStyle::kSakuraPink;
    return ContactLensStyle::kHazel;
}

BackgroundMode parseBackgroundMode(const std::string& s) {
    if (s == "blur") return BackgroundMode::kBlur;
    if (s == "virtual_image" || s == "image") return BackgroundMode::kVirtualImage;
    if (s == "virtual_blur" || s == "blur_image") return BackgroundMode::kVirtualBlur;
    return BackgroundMode::kNone;
}

math::RelightStyle parseRelightStyle(const std::string& s) {
    if (s == "natural") return math::RelightStyle::kNatural;
    if (s == "butterfly") return math::RelightStyle::kButterfly;
    if (s == "rembrandt") return math::RelightStyle::kRembrandt;
    return math::RelightStyle::kStudioSoftbox;
}

math::PerformanceTier parsePerformanceTier(const std::string& s) {
    if (s == "lite") return math::PerformanceTier::kLite;
    if (s == "standard") return math::PerformanceTier::kStandard;
    if (s == "ultra") return math::PerformanceTier::kUltra;
    return math::PerformanceTier::kAuto;
}

}  // namespace livebeauty
