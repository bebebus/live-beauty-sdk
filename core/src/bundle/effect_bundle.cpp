#include "livebeauty/bundle/effect_bundle.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>

namespace livebeauty::bundle {

namespace {

struct JsonValue {
    enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
    Kind kind = Kind::kNull;
    bool boolVal = false;
    double numVal = 0.0;
    std::string strVal;
    std::vector<JsonValue> arrVal;
    std::map<std::string, JsonValue> objVal;

    const JsonValue* find(const char* key) const {
        if (kind != Kind::kObject) return nullptr;
        auto it = objVal.find(key);
        return (it != objVal.end()) ? &it->second : nullptr;
    }
};

class MiniJsonReader {
public:
    explicit MiniJsonReader(const std::string& s) : s_(s) {}

    bool parseRoot(JsonValue& out, std::string& err) {
        skipWs();
        if (!parseValue(out, err, 0)) return false;
        skipWs();
        if (pos_ != s_.size()) {
            err = "trailing characters after JSON root";
            return false;
        }
        return true;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    void skipWs() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    bool matchLiteral(const char* lit) {
        size_t i = 0;
        while (lit[i] != '\0') {
            if (pos_ + i >= s_.size() || s_[pos_ + i] != lit[i]) return false;
            ++i;
        }
        pos_ += i;
        return true;
    }

    bool parseString(std::string& out, std::string& err) {
        if (pos_ >= s_.size() || s_[pos_] != '"') {
            err = "expected '\"'";
            return false;
        }
        ++pos_;
        out.clear();
        while (pos_ < s_.size()) {
            const char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) {
                    err = "unterminated escape";
                    return false;
                }
                const char esc = s_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        auto readHex4 = [&](unsigned int& dst) -> bool {
                            if (pos_ + 4 > s_.size()) return false;
                            dst = 0;
                            for (int k = 0; k < 4; ++k) {
                                const char hc = s_[pos_++];
                                dst <<= 4;
                                if (hc >= '0' && hc <= '9') dst |= static_cast<unsigned int>(hc - '0');
                                else if (hc >= 'a' && hc <= 'f') dst |= static_cast<unsigned int>(10 + hc - 'a');
                                else if (hc >= 'A' && hc <= 'F') dst |= static_cast<unsigned int>(10 + hc - 'A');
                                else return false;
                            }
                            return true;
                        };
                        unsigned int code = 0;
                        if (!readHex4(code)) {
                            err = "invalid unicode escape";
                            return false;
                        }
                        if (code >= 0xD800u && code <= 0xDBFFu) {
                            if (pos_ + 6 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                                const size_t saved = pos_;
                                pos_ += 2;
                                unsigned int low = 0;
                                if (readHex4(low) && low >= 0xDC00u && low <= 0xDFFFu) {
                                    code = 0x10000u + (((code - 0xD800u) << 10) | (low - 0xDC00u));
                                } else {
                                    pos_ = saved;
                                    code = 0xFFFDu;
                                }
                            } else {
                                code = 0xFFFDu;
                            }
                        } else if (code >= 0xDC00u && code <= 0xDFFFu) {
                            code = 0xFFFDu;
                        }
                        if (code <= 0x7Fu) {
                            out.push_back(static_cast<char>(code));
                        } else if (code <= 0x7FFu) {
                            out.push_back(static_cast<char>(0xC0u | (code >> 6)));
                            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                        } else if (code <= 0xFFFFu) {
                            out.push_back(static_cast<char>(0xE0u | (code >> 12)));
                            out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                        } else {
                            out.push_back(static_cast<char>(0xF0u | (code >> 18)));
                            out.push_back(static_cast<char>(0x80u | ((code >> 12) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                        }
                        break;
                    }
                    default:
                        err = "unsupported escape character";
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        err = "unterminated string";
        return false;
    }

    // Locale-independent strict RFC 8259 decimal number parser (never affected by LC_NUMERIC ',')
    bool parseNumber(double& out, std::string& err) {
        size_t i = pos_;
        bool neg = false;
        if (i < s_.size() && s_[i] == '-') {
            neg = true;
            ++i;
        }
        if (i >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[i]))) {
            err = "invalid number";
            return false;
        }
        double mantissa = 0.0;
        if (s_[i] == '0') {
            ++i;
            // Reject hex (0x...) or leading zeros (01)
            if (i < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i])) || s_[i] == 'x' || s_[i] == 'X')) {
                err = "invalid leading zero or hex number";
                return false;
            }
        } else {
            while (i < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i]))) {
                mantissa = mantissa * 10.0 + static_cast<double>(s_[i] - '0');
                ++i;
            }
        }
        if (i < s_.size() && s_[i] == '.') {
            ++i;
            if (i >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[i]))) {
                err = "invalid fractional number";
                return false;
            }
            double place = 0.1;
            while (i < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i]))) {
                mantissa += static_cast<double>(s_[i] - '0') * place;
                place *= 0.1;
                ++i;
            }
        }
        int expSign = 1;
        int expVal = 0;
        if (i < s_.size() && (s_[i] == 'e' || s_[i] == 'E')) {
            ++i;
            if (i < s_.size() && (s_[i] == '+' || s_[i] == '-')) {
                expSign = (s_[i] == '-') ? -1 : 1;
                ++i;
            }
            if (i >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[i]))) {
                err = "invalid exponent in number";
                return false;
            }
            while (i < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i]))) {
                if (expVal < 1000) expVal = expVal * 10 + (s_[i] - '0');
                ++i;
            }
        }
        double val = mantissa * std::pow(10.0, static_cast<double>(expSign * expVal));
        if (neg) val = -val;
        if (!std::isfinite(val)) {
            err = "number out of finite range";
            return false;
        }
        out = val;
        pos_ = i;
        return true;
    }

    bool parseValue(JsonValue& out, std::string& err, int depth) {
        if (depth > 24) {
            err = "JSON nesting too deep";
            return false;
        }
        skipWs();
        if (pos_ >= s_.size()) {
            err = "unexpected end of JSON";
            return false;
        }
        const char c = s_[pos_];
        if (c == '{') {
            ++pos_;
            out = JsonValue{};
            out.kind = JsonValue::Kind::kObject;
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            while (true) {
                skipWs();
                std::string key;
                if (!parseString(key, err)) return false;
                skipWs();
                if (pos_ >= s_.size() || s_[pos_] != ':') {
                    err = "expected ':' in object";
                    return false;
                }
                ++pos_;
                JsonValue val;
                if (!parseValue(val, err, depth + 1)) return false;
                out.objVal[std::move(key)] = std::move(val);
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == '}') {
                    ++pos_;
                    return true;
                }
                err = "expected ',' or '}' in object";
                return false;
            }
        }
        if (c == '[') {
            ++pos_;
            out = JsonValue{};
            out.kind = JsonValue::Kind::kArray;
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            while (true) {
                JsonValue elem;
                if (!parseValue(elem, err, depth + 1)) return false;
                out.arrVal.push_back(std::move(elem));
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == ']') {
                    ++pos_;
                    return true;
                }
                err = "expected ',' or ']' in array";
                return false;
            }
        }
        if (c == '"') {
            out = JsonValue{};
            out.kind = JsonValue::Kind::kString;
            return parseString(out.strVal, err);
        }
        if (matchLiteral("true")) {
            out = JsonValue{};
            out.kind = JsonValue::Kind::kBool;
            out.boolVal = true;
            return true;
        }
        if (matchLiteral("false")) {
            out = JsonValue{};
            out.kind = JsonValue::Kind::kBool;
            out.boolVal = false;
            return true;
        }
        if (matchLiteral("null")) {
            out = JsonValue{};
            out.kind = JsonValue::Kind::kNull;
            return true;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            out = JsonValue{};
            out.kind = JsonValue::Kind::kNumber;
            return parseNumber(out.numVal, err);
        }
        err = "unexpected token in JSON";
        return false;
    }
};

float optFloat(const JsonValue& obj, const char* key, float fallback, float lo, float hi) {
    const auto* v = obj.find(key);
    if (!v || v->kind != JsonValue::Kind::kNumber) return fallback;
    return detail::clampf(static_cast<float>(v->numVal), lo, hi);
}

bool optBool(const JsonValue& obj, const char* key, bool fallback) {
    const auto* v = obj.find(key);
    if (!v || v->kind != JsonValue::Kind::kBool) return fallback;
    return v->boolVal;
}

std::string optStr(const JsonValue& obj, const char* key, const std::string& fallback = {}) {
    const auto* v = obj.find(key);
    if (!v || v->kind != JsonValue::Kind::kString) return fallback;
    return v->strVal;
}

BeautyConfig baseFromPresetName(const std::string& preset) {
    if (preset == "natural_studio") return BeautyConfig::naturalStudio();
    if (preset == "clear_cold_white") return BeautyConfig::clearColdWhite();
    if (preset == "korean_water_glow") return BeautyConfig::koreanWaterGlow();
    if (preset == "mens_clean") return BeautyConfig::mensClean();
    if (preset == "original") return BeautyConfig::original();
    return BeautyConfig{};
}

}  // namespace

BundleTriggerRule EffectBundleParser::parseTriggerRule(const std::string& s) {
    if (s == "smile") return BundleTriggerRule::kSmile;
    if (s == "mouth_open" || s == "jaw_open") return BundleTriggerRule::kMouthOpen;
    if (s == "brow_up") return BundleTriggerRule::kBrowUp;
    if (s == "blink" || s == "wink") return BundleTriggerRule::kBlink;
    if (s == "cheek_puff") return BundleTriggerRule::kCheekPuff;
    if (s == "tongue_out") return BundleTriggerRule::kTongueOut;
    if (s == "gesture_heart" || s == "heart") return BundleTriggerRule::kGestureHeart;
    if (s == "gesture_thumbs_up" || s == "thumbs_up") return BundleTriggerRule::kGestureThumbsUp;
    if (s == "gesture_victory" || s == "victory") return BundleTriggerRule::kGestureVictory;
    if (s == "gesture_palm" || s == "open_palm") return BundleTriggerRule::kGestureOpenPalm;
    return BundleTriggerRule::kAlways;
}

math::StickerAnchor EffectBundleParser::parseStickerAnchor(const std::string& s) {
    if (s == "glasses" || s == "eye_bridge") return math::StickerAnchor::kEyesBridge;
    if (s == "nose" || s == "nose_tip") return math::StickerAnchor::kNoseTip;
    if (s == "cheeks") return math::StickerAnchor::kCheeks;
    if (s == "full_face") return math::StickerAnchor::kFullFace;
    return math::StickerAnchor::kForeheadCrown;
}

math::Ar3dAnchor EffectBundleParser::parseAr3dAnchor(const std::string& s) {
    if (s == "glasses_bridge" || s == "glasses") return math::Ar3dAnchor::kGlassesBridge;
    if (s == "left_ear") return math::Ar3dAnchor::kLeftEar;
    if (s == "right_ear") return math::Ar3dAnchor::kRightEar;
    if (s == "nose_tip" || s == "nose") return math::Ar3dAnchor::kNoseTip;
    if (s == "chin") return math::Ar3dAnchor::kChin;
    return math::Ar3dAnchor::kHeadCrown;
}

math::Ar3dMeshPreset EffectBundleParser::parseMeshPreset(const std::string& s) {
    if (s == "curved_glasses") return math::Ar3dMeshPreset::kCurvedGlasses;
    if (s == "curved_tiara" || s == "tiara") return math::Ar3dMeshPreset::kCurvedTiara;
    return math::Ar3dMeshPreset::kBillboardQuad;
}

float EffectBundleParser::evaluateTriggerGain(
    BundleTriggerRule rule,
    const math::ExpressionReaction& rx,
    math::GestureType activeGesture) {
    switch (rule) {
        case BundleTriggerRule::kSmile:
            return detail::smoothstep(0.18f, 0.45f, rx.smile);
        case BundleTriggerRule::kMouthOpen:
            return detail::smoothstep(0.16f, 0.40f, rx.mouthOpen);
        case BundleTriggerRule::kBrowUp:
            return detail::smoothstep(0.18f, 0.44f, rx.browUp);
        case BundleTriggerRule::kBlink:
            return detail::smoothstep(0.25f, 0.55f, rx.blink);
        case BundleTriggerRule::kCheekPuff:
            return detail::smoothstep(0.16f, 0.40f, rx.cheekPuff);
        case BundleTriggerRule::kTongueOut:
            return detail::smoothstep(0.18f, 0.42f, rx.tongueOut);
        case BundleTriggerRule::kGestureHeart:
            return (activeGesture == math::GestureType::kHeart) ? 1.0f : 0.0f;
        case BundleTriggerRule::kGestureThumbsUp:
            return (activeGesture == math::GestureType::kThumbsUp) ? 1.0f : 0.0f;
        case BundleTriggerRule::kGestureVictory:
            return (activeGesture == math::GestureType::kVictory) ? 1.0f : 0.0f;
        case BundleTriggerRule::kGestureOpenPalm:
            return (activeGesture == math::GestureType::kOpenPalm) ? 1.0f : 0.0f;
        case BundleTriggerRule::kAlways:
        default:
            return 1.0f;
    }
}

bool EffectBundleParser::parseJson(
    const std::string& jsonText, EffectBundleManifest& out, std::string* outError) {
    out = EffectBundleManifest{};
    JsonValue root;
    std::string err;
    MiniJsonReader reader(jsonText);
    if (!reader.parseRoot(root, err) || root.kind != JsonValue::Kind::kObject) {
        if (outError) *outError = err.empty() ? "root must be JSON object" : err;
        return false;
    }

    if (const auto* sv = root.find("schemaVersion")) {
        if (sv->kind == JsonValue::Kind::kNumber) {
            out.schemaVersion = static_cast<int>(sv->numVal);
        } else if (sv->kind == JsonValue::Kind::kString) {
            out.schemaVersion = std::max(1, std::atoi(sv->strVal.c_str()));
        }
    }
    if (out.schemaVersion < 1 || out.schemaVersion > 2) {
        if (outError) *outError = "unsupported .lbfx schemaVersion";
        return false;
    }

    out.id = optStr(root, "id");
    out.name = optStr(root, "name");
    out.author = optStr(root, "author");

    if (const auto* b = root.find("beauty")) {
        if (b->kind != JsonValue::Kind::kObject) {
            if (outError) *outError = "'beauty' must be an object";
            return false;
        }
        out.hasBeautySection = true;
        BeautyConfig c = baseFromPresetName(optStr(*b, "preset"));
        c.enabled = optBool(*b, "enabled", c.enabled);
        c.skinOnly = optBool(*b, "skinOnly", c.skinOnly);
        c.smoothing = optFloat(*b, "smoothing", c.smoothing, 0.0f, 1.0f);
        c.brightness = optFloat(*b, "brightness", c.brightness, -1.0f, 1.0f);
        c.sharpness = optFloat(*b, "sharpness", c.sharpness, 0.0f, 1.0f);
        c.saturation = optFloat(*b, "saturation", c.saturation, 0.0f, 2.0f);
        c.filterIntensity = optFloat(*b, "filterIntensity", c.filterIntensity, 0.0f, 1.0f);
        if (b->find("filter")) c.filter = parseFilterStyle(optStr(*b, "filter"));

        c.whiten = optFloat(*b, "whiten", c.whiten, 0.0f, 1.0f);
        c.rosy = optFloat(*b, "rosy", c.rosy, 0.0f, 1.0f);
        c.skinTexture = optFloat(*b, "skinTexture", c.skinTexture, 0.0f, 1.0f);
        c.radiance = optFloat(*b, "radiance", c.radiance, 0.0f, 1.0f);
        c.faceSlim = optFloat(*b, "faceSlim", c.faceSlim, 0.0f, 1.0f);
        c.eyeEnlarge = optFloat(*b, "eyeEnlarge", c.eyeEnlarge, 0.0f, 1.0f);
        c.vFace = optFloat(*b, "vFace", c.vFace, 0.0f, 1.0f);
        c.faceNarrow = optFloat(*b, "faceNarrow", c.faceNarrow, 0.0f, 1.0f);
        c.faceSmall = optFloat(*b, "faceSmall", c.faceSmall, 0.0f, 1.0f);
        c.cheekboneSlim = optFloat(*b, "cheekboneSlim", c.cheekboneSlim, 0.0f, 1.0f);
        c.chinShape = optFloat(*b, "chinShape", c.chinShape, -1.0f, 1.0f);
        c.foreheadShape = optFloat(*b, "foreheadShape", c.foreheadShape, -1.0f, 1.0f);
        c.cranialTop = optFloat(*b, "cranialTop", c.cranialTop, 0.0f, 1.0f);
        c.noseSlim = optFloat(*b, "noseSlim", c.noseSlim, 0.0f, 1.0f);
        c.noseLength = optFloat(*b, "noseLength", c.noseLength, -1.0f, 1.0f);
        c.eyeDistance = optFloat(*b, "eyeDistance", c.eyeDistance, -1.0f, 1.0f);
        c.eyeCorner = optFloat(*b, "eyeCorner", c.eyeCorner, -1.0f, 1.0f);
        c.mouthSize = optFloat(*b, "mouthSize", c.mouthSize, -1.0f, 1.0f);
        c.mouthSmile = optFloat(*b, "mouthSmile", c.mouthSmile, 0.0f, 1.0f);

        c.darkCircles = optFloat(*b, "darkCircles", c.darkCircles, 0.0f, 1.0f);
        c.brightEyes = optFloat(*b, "brightEyes", c.brightEyes, 0.0f, 1.0f);
        c.teethWhiten = optFloat(*b, "teethWhiten", c.teethWhiten, 0.0f, 1.0f);
        c.nasolabial = optFloat(*b, "nasolabial", c.nasolabial, 0.0f, 1.0f);
        c.blemishRemove = optFloat(*b, "blemishRemove", c.blemishRemove, 0.0f, 1.0f);
        c.oilControl = optFloat(*b, "oilControl", c.oilControl, 0.0f, 1.0f);
        c.neckTone = optFloat(*b, "neckTone", c.neckTone, 0.0f, 1.0f);
        c.relight = optFloat(*b, "relight", c.relight, 0.0f, 1.0f);
        if (b->find("relightStyle")) c.relightStyle = parseRelightStyle(optStr(*b, "relightStyle"));

        c.lipstick = optFloat(*b, "lipstick", c.lipstick, 0.0f, 1.0f);
        if (b->find("lipstickColor")) c.lipstickColor = parseLipstickColor(optStr(*b, "lipstickColor"));
        if (b->find("lipstickFinish")) c.lipstickFinish = parseLipstickFinish(optStr(*b, "lipstickFinish"));
        c.blush = optFloat(*b, "blush", c.blush, 0.0f, 1.0f);
        if (b->find("blushStyle")) c.blushStyle = parseBlushStyle(optStr(*b, "blushStyle"));
        if (b->find("blushColor")) c.blushColor = parseBlushColor(optStr(*b, "blushColor"));
        c.eyebrow = optFloat(*b, "eyebrow", c.eyebrow, 0.0f, 1.0f);
        if (b->find("eyebrowStyle")) c.eyebrowStyle = parseEyebrowStyle(optStr(*b, "eyebrowStyle"));
        if (b->find("eyebrowColor")) c.eyebrowColor = parseEyebrowColor(optStr(*b, "eyebrowColor"));
        c.eyeshadow = optFloat(*b, "eyeshadow", c.eyeshadow, 0.0f, 1.0f);
        if (b->find("eyeshadowColor")) c.eyeshadowColor = parseEyeshadowColor(optStr(*b, "eyeshadowColor"));
        c.eyeliner = optFloat(*b, "eyeliner", c.eyeliner, 0.0f, 1.0f);
        c.aegyoSal = optFloat(*b, "aegyoSal", c.aegyoSal, 0.0f, 1.0f);
        c.contour = optFloat(*b, "contour", c.contour, 0.0f, 1.0f);
        c.highlight = optFloat(*b, "highlight", c.highlight, 0.0f, 1.0f);
        c.contactLens = optFloat(*b, "contactLens", c.contactLens, 0.0f, 1.0f);
        if (b->find("contactLensStyle")) c.contactLensStyle = parseContactLensStyle(optStr(*b, "contactLensStyle"));
        c.catchlight = optFloat(*b, "catchlight", c.catchlight, 0.0f, 1.0f);

        c.hairDye = optFloat(*b, "hairDye", c.hairDye, 0.0f, 1.0f);
        if (b->find("hairDyeColor")) c.hairDyeColor = parseHairDyeColor(optStr(*b, "hairDyeColor"));
        c.hairShine = optFloat(*b, "hairShine", c.hairShine, 0.0f, 1.0f);

        c.bloom = optFloat(*b, "bloom", c.bloom, 0.0f, 1.0f);
        c.filmGrain = optFloat(*b, "filmGrain", c.filmGrain, 0.0f, 1.0f);
        c.colorTemp = optFloat(*b, "colorTemp", c.colorTemp, -1.0f, 1.0f);
        c.colorTint = optFloat(*b, "colorTint", c.colorTint, -1.0f, 1.0f);

        c.waistSlim = optFloat(*b, "waistSlim", c.waistSlim, 0.0f, 1.0f);
        c.bustEnhance = optFloat(*b, "bustEnhance", c.bustEnhance, 0.0f, 1.0f);
        c.legLengthen = optFloat(*b, "legLengthen", c.legLengthen, 0.0f, 1.0f);
        c.shoulderSlim = optFloat(*b, "shoulderSlim", c.shoulderSlim, 0.0f, 1.0f);
        c.neckLengthen = optFloat(*b, "neckLengthen", c.neckLengthen, 0.0f, 1.0f);
        c.ambientMakeupBlend = optFloat(*b, "ambientMakeupBlend", c.ambientMakeupBlend, 0.0f, 1.0f);
        c.matteTemporal = optFloat(*b, "matteTemporal", c.matteTemporal, 0.0f, 1.0f);

        if (b->find("backgroundBlur") || b->find("backgroundMode") || b->find("virtualBgStrength") ||
            b->find("chromaKeyColor") || b->find("chromaKeySimilarity") ||
            b->find("chromaKeySmoothness") || b->find("chromaKeyDespill")) {
            out.overridesBackground = true;
            c.backgroundBlur = optFloat(*b, "backgroundBlur", c.backgroundBlur, 0.0f, 1.0f);
            if (b->find("backgroundMode")) c.backgroundMode = parseBackgroundMode(optStr(*b, "backgroundMode"));
            c.virtualBgStrength = optFloat(*b, "virtualBgStrength", c.virtualBgStrength, 0.0f, 1.0f);
            if (b->find("chromaKeyColor")) c.chromaKeyColor = parseChromaKeyColor(optStr(*b, "chromaKeyColor"));
            c.chromaKeySimilarity = optFloat(*b, "chromaKeySimilarity", c.chromaKeySimilarity, 0.05f, 0.95f);
            c.chromaKeySmoothness = optFloat(*b, "chromaKeySmoothness", c.chromaKeySmoothness, 0.01f, 0.80f);
            c.chromaKeyDespill = optFloat(*b, "chromaKeyDespill", c.chromaKeyDespill, 0.0f, 1.0f);
        }
        if (b->find("performanceTier") || b->find("preferHalfFloat") ||
            b->find("maxFaces") || b->find("frameDelayFrames")) {
            out.overridesPipelineOptions = true;
            if (b->find("performanceTier")) c.performanceTier = parsePerformanceTier(optStr(*b, "performanceTier"));
            c.preferHalfFloat = optBool(*b, "preferHalfFloat", c.preferHalfFloat);
            c.maxFaces = static_cast<int>(optFloat(*b, "maxFaces", static_cast<float>(c.maxFaces), 1.0f, 2.0f));
            c.frameDelayFrames = static_cast<int>(optFloat(*b, "frameDelayFrames", static_cast<float>(c.frameDelayFrames), 0.0f, 4.0f));
        }

        if (b->find("lutUrl")) c.lutUrl = optStr(*b, "lutUrl");
        c.lutIntensity = optFloat(*b, "lutIntensity", c.lutIntensity, 0.0f, 1.0f);
        out.beauty = c;
    }

    if (const auto* bg = root.find("background")) {
        if (bg->kind == JsonValue::Kind::kObject) {
            out.overridesBackground = true;
            out.beauty.backgroundBlur = optFloat(*bg, "blur", optFloat(*bg, "backgroundBlur", out.beauty.backgroundBlur, 0.0f, 1.0f), 0.0f, 1.0f);
            if (bg->find("mode")) out.beauty.backgroundMode = parseBackgroundMode(optStr(*bg, "mode"));
            else if (bg->find("backgroundMode")) out.beauty.backgroundMode = parseBackgroundMode(optStr(*bg, "backgroundMode"));
            out.beauty.virtualBgStrength = optFloat(*bg, "strength", optFloat(*bg, "virtualBgStrength", out.beauty.virtualBgStrength, 0.0f, 1.0f), 0.0f, 1.0f);
            if (bg->find("keyColor")) out.beauty.chromaKeyColor = parseChromaKeyColor(optStr(*bg, "keyColor"));
            else if (bg->find("chromaKeyColor")) out.beauty.chromaKeyColor = parseChromaKeyColor(optStr(*bg, "chromaKeyColor"));
            out.beauty.chromaKeySimilarity = optFloat(*bg, "similarity", optFloat(*bg, "chromaKeySimilarity", out.beauty.chromaKeySimilarity, 0.05f, 0.95f), 0.05f, 0.95f);
            out.beauty.chromaKeySmoothness = optFloat(*bg, "smoothness", optFloat(*bg, "chromaKeySmoothness", out.beauty.chromaKeySmoothness, 0.01f, 0.80f), 0.01f, 0.80f);
            out.beauty.chromaKeyDespill = optFloat(*bg, "despill", optFloat(*bg, "chromaKeyDespill", out.beauty.chromaKeyDespill, 0.0f, 1.0f), 0.0f, 1.0f);
        }
    }
    if (const auto* po = root.find("pipelineOptions")) {
        if (po->kind == JsonValue::Kind::kObject) {
            out.overridesPipelineOptions = true;
            if (po->find("performanceTier")) out.beauty.performanceTier = parsePerformanceTier(optStr(*po, "performanceTier"));
            out.beauty.preferHalfFloat = optBool(*po, "preferHalfFloat", out.beauty.preferHalfFloat);
            out.beauty.maxFaces = static_cast<int>(optFloat(*po, "maxFaces", static_cast<float>(out.beauty.maxFaces), 1.0f, 2.0f));
            out.beauty.frameDelayFrames = static_cast<int>(optFloat(*po, "frameDelayFrames", static_cast<float>(out.beauty.frameDelayFrames), 0.0f, 4.0f));
        }
    }

    if (const auto* stArr = root.find("stickers")) {
        if (stArr->kind != JsonValue::Kind::kArray) {
            if (outError) *outError = "'stickers' must be an array";
            return false;
        }
        int nextSlot = 0;
        int parsedCount = 0;
        for (const auto& item : stArr->arrVal) {
            if (item.kind != JsonValue::Kind::kObject || parsedCount >= math::kMaxStickerLayers) continue;
            BundleStickerSpec spec{};
            const bool explicitSlot = (item.find("slot") != nullptr);
            spec.slot = std::clamp(static_cast<int>(optFloat(item, "slot", static_cast<float>(nextSlot), 0.0f, 3.0f)),
                                   0, math::kMaxStickerLayers - 1);
            if (!explicitSlot && nextSlot < math::kMaxStickerLayers - 1) ++nextSlot;
            ++parsedCount;
            spec.assetPath = optStr(item, "assetPath");
            spec.trigger = parseTriggerRule(optStr(item, "trigger", "always"));
            spec.layer.enabled = optBool(item, "enabled", true);
            spec.layer.anchor = parseStickerAnchor(optStr(item, "anchor", "forehead_crown"));
            spec.layer.scale = optFloat(item, "scale", 1.0f, 0.1f, 4.0f);
            spec.layer.offsetX = optFloat(item, "offsetX", 0.0f, -2.0f, 2.0f);
            spec.layer.offsetY = optFloat(item, "offsetY", 0.0f, -2.0f, 2.0f);
            const float rotDeg = optFloat(item, "rotationDeg", 0.0f, -180.0f, 180.0f);
            spec.layer.rotationRad = rotDeg * (3.14159265f / 180.0f);
            spec.layer.opacity = optFloat(item, "opacity", 1.0f, 0.0f, 1.0f);
            spec.layer.textureAspect = optFloat(item, "aspect", 1.0f, 0.1f, 10.0f);
            out.stickers.push_back(std::move(spec));
        }
    }

    if (const auto* arArr = root.find("ar3d")) {
        if (arArr->kind != JsonValue::Kind::kArray) {
            if (outError) *outError = "'ar3d' must be an array";
            return false;
        }
        int nextSlot = 0;
        int parsedCount = 0;
        for (const auto& item : arArr->arrVal) {
            if (item.kind != JsonValue::Kind::kObject || parsedCount >= math::kMaxAr3dLayers) continue;
            BundleAr3dSpec spec{};
            const bool explicitSlot = (item.find("slot") != nullptr);
            spec.slot = std::clamp(static_cast<int>(optFloat(item, "slot", static_cast<float>(nextSlot), 0.0f, 3.0f)),
                                   0, math::kMaxAr3dLayers - 1);
            if (!explicitSlot && nextSlot < math::kMaxAr3dLayers - 1) ++nextSlot;
            ++parsedCount;
            spec.assetPath = optStr(item, "assetPath");
            spec.trigger = parseTriggerRule(optStr(item, "trigger", "always"));
            spec.layer.enabled = optBool(item, "enabled", true);
            spec.layer.anchor = parseAr3dAnchor(optStr(item, "anchor", "head_crown"));
            spec.layer.meshPreset = parseMeshPreset(optStr(item, "meshPreset", "quad"));
            spec.layer.scale = optFloat(item, "scale", 1.0f, 0.1f, 3.0f);
            spec.layer.offsetX = optFloat(item, "offsetX", 0.0f, -2.0f, 2.0f);
            spec.layer.offsetY = optFloat(item, "offsetY", 0.0f, -2.0f, 2.0f);
            spec.layer.offsetZ = optFloat(item, "offsetZ", 0.0f, -2.0f, 2.0f);
            spec.layer.yawOffsetRad = optFloat(item, "yawDeg", 0.0f, -180.0f, 180.0f) * (3.14159265f / 180.0f);
            spec.layer.pitchOffsetRad = optFloat(item, "pitchDeg", 0.0f, -180.0f, 180.0f) * (3.14159265f / 180.0f);
            spec.layer.rollOffsetRad = optFloat(item, "rollDeg", 0.0f, -180.0f, 180.0f) * (3.14159265f / 180.0f);
            spec.layer.opacity = optFloat(item, "opacity", 1.0f, 0.0f, 1.0f);
            spec.layer.textureAspect = optFloat(item, "aspect", 1.0f, 0.1f, 10.0f);
            spec.layer.springStiffness = optFloat(item, "springStiffness", 0.0f, 0.0f, 80.0f);
            spec.layer.springDamping = optFloat(item, "springDamping", 4.5f, 1.0f, 20.0f);
            spec.layer.headOcclusion = optBool(item, "headOcclusion", true);
            out.ar3d.push_back(std::move(spec));
        }
    }

    return true;
}

}  // namespace livebeauty::bundle
