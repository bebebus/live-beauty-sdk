#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "livebeauty/config.h"
#include "livebeauty/math/ar_3d_accessory.h"
#include "livebeauty/math/face_aux_vision.h"
#include "livebeauty/math/sticker_transform.h"
#include "livebeauty/rhi/rhi_types.h"

namespace livebeauty::bundle {

/** Expression-gated visibility rule for declarative creator effect layers (.lbfx). */
enum class BundleTriggerRule : uint8_t {
    kAlways = 0,
    kSmile = 1,
    kMouthOpen = 2,
    kBrowUp = 3,
    kBlink = 4,
    kCheekPuff = 5,
    kTongueOut = 6,
};

struct BundleStickerSpec {
    int slot = 0;
    std::string assetPath;
    math::StickerLayerConfig layer{};
    BundleTriggerRule trigger = BundleTriggerRule::kAlways;
};

struct BundleAr3dSpec {
    int slot = 0;
    std::string assetPath;
    math::Ar3dAccessoryConfig layer{};
    BundleTriggerRule trigger = BundleTriggerRule::kAlways;
};

/**
 * Parsed representation of a declarative `.lbfx` Effect Bundle manifest (`manifest.json`).
 * Encapsulates BeautyConfig look overrides, LUT path, 2D stickers, 3D head-occluded meshes,
 * real-time hair dye settings, and expression-reactive trigger rules.
 */
struct EffectBundleManifest {
    int schemaVersion = 1;
    std::string id;
    std::string name;
    std::string author;
    BeautyConfig beauty{};
    bool hasBeautySection = false;
    std::vector<BundleStickerSpec> stickers;
    std::vector<BundleAr3dSpec> ar3d;
};

using BundleTextureResolver = std::function<rhi::TextureId(const std::string& assetPath)>;

class EffectBundleParser {
public:
    static BundleTriggerRule parseTriggerRule(const std::string& s);
    static math::StickerAnchor parseStickerAnchor(const std::string& s);
    static math::Ar3dAnchor parseAr3dAnchor(const std::string& s);
    static math::Ar3dMeshPreset parseMeshPreset(const std::string& s);

    /** Evaluate [0..1] opacity multiplier for a bundle layer given current smoothed expression state. */
    static float evaluateTriggerGain(BundleTriggerRule rule, const math::ExpressionReaction& reaction);

    /** Parse a `.lbfx` JSON manifest string into `out` (zero external dependencies). */
    static bool parseJson(const std::string& jsonText, EffectBundleManifest& out, std::string* outError = nullptr);
};

}  // namespace livebeauty::bundle
