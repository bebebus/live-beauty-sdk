#pragma once

#include <cstdint>
#include <string>

namespace livebeauty::math {

enum class PerformanceTier : int {
    kAuto = 0,      // Resolve automatically from GPU/SoC hardware profile + runtime frame budget
    kLite = 1,      // Entry/legacy devices: core guided smooth + basic warp, skip heavy optional passes
    kStandard = 2,  // Mid-range devices: 3D relight, oil control, neck continuity, hair alpha, makeup ambient WB
    kUltra = 3,     // Flagship devices: all features + temporal matte refinement + blemish heal + 3D AR + dual-face
};

struct TierCapabilities {
    PerformanceTier resolvedTier = PerformanceTier::kStandard;
    bool enableRelight3d = true;
    bool enableNeckTone = true;
    bool enableOilControl = true;
    bool enableHairAlpha = true;
    bool enableMatteRefine = true;
    bool enableBlemishHeal = true;
    bool enableAr3d = true;
    bool enableAmbientMakeup = true;
    int maxFacesCap = 2;
};

/**
 * Hardware/SoC tier classifier & dynamic frame-budget scheduler.
 *
 * Classifies Android (Adreno/Mali/Immortalis/PowerVR/Xclipse) and Apple (A-series/M-series) GPUs
 * into Lite / Standard / Ultra tiers, and supports smooth one-step runtime downshift if sustained
 * frame time exceeds 30fps budget.
 */
class DeviceTierPolicy {
public:
    /** Classify device hardware into a concrete tier (kLite / kStandard / kUltra). */
    static PerformanceTier classify(const std::string& gpuRenderer, int cpuCores, int totalRamMb);

    /** Return feature capability gates for a concrete tier (kAuto resolves to kStandard). */
    static TierCapabilities capabilitiesFor(PerformanceTier tier);

    /** Resolve final capabilities given user preference, hardware profile, and runtime load governor level (0..3). */
    static TierCapabilities resolve(
        PerformanceTier requested,
        PerformanceTier hardwareDetected,
        int governorLoadLevel);
};

}  // namespace livebeauty::math
