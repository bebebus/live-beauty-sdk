#include "livebeauty/math/device_tier.h"

#include <algorithm>
#include <cctype>

namespace livebeauty::math {

namespace {

std::string toLowerAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

int extractFirstNumberAfter(const std::string& s, const std::string& token) {
    const size_t pos = s.find(token);
    if (pos == std::string::npos) return -1;
    size_t i = pos + token.size();
    const size_t scanLimit = std::min(s.size(), i + 12);
    while (i < scanLimit && !std::isdigit(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    int val = 0;
    bool any = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        val = val * 10 + (s[i] - '0');
        any = true;
        ++i;
    }
    return any ? val : -1;
}

}  // namespace

PerformanceTier DeviceTierPolicy::classify(const std::string& gpuRenderer, int cpuCores, int totalRamMb) {
    const std::string gpu = toLowerAscii(gpuRenderer);

    // 1. Apple Silicon (iOS / macOS Metal)
    if (gpu.find("apple m") != std::string::npos) return PerformanceTier::kUltra;
    if (gpu.find("apple a") != std::string::npos) {
        const int gen = extractFirstNumberAfter(gpu, "apple a");
        if (gen >= 15) return PerformanceTier::kUltra;
        if (gen >= 12) return PerformanceTier::kStandard;
        if (gen > 0) return PerformanceTier::kLite;
    }
    if (gpu.find("apple gpu") != std::string::npos) {
        return (totalRamMb >= 5500 || cpuCores >= 6) ? PerformanceTier::kUltra : PerformanceTier::kStandard;
    }

    // 2. Qualcomm Adreno
    if (gpu.find("adreno") != std::string::npos) {
        const int model = extractFirstNumberAfter(gpu, "adreno");
        if (model >= 730 || (model >= 800 && model < 900)) return PerformanceTier::kUltra;
        if (model >= 618) return PerformanceTier::kStandard;
        if (model > 0) return PerformanceTier::kLite;
    }

    // 3. ARM Immortalis / Mali
    if (gpu.find("immortalis") != std::string::npos) return PerformanceTier::kUltra;
    if (gpu.find("mali-g") != std::string::npos) {
        const int model = extractFirstNumberAfter(gpu, "mali-g");
        if (model >= 710 || model == 78 || model == 77) return PerformanceTier::kUltra;
        if (model >= 57 || model == 68 || model == 610 || model == 615) return PerformanceTier::kStandard;
        if (model > 0) return PerformanceTier::kLite;
    }
    if (gpu.find("mali-t") != std::string::npos || gpu.find("mali-4") != std::string::npos) {
        return PerformanceTier::kLite;
    }

    // 4. Samsung Xclipse (AMD RDNA mobile)
    if (gpu.find("xclipse") != std::string::npos) {
        const int model = extractFirstNumberAfter(gpu, "xclipse");
        return (model >= 920) ? PerformanceTier::kUltra : PerformanceTier::kStandard;
    }

    // 5. PowerVR / SwiftShader / software emulators
    if (gpu.find("powervr") != std::string::npos || gpu.find("swiftshader") != std::string::npos ||
        gpu.find("llvmpipe") != std::string::npos) {
        return PerformanceTier::kLite;
    }

    // 6. Fallback heuristic using RAM & CPU core count
    if (totalRamMb >= 7500 && cpuCores >= 8) return PerformanceTier::kUltra;
    if (totalRamMb > 0 && totalRamMb <= 3500) return PerformanceTier::kLite;
    if (cpuCores > 0 && cpuCores <= 4) return PerformanceTier::kLite;
    return PerformanceTier::kStandard;
}

TierCapabilities DeviceTierPolicy::capabilitiesFor(PerformanceTier tier) {
    TierCapabilities cap{};
    const PerformanceTier concrete = (tier == PerformanceTier::kAuto) ? PerformanceTier::kStandard : tier;
    cap.resolvedTier = concrete;
    switch (concrete) {
        case PerformanceTier::kLite:
            cap.enableRelight3d = false;
            cap.enableNeckTone = true;
            cap.enableOilControl = true;
            cap.enableHairAlpha = false;
            cap.enableMatteRefine = false;
            cap.enableBlemishHeal = false;
            cap.enableAr3d = false;
            cap.enableAmbientMakeup = true;
            cap.maxFacesCap = 1;
            break;
        case PerformanceTier::kStandard:
            cap.enableRelight3d = true;
            cap.enableNeckTone = true;
            cap.enableOilControl = true;
            cap.enableHairAlpha = true;
            cap.enableMatteRefine = true;
            cap.enableBlemishHeal = true;
            cap.enableAr3d = true;
            cap.enableAmbientMakeup = true;
            cap.maxFacesCap = 2;
            break;
        case PerformanceTier::kUltra:
        case PerformanceTier::kAuto:
        default:
            cap.enableRelight3d = true;
            cap.enableNeckTone = true;
            cap.enableOilControl = true;
            cap.enableHairAlpha = true;
            cap.enableMatteRefine = true;
            cap.enableBlemishHeal = true;
            cap.enableAr3d = true;
            cap.enableAmbientMakeup = true;
            cap.maxFacesCap = 2;
            break;
    }
    return cap;
}

TierCapabilities DeviceTierPolicy::resolve(
    PerformanceTier requested,
    PerformanceTier hardwareDetected,
    int governorLoadLevel) {
    PerformanceTier base = (requested == PerformanceTier::kAuto)
        ? (hardwareDetected == PerformanceTier::kAuto ? PerformanceTier::kStandard : hardwareDetected)
        : requested;
    if (governorLoadLevel >= 2 && base == PerformanceTier::kUltra) {
        base = PerformanceTier::kStandard;
    } else if (governorLoadLevel >= 2 && base == PerformanceTier::kStandard) {
        base = PerformanceTier::kLite;
    }
    TierCapabilities cap = capabilitiesFor(base);
    if (governorLoadLevel >= 1) {
        cap.enableBlemishHeal = false;
        cap.enableMatteRefine = false;
    }
    return cap;
}

}  // namespace livebeauty::math
