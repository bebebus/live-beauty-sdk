#pragma once

#include <cstdint>
#include <string>
#include "livebeauty/types.h"

namespace livebeauty::math {

class FaceRoiSelector {
public:
    static constexpr float kMinFaceWidth = 0.045f;
    static constexpr float kEnterFaceWidth = 0.30f;
    static constexpr float kExitFaceWidth = 0.36f;
    static constexpr float kExpandFactor = 2.80f;
    static constexpr float kMinCropSpan = 0.28f;
    static constexpr float kMaxCropSpan = 0.85f;
    static constexpr int kRoiSidePx = 256;
    static constexpr int64_t kStaleMs = 350;
    static constexpr int64_t kMaxSampleAgeMs = 650;

    bool isActive() const { return active_; }

    void reset();
    void onFace(float glCenterX, float glCenterY, float faceWidth, int64_t sampleMs, int64_t receivedMs);
    void onFace(float glCenterX, float glCenterY, float faceWidth, int64_t sampleMs) {
        onFace(glCenterX, glCenterY, faceWidth, sampleMs, sampleMs);
    }
    void onMiss();
    CropRect select(int width, int height, int64_t nowMs);

private:
    static constexpr float kEmaAlpha = 0.45f;
    bool active_ = false;
    float smoothCx_ = 0.5f;
    float smoothCy_ = 0.5f;
    float smoothFaceWidth_ = 0.0f;
    int64_t lastSampleMs_ = kTimeInvalid;
    int64_t lastReceivedMs_ = kTimeInvalid;
};

bool isEs3VersionString(const char* glVersion);

}  // namespace livebeauty::math
