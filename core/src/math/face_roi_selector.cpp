#include "livebeauty/math/face_roi_selector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

namespace livebeauty::math {

void FaceRoiSelector::reset() {
    active_ = false;
    smoothCx_ = 0.5f;
    smoothCy_ = 0.5f;
    smoothFaceWidth_ = 0.0f;
    lastSampleMs_ = kTimeInvalid;
    lastReceivedMs_ = kTimeInvalid;
}

void FaceRoiSelector::onFace(
    float glCenterX, float glCenterY, float faceWidth, int64_t sampleMs, int64_t receivedMs) {
    if (!std::isfinite(glCenterX) || !std::isfinite(glCenterY) ||
        !std::isfinite(faceWidth) || faceWidth < kMinFaceWidth) {
        return;
    }
    if (receivedMs < sampleMs || receivedMs - sampleMs > kMaxSampleAgeMs) {
        return;
    }
    if (lastSampleMs_ != kTimeInvalid && sampleMs < lastSampleMs_) {
        return;
    }
    const float mpCx = detail::clampf(glCenterX, 0.0f, 1.0f);
    const float mpCy = detail::clampf(1.0f - glCenterY, 0.0f, 1.0f);
    if (lastReceivedMs_ == kTimeInvalid || receivedMs < lastReceivedMs_ || receivedMs - lastReceivedMs_ > kStaleMs) {
        smoothCx_ = mpCx;
        smoothCy_ = mpCy;
        smoothFaceWidth_ = faceWidth;
    } else {
        smoothCx_ += (mpCx - smoothCx_) * kEmaAlpha;
        smoothCy_ += (mpCy - smoothCy_) * kEmaAlpha;
        smoothFaceWidth_ += (faceWidth - smoothFaceWidth_) * kEmaAlpha;
    }
    lastSampleMs_ = sampleMs;
    lastReceivedMs_ = receivedMs;
    active_ = active_ ? (smoothFaceWidth_ < kExitFaceWidth) : (smoothFaceWidth_ <= kEnterFaceWidth);
}

void FaceRoiSelector::onMiss() {
    active_ = false;
    lastSampleMs_ = kTimeInvalid;
    lastReceivedMs_ = kTimeInvalid;
}

CropRect FaceRoiSelector::select(int width, int height, int64_t nowMs) {
    if (!active_ || width <= 0 || height <= 0 ||
        lastReceivedMs_ == kTimeInvalid || nowMs < lastReceivedMs_ || nowMs - lastReceivedMs_ > kStaleMs ||
        lastSampleMs_ == kTimeInvalid || nowMs < lastSampleMs_ || nowMs - lastSampleMs_ > kMaxSampleAgeMs) {
        active_ = false;
        return CropRect::full();
    }
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    if (!std::isfinite(aspect) || aspect <= 0.0f) return CropRect::full();

    const float desiredW = detail::clampf(smoothFaceWidth_ * kExpandFactor, kMinCropSpan, kMaxCropSpan);
    float cropW = 1.0f;
    float cropH = 1.0f;
    if (aspect <= 1.0f) {
        cropW = std::min(desiredW, 1.0f);
        cropH = detail::clampf(cropW * aspect, kMinCropSpan * aspect, 1.0f);
    } else {
        cropH = detail::clampf(desiredW * aspect, kMinCropSpan, kMaxCropSpan);
        cropW = std::min(cropH / aspect, 1.0f);
    }
    if (cropW >= 0.96f || cropH >= 0.96f) return CropRect::full();

    const float left = detail::clampf(smoothCx_ - 0.5f * cropW, 0.0f, 1.0f - cropW);
    const float top = detail::clampf(smoothCy_ - 0.5f * cropH, 0.0f, 1.0f - cropH);
    return CropRect{left, top, cropW, cropH};
}

bool isEs3VersionString(const char* glVersion) {
    if (!glVersion || *glVersion == '\0') return false;
    const char* prefix = "OpenGL ES ";
    const char* found = std::strstr(glVersion, prefix);
    const char* p = found ? (found + std::strlen(prefix)) : glVersion;
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    int major = 0;
    while (*p && std::isdigit(static_cast<unsigned char>(*p))) {
        major = major * 10 + (*p - '0');
        ++p;
    }
    return major >= 3;
}

}  // namespace livebeauty::math
