#include "livebeauty/math/body_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "livebeauty/math/face_geometry.h"

namespace livebeauty::math {

bool BodyGeometry::fromPose(const PoseLandmark* points, int count, float aspect, BodyGeometry& out) {
    if (!points || count < 33 || !std::isfinite(aspect) || aspect <= 0.0f) return false;
    constexpr int kLShoulder = 11;
    constexpr int kRShoulder = 12;
    constexpr int kLHip = 23;
    constexpr int kRHip = 24;
    constexpr int kLKnee = 25;
    constexpr int kRKnee = 26;
    constexpr float kMinVis = 0.5f;

    auto visible = [&](int idx, float margin, float minVis) {
        const auto& pt = points[idx];
        return std::isfinite(pt.x) && std::isfinite(pt.y) && pt.visibility >= minVis &&
               pt.x >= -margin && pt.x <= 1.0f + margin &&
               pt.y >= -margin && pt.y <= 1.0f + margin;
    };
    if (!visible(kLShoulder, 0.1f, kVisibilityKeep) || !visible(kRShoulder, 0.1f, kVisibilityKeep) ||
        !visible(kLHip, 0.1f, kVisibilityKeep) || !visible(kRHip, 0.1f, kVisibilityKeep)) {
        return false;
    }

    const double ratio = static_cast<double>(aspect);
    auto px = [&](int idx) { return static_cast<double>(points[idx].x); };
    auto py = [&](int idx) { return 1.0 - static_cast<double>(points[idx].y); };
    auto mx = [&](int idx) { return px(idx); };
    auto my = [&](int idx) { return py(idx) / ratio; };

    const double shX = 0.5 * (px(kLShoulder) + px(kRShoulder));
    const double shY = 0.5 * (py(kLShoulder) + py(kRShoulder));
    const double hipX = 0.5 * (px(kLHip) + px(kRHip));
    const double hipY = 0.5 * (py(kLHip) + py(kRHip));

    const double msX = shX;
    const double msY = shY / ratio;
    const double mhX = hipX;
    const double mhY = hipY / ratio;
    const double dx = mhX - msX;
    const double dy = mhY - msY;
    const double torso = std::hypot(dx, dy);
    if (torso < 0.05 || torso > 1.5) return false;

    const double shSpan = std::hypot(mx(kLShoulder) - mx(kRShoulder), my(kLShoulder) - my(kRShoulder));
    const double hipSpan = std::hypot(mx(kLHip) - mx(kRHip), my(kLHip) - my(kRHip));
    const double half = 0.5 * std::max(shSpan, hipSpan);
    if (half < 0.02 || half > 0.8) return false;

    const double ax = dx / torso;
    const double ay = dy / torso;
    if (-ay < 0.64) return false;

    auto belowHip = [&](int idx, float minVis) {
        if (!visible(idx, 0.05f, minVis)) return false;
        return (mx(idx) - mhX) * ax + (my(idx) - mhY) * ay > 0.3 * torso;
    };
    const bool kneesBelow = belowHip(kLKnee, 0.35f) && belowHip(kRKnee, 0.35f);
    const float kneeVis = kneesBelow
        ? std::min(points[kLKnee].visibility, points[kRKnee].visibility)
        : 0.0f;
    const float legs = (belowHip(kLKnee, kMinVis) && belowHip(kRKnee, kMinVis)) ? 1.0f : 0.0f;
    const float vis = std::min({points[kLShoulder].visibility, points[kRShoulder].visibility,
                                points[kLHip].visibility, points[kRHip].visibility});

    out.hipX = static_cast<float>(hipX);
    out.hipY = static_cast<float>(hipY);
    out.axisX = static_cast<float>(ax);
    out.axisY = static_cast<float>(ay);
    out.torso = static_cast<float>(torso);
    out.half = static_cast<float>(half);
    out.legs = legs;
    out.visibility = vis;
    out.kneeVisibility = kneeVis;
    return true;
}

void BodyGeometrySmoother::reset() {
    hasState_ = false;
    legsOn_ = false;
    legsDropCount_ = 0;
    legsWeight_ = 0.0f;
    ts_ = kTimeInvalid;
    presence_ = 0.0f;
    missing_ = 0.0f;
    last_ = kTimeInvalid;
}

const float* BodyGeometrySmoother::update(
    const float* targetState,
    int64_t sampleTimestampMs,
    int64_t nowMs,
    float torsoVis,
    float kneeVis) {
    const float dt = (last_ == kTimeInvalid) ? 0.0f : static_cast<float>(std::clamp<int64_t>(nowMs - last_, 0, 100));
    last_ = nowMs;
    // v3.9.4 (shape-v2): Stateful hysteresis for torso acquisition (kVisibilityAcquire=0.6 vs kVisibilityKeep=0.4)
    // and knee visibility (acquire >=0.6, drop after 3 consecutive samples <0.35) + 300ms smooth legs ramp.
    const bool acceptTarget = (targetState != nullptr) &&
        (hasState_ ? (torsoVis >= BodyGeometry::kVisibilityKeep) : (torsoVis >= BodyGeometry::kVisibilityAcquire));
    if (acceptTarget) {
        missing_ = 0.0f;
        const bool newSample = !hasState_ || (sampleTimestampMs != ts_);
        if (!hasState_) {
            std::memcpy(state_.data(), targetState, sizeof(float) * BodyGeometry::kStateSize);
            legsOn_ = (kneeVis >= 0.0f) ? (kneeVis >= 0.6f) : (targetState[6] > 0.5f);
            legsDropCount_ = 0;
            legsWeight_ = legsOn_ ? 1.0f : 0.0f;
            state_[6] = legsWeight_;
            hasState_ = true;
        } else if (newSample) {
            for (int i = 0; i < BodyGeometry::kStateSize - 1; ++i) {
                state_[i] += (targetState[i] - state_[i]) * kAlpha;
            }
            const float len = std::hypot(state_[2], state_[3]);
            if (len > 1e-4f) {
                state_[2] /= len;
                state_[3] /= len;
            }
            if (kneeVis >= 0.0f) {
                if (!legsOn_) {
                    if (kneeVis >= 0.60f) {
                        legsOn_ = true;
                        legsDropCount_ = 0;
                    }
                } else {
                    if (kneeVis < 0.35f) {
                        if (++legsDropCount_ >= 3) {
                            legsOn_ = false;
                            legsDropCount_ = 0;
                        }
                    } else {
                        legsDropCount_ = 0;
                    }
                }
            } else {
                if (!legsOn_) {
                    if (targetState[6] > 0.5f) {
                        legsOn_ = true;
                        legsDropCount_ = 0;
                    }
                } else {
                    if (targetState[6] <= 0.5f) {
                        if (++legsDropCount_ >= 3) {
                            legsOn_ = false;
                            legsDropCount_ = 0;
                        }
                    } else {
                        legsDropCount_ = 0;
                    }
                }
            }
        }
        const float legsStep = dt / kLegsRampMs;
        legsWeight_ = legsOn_
            ? std::min(legsWeight_ + legsStep, 1.0f)
            : std::max(legsWeight_ - legsStep, 0.0f);
        state_[6] = legsWeight_;
        ts_ = sampleTimestampMs;
        presence_ = std::min(presence_ + dt / kFadeInMs, 1.0f);
    } else if (hasState_) {
        missing_ += dt;
        if (missing_ > kHoldMs) {
            presence_ = std::max(presence_ - dt / kFadeOutMs, 0.0f);
        }
        if (presence_ <= 0.0f) reset();
    }
    return hasState_ ? state_.data() : nullptr;
}

void NeckLift::applyFaceState(
    const float* face41, const float* body7, float neck, float aspect, float* out41, float bandOverride) {
    if (!face41 || !body7 || !out41) return;
    if (out41 != face41) {
        std::memcpy(out41, face41, sizeof(float) * FaceGeometryConstants::kStateSize);
    }
    if (!(neck > 0.0f) || !(aspect > 0.0f)) return;
    const float len = std::hypot(body7[2], body7[3]);
    if (len < 1e-4f) return;
    const float axisX = body7[2] / len;
    const float axisY = body7[3] / len;
    const float band = (bandOverride > 0.0f) ? bandOverride : (0.3f * body7[4]);
    const float width = body7[5];
    const float sideX = -axisY;
    const float sideY = axisX;
    const float s = (face41[22] - body7[0]) * sideX + (face41[23] / aspect - body7[1] / aspect) * sideY;
    const float t = detail::clampf((std::fabs(s) - 1.2f * width) / width, 0.0f, 1.0f);
    const float keep = 1.0f - t * t * (3.0f - 2.0f * t);
    const float lift = band * neck * keep;
    const float dx = -axisX * lift;
    const float dy = -axisY * lift * aspect;
    constexpr int kCentres[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 28, 32, 36};
    for (int i : kCentres) {
        out41[i] += dx;
        out41[i + 1] += dy;
    }
}

void NeckLift::applyLandmarks(
    const float* landmarks, int coordCount,
    float faceCenterX, float faceCenterY,
    const float* body7, float neck, float aspect,
    float* outLandmarks, float bandOverride) {
    if (!landmarks || !body7 || !outLandmarks || coordCount < FaceMeshFrame::kCoordCount) return;
    if (outLandmarks != landmarks) {
        std::memcpy(outLandmarks, landmarks, sizeof(float) * FaceMeshFrame::kCoordCount);
    }
    if (!(neck > 0.0f) || !(aspect > 0.0f)) return;
    const float len = std::hypot(body7[2], body7[3]);
    if (len < 1e-4f) return;
    const float axisX = body7[2] / len;
    const float axisY = body7[3] / len;
    const float band = (bandOverride > 0.0f) ? bandOverride : (0.3f * body7[4]);
    const float width = body7[5];
    const float sideX = -axisY;
    const float sideY = axisX;
    const float s = (faceCenterX - body7[0]) * sideX + (faceCenterY / aspect - body7[1] / aspect) * sideY;
    const float t = detail::clampf((std::fabs(s) - 1.2f * width) / width, 0.0f, 1.0f);
    const float keep = 1.0f - t * t * (3.0f - 2.0f * t);
    const float lift = band * neck * keep;
    const float dx = -axisX * lift;
    const float dy = -axisY * lift * aspect;
    for (int i = 0; i < FaceMeshFrame::kLandmarkCount; ++i) {
        outLandmarks[i * 2] += dx;
        outLandmarks[i * 2 + 1] += dy;
    }
}

}  // namespace livebeauty::math
