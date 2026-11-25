#include "livebeauty/math/face_timeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace livebeauty::math {

namespace {

constexpr int kOvalIndex = 22;
constexpr int kRxIndex = 24;
constexpr int kAxisIndex = 26;
constexpr int kEyeRadiusIndex = 18;
constexpr int kXPositions[14] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 28, 32, 36};
constexpr int kYPositions[14] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 23, 29, 33, 37};
constexpr int kEllipses[4] = {kOvalIndex, 28, 32, 36};
constexpr float kCores[4] = {0.50f, 0.65f, 0.55f, 0.55f};

constexpr float kRadiusMin = 1e-3f;
constexpr float kRadiusFloor = 0.01f;
constexpr float kMinShift = 1e-3f;
constexpr float kFcMinHz = 2.8f;
constexpr float kBeta = 35.0f;
constexpr float kSpeedDead = 0.25f;
constexpr float kDCutoffHz = 6.0f;
constexpr int64_t kSpeedBaselineMs = 100;
constexpr float kHoldDecayMs = 45.0f;
constexpr int64_t kBaselineMs = 200;
constexpr float kStepJitter = 0.015f;
constexpr float kIntervalTauMs = 500.0f;
constexpr float kIntervalMinMs = 16.0f;
constexpr float kIntervalMaxMs = 200.0f;
constexpr int64_t kDtClampMs = 100;
constexpr float kEyeFade[2] = {0.35f, 0.8f};
constexpr float kFaceFade[2] = {0.15f, 0.35f};
constexpr float kEyeMakeupMotion[2] = {0.22f, 0.52f};
constexpr float kEyeMakeupExtrap[2] = {0.14f, 0.38f};
constexpr float kEyeMakeupOutMs = 30.0f;
constexpr float kEyeMakeupInMs = 36.0f;
constexpr float kDenoiseFade[2] = {0.05f, 0.12f};

}  // namespace

FaceTimeline::FaceTimeline() {
    for (int i = 0; i < 32; ++i) {
        const double ang = 2.0 * 3.14159265358979323846 * static_cast<double>(i) / 32.0;
        cosTable_[i] = static_cast<float>(std::cos(ang));
        sinTable_[i] = static_cast<float>(std::sin(ang));
    }
    meshPresent_.fill(false);
    meshHasIris_.fill(false);
    eventFace_.fill(false);
}

int64_t FaceTimeline::latest() const {
    return (count_ == 0) ? kTimeInvalid : times_[(head_ - 1 + kHistory) % kHistory];
}

int64_t FaceTimeline::missSince() const {
    if (eventCount_ == 0) return kTimeInvalid;
    return eventFace_[(eventHead_ - 1 + kEvents) % kEvents] ? kTimeInvalid : missRunStart(0);
}

float FaceTimeline::horizonMs() const {
    return detail::clampf(kExtrapolateIntervals * sampleIntervalMs_, kExtrapolateMinMs, kExtrapolateMaxMs);
}

float FaceTimeline::missDebounceMs() const {
    return std::max(kMissDebounceMinMs, kMissDebounceIntervals * sampleIntervalMs_);
}

float FaceTimeline::busyHoldMs() const {
    return detail::clampf(2.0f * sampleIntervalMs_ + 50.0f, kBusyHoldMinMs, kBusyHoldMaxMs);
}

void FaceTimeline::reset() {
    clearHistory();
    meshFilter_.reset(false);
    sampleIntervalMs_ = kDefaultIntervalMs;
    intervalKnown_ = false;
    last_ = kTimeInvalid;
    pendingDt_ = 0.0f;
    steadyEyeTarget_ = 1.0f;
    steadyFaceTarget_ = 1.0f;
    eyeMakeupTarget_ = 1.0f;
    presence_ = 0.0f;
    offsetX_ = 0.0f;
    offsetY_ = 0.0f;
    speed_ = 0.0f;
    steadyEye_ = 1.0f;
    steadyFace_ = 1.0f;
    eyeMakeupKeep_ = 1.0f;
    denoiseKeep_ = 1.0f;
    holdX_ = 0.0f;
    holdY_ = 0.0f;
    ++runId_;
}

void FaceTimeline::clearHistory() {
    count_ = 0;
    head_ = 0;
    edxX_ = 0.0f;
    edxY_ = 0.0f;
    velocityX_ = 0.0f;
    velocityY_ = 0.0f;
    sampleSpeed_ = 0.0f;
    sampleMotion_ = 0.0f;
    holdX_ = 0.0f;
    holdY_ = 0.0f;
    steadyEye_ = 1.0f;
    steadyFace_ = 1.0f;
    eyeMakeupKeep_ = 1.0f;
    steadyEyeTarget_ = 1.0f;
    steadyFaceTarget_ = 1.0f;
    eyeMakeupTarget_ = 1.0f;
    poseVelValid_ = false;
    poseMoving_ = false;
    poseVelGain_ = 0.0f;
    meshPresent_.fill(false);
    meshHasIris_.fill(false);
    eventHead_ = 0;
    eventCount_ = 0;
    runFrom_ = kTimeInvalid;
    runStart_ = kTimeInvalid;
    prevRunStart_ = kTimeInvalid;
    shownMax_ = kTimeInvalid;
    meshFilter_.reset(true);
}

void FaceTimeline::dropSamples() {
    count_ = 0;
    edxX_ = 0.0f;
    edxY_ = 0.0f;
    velocityX_ = 0.0f;
    velocityY_ = 0.0f;
    sampleSpeed_ = 0.0f;
    sampleMotion_ = 0.0f;
    holdX_ = 0.0f;
    holdY_ = 0.0f;
    steadyEye_ = 1.0f;
    steadyFace_ = 1.0f;
    eyeMakeupKeep_ = 1.0f;
    steadyEyeTarget_ = 1.0f;
    steadyFaceTarget_ = 1.0f;
    eyeMakeupTarget_ = 1.0f;
    poseVelValid_ = false;
    poseMoving_ = false;
    poseVelGain_ = 0.0f;
    meshPresent_.fill(false);
    meshHasIris_.fill(false);
}

void FaceTimeline::add(int64_t sampleTs, const float* targetState41, const FaceMeshFrame* mesh) {
    if (!targetState41) return;
    if (count_ > 0 && sampleTs <= latest()) return;
    const int64_t ms = (count_ > 0) ? missSince() : kTimeInvalid;
    const bool exitedDuringMiss = (ms != kTimeInvalid) &&
        (static_cast<float>(sampleTs - ms) >= missDebounceMs() + kExitRampMs);
    if (count_ > 0 && (sampleTs - latest() > kMaxGapMs || exitedDuringMiss)) {
        dropSamples();
        presence_ = 0.0f;
    }
    const int slot = head_;
    if (mesh != nullptr) {
        if (count_ == 0) meshFilter_.reset(true);
        meshFilter_.filter(sampleTs, mesh->landmarks.data(), aspect_, meshCoords_[slot].data(), meshPose_[slot].data());
        if (meshFilter_.jumped() && count_ > 0) dropSamples();
        auto& meta = meshMeta_[slot];
        if (!FaceMeshFrame::fillMeta(meshCoords_[slot].data(), FaceMeshFrame::kCoordCount, aspect_, meta.data())) {
            meta = mesh->meta;
        }
        meta[FaceMeshFrame::kMetaRightIrisGate] = meshFilter_.rightGate();
        meta[FaceMeshFrame::kMetaLeftIrisGate] = meshFilter_.leftGate();
        meshPresent_[slot] = true;
        meshHasIris_[slot] = mesh->hasIris;
    } else {
        meshPresent_[slot] = false;
        meshHasIris_[slot] = false;
    }

    const float x = targetState41[kOvalIndex];
    const float y = targetState41[kOvalIndex + 1] / aspect_;
    const int prevSlot = (head_ - 1 + kHistory) % kHistory;
    if (mesh != nullptr && FaceMeshFrame::fillGeometryState(
            meshCoords_[slot].data(), FaceMeshFrame::kCoordCount,
            meshMeta_[slot][FaceMeshFrame::kMetaAxisX], meshMeta_[slot][FaceMeshFrame::kMetaAxisY],
            aspect_, targetState41[kStateSize - 1], states_[slot].data())) {
        if (count_ == 0) {
            edxX_ = 0.0f;
            edxY_ = 0.0f;
        }
    } else if (count_ == 0) {
        edxX_ = 0.0f;
        edxY_ = 0.0f;
        std::memcpy(states_[slot].data(), targetState41, sizeof(float) * kStateSize);
    } else {
        const auto& previous = states_[prevSlot];
        const float te = static_cast<float>(sampleTs - latest());
        int base = prevSlot;
        for (int k = 1; k < count_; ++k) {
            const int i = (head_ - 1 - k + kHistory) % kHistory;
            if (sampleTs - times_[i] > kSpeedBaselineMs) break;
            base = i;
        }
        const float span = static_cast<float>(sampleTs - times_[base]);
        const float d = detail::oneEuroAlpha(kDCutoffHz, te);
        edxX_ += d * ((x - rawX_[base]) / span * 1000.0f - edxX_);
        edxY_ += d * ((y - rawY_[base]) / span * 1000.0f - edxY_);
        const float rx = std::max(previous[kRxIndex], kRadiusMin);
        const float centerExcess = std::max(0.0f, std::hypot(edxX_, edxY_) / rx - kSpeedDead);
        const float a = detail::oneEuroAlpha(kFcMinHz + kBeta * centerExcess, te);
        auto& out = states_[slot];
        for (int i = 0; i < kStateSize; ++i) {
            out[i] = previous[i] + a * (targetState41[i] - previous[i]);
        }
    }

    const bool hadVel = poseVelValid_;
    if (mesh != nullptr) {
        const float dead = meshFilter_.poseDead();
        poseVelGain_ = detail::smoothstep(dead * 0.75f, dead * 1.65f + 0.012f, meshFilter_.poseSpeed());
        poseMoving_ = (poseVelGain_ > 1e-3f);
    } else {
        poseVelGain_ = 0.0f;
        poseMoving_ = false;
    }
    poseVelValid_ = (mesh != nullptr && count_ > 0 && meshPresent_[prevSlot]);
    if (poseVelValid_) {
        const float dt = static_cast<float>(sampleTs - times_[prevSlot]);
        const int prev2Slot = (head_ - 2 + kHistory) % kHistory;
        const bool hasPrev2 = (count_ >= 2 && meshPresent_[prev2Slot] && times_[prevSlot] > times_[prev2Slot]);
        const float dtPrev = hasPrev2 ? static_cast<float>(times_[prevSlot] - times_[prev2Slot]) : dt;
        for (int k = 0; k < FaceMeshFilter::kPoseSize; ++k) {
            const float v01 = (meshPose_[slot][k] - meshPose_[prevSlot][k]) / dt;
            if (k < 2) {
                float leadVel = v01;
                if (hasPrev2 && dtPrev <= static_cast<float>(kMaxGapMs)) {
                    const float v12 = (meshPose_[prevSlot][k] - meshPose_[prev2Slot][k]) / dtPrev;
                    // Gate 2nd-derivative curvature lead by poseVelGain_ and frame rate so low-fps turnarounds never overshoot.
                    const float rateScale = std::min(1.0f, 35.0f / std::max(dt, 1.0f));
                    leadVel = v01 + 0.45f * rateScale * poseVelGain_ * (dt / (0.5f * (dt + dtPrev))) * (v01 - v12);
                    if (v01 * leadVel < 0.0f) leadVel = 0.0f;
                }
                poseVel_[k] = hadVel ? (poseVel_[k] + 0.90f * (leadVel - poseVel_[k])) : leadVel;
            } else {
                poseVel_[k] = hadVel ? (poseVel_[k] + 0.55f * (v01 - poseVel_[k])) : v01;
            }
        }
    } else {
        poseVel_.fill(0.0f);
    }
    const bool restart = (count_ == 0);
    rawX_[slot] = x;
    rawY_[slot] = y;
    normalizeAxis(states_[slot].data());
    times_[slot] = sampleTs;
    head_ = (head_ + 1) % kHistory;
    count_ = std::min(count_ + 1, kHistory);
    if (restart) startRun(sampleTs, std::max(presence_, 0.50f), true);
    else redetect(sampleTs);
    pushEvent(sampleTs, true);
    updateVelocity(sampleTs, slot);
    if (mesh != nullptr) {
        const float s = meshFilter_.poseScale();
        sampleSpeed_ = s * std::max(0.0f, meshFilter_.poseSpeed() - meshFilter_.poseDead());
        sampleMotion_ = meshFilter_.motionSpeed();
    } else {
        sampleSpeed_ = std::hypot(velocityX_, velocityY_) * 1000.0f;
        sampleMotion_ = 0.0f;
    }
}

void FaceTimeline::addMiss(int64_t sampleTs) {
    pushEvent(sampleTs, false);
}

void FaceTimeline::update(int64_t nowMs) {
    const float dt = (last_ == kTimeInvalid) ? 0.0f
        : static_cast<float>(std::clamp<int64_t>(nowMs - last_, 0, kDtClampMs));
    last_ = nowMs;
    if (count_ > 0 && nowMs - latest() > kClearAfterMs) clearHistory();
    if (count_ > 0 && nowMs - latest() > kMaxGapMs) {
        velocityX_ = 0.0f;
        velocityY_ = 0.0f;
        sampleSpeed_ = 0.0f;
        sampleMotion_ = 0.0f;
        poseMoving_ = false;
        poseVelGain_ = 0.0f;
    }
    if (count_ == 0) {
        sampleSpeed_ = 0.0f;
        sampleMotion_ = 0.0f;
    }
    speed_ = sampleSpeed_;
    const bool inMissRun = (count_ > 0 && missSince() != kTimeInvalid);
    const float denoiseTarget = inMissRun
        ? 0.0f
        : (1.0f - detail::smoothstep(kDenoiseFade[0], kDenoiseFade[1], std::max(speed_, sampleMotion_)));
    denoiseKeep_ = approach(denoiseKeep_, denoiseTarget, dt, 16.0f, 240.0f);
    steadyEyeTarget_ = 1.0f;
    steadyFaceTarget_ = 1.0f;
    eyeMakeupTarget_ = 1.0f;
    pendingDt_ = dt;
}

const float* FaceTimeline::at(int64_t t) {
    presence_ = presenceAt(t);
    if (count_ == 0) {
        offsetX_ = 0.0f;
        offsetY_ = 0.0f;
        settle();
        return nullptr;
    }
    const int newest = (head_ - 1 + kHistory) % kHistory;
    const int oldest = (head_ - count_ + kHistory) % kHistory;
    offsetX_ = 0.0f;
    offsetY_ = 0.0f;
    if (t >= times_[newest]) {
        output_ = states_[newest];
        const float horizon = std::min(static_cast<float>(t - times_[newest]), horizonMs());
        float mx = 0.0f, my = 0.0f;
        if (meshPresent_[newest]) {
            if (poseVelValid_ && poseVelGain_ > 1e-3f && horizon > 0.0f) {
                float shift[2];
                FaceMeshFilter::centerShift(
                    meshCoords_[newest].data(), meshPose_[newest].data(), poseVel_.data(),
                    horizon * kExtrapolateDamp * poseVelGain_, aspect_, shift);
                mx = shift[0];
                my = shift[1];
            }
        } else {
            mx = velocityX_ * horizon;
            my = velocityY_ * horizon;
        }
        if (mx != 0.0f || my != 0.0f) {
            for (int idx : kXPositions) output_[idx] += mx;
            for (int idx : kYPositions) output_[idx] += my * aspect_;
            offsetX_ = -mx;
            offsetY_ = -my;
        }
    } else if (t <= times_[oldest]) {
        output_ = states_[oldest];
    } else {
        int k = newest;
        while (times_[(k - 1 + kHistory) % kHistory] > t) k = (k - 1 + kHistory) % kHistory;
        const int ia = (k - 1 + kHistory) % kHistory;
        const auto& a = states_[ia];
        const auto& b = states_[k];
        const float g = static_cast<float>(t - times_[ia]) / static_cast<float>(times_[k] - times_[ia]);
        for (int i = 0; i < kStateSize; ++i) output_[i] = a[i] + g * (b[i] - a[i]);
        normalizeAxis(output_.data());
    }

    if (std::hypot(offsetX_, offsetY_) >= std::hypot(holdX_, holdY_)) {
        holdX_ = offsetX_;
        holdY_ = offsetY_;
    } else {
        const float k = std::exp(-pendingDt_ / kHoldDecayMs);
        holdX_ = offsetX_ + (holdX_ - offsetX_) * k;
        holdY_ = offsetY_ + (holdY_ - offsetY_) * k;
    }
    const float distance = std::hypot(holdX_, holdY_);
    const float eye = std::max({output_[kEyeRadiusIndex], output_[kEyeRadiusIndex + 3], kRadiusFloor});
    const float face = std::max(output_[kRxIndex], kRadiusFloor);
    const float motion = std::max(speed_, sampleMotion_);
    // Aligned frames (distance == 0) with un-double-filtered landmarks track within 2-3 px during handheld shake:
    // do NOT crush eye/nose/mouth beauty strength down to 0.25 on normal shake (only softly damp extreme motion blur).
    const float shakeEyeKeep = 1.0f - 0.45f * detail::smoothstep(1.35f, 2.50f, motion);
    steadyEyeTarget_ = (1.0f - detail::smoothstep(kEyeFade[0], kEyeFade[1], distance / eye)) * shakeEyeKeep;
    // Active landmark tracking must NEVER pump face-shape strength on speech or in-frame head sway;
    // only fade gently if tracking is coasting in a miss run or beyond 0.55..1.10 face half-widths.
    if (missSince() == kTimeInvalid) {
        steadyFaceTarget_ = 1.0f - 0.50f * detail::smoothstep(0.55f, 1.10f, distance / face);
    } else {
        steadyFaceTarget_ = 1.0f - detail::smoothstep(kFaceFade[0], kFaceFade[1], distance / face);
    }
    const float motionKeep = 1.0f - 0.85f * detail::smoothstep(kEyeMakeupMotion[0], kEyeMakeupMotion[1], motion);
    const float extrapKeep = 1.0f - detail::smoothstep(kEyeMakeupExtrap[0], kEyeMakeupExtrap[1], distance / eye);
    eyeMakeupTarget_ = motionKeep * extrapKeep;
    settle();
    return output_.data();
}

bool FaceTimeline::poseAt(int64_t t, float outPose6[kPose6Size]) const {
    if (!outPose6 || count_ == 0) return false;
    const int newest = (head_ - 1 + kHistory) % kHistory;
    const int oldest = (head_ - count_ + kHistory) % kHistory;
    if (t < times_[oldest] || t > times_[newest]) return false;
    if (t == times_[oldest]) {
        std::memcpy(outPose6, states_[oldest].data() + kOvalIndex, sizeof(float) * kPose6Size);
        return true;
    }
    int k = newest;
    while (times_[(k - 1 + kHistory) % kHistory] >= t) k = (k - 1 + kHistory) % kHistory;
    const int ia = (k - 1 + kHistory) % kHistory;
    const auto& a = states_[ia];
    const auto& b = states_[k];
    const float g = static_cast<float>(t - times_[ia]) / static_cast<float>(times_[k] - times_[ia]);
    for (int i = 0; i < kPose6Size; ++i) {
        outPose6[i] = a[kOvalIndex + i] + g * (b[kOvalIndex + i] - a[kOvalIndex + i]);
    }
    const float n = std::hypot(outPose6[4], outPose6[5]);
    if (n > 1e-6f) {
        outPose6[4] /= n;
        outPose6[5] /= n;
    }
    return true;
}

bool FaceTimeline::meshAt(int64_t t, SmoothedMeshView& outView) {
    presence_ = presenceAt(t);
    if (count_ == 0) return false;
    const int newest = (head_ - 1 + kHistory) % kHistory;
    const int oldest = (head_ - count_ + kHistory) % kHistory;
    bool hasIris = false;
    if (t >= times_[newest]) {
        if (!meshPresent_[newest]) return false;
        meshOutMeta_ = meshMeta_[newest];
        const float horizon = std::min(static_cast<float>(t - times_[newest]), horizonMs()) * kExtrapolateDamp * poseVelGain_;
        if (poseVelValid_ && poseVelGain_ > 1e-3f && horizon > 0.0f) {
            FaceMeshFilter::extrapolate(
                meshCoords_[newest].data(), meshPose_[newest].data(), poseVel_.data(),
                horizon, aspect_, meshOutCoords_.data());
            float shift[2];
            FaceMeshFilter::centerShift(
                meshCoords_[newest].data(), meshPose_[newest].data(), poseVel_.data(),
                horizon, aspect_, shift);
            meshOutMeta_[FaceMeshFrame::kMetaCenterX] += shift[0];
            meshOutMeta_[FaceMeshFrame::kMetaCenterY] += shift[1] * aspect_;
            const float omega = FaceMeshFilter::angularVelOf(meshPose_[newest].data(), poseVel_.data());
            const float dTheta = horizon * omega;
            const float c = std::cos(dTheta);
            const float sn = std::sin(dTheta);
            const float ax = meshOutMeta_[FaceMeshFrame::kMetaAxisX];
            const float ay = meshOutMeta_[FaceMeshFrame::kMetaAxisY];
            meshOutMeta_[FaceMeshFrame::kMetaAxisX] = c * ax - sn * ay;
            meshOutMeta_[FaceMeshFrame::kMetaAxisY] = sn * ax + c * ay;
            normalizeMetaAxis(meshOutMeta_.data());
        } else {
            meshOutCoords_ = meshCoords_[newest];
        }
        hasIris = meshHasIris_[newest];
    } else if (t <= times_[oldest]) {
        if (!meshPresent_[oldest]) return false;
        meshOutCoords_ = meshCoords_[oldest];
        meshOutMeta_ = meshMeta_[oldest];
        hasIris = meshHasIris_[oldest];
    } else {
        int k = newest;
        while (times_[(k - 1 + kHistory) % kHistory] > t) k = (k - 1 + kHistory) % kHistory;
        const int iA = (k - 1 + kHistory) % kHistory;
        const int iB = k;
        const bool hasA = meshPresent_[iA];
        const bool hasB = meshPresent_[iB];
        if (!hasA && !hasB) return false;
        if (hasA && hasB) {
            const float g = static_cast<float>(t - times_[iA]) / static_cast<float>(times_[iB] - times_[iA]);
            for (int i = 0; i < FaceMeshFrame::kCoordCount; ++i) {
                meshOutCoords_[i] = meshCoords_[iA][i] + g * (meshCoords_[iB][i] - meshCoords_[iA][i]);
            }
            for (int i = 0; i < FaceMeshFrame::kMetaSize; ++i) {
                meshOutMeta_[i] = meshMeta_[iA][i] + g * (meshMeta_[iB][i] - meshMeta_[iA][i]);
            }
            normalizeMetaAxis(meshOutMeta_.data());
            hasIris = meshHasIris_[iA] && meshHasIris_[iB];
        } else {
            const int src = hasB ? iB : iA;
            meshOutCoords_ = meshCoords_[src];
            meshOutMeta_ = meshMeta_[src];
            hasIris = meshHasIris_[src];
        }
    }
    outView.landmarks = meshOutCoords_.data();
    outView.meta = meshOutMeta_.data();
    outView.hasIris = hasIris;
    return true;
}

float FaceTimeline::presenceAt(int64_t t) {
    if (t > shownMax_) shownMax_ = t;
    if (count_ == 0 || runFrom_ == kTimeInvalid) return 0.0f;
    const int64_t start = (t >= runFrom_) ? runStart_ : prevRunStart_;
    if (start == kTimeInvalid || t <= start) return 0.0f;
    const float fadeIn = detail::smoothstep(0.0f, 1.0f, static_cast<float>(t - start) / kTInMs);
    if (fadeIn <= 0.0f) return 0.0f;
    int e = -1;
    for (int k = 0; k < eventCount_; ++k) {
        const int i = (eventHead_ - 1 - k + kEvents) % kEvents;
        if (eventTimes_[i] <= t) {
            e = k;
            break;
        }
    }
    if (e < 0) return fadeIn;
    const int idx = (eventHead_ - 1 - e + kEvents) % kEvents;
    const float fastMotion = detail::smoothstep(0.28f, 0.70f, std::max(speed_, sampleMotion_));
    const float effExitRamp = std::max(kExitRampMs * (1.0f - 0.55f * fastMotion), 12.0f);
    float exitVal = 1.0f;
    if (eventFace_[idx]) {
        if (e == 0) {
            const float effBusyHold = busyHoldMs() * (1.0f - 0.70f * fastMotion);
            const float over = detail::clampf(static_cast<float>(t - eventTimes_[idx]) - effBusyHold, 0.0f, effExitRamp);
            exitVal = 1.0f - over / effExitRamp;
        }
    } else {
        const int64_t ms = missRunStart(e);
        const float effMissDebounce = missDebounceMs() * (1.0f - 0.85f * fastMotion);
        const float over = detail::clampf(static_cast<float>(t - ms) - effMissDebounce, 0.0f, effExitRamp);
        exitVal = 1.0f - over / effExitRamp;
    }
    return fadeIn * exitVal;
}

int64_t FaceTimeline::missRunStart(int fromNewest) const {
    int64_t start = eventTimes_[(eventHead_ - 1 - fromNewest + kEvents) % kEvents];
    for (int k = fromNewest + 1; k < eventCount_; ++k) {
        const int i = (eventHead_ - 1 - k + kEvents) % kEvents;
        if (eventFace_[i]) break;
        start = eventTimes_[i];
    }
    return start;
}

void FaceTimeline::startRun(int64_t sampleTs, float level, bool hard) {
    prevRunStart_ = hard ? kTimeInvalid : runStart_;
    runFrom_ = sampleTs;
    const float x = (level <= 0.0f) ? 0.0f
        : static_cast<float>(0.5 - std::sin(std::asin(1.0 - 2.0 * static_cast<double>(detail::clampf(level, 0.0f, 1.0f))) / 3.0));
    runStart_ = sampleTs - static_cast<int64_t>(x * kTInMs);
    ++runId_;
}

void FaceTimeline::redetect(int64_t sampleTs) {
    if (eventCount_ == 0) return;
    const int newest = (eventHead_ - 1 + kEvents) % kEvents;
    if (eventFace_[newest]) return;
    const int64_t ms = missRunStart(0);
    if (static_cast<float>(sampleTs - ms) <= missDebounceMs() + kExitRampMs &&
        shownMax_ <= ms + static_cast<int64_t>(missDebounceMs())) {
        for (int k = 0; k < eventCount_; ++k) {
            const int i = (eventHead_ - 1 - k + kEvents) % kEvents;
            if (eventFace_[i]) break;
            eventFace_[i] = true;
        }
        return;
    }
    const float missed = static_cast<float>(sampleTs - ms) - missDebounceMs();
    if (missed <= 0.0f) return;
    const float level = 1.0f - detail::clampf(missed, 0.0f, kExitRampMs) / kExitRampMs;
    startRun(sampleTs, std::max(level, 0.50f), false);
}

void FaceTimeline::pushEvent(int64_t ts, bool face) {
    if (eventCount_ > 0) {
        const int64_t lastTs = eventTimes_[(eventHead_ - 1 + kEvents) % kEvents];
        if (ts <= lastTs) return;
        const float dt = static_cast<float>(ts - lastTs);
        if (dt <= static_cast<float>(kMaxGapMs)) {
            sampleIntervalMs_ = !intervalKnown_ ? dt
                : sampleIntervalMs_ + (1.0f - std::exp(-dt / kIntervalTauMs)) * (dt - sampleIntervalMs_);
            sampleIntervalMs_ = detail::clampf(sampleIntervalMs_, kIntervalMinMs, kIntervalMaxMs);
            intervalKnown_ = true;
        }
    }
    eventTimes_[eventHead_] = ts;
    eventFace_[eventHead_] = face;
    eventHead_ = (eventHead_ + 1) % kEvents;
    eventCount_ = std::min(eventCount_ + 1, kEvents);
}

void FaceTimeline::updateVelocity(int64_t sampleTs, int slot) {
    int oldest = -1;
    for (int k = 1; k < count_; ++k) {
        const int i = (head_ - 1 - k + kHistory) % kHistory;
        if (sampleTs - times_[i] > kBaselineMs) break;
        oldest = i;
    }
    if (oldest < 0) {
        velocityX_ = 0.0f;
        velocityY_ = 0.0f;
        return;
    }
    const float span = static_cast<float>(sampleTs - times_[oldest]);
    const auto& newest = states_[slot];
    const auto& base = states_[oldest];
    float mx = newest[kOvalIndex] - base[kOvalIndex];
    float my = (newest[kOvalIndex + 1] - base[kOvalIndex + 1]) / aspect_;
    const float len = std::hypot(mx, my);
    const float jitter = kStepJitter * std::max(newest[kRxIndex], kRadiusMin);
    const float keep = (len <= jitter) ? 0.0f : (len - jitter) / len;
    velocityX_ = (mx * keep) / span;
    velocityY_ = (my * keep) / span;
}

void FaceTimeline::settle() {
    steadyEye_ = approach(steadyEye_, steadyEyeTarget_, pendingDt_, 28.0f, 32.0f);
    steadyFace_ = approach(steadyFace_, steadyFaceTarget_, pendingDt_, 160.0f, 100.0f);
    eyeMakeupKeep_ = approach(eyeMakeupKeep_, eyeMakeupTarget_, pendingDt_, kEyeMakeupOutMs, kEyeMakeupInMs);
    pendingDt_ = 0.0f;
}

const float* FaceTimeline::dilate(const float* geometry41, float* out41) const {
    const float ex = holdX_ * 0.5f;
    const float ey = holdY_ * 0.5f;
    if (std::hypot(ex, ey) < kMinShift) return geometry41;
    const float n = std::hypot(geometry41[kAxisIndex], geometry41[kAxisIndex + 1]);
    if (n < 1e-6f) return geometry41;
    std::memcpy(out41, geometry41, sizeof(float) * kStateSize);
    const float ux = geometry41[kAxisIndex] / n;
    const float uy = geometry41[kAxisIndex + 1] / n;
    const float along = ex * ux + ey * uy;
    const float across = -ex * uy + ey * ux;
    const float limit = std::max(geometry41[kOvalIndex + 2], geometry41[kOvalIndex + 3]);
    for (int i = 0; i < 4; ++i) {
        const int base = kEllipses[i];
        const float core = std::sqrt(kCores[i]);
        const float rx = std::max(geometry41[base + 2], kRadiusMin);
        const float ry = std::max(geometry41[base + 3], kRadiusMin);
        const float x = std::fabs(along / rx) / core;
        const float y = std::fabs(across / ry) / core;
        const float px = 1.0f + x;
        const float py = 1.0f + y;
        float worst = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float a = (cosTable_[j] + x) / px;
            const float b = (sinTable_[j] + y) / py;
            worst = std::max(worst, a * a + b * b);
        }
        const float s = std::sqrt(worst) * 1.01f;
        out41[base] = geometry41[base] + ex;
        out41[base + 1] = geometry41[base + 1] + ey * aspect_;
        const float cap = (base == kOvalIndex) ? std::numeric_limits<float>::max()
                                               : std::max({limit, geometry41[base + 2], geometry41[base + 3]});
        out41[base + 2] = std::min(geometry41[base + 2] * px * s, cap);
        out41[base + 3] = std::min(geometry41[base + 3] * py * s, cap);
    }
    return out41;
}

void FaceTimeline::normalizeAxis(float* state41) {
    const float n = std::hypot(state41[kAxisIndex], state41[kAxisIndex + 1]);
    if (n > 1e-6f) {
        state41[kAxisIndex] /= n;
        state41[kAxisIndex + 1] /= n;
    }
}

void FaceTimeline::normalizeMetaAxis(float* meta15) {
    const float n = std::hypot(meta15[FaceMeshFrame::kMetaAxisX], meta15[FaceMeshFrame::kMetaAxisY]);
    if (n > 1e-6f) {
        meta15[FaceMeshFrame::kMetaAxisX] /= n;
        meta15[FaceMeshFrame::kMetaAxisY] /= n;
    }
}

float FaceTimeline::approach(float current, float target, float dt, float outMs, float inMs) {
    return (target < current) ? std::max(target, current - dt / outMs)
                              : std::min(target, current + dt / inMs);
}

}  // namespace livebeauty::math
