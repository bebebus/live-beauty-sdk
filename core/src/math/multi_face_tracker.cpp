#include "livebeauty/math/multi_face_tracker.h"

#include <algorithm>
#include <cmath>

namespace livebeauty::math {

void MultiFaceTracker::setAspect(float aspect) {
    aspect_ = aspect;
    for (auto& tl : timelines_) tl.setAspect(aspect);
}

void MultiFaceTracker::reset() {
    for (int i = 0; i < kMaxFaces; ++i) {
        timelines_[i].reset();
        hasLastCenter_[i] = false;
        lastCx_[i] = 0.5f;
        lastCy_[i] = 0.5f;
    }
}

void MultiFaceTracker::submitAllMiss(int64_t sampleTsMs) {
    for (auto& tl : timelines_) tl.addMiss(sampleTsMs);
}

void MultiFaceTracker::update(int64_t nowMs) {
    for (int i = 0; i < kMaxFaces; ++i) {
        timelines_[i].update(nowMs);
        if (timelines_[i].latest() == kTimeInvalid) {
            hasLastCenter_[i] = false;
        }
    }
}

int MultiFaceTracker::activeFaceCount() const {
    int n = 0;
    for (int i = 0; i < kMaxFaces; ++i) {
        if (timelines_[i].presence() > 1e-4f && timelines_[i].latest() != kTimeInvalid) {
            ++n;
        }
    }
    return n;
}

bool MultiFaceTracker::parseOne(const RawFaceInput& raw, float aspect, ParsedFace& out) const {
    if (!raw.mpXy || raw.pointCount < 468) return false;
    if (!FaceMeshFrame::fromMediaPipe(raw.mpXy, raw.pointCount, aspect, out.mesh)) return false;
    if (!FaceMeshFrame::faceGeometryFromMediaPipe(raw.mpXy, raw.pointCount, aspect, out.state41.data())) {
        if (!FaceMeshFrame::fillGeometryState(
                out.mesh.landmarks.data(), FaceMeshFrame::kCoordCount,
                out.mesh.axisX(), out.mesh.axisY(), aspect, 0.0f, out.state41.data())) {
            return false;
        }
    }
    out.cx = out.mesh.centerX();
    out.cy = out.mesh.centerY();
    out.fw = out.mesh.faceWidth();
    out.valid = true;
    return true;
}

float MultiFaceTracker::distSq(float x0, float y0, float x1, float y1, float aspect) {
    const float dx = x0 - x1;
    const float dy = (y0 - y1) / std::max(aspect, 0.01f);
    return dx * dx + dy * dy;
}

int MultiFaceTracker::submitFaces(
    int64_t sampleTsMs, const RawFaceInput* faces, int faceCount,
    int frameWidth, int frameHeight, int64_t /*receivedMs*/,
    float* outPrimaryCenterX, float* outPrimaryCenterY, float* outPrimaryFaceWidth) {
    if (frameWidth <= 0 || frameHeight <= 0) return 0;
    const float aspect = static_cast<float>(frameWidth) / static_cast<float>(frameHeight);
    if (std::fabs(aspect - aspect_) > 1e-5f) setAspect(aspect);

    const int nRaw = (faces && faceCount > 0) ? std::min(faceCount, kMaxFaces) : 0;
    std::array<ParsedFace, kMaxFaces> parsed{};
    int validCount = 0;
    for (int i = 0; i < nRaw; ++i) {
        if (parseOne(faces[i], aspect, parsed[validCount])) {
            ++validCount;
        }
    }

    if (validCount == 0) {
        submitAllMiss(sampleTsMs);
        return 0;
    }

    // Slot assignment: assign[slotIdx] = parsedIdx (-1 means miss for that slot)
    int assign[kMaxFaces] = {-1, -1};
    if (validCount == 1) {
        const bool s0 = hasLastCenter_[0] && timelines_[0].latest() != kTimeInvalid;
        const bool s1 = hasLastCenter_[1] && timelines_[1].latest() != kTimeInvalid;
        if (s0 && s1) {
            const float d0 = distSq(parsed[0].cx, parsed[0].cy, lastCx_[0], lastCy_[0], aspect);
            const float d1 = distSq(parsed[0].cx, parsed[0].cy, lastCx_[1], lastCy_[1], aspect);
            assign[d0 <= d1 ? 0 : 1] = 0;
        } else if (s1 && !s0) {
            assign[1] = 0;
        } else {
            assign[0] = 0;
        }
    } else {
        // validCount == 2: exact minimum-cost 2x2 assignment to prevent ID swaps on crossing
        const bool s0 = hasLastCenter_[0] && timelines_[0].latest() != kTimeInvalid;
        const bool s1 = hasLastCenter_[1] && timelines_[1].latest() != kTimeInvalid;
        if (s0 || s1) {
            const float c00 = s0 ? distSq(parsed[0].cx, parsed[0].cy, lastCx_[0], lastCy_[0], aspect) : 0.0f;
            const float c11 = s1 ? distSq(parsed[1].cx, parsed[1].cy, lastCx_[1], lastCy_[1], aspect) : 0.0f;
            const float c01 = s1 ? distSq(parsed[0].cx, parsed[0].cy, lastCx_[1], lastCy_[1], aspect) : 0.0f;
            const float c10 = s0 ? distSq(parsed[1].cx, parsed[1].cy, lastCx_[0], lastCy_[0], aspect) : 0.0f;
            if ((c00 + c11) <= (c01 + c10)) {
                assign[0] = 0;
                assign[1] = 1;
            } else {
                assign[0] = 1;
                assign[1] = 0;
            }
        } else {
            // Fresh dual-face entry: sort by larger faceWidth -> slot 0 (primary), smaller -> slot 1
            if (parsed[0].fw >= parsed[1].fw) {
                assign[0] = 0;
                assign[1] = 1;
            } else {
                assign[0] = 1;
                assign[1] = 0;
            }
        }
    }

    for (int s = 0; s < kMaxFaces; ++s) {
        if (assign[s] >= 0) {
            const auto& p = parsed[assign[s]];
            timelines_[s].add(sampleTsMs, p.state41.data(), &p.mesh);
            lastCx_[s] = p.cx;
            lastCy_[s] = p.cy;
            hasLastCenter_[s] = true;
            if (s == 0) {
                if (outPrimaryCenterX) *outPrimaryCenterX = p.cx;
                if (outPrimaryCenterY) *outPrimaryCenterY = p.cy;
                if (outPrimaryFaceWidth) *outPrimaryFaceWidth = p.fw;
            }
        } else {
            timelines_[s].addMiss(sampleTsMs);
        }
    }
    return validCount;
}

}  // namespace livebeauty::math
