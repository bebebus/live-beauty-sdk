#include "livebeauty/math/particle_system.h"

#include <cmath>

namespace livebeauty::math {

ParticleBurstSolver::ParticleBurstSolver() {
    reset();
}

void ParticleBurstSolver::reset() {
    for (auto& p : particles_) p.active = false;
    nextSlot_ = 0;
}

float ParticleBurstSolver::nextRand01() {
    // xorshift32 deterministic PRNG
    uint32_t x = rngState_;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rngState_ = x ? x : 0x9E3779B9u;
    return static_cast<float>(rngState_ & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

float ParticleBurstSolver::nextSigned() {
    return nextRand01() * 2.0f - 1.0f;
}

void ParticleBurstSolver::triggerBurst(GestureType gesture, float normX, float normY, int64_t nowMs) {
    if (gesture == GestureType::kNone) return;
    const float cx = detail::clampf(normX, 0.08f, 0.92f);
    const float cy = detail::clampf(normY, 0.10f, 0.90f);
    constexpr int kSpawnCount = 18;

    for (int i = 0; i < kSpawnCount; ++i) {
        Particle& p = particles_[nextSlot_];
        nextSlot_ = (nextSlot_ + 1) % kMaxParticles;

        p.active = true;
        p.birthMs = nowMs + static_cast<int64_t>((i % 5) * 28);
        p.durationMs = 1050 + static_cast<int64_t>(nextRand01() * 520.0f);
        p.startX = cx + nextSigned() * 0.045f;
        p.startY = cy + nextSigned() * 0.045f;

        const float theta = -kPi * 0.5f + nextSigned() * (kPi * 0.46f);
        const float speed = 0.20f + nextRand01() * 0.34f;
        p.vx = std::cos(theta) * speed;
        p.vy = std::sin(theta) * speed;
        p.drag = 1.35f + nextRand01() * 0.55f;
        p.angle0 = nextSigned() * 0.45f;
        p.spinVel = nextSigned() * 2.2f;

        switch (gesture) {
            case GestureType::kHeart: {
                p.gravity = -0.14f;  // buoyant float upward
                p.baseRadius = (i % 4 == 3) ? (0.022f + nextRand01() * 0.012f)
                                            : (0.032f + nextRand01() * 0.022f);
                p.shape = (i % 4 == 3) ? ParticleShape::kSparkle : ParticleShape::kHeart;
                if (i % 3 == 0) {
                    p.color = {1.00f, 0.30f, 0.54f};  // vibrant rose pink
                } else if (i % 3 == 1) {
                    p.color = {1.00f, 0.52f, 0.66f};  // soft sakura coral
                } else {
                    p.color = {1.00f, 0.84f, 0.46f};  // champagne gold accent
                }
                break;
            }
            case GestureType::kThumbsUp: {
                p.gravity = -0.10f;
                p.baseRadius = 0.028f + nextRand01() * 0.020f;
                p.shape = (i % 3 == 0) ? ParticleShape::kSparkle : ParticleShape::kStar;
                p.color = (i % 2 == 0) ? Vec3{1.00f, 0.82f, 0.26f} : Vec3{1.00f, 0.58f, 0.24f};
                break;
            }
            case GestureType::kVictory: {
                p.gravity = 0.26f;  // celebratory confetti arc
                p.vy -= 0.14f;
                p.baseRadius = 0.026f + nextRand01() * 0.018f;
                p.shape = (i % 2 == 0) ? ParticleShape::kDiamond : ParticleShape::kStar;
                if (i % 3 == 0) p.color = {0.42f, 0.88f, 1.00f};
                else if (i % 3 == 1) p.color = {0.82f, 0.46f, 1.00f};
                else p.color = {1.00f, 0.76f, 0.34f};
                break;
            }
            case GestureType::kOpenPalm:
            default: {
                p.gravity = -0.06f;
                p.baseRadius = 0.024f + nextRand01() * 0.020f;
                p.shape = (i % 3 == 0) ? ParticleShape::kStar : ParticleShape::kSparkle;
                if (i % 3 == 0) p.color = {0.56f, 0.96f, 0.88f};
                else if (i % 3 == 1) p.color = {1.00f, 0.90f, 0.56f};
                else p.color = {0.96f, 0.64f, 1.00f};
                break;
            }
        }
    }
}

bool ParticleBurstSolver::hasActiveParticles(int64_t nowMs) const {
    for (const auto& p : particles_) {
        if (!p.active) continue;
        if (nowMs <= p.birthMs + p.durationMs) return true;
    }
    return false;
}

int ParticleBurstSolver::buildBatch(int64_t nowMs, float aspect, float* outVertices) {
    if (!outVertices || !(aspect > 0.0f)) return 0;
    int quadCount = 0;

    static constexpr float kCorners[4][2] = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        {-1.0f,  1.0f},
        { 1.0f,  1.0f},
    };

    for (auto& p : particles_) {
        if (!p.active) continue;
        if (nowMs < p.birthMs) continue;
        const int64_t elapsedMs = nowMs - p.birthMs;
        if (elapsedMs >= p.durationMs || p.durationMs <= 0) {
            p.active = false;
            continue;
        }

        const float tSec = static_cast<float>(elapsedMs) * 0.001f;
        const float life01 = detail::clampf(
            static_cast<float>(elapsedMs) / static_cast<float>(p.durationMs), 0.0f, 1.0f);

        // Exponential air-drag integration + constant buoyancy/gravity
        const float dragScale = (p.drag > 1e-3f) ? ((1.0f - std::exp(-p.drag * tSec)) / p.drag) : tSec;
        const float dxIso = p.vx * dragScale + 0.018f * std::sin(tSec * 5.2f + p.angle0 * 4.0f);
        const float dyIso = p.vy * dragScale + 0.5f * p.gravity * tSec * tSec;

        const float cx = p.startX + dxIso;
        const float cyBottomUp = (1.0f - p.startY) - dyIso * aspect;
        if (cx < -0.15f || cx > 1.15f || cyBottomUp < -0.15f || cyBottomUp > 1.15f) {
            continue;
        }

        // Pop-in [0..0.16] with gentle overshoot, then smooth fade-out [0.58..1.0]
        const float popIn = detail::smoothstep(0.0f, 0.16f, life01);
        const float overshoot = 1.0f + 0.18f * std::sin(detail::clampf(life01 / 0.28f, 0.0f, 1.0f) * kPi);
        const float fadeOut = 1.0f - detail::smoothstep(0.56f, 1.0f, life01);
        const float alpha = detail::clampf(popIn * fadeOut * 0.96f, 0.0f, 0.995f);
        if (alpha <= 0.005f) continue;

        const float radius = p.baseRadius * popIn * overshoot;
        const float ang = p.angle0 + p.spinVel * tSec;
        const float ca = std::cos(ang);
        const float sa = std::sin(ang);
        const float packedShapeAlpha = static_cast<float>(static_cast<int>(p.shape)) * 2.0f + alpha;

        float* vBase = outVertices + quadCount * kVertsPerQuad * kFloatsPerVertex;
        for (int c = 0; c < 4; ++c) {
            const float lx = kCorners[c][0];
            const float ly = kCorners[c][1];
            const float rx = (lx * ca - ly * sa) * radius;
            const float ry = (lx * sa + ly * ca) * radius * aspect;
            const float ux = cx + rx;
            const float uy = cyBottomUp + ry;

            float* v = vBase + c * kFloatsPerVertex;
            v[0] = ux * 2.0f - 1.0f;
            v[1] = uy * 2.0f - 1.0f;
            v[2] = lx;
            v[3] = ly;
            v[4] = p.color.x;
            v[5] = p.color.y;
            v[6] = p.color.z;
            v[7] = packedShapeAlpha;
        }
        ++quadCount;
        if (quadCount >= kMaxParticles) break;
    }
    return quadCount;
}

const std::array<uint16_t, ParticleBurstSolver::kMaxIndices>& ParticleBurstSolver::quadIndices() {
    static const std::array<uint16_t, kMaxIndices> kIndices = []() {
        std::array<uint16_t, kMaxIndices> arr{};
        for (int q = 0; q < kMaxParticles; ++q) {
            const uint16_t b = static_cast<uint16_t>(q * 4);
            const size_t off = static_cast<size_t>(q * 6);
            arr[off + 0] = b + 0;
            arr[off + 1] = b + 1;
            arr[off + 2] = b + 2;
            arr[off + 3] = b + 2;
            arr[off + 4] = b + 1;
            arr[off + 5] = b + 3;
        }
        return arr;
    }();
    return kIndices;
}

}  // namespace livebeauty::math
