#pragma once

#include <array>
#include <cstdint>

#include "livebeauty/types.h"

namespace livebeauty::math {

/** High-frequency interactive hand gestures recognized from 21-point single/dual-hand landmarks. */
enum class GestureType : uint8_t {
    kNone = 0,
    kHeart = 1,     // Two-hand heart or finger heart -> glowing rose/coral hearts & sparkles
    kThumbsUp = 2,  // Thumbs up -> golden stars & rising sparks
    kVictory = 3,   // Peace / V-sign -> neon confetti diamonds & stars
    kOpenPalm = 4,  // Open 5-finger palm -> aurora fairy dust & stars
};

/** Analytical SDF particle shape rendered in `ShaderId::kParticleBurst` (zero external PNG dependency). */
enum class ParticleShape : uint8_t {
    kHeart = 0,
    kStar = 1,
    kDiamond = 2,
    kSparkle = 3,
};

/**
 * Real-time GPU-batched 2D SDF interactive particle burst solver (`v3.6`).
 * Spawns physically damped rising/floating particles at normalized `[0..1]` top-down hand/face anchors
 * and packs all active particles into a single `VertexLayout::kRetouchPos2Uv2A4` (`8 floats/vertex`)
 * triangle batch (`<= 64` quads = `1` draw call per frame).
 */
class ParticleBurstSolver {
public:
    static constexpr int kMaxParticles = 64;
    static constexpr int kFloatsPerVertex = 8;
    static constexpr int kVertsPerQuad = 4;
    static constexpr int kIndicesPerQuad = 6;
    static constexpr int kMaxVertexFloats = kMaxParticles * kVertsPerQuad * kFloatsPerVertex;
    static constexpr int kMaxIndices = kMaxParticles * kIndicesPerQuad;

    ParticleBurstSolver();
    void reset();

    /** Spawn a burst of interactive particles centered at top-down normalized `(normX, normY)` in `[0..1]`. */
    void triggerBurst(GestureType gesture, float normX, float normY, int64_t nowMs);

    /** Returns true if any particle might still be alive at `nowMs`. */
    bool hasActiveParticles(int64_t nowMs) const;

    /**
     * Advance physics to `nowMs` and emit interleaved `kRetouchPos2Uv2A4` vertices:
     * `[ndcX, ndcY, localU(-1..1), localV(-1..1), r, g, b, shapeId * 2.0f + alpha01]`.
     * Returns number of active quads written (`0..kMaxParticles`).
     */
    int buildBatch(int64_t nowMs, float aspect, float* outVertices);

    /** Prebuilt static index table (`0,1,2, 2,1,3, ...`) for up to `kMaxParticles` quads. */
    static const std::array<uint16_t, kMaxIndices>& quadIndices();

private:
    struct Particle {
        bool active = false;
        int64_t birthMs = 0;
        int64_t durationMs = 1200;
        float startX = 0.5f;  // top-down [0..1]
        float startY = 0.5f;  // top-down [0..1]
        float vx = 0.0f;      // units/sec in isotropic space
        float vy = 0.0f;      // units/sec (negative = upward in top-down)
        float gravity = -0.18f;
        float drag = 1.4f;
        float baseRadius = 0.035f;
        float angle0 = 0.0f;
        float spinVel = 0.0f;
        Vec3 color{1.0f, 0.4f, 0.6f};
        ParticleShape shape = ParticleShape::kHeart;
    };

    uint32_t rngState_ = 0x9E3779B9u;
    int nextSlot_ = 0;
    std::array<Particle, kMaxParticles> particles_{};

    float nextRand01();
    float nextSigned();
};

}  // namespace livebeauty::math
