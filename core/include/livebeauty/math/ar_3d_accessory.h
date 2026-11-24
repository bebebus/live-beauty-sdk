#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "livebeauty/math/face_mesh_topology.h"
#include "livebeauty/math/face_relight.h"
#include "livebeauty/rhi/rhi_types.h"
#include "livebeauty/types.h"

namespace livebeauty::math {

enum class Ar3dAnchor : int {
    kHeadCrown = 0,      // Headband / cat ears / crown (anchored above forehead)
    kGlassesBridge = 1,  // 3D glasses (bridge + temples occluded behind head when turning)
    kLeftEar = 2,        // Left dangling earring with spring pendulum physics
    kRightEar = 3,       // Right dangling earring with spring pendulum physics
    kNoseTip = 4,        // 3D nose prop
    kChin = 5,           // Beard / neck collar prop
};

enum class Ar3dMeshPreset : int {
    kBillboardQuad = 0,   // Perspective-foreshortened 3D quad
    kCurvedGlasses = 1,   // 3D front frame + left/right ear temples extending along -Z
    kCurvedTiara = 2,     // Cylindrical 3D crown/headband wrapping around forehead cranium
};

/** Custom 3D accessory mesh vertex in normalized local half-width space (x,y,z in ~[-1.5..+1.5], u,v in [0..1]). */
struct Ar3dMeshVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
};

struct Ar3dAccessoryConfig {
    bool enabled = false;
    Ar3dAnchor anchor = Ar3dAnchor::kHeadCrown;
    Ar3dMeshPreset meshPreset = Ar3dMeshPreset::kBillboardQuad;
    rhi::TextureId texture = rhi::kInvalidTexture;
    float scale = 1.0f;
    float offsetX = 0.0f;          // In canonical face-width units (+X = subject left)
    float offsetY = 0.0f;          // +Y = up
    float offsetZ = 0.0f;          // +Z = toward camera (negative = behind face plane for occlusion)
    float yawOffsetRad = 0.0f;
    float pitchOffsetRad = 0.0f;
    float rollOffsetRad = 0.0f;
    float opacity = 1.0f;
    float textureAspect = 1.0f;
    float springStiffness = 0.0f;  // 0 = rigid, >0 = damped spring sway (e.g. 18.0 for earrings/ears)
    float springDamping = 4.5f;    // Damping factor (higher = quicker settle)
    bool headOcclusion = true;     // Occlude pixels behind the 3D head depth shell

    // v3.5: Optional custom indexed 3D triangle mesh (overrides meshPreset when non-null & valid)
    const Ar3dMeshVertex* customVertices = nullptr;
    int customVertexCount = 0;
    const uint16_t* customIndices = nullptr;
    int customIndexCount = 0;
};

constexpr int kMaxAr3dLayers = 4;

/**
 * 2nd-order damped spring pendulum driven by head acceleration & rotation velocity.
 * Strictly bounded and dt-clamped so earrings / cat ears sway naturally without ever diverging.
 */
class SpringPendulum {
public:
    void reset();

    /** Advance spring state given current anchor screen position & head roll; returns angular/linear sway offsets. */
    void update(float anchorX, float anchorY, float headRollRad, float stiffness, float damping, int64_t nowMs);

    float swayAngleRad() const { return angle_; }
    float swayOffsetX() const { return offsetX_; }
    float swayOffsetY() const { return offsetY_; }

private:
    bool initialized_ = false;
    int64_t lastMs_ = kTimeInvalid;
    float prevAnchorX_ = 0.0f;
    float prevAnchorY_ = 0.0f;
    float prevVelX_ = 0.0f;
    float prevVelY_ = 0.0f;
    float prevRoll_ = 0.0f;
    float angle_ = 0.0f;
    float angleVel_ = 0.0f;
    float offsetX_ = 0.0f;
    float offsetY_ = 0.0f;
};

/**
 * 3D AR Accessory & Head-Occluder geometry generator.
 *
 * - Builds a 3D depth-shell vertex stream from the 468 canonical landmarks (`kAr3dPos2Uv2Z1` layout:
 *   ndcX, ndcY, u, v, normalizedDepth) representing the front/side skull volume.
 * - Projects perspective-foreshortened 3D accessory quads with optional spring-physics sway and
 *   per-vertex normalized depth so the fragment shader cleanly occludes temples/earrings behind the head!
 */
class Ar3dSolver {
public:
    static constexpr int kQuadVertices = 4;
    static constexpr int kFloatsPerVertex = 5;  // ndcX, ndcY, u, v, depth01 (larger = closer to camera)
    static constexpr int kQuadFloats = kQuadVertices * kFloatsPerVertex;
    static constexpr int kMaxMeshVertices = 128;
    static constexpr int kMaxMeshIndices = 384;
    static constexpr int kMaxMeshFloats = kMaxMeshVertices * kFloatsPerVertex;
    static constexpr int kHeadShellFloats = FaceMeshTopology::kFaceVertexCount * kFloatsPerVertex;

    /** Built-in 3D curved glasses mesh (front bridge + 2 perspective side temples receding along -Z). */
    static void getCurvedGlassesMesh(const Ar3dMeshVertex*& outVerts, int& outVertCount,
                                     const uint16_t*& outIndices, int& outIndexCount);

    /** Built-in 3D cylindrical crown/tiara mesh wrapping around forehead cranium along -Z arc. */
    static void getCurvedTiaraMesh(const Ar3dMeshVertex*& outVerts, int& outVertCount,
                                   const uint16_t*& outIndices, int& outIndexCount);

    /** Build 468-vertex 3D head-occluder shell (depth normalized to [0..1], higher = closer to camera). */
    static void buildHeadDepthShell(
        const float* landmarks478x2,
        const HeadPose3D& pose,
        float* outShell468x5);

    /**
     * Project a 3D AR accessory quad into NDC + UV + normalized depth, incorporating 3D head rotation
     * (perspective foreshortening on yaw/pitch) and optional spring pendulum sway.
     */
    static bool buildAccessoryQuad(
        const Ar3dAccessoryConfig& cfg,
        const float* landmarks478x2,
        const HeadPose3D& pose,
        float aspect,
        const SpringPendulum* spring,
        float* outQuad20,
        float& outAnchorX,
        float& outAnchorY);

    /**
     * v3.5: Project an arbitrary indexed 3D mesh (custom or preset) into `kAr3dPos2Uv2Z1` vertex stream.
     * Writes `outVertCount * 5` floats into `outVertices` and sets `outIndices` / `outIndexCount`.
     */
    static bool buildAccessoryMesh(
        const Ar3dAccessoryConfig& cfg,
        const float* landmarks478x2,
        const HeadPose3D& pose,
        float aspect,
        const SpringPendulum* spring,
        float* outVertices,
        int maxVertices,
        int& outVertCount,
        const uint16_t*& outIndices,
        int& outIndexCount,
        float& outAnchorX,
        float& outAnchorY);
};

}  // namespace livebeauty::math
