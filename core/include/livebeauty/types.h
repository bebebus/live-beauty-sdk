#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace livebeauty {

constexpr float kPi = 3.14159265358979323846f;
constexpr int64_t kTimeInvalid = std::numeric_limits<int64_t>::min();

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct CropRect {
    float left = 0.0f;
    float top = 0.0f;
    float width = 1.0f;
    float height = 1.0f;

    bool isFullFrame() const {
        return left == 0.0f && top == 0.0f && width == 1.0f && height == 1.0f;
    }

    float unprojectX(float rx) const { return left + rx * width; }
    float unprojectY(float ry) const { return top + ry * height; }

    static constexpr CropRect full() { return CropRect{0.0f, 0.0f, 1.0f, 1.0f}; }
};

namespace detail {

inline float clampf(float v, float lo, float hi) {
    if (v != v) return (lo <= 0.0f && hi >= 0.0f) ? 0.0f : lo;
    return std::max(lo, std::min(hi, v));
}

inline double clampd(double v, double lo, double hi) {
    if (v != v) return (lo <= 0.0 && hi >= 0.0) ? 0.0 : lo;
    return std::max(lo, std::min(hi, v));
}

inline float smoothstep(float e0, float e1, float x) {
    const float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline double smoothstepd(double e0, double e1, double x) {
    const double t = clampd((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline bool isFinite(float v) {
    return std::isfinite(v);
}

inline float oneEuroAlpha(float hz, float teMs) {
    return static_cast<float>(1.0 - std::exp(-2.0 * 3.14159265358979323846 * static_cast<double>(hz) * static_cast<double>(teMs) / 1000.0));
}

}  // namespace detail

}  // namespace livebeauty
