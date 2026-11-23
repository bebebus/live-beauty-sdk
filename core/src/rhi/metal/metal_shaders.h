#pragma once

namespace livebeauty::rhi::metal {

/**
 * Hand-tuned Apple Metal Shading Language (MSL) equivalent of all GLSL beauty passes:
 * - PassThrough, Downsample (2/3 tap 3x3 box), Blur5 (RGB/Alpha), GuidedDiff, Dual-Band GuidedCoeff (1/3 + 1/9)
 * - 548-vertex Diffeomorphic MeshWarp & 504-vertex SkinMask
 * - 468-pt In-Place Anatomical Sub-Mesh Retouch (Aegyo-sal preserving dark circles, curved nasolabial, eye & teeth)
 * - 6-Layer UV Makeup (SoftLight/Multiply/ScreenLift + Eyelid Skin Gate + Beer-Lambert Lipstick + Contact Lens/Catchlight)
 * - 5-Axis BodyWarp, CompositePro + 64^3 3D LUT, TemporalDenoise, Background Bokeh (Premask/Blur/Mix)
 */
inline constexpr const char* kMetalBeautyShadersSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4 u[16];
};

inline float2 toMtlUv(float2 glUv) {
    return float2(glUv.x, 1.0f - glUv.y);
}

inline float luma709(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float luma601(float3 c) {
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

inline float skinWeight(float3 c) {
    float cb = 0.5f + dot(c, float3(-0.168736f, -0.331264f, 0.5f));
    float cr = 0.5f + dot(c, float3(0.5f, -0.418688f, -0.081312f));
    return smoothstep(0.26f, 0.34f, cb) * (1.0f - smoothstep(0.48f, 0.56f, cb)) *
           smoothstep(0.50f, 0.55f, cr) * (1.0f - smoothstep(0.72f, 0.80f, cr));
}

inline float ellipseE(float2 uv, float4 c, float2 u, float aspect) {
    float2 d = float2(uv.x - c.x, (uv.y - c.y) / aspect);
    float a = clamp(dot(d, u) / max(c.z, 0.0001f), -4.0f, 4.0f);
    float b = clamp(dot(d, float2(-u.y, u.x)) / max(c.w, 0.0001f), -4.0f, 4.0f);
    return a * a + b * b;
}

inline float2 faceMasks(float2 uv, float aspect, float presence, float4 oval, float2 axis,
                        float4 mouth, float4 eye0, float4 eye1) {
    if (presence <= 0.0f) return float2(0.0f);
    float inside = 1.0f - smoothstep(0.50f, 1.0f, ellipseE(uv, oval, axis, aspect));
    float e0 = 1.0f - smoothstep(0.55f, 1.0f, ellipseE(uv, eye0, axis, aspect));
    float e1 = 1.0f - smoothstep(0.55f, 1.0f, ellipseE(uv, eye1, axis, aspect));
    float lips = 1.0f - smoothstep(0.65f, 1.0f, ellipseE(uv, mouth, axis, aspect));
    float keep = (1.0f - e0) * (1.0f - e1) * (1.0f - lips);
    return presence * inside * float2(keep, 1.0f - keep);
}

inline float3 softLightBlend(float3 b, float3 s) {
    return (1.0f - 2.0f * s) * b * b + 2.0f * s * b;
}

inline float3 fitGamut(float3 c, float y) {
    float hi = max(max(c.r, c.g), c.b);
    float lo = min(min(c.r, c.g), c.b);
    float s = 1.0f;
    if (hi > 1.0f) s = min(s, (1.0f - y) / max(hi - y, 0.0001f));
    if (lo < 0.0f) s = min(s, y / max(y - lo, 0.0001f));
    return float3(y) + (c - float3(y)) * s;
}

inline float3 gradePro(float3 color, float style) {
    float y = luma709(color);
    float bell = 4.0f * y * (1.0f - y);
    float3 graded = color;
    if (style > 4.5f) graded += float3(0.045f, -0.005f, 0.025f) * bell;
    else if (style > 3.5f) graded = float3(
        dot(color, float3(0.393f, 0.769f, 0.189f)),
        dot(color, float3(0.349f, 0.686f, 0.168f)),
        dot(color, float3(0.272f, 0.534f, 0.131f)));
    else if (style > 2.5f) graded = float3(dot(color, float3(0.299f, 0.587f, 0.114f)));
    else if (style > 1.5f) graded += float3(-0.025f, 0.0f, 0.045f) * bell;
    else if (style > 0.5f) graded += float3(0.045f, 0.008f, -0.025f) * bell;
    return graded;
}

inline float2 lutCell(float level, float2 rg) {
    float row = floor(level / 8.0f);
    float col = level - row * 8.0f;
    return float2(col, row) * 0.125f + (0.5f / 512.0f) + (0.125f - 1.0f / 512.0f) * rg;
}

constexpr sampler kLinearClamp(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

// ---------------- Vertex Types & Shaders ----------------

struct QuadIn {
    float2 pos [[attribute(0)]];
    float2 uv  [[attribute(1)]];
};

struct QuadOut {
    float4 pos [[position]];
    float2 uv;
};

vertex QuadOut vs_quad(QuadIn in [[stage_in]]) {
    QuadOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    out.uv = in.uv;
    return out;
}

struct DownOut {
    float4 pos [[position]];
    float2 tap0;
    float2 tap1;
    float2 tap2;
    float2 tap3;
};

vertex DownOut vs_down(QuadIn in [[stage_in]], constant Uniforms& uni [[buffer(1)]]) {
    DownOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    float2 t = uni.u[0].xy;
    out.tap0 = in.uv + float2(-t.x, -t.y);
    out.tap1 = in.uv + float2( t.x, -t.y);
    out.tap2 = in.uv + float2(-t.x,  t.y);
    out.tap3 = in.uv + float2( t.x,  t.y);
    return out;
}

struct BlurOut {
    float4 pos [[position]];
    float2 c;
    float2 a1;
    float2 b1;
    float2 a2;
    float2 b2;
};

vertex BlurOut vs_blur(QuadIn in [[stage_in]], constant Uniforms& uni [[buffer(1)]]) {
    BlurOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    float2 d = uni.u[0].xy;
    out.c = in.uv;
    out.a1 = in.uv + d * 1.4072f;
    out.b1 = in.uv - d * 1.4072f;
    out.a2 = in.uv + d * 3.2939f;
    out.b2 = in.uv - d * 3.2939f;
    return out;
}

struct SkinMaskIn {
    float2 pos [[attribute(0)]];
    float3 w   [[attribute(1)]];
};

struct SkinMaskOut {
    float4 pos [[position]];
    float3 w;
};

vertex SkinMaskOut vs_skin_mask(SkinMaskIn in [[stage_in]]) {
    SkinMaskOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    out.w = in.w;
    return out;
}

struct RetouchIn {
    float2 pos  [[attribute(0)]];
    float2 src  [[attribute(1)]];
    float4 attr [[attribute(2)]];
};

struct RetouchOut {
    float4 pos  [[position]];
    float2 src;
    float4 attr;
};

vertex RetouchOut vs_retouch(RetouchIn in [[stage_in]]) {
    RetouchOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    out.src = in.src;
    out.attr = in.attr;
    return out;
}

struct MakeupIn {
    float2 pos      [[attribute(0)]];
    float2 screenUv [[attribute(1)]];
    float2 makeupUv [[attribute(2)]];
};

struct MakeupOut {
    float4 pos [[position]];
    float2 screenUv;
    float2 makeupUv;
};

vertex MakeupOut vs_makeup(MakeupIn in [[stage_in]]) {
    MakeupOut out;
    out.pos = float4(in.pos, 0.0f, 1.0f);
    out.screenUv = in.screenUv;
    out.makeupUv = in.makeupUv;
    return out;
}

// ---------------- Fragment Shaders ----------------

fragment float4 fs_pass_through(QuadOut in [[stage_in]], texture2d<float> tex0 [[texture(0)]]) {
    return tex0.sample(kLinearClamp, toMtlUv(clamp(in.uv, 0.0f, 1.0f)));
}

fragment float4 fs_down(DownOut in [[stage_in]], texture2d<float> tex0 [[texture(0)]]) {
    return (tex0.sample(kLinearClamp, toMtlUv(in.tap0)) +
            tex0.sample(kLinearClamp, toMtlUv(in.tap1)) +
            tex0.sample(kLinearClamp, toMtlUv(in.tap2)) +
            tex0.sample(kLinearClamp, toMtlUv(in.tap3))) * 0.25f;
}

inline float4 blur5Sample(texture2d<float> tex0, BlurOut in) {
    return tex0.sample(kLinearClamp, toMtlUv(in.c)) * 0.2042f +
           (tex0.sample(kLinearClamp, toMtlUv(in.a1)) + tex0.sample(kLinearClamp, toMtlUv(in.b1))) * 0.3040f +
           (tex0.sample(kLinearClamp, toMtlUv(in.a2)) + tex0.sample(kLinearClamp, toMtlUv(in.b2))) * 0.0939f;
}

fragment float4 fs_blur_rgb(BlurOut in [[stage_in]], texture2d<float> tex0 [[texture(0)]]) {
    return blur5Sample(tex0, in);
}

fragment float4 fs_blur_alpha(BlurOut in [[stage_in]], texture2d<float> tex0 [[texture(0)]]) {
    float3 rgb = tex0.sample(kLinearClamp, toMtlUv(in.c)).rgb;
    return float4(rgb, blur5Sample(tex0, in).a);
}

fragment float4 fs_guided_diff(BlurOut in [[stage_in]],
                               texture2d<float> tex0 [[texture(0)]],
                               texture2d<float> tex1 [[texture(1)]]) {
    float3 mean = blur5Sample(tex0, in).rgb;
    float3 diff = tex1.sample(kLinearClamp, toMtlUv(in.c)).rgb - mean;
    return float4(mean, clamp(length(diff) * 2.5f, 0.0f, 1.0f));
}

fragment float4 fs_guided_coeff(BlurOut in [[stage_in]],
                                texture2d<float> tex0 [[texture(0)]],
                                texture2d<float> tex1 [[texture(1)]],
                                constant Uniforms& uni [[buffer(1)]]) {
    float s = blur5Sample(tex0, in).a;
    float eps = max(uni.u[0].z, 0.005f);
    float evenness = uni.u[0].w;
    float3 fineMean = tex0.sample(kLinearClamp, toMtlUv(in.c)).rgb;
    float3 mean = fineMean;
    if (evenness > 0.0f) {
        float3 macroMean = tex1.sample(kLinearClamp, toMtlUv(in.c)).rgb;
        float dy = abs(luma709(macroMean - fineMean)) / 0.068f;
        float dy2 = dy * dy;
        float boneKeep = 1.0f / (1.0f + dy2 * dy2);
        float flatKeep = 1.0f - smoothstep(0.05f, 0.14f, s);
        mean = mix(fineMean, macroMean, evenness * boneKeep * flatKeep);
    }
    float r = s * s / eps;
    float rp = r * sqrt(r);
    float lo = clamp(sqrt(eps) - 0.09f, 0.12f, 0.18f);
    float a = max(rp / (1.0f + rp), smoothstep(lo, 0.30f, s));
    return float4((1.0f - a) * mean, a);
}

fragment float4 fs_skin_mask(SkinMaskOut in [[stage_in]]) {
    return float4(clamp(in.w, 0.0f, 1.0f), 1.0f);
}

fragment float4 fs_retouch_skin(RetouchOut in [[stage_in]],
                                texture2d<float> tex0 [[texture(0)]],
                                constant Uniforms& uni [[buffer(1)]]) {
    float wTear = clamp(in.attr.x * uni.u[0].x * 0.85f, 0.0f, 1.0f);
    float wFold = clamp(in.attr.y * uni.u[0].y * 0.82f, 0.0f, 1.0f);
    float alpha = clamp(wTear + wFold, 0.0f, 1.0f);
    if (alpha <= 0.002f) discard_fragment();
    float3 c = tex0.sample(kLinearClamp, toMtlUv(in.src)).rgb;
    float2 off = in.attr.zw;
    float3 target = c;
    if (wTear > wFold) {
        float3 cheekRef = tex0.sample(kLinearClamp, toMtlUv(clamp(in.src + off, 0.0f, 1.0f))).rgb;
        float yBase = luma709(c);
        float yRef = luma709(cheekRef);
        float deficit = clamp((yRef - yBase) * 2.4f, 0.0f, 0.22f);
        float liftMask = clamp(deficit * 3.8f, 0.0f, 0.75f);
        target = clamp(c + (cheekRef - c) * liftMask + float3(0.058f, 0.046f, 0.032f) * liftMask * (1.0f - yBase), 0.0f, 1.0f);
    } else {
        float3 sideA = tex0.sample(kLinearClamp, toMtlUv(clamp(in.src + off, 0.0f, 1.0f))).rgb;
        float3 sideB = tex0.sample(kLinearClamp, toMtlUv(clamp(in.src - off, 0.0f, 1.0f))).rgb;
        target = max(c, 0.5f * (sideA + sideB));
    }
    return float4(target, alpha);
}

fragment float4 fs_retouch_eye_teeth(RetouchOut in [[stage_in]],
                                     texture2d<float> tex0 [[texture(0)]],
                                     constant Uniforms& uni [[buffer(1)]]) {
    float strength = clamp(uni.u[0].x, 0.0f, 1.0f);
    if (strength <= 0.002f) discard_fragment();
    float3 c = tex0.sample(kLinearClamp, toMtlUv(in.src)).rgb;
    float y = luma601(c);
    float chroma = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    if (uni.u[0].y < 0.5f) {
        float sclera = smoothstep(0.38f, 0.62f, y) * (1.0f - smoothstep(0.12f, 0.28f, chroma));
        float3 contrasted = clamp((c - 0.42f) * 1.16f + 0.45f, 0.0f, 1.0f);
        float3 cleanSclera = mix(contrasted, float3(min(y * 1.07f + 0.025f, 1.0f)), sclera * 0.55f);
        return float4(cleanSclera, clamp(strength * 0.72f, 0.0f, 1.0f));
    } else {
        float toothMask = smoothstep(0.28f, 0.50f, y) * (1.0f - smoothstep(0.20f, 0.38f, chroma));
        float alpha = clamp(strength * toothMask * 0.84f, 0.0f, 1.0f);
        if (alpha <= 0.002f) discard_fragment();
        float targetY = clamp(y * 1.12f + 0.035f, 0.0f, 0.96f);
        float3 neutralWhite = float3(targetY, targetY, min(targetY * 1.015f, 0.98f));
        float3 whitened = mix(c * (targetY / max(y, 0.001f)), neutralWhite, 0.68f);
        return float4(clamp(whitened, 0.0f, 1.0f), alpha);
    }
}

fragment float4 fs_makeup_layer(MakeupOut in [[stage_in]],
                                texture2d<float> tex0 [[texture(0)]],
                                texture2d<float> tex1 [[texture(1)]],
                                constant Uniforms& uni [[buffer(1)]]) {
    float4 uColor = uni.u[0];
    float4 uChannel = uni.u[1];
    float4 uMode = uni.u[2];
    float2 atlasUv = in.makeupUv;
    if (uMode.y > 0.5f) {
        float isLeft = step(0.5f, in.makeupUv.x);
        float eu = isLeft < 0.5f ? (in.makeupUv.x - 0.18f) / 0.30f : (in.makeupUv.x - 0.52f) / 0.30f;
        float ev = (in.makeupUv.y - 0.21f) / 0.24f;
        if (eu <= 0.002f || eu >= 0.998f || ev <= 0.002f || ev >= 0.998f) discard_fragment();
        atlasUv = float2(eu, isLeft * 0.5f + ev * 0.5f);
    }
    float mask = dot(tex1.sample(kLinearClamp, atlasUv), uChannel);
    float alpha = clamp(mask * uColor.a, 0.0f, 1.0f);
    if (alpha <= 0.001f) discard_fragment();
    float3 base = tex0.sample(kLinearClamp, toMtlUv(clamp(in.screenUv, 0.0f, 1.0f))).rgb;
    if (uMode.z > 0.5f) {
        float baseY = luma601(base);
        float darkGate = smoothstep(0.14f, 0.30f, baseY);
        float skinGate = smoothstep(0.025f, 0.095f, base.r - base.b) * smoothstep(0.015f, 0.065f, base.r - base.g);
        alpha *= darkGate * skinGate;
        if (alpha <= 0.001f) discard_fragment();
    }
    float3 result;
    if (uMode.x < 0.5f) {
        result = mix(softLightBlend(base, uColor.rgb), base * mix(float3(1.0f), uColor.rgb, 0.68f), 0.45f);
    } else if (uMode.x < 1.5f) {
        float3 mult = base * mix(float3(1.0f), uColor.rgb, uMode.z > 0.5f ? 0.78f : 0.92f);
        result = uMode.z > 0.5f ? mix(mult, softLightBlend(base, uColor.rgb), 0.45f) : mult;
    } else {
        result = 1.0f - (1.0f - base) * (1.0f - uColor.rgb * 0.58f);
    }
    return float4(clamp(result, 0.0f, 1.0f), alpha);
}

fragment float4 fs_makeup_lip(MakeupOut in [[stage_in]],
                              texture2d<float> tex0 [[texture(0)]],
                              texture2d<float> tex1 [[texture(1)]],
                              constant Uniforms& uni [[buffer(1)]]) {
    float4 uLipColor = uni.u[0];
    float4 uLipStyle = uni.u[1];
    float feather = tex1.sample(kLinearClamp, in.makeupUv).b;
    if (feather * uLipColor.a <= 0.001f) discard_fragment();
    float2 uv = clamp(in.screenUv, 0.0f, 1.0f);
    float3 base = tex0.sample(kLinearClamp, toMtlUv(uv)).rgb;
    float baseY = luma601(base);
    float darkGate = smoothstep(0.06f, 0.20f, baseY);
    float redExcess = base.r - max(base.g, base.b);
    float teethLike = smoothstep(0.52f, 0.74f, baseY) * (1.0f - smoothstep(0.035f, 0.115f, redExcess));
    float teethGate = 1.0f - teethLike;
    float lipHue = (base.r - base.g) - 0.75f * max(base.g - base.b, 0.0f);
    float rimZone = 1.0f - smoothstep(0.35f, 0.92f, feather);
    float skinRimGate = mix(1.0f, smoothstep(0.045f, 0.155f, lipHue), rimZone * 0.92f);
    float effFeather = feather * darkGate * teethGate * skinRimGate;
    float alpha = clamp(effFeather * uLipColor.a, 0.0f, 0.90f);
    if (alpha <= 0.001f) discard_fragment();
    float smoothY = baseY;
    if (uLipStyle.y > 0.0f && uLipStyle.z > 0.0f) {
        float2 stepUv = uLipStyle.yz * 2.0f;
        float3 avg = 0.25f * (
            tex0.sample(kLinearClamp, toMtlUv(clamp(uv + float2(-stepUv.x, -stepUv.y), 0.0f, 1.0f))).rgb +
            tex0.sample(kLinearClamp, toMtlUv(clamp(uv + float2( stepUv.x, -stepUv.y), 0.0f, 1.0f))).rgb +
            tex0.sample(kLinearClamp, toMtlUv(clamp(uv + float2(-stepUv.x,  stepUv.y), 0.0f, 1.0f))).rgb +
            tex0.sample(kLinearClamp, toMtlUv(clamp(uv + float2( stepUv.x,  stepUv.y), 0.0f, 1.0f))).rgb);
        smoothY = luma601(avg);
    }
    float lipDetail = clamp(baseY - smoothY, -0.09f, 0.09f);
    float midLift = 1.0f + 0.14f * smoothstep(0.14f, 0.26f, smoothY) * (1.0f - smoothstep(0.32f, 0.48f, smoothY));
    float illum = (smoothY / 0.54f) * midLift;
    float illumSoft = illum / (1.0f + 0.52f * max(illum - 0.82f, 0.0f));
    float3 stainTint = clamp(uLipColor.rgb * 1.15f, 0.05f, 1.02f);
    float3 multLip = clamp(base * mix(float3(1.0f), stainTint, 0.92f), 0.0f, 1.0f);
    float isGlossy = step(0.5f, uLipStyle.x);
    float3 pigmentLip = clamp(uLipColor.rgb * illumSoft * mix(1.0f, 0.88f, isGlossy), 0.0f, 1.0f);
    float brightStainBias = smoothstep(0.46f, 0.68f, smoothY) * 0.16f;
    float coverage = clamp(mix(0.48f, 0.76f, clamp(uLipColor.a, 0.0f, 1.0f)) - brightStainBias, 0.32f, 0.76f);
    float3 result = mix(multLip, pigmentLip, coverage);
    result = clamp(result + float3(lipDetail * mix(1.15f, 0.75f, isGlossy)), 0.0f, 1.0f);
    if (isGlossy > 0.5f) {
        float lumaDrive = smoothstep(0.24f, 0.58f, baseY);
        float centerBelly = 1.0f - smoothstep(0.015f, 0.085f, abs(in.makeupUv.x - 0.5f));
        float ridgeMask = clamp(0.85f + lipDetail * 5.0f, 0.20f, 1.40f);
        float dyLo = (in.makeupUv.y - 0.714f) * 145.0f;
        float dyHi = (in.makeupUv.y - 0.673f) * 165.0f;
        float lowerBand = exp(-(dyLo * dyLo));
        float upperBand = exp(-(dyHi * dyHi)) * 0.38f;
        float naturalHot = smoothstep(0.78f, 0.95f, baseY) * 0.24f;
        float spec = clamp(((lowerBand + upperBand) * centerBelly * lumaDrive * 0.78f + naturalHot) * ridgeMask * effFeather, 0.0f, 0.68f);
        float3 specCol = float3(1.0f, 0.99f, 0.97f);
        result = clamp(result + (specCol - result) * (spec * 0.52f), 0.0f, 1.0f);
    }
    return float4(clamp(result, 0.0f, 1.0f), alpha);
}

fragment float4 fs_makeup_iris(MakeupOut in [[stage_in]],
                               texture2d<float> tex0 [[texture(0)]],
                               texture2d<float> tex1 [[texture(1)]],
                               constant Uniforms& uni [[buffer(1)]]) {
    float4 uIris = uni.u[0];
    float4 uIrisTile = uni.u[1];
    float4 uIrisStrength = uni.u[2];
    float blink = smoothstep(0.10f, 0.22f, uIrisTile.z);
    if (blink <= 0.001f) discard_fragment();
    float aspect = max(uIris.w, 0.001f);
    float2 m = float2(in.screenUv.x, in.screenUv.y / aspect);
    float2 c = float2(uIris.x, uIris.y / aspect);
    float2 d = (m - c) / max(uIris.z, 0.0001f);
    if (length(d) >= 1.0f) discard_fragment();
    float3 base = tex0.sample(kLinearClamp, toMtlUv(clamp(in.screenUv, 0.0f, 1.0f))).rgb;
    float baseY = luma601(base);
    float2 localUv = clamp(d * 0.5f + 0.5f, 0.005f, 0.995f);
    float2 atlasUv = float2(uIrisTile.x + localUv.x * 0.5f, uIrisTile.y + (1.0f - localUv.y) * 0.5f);
    float4 lens = tex1.sample(kLinearClamp, atlasUv);
    float lensA = lens.a * uIrisStrength.x * blink;
    float lidShadow = mix(1.0f, 0.65f, smoothstep(0.25f, 0.95f, d.y));
    float3 lensRgb = clamp(lens.rgb * mix(0.45f, 1.15f, clamp(baseY * 2.4f, 0.0f, 1.0f)) * lidShadow, 0.0f, 1.0f);
    float3 outRgb = mix(base, lensRgb, lensA);
    float clA = 0.0f;
    if (uIrisStrength.y > 0.0f) {
        float2 p0 = d - float2(-0.26f, 0.28f);
        float2 p1 = d - float2(0.22f, -0.18f);
        float spot = exp(-dot(p0, p0) * 32.0f) + 0.45f * exp(-dot(p1, p1) * 58.0f);
        clA = clamp(spot * uIrisStrength.y * blink, 0.0f, 0.92f);
        outRgb = mix(outRgb, float3(1.0f, 0.99f, 0.97f), clA);
    }
    float finalA = clamp(max(lensA, clA), 0.0f, 1.0f);
    if (finalA <= 0.001f) discard_fragment();
    return float4(outRgb, finalA);
}

fragment float4 fs_body_warp(QuadOut in [[stage_in]],
                             texture2d<float> tex0 [[texture(0)]],
                             constant Uniforms& uni [[buffer(1)]]) {
    float4 a = uni.u[0];
    float4 b = uni.u[1];
    float4 c = uni.u[2];
    float aspect = max(c.z, 0.001f);
    float axisLen = length(a.zw);
    float2 uv = in.uv;
    if (axisLen >= 0.5f) {
        float2 axis = a.zw / axisLen;
        float legs = axisLen - 1.0f;
        float2 side = float2(-axis.y, axis.x);
        float2 hip = float2(a.x, a.y / aspect);
        float2 d = float2(uv.x, uv.y / aspect) - hip;
        float t = dot(d, axis);
        float s = dot(d, side);
        float torso = b.x;
        float width = b.y;
        if (legs > 0.0f && t > 0.0f) {
            float legKeep = 1.0f - smoothstep(1.15f * width, 1.85f * width, abs(s));
            t = t / (1.0f + legs * legKeep * smoothstep(0.0f, 0.15f * torso, t));
        }
        if (c.y > 0.0f) {
            float u = -(t + torso);
            if (u > 0.0f) {
                float band = 0.3f * torso;
                float lifted = u <= band * (1.0f + c.y) ? u / (1.0f + c.y) : u - band * c.y;
                float keep = 1.0f - smoothstep(1.2f * width, 2.2f * width, abs(s));
                t = -torso - mix(u, lifted, keep);
            }
        }
        if (b.z > 0.0f) {
            float tw = (t + 0.3f * torso) / (0.22f * torso);
            float r = abs(s) / (1.65f * width);
            float f = r < 1.0f ? (1.0f - r * r) * (1.0f - r * r) : 0.0f;
            s *= 1.0f + b.z * exp(-0.5f * tw * tw) * f;
        }
        if (c.x > 0.0f) {
            float ts = (t + 0.95f * torso) / (0.18f * torso);
            float r = abs(s) / (1.60f * width);
            float f = r < 1.0f ? (1.0f - r * r) * (1.0f - r * r) : 0.0f;
            s *= 1.0f + c.x * exp(-0.5f * ts * ts) * f;
        }
        float2 m = hip + axis * t + side * s;
        if (b.w > 0.0f) {
            for (int i = 0; i < 2; ++i) {
                float2 center = hip - axis * (0.65f * torso) + side * ((float(i) * 2.0f - 1.0f) * 0.45f * width);
                float ratio = length(m - center) / max(0.42f * width, 0.001f);
                m = center + (m - center) * clamp(1.0f - (1.0f - ratio * ratio) * b.w, 0.0f, 1.0f);
            }
        }
        uv = float2(m.x, m.y * aspect);
    }
    return tex0.sample(kLinearClamp, toMtlUv(clamp(uv, 0.0f, 1.0f)));
}

inline float3 diag4Sample(texture2d<float> tex0, float2 uv, float2 d) {
    return 0.25f * (tex0.sample(kLinearClamp, toMtlUv(uv + float2(-d.x, -d.y))).rgb +
                    tex0.sample(kLinearClamp, toMtlUv(uv + float2( d.x, -d.y))).rgb +
                    tex0.sample(kLinearClamp, toMtlUv(uv + float2(-d.x,  d.y))).rgb +
                    tex0.sample(kLinearClamp, toMtlUv(uv + float2( d.x,  d.y))).rgb);
}

fragment float4 fs_composite_pro(QuadOut in [[stage_in]],
                                 texture2d<float> tex0 [[texture(0)]],
                                 texture2d<float> tex1 [[texture(1)]],
                                 texture2d<float> tex2 [[texture(2)]],
                                 texture2d<float> tex3 [[texture(3)]],
                                 constant Uniforms& uni [[buffer(1)]]) {
    float4 uFrame = uni.u[0];
    float4 uParams = uni.u[1];
    float4 uDetails = uni.u[2];
    float4 uTone = uni.u[3];
    float4 uEye0 = uni.u[4];
    float4 uEye1 = uni.u[5];
    float4 uOval = uni.u[6];
    float4 uMouth = uni.u[7];
    float4 uEyeMask0 = uni.u[8];
    float4 uEyeMask1 = uni.u[9];
    float4 uGrade = uni.u[10];

    float2 uv = in.uv;
    if (uParams.x < -0.5f) return tex0.sample(kLinearClamp, toMtlUv(uv));
    float4 original = tex0.sample(kLinearClamp, toMtlUv(uv));
    float3 guidedB = original.rgb;
    float guidedA = 0.0f;
    if (uParams.y > 0.0f) {
        float4 coeff = tex1.sample(kLinearClamp, toMtlUv(uv));
        guidedB = coeff.rgb;
        guidedA = coeff.a;
    }
    float3 fineLow = original.rgb;
    float3 wideLow = original.rgb;
    if (uParams.y > 0.0f || uDetails.y > 0.0f) {
        fineLow = diag4Sample(tex0, uv, uFrame.xy * 0.5f);
        wideLow = diag4Sample(tex0, uv, uFrame.xy * 1.5f);
    }
    float4 sem = tex2.sample(kLinearClamp, toMtlUv(uv));
    float nonHair = clamp(sem.a * 1.04f - 0.02f, 0.0f, 1.0f);
    float faceM = 0.0f, exclM = 0.0f, hiM = 0.0f;
    if (uFrame.w >= 3.5f) {
        float pres = clamp(uFrame.w - 4.0f, 0.0f, 1.0f);
        float3 poly = sem.rgb * pres;
        faceM = poly.r; exclM = poly.g; hiM = poly.b;
    } else if (uFrame.w > 0.0f) {
        float2 em = faceMasks(uv, uFrame.z, uFrame.w, uOval, float2(uEye0.w, uEye1.w), uMouth, uEyeMask0, uEyeMask1);
        faceM = em.x; exclM = em.y; hiM = em.x * 0.3f;
    }
    float skinOnly = step(15.5f, uParams.x);
    float style = uParams.x - skinOnly * 16.0f;
    float gate = skinWeight(fineLow);
    float chromaGate = smoothstep(0.045f, 0.10f, fineLow.r - fineLow.b);
    float polyGate = smoothstep(0.06f, 0.18f, luma709(fineLow)) * mix(0.30f, 1.0f, smoothstep(0.0f, 0.25f, gate));
    float skin = clamp(faceM * polyGate + (1.0f - faceM) * gate * chromaGate * (1.0f - 0.70f * exclM), 0.0f, 1.0f) * nonHair;
    float3 guided = clamp(original.rgb * guidedA + guidedB, 0.0f, 1.0f);
    float bleedGuard = 1.0f - (1.0f - smoothstep(0.22f, 0.31f, luma709(fineLow))) *
        max(smoothstep(0.020f, 0.055f, luma709(guided) - luma709(fineLow)),
            smoothstep(0.24f, 0.48f, guidedA) * smoothstep(0.003f, 0.018f, luma709(guided) - luma709(fineLow)));
    float fineDark = (1.0f - smoothstep(0.02f, 0.06f, luma709(fineLow) - luma709(original.rgb)) * smoothstep(0.25f, 0.55f, guidedA)) * bleedGuard;
    float featureKeep = clamp(1.0f - exclM, 0.0f, 1.0f);
    float core = max(uGrade.y, 0.005f);
    float band = luma709(fineLow - wideLow);
    float3 color = original.rgb;
    if (uParams.y > 0.0f) {
        float rEdge = length(original.rgb - guided) / 0.36f;
        float rEdge2 = rEdge * rEdge;
        float k = uParams.y * 0.96f * (skinOnly > 0.5f ? skin : 1.0f) * featureKeep * fineDark / (1.0f + rEdge2 * rEdge2);
        float tex = clamp(band, -0.035f, 0.035f) * smoothstep(core, 2.5f * core, abs(band));
        color = mix(original.rgb, guided, k) + float3(tex * mix(0.35f, 0.85f, uParams.w) * k);
    }
    if (uDetails.y > 0.0f) {
        float edge = smoothstep(1.5f * core, 4.0f * core, abs(band));
        float region = mix(0.35f, 1.0f, exclM) * (1.0f - 0.85f * skin * featureKeep * faceM);
        color += float3(clamp(band, -0.06f, 0.06f) * edge * uDetails.y * 1.6f * region);
    }
    color = clamp(color, 0.0f, 1.0f);
    float w = skin * featureKeep * fineDark;
    float liftA = (0.60f * uTone.x + 0.40f * uParams.z) * w;
    float rosyA = 0.25f * uTone.y * w;
    float sculpt = 0.30f * uDetails.x * (hiM - 0.3f * faceM) * w * faceM;
    if (liftA > 0.0f || rosyA > 0.0f || sculpt != 0.0f) {
        float y0 = luma709(color);
        if (uTone.x > 0.0f) {
            float chroma = max(color.r - color.b, 0.02f);
            float excess = smoothstep(0.40f, 0.60f, (color.g - color.b) / chroma);
            float target = color.b + 0.38f * chroma;
            color.g -= min(max(color.g - target, 0.0f) * excess * 0.6f * uTone.x, 0.02f) * w;
        }
        float y1 = max(luma709(color), 0.0001f);
        float a = liftA * smoothstep(0.12f, 0.28f, y0) * mix(0.4f, 1.0f, smoothstep(0.15f, 0.5f, y0));
        float y2 = y0 * (1.0f + a) / (1.0f + a * y0);
        if (sculpt != 0.0f) {
            float s = 0.5f + sculpt;
            y2 = (1.0f - 2.0f * s) * y2 * y2 + 2.0f * s * y2;
        }
        color *= y2 / y1;
        if (rosyA > 0.0f) color = float3(y2) + (color - float3(y2)) * (1.0f + rosyA * 4.0f * y2 * (1.0f - y2));
        color = fitGamut(color, y2);
    }
    color = mix(float3(luma709(color)), color, uDetails.z);
    color = mix(color, clamp(gradePro(color, style), 0.0f, 1.0f), uDetails.w);
    color = clamp(color, 0.0f, 1.0f);
    if (uGrade.x > 0.0f) {
        float level = color.b * 63.0f;
        float3 low = tex3.sample(kLinearClamp, lutCell(floor(level), color.rg)).rgb;
        float3 high = tex3.sample(kLinearClamp, lutCell(ceil(level), color.rg)).rgb;
        color = mix(color, mix(low, high, level - floor(level)), uGrade.x);
    }
    return float4(color, original.a);
}

fragment float4 fs_temporal_denoise(QuadOut in [[stage_in]],
                                    texture2d<float> tex0 [[texture(0)]],
                                    texture2d<float> tex1 [[texture(1)]],
                                    constant Uniforms& uni [[buffer(1)]]) {
    float2 muv = toMtlUv(in.uv);
    float4 current = tex0.sample(kLinearClamp, muv);
    float3 previous = tex1.sample(kLinearClamp, muv).rgb;
    float3 delta = abs(current.rgb - previous);
    float diff = max(max(delta.r, delta.g), delta.b);
    float weight = uni.u[0].x * (1.0f - smoothstep(uni.u[0].y, uni.u[0].z, diff));
    return float4(mix(current.rgb, previous, weight), current.a);
}

fragment float4 fs_bg_premask(QuadOut in [[stage_in]],
                              texture2d<float> tex0 [[texture(0)]],
                              texture2d<float> tex1 [[texture(1)]],
                              constant Uniforms& uni [[buffer(1)]]) {
    float2 muv = toMtlUv(in.uv);
    float3 color = tex0.sample(kLinearClamp, muv).rgb;
    float person = tex1.sample(kLinearClamp, muv).r;
    float bg = 1.0f - smoothstep(uni.u[0].y, uni.u[0].z, person);
    return float4(color * bg, bg);
}

fragment float4 fs_bg_blur(QuadOut in [[stage_in]],
                           texture2d<float> tex0 [[texture(0)]],
                           constant Uniforms& uni [[buffer(1)]]) {
    float2 step = uni.u[0].xy;
    float4 sum = tex0.sample(kLinearClamp, toMtlUv(in.uv)) * 0.2270270f;
    sum += (tex0.sample(kLinearClamp, toMtlUv(in.uv + step * 1.3846154f)) +
            tex0.sample(kLinearClamp, toMtlUv(in.uv - step * 1.3846154f))) * 0.3162162f;
    sum += (tex0.sample(kLinearClamp, toMtlUv(in.uv + step * 3.2307692f)) +
            tex0.sample(kLinearClamp, toMtlUv(in.uv - step * 3.2307692f))) * 0.0702703f;
    return sum;
}

fragment float4 fs_bg_mix(QuadOut in [[stage_in]],
                          texture2d<float> tex0 [[texture(0)]],
                          texture2d<float> tex1 [[texture(1)]],
                          texture2d<float> tex2 [[texture(2)]],
                          constant Uniforms& uni [[buffer(1)]]) {
    float2 muv = toMtlUv(in.uv);
    float4 color = tex0.sample(kLinearClamp, muv);
    float person = tex2.sample(kLinearClamp, muv).r;
    float4 blurSample = tex1.sample(kLinearClamp, muv);
    float3 blurred = blurSample.a > 0.02f ? clamp(blurSample.rgb / blurSample.a, 0.0f, 1.0f) : color.rgb;
    float head = 1.0f - smoothstep(2.0f, 2.8f, sqrt(ellipseE(in.uv, uni.u[1], uni.u[2].xy, max(uni.u[2].z, 0.0001f))));
    float fg = smoothstep(0.15f, 0.30f, abs(luma709(color.rgb - blurred))) * smoothstep(0.02f, 0.10f, person) * head;
    float keep = max(smoothstep(uni.u[0].y, uni.u[0].z, person), uni.u[0].w * fg);
    return float4(mix(color.rgb, mix(blurred, color.rgb, keep), uni.u[0].x), color.a);
}
)MSL";

}  // namespace livebeauty::rhi::metal
