#include "gles_shaders.h"

namespace livebeauty::rhi::gles {

namespace {

constexpr const char* kQuadVert = R"(
attribute vec2 aPosition;
attribute vec2 aTextureCoord;
varying vec2 vTextureCoord;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTextureCoord = aTextureCoord;
}
)";

constexpr const char* kDownVert = R"(
attribute vec2 aPosition;
attribute vec2 aTextureCoord;
uniform vec4 uU0;
varying vec2 vTap0;
varying vec2 vTap1;
varying vec2 vTap2;
varying vec2 vTap3;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTap0 = aTextureCoord + vec2(-uU0.x, -uU0.y);
    vTap1 = aTextureCoord + vec2(uU0.x, -uU0.y);
    vTap2 = aTextureCoord + vec2(-uU0.x, uU0.y);
    vTap3 = aTextureCoord + vec2(uU0.x, uU0.y);
}
)";

constexpr const char* kBlurVert = R"(
attribute vec2 aPosition;
attribute vec2 aTextureCoord;
uniform vec4 uU0;
varying vec2 vC;
varying vec2 vA1;
varying vec2 vB1;
varying vec2 vA2;
varying vec2 vB2;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vC = aTextureCoord;
    vA1 = aTextureCoord + uU0.xy * 1.4072;
    vB1 = aTextureCoord - uU0.xy * 1.4072;
    vA2 = aTextureCoord + uU0.xy * 3.2939;
    vB2 = aTextureCoord - uU0.xy * 3.2939;
}
)";

constexpr const char* kSkinMaskVert = R"(
attribute vec2 aPosition;
attribute vec3 aMaskWeights;
varying vec3 vMaskWeights;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vMaskWeights = aMaskWeights;
}
)";

constexpr const char* kRetouchVert = R"(
attribute vec2 aPosition;
attribute vec2 aSrcCoord;
attribute vec4 aRetouchAttr;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vSrcCoord = aSrcCoord;
    vRetouchAttr = aRetouchAttr;
}
)";

constexpr const char* kMakeupVert = R"(
attribute vec2 aPosition;
attribute vec2 aScreenUv;
attribute vec2 aMakeupUv;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vScreenUv = aScreenUv;
    vMakeupUv = aMakeupUv;
}
)";

#define LB_GLSL_PREC \
    "#ifdef GL_ES\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\n#endif\n"

#define LB_GLSL_BLUR_COMMON \
    LB_GLSL_PREC \
    "uniform sampler2D uTex0;\n" \
    "varying vec2 vC; varying vec2 vA1; varying vec2 vB1; varying vec2 vA2; varying vec2 vB2;\n" \
    "vec4 blur5() {\n" \
    "  return texture2D(uTex0, vC) * 0.2042 + (texture2D(uTex0, vA1) + texture2D(uTex0, vB1)) * 0.3040 + (texture2D(uTex0, vA2) + texture2D(uTex0, vB2)) * 0.0939;\n" \
    "}\n"

constexpr const char* kPassThroughFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
varying vec2 vTextureCoord;
void main() { gl_FragColor = texture2D(uTex0, clamp(vTextureCoord, 0.0, 1.0)); }
)";

constexpr const char* kDownFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
varying vec2 vTap0; varying vec2 vTap1; varying vec2 vTap2; varying vec2 vTap3;
void main() {
    gl_FragColor = (texture2D(uTex0, vTap0) + texture2D(uTex0, vTap1) +
                    texture2D(uTex0, vTap2) + texture2D(uTex0, vTap3)) * 0.25;
}
)";

constexpr const char* kBlurRgbFrag = LB_GLSL_BLUR_COMMON R"(
void main() { gl_FragColor = blur5(); }
)";

constexpr const char* kBlurAlphaFrag = LB_GLSL_BLUR_COMMON R"(
void main() { gl_FragColor = vec4(texture2D(uTex0, vC).rgb, blur5().a); }
)";

constexpr const char* kGuidedDiffFrag = LB_GLSL_BLUR_COMMON R"(
uniform sampler2D uTex1;
void main() {
    vec3 mean = blur5().rgb;
    vec3 diff = texture2D(uTex1, vC).rgb - mean;
    gl_FragColor = vec4(mean, clamp(length(diff) * 2.5, 0.0, 1.0));
}
)";

constexpr const char* kGuidedCoeffFrag = LB_GLSL_BLUR_COMMON R"(
uniform sampler2D uTex1;
uniform vec4 uU0;
float guidedCoeff(float s, float eps) {
    float e = max(eps, 0.005);
    float r = s * s / e;
    float rp = r * sqrt(r);
    float lo = clamp(sqrt(e) - 0.09, 0.12, 0.18);
    return max(rp / (1.0 + rp), smoothstep(lo, 0.30, s));
}
vec3 evenMean(vec3 fineMean, float s, float evenness) {
    if (evenness <= 0.0) return fineMean;
    vec3 macroMean = texture2D(uTex1, vC).rgb;
    float dy = abs(dot(macroMean - fineMean, vec3(0.2126, 0.7152, 0.0722))) / 0.068;
    float dy2 = dy * dy;
    float boneKeep = 1.0 / (1.0 + dy2 * dy2);
    float flatKeep = 1.0 - smoothstep(0.05, 0.14, s);
    return mix(fineMean, macroMean, evenness * boneKeep * flatKeep);
}
void main() {
    float s = blur5().a;
    vec3 mean = evenMean(texture2D(uTex0, vC).rgb, s, uU0.w);
    float a = guidedCoeff(s, uU0.z);
    gl_FragColor = vec4((1.0 - a) * mean, a);
}
)";

constexpr const char* kGuidedWeightBlurFrag = LB_GLSL_BLUR_COMMON R"(
uniform sampler2D uTex1;
vec4 weightedSample(vec2 uv) {
    float w = clamp(texture2D(uTex1, uv).a * 1.04 - 0.02, 0.0, 1.0);
    return vec4(texture2D(uTex0, uv).rgb * w, w);
}
void main() {
    gl_FragColor = weightedSample(vC) * 0.2042 +
                   (weightedSample(vA1) + weightedSample(vB1)) * 0.3040 +
                   (weightedSample(vA2) + weightedSample(vB2)) * 0.0939;
}
)";

constexpr const char* kGuidedWeightDiffFrag = LB_GLSL_BLUR_COMMON R"(
uniform sampler2D uTex1;
void main() {
    vec4 sum = blur5();
    vec3 mean = clamp(sum.rgb / max(sum.a, 0.02), 0.0, 1.0);
    vec3 diff = texture2D(uTex1, vC).rgb - mean;
    gl_FragColor = vec4(mean, clamp(length(diff) * 2.5, 0.0, 1.0));
}
)";

constexpr const char* kGuidedWeightCoeffFrag = LB_GLSL_BLUR_COMMON R"(
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
float guidedCoeff(float s, float eps) {
    float e = max(eps, 0.005);
    float r = s * s / e;
    float rp = r * sqrt(r);
    float lo = clamp(sqrt(e) - 0.09, 0.12, 0.18);
    return max(rp / (1.0 + rp), smoothstep(lo, 0.30, s));
}
vec3 evenMean(vec3 fineMean, float s, float evenness, float maskW) {
    if (evenness <= 0.0) return fineMean;
    vec3 macroMean = texture2D(uTex1, vC).rgb;
    float dy = abs(dot(macroMean - fineMean, vec3(0.2126, 0.7152, 0.0722))) / 0.068;
    float dy2 = dy * dy;
    float boneKeep = 1.0 / (1.0 + dy2 * dy2);
    float flatKeep = 1.0 - smoothstep(0.05, 0.14, s);
    return mix(fineMean, macroMean, evenness * boneKeep * flatKeep * maskW);
}
void main() {
    float s = blur5().a;
    float w = clamp(texture2D(uTex2, vC).a * 1.04 - 0.02, 0.0, 1.0);
    vec3 mean = evenMean(texture2D(uTex0, vC).rgb, s, uU0.w, w);
    float a = mix(1.0, guidedCoeff(s, uU0.z), w);
    gl_FragColor = vec4((1.0 - a) * mean, a);
}
)";

constexpr const char* kSkinMaskFrag = LB_GLSL_PREC R"(
varying vec3 vMaskWeights;
void main() { gl_FragColor = vec4(clamp(vMaskWeights, 0.0, 1.0), 1.0); }
)";

constexpr const char* kRetouchSkinFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform vec4 uU0;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    float wTear = clamp(vRetouchAttr.x * uU0.x * 0.85, 0.0, 1.0);
    float wFold = clamp(vRetouchAttr.y * uU0.y * 0.82, 0.0, 1.0);
    float alpha = clamp(wTear + wFold, 0.0, 1.0);
    if (alpha <= 0.002) { discard; }
    vec3 c = texture2D(uTex0, vSrcCoord).rgb;
    vec2 off = vRetouchAttr.zw;
    vec3 target = c;
    if (wTear > wFold) {
        vec3 cheekRef = texture2D(uTex0, clamp(vSrcCoord + off, 0.0, 1.0)).rgb;
        float yBase = dot(c, vec3(0.2126, 0.7152, 0.0722));
        float yRef = dot(cheekRef, vec3(0.2126, 0.7152, 0.0722));
        float deficit = clamp((yRef - yBase) * 2.4, 0.0, 0.22);
        float liftMask = clamp(deficit * 3.8, 0.0, 0.75);
        target = clamp(c + (cheekRef - c) * liftMask + vec3(0.058, 0.046, 0.032) * liftMask * (1.0 - yBase), 0.0, 1.0);
    } else {
        vec3 sideA = texture2D(uTex0, clamp(vSrcCoord + off, 0.0, 1.0)).rgb;
        vec3 sideB = texture2D(uTex0, clamp(vSrcCoord - off, 0.0, 1.0)).rgb;
        target = max(c, 0.5 * (sideA + sideB));
    }
    gl_FragColor = vec4(target, alpha);
}
)";

constexpr const char* kRetouchEyeTeethFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform vec4 uU0;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    float strength = clamp(uU0.x, 0.0, 1.0);
    if (strength <= 0.002) { discard; }
    vec3 c = texture2D(uTex0, vSrcCoord).rgb;
    float y = dot(c, vec3(0.299, 0.587, 0.114));
    float chroma = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    if (uU0.y < 0.5) {
        float sclera = smoothstep(0.38, 0.62, y) * (1.0 - smoothstep(0.12, 0.28, chroma));
        vec3 contrasted = clamp((c - 0.42) * 1.16 + 0.45, 0.0, 1.0);
        vec3 cleanSclera = mix(contrasted, vec3(min(y * 1.07 + 0.025, 1.0)), sclera * 0.55);
        gl_FragColor = vec4(cleanSclera, clamp(strength * 0.72, 0.0, 1.0));
    } else {
        float toothMask = smoothstep(0.28, 0.50, y) * (1.0 - smoothstep(0.20, 0.38, chroma));
        float alpha = clamp(strength * toothMask * 0.84, 0.0, 1.0);
        if (alpha <= 0.002) { discard; }
        float targetY = clamp(y * 1.12 + 0.035, 0.0, 0.96);
        vec3 neutralWhite = vec3(targetY, targetY, min(targetY * 1.015, 0.98));
        vec3 whitened = mix(c * (targetY / max(y, 0.001)), neutralWhite, 0.68);
        gl_FragColor = vec4(clamp(whitened, 0.0, 1.0), alpha);
    }
}
)";

constexpr const char* kMakeupLayerFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
vec3 softLightBlend(vec3 b, vec3 s) { return (1.0 - 2.0 * s) * b * b + 2.0 * s * b; }
void main() {
    vec2 atlasUv = vMakeupUv;
    if (uU2.y > 0.5) {
        float isLeft = step(0.5, vMakeupUv.x);
        float eu = isLeft < 0.5 ? (vMakeupUv.x - 0.18) / 0.30 : (vMakeupUv.x - 0.52) / 0.30;
        float ev = (vMakeupUv.y - 0.21) / 0.24;
        if (eu <= 0.002 || eu >= 0.998 || ev <= 0.002 || ev >= 0.998) { discard; }
        atlasUv = vec2(eu, isLeft * 0.5 + ev * 0.5);
    }
    float mask = dot(texture2D(uTex1, atlasUv), uU1);
    float alpha = clamp(mask * uU0.a, 0.0, 1.0);
    if (alpha <= 0.001) { discard; }
    vec3 base = texture2D(uTex0, clamp(vScreenUv, 0.0, 1.0)).rgb;
    if (uU2.z > 0.5) {
        float baseY = dot(base, vec3(0.299, 0.587, 0.114));
        float darkGate = smoothstep(0.14, 0.30, baseY);
        float skinGate = smoothstep(0.025, 0.095, base.r - base.b) * smoothstep(0.015, 0.065, base.r - base.g);
        alpha *= darkGate * skinGate;
        if (alpha <= 0.001) { discard; }
    }
    vec3 result;
    if (uU2.x < 0.5) {
        result = mix(softLightBlend(base, uU0.rgb), base * mix(vec3(1.0), uU0.rgb, 0.68), 0.45);
    } else if (uU2.x < 1.5) {
        vec3 mult = base * mix(vec3(1.0), uU0.rgb, uU2.z > 0.5 ? 0.78 : 0.92);
        result = uU2.z > 0.5 ? mix(mult, softLightBlend(base, uU0.rgb), 0.45) : mult;
    } else {
        result = 1.0 - (1.0 - base) * (1.0 - uU0.rgb * 0.58);
    }
    gl_FragColor = vec4(clamp(result, 0.0, 1.0), alpha);
}
)";

constexpr const char* kMakeupLipFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
void main() {
    float feather = texture2D(uTex1, vMakeupUv).b;
    if (feather * uU0.a <= 0.001) { discard; }
    vec2 uv = clamp(vScreenUv, 0.0, 1.0);
    vec3 base = texture2D(uTex0, uv).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    float darkGate = smoothstep(0.06, 0.20, baseY);
    float redExcess = base.r - max(base.g, base.b);
    float teethLike = smoothstep(0.52, 0.74, baseY) * (1.0 - smoothstep(0.035, 0.115, redExcess));
    float teethGate = 1.0 - teethLike;
    float lipHue = (base.r - base.g) - 0.75 * max(base.g - base.b, 0.0);
    float rimZone = 1.0 - smoothstep(0.35, 0.92, feather);
    float skinRimGate = mix(1.0, smoothstep(0.045, 0.155, lipHue), rimZone * 0.92);
    float effFeather = feather * darkGate * teethGate * skinRimGate;
    float alpha = clamp(effFeather * uU0.a, 0.0, 0.90);
    if (alpha <= 0.001) { discard; }
    float smoothY = baseY;
    if (uU1.y > 0.0 && uU1.z > 0.0) {
        vec2 stepUv = uU1.yz * 2.0;
        vec3 avg = 0.25 * (
            texture2D(uTex0, clamp(uv + vec2(-stepUv.x, -stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(uv + vec2(stepUv.x, -stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(uv + vec2(-stepUv.x, stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(uv + vec2(stepUv.x, stepUv.y), 0.0, 1.0)).rgb
        );
        smoothY = dot(avg, vec3(0.299, 0.587, 0.114));
    }
    float lipDetail = clamp(baseY - smoothY, -0.09, 0.09);
    float midLift = 1.0 + 0.14 * smoothstep(0.14, 0.26, smoothY) * (1.0 - smoothstep(0.32, 0.48, smoothY));
    float illum = (smoothY / 0.54) * midLift;
    float illumSoft = illum / (1.0 + 0.52 * max(illum - 0.82, 0.0));
    vec3 stainTint = clamp(uU0.rgb * 1.15, 0.05, 1.02);
    vec3 multLip = clamp(base * mix(vec3(1.0), stainTint, 0.92), 0.0, 1.0);
    float isGlossy = step(0.5, uU1.x);
    vec3 pigmentLip = clamp(uU0.rgb * illumSoft * mix(1.0, 0.88, isGlossy), 0.0, 1.0);
    float brightStainBias = smoothstep(0.46, 0.68, smoothY) * 0.16;
    float coverage = clamp(mix(0.48, 0.76, clamp(uU0.a, 0.0, 1.0)) - brightStainBias, 0.32, 0.76);
    vec3 result = mix(multLip, pigmentLip, coverage);
    result = clamp(result + vec3(lipDetail * mix(1.15, 0.75, isGlossy)), 0.0, 1.0);
    if (isGlossy > 0.5) {
        float lumaDrive = smoothstep(0.24, 0.58, baseY);
        float centerBelly = 1.0 - smoothstep(0.015, 0.085, abs(vMakeupUv.x - 0.5));
        float ridgeMask = clamp(0.85 + lipDetail * 5.0, 0.20, 1.40);
        float dyLo = (vMakeupUv.y - 0.714) * 145.0;
        float dyHi = (vMakeupUv.y - 0.673) * 165.0;
        float lowerBand = exp(-(dyLo * dyLo));
        float upperBand = exp(-(dyHi * dyHi)) * 0.38;
        float naturalHot = smoothstep(0.78, 0.95, baseY) * 0.24;
        float spec = clamp(((lowerBand + upperBand) * centerBelly * lumaDrive * 0.78 + naturalHot) * ridgeMask * effFeather, 0.0, 0.68);
        vec3 specCol = vec3(1.0, 0.99, 0.97);
        result = clamp(result + (specCol - result) * (spec * 0.52), 0.0, 1.0);
    }
    gl_FragColor = vec4(clamp(result, 0.0, 1.0), alpha);
}
)";

constexpr const char* kMakeupIrisFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
void main() {
    float blink = smoothstep(0.10, 0.22, uU1.z);
    if (blink <= 0.001) { discard; }
    float aspect = max(uU0.w, 0.001);
    vec2 m = vec2(vScreenUv.x, vScreenUv.y / aspect);
    vec2 c = vec2(uU0.x, uU0.y / aspect);
    vec2 d = (m - c) / max(uU0.z, 0.0001);
    if (length(d) >= 1.0) { discard; }
    vec3 base = texture2D(uTex0, clamp(vScreenUv, 0.0, 1.0)).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    vec2 localUv = clamp(d * 0.5 + 0.5, 0.005, 0.995);
    vec2 atlasUv = vec2(uU1.x + localUv.x * 0.5, uU1.y + (1.0 - localUv.y) * 0.5);
    vec4 lens = texture2D(uTex1, atlasUv);
    float lensA = lens.a * uU2.x * blink;
    float lidShadow = mix(1.0, 0.65, smoothstep(0.25, 0.95, d.y));
    vec3 lensRgb = clamp(lens.rgb * mix(0.45, 1.15, clamp(baseY * 2.4, 0.0, 1.0)) * lidShadow, 0.0, 1.0);
    vec3 outRgb = mix(base, lensRgb, lensA);
    float clA = 0.0;
    if (uU2.y > 0.0) {
        vec2 p0 = d - vec2(-0.26, 0.28);
        vec2 p1 = d - vec2(0.22, -0.18);
        float spot = exp(-dot(p0, p0) * 32.0) + 0.45 * exp(-dot(p1, p1) * 58.0);
        clA = clamp(spot * uU2.y * blink, 0.0, 0.92);
        outRgb = mix(outRgb, vec3(1.0, 0.99, 0.97), clA);
    }
    float finalA = clamp(max(lensA, clA), 0.0, 1.0);
    if (finalA <= 0.001) { discard; }
    gl_FragColor = vec4(outRgb, finalA);
}
)";

constexpr const char* kBodyWarpFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
varying vec2 vTextureCoord;
void main() {
    float aspect = max(uU2.z, 0.001);
    float axisLen = length(uU0.zw);
    vec2 uv = vTextureCoord;
    if (axisLen >= 0.5) {
        vec2 axis = uU0.zw / axisLen;
        float legs = axisLen - 1.0;
        vec2 side = vec2(-axis.y, axis.x);
        vec2 hip = vec2(uU0.x, uU0.y / aspect);
        vec2 d = vec2(uv.x, uv.y / aspect) - hip;
        float t = dot(d, axis);
        float s = dot(d, side);
        float torso = uU1.x;
        float width = uU1.y;
        if (legs > 0.0 && t > 0.0) {
            float legKeep = 1.0 - smoothstep(1.15 * width, 1.85 * width, abs(s));
            t = t / (1.0 + legs * legKeep * smoothstep(0.0, 0.15 * torso, t));
        }
        if (uU2.y > 0.0) {
            float u = -(t + torso);
            if (u > 0.0) {
                float band = 0.3 * torso;
                float lifted = u <= band * (1.0 + uU2.y) ? u / (1.0 + uU2.y) : u - band * uU2.y;
                float keep = 1.0 - smoothstep(1.2 * width, 2.2 * width, abs(s));
                t = -torso - mix(u, lifted, keep);
            }
        }
        if (uU1.z > 0.0) {
            float tw = (t + 0.3 * torso) / (0.22 * torso);
            float r = abs(s) / (1.65 * width);
            float f = r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
            s *= 1.0 + uU1.z * exp(-0.5 * tw * tw) * f;
        }
        if (uU2.x > 0.0) {
            float ts = (t + 0.95 * torso) / (0.18 * torso);
            float r = abs(s) / (1.60 * width);
            float f = r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
            s *= 1.0 + uU2.x * exp(-0.5 * ts * ts) * f;
        }
        vec2 m = hip + axis * t + side * s;
        if (uU1.w > 0.0) {
            for (int i = 0; i < 2; i++) {
                vec2 c = hip - axis * (0.65 * torso) + side * ((float(i) * 2.0 - 1.0) * 0.45 * width);
                float ratio = length(m - c) / max(0.42 * width, 0.001);
                m = c + (m - c) * clamp(1.0 - (1.0 - ratio * ratio) * uU1.w, 0.0, 1.0);
            }
        }
        uv = vec2(m.x, m.y * aspect);
    }
    gl_FragColor = texture2D(uTex0, clamp(uv, 0.0, 1.0));
}
)";

constexpr const char* kCompositeProFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform sampler2D uTex3;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
uniform vec4 uU4;
uniform vec4 uU5;
uniform vec4 uU6;
uniform vec4 uU7;
uniform vec4 uU8;
uniform vec4 uU9;
uniform vec4 uU10;
varying vec2 vTextureCoord;
float luma(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }
float skinWeight(vec3 color) {
    float cb = 0.5 + dot(color, vec3(-0.168736, -0.331264, 0.5));
    float cr = 0.5 + dot(color, vec3(0.5, -0.418688, -0.081312));
    return smoothstep(0.26, 0.34, cb) * (1.0 - smoothstep(0.48, 0.56, cb)) *
           smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.72, 0.80, cr));
}
float ellipseE(vec2 uv, vec4 c, vec2 u, float aspect) {
    vec2 d = vec2(uv.x - c.x, (uv.y - c.y) / aspect);
    float a = clamp(dot(d, u) / max(c.z, 0.0001), -4.0, 4.0);
    float b = clamp(dot(d, vec2(-u.y, u.x)) / max(c.w, 0.0001), -4.0, 4.0);
    return a * a + b * b;
}
vec2 faceMasks(vec2 uv, float aspect, float presence, vec4 oval, vec2 axis, vec4 mouth, vec4 eye0, vec4 eye1) {
    if (presence <= 0.0) return vec2(0.0);
    float inside = 1.0 - smoothstep(0.50, 1.0, ellipseE(uv, oval, axis, aspect));
    float e0 = 1.0 - smoothstep(0.55, 1.0, ellipseE(uv, eye0, axis, aspect));
    float e1 = 1.0 - smoothstep(0.55, 1.0, ellipseE(uv, eye1, axis, aspect));
    float lips = 1.0 - smoothstep(0.65, 1.0, ellipseE(uv, mouth, axis, aspect));
    float keep = (1.0 - e0) * (1.0 - e1) * (1.0 - lips);
    return presence * inside * vec2(keep, 1.0 - keep);
}
vec3 gradePro(vec3 color, float style) {
    float y = luma(color);
    float bell = 4.0 * y * (1.0 - y);
    vec3 graded = color;
    if (style > 4.5) graded += vec3(0.045, -0.005, 0.025) * bell;
    else if (style > 3.5) graded = vec3(dot(color, vec3(0.393, 0.769, 0.189)), dot(color, vec3(0.349, 0.686, 0.168)), dot(color, vec3(0.272, 0.534, 0.131)));
    else if (style > 2.5) graded = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    else if (style > 1.5) graded += vec3(-0.025, 0.0, 0.045) * bell;
    else if (style > 0.5) graded += vec3(0.045, 0.008, -0.025) * bell;
    return graded;
}
vec3 fitGamut(vec3 c, float y) {
    float hi = max(max(c.r, c.g), c.b);
    float lo = min(min(c.r, c.g), c.b);
    float s = 1.0;
    if (hi > 1.0) s = min(s, (1.0 - y) / max(hi - y, 0.0001));
    if (lo < 0.0) s = min(s, y / max(y - lo, 0.0001));
    return vec3(y) + (c - vec3(y)) * s;
}
vec2 lutCell(float level, vec2 rg) {
    float row = floor(level / 8.0);
    float column = level - row * 8.0;
    return vec2(column, row) * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * rg;
}
vec3 diag4(vec2 uv, vec2 d) {
    return 0.25 * (texture2D(uTex0, uv + vec2(-d.x, -d.y)).rgb +
                   texture2D(uTex0, uv + vec2(d.x, -d.y)).rgb +
                   texture2D(uTex0, uv + vec2(-d.x, d.y)).rgb +
                   texture2D(uTex0, uv + vec2(d.x, d.y)).rgb);
}
void main() {
    if (uU1.x < -0.5) { gl_FragColor = texture2D(uTex0, vTextureCoord); return; }
    vec2 uv = vTextureCoord;
    vec4 original = texture2D(uTex0, uv);
    vec3 guidedB = original.rgb;
    float guidedA = 0.0;
    if (uU1.y > 0.0) {
        vec4 coeff = texture2D(uTex1, uv);
        guidedB = coeff.rgb;
        guidedA = coeff.a;
    }
    vec3 fineLow = original.rgb;
    vec3 wideLow = original.rgb;
    if (uU1.y > 0.0 || uU2.y > 0.0) {
        fineLow = diag4(uv, uU0.xy * 0.5);
        wideLow = diag4(uv, uU0.xy * 1.5);
    }
    vec4 sem = texture2D(uTex2, uv);
    float nonHair = clamp(sem.a * 1.04 - 0.02, 0.0, 1.0);
    float faceM = 0.0; float exclM = 0.0; float hiM = 0.0;
    if (uU0.w >= 3.5) {
        float pres = clamp(uU0.w - 4.0, 0.0, 1.0);
        vec3 poly = sem.rgb * pres;
        faceM = poly.r; exclM = poly.g; hiM = poly.b;
    } else if (uU0.w > 0.0) {
        vec2 em = faceMasks(uv, uU0.z, uU0.w, uU6, vec2(uU4.w, uU5.w), uU7, uU8, uU9);
        faceM = em.x; exclM = em.y; hiM = em.x * 0.3;
    }
    float skinOnly = step(15.5, uU1.x);
    float style = uU1.x - skinOnly * 16.0;
    float gate = skinWeight(fineLow);
    float chromaGate = smoothstep(0.045, 0.10, fineLow.r - fineLow.b);
    float polyGate = smoothstep(0.06, 0.18, luma(fineLow)) * mix(0.30, 1.0, smoothstep(0.0, 0.25, gate));
    float skin = clamp(faceM * polyGate + (1.0 - faceM) * gate * chromaGate * (1.0 - 0.70 * exclM), 0.0, 1.0) * nonHair;
    vec3 guided = clamp(original.rgb * guidedA + guidedB, 0.0, 1.0);
    float bleedGuard = 1.0 - (1.0 - smoothstep(0.22, 0.31, luma(fineLow))) *
        max(smoothstep(0.020, 0.055, luma(guided) - luma(fineLow)),
            smoothstep(0.24, 0.48, guidedA) * smoothstep(0.003, 0.018, luma(guided) - luma(fineLow)));
    float fineDark = (1.0 - smoothstep(0.02, 0.06, luma(fineLow) - luma(original.rgb)) * smoothstep(0.25, 0.55, guidedA)) * bleedGuard;
    float featureKeep = clamp(1.0 - exclM, 0.0, 1.0);
    float core = max(uU10.y, 0.005);
    float band = luma(fineLow - wideLow);
    vec3 color = original.rgb;
    if (uU1.y > 0.0) {
        float rEdge = length(original.rgb - guided) / 0.36;
        float rEdge2 = rEdge * rEdge;
        float k = uU1.y * 0.96 * (skinOnly > 0.5 ? skin : 1.0) * featureKeep * fineDark / (1.0 + rEdge2 * rEdge2);
        float tex = clamp(band, -0.035, 0.035) * smoothstep(core, 2.5 * core, abs(band));
        color = mix(original.rgb, guided, k) + vec3(tex * mix(0.35, 0.85, uU1.w) * k);
    }
    if (uU2.y > 0.0) {
        float edge = smoothstep(1.5 * core, 4.0 * core, abs(band));
        float region = mix(0.35, 1.0, exclM) * (1.0 - 0.85 * skin * featureKeep * faceM);
        color += vec3(clamp(band, -0.06, 0.06) * edge * uU2.y * 1.6 * region);
    }
    color = clamp(color, 0.0, 1.0);
    float w = skin * featureKeep * fineDark;
    float liftA = (0.60 * uU3.x + 0.40 * uU1.z) * w;
    float rosyA = 0.25 * uU3.y * w;
    float sculpt = 0.30 * uU2.x * (hiM - 0.3 * faceM) * w * faceM;
    if (liftA > 0.0 || rosyA > 0.0 || sculpt != 0.0) {
        float y0 = luma(color);
        if (uU3.x > 0.0) {
            float chroma = max(color.r - color.b, 0.02);
            float excess = smoothstep(0.40, 0.60, (color.g - color.b) / chroma);
            float target = color.b + 0.38 * chroma;
            color.g -= min(max(color.g - target, 0.0) * excess * 0.6 * uU3.x, 0.02) * w;
        }
        float y1 = max(luma(color), 0.0001);
        float a = liftA * smoothstep(0.12, 0.28, y0) * mix(0.4, 1.0, smoothstep(0.15, 0.5, y0));
        float y2 = y0 * (1.0 + a) / (1.0 + a * y0);
        if (sculpt != 0.0) {
            float s = 0.5 + sculpt;
            y2 = (1.0 - 2.0 * s) * y2 * y2 + 2.0 * s * y2;
        }
        color *= y2 / y1;
        if (rosyA > 0.0) color = vec3(y2) + (color - vec3(y2)) * (1.0 + rosyA * 4.0 * y2 * (1.0 - y2));
        color = fitGamut(color, y2);
    }
    color = mix(vec3(luma(color)), color, uU2.z);
    color = mix(color, clamp(gradePro(color, style), 0.0, 1.0), uU2.w);
    color = clamp(color, 0.0, 1.0);
    if (uU10.x > 0.0) {
        float level = color.b * 63.0;
        vec3 low = texture2D(uTex3, lutCell(floor(level), color.rg)).rgb;
        vec3 high = texture2D(uTex3, lutCell(ceil(level), color.rg)).rgb;
        color = mix(color, mix(low, high, level - floor(level)), uU10.x);
    }
    gl_FragColor = vec4(color, original.a);
}
)";

constexpr const char* kTemporalDenoiseFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
varying vec2 vTextureCoord;
void main() {
    vec4 current = texture2D(uTex0, vTextureCoord);
    vec3 previous = texture2D(uTex1, vTextureCoord).rgb;
    vec3 delta = abs(current.rgb - previous);
    float diff = max(max(delta.r, delta.g), delta.b);
    float weight = uU0.x * (1.0 - smoothstep(uU0.y, uU0.z, diff));
    gl_FragColor = vec4(mix(current.rgb, previous, weight), current.a);
}
)";

constexpr const char* kBgPremaskFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
varying vec2 vTextureCoord;
void main() {
    vec3 color = texture2D(uTex0, vTextureCoord).rgb;
    float person = texture2D(uTex1, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
    float bg = 1.0 - smoothstep(uU0.y, uU0.z, person);
    gl_FragColor = vec4(color * bg, bg);
}
)";

constexpr const char* kBgBlurFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform vec4 uU0;
varying vec2 vTextureCoord;
void main() {
    vec4 sum = texture2D(uTex0, vTextureCoord) * 0.2270270;
    sum += (texture2D(uTex0, vTextureCoord + uU0.xy * 1.3846154) + texture2D(uTex0, vTextureCoord - uU0.xy * 1.3846154)) * 0.3162162;
    sum += (texture2D(uTex0, vTextureCoord + uU0.xy * 3.2307692) + texture2D(uTex0, vTextureCoord - uU0.xy * 3.2307692)) * 0.0702703;
    gl_FragColor = sum;
}
)";

constexpr const char* kBgMixFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
varying vec2 vTextureCoord;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float ellipseE(vec2 uv, vec4 c, vec2 u, float aspect) {
    vec2 d = vec2(uv.x - c.x, (uv.y - c.y) / aspect);
    float a = clamp(dot(d, u) / max(c.z, 0.0001), -4.0, 4.0);
    float b = clamp(dot(d, vec2(-u.y, u.x)) / max(c.w, 0.0001), -4.0, 4.0);
    return a * a + b * b;
}
void main() {
    vec4 color = texture2D(uTex0, vTextureCoord);
    float person = texture2D(uTex2, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
    vec4 blurSample = texture2D(uTex1, vTextureCoord);
    vec3 blurred = blurSample.a > 0.02 ? clamp(blurSample.rgb / blurSample.a, 0.0, 1.0) : color.rgb;
    float head = 1.0 - smoothstep(2.0, 2.8, sqrt(ellipseE(vTextureCoord, uU1, uU2.xy, max(uU2.z, 0.0001))));
    float fg = smoothstep(0.15, 0.30, abs(luma(color.rgb - blurred))) * smoothstep(0.02, 0.10, person) * head;
    float keep = max(smoothstep(uU0.y, uU0.z, person), uU0.w * fg);
    gl_FragColor = vec4(mix(color.rgb, mix(blurred, color.rgb, keep), uU0.x), color.a);
}
)";

// uU0=(a00, a01, a02, shown), uU1=(a10, a11, a12, 0), uU2=oval(cx,cy,rx,ry), uU3=(ax,ay,aspect,hasFace)
const char* kHairAlphaFrag = R"(
precision mediump float;
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
float ellipseE(vec2 p, vec4 o, vec2 a, float aspect) {
    vec2 d = vec2(p.x - o.x, (p.y - o.y) / aspect);
    float u = dot(d, a) / max(o.z, 0.0001);
    float v = dot(d, vec2(-a.y, a.x)) / max(o.w, 0.0001);
    return u * u + v * v;
}
void main() {
    vec2 muv = vec2(uU0.x * vTextureCoord.x + uU0.y * vTextureCoord.y + uU0.z,
                    uU1.x * vTextureCoord.x + uU1.y * vTextureCoord.y + uU1.z);
    float inBounds = step(0.0, muv.x) * step(muv.x, 1.0) * step(0.0, muv.y) * step(muv.y, 1.0);
    float hair = texture2D(uTex0, vec2(muv.x, 1.0 - muv.y)).r * inBounds;
    float inFace = uU3.w > 0.5 ? (1.0 - smoothstep(0.85, 1.15, ellipseE(vTextureCoord, uU2, uU3.xy, max(uU3.z, 0.0001)))) : 0.0;
    float bias = 0.15 * inFace;
    float h = smoothstep(0.35 + bias, 0.65 + bias, hair);
    float nonHair = clamp(1.0 - uU0.w * h, 0.0, 1.0);
    gl_FragColor = vec4(0.0, 0.0, 0.0, nonHair);
}
)";

// Renders in top-down mask space matching personMaskTex so downstream BgPremask/BgMix/BgVirtualMix sample identically.
// uU0=(a00, a01, a02, shown), uU1=(a10, a11, a12, 0), uU2=oval, uU3=(ax,ay,aspect,hasFace)
const char* kBgHairMergeFrag = R"(
precision mediump float;
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
float ellipseE(vec2 p, vec4 o, vec2 a, float aspect) {
    vec2 d = vec2(p.x - o.x, (p.y - o.y) / aspect);
    float u = dot(d, a) / max(o.z, 0.0001);
    float v = dot(d, vec2(-a.y, a.x)) / max(o.w, 0.0001);
    return u * u + v * v;
}
void main() {
    float person = texture2D(uTex0, vTextureCoord).r;
    vec2 dispUv = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y);
    vec2 muv = vec2(uU0.x * dispUv.x + uU0.y * dispUv.y + uU0.z,
                    uU1.x * dispUv.x + uU1.y * dispUv.y + uU1.z);
    float inBounds = step(0.0, muv.x) * step(muv.x, 1.0) * step(0.0, muv.y) * step(muv.y, 1.0);
    float hair = texture2D(uTex1, vec2(muv.x, 1.0 - muv.y)).r * inBounds;
    float head = uU3.w > 0.5 ? (1.0 - smoothstep(2.0, 2.8, sqrt(ellipseE(dispUv, uU2, uU3.xy, max(uU3.z, 0.0001))))) : 1.0;
    float merged = max(person, head * uU0.w * smoothstep(0.30, 0.65, hair));
    gl_FragColor = vec4(merged, merged, merged, 1.0);
}
)";

// uU0=(strength, lo, hi, detailKeep), uU1=(uvScaleX, uvScaleY, uvOffX, uvOffY), uU2=oval, uU3=(ax,ay,aspect,0)
const char* kBgVirtualMixFrag = R"(
precision mediump float;
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float ellipseE(vec2 p, vec4 o, vec2 a, float aspect) {
    vec2 d = vec2(p.x - o.x, (p.y - o.y) / aspect);
    float u = dot(d, a) / max(o.z, 0.0001);
    float v = dot(d, vec2(-a.y, a.x)) / max(o.w, 0.0001);
    return u * u + v * v;
}
void main() {
    vec4 color = texture2D(uTex0, vTextureCoord);
    vec2 bgUv = vec2(vTextureCoord.x * uU1.x + uU1.z, (1.0 - vTextureCoord.y) * uU1.y + uU1.w);
    vec3 vbg = texture2D(uTex1, clamp(bgUv, 0.0, 1.0)).rgb;
    float person = texture2D(uTex2, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
    float head = 1.0 - smoothstep(2.0, 2.8, sqrt(ellipseE(vTextureCoord, uU2, uU3.xy, max(uU3.z, 0.0001))));
    float fg = smoothstep(0.06, 0.16, person) * head;
    float keep = clamp(max(smoothstep(uU0.y, uU0.z, person), uU0.w * 0.45 * fg), 0.0, 1.0);
    gl_FragColor = vec4(mix(color.rgb, mix(vbg, color.rgb, keep), uU0.x), color.a);
}
)";

// uU0=(opacity, 0, 0, 0)
const char* kStickerQuadFrag = R"(
precision mediump float;
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform vec4 uU0;
void main() {
    vec4 s = texture2D(uTex0, vTextureCoord);
    gl_FragColor = vec4(s.rgb, s.a * uU0.x);
}
)";

}  // namespace

ShaderSourcePair getShaderSources(ShaderId id) {
    switch (id) {
        case ShaderId::kDownsample:      return {kDownVert, kDownFrag};
        case ShaderId::kBlurRgb:         return {kBlurVert, kBlurRgbFrag};
        case ShaderId::kBlurAlpha:       return {kBlurVert, kBlurAlphaFrag};
        case ShaderId::kGuidedDiff:         return {kBlurVert, kGuidedDiffFrag};
        case ShaderId::kGuidedCoeff:        return {kBlurVert, kGuidedCoeffFrag};
        case ShaderId::kGuidedWeightBlur:   return {kBlurVert, kGuidedWeightBlurFrag};
        case ShaderId::kGuidedWeightDiff:   return {kBlurVert, kGuidedWeightDiffFrag};
        case ShaderId::kGuidedWeightCoeff:  return {kBlurVert, kGuidedWeightCoeffFrag};
        case ShaderId::kMeshWarp:        return {kQuadVert, kPassThroughFrag};
        case ShaderId::kSkinMask:        return {kSkinMaskVert, kSkinMaskFrag};
        case ShaderId::kRetouchSkin:     return {kRetouchVert, kRetouchSkinFrag};
        case ShaderId::kRetouchEyeTeeth: return {kRetouchVert, kRetouchEyeTeethFrag};
        case ShaderId::kMakeupLayer:     return {kMakeupVert, kMakeupLayerFrag};
        case ShaderId::kMakeupLip:       return {kMakeupVert, kMakeupLipFrag};
        case ShaderId::kMakeupIris:      return {kMakeupVert, kMakeupIrisFrag};
        case ShaderId::kBodyWarp:        return {kQuadVert, kBodyWarpFrag};
        case ShaderId::kCompositePro:    return {kQuadVert, kCompositeProFrag};
        case ShaderId::kTemporalDenoise: return {kQuadVert, kTemporalDenoiseFrag};
        case ShaderId::kBgPremask:       return {kQuadVert, kBgPremaskFrag};
        case ShaderId::kBgBlur:          return {kQuadVert, kBgBlurFrag};
        case ShaderId::kBgMix:           return {kQuadVert, kBgMixFrag};
        case ShaderId::kHairAlpha:       return {kQuadVert, kHairAlphaFrag};
        case ShaderId::kBgHairMerge:     return {kQuadVert, kBgHairMergeFrag};
        case ShaderId::kBgVirtualMix:    return {kQuadVert, kBgVirtualMixFrag};
        case ShaderId::kStickerQuad:     return {kQuadVert, kStickerQuadFrag};
        case ShaderId::kPassThrough:
        default:                         return {kQuadVert, kPassThroughFrag};
    }
}

}  // namespace livebeauty::rhi::gles
