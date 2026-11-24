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

constexpr const char* kSemanticVert = R"(
attribute vec2 aPosition;
attribute vec2 aTextureCoord;
varying vec2 vSrc;
varying vec2 vDst;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vSrc = aTextureCoord;
    vDst = aPosition * 0.5 + 0.5;
}
)";

constexpr const char* kAr3dVert = R"(
attribute vec2 aPosition;
attribute vec2 aTextureCoord;
attribute float aDepth;
varying vec2 vTextureCoord;
varying vec2 vScreenUv;
varying float vDepth;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTextureCoord = aTextureCoord;
    vScreenUv = aPosition * 0.5 + 0.5;
    vDepth = aDepth;
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
uniform vec4 uU1;
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
    vec3 mean = evenMean(texture2D(uTex0, vC).rgb, s, uU1.y);
    float a = guidedCoeff(s, uU1.x);
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
uniform vec4 uU1;
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
    vec3 mean = evenMean(texture2D(uTex0, vC).rgb, s, uU1.y, w);
    float a = mix(1.0, guidedCoeff(s, uU1.x), w);
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
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
uniform vec4 uU4;
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
    vec2 suv = clamp(vScreenUv, 0.0, 1.0);
    float vis = clamp(texture2D(uTex2, suv).a * 1.04 - 0.02, 0.0, 1.0);
    alpha *= vis;
    if (alpha <= 0.001) { discard; }
    vec3 base = texture2D(uTex0, suv).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    if (uU2.z > 0.5) {
        float darkGate = smoothstep(0.14, 0.30, baseY);
        float skinGate = smoothstep(0.025, 0.095, base.r - base.b) * smoothstep(0.015, 0.065, base.r - base.g);
        alpha *= darkGate * skinGate;
        if (alpha <= 0.001) { discard; }
    }
    vec3 amb = mix(vec3(1.0), clamp(uU3.rgb, 0.65, 1.35), clamp(uU3.w, 0.0, 1.0));
    vec3 pigment = clamp(uU0.rgb * amb, 0.0, 1.0);
    vec3 result;
    if (uU2.x < 0.5) {
        result = mix(softLightBlend(base, pigment), base * mix(vec3(1.0), pigment, 0.68), 0.45);
    } else if (uU2.x < 1.5) {
        vec3 mult = base * mix(vec3(1.0), pigment, uU2.z > 0.5 ? 0.78 : 0.92);
        result = uU2.z > 0.5 ? mix(mult, softLightBlend(base, pigment), 0.45) : mult;
    } else {
        result = 1.0 - (1.0 - base) * (1.0 - pigment * 0.58);
    }
    if (uU4.x > 0.0 && uU4.y > 0.0 && (uU4.z > 0.0 || uU4.w > 0.0)) {
        vec2 st = uU4.xy * 1.5;
        vec3 avg = 0.25 * (
            texture2D(uTex0, clamp(suv + vec2(-st.x, -st.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(suv + vec2( st.x, -st.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(suv + vec2(-st.x,  st.y), 0.0, 1.0)).rgb +
            texture2D(uTex0, clamp(suv + vec2( st.x,  st.y), 0.0, 1.0)).rgb
        );
        float poreHi = clamp(baseY - dot(avg, vec3(0.299, 0.587, 0.114)), -0.045, 0.045);
        result += vec3(poreHi * uU4.z * 0.55);
        if (uU2.z > 0.5 && uU4.w > 0.0) {
            result += vec3(min(poreHi, 0.0) * uU4.w * 1.35);
        }
    }
    gl_FragColor = vec4(clamp(result, 0.0, 1.0), alpha);
}
)";

constexpr const char* kMakeupLipFrag = LB_GLSL_PREC R"(
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
void main() {
    float feather = texture2D(uTex1, vMakeupUv).b;
    if (feather * uU0.a <= 0.001) { discard; }
    vec2 uv = clamp(vScreenUv, 0.0, 1.0);
    float vis = clamp(texture2D(uTex2, uv).a * 1.04 - 0.02, 0.0, 1.0);
    if (vis <= 0.002) { discard; }
    vec3 base = texture2D(uTex0, uv).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    float darkGate = smoothstep(0.11, 0.24, baseY);
    float redExcess = base.r - max(base.g, base.b);
    float teethLike = smoothstep(0.42, 0.66, baseY) * (1.0 - smoothstep(0.025, 0.095, redExcess));
    float teethGate = 1.0 - teethLike;
    float lipHue = (base.r - base.g) - 0.75 * max(base.g - base.b, 0.0);
    float rimZone = 1.0 - smoothstep(0.35, 0.92, feather);
    float skinRimGate = mix(1.0, smoothstep(0.045, 0.155, lipHue), rimZone * 0.92);
    float effFeather = feather * darkGate * teethGate * skinRimGate * vis;
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
    vec3 amb = mix(vec3(1.0), clamp(uU2.rgb, 0.65, 1.35), clamp(uU2.w, 0.0, 1.0));
    vec3 lipPigment = clamp(uU0.rgb * amb, 0.0, 1.0);
    float lipDetail = clamp(baseY - smoothY, -0.09, 0.09);
    float isGlossy = step(0.5, uU1.x);
    float liftedY = mix(smoothY, max(smoothY, 0.48), smoothstep(0.18, 0.34, smoothY));
    float illum = mix(0.88, 0.99, smoothstep(0.28, 0.68, liftedY));
    vec3 softTint = clamp(mix(base, lipPigment * illum, 0.68), 0.0, 1.0);
    vec3 pigmentLip = clamp(lipPigment * illum, 0.0, 1.0);
    float coverage = clamp(mix(0.56, 0.84, clamp(uU0.a, 0.0, 1.0)), 0.45, 0.86);
    vec3 result = mix(softTint, pigmentLip, coverage);
    result = clamp(result + vec3(lipDetail * mix(1.05, 0.75, isGlossy)), 0.0, 1.0);
    if (isGlossy > 0.5) {
        float lumaDrive = smoothstep(0.22, 0.56, baseY);
        float yawShift = clamp(uU1.w * 0.18, -0.14, 0.14);
        float centerBelly = 1.0 - smoothstep(0.015, 0.090, abs(vMakeupUv.x - (0.5 + yawShift)));
        float fresnelRim = clamp(abs(uU1.w) * 0.22, 0.0, 0.14) * smoothstep(0.03, 0.14, abs(vMakeupUv.x - 0.5));
        float ridgeMask = clamp(0.88 + lipDetail * 5.2, 0.20, 1.45);
        float dyLo = (vMakeupUv.y - 0.714) * 145.0;
        float dyHi = (vMakeupUv.y - 0.673) * 165.0;
        float lowerBand = exp(-(dyLo * dyLo));
        float upperBand = exp(-(dyHi * dyHi)) * 0.44;
        float microGlint = smoothstep(0.014, 0.048, lipDetail) * centerBelly * 0.26;
        float naturalHot = smoothstep(0.76, 0.94, baseY) * 0.26;
        float spec = clamp(((lowerBand + upperBand) * (centerBelly + fresnelRim) * lumaDrive * 0.82 + microGlint + naturalHot) * ridgeMask * effFeather, 0.0, 0.74);
        vec3 specCol = clamp(vec3(1.0, 0.99, 0.97) * mix(vec3(1.0), amb, 0.35), 0.0, 1.0);
        result = clamp(result + (specCol - result) * (spec * 0.56), 0.0, 1.0);
    } else {
        float velvetSoft = smoothstep(0.68, 0.92, baseY) * 0.08 * effFeather;
        result = clamp(mix(result, lipPigment * 0.96 + vec3(0.03), velvetSoft), 0.0, 1.0);
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
        vec2 parallax = vec2(clamp(uU1.w * 0.18, -0.12, 0.12), 0.0);
        vec2 p0 = d - (vec2(-0.26, 0.28) + parallax);
        vec2 p1 = d - (vec2(0.22, -0.18) + parallax * 0.65);
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
uniform vec4 uU3;
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
            float handKeep = 1.0;
            if (uU2.w > 0.5) {
                if (uU3.x >= 0.0) {
                    vec2 dh0 = vec2(uv.x - uU3.x, (uv.y - uU3.y) / aspect);
                    handKeep *= smoothstep(0.12 * max(torso, 0.15), 0.34 * max(torso, 0.15), length(dh0));
                }
                if (uU3.z >= 0.0) {
                    vec2 dh1 = vec2(uv.x - uU3.z, (uv.y - uU3.w) / aspect);
                    handKeep *= smoothstep(0.12 * max(torso, 0.15), 0.34 * max(torso, 0.15), length(dh1));
                }
            }
            s *= 1.0 + uU1.z * handKeep * exp(-0.5 * tw * tw) * f;
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
    float warmPoly = smoothstep(0.018, 0.055, fineLow.r - fineLow.b) *
        smoothstep(0.0, 0.025, fineLow.r - fineLow.g + 0.010);
    float polyGate = smoothstep(0.06, 0.18, luma(fineLow)) *
        mix(0.30 * smoothstep(0.42, 0.72, faceM) * warmPoly, 1.0, smoothstep(0.0, 0.25, gate));
    float skin = clamp(faceM * polyGate + (1.0 - faceM) * gate * chromaGate * (1.0 - 0.70 * exclM), 0.0, 1.0) * nonHair;
    vec3 guided = clamp(original.rgb * guidedA + guidedB, 0.0, 1.0);
    float bleedGuard = 1.0 - (1.0 - smoothstep(0.22, 0.31, luma(fineLow))) *
        max(smoothstep(0.020, 0.055, luma(guided) - luma(fineLow)),
            smoothstep(0.24, 0.48, guidedA) * smoothstep(0.003, 0.018, luma(guided) - luma(fineLow)));
    float fineDark = (1.0 - smoothstep(0.02, 0.06, luma(fineLow) - luma(original.rgb)) * smoothstep(0.25, 0.55, guidedA)) * bleedGuard;
    float featureKeep = clamp(1.0 - exclM, 0.0, 1.0);
    float core = max(uU10.y, 0.005);
    float band = luma(fineLow - wideLow);
    float micro = luma(original.rgb - fineLow);
    vec3 color = original.rgb;
    if (uU1.y > 0.0) {
        float rEdge = length(original.rgb - guided) / 0.36;
        float rEdge2 = rEdge * rEdge;
        float k = uU1.y * 0.96 * (skinOnly > 0.5 ? skin : 1.0) * featureKeep * fineDark / (1.0 + rEdge2 * rEdge2);
        float tex = clamp(band, -0.035, 0.035) * smoothstep(core, 2.5 * core, abs(band));
        float microTex = clamp(micro, -0.022, 0.022) * smoothstep(core * 0.85, core * 2.2, abs(micro));
        color = mix(original.rgb, guided, k) + vec3((tex * mix(0.35, 0.85, uU1.w) + microTex * uU10.z) * k);
    }
    float mouthZone = 0.0;
    if (uU7.z > 0.001 && exclM > 0.25) {
        mouthZone = (1.0 - smoothstep(0.85, 1.45, ellipseE(uv, uU7, vec2(uU4.w, uU5.w), max(uU0.z, 0.0001)))) *
                    smoothstep(0.32, 0.72, exclM) * nonHair;
    }
    if (uU2.y > 0.0) {
        float edge = smoothstep(1.5 * core, 4.0 * core, abs(band));
        float region = mix(0.35, 1.0, exclM) * (1.0 - 0.85 * skin * featureKeep * faceM);
        float sharpDelta = clamp(band, mix(-0.06, -0.01, mouthZone), 0.06);
        color += vec3(sharpDelta * edge * uU2.y * 1.6 * region);
    }
    color = clamp(color, 0.0, 1.0);
    float w = skin * featureKeep * fineDark;
    if (uU3.z > 0.0 && w > 0.0) {
        float yCur = luma(color);
        float chCur = max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
        float hotY = smoothstep(0.62, 0.90, yCur);
        float lowSat = 1.0 - smoothstep(0.08, 0.30, chCur);
        float tZone = mix(0.45, 1.0, clamp(hiM, 0.0, 1.0));
        float oilMask = clamp(hotY * lowSat * tZone * w * uU3.z, 0.0, 0.65);
        if (oilMask > 0.001) {
            vec3 matteRef = mix(color * 0.86, min(color, guided * 0.94 + vec3(0.03, 0.02, 0.01)), step(0.001, uU1.y));
            float yMat = luma(matteRef);
            vec3 warmSkin = clamp(mix(matteRef, vec3(yMat * 1.06, yMat * 0.97, yMat * 0.90), 0.28), 0.0, 1.0);
            color = mix(color, warmSkin, oilMask);
        }
    }
    float yPre = luma(color);
    float relightFill = 0.22 * uU3.w * w * faceM * (1.0 - smoothstep(0.20, 0.55, yPre));
    float liftA = (1.12 * uU3.x + 0.48 * uU1.z) * w + relightFill;
    float rosyA = 0.25 * uU3.y * w;
    float sculpt = (0.30 * uU2.x + 0.42 * uU3.w) * (hiM - 0.30 * faceM) * w * faceM;
    if (liftA > 0.0 || rosyA > 0.0 || sculpt != 0.0) {
        float y0 = luma(color);
        if (uU3.x > 0.0) {
            float chroma = max(color.r - color.b, 0.02);
            float excess = smoothstep(0.32, 0.56, (color.g - color.b) / chroma);
            float target = color.b + 0.34 * chroma;
            color.g -= min(max(color.g - target, 0.0) * excess * 0.75 * uU3.x, 0.035) * w;
        }
        float y1 = max(luma(color), 0.0001);
        float shadowGate = smoothstep(0.07, 0.20, y0) * mix(0.68, 1.0, smoothstep(0.14, 0.38, y0));
        float hiRoll = 1.0 - 0.55 * smoothstep(0.78, 0.96, y0);
        float a = liftA * shadowGate * hiRoll;
        float y2 = y0 * (1.0 + a) / (1.0 + a * y0);
        if (sculpt != 0.0) {
            float s = 0.5 + sculpt;
            y2 = (1.0 - 2.0 * s) * y2 * y2 + 2.0 * s * y2;
        }
        color *= y2 / y1;
        if (uU3.x > 0.0) {
            float coolW = clamp(uU3.x * w, 0.0, 1.0);
            vec3 ivory = vec3(min(y2 * 1.04, 1.0), y2 * 0.99, min(y2 * 1.03, 1.0));
            color = mix(color, ivory, 0.32 * coolW);
            y2 = luma(color);
        }
        if (rosyA > 0.0) color = vec3(y2) + (color - vec3(y2)) * (1.0 + rosyA * 4.0 * y2 * (1.0 - y2));
        color = fitGamut(color, y2);
    }
    if (mouthZone > 0.001 && (uU3.x > 0.0 || uU3.y > 0.0)) {
        float yLip = luma(color);
        float redEx = color.r - max(color.g, color.b);
        float notSeam = smoothstep(0.14, 0.26, yLip);
        float notTeeth = 1.0 - smoothstep(0.42, 0.64, yLip) * (1.0 - smoothstep(0.025, 0.095, redEx));
        float darkDeficit = 1.0 - 0.72 * smoothstep(0.38, 0.58, yLip);
        float drive = clamp(smoothstep(0.0, 0.35, uU3.x + uU3.y) * (0.58 + 0.42 * clamp(uU3.x + uU3.y, 0.0, 1.0)), 0.0, 1.0);
        float lipCare = clamp(mouthZone * drive * notSeam * notTeeth * darkDeficit, 0.0, 0.75);
        float liftedLipY = clamp(max(yLip * 1.16 + 0.055, mix(yLip, 0.46, 0.58)), 0.0, 0.68);
        vec3 healthyLip = clamp(vec3(liftedLipY * 1.38 + 0.045, liftedLipY * 0.85 + 0.008, liftedLipY * 0.87 + 0.014), 0.0, 1.0);
        color = mix(color, healthyLip, lipCare * 0.68);
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
    if (uU0.w > 0.0) {
        vec2 stepUv = vec2(uU0.w * 2.5, uU0.w * 2.5);
        vec3 d0 = abs(texture2D(uTex0, clamp(vTextureCoord + stepUv, 0.0, 1.0)).rgb -
                      texture2D(uTex1, clamp(vTextureCoord + stepUv, 0.0, 1.0)).rgb);
        vec3 d1 = abs(texture2D(uTex0, clamp(vTextureCoord - stepUv, 0.0, 1.0)).rgb -
                      texture2D(uTex1, clamp(vTextureCoord - stepUv, 0.0, 1.0)).rgb);
        vec3 d2 = abs(texture2D(uTex0, clamp(vTextureCoord + vec2(stepUv.x, -stepUv.y), 0.0, 1.0)).rgb -
                      texture2D(uTex1, clamp(vTextureCoord + vec2(stepUv.x, -stepUv.y), 0.0, 1.0)).rgb);
        vec3 d3 = abs(texture2D(uTex0, clamp(vTextureCoord + vec2(-stepUv.x, stepUv.y), 0.0, 1.0)).rgb -
                      texture2D(uTex1, clamp(vTextureCoord + vec2(-stepUv.x, stepUv.y), 0.0, 1.0)).rgb);
        vec3 dMax = max(max(d0, d1), max(d2, d3));
        diff = max(diff, max(max(dMax.r, dMax.g), dMax.b));
    }
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
    float rFace = sqrt(ellipseE(vTextureCoord, uU1, uU2.xy, max(uU2.z, 0.0001)));
    float head = 1.0 - smoothstep(2.0, 2.8, rFace);
    float faceCore = (1.0 - smoothstep(0.84, 1.08, rFace)) * clamp(uU2.w, 0.0, 1.0) * 0.98;
    float fg = smoothstep(0.15, 0.30, abs(luma(color.rgb - blurred))) * smoothstep(0.02, 0.10, person) * head;
    float keep = max(max(smoothstep(uU0.y, uU0.z, person), uU0.w * fg), faceCore);
    gl_FragColor = vec4(mix(color.rgb, mix(blurred, color.rgb, keep), uU0.x), color.a);
}
)";

// uU0=(a00, a01, a02, hairShown), uU1=(a10, a11, a12, hasOcclusion), uU2=oval(cx,cy,rx,ry), uU3=(ax,ay,aspect,hasFace)
constexpr const char* kHairAlphaFrag = LB_GLSL_PREC R"(
varying vec2 vSrc;
varying vec2 vDst;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
float ellipseE(vec2 p, vec4 o, vec2 a, float aspect) {
    vec2 d = vec2(p.x - o.x, (p.y - o.y) / aspect);
    float u = clamp(dot(d, a) / max(o.z, 0.0001), -4.0, 4.0);
    float v = clamp(dot(d, vec2(-a.y, a.x)) / max(o.w, 0.0001), -4.0, 4.0);
    return u * u + v * v;
}
void main() {
    float nonHair = 1.0;
    if (uU0.w > 0.001) {
        vec2 muv = vec2(uU0.x * vSrc.x + uU0.y * vSrc.y + uU0.z,
                        uU1.x * vSrc.x + uU1.y * vSrc.y + uU1.z);
        float inBounds = step(0.0, muv.x) * step(muv.x, 1.0) * step(0.0, muv.y) * step(muv.y, 1.0);
        float hair = texture2D(uTex0, vec2(muv.x, 1.0 - muv.y)).r * inBounds;
        float inFace = uU3.w > 0.5 ? (1.0 - smoothstep(0.50, 1.0, ellipseE(vDst, uU2, uU3.xy, max(uU3.z, 0.0001)))) : 0.0;
        float bias = 0.15 * inFace;
        float h = smoothstep(0.35 + bias, 0.65 + bias, hair);
        nonHair = clamp(1.0 - uU0.w * h, 0.0, 1.0);
    }
    if (uU1.w > 0.001) {
        vec2 ouv = vec2(uU0.x * vSrc.x + uU0.y * vSrc.y + uU0.z,
                        uU1.x * vSrc.x + uU1.y * vSrc.y + uU1.z);
        float occIn = step(0.0, ouv.x) * step(ouv.x, 1.0) * step(0.0, ouv.y) * step(ouv.y, 1.0);
        float occ = mix(1.0, texture2D(uTex1, vec2(ouv.x, 1.0 - ouv.y)).r, occIn);
        nonHair *= mix(1.0, smoothstep(0.25, 0.70, occ), clamp(uU1.w, 0.0, 1.0));
    }
    gl_FragColor = vec4(0.0, 0.0, 0.0, nonHair);
}
)";

// Renders in top-down mask space matching personMaskTex so downstream BgPremask/BgMix/BgVirtualMix sample identically.
// uU0=(a00, a01, a02, shown), uU1=(a10, a11, a12, 0), uU2=oval, uU3=(ax,ay,aspect,hasFace)
constexpr const char* kBgHairMergeFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
float ellipseE(vec2 p, vec4 o, vec2 a, float aspect) {
    vec2 d = vec2(p.x - o.x, (p.y - o.y) / aspect);
    float u = clamp(dot(d, a) / max(o.z, 0.0001), -4.0, 4.0);
    float v = clamp(dot(d, vec2(-a.y, a.x)) / max(o.w, 0.0001), -4.0, 4.0);
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

// uU0=(strength, lo, hi, primaryFacePresence), uU1=(uvScaleX, uvScaleY, uvOffX, uvOffY),
// uU2=oval, uU3=(ax, ay, aspect, secFacePresence), uU4=secOval, uU5=(secAx, secAy, 0, 0)
constexpr const char* kBgVirtualMixFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
uniform vec4 uU4;
uniform vec4 uU5;
float personAnchorShield(vec2 uv, vec4 o, vec2 u, float aspect, float presence) {
    if (presence <= 0.001) return 0.0;
    vec2 d = vec2(uv.x - o.x, (uv.y - o.y) / aspect);
    float a = clamp(dot(d, u) / max(o.z, 0.0001), -6.0, 6.0);
    float b = clamp(dot(d, vec2(-u.y, u.x)) / max(o.w, 0.0001), -6.0, 6.0);
    float rFace = sqrt(a * a + b * b);
    float faceCore = (1.0 - smoothstep(0.80, 1.00, rFace)) * 0.99;
    float down = -b;
    float neckBand = smoothstep(0.60, 0.85, down) * (1.0 - smoothstep(1.35, 1.62, down));
    float neckCore = neckBand * (1.0 - smoothstep(0.46, 0.68, abs(a))) * 0.94;
    return clamp(max(faceCore, neckCore) * presence, 0.0, 1.0);
}
void main() {
    vec4 color = texture2D(uTex0, vTextureCoord);
    vec2 bgUv = clamp(vec2(vTextureCoord.x * uU1.x + uU1.z, (1.0 - vTextureCoord.y) * uU1.y + uU1.w), 0.0, 1.0);
    vec3 vbg = texture2D(uTex1, bgUv).rgb;
    float person = texture2D(uTex2, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
    float asp = max(uU3.z, 0.0001);
    float shield0 = personAnchorShield(vTextureCoord, uU2, uU3.xy, asp, clamp(uU0.w, 0.0, 1.0));
    float shield1 = personAnchorShield(vTextureCoord, uU4, uU5.xy, asp, clamp(uU3.w, 0.0, 1.0));
    float keep = clamp(max(smoothstep(uU0.y, uU0.z, person), max(shield0, shield1)), 0.0, 1.0);
    // Subtle Light Wrap strictly on outer translucent boundary (keep in 0.15..0.88; 0% tint on bg or solid fg)
    vec2 wrapD = vec2(0.020, 0.020);
    vec3 vbgAmb = 0.20 * (
        vbg +
        texture2D(uTex1, clamp(bgUv + vec2( wrapD.x, 0.0), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(-wrapD.x, 0.0), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(0.0,  wrapD.y), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(0.0, -wrapD.y), 0.0, 1.0)).rgb
    );
    float edgeBand = smoothstep(0.15, 0.48, keep) * (1.0 - smoothstep(0.55, 0.88, keep));
    vec3 rimGlow = max(color.rgb, vbgAmb * 0.90);
    vec3 fgWrapped = mix(color.rgb, mix(rimGlow, vbgAmb, 0.14), edgeBand * 0.14);
    gl_FragColor = vec4(mix(color.rgb, mix(vbg, fgWrapped, keep), uU0.x), color.a);
}
)";

// uU0=(opacity, 0, 0, 0)
constexpr const char* kStickerQuadFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform vec4 uU0;
void main() {
    vec4 s = texture2D(uTex0, vTextureCoord);
    gl_FragColor = vec4(s.rgb, s.a * uU0.x);
}
)";

// P2: RGB-guided 5-tap joint bilateral + motion-adaptive temporal portrait matte refinement.
// Operates in top-down mask space matching personMaskTex / rtBgHair_.
// uTex0=curMask(top-down), uTex1=curRgb(bottom-up), uTex2=prevMask(top-down), uTex3=prevRgb(bottom-up)
// uU0=(texelX, texelY, temporalWeight, hasHistory), uU1=(sigmaColor, motionLo, motionHi, 0)
constexpr const char* kMatteRefineFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform sampler2D uTex3;
uniform vec4 uU0;
uniform vec4 uU1;
void main() {
    vec2 muv = vTextureCoord;
    vec2 ruv = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y);
    float cMask = texture2D(uTex0, muv).r;
    vec3 cRgb = texture2D(uTex1, ruv).rgb;
    float inv2Sig2 = 0.5 / max(uU1.x * uU1.x, 0.001);
    vec2 d = uU0.xy * 1.25;
    float sumW = 1.4;
    float sumM = cMask * 1.4;

    vec2 o0 = vec2(-d.x, -d.y);
    vec3 dc0 = texture2D(uTex1, clamp(ruv + vec2(o0.x, -o0.y), 0.0, 1.0)).rgb - cRgb;
    float w0 = exp(-dot(dc0, dc0) * inv2Sig2);
    sumW += w0; sumM += w0 * texture2D(uTex0, clamp(muv + o0, 0.0, 1.0)).r;

    vec2 o1 = vec2(d.x, -d.y);
    vec3 dc1 = texture2D(uTex1, clamp(ruv + vec2(o1.x, -o1.y), 0.0, 1.0)).rgb - cRgb;
    float w1 = exp(-dot(dc1, dc1) * inv2Sig2);
    sumW += w1; sumM += w1 * texture2D(uTex0, clamp(muv + o1, 0.0, 1.0)).r;

    vec2 o2 = vec2(-d.x, d.y);
    vec3 dc2 = texture2D(uTex1, clamp(ruv + vec2(o2.x, -o2.y), 0.0, 1.0)).rgb - cRgb;
    float w2 = exp(-dot(dc2, dc2) * inv2Sig2);
    sumW += w2; sumM += w2 * texture2D(uTex0, clamp(muv + o2, 0.0, 1.0)).r;

    vec2 o3 = vec2(d.x, d.y);
    vec3 dc3 = texture2D(uTex1, clamp(ruv + vec2(o3.x, -o3.y), 0.0, 1.0)).rgb - cRgb;
    float w3 = exp(-dot(dc3, dc3) * inv2Sig2);
    sumW += w3; sumM += w3 * texture2D(uTex0, clamp(muv + o3, 0.0, 1.0)).r;

    float guidedMask = clamp(sumM / sumW, 0.0, 1.0);
    float outMask = guidedMask;
    if (uU0.w > 0.5 && uU0.z > 0.001) {
        float pMask = texture2D(uTex2, muv).r;
        vec3 pRgb = texture2D(uTex3, ruv).rgb;
        vec3 pRgb0 = texture2D(uTex3, clamp(ruv + vec2(o0.x, -o0.y), 0.0, 1.0)).rgb;
        vec3 pRgb3 = texture2D(uTex3, clamp(ruv + vec2(o3.x, -o3.y), 0.0, 1.0)).rgb;
        vec3 dC = abs(cRgb - pRgb);
        vec3 d0 = abs((dc0 + cRgb) - pRgb0);
        vec3 d3 = abs((dc3 + cRgb) - pRgb3);
        float rgbDiff = max(max(dC.r, max(dC.g, dC.b)),
                        0.75 * max(max(d0.r, max(d0.g, d0.b)), max(d3.r, max(d3.g, d3.b))));
        float shrinkClear = max((pMask - guidedMask) - 0.45, 0.0) * 0.12;
        float motion = max(rgbDiff, shrinkClear);
        float keep = uU0.z * (1.0 - smoothstep(uU1.y, uU1.z, motion));
        outMask = mix(guidedMask, pMask, keep);
    }
    gl_FragColor = vec4(outMask, outMask, outMask, 1.0);
}
)";

// P3: Multi-scale isolated blemish / acne / redness healer strictly inside face skin mask.
// uTex0=currentRgb, uTex1=rtSkinMask_ (R=faceM, G=exclM, A=nonHair*occ)
// uU0=(texelX, texelY, strength, presence)
constexpr const char* kBlemishHealFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
float luma7(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main() {
    vec2 uv = vTextureCoord;
    vec4 orig = texture2D(uTex0, uv);
    vec4 sem = texture2D(uTex1, uv);
    float skinMask = sem.r * (1.0 - sem.g) * clamp(sem.a * 1.04 - 0.02, 0.0, 1.0) * uU0.w;
    if (skinMask <= 0.01 || uU0.z <= 0.001) {
        gl_FragColor = orig;
        return;
    }
    vec2 r = uU0.xy * 5.5;
    vec2 rd = r * 0.7071;
    vec3 s0 = texture2D(uTex0, clamp(uv + vec2( r.x, 0.0), 0.0, 1.0)).rgb;
    vec3 s1 = texture2D(uTex0, clamp(uv + vec2(-r.x, 0.0), 0.0, 1.0)).rgb;
    vec3 s2 = texture2D(uTex0, clamp(uv + vec2(0.0,  r.y), 0.0, 1.0)).rgb;
    vec3 s3 = texture2D(uTex0, clamp(uv + vec2(0.0, -r.y), 0.0, 1.0)).rgb;
    vec3 s4 = texture2D(uTex0, clamp(uv + vec2( rd.x,  rd.y), 0.0, 1.0)).rgb;
    vec3 s5 = texture2D(uTex0, clamp(uv + vec2(-rd.x,  rd.y), 0.0, 1.0)).rgb;
    vec3 s6 = texture2D(uTex0, clamp(uv + vec2( rd.x, -rd.y), 0.0, 1.0)).rgb;
    vec3 s7 = texture2D(uTex0, clamp(uv + vec2(-rd.x, -rd.y), 0.0, 1.0)).rgb;
    vec3 ringAvg = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7) * 0.125;

    float spread = 0.25 * (
        length(s0 - ringAvg) + length(s1 - ringAvg) +
        length(s2 - ringAvg) + length(s3 - ringAvg)
    );
    float yC = luma7(orig.rgb);
    float yR = luma7(ringAvg);
    float darkDeficit = max(yR - yC, 0.0);
    float redC = orig.r - 0.5 * (orig.g + orig.b);
    float redR = ringAvg.r - 0.5 * (ringAvg.g + ringAvg.b);
    float redExcess = max(redC - redR, 0.0);

    float uniformRing = 1.0 - smoothstep(0.055, 0.145, spread);
    float spot = max(smoothstep(0.032, 0.110, darkDeficit), smoothstep(0.028, 0.095, redExcess)) * uniformRing;
    float notBlack = smoothstep(0.08, 0.18, yC);
    float healAlpha = clamp(spot * notBlack * skinMask * uU0.z * 0.92, 0.0, 0.90);
    gl_FragColor = vec4(mix(orig.rgb, ringAvg, healAlpha), orig.a);
}
)";

// P3: Pluggable Face-ROI Neural Residual Delta Fusion (RGBA8, 0.5=zero delta, A=confidence mask)
// uTex0=currentRgb, uTex1=aiResidualTex(top-down ROI), uTex2=rtSkinMask_
// uU0=(roiLeft, roiTop, roiWidth, roiHeight), uU1=(strength, hasSkinMask, 0, 0)
constexpr const char* kAiResidualBlendFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
void main() {
    vec4 base = texture2D(uTex0, vTextureCoord);
    vec2 topUv = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y);
    vec2 roiUv = (topUv - uU0.xy) / max(uU0.zw, vec2(0.001));
    if (roiUv.x <= 0.0 || roiUv.x >= 1.0 || roiUv.y <= 0.0 || roiUv.y >= 1.0 || uU1.x <= 0.001) {
        gl_FragColor = base;
        return;
    }
    float edge = smoothstep(0.0, 0.06, roiUv.x) * (1.0 - smoothstep(0.94, 1.0, roiUv.x)) *
                 smoothstep(0.0, 0.06, roiUv.y) * (1.0 - smoothstep(0.94, 1.0, roiUv.y));
    vec4 res = texture2D(uTex1, clamp(roiUv, 0.0, 1.0));
    vec4 sem = texture2D(uTex2, vTextureCoord);
    float semMask = mix(1.0, sem.r * (1.0 - sem.g) * clamp(sem.a * 1.04 - 0.02, 0.0, 1.0), uU1.y);
    vec3 delta = (res.rgb - vec3(0.5)) * 0.50;
    vec3 outRgb = clamp(base.rgb + delta * (res.a * edge * semMask * uU1.x), 0.0, 1.0);
    gl_FragColor = vec4(outRgb, base.a);
}
)";

// P4: 3D head-occluder depth shell (higher depth = closer to camera)
constexpr const char* kHeadDepthShellFrag = LB_GLSL_PREC R"(
varying float vDepth;
void main() {
    float d = clamp(vDepth, 0.0, 1.0);
    gl_FragColor = vec4(d, d, d, 1.0);
}
)";

// P4: 3D perspective AR accessory quad with head-depth soft occlusion
// uTex0=accessoryRgba, uTex1=rtHeadDepth_
// uU0=(opacity, headOcclusionEnable, 0, 0)
constexpr const char* kAr3dMeshFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
varying vec2 vScreenUv;
varying float vDepth;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
void main() {
    vec4 s = texture2D(uTex0, vTextureCoord);
    float a = s.a * uU0.x;
    if (a <= 0.002) { discard; }
    if (uU0.y > 0.5) {
        float headD = texture2D(uTex1, clamp(vScreenUv, 0.0, 1.0)).r;
        float vis = smoothstep(headD - 0.030, headD - 0.004, vDepth);
        a *= vis;
        if (a <= 0.002) { discard; }
    }
    gl_FragColor = vec4(s.rgb, clamp(a, 0.0, 1.0));
}
)";

// v3.4/v3.6: Real-time strand-preserving salon hair dyeing (melanin bleach curve + micro-strand detail) & dual-lobe Angel-Ring sheen
// uTex0=currentRgb, uTex1=hairMaskTex(top-down), uTex2=rtSkinMask_
// uU0=(a00, a01, a02, hairDye), uU1=(a10, a11, a12, hairShine)
// uU2=(dyeR, dyeG, dyeB, hasFacePoly), uU3=(crownCx, crownCy, crownRx, crownRy), uU4=(axisUx, axisUy, aspect, signedYaw), uU5=(texelX, texelY, 0, 0)
constexpr const char* kHairDyeFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
uniform vec4 uU4;
uniform vec4 uU5;
float lumaHair(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 fitHairGamut(vec3 c, float y) {
    float hi = max(max(c.r, c.g), c.b);
    float lo = min(min(c.r, c.g), c.b);
    float s = 1.0;
    if (hi > 1.0) s = min(s, (1.0 - y) / max(hi - y, 0.0001));
    if (lo < 0.0) s = min(s, y / max(y - lo, 0.0001));
    return vec3(y) + (c - vec3(y)) * s;
}
void main() {
    vec2 uv = vTextureCoord;
    vec4 orig = texture2D(uTex0, uv);
    vec2 muv = vec2(uU0.x * uv.x + uU0.y * uv.y + uU0.z,
                    uU1.x * uv.x + uU1.y * uv.y + uU1.z);
    float inBounds = step(0.0, muv.x) * step(muv.x, 1.0) * step(0.0, muv.y) * step(muv.y, 1.0);
    float rawHair = texture2D(uTex1, vec2(muv.x, 1.0 - muv.y)).r * inBounds;
    float hairMask = smoothstep(0.28, 0.72, rawHair);
    if (uU2.w > 0.5) {
        vec4 sem = texture2D(uTex2, uv);
        float bareFace = clamp(sem.r * clamp(sem.a * 1.04 - 0.02, 0.0, 1.0) * 0.85 + sem.g, 0.0, 1.0);
        hairMask *= (1.0 - smoothstep(0.45, 0.85, bareFace));
    }
    if (hairMask <= 0.002) {
        gl_FragColor = orig;
        return;
    }
    vec3 color = orig.rgb;
    if (uU0.w > 0.001) {
        float y = lumaHair(color);
        float strandLift = smoothstep(0.015, 0.12, y);
        float dyeY = max(lumaHair(uU2.rgb), 0.08);
        float bleachNeed = smoothstep(0.28, 0.72, dyeY) * smoothstep(0.012, 0.09, y) * (1.0 - smoothstep(0.18, 0.45, y));
        float strandDetail = 0.0;
        if (uU5.x > 0.0001) {
            vec2 stepLat = uU4.xy * uU5.x * 1.5;
            float avgLat = 0.5 * (
                lumaHair(texture2D(uTex0, clamp(uv + stepLat, 0.0, 1.0)).rgb) +
                lumaHair(texture2D(uTex0, clamp(uv - stepLat, 0.0, 1.0)).rgb)
            );
            strandDetail = clamp(y - avgLat, -0.06, 0.06);
        }
        float liftedY = clamp(
            mix(y, clamp(y * (0.65 + dyeY * 1.35) + dyeY * 0.36 * strandLift + bleachNeed * dyeY * 0.32, 0.0, 0.95), 0.80) +
            strandDetail * 1.20,
            0.0, 0.96);
        vec3 tinted = fitHairGamut(clamp(uU2.rgb * (liftedY / dyeY), 0.0, 1.28), liftedY);
        color = mix(color, clamp(tinted, 0.0, 1.0), clamp(hairMask * uU0.w * (0.36 + 0.64 * strandLift), 0.0, 0.92));
    }
    if (uU1.w > 0.001) {
        vec2 axisU = uU4.xy;
        vec2 axisV = vec2(-axisU.y, axisU.x);
        vec2 d = vec2(uv.x - uU3.x, (uv.y - uU3.y) / max(uU4.z, 0.001));
        float su = dot(d, axisU) / max(uU3.z, 0.04);
        float sv = dot(d, axisV) / max(uU3.w, 0.04);
        float latKeep = 1.0 - smoothstep(0.65, 1.35, abs(su));
        float arcR = (sv + 0.18 * su * su) * 4.2;
        float arcTrt = (sv + 0.18 * su * su - 0.12) * 6.5;
        float ringBand = (exp(-(arcR * arcR)) + 0.32 * exp(-(arcTrt * arcTrt))) * latKeep;
        float yawPeak = clamp(-uU4.w * 0.45, -0.35, 0.35);
        float du = su - yawPeak;
        float viewLobe = 0.65 + 0.35 * exp(-(du * du) * 2.5);
        float strandCatch = smoothstep(0.04, 0.32, lumaHair(color)) * (0.55 + 0.45 * smoothstep(0.10, 0.45, orig.r));
        float sheen = clamp(ringBand * viewLobe * strandCatch * hairMask * uU1.w * 0.50, 0.0, 0.46);
        vec3 sheenCol = mix(vec3(1.0, 0.99, 0.97), clamp(uU2.rgb * 1.35 + vec3(0.25), 0.0, 1.0), clamp(uU0.w * 0.45, 0.0, 0.45));
        color = clamp(color + (sheenCol - color) * sheen, 0.0, 1.0);
    }
    gl_FragColor = vec4(color, orig.a);
}
)";

// v3.6: Studio YCbCr Chroma Key (green/blue screen) + edge despill + 5-tap virtual background Light Wrap
// uTex0=currentRgb, uTex1=virtualBgRgba(top-down)
// uU0=(strength, similarity, smoothness, despill), uU1=(uvScaleX, uvScaleY, uvOffX, uvOffY),
// uU2=(keyR, keyG, keyB, isBlueKey), uU3=oval, uU4=(ax, ay, aspect, facePresence)
constexpr const char* kChromaKeyFrag = LB_GLSL_PREC R"(
varying vec2 vTextureCoord;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uU0;
uniform vec4 uU1;
uniform vec4 uU2;
uniform vec4 uU3;
uniform vec4 uU4;
float lumaCk(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec2 toCbCr(vec3 c) {
    return vec2(0.5 + dot(c, vec3(-0.168736, -0.331264, 0.5)),
                0.5 + dot(c, vec3(0.5, -0.418688, -0.081312)));
}
void main() {
    vec4 color = texture2D(uTex0, vTextureCoord);
    vec2 bgUv = clamp(vec2(vTextureCoord.x * uU1.x + uU1.z, (1.0 - vTextureCoord.y) * uU1.y + uU1.w), 0.0, 1.0);
    vec3 vbg = texture2D(uTex1, bgUv).rgb;
    vec2 cc = toCbCr(color.rgb);
    vec2 kc = toCbCr(uU2.rgb);
    float dist = length(cc - kc);
    float sim = clamp(uU0.y, 0.05, 0.95) * 0.55;
    float sm = max(uU0.z * 0.45, 0.008);
    float rawKeyed = 1.0 - smoothstep(sim, sim + sm, dist);
    float dom = (uU2.w < 0.5)
        ? (color.g - max(color.r, color.b))
        : (color.b - max(color.r, color.g));
    float domGate = smoothstep(0.015, 0.075, dom);
    float faceShield = 0.0;
    if (uU4.w > 0.001) {
        vec2 df = vec2(vTextureCoord.x - uU3.x, (vTextureCoord.y - uU3.y) / max(uU4.z, 0.0001));
        float af = clamp(dot(df, uU4.xy) / max(uU3.z, 0.0001), -4.0, 4.0);
        float bf = clamp(dot(df, vec2(-uU4.y, uU4.x)) / max(uU3.w, 0.0001), -4.0, 4.0);
        faceShield = (1.0 - smoothstep(0.86, 1.12, sqrt(af * af + bf * bf))) * clamp(uU4.w, 0.0, 1.0);
    }
    float keep = clamp(max(1.0 - rawKeyed * domGate, faceShield), 0.0, 1.0);

    float baseY = lumaCk(color.rgb);
    vec3 fg = color.rgb;
    if (uU2.w < 0.5) {
        float refMax = mix(0.5 * (fg.r + fg.b), max(fg.r, fg.b), 0.35);
        float spill = max(fg.g - refMax, 0.0) * clamp(uU0.w, 0.0, 1.0);
        fg.g -= spill;
    } else {
        float refMax = mix(0.5 * (fg.r + fg.g), max(fg.r, fg.g), 0.35);
        float spill = max(fg.b - refMax, 0.0) * clamp(uU0.w, 0.0, 1.0);
        fg.b -= spill;
    }
    fg = clamp(fg + vec3((baseY - lumaCk(fg)) * 0.65), 0.0, 1.0);

    vec2 wrapD = vec2(0.022, 0.022);
    vec3 vbgAmb = 0.20 * (
        vbg +
        texture2D(uTex1, clamp(bgUv + vec2( wrapD.x, 0.0), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(-wrapD.x, 0.0), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(0.0,  wrapD.y), 0.0, 1.0)).rgb +
        texture2D(uTex1, clamp(bgUv + vec2(0.0, -wrapD.y), 0.0, 1.0)).rgb
    );
    float edgeBand = smoothstep(0.04, 0.45, keep) * (1.0 - smoothstep(0.55, 0.96, keep));
    vec3 fgWrapped = mix(fg, mix(max(fg, vbgAmb * 0.92), vbgAmb, 0.20), edgeBand * 0.24);
    gl_FragColor = vec4(mix(color.rgb, mix(vbg, fgWrapped, keep), clamp(uU0.x, 0.0, 1.0)), color.a);
}
)";

// v3.6: Batched 2D Analytical SDF Particle Burst (0=heart, 1=5pt star, 2=diamond, 3=4-ray sparkle)
// Uses kRetouchVert: vSrcCoord=localXY[-1..1], vRetouchAttr=(r, g, b, shapeId * 2.0 + alpha01)
constexpr const char* kParticleBurstFrag = LB_GLSL_PREC R"(
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
float sdHeart(vec2 p) {
    vec2 q = vec2(abs(p.x) * 1.15, p.y * 1.15 + 0.55);
    if (q.y + q.x > 1.0) {
        return length(q - vec2(0.25, 0.75)) - 0.3536;
    }
    vec2 a = q - vec2(0.0, 1.0);
    float h = clamp(0.5 * (q.x + q.y), 0.0, 1.0);
    vec2 b = q - vec2(h, h);
    float d = sqrt(min(dot(a, a), dot(b, b)));
    float s = (q.x - q.y < 0.0) ? -1.0 : 1.0;
    return d * s;
}
float sdStar5(vec2 p) {
    vec2 k1 = vec2(0.809017, -0.587785);
    vec2 k2 = vec2(-0.809017, -0.587785);
    vec2 q = vec2(abs(p.x), p.y) * 1.28;
    q -= 2.0 * max(dot(k1, q), 0.0) * k1;
    q -= 2.0 * max(dot(k2, q), 0.0) * k2;
    q.x = abs(q.x);
    q.y -= 1.0;
    vec2 ba = vec2(0.270381, -0.627852);
    float h = clamp(dot(q, ba) / 0.467304, 0.0, 1.0);
    float s = (q.y * ba.x - q.x * ba.y < 0.0) ? -1.0 : 1.0;
    return length(q - ba * h) * s * 0.78;
}
void main() {
    float shape = floor(vRetouchAttr.w * 0.5 + 0.001);
    float alpha = clamp(vRetouchAttr.w - shape * 2.0, 0.0, 1.0);
    if (alpha <= 0.003) { discard; }
    vec2 p = vSrcCoord;
    float sd = 1.0;
    if (shape < 0.5) {
        sd = sdHeart(p);
    } else if (shape < 1.5) {
        sd = sdStar5(p);
    } else if (shape < 2.5) {
        vec2 a = abs(p);
        sd = (a.x * 1.25 + a.y * 0.95 - 0.72) * 0.64;
    } else {
        vec2 a = sqrt(abs(p) + vec2(0.004));
        sd = (a.x + a.y - 0.94) * 0.85;
    }
    float bodyMask = 1.0 - smoothstep(-0.06, 0.05, sd);
    float halo = exp(-max(sd, 0.0) * 6.5) * 0.36;
    float finalA = clamp((bodyMask + halo) * alpha, 0.0, 1.0);
    if (finalA <= 0.003) { discard; }
    float core = smoothstep(-0.02, -0.32, sd) * 0.55;
    vec3 outRgb = clamp(mix(vRetouchAttr.rgb, vec3(1.0, 0.99, 0.96), core), 0.0, 1.0);
    gl_FragColor = vec4(outRgb, finalA);
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
        case ShaderId::kHairAlpha:       return {kSemanticVert, kHairAlphaFrag};
        case ShaderId::kBgHairMerge:     return {kQuadVert, kBgHairMergeFrag};
        case ShaderId::kBgVirtualMix:    return {kQuadVert, kBgVirtualMixFrag};
        case ShaderId::kStickerQuad:     return {kQuadVert, kStickerQuadFrag};
        case ShaderId::kMatteRefine:     return {kQuadVert, kMatteRefineFrag};
        case ShaderId::kBlemishHeal:     return {kQuadVert, kBlemishHealFrag};
        case ShaderId::kAiResidualBlend: return {kQuadVert, kAiResidualBlendFrag};
        case ShaderId::kHeadDepthShell:  return {kAr3dVert, kHeadDepthShellFrag};
        case ShaderId::kAr3dMesh:        return {kAr3dVert, kAr3dMeshFrag};
        case ShaderId::kHairDye:         return {kQuadVert, kHairDyeFrag};
        case ShaderId::kChromaKey:       return {kQuadVert, kChromaKeyFrag};
        case ShaderId::kParticleBurst:   return {kRetouchVert, kParticleBurstFrag};
        case ShaderId::kPassThrough:
        default:                         return {kQuadVert, kPassThroughFrag};
    }
}

}  // namespace livebeauty::rhi::gles
