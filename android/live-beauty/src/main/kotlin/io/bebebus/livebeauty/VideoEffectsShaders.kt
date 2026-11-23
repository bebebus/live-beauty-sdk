package io.bebebus.livebeauty

/**
 * 美颜全部 GLSL ES 1.00 文本（规格 v1 第 4 节）。纯 Kotlin 常量，不依赖 Android，宿主机门禁直接从本文件提取。
 *
 * 规则：只用 `$NAME` 引用本文件其他片段；不 trimIndent、不在运行期拼接、不写 #version；
 * 浮点字面量都带小数点；同一 uniform 不在两个阶段同时声明；不用 packed/input/output/half/fixed/filter/sample
 * 作标识符；循环上界为常量，uniform 数组只用循环变量或常量下标；着色器文本内只写 ASCII 注释。
 * 片段只定义函数、不读 uniform，测试可以用同一片段拼出调试程序。
 */
internal object VideoEffectsShaders {
    /** 片元着色器固定开头：有 highp 用 highp，否则 mediump；桌面 GLSL 下整段跳过。 */
    const val PRECISION = """#ifdef GL_ES
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
#endif
"""

    /** 合成与原画程序共用的顶点着色器。 */
    const val VERTEX = """attribute vec4 aPosition;
attribute vec2 aTextureCoord;
varying vec2 vTextureCoord;
void main() { gl_Position = aPosition; vTextureCoord = aTextureCoord; }
"""

    /** 原画程序：合成程序编译失败或渲染故障后使用。 */
    const val ORIGINAL = """$PRECISION
uniform sampler2D uSampler;
varying vec2 vTextureCoord;
void main() { gl_FragColor = texture2D(uSampler, vTextureCoord); }
"""

    /** 478 点 2D 三角网格微整顶点着色器（仅 Android，FaceMeshWarpPass）。 */
    const val MESH_WARP_VERTEX = """attribute vec2 aPosition;
attribute vec2 aTextureCoord;
varying vec2 vTextureCoord;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); vTextureCoord = aTextureCoord; }
"""

    /** 478 点 2D 三角网格微整片元着色器（仅 1 个 sampler uniform，1 行采样，FaceMeshWarpPass）。 */
    const val MESH_WARP_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
varying vec2 vTextureCoord;
// Forward-warped 2D face mesh sampling with safe edge clamping.
void main() { gl_FragColor = texture2D(uSampler, clamp(vTextureCoord, 0.0, 1.0)); }
"""

    /** 468 点多边形光栅化皮肤遮罩顶点着色器（仅 Android，SkinMaskPass）。 */
    const val SKIN_MASK_VERTEX = """attribute vec2 aPosition;
attribute vec3 aMaskWeights;
varying vec3 vMaskWeights;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vMaskWeights = aMaskWeights;
}
"""

    /** 468 点多边形光栅化皮肤遮罩片元着色器（输出 R=skin, G=excl, B=hiPlane，零 uniform）。 */
    const val SKIN_MASK_FRAGMENT = """$PRECISION
varying vec3 vMaskWeights;
void main() {
    gl_FragColor = vec4(clamp(vMaskWeights, 0.0, 1.0), 1.0);
}
"""

    /** 实时美妆子网格顶点着色器（仅 Android，MakeupPass）：输出屏幕坐标与规范人脸 UV。 */
    const val MAKEUP_VERTEX = """attribute vec2 aPosition;
attribute vec2 aScreenUv;
attribute vec2 aMakeupUv;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vScreenUv = aScreenUv;
    vMakeupUv = aMakeupUv;
}
"""

    /**
     * 实时美妆通用层片元着色器（腮红 / 修容 / 高光 / 眉妆 / 眼影 / 眼线 / 卧蚕，仅 5 个 uniform）。
     * uMode.x: 0 = SoftLight (blush), 1 = Multiply (contour/brow/liner), 2 = ScreenLift (highlight/aegyo).
     * uMode.y > 0.5: 启用上下分屏局部放大高清眉眼 UV（上半右眉眼 [0.18..0.48, 0.21..0.45]，下半左眉眼 [0.52..0.82, 0.21..0.45]）。
     * uMode.z > 0.5: 启用眼睑皮肤保护门 + 柔粉混合（眼影专用：过滤白眼球巩膜、深色瞳孔/睫毛/眉毛与非肤色背景，避免移动滞后拖出脏残影）。
     */
    const val MAKEUP_LAYER_FRAGMENT = """$PRECISION
uniform sampler2D uBase;
uniform sampler2D uAtlas;
uniform vec4 uColor;
uniform vec4 uChannel;
uniform vec4 uMode;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
// SoftLight / Multiply / ScreenLift blending with optional split-eye UV and eyelid-skin shadow protection.
vec3 softLightBlend(vec3 b, vec3 s) {
    return (1.0 - 2.0 * s) * b * b + 2.0 * s * b;
}
void main() {
    vec2 atlasUv = vMakeupUv;
    if (uMode.y > 0.5) {
        float isLeft = step(0.5, vMakeupUv.x);
        float eu = isLeft < 0.5 ? (vMakeupUv.x - 0.18) / 0.30 : (vMakeupUv.x - 0.52) / 0.30;
        float ev = (vMakeupUv.y - 0.21) / 0.24;
        if (eu <= 0.002 || eu >= 0.998 || ev <= 0.002 || ev >= 0.998) { discard; }
        atlasUv = vec2(eu, isLeft * 0.5 + ev * 0.5);
    }
    float mask = dot(texture2D(uAtlas, atlasUv), uChannel);
    float alpha = clamp(mask * uColor.a, 0.0, 1.0);
    if (alpha <= 0.001) { discard; }
    vec3 base = texture2D(uBase, clamp(vScreenUv, 0.0, 1.0)).rgb;
    if (uMode.z > 0.5) {
        float baseY = dot(base, vec3(0.299, 0.587, 0.114));
        float darkGate = smoothstep(0.14, 0.30, baseY);
        float skinGate = smoothstep(0.025, 0.095, base.r - base.b) * smoothstep(0.015, 0.065, base.r - base.g);
        alpha *= darkGate * skinGate;
        if (alpha <= 0.001) { discard; }
    }
    vec3 result;
    if (uMode.x < 0.5) {
        result = mix(softLightBlend(base, uColor.rgb), base * mix(vec3(1.0), uColor.rgb, 0.68), 0.45);
    } else if (uMode.x < 1.5) {
        vec3 mult = base * mix(vec3(1.0), uColor.rgb, uMode.z > 0.5 ? 0.78 : 0.92);
        result = uMode.z > 0.5 ? mix(mult, softLightBlend(base, uColor.rgb), 0.45) : mult;
    } else {
        result = 1.0 - (1.0 - base) * (1.0 - uColor.rgb * 0.58);
    }
    gl_FragColor = vec4(clamp(result, 0.0, 1.0), alpha);
}
"""

    /**
     * 实时口红 / 唇釉片元着色器（仅在 120 个纯唇带三角形上光栅化，内唇口腔零三角形物理防涂牙）。
     * 采用物理透色柔焦唇模型：
     * 1. 三级像素保护门控：暗缝/口腔/胡茬保护（darkGate）、露齿防染（teethGate）、外缘嘴周黄肤自适应柔化（skinRimGate）；
     * 2. 等比光照透色着色：去除固定亮底，按真实局部明暗 smoothY 等比驱动并保留环境色温与高频唇纹，立体暗面与闭合唇缝自然清晰；
     * 3. 真实镜面水光：湿润基底加深显色 + 紧致中性暖白高光（随真实亮面与唇腹曲率触发，杜绝大面积粉白带）。
     * uLipStyle = (isGlossy, texelX, texelY, 0).
     */
    const val MAKEUP_LIP_FRAGMENT = """$PRECISION
uniform sampler2D uBase;
uniform sampler2D uAtlas;
uniform vec4 uLipColor;
uniform vec4 uLipStyle;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
// Physically-based translucent lip shading with shadow/teeth/skin-rim gates and crisp neutral-white gloss.
void main() {
    float feather = texture2D(uAtlas, vMakeupUv).b;
    if (feather * uLipColor.a <= 0.001) { discard; }
    vec2 uv = clamp(vScreenUv, 0.0, 1.0);
    vec3 base = texture2D(uBase, uv).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    // 1. Adaptive anatomical gates: keep dark lip parting seam / cavity / beard dark, teeth white, and perioral skin clean.
    float darkGate = smoothstep(0.06, 0.20, baseY);
    float redExcess = base.r - max(base.g, base.b);
    float teethLike = smoothstep(0.52, 0.74, baseY) * (1.0 - smoothstep(0.035, 0.115, redExcess));
    float teethGate = 1.0 - teethLike;
    float lipHue = (base.r - base.g) - 0.75 * max(base.g - base.b, 0.0);
    float rimZone = 1.0 - smoothstep(0.35, 0.92, feather);
    float skinRimGate = mix(1.0, smoothstep(0.045, 0.155, lipHue), rimZone * 0.92);
    float effFeather = feather * darkGate * teethGate * skinRimGate;
    float alpha = clamp(effFeather * uLipColor.a, 0.0, 0.90);
    if (alpha <= 0.001) { discard; }
    // 2. Local 4-tap smooth luma to isolate real vertical lip creases without washing out 3D curvature.
    float smoothY = baseY;
    if (uLipStyle.y > 0.0 && uLipStyle.z > 0.0) {
        vec2 stepUv = uLipStyle.yz * 2.0;
        vec3 avg = 0.25 * (
            texture2D(uBase, clamp(uv + vec2(-stepUv.x, -stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uBase, clamp(uv + vec2(stepUv.x, -stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uBase, clamp(uv + vec2(-stepUv.x, stepUv.y), 0.0, 1.0)).rgb +
            texture2D(uBase, clamp(uv + vec2(stepUv.x, stepUv.y), 0.0, 1.0)).rgb
        );
        smoothY = dot(avg, vec3(0.299, 0.587, 0.114));
    }
    float lipDetail = clamp(baseY - smoothY, -0.09, 0.09);
    // 3. Proportional illumination shading (zero flat shadow floor) + midtone-only bare-lip de-dulling.
    float midLift = 1.0 + 0.22 * smoothstep(0.16, 0.30, smoothY) * (1.0 - smoothstep(0.42, 0.66, smoothY));
    float illum = (smoothY / 0.44) * midLift;
    float illumSoft = illum / (1.0 + 0.24 * max(illum - 0.85, 0.0));
    float cY = max(dot(uLipColor.rgb, vec3(0.299, 0.587, 0.114)), 0.15);
    vec3 normC = clamp(uLipColor.rgb / max(cY * 1.12, 0.34), 0.12, 1.45);
    vec3 multLip = clamp(base * mix(vec3(1.0), normC, 0.88), 0.0, 1.0);
    float isGlossy = step(0.5, uLipStyle.x);
    vec3 pigmentLip = clamp(uLipColor.rgb * illumSoft * mix(1.0, 0.90, isGlossy), 0.0, 1.0);
    float coverage = mix(0.48, 0.74, clamp(uLipColor.a, 0.0, 1.0));
    vec3 result = mix(multLip, pigmentLip, coverage);
    result = clamp(result + vec3(lipDetail * mix(1.15, 0.75, isGlossy)), 0.0, 1.0);
    // 4. Wet clearcoat specular: tight neutral warm-white reflection driven by real surface brightness and lip creases.
    if (isGlossy > 0.5) {
        float lumaDrive = smoothstep(0.24, 0.58, baseY);
        float centerBelly = 1.0 - smoothstep(0.02, 0.095, abs(vMakeupUv.x - 0.5));
        float ridgeMask = clamp(0.80 + lipDetail * 5.0, 0.20, 1.40);
        float dyLo = (vMakeupUv.y - 0.714) * 100.0;
        float dyHi = (vMakeupUv.y - 0.673) * 120.0;
        float lowerBand = exp(-(dyLo * dyLo));
        float upperBand = exp(-(dyHi * dyHi)) * 0.42;
        float naturalHot = smoothstep(0.48, 0.80, baseY) * 0.42;
        float spec = clamp(((lowerBand + upperBand) * centerBelly * lumaDrive * 0.78 + naturalHot) * ridgeMask * effFeather, 0.0, 0.78);
        vec3 specCol = vec3(1.0, 0.99, 0.97);
        result = clamp(result + (specCol - result) * (spec * 0.65), 0.0, 1.0);
    }
    gl_FragColor = vec4(clamp(result, 0.0, 1.0), alpha);
}
"""

    /**
     * 实时美瞳 + 眼神光片元着色器（仅在 14+14 个眼裂内部三角形内光栅化，物理眼皮裁切 + 眨眼门控）。
     * uIris = (cx, cy, radiusMetric, aspect), uIrisTile = (tileX, tileY, eyeOpenness, 0),
     * uIrisStrength = (lensStrength, catchlightStrength, 0, 0).
     */
    const val MAKEUP_IRIS_FRAGMENT = """$PRECISION
uniform sampler2D uBase;
uniform sampler2D uAtlas;
uniform vec4 uIris;
uniform vec4 uIrisTile;
uniform vec4 uIrisStrength;
varying vec2 vScreenUv;
varying vec2 vMakeupUv;
// Eyelid-clipped contact lens with blink gate and twin studio catchlights.
void main() {
    float blink = smoothstep(0.10, 0.22, uIrisTile.z);
    if (blink <= 0.001) { discard; }
    float aspect = max(uIris.w, 0.001);
    vec2 m = vec2(vScreenUv.x, vScreenUv.y / aspect);
    vec2 c = vec2(uIris.x, uIris.y / aspect);
    vec2 d = (m - c) / max(uIris.z, 0.0001);
    float r = length(d);
    if (r >= 1.0) { discard; }
    vec3 base = texture2D(uBase, clamp(vScreenUv, 0.0, 1.0)).rgb;
    float baseY = dot(base, vec3(0.299, 0.587, 0.114));
    vec2 localUv = clamp(d * 0.5 + 0.5, 0.005, 0.995);
    vec2 atlasUv = vec2(uIrisTile.x + localUv.x * 0.5, uIrisTile.y + (1.0 - localUv.y) * 0.5);
    vec4 lens = texture2D(uAtlas, atlasUv);
    float lensA = lens.a * uIrisStrength.x * blink;
    // Preserve upper-eyelid shadow over the top of the iris so lenses blend realistically.
    float lidShadow = mix(1.0, 0.65, smoothstep(0.25, 0.95, d.y));
    vec3 lensRgb = clamp(lens.rgb * mix(0.45, 1.15, clamp(baseY * 2.4, 0.0, 1.0)) * lidShadow, 0.0, 1.0);
    vec3 outRgb = mix(base, lensRgb, lensA);
    float clA = 0.0;
    if (uIrisStrength.y > 0.0) {
        vec2 p0 = d - vec2(-0.26, 0.28);
        vec2 p1 = d - vec2(0.22, -0.18);
        float spot = exp(-dot(p0, p0) * 32.0) + 0.45 * exp(-dot(p1, p1) * 58.0);
        clA = clamp(spot * uIrisStrength.y * blink, 0.0, 0.92);
        outRgb = mix(outRgb, vec3(1.0, 0.99, 0.97), clA);
    }
    float finalA = clamp(max(lensA, clA), 0.0, 1.0);
    if (finalA <= 0.001) { discard; }
    gl_FragColor = vec4(outRgb, finalA);
}
"""

    /**
     * 更多脸型与局部修饰（仅 Android）：下巴、额头、瘦鼻（变形），去黑眼圈、亮眼、白牙（调色）。单独一道全分辨率绘制
     * （合成程序的 uniform 已满），在美体之前。uChin / uForehead / uNose = (中心 x, 中心 y（纹理坐标）, 半径（度量）, 强度)；
     * uEye0 / uEye1 / uMouth = (中心 x, 中心 y（纹理坐标）, 横半径, 竖半径（度量）)；uRetouch = (黑眼圈, 亮眼, 白牙, 法令纹)；
     * uFaceAxis = (u.x, u.y, aspect, 未用)，u 为横过脸的度量单位向量，v = (-u.y, u.x) 指向额头。
     * 先变形取色，再在取色位置（即原帧人脸几何）上调色；全部强度为 0 时严格直通。见 FaceShapePass。
     */
    const val FACE_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform vec4 uChin;
uniform vec4 uForehead;
uniform vec4 uNose;
uniform vec4 uEye0;
uniform vec4 uEye1;
uniform vec4 uMouth;
uniform vec4 uRetouch;
uniform vec4 uFaceAxis;
varying vec2 vTextureCoord;
// Local inverse warps in metric space m = (x, y/aspect); falloff (1 - r^2)^2 inside each radius.
float falloff(vec2 m, vec4 p, float aspect) {
    float r = length(m - vec2(p.x, p.y / aspect)) / max(p.z, 0.0001);
    return r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
}
// Rotated ellipse weight (1 - r^2)^2; e = (centre xy in texture space, radius along u, radius along v), shifted along v.
float ellipseWeight(vec2 m, vec4 e, vec2 u, vec2 v, float aspect, float shift, vec2 scale) {
    vec2 d = m - (vec2(e.x, e.y / aspect) + v * shift);
    float r = length(vec2(dot(d, u) / max(e.z * scale.x, 0.0001), dot(d, v) / max(e.w * scale.y, 0.0001)));
    return r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
}
// Distance from m to segment ab, normalised so the capsule edge is 1.
float capsule(vec2 m, vec2 a, vec2 b, float radius) {
    vec2 ab = b - a;
    float t = clamp(dot(m - a, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
    return length(m - (a + ab * t)) / max(radius, 0.0001);
}
// Nasolabial folds: from just outside each nose wing to just outside each mouth corner. Inside the capsule, pixels darker
// than the skin on either side (sampled across the fold) are pulled up towards it; brighter pixels are left alone.
vec3 softenFolds(vec3 c, vec2 m, vec2 u, vec2 v, float aspect) {
    vec2 nose = vec2(uNose.x, uNose.y / aspect);
    vec2 mouth = vec2(uMouth.x, uMouth.y / aspect);
    float width = 0.35 * uNose.z;
    vec2 a0 = nose - u * (1.05 * uNose.z);
    vec2 b0 = mouth - u * (uMouth.z / 1.2 + 0.15 * uMouth.z);
    vec2 a1 = nose + u * (1.05 * uNose.z);
    vec2 b1 = mouth + u * (uMouth.z / 1.2 + 0.15 * uMouth.z);
    float r0 = capsule(m, a0, b0, width);
    float r1 = capsule(m, a1, b1, width);
    float r = min(r0, r1);
    if (r >= 1.0) return c;
    vec2 along = r0 < r1 ? normalize(b0 - a0) : normalize(b1 - a1);
    vec2 across = vec2(-along.y, along.x) * (1.4 * width);
    vec3 left = texture2D(uSampler, clamp(vec2(m.x + across.x, (m.y + across.y) * aspect), 0.0, 1.0)).rgb;
    vec3 right = texture2D(uSampler, clamp(vec2(m.x - across.x, (m.y - across.y) * aspect), 0.0, 1.0)).rgb;
    vec3 skin = 0.5 * (left + right);
    float f = (1.0 - r * r) * (1.0 - r * r);
    return mix(c, max(c, skin), uRetouch.w * f * 0.8);
}
vec3 retouch(vec3 c, vec2 m, vec2 u, vec2 v, float aspect) {
    if (uRetouch.w > 0.0) c = softenFolds(c, m, u, v, aspect);
    // Dark circles: lift shadows in a band just below each eye.
    if (uRetouch.x > 0.0) {
        float f = max(ellipseWeight(m, uEye0, u, v, aspect, -1.6 * uEye0.w, vec2(0.9, 1.1)),
                      ellipseWeight(m, uEye1, u, v, aspect, -1.6 * uEye1.w, vec2(0.9, 1.1)));
        c += uRetouch.x * f * (1.0 - c) * 0.18;
    }
    // Bright eyes: a little more contrast and light inside the eye openings.
    if (uRetouch.y > 0.0) {
        float f = max(ellipseWeight(m, uEye0, u, v, aspect, 0.0, vec2(1.0)),
                      ellipseWeight(m, uEye1, u, v, aspect, 0.0, vec2(1.0)));
        c = mix(c, clamp((c - 0.5) * 1.15 + 0.53, 0.0, 1.0), uRetouch.y * f * 0.6);
    }
    // Teeth: only bright, low-saturation pixels inside the mouth (lips are saturated and stay untouched).
    if (uRetouch.z > 0.0) {
        float f = ellipseWeight(m, uMouth, u, v, aspect, 0.0, vec2(1.0));
        float luma = dot(c, vec3(0.299, 0.587, 0.114));
        float chroma = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
        float teeth = smoothstep(0.35, 0.55, luma) * (1.0 - smoothstep(0.18, 0.35, chroma));
        c = mix(c, vec3(clamp(luma * 1.08 + 0.02, 0.0, 1.0)), uRetouch.z * f * teeth * 0.7);
    }
    return c;
}
void main() {
    if (uChin.w <= 0.0 && uForehead.w <= 0.0 && uNose.w <= 0.0 && max(max(uRetouch.x, uRetouch.y), max(uRetouch.z, uRetouch.w)) <= 0.0) {
        gl_FragColor = texture2D(uSampler, vTextureCoord);
        return;
    }
    float aspect = uFaceAxis.z;
    vec2 u = uFaceAxis.xy;
    vec2 v = vec2(-u.y, u.x);
    vec2 m = vec2(vTextureCoord.x, vTextureCoord.y / aspect);
    // Chin: sample from above so the chin reaches further down. Forehead: sample from below so it rises.
    m += v * (uChin.w * uChin.z * falloff(m, uChin, aspect));
    m -= v * (uForehead.w * uForehead.z * falloff(m, uForehead, aspect));
    // Nose: sample further out sideways inside an ellipse taller than wide, so the wings look narrower.
    if (uNose.w > 0.0) {
        vec2 c = vec2(uNose.x, uNose.y / aspect);
        vec2 d = m - c;
        float du = dot(d, u);
        float dv = dot(d, v);
        float r = length(vec2(du, dv / 1.3)) / max(uNose.z, 0.0001);
        float f = r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
        m = c + u * (du * (1.0 + uNose.w * f)) + v * dv;
    }
    vec4 color = texture2D(uSampler, clamp(vec2(m.x, m.y * aspect), 0.0, 1.0));
    gl_FragColor = vec4(retouch(color.rgb, m, u, v, aspect), color.a);
}
"""

    /** 468 点解剖子网格局部精修顶点着色器（仅 Android，FaceRetouchMeshPass）：原位叠加在 FaceMeshWarpPass FBO 上，零额外整屏拷贝。 */
    const val RETOUCH_VERTEX = """attribute vec2 aPosition;
attribute vec2 aSrcCoord;
attribute vec4 aRetouchAttr;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vSrcCoord = aSrcCoord;
    vRetouchAttr = aRetouchAttr;
}
"""

    /**
     * 468 点解剖子网格泪沟去黑眼圈（精准避开卧蚕带）+ 弧形鼻唇沟淡法令纹片元着色器（仅 Android，FaceRetouchMeshPass）。
     * vRetouchAttr = (tearWeight, foldWeight, sampleOffsetX, sampleOffsetY)；uRetouchStrength = (darkCircles, nasolabial, 0, 0)。
     */
    const val RETOUCH_SKIN_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform vec4 uRetouchStrength;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    float wTear = clamp(vRetouchAttr.x * uRetouchStrength.x * 0.85, 0.0, 1.0);
    float wFold = clamp(vRetouchAttr.y * uRetouchStrength.y * 0.82, 0.0, 1.0);
    float alpha = clamp(wTear + wFold, 0.0, 1.0);
    if (alpha <= 0.002) { discard; }
    vec3 c = texture2D(uSampler, vSrcCoord).rgb;
    vec2 off = vRetouchAttr.zw;
    vec3 target = c;
    if (wTear > wFold) {
        vec3 cheekRef = texture2D(uSampler, clamp(vSrcCoord + off, 0.0, 1.0)).rgb;
        float yBase = dot(c, vec3(0.2126, 0.7152, 0.0722));
        float yRef = dot(cheekRef, vec3(0.2126, 0.7152, 0.0722));
        float deficit = clamp((yRef - yBase) * 2.4, 0.0, 0.22);
        float liftMask = clamp(deficit * 3.8, 0.0, 0.75);
        target = clamp(c + (cheekRef - c) * liftMask + vec3(0.058, 0.046, 0.032) * liftMask * (1.0 - yBase), 0.0, 1.0);
    } else {
        vec3 sideA = texture2D(uSampler, clamp(vSrcCoord + off, 0.0, 1.0)).rgb;
        vec3 sideB = texture2D(uSampler, clamp(vSrcCoord - off, 0.0, 1.0)).rgb;
        target = max(c, 0.5 * (sideA + sideB));
    }
    gl_FragColor = vec4(target, alpha);
}
"""

    /**
     * 468 点解剖子网格眼裂精准亮眼（28 个眼裂内部三角形）+ 内唇口腔物理防染唇白牙（18 个内唇孔洞三角形）。
     * uRetouchMode = (strength, isTeethMode, 0, 0)。
     */
    const val RETOUCH_EYE_TEETH_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform vec4 uRetouchMode;
varying vec2 vSrcCoord;
varying vec4 vRetouchAttr;
void main() {
    float strength = clamp(uRetouchMode.x, 0.0, 1.0);
    if (strength <= 0.002) { discard; }
    vec3 c = texture2D(uSampler, vSrcCoord).rgb;
    float y = dot(c, vec3(0.299, 0.587, 0.114));
    float chroma = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    if (uRetouchMode.y < 0.5) {
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
"""

    /**
     * 背景虚化（仅 Android）的预乘降采样：将前景人像置零（rgb * bg, bg），后续高斯模糊只对背景区域传播，
     * 彻底消除头发、肩膀向模糊背景渗色造成的深色残影光晕。见 BackgroundBlurPass。
     */
    const val BG_PREMASK_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform sampler2D uMask;
uniform vec4 uBackground;
varying vec2 vTextureCoord;
// Premultiply background by (1 - personMask) so dark hair/shoulders never bleed into blurred bokeh.
void main() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  float person = texture2D(uMask, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
  float bg = 1.0 - smoothstep(uBackground.y, uBackground.z, person);
  gl_FragColor = vec4(color * bg, bg);
}
"""

    /**
     * 背景虚化（仅 Android）的模糊：在 1/4 分辨率上沿 uStep 方向做 9 次线性采样等效的 RGBA 高斯模糊，水平、竖直各两遍。
     * 同步模糊预乘后的颜色与背景权重 alpha，供 BG_MIX_FRAGMENT 做归一化还原。见 BackgroundBlurPass。
     */
    const val BG_BLUR_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform vec2 uStep;
varying vec2 vTextureCoord;
// 9-tap RGBA Gaussian folded into 5 bilinear fetches along uStep (preserves premultiplied background weight).
void main() {
  vec4 sum = texture2D(uSampler, vTextureCoord) * 0.2270270;
  sum += (texture2D(uSampler, vTextureCoord + uStep * 1.3846154) + texture2D(uSampler, vTextureCoord - uStep * 1.3846154)) * 0.3162162;
  sum += (texture2D(uSampler, vTextureCoord + uStep * 3.2307692) + texture2D(uSampler, vTextureCoord - uStep * 3.2307692)) * 0.0702703;
  gl_FragColor = sum;
}
"""

    /**
     * LUT 查表的坐标换算（仅 Android，纯函数片段，FRAGMENT_PRO 引用）。uLut 为 512x512 的 64 级色彩立方（8x8 格，每格 64x64，
     * 格内 x = 红、y = 绿，格序 = 蓝；位图第 0 行在纹理 t = 0），蓝色在相邻两格之间线性插值。
     */
    const val LUT_SAMPLE = """
// Standard 64^3 lookup laid out as 8x8 tiles of 64x64 in a 512x512 texture.
vec2 lutCell(float level, vec2 rg) {
  float row = floor(level / 8.0);
  float column = level - row * 8.0;
  return vec2(column, row) * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * rg;
}
"""

    /**
     * 暗光时域降噪（仅 Android）：磨皮之前单独一道全分辨率绘制，把当前帧与上一帧降噪结果按逐像素差异混合。
     * 差异小（静止、只有噪点）时最多混入 uDenoise.x 的历史，差异大（运动）时直通，避免拖影。
     * uDenoise = (最大历史权重, 差异下限, 差异上限, 未用)；权重为 0 时严格直通。见 TemporalDenoisePass。
     */
    const val DENOISE_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform sampler2D uHistory;
uniform vec4 uDenoise;
varying vec2 vTextureCoord;
// Recursive temporal filter: static noisy pixels lean on history, moving pixels pass through.
void main() {
  vec4 current = texture2D(uSampler, vTextureCoord);
  vec3 previous = texture2D(uHistory, vTextureCoord).rgb;
  vec3 delta = abs(current.rgb - previous);
  float diff = max(max(delta.r, delta.g), delta.b);
  float weight = uDenoise.x * (1.0 - smoothstep(uDenoise.y, uDenoise.z, diff));
  gl_FragColor = vec4(mix(current.rgb, previous, weight), current.a);
}
"""

    /**
     * 美体（仅 Android）：在合成之前单独一道全分辨率绘制，把相机帧按人体几何变形到中间纹理，合成程序再读它，
     * 合成程序的 uniform 数不变。uBodyA = (胯中点.xy, 身体方向 × (1 + 长腿强度))，
     * uBodyB = (躯干长, 半宽, 细腰强度, 丰胸强度)，uBodyC = (瘦肩强度, 天鹅颈强度, 未用, 未用)，uAspect = 宽 / 高；见 BodyWarpPass。
     */
    const val BODY_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform vec4 uBodyA;
uniform vec4 uBodyB;
uniform vec4 uBodyC;
uniform float uAspect;
varying vec2 vTextureCoord;
// Body shaping (inverse map in metric space m = (x, y/aspect)): a = (hip mid xy, body axis * (1 + leg strength)),
// b = (torso length, half width, waist strength, bust strength), c = (shoulder strength, neck strength, unused, unused);
// |a.zw| < 0.5 means no body. Order: legs, neck, waist, shoulders, bust.
vec2 bodyWarp(vec2 uv, vec4 a, vec4 b, vec4 c, float aspect) {
    float axisLength = length(a.zw);
    if (axisLength < 0.5) return uv;
    vec2 axis = a.zw / axisLength;
    float legs = axisLength - 1.0;
    vec2 side = vec2(-axis.y, axis.x);
    vec2 hip = vec2(a.x, a.y / aspect);
    vec2 d = vec2(uv.x, uv.y / aspect) - hip;
    float t = dot(d, axis);
    float s = dot(d, side);
    float torso = b.x;
    float width = b.y;
    // Legs: stretch below the hips within a horizontal body corridor so side floors/walls stay untouched.
    if (legs > 0.0 && t > 0.0) {
        float legKeep = 1.0 - smoothstep(1.15 * width, 1.85 * width, abs(s));
        t = t / (1.0 + legs * legKeep * smoothstep(0.0, 0.15 * torso, t));
    }
    // Neck: stretch a band of 0.3 torso above the shoulder line and lift everything above it (the head moves up,
    // it is not stretched); fades out sideways between 1.2 and 2.2 half widths so the background does not tear.
    if (c.y > 0.0) {
        float u = -(t + torso);
        if (u > 0.0) {
            float band = 0.3 * torso;
            float lifted = u <= band * (1.0 + c.y) ? u / (1.0 + c.y) : u - band * c.y;
            float keep = 1.0 - smoothstep(1.2 * width, 2.2 * width, abs(s));
            t = -torso - mix(u, lifted, keep);
        }
    }
    // Waist: sample further out around the waist line (0.3 torso above the hips); compact 1.65 half-width corridor.
    if (b.z > 0.0) {
        float tw = (t + 0.3 * torso) / (0.22 * torso);
        float r = abs(s) / (1.65 * width);
        float f = r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
        s *= 1.0 + b.z * exp(-0.5 * tw * tw) * f;
    }
    // Shoulders: sample further out around the shoulder line (0.05 torso below the shoulder joints); 1.60 half-widths.
    if (c.x > 0.0) {
        float ts = (t + 0.95 * torso) / (0.18 * torso);
        float r = abs(s) / (1.60 * width);
        float f = r < 1.0 ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
        s *= 1.0 + c.x * exp(-0.5 * ts * ts) * f;
    }
    vec2 m = hip + axis * t + side * s;
    // Bust: local magnify (same as eye enlarge) around two centres 0.65 torso above the hips, 0.45 half widths apart.
    if (b.w > 0.0) {
        for (int i = 0; i < 2; i++) {
            vec2 c = hip - axis * (0.65 * torso) + side * ((float(i) * 2.0 - 1.0) * 0.45 * width);
            float ratio = length(m - c) / max(0.42 * width, 0.001);
            m = c + (m - c) * clamp(1.0 - (1.0 - ratio * ratio) * b.w, 0.0, 1.0);
        }
    }
    return vec2(m.x, m.y * aspect);
}
void main() { gl_FragColor = texture2D(uSampler, clamp(bodyWarp(vTextureCoord, uBodyA, uBodyB, uBodyC, uAspect), 0.0, 1.0)); }
"""

    /**
     * COMMON：亮度、肤色门限（既有 YCbCr 平滑门限，常量不变，是颜色区域估计而不是人脸检测）、
     * 端点不动的提亮曲线、遮罩椭圆距离 E（第 2 节，商先截到 ±4 再平方，mediump 也不溢出）。
     */
    const val COMMON = """
float luma(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }
float skinWeight(vec3 color) {
    float cb = 0.5 + dot(color, vec3(-0.168736, -0.331264, 0.5));
    float cr = 0.5 + dot(color, vec3(0.5, -0.418688, -0.081312));
    return smoothstep(0.26, 0.34, cb) * (1.0 - smoothstep(0.48, 0.56, cb)) *
        smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.72, 0.80, cr));
}
vec3 lift(vec3 x, float a) { return x * (1.0 + a) / (1.0 + a * x); }
float ellipseE(vec2 uv, vec4 c, vec2 u, float aspect) {
    vec2 d = vec2(uv.x - c.x, (uv.y - c.y) / aspect);
    float a = clamp(dot(d, u) / max(c.z, 0.0001), -4.0, 4.0);
    float b = clamp(dot(d, vec2(-u.y, u.x)) / max(c.w, 0.0001), -4.0, 4.0);
    return a * a + b * b;
}
"""

    /** WARP：既有脸颊收缩与大眼逆向变形，公式不变；strength 已乘 presence 与 warpWeight。 */
    const val WARP = """
// Adapted from GPUPixel v1.3.1 FaceReshapeFilter (Apache-2.0, PixPark).
// Changed to MediaPipe points, four bounded cheek warps, combined color pass.
float faceDistance(vec2 a, vec2 b, float aspect) {
    return length((a - b) * vec2(1.0, 1.0 / aspect));
}
vec2 cheek(vec2 uv, vec4 pair, float strength, float aspect) {
    vec2 origin = pair.xy;
    vec2 target = pair.zw;
    float radius = max(faceDistance(origin, target, aspect), 0.001);
    float weight = clamp(1.0 - faceDistance(uv, origin, aspect) / radius, 0.0, 1.0);
    return uv - (target - origin) * strength * weight;
}
vec2 eye(vec2 uv, vec3 centre, float strength, float aspect) {
    float ratio = faceDistance(uv, centre.xy, aspect) / max(centre.z, 0.001);
    float weight = clamp(1.0 - (1.0 - ratio * ratio) * strength, 0.0, 1.0);
    return centre.xy + (uv - centre.xy) * weight;
}
"""

    /** MASKS：采样坐标处的 (face, excl)；presence == 0 时不计算，直接为 0。依赖 COMMON。 */
    const val MASKS = """
vec2 faceMasks(vec2 uv, float aspect, float presence, vec4 oval, vec2 axis, vec4 mouth, vec4 eye0, vec4 eye1) {
    if (presence <= 0.0) return vec2(0.0);
    float inside = 1.0 - smoothstep(0.50, 1.0, ellipseE(uv, oval, axis, aspect));
    float e0 = 1.0 - smoothstep(0.55, 1.0, ellipseE(uv, eye0, axis, aspect));
    float e1 = 1.0 - smoothstep(0.55, 1.0, ellipseE(uv, eye1, axis, aspect));
    float lips = 1.0 - smoothstep(0.65, 1.0, ellipseE(uv, mouth, axis, aspect));
    float keep = (1.0 - e0) * (1.0 - e1) * (1.0 - lips);
    return presence * inside * vec2(keep, 1.0 - keep);
}
"""

    /**
     * COMPOSITE：规格 4.2 的 beautyColor()，假定 processing 为真。依赖 COMMON。
     * params = (processing, smoothing, brightness, style)，details = (skinOnly, sharpness, saturation, filterIntensity)，
     * tone = (whiten, rosy, 塑形强度…)；顺序：磨皮 → 锐化 → clamp → 美白 → 红润 → 提亮 → 饱和度 → 滤镜 → clamp。
     * 方差自适应混合移植自 GPUPixel BeautyFaceUnitFilter（Apache-2.0）。滤镜编号与公式沿用既有着色器。
     */
    const val COMPOSITE = """
vec3 grade(vec3 color, float style) {
    vec3 graded = color;
    if (style > 4.5) graded += vec3(0.045, -0.005, 0.025);
    else if (style > 3.5) graded = vec3(
        dot(color, vec3(0.393, 0.769, 0.189)),
        dot(color, vec3(0.349, 0.686, 0.168)),
        dot(color, vec3(0.272, 0.534, 0.131)));
    else if (style > 2.5) graded = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    else if (style > 1.5) graded += vec3(-0.025, 0.0, 0.045);
    else if (style > 0.5) graded += vec3(0.045, 0.008, -0.025);
    return graded;
}
vec3 beautyColor(vec3 base, vec3 mean, float variance, vec3 around, float face, float excl,
        vec4 params, vec4 details, vec4 tone) {
    float gate = skinWeight(base);
    float skin = face * mix(0.85, 1.0, gate) + (1.0 - face) * gate * (1.0 - 0.55 * excl);
    vec3 color = base;
    float k = 0.0;
    if (params.y > 0.0) {
        float r = variance / 0.15;
        k = params.y * 0.9 * (details.x > 0.5 ? skin : 1.0) / (1.0 + r * r);
        color = mix(base, mean, k);
    }
    if (details.y > 0.0) {
        float detail = luma(base - around);
        detail *= smoothstep(0.008, 0.03, abs(detail));
        color += clamp(detail, -0.1, 0.1) * details.y * 0.8 * (1.0 - 0.6 * skin) * (1.0 - k);
    }
    color = clamp(color, 0.0, 1.0);
    if (tone.x > 0.0) {
        float y = clamp(luma(color), 0.0, 1.0);
        vec3 white = min(color * (1.0 + 0.5 * tone.x * (1.0 - mix(color, vec3(y), 0.5))), 1.0);
        color = mix(color, white, mix(0.15, 1.0, skin));
    }
    if (tone.y > 0.0) {
        float shift = 0.03 * tone.y * skin;
        color += vec3(1.402 * shift, -0.714 * shift, 0.0);
    }
    if (params.z > 0.0) color = lift(color, 0.5 * params.z);
    color = mix(vec3(luma(color)), color, details.z);
    color = mix(color, clamp(grade(color, params.w), 0.0, 1.0), details.w);
    return clamp(color, 0.0, 1.0);
}
"""

    /**
     * 全分辨率合成程序（14 个非采样器 vec4 + 2 个采样器 = 16，正好是 GLES2 片元着色器的最低保证；不要再加，美体另走 BODY_FRAGMENT），由 BaseFilterRender.draw() 自己的 glDrawArrays 绘制。
     * mean/var、四邻域与遮罩都在逆向变形后的坐标处取；不处理时按原坐标严格直通。
     */
    const val FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform sampler2D uBlur;
uniform vec4 uFrame;
uniform vec4 uParams;
uniform vec4 uDetails;
uniform vec4 uTone;
uniform vec4 uCheeks[4];
uniform vec4 uEyes[2];
uniform vec4 uOval;
uniform vec4 uMouth;
uniform vec4 uEyeMask[2];
varying vec2 vTextureCoord;
$COMMON
$WARP
$MASKS
$COMPOSITE
vec2 warpCoordinate(vec2 uv) {
    if (uTone.z > 0.0) {
        for (int i = 0; i < 4; i++) uv = cheek(uv, uCheeks[i], uTone.z, uFrame.z);
    }
    if (uTone.w > 0.0) {
        for (int i = 0; i < 2; i++) uv = eye(uv, uEyes[i].xyz, uTone.w, uFrame.z);
    }
    return clamp(uv, 0.0, 1.0);
}
void main() {
    if (uParams.x < 0.5) { gl_FragColor = texture2D(uSampler, vTextureCoord); return; }
    vec2 coordinate = warpCoordinate(vTextureCoord);
    vec4 original = texture2D(uSampler, coordinate);
    vec3 mean = original.rgb;
    float variance = 0.0;
    if (uParams.y > 0.0) {
        vec4 blurred = texture2D(uBlur, coordinate);
        mean = blurred.rgb;
        variance = blurred.a;
    }
    vec3 around = original.rgb;
    if (uDetails.y > 0.0) {
        around = 0.25 * (texture2D(uSampler, coordinate + vec2(uFrame.x, 0.0)).rgb +
            texture2D(uSampler, coordinate - vec2(uFrame.x, 0.0)).rgb +
            texture2D(uSampler, coordinate + vec2(0.0, uFrame.y)).rgb +
            texture2D(uSampler, coordinate - vec2(0.0, uFrame.y)).rgb);
    }
    vec2 masks = faceMasks(coordinate, uFrame.z, uFrame.w, uOval, vec2(uEyes[0].w, uEyes[1].w),
        uMouth, uEyeMask[0], uEyeMask[1]);
    gl_FragColor = vec4(beautyColor(original.rgb, mean, variance, around, masks.x, masks.y,
        uParams, uDetails, uTone), original.a);
}
"""

    /**
     * COMPOSITE_PRO：Android 主合成的颜色链（v5），依赖 COMMON。
     * - 磨皮：导向滤波在全分辨率重构 `guided = base * guidedA + guidedB`，再按肤色、五官保护与 RGB 距离护栏混合；
     * - 质感与锐化共用一个带通 `band = luma(fineLow - wideLow)`（±0.5 与 ±1.5 texel 的四角均值之差，4px 周期响应最大、
     *   2px 噪声为 0），按 coring 门限 grade.y 去掉噪声级起伏；锐化在五官全量、背景 0.35、平滑后的脸部皮肤约 0.05。
     *   锐化压低只乘在脸内（region 乘 face）：脸外被肤色门误判成皮肤的暖色头发、衣物按背景级 0.35 锐化，发丝纹理不被磨平；
     * - 胡须与胡须边界门 fineDark：1) 比 2px 低通暗出 0.02–0.06 的细暗丝且局部纹理密度 guidedA 在 0.25–0.55 以上（胡茬、胡须区）时
     *   不磨皮、不调色，孤立黑头/雀斑照常磨掉；2) bleedGuard 仅在暗调区（luma < 0.30）且导向均值比局部低通显著偏亮（+0.042..+0.105）
     *   的亮肤-暗须交界处平滑阻断亮肤色向胡须内侧晕染，配合深暗部美白滚降消除「胡须粘着一层皮肤」，同时保留脸内 0.30 保底避免冷光肤色斑驳。
     *   只乘进 k 与 w，不乘锐化 region（避免胡须、刘海过锐）；
     * - 头发遮罩 nonHair（FRAGMENT_PRO 由 uSkinMask.a 换算，1 = 非头发）乘进 skin，经 skin 同时作用于磨皮 k、锐化 region 与
     *   调色 w：头发处 skin = 0，不磨皮、不调色，只按背景级 0.35 锐化。nonHair = 1 时与不接头发遮罩逐字节相同；
     * - 调色只作用于肤色权重 w = skin * featureKeep * fineDark，非肤色像素逐字节直通：去黄（只拉偏黄的色相，G 最多 -0.02）→
     *   亮度提亮曲线（美白 0.60 + 提亮 0.40）→ 仅脸部的亮度 SoftLight 立体光（hiPlane 驱动）→ RGB 按亮度比例缩放 →
     *   中间调色度提升（红润）→ 保亮度的色域收敛；之后是全局饱和度与旧常量滤镜 gradePro（加性偏移乘 4Y(1-Y)，黑白场不变）。
     * params = (style + 16 * skinOnly, smoothing, brightness, skinTexture)，details = (radiance, sharpness, saturation, filterIntensity)，
     * tone = (whiten, rosy, 未用, 未用)，grade = (LUT 强度, coring 门限（不低于 0.005，即 BeautyQualityTuning.CORE_BASE）,
     * 回退预处理标记, 未用)。
     */
    const val COMPOSITE_PRO = """
// Legacy constant filters; additive tints are weighted by 4Y(1-Y) so black and white stay untouched.
vec3 gradePro(vec3 color, float style) {
    float y = luma(color);
    float bell = 4.0 * y * (1.0 - y);
    vec3 graded = color;
    if (style > 4.5) graded += vec3(0.045, -0.005, 0.025) * bell;
    else if (style > 3.5) graded = vec3(
        dot(color, vec3(0.393, 0.769, 0.189)),
        dot(color, vec3(0.349, 0.686, 0.168)),
        dot(color, vec3(0.272, 0.534, 0.131)));
    else if (style > 2.5) graded = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    else if (style > 1.5) graded += vec3(-0.025, 0.0, 0.045) * bell;
    else if (style > 0.5) graded += vec3(0.045, 0.008, -0.025) * bell;
    return graded;
}
// Pull chroma towards luma y just enough to fit [0, 1]; luma is unchanged.
vec3 fitGamut(vec3 c, float y) {
    float hi = max(max(c.r, c.g), c.b);
    float lo = min(min(c.r, c.g), c.b);
    float s = 1.0;
    if (hi > 1.0) s = min(s, (1.0 - y) / max(hi - y, 0.0001));
    if (lo < 0.0) s = min(s, y / max(y - lo, 0.0001));
    return vec3(y) + (c - vec3(y)) * s;
}
vec3 beautyColorPro(vec3 base, vec3 guidedB, float guidedA, vec3 fineLow, vec3 wideLow, float face, float excl,
        float hiPlane, float nonHair, vec4 params, vec4 details, vec4 tone, vec4 grade) {
    float skinOnly = step(15.5, params.x);
    float style = params.x - skinOnly * 16.0;
    // Skin gates read the 2px low-pass (less per-pixel chroma noise); off-face also needs some red-over-blue chroma.
    float gate = skinWeight(fineLow);
    float chromaGate = smoothstep(0.045, 0.10, fineLow.r - fineLow.b);
    float polyGate = smoothstep(0.06, 0.18, luma(fineLow)) * mix(0.30, 1.0, smoothstep(0.0, 0.25, gate));
    float skin = clamp(face * polyGate + (1.0 - face) * gate * chromaGate * (1.0 - 0.70 * excl), 0.0, 1.0) * nonHair;
    vec3 guided = clamp(base * guidedA + guidedB, 0.0, 1.0);
    // Beard & beard-border gate: thin strands darker than 2px low-pass inside dense texture stay unsmoothed;
    // bleedGuard blocks bright cheek/lip skin (guided >> fineLow) from smearing onto dark beard borders without blotching shadowed skin.
    float bleedGuard = 1.0 - (1.0 - smoothstep(0.22, 0.31, luma(fineLow))) *
        max(smoothstep(0.020, 0.055, luma(guided) - luma(fineLow)),
            smoothstep(0.24, 0.48, guidedA) * smoothstep(0.003, 0.018, luma(guided) - luma(fineLow)));
    float fineDark = (1.0 - smoothstep(0.02, 0.06, luma(fineLow) - luma(base)) * smoothstep(0.25, 0.55, guidedA)) * bleedGuard;
    float featureKeep = clamp(1.0 - excl, 0.0, 1.0);
    // Coring threshold never below the indoor floor (also covers an unset uGrade).
    float core = max(grade.y, 0.005);
    float band = luma(fineLow - wideLow);
    vec3 color = base;
    float k = 0.0;
    if (params.y > 0.0) {
        // Guided filter reconstruction q = guidedA * I + guidedB; the RGB distance guard keeps hard edges.
        float rEdge = length(base - guided) / 0.36;
        float rEdge2 = rEdge * rEdge;
        k = params.y * 0.96 * (skinOnly > 0.5 ? skin : 1.0) * featureKeep * fineDark / (1.0 + rEdge2 * rEdge2);
        // Pore-scale texture restore from the band-pass, cored above the noise floor.
        float tex = clamp(band, -0.035, 0.035) * smoothstep(core, 2.5 * core, abs(band));
        color = mix(base, guided, k) + vec3(tex * mix(0.35, 0.85, params.w) * k);
    }
    if (details.y > 0.0) {
        // Same band for sharpening: full on features, 0.35 on background, about 0.05 on smoothed face skin
        // (off-face skin-coloured hair stays at the background 0.35).
        float edge = smoothstep(1.5 * core, 4.0 * core, abs(band));
        float region = mix(0.35, 1.0, excl) * (1.0 - 0.85 * skin * featureKeep * face);
        color += vec3(clamp(band, -0.06, 0.06) * edge * details.y * 1.6 * region);
    }
    color = clamp(color, 0.0, 1.0);
    float w = skin * featureKeep * fineDark;
    float liftA = (0.60 * tone.x + 0.40 * params.z) * w;
    float rosyA = 0.25 * tone.y * w;
    float sculpt = 0.30 * details.x * (hiPlane - 0.3 * face) * w * face;
    if (liftA > 0.0 || rosyA > 0.0 || sculpt != 0.0) {
        float y0 = luma(color);
        if (tone.x > 0.0) {
            // De-yellow only hues more yellow than common skin: pull G towards B + 0.38 (R - B), at most 0.02.
            float chroma = max(color.r - color.b, 0.02);
            float excess = smoothstep(0.40, 0.60, (color.g - color.b) / chroma);
            float target = color.b + 0.38 * chroma;
            color.g -= min(max(color.g - target, 0.0) * excess * 0.6 * tone.x, 0.02) * w;
        }
        float y1 = max(luma(color), 0.0001);
        float a = liftA * smoothstep(0.12, 0.28, y0) * mix(0.4, 1.0, smoothstep(0.15, 0.5, y0));
        float y2 = y0 * (1.0 + a) / (1.0 + a * y0);
        if (sculpt != 0.0) {
            // Luma soft-light on the face only: high planes brighter, the rest slightly darker, black point fixed.
            float s = 0.5 + sculpt;
            y2 = (1.0 - 2.0 * s) * y2 * y2 + 2.0 * s * y2;
        }
        color *= y2 / y1;
        // Rosy: midtone chroma boost around the new luma, no added blue.
        if (rosyA > 0.0) color = vec3(y2) + (color - vec3(y2)) * (1.0 + rosyA * 4.0 * y2 * (1.0 - y2));
        color = fitGamut(color, y2);
    }
    color = mix(vec3(luma(color)), color, details.z);
    color = mix(color, clamp(gradePro(color, style), 0.0, 1.0), details.w);
    return clamp(color, 0.0, 1.0);
}
"""

    /**
     * Android 主合成程序：4 个 sampler + 11 个 vec4 = 15 个活跃 uniform 向量（GLES2 片元着色器最低保证 16），
     * 新增合成层需求只剩 1 个向量的余量（门禁要求恰为 15，防止余量被悄悄占掉）。uFrame.w >= 3.5 时采样 uSkinMask 多边形三通道
     * 遮罩（presence = uFrame.w - 4.0），否则回退椭圆遮罩（hiPlane 取 0.3 * face，立体光中性）。
     * uSkinMask.a 是非头发权重（1 = 非头发，SemanticAlphaPass 写入），不论走哪条路径都无条件取一次，换算成
     * nonHair = clamp(a * 1.04 - 0.02, 0, 1)：容忍 RGBA8 与模糊丢掉的 1 LSB，A = 254/255 都读成 1。没有头发遮罩时绑的
     * 皮肤遮罩或回退纹理 A 恒为 255，画面与不接头发遮罩逐字节相同。uBlur 是导向滤波的 (b, a)；uGrade.z > 0.5 表示绑的是
     * BeautyPrePass 回退的 (mean, var)，按同一软映射换算成 (b, a)。uGrade.x > 0 时在合成结果上查 uLut（纹理单元 3），
     * 在美妆之前；uParams.x < -0.5 时不处理、严格直通。
     */
    const val FRAGMENT_PRO = """$PRECISION
uniform sampler2D uSampler;
uniform sampler2D uBlur;
uniform sampler2D uSkinMask;
uniform sampler2D uLut;
uniform vec4 uFrame;
uniform vec4 uParams;
uniform vec4 uDetails;
uniform vec4 uTone;
uniform vec4 uEyes[2];
uniform vec4 uOval;
uniform vec4 uMouth;
uniform vec4 uEyeMask[2];
uniform vec4 uGrade;
varying vec2 vTextureCoord;
$COMMON
$WARP
$MASKS
$COMPOSITE_PRO
$LUT_SAMPLE
vec2 warpCoordinatePro(vec2 uv) {
    if (uTone.w > 0.0) {
        for (int i = 0; i < 2; i++) uv = eye(uv, uEyes[i].xyz, uTone.w, uFrame.z);
    }
    return clamp(uv, 0.0, 1.0);
}
// Mean of four bilinear taps at (+-d.x, +-d.y): half a texel is the exact [1,2,1]^2 binomial.
vec3 diag4(vec2 uv, vec2 d) {
    return 0.25 * (texture2D(uSampler, uv + vec2(-d.x, -d.y)).rgb +
        texture2D(uSampler, uv + vec2(d.x, -d.y)).rgb +
        texture2D(uSampler, uv + vec2(-d.x, d.y)).rgb +
        texture2D(uSampler, uv + vec2(d.x, d.y)).rgb);
}
void main() {
    if (uParams.x < -0.5) { gl_FragColor = texture2D(uSampler, vTextureCoord); return; }
    vec2 coordinate = warpCoordinatePro(vTextureCoord);
    vec4 original = texture2D(uSampler, coordinate);
    vec3 guidedB = original.rgb;
    float guidedA = 0.0;
    if (uParams.y > 0.0) {
        vec4 coeff = texture2D(uBlur, coordinate);
        guidedB = coeff.rgb;
        guidedA = coeff.a;
        if (uGrade.z > 0.5) {
            // Fallback prepass binds (mean, var), var = 50 * dY^2: same soft mapping with sigma0 = 12/255 (0.111).
            float r = coeff.a / 0.111;
            float rp = r * sqrt(r);
            guidedA = rp / (1.0 + rp);
            guidedB = (1.0 - guidedA) * coeff.rgb;
        }
    }
    vec3 fineLow = original.rgb;
    vec3 wideLow = original.rgb;
    if (uParams.y > 0.0 || uDetails.y > 0.0) {
        fineLow = diag4(coordinate, uFrame.xy * 0.5);
        wideLow = diag4(coordinate, uFrame.xy * 1.5);
    }
    // The skin mask alpha is the non-hair weight on every path (no face too); 1 LSB lost to RGBA8 or blur still reads 1.
    vec4 sem = texture2D(uSkinMask, coordinate);
    float nonHair = clamp(sem.a * 1.04 - 0.02, 0.0, 1.0);
    float faceM = 0.0;
    float exclM = 0.0;
    float hiM = 0.0;
    if (uFrame.w >= 3.5) {
        float pres = clamp(uFrame.w - 4.0, 0.0, 1.0);
        vec3 poly = sem.rgb * pres;
        faceM = poly.r;
        exclM = poly.g;
        hiM = poly.b;
    } else if (uFrame.w > 0.0) {
        vec2 em = faceMasks(coordinate, uFrame.z, uFrame.w, uOval, vec2(uEyes[0].w, uEyes[1].w),
            uMouth, uEyeMask[0], uEyeMask[1]);
        faceM = em.x;
        exclM = em.y;
        hiM = em.x * 0.3;
    }
    vec3 color = beautyColorPro(original.rgb, guidedB, guidedA, fineLow, wideLow, faceM, exclM, hiM, nonHair,
        uParams, uDetails, uTone, uGrade);
    if (uGrade.x > 0.0) {
        vec3 rgb = clamp(color, 0.0, 1.0);
        float level = rgb.b * 63.0;
        vec3 low = texture2D(uLut, lutCell(floor(level), rgb.rg)).rgb;
        vec3 high = texture2D(uLut, lutCell(ceil(level), rgb.rg)).rgb;
        vec3 graded = mix(low, high, level - floor(level));
        color = mix(color, graded, uGrade.x);
    }
    gl_FragColor = vec4(color, original.a);
}
"""

    /**
     * P1 DOWN 顶点：4 个双线性抽头位于 ±uTexel。
     * - BeautyPrePass（1/4 降采样）传 uTexel = (1/w, 1/h)：抽头位于 ±1 原始像素，尺寸为 4 的倍数时恰为 4×4 盒式均值。
     * - GuidedFilterPrePass（1/3 降采样）传 uTexel = BeautyQualityTuning.DOWN_TAP_SCALE / (w, h)，即 ±2/3 原始像素：每个抽头按
     *   2:1 混合相邻两个像素，4 次双线性取样恰好等于 3×3 盒式均值（每个像素权重 1/9）。若按 ±1 像素取，只采到 3×3 的四个角，
     *   1px 发丝落在中间行列时整根漏采，保留率随相位跳变。
     */
    const val DOWN_VERTEX = """attribute vec4 aPosition;
attribute vec2 aTextureCoord;
uniform vec2 uTexel;
varying vec2 vTap0;
varying vec2 vTap1;
varying vec2 vTap2;
varying vec2 vTap3;
void main() {
    gl_Position = aPosition;
    vTap0 = aTextureCoord + vec2(-uTexel.x, -uTexel.y);
    vTap1 = aTextureCoord + vec2(uTexel.x, -uTexel.y);
    vTap2 = aTextureCoord + vec2(-uTexel.x, uTexel.y);
    vTap3 = aTextureCoord + vec2(uTexel.x, uTexel.y);
}
"""

    /** P1 DOWN 片元：src → T0。 */
    const val DOWN_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
varying vec2 vTap0;
varying vec2 vTap1;
varying vec2 vTap2;
varying vec2 vTap3;
void main() {
    gl_FragColor = (texture2D(uSampler, vTap0) + texture2D(uSampler, vTap1) +
        texture2D(uSampler, vTap2) + texture2D(uSampler, vTap3)) * 0.25;
}
"""

    /** P2–P5 顶点：sigma 2 的 9 抽头高斯写成 5 个线性抽头，全部是 varying；uDir = (1/sw, 0) 或 (0, 1/sh)。 */
    const val BLUR_VERTEX = """attribute vec4 aPosition;
attribute vec2 aTextureCoord;
uniform vec2 uDir;
varying vec2 vC;
varying vec2 vA1;
varying vec2 vB1;
varying vec2 vA2;
varying vec2 vB2;
void main() {
    gl_Position = aPosition;
    vC = aTextureCoord;
    vA1 = aTextureCoord + uDir * 1.4072;
    vB1 = aTextureCoord - uDir * 1.4072;
    vA2 = aTextureCoord + uDir * 3.2939;
    vB2 = aTextureCoord - uDir * 3.2939;
}
"""

    /** 模糊片元的公共部分：blur5() 按权重 0.2042 / 0.3040 / 0.0939 取 5 个抽头。 */
    const val BLUR = """$PRECISION
uniform sampler2D uSampler;
varying vec2 vC;
varying vec2 vA1;
varying vec2 vB1;
varying vec2 vA2;
varying vec2 vB2;
vec4 blur5() {
    return texture2D(uSampler, vC) * 0.2042 +
        (texture2D(uSampler, vA1) + texture2D(uSampler, vB1)) * 0.3040 +
        (texture2D(uSampler, vA2) + texture2D(uSampler, vB2)) * 0.0939;
}
"""

    /** P2 BLUR_RGB：T0 → T1，横向。 */
    const val BLUR_RGB_FRAGMENT = """$BLUR
void main() { gl_FragColor = blur5(); }
"""

    /** P3 BLUR_DIFF：T1 → T2，纵向得 mean；uOrig = T0（unit 1），输出 (mean.rgb, v0)，v0 = min((dot(T0 − mean, L) × 7.07)², 1)。 */
    const val BLUR_DIFF_FRAGMENT = """$BLUR
uniform sampler2D uOrig;
void main() {
    vec3 mean = blur5().rgb;
    float d = dot(texture2D(uOrig, vC).rgb - mean, vec3(0.2126, 0.7152, 0.0722)) * 7.07;
    gl_FragColor = vec4(mean, min(d * d, 1.0));
}
"""

    /** P4/P5 BLUR_ALPHA：只模糊 alpha，rgb 取中心抽头原样带过。 */
    const val BLUR_ALPHA_FRAGMENT = """$BLUR
void main() { gl_FragColor = vec4(texture2D(uSampler, vC).rgb, blur5().a); }
"""

    /**
     * 导向滤波 P3：纵向得局部均值 mean，alpha 存 RGB 偏差的线性能量 clamp(|I - mean| * 2.5)。线性编码让 RGBA8 在 σ ≈ 1/255
     * 处仍有分辨率，饱和点约每通道 59/255；系数须与 BeautyQualityTuning.ENERGY_SCALE 一致（门禁核对）。
     */
    const val GUIDED_DIFF_FRAGMENT = """$BLUR
uniform sampler2D uOrig;
void main() {
    vec3 mean = blur5().rgb;
    vec3 diff = texture2D(uOrig, vC).rgb - mean;
    gl_FragColor = vec4(mean, clamp(length(diff) * 2.5, 0.0, 1.0));
}
"""

    /**
     * 导向滤波 P5 的系数映射（片段，GUIDED_COEFF_FRAGMENT 与 GUIDED_WEIGHT_COEFF_FRAGMENT 共用）：模糊后的能量 s →
     * a = max(r^1.5 / (1 + r^1.5), smoothstep(lo, 0.30, s))，r = s^2 / eps。a 从 0.06 到 0.94 跨 σ 约 6 倍（旧 r^4 只跨 2 倍），
     * 强边缘再由 smoothstep 锁到 1。边缘锁下限随磨皮档自适应：lo = clamp(√eps − 0.09, 0.12, 0.18)，磨皮 0.55 → 0.121、
     * 0.65 → 0.134、0.8 → 0.155、1.0 → 0.180。棕发、染发的发缘能量 s 只有 0.2–0.3，旧的固定锁 0.30–0.55 锁不住，发缘两侧被
     * 磨出 8–15px 的光晕和色渗；但 Δ60 的法令纹、黑眼圈 s 也在 0.24–0.27，固定把下限压到 0.12 会让满档磨不掉它们。所以默认档
     * 按发缘取 0.12，越往满档越让位给瑕疵，满档取 0.18（满档发缘光晕留约 6/255，见计划 2026-11-21 的 H3）。
     * eps 下限 0.005 让 r^1.5 在 mediump 下不溢出（实际 eps >= 0.018，见 BeautyQualityTuning.guidedEps）。
     * 门禁 scripts/glsl/glsl_gate.py 的 guidedCoeffCurve 探针守护此曲线（eps(0.55) 下 s = s0/2、s0、1.25·s0、2·s0 与 0.60，
     * s0 = √eps；另加 eps(1.0) 下 s = 0.24 守自适应下限），调整映射时要同步改其中的 GUIDED_COEFF_CURVE 区间。
     */
    const val GUIDED_COEFF = """$BLUR
uniform float uGuidedEps;
uniform sampler2D uMacro;
uniform float uEvenness;
float guidedCoeff(float s) {
    float r = s * s / max(uGuidedEps, 0.005);
    float rp = r * sqrt(r);
    float lo = clamp(sqrt(max(uGuidedEps, 0.005)) - 0.09, 0.12, 0.18);
    float a = max(rp / (1.0 + rp), smoothstep(lo, 0.30, s));
    return a;
}
vec3 evenMean(vec3 fineMean, float s, float maskW) {
    if (uEvenness <= 0.0) return fineMean;
    vec3 macroMean = texture2D(uMacro, vC).rgb;
    float dy = abs(dot(macroMean - fineMean, vec3(0.2126, 0.7152, 0.0722))) / 0.068;
    float dy2 = dy * dy;
    float boneKeep = 1.0 / (1.0 + dy2 * dy2);
    float flatKeep = 1.0 - smoothstep(0.05, 0.14, s);
    return mix(fineMean, macroMean, uEvenness * boneKeep * flatKeep * maskW);
}
"""

    /** 导向滤波 P5：纵向模糊能量得 s，按 GUIDED_COEFF 的映射求系数 a，输出 ((1 - a) * evenMean, a)。 */
    const val GUIDED_COEFF_FRAGMENT = """$GUIDED_COEFF
void main() {
    float s = blur5().a;
    vec3 mean = evenMean(texture2D(uSampler, vC).rgb, s, 1.0);
    float a = guidedCoeff(s);
    gl_FragColor = vec4((1.0 - a) * mean, a);
}
"""

    /**
     * 头发加权导向滤波 P2'（仅 Android，GuidedFilterPrePass 传入头发权重纹理时代替 P2）：横向模糊 (w·I, w)，w 是非头发权重——
     * uWeight 为本帧 SemanticAlphaPass 写过 A 的 SkinMaskPass 输出，按与 FRAGMENT_PRO 相同的换算 clamp(a·1.04 − 0.02, 0, 1)
     * 读 A（1 = 非头发）。思路同背景虚化的 BG_PREMASK：先按权重预乘、模糊后再归一化，皮肤侧的局部均值就不再混进头发。
     * 5 个抽头的位置与权重同 BLUR 的 blur5（门禁核对权重）。
     */
    const val GUIDED_WEIGHT_BLUR_FRAGMENT = """$BLUR
uniform sampler2D uWeight;
// Premultiply by the non-hair weight (skin mask alpha, same mapping as FRAGMENT_PRO), then blur like blur5().
vec4 weighted(vec2 uv) {
    float w = clamp(texture2D(uWeight, uv).a * 1.04 - 0.02, 0.0, 1.0);
    return vec4(texture2D(uSampler, uv).rgb * w, w);
}
void main() {
    gl_FragColor = weighted(vC) * 0.2042 + (weighted(vA1) + weighted(vB1)) * 0.3040 +
        (weighted(vA2) + weighted(vB2)) * 0.0939;
}
"""

    /**
     * 头发加权导向滤波 P3'（代替 P3）：纵向模糊 P2' 的 (Σw·I, Σw) 后归一化得只含非头发像素的均值 mean（Σw ≤ 0.02 的纯头发区按
     * 0.02 归一，其系数在 P5' 取恒等，不影响结果）；能量仍按 GUIDED_DIFF 的写法对这个均值求，不加权（能量也加权会让发缘皮肤侧
     * 被磨得更狠、错位遮罩下头发侧光晕更重，评审 2026-11-21 #0 实测否决）。系数须与 BeautyQualityTuning.ENERGY_SCALE 一致（门禁核对）。
     */
    const val GUIDED_WEIGHT_DIFF_FRAGMENT = """$BLUR
uniform sampler2D uOrig;
// Mean over non-hair pixels only; the energy is measured against that mean exactly as in GUIDED_DIFF_FRAGMENT.
void main() {
    vec4 sum = blur5();
    vec3 mean = clamp(sum.rgb / max(sum.a, 0.02), 0.0, 1.0);
    vec3 diff = texture2D(uOrig, vC).rgb - mean;
    gl_FragColor = vec4(mean, clamp(length(diff) * 2.5, 0.0, 1.0));
}
"""

    /**
     * 头发加权导向滤波 P5'（代替 P5）：系数映射同 GUIDED_COEFF，再按本像素的非头发权重向恒等系数 (a = 1, b = 0) 收：头发处
     * q = I。P3' 给头发像素算出的是邻近皮肤的均值，软遮罩过渡带里只有部分权重、主合成仍会轻度磨皮的发梢若照用它，会被往肤色拉；
     * 取恒等后这些像素的重构就是自己。恒等系数经 P6/P7 混进发缘皮肤时，对皮肤像素仍有 q ≈ I，不引入头发的颜色。
     * 实测（评审 2026-11-21 #0）：对齐的软遮罩下与不取恒等相差 ±0.5/255 以内，遮罩比头发少 3px 时满档头发侧光晕低至多 1.25/255；
     * 门禁的 G1 矩阵区分不出这一项，属保守做法。
     */
    const val GUIDED_WEIGHT_COEFF_FRAGMENT = """$GUIDED_COEFF
uniform sampler2D uWeight;
// Hair (weight 0) gets the identity coefficients a = 1, b = 0, so the coefficient blur never pulls hair colour into skin.
void main() {
    float s = blur5().a;
    float w = clamp(texture2D(uWeight, vC).a * 1.04 - 0.02, 0.0, 1.0);
    vec3 mean = evenMean(texture2D(uSampler, vC).rgb, s, w);
    float a = mix(1.0, guidedCoeff(s), w);
    gl_FragColor = vec4((1.0 - a) * mean, a);
}
"""

    /**
     * 头发遮罩写皮肤遮罩 A 通道的顶点着色器（仅 Android，SemanticAlphaPass）：aPosition 是目标 NDC，aTextureCoord 是源 uv
     * （相机帧纹理坐标）。全屏四边形两者相同；做了网格微整时再用 FaceMeshDeformer.vertexData（dst NDC、src 纹理坐标）画一遍，
     * 遮罩在相机系取样、按网格正向变形落点，不需要逆映射。vDst 是目标处的帧纹理坐标（判断是否在脸内）。
     */
    const val SEMANTIC_VERTEX = """attribute vec2 aPosition;
attribute vec2 aTextureCoord;
varying vec2 vSrc;
varying vec2 vDst;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vSrc = aTextureCoord;
    vDst = aPosition * 0.5 + 0.5;
}
"""

    /**
     * 头发遮罩 → 皮肤遮罩 A 通道（仅 Android，SemanticAlphaPass，colorMask 只开 A）：A = 1 − 强度 · smoothstep(lo, hi, hair')，
     * 1 = 非头发，FRAGMENT_PRO 读它得到 nonHair。uSemantic 为头发置信度（LUMINANCE，第 0 行是画面顶部）；uHairX0 / uHairX1 是
     * 「显示帧纹理坐标 → 遮罩时刻纹理坐标」2×3 仿射的两行（HairMaskTuning.affine，w 未用），落到遮罩外按无头发；
     * uHair = (强度, 脸外下限, 脸外上限, 脸内偏置)；uFaceOval = (cx, cy, rx, ry)、uFaceAxis = (u.x, u.y, aspect, 未用)，同 MASKS：
     * 在目标处按脸椭圆羽化，脸内两个阈值都抬高偏置，压住脸上被误检成头发、出现不磨斑块的情况。强度为 0 时 A 恒为 1。
     */
    const val HAIR_ALPHA_FRAGMENT = """$PRECISION
uniform sampler2D uSemantic;
uniform vec4 uHair;
uniform vec4 uHairX0;
uniform vec4 uHairX1;
uniform vec4 uFaceOval;
uniform vec4 uFaceAxis;
varying vec2 vSrc;
varying vec2 vDst;
$COMMON
// Hair confidence at the source uv moved to mask time; the mask's row 0 is the picture top, outside it counts as no hair.
void main() {
    vec2 q = vec2(dot(uHairX0.xyz, vec3(vSrc, 1.0)), dot(uHairX1.xyz, vec3(vSrc, 1.0)));
    float hair = texture2D(uSemantic, vec2(q.x, 1.0 - q.y)).r * step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    // Inside the face oval both thresholds rise, so skin misread as hair does not leave unsmoothed patches.
    float bias = uHair.w * (1.0 - smoothstep(0.50, 1.0, ellipseE(vDst, uFaceOval, uFaceAxis.xy, max(uFaceAxis.z, 0.0001))));
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0 - uHair.x * smoothstep(uHair.y + bias, uHair.z + bias, hair));
}
"""

    /**
     * 背景虚化的头发合并道（仅 Android，BackgroundBlurPass，与 VERTEX 配对，在人像遮罩分辨率上绘制）：
     * R = max(person, 头部先验 · shown · smoothstep(lo, hi, hair'))。selfie 人像遮罩看不见的低对比飞发（如白墙前的金发）靠头发类
     * 找回；头部先验 1 − smoothstep(2.0, 2.8, d)（d 为到脸椭圆中心的距离，以椭圆半径为单位，推断初值，待真机调）压住背景物被误检
     * 成头发。输出与 uPerson（LUMINANCE）同为「第 0 行是画面顶部」：本道按纹素原位读 uPerson、在 (x, 1 − y) 处求帧纹理坐标，
     * BG_PREMASK / BG_MIX 原文不用改（读 .r）。uSemantic、uHairX0 / uHairX1、uFaceOval / uFaceAxis 同 HAIR_ALPHA_FRAGMENT；
     * uHair = (shown, 下限, 上限, 未用)，下限 / 上限即 BackgroundBlurPass.HAIR_LOW / HAIR_HIGH；shown 为 0 时输出逐字节等于 person。
     */
    const val BG_HAIR_FRAGMENT = """$PRECISION
uniform sampler2D uPerson;
uniform sampler2D uSemantic;
uniform vec4 uHair;
uniform vec4 uHairX0;
uniform vec4 uHairX1;
uniform vec4 uFaceOval;
uniform vec4 uFaceAxis;
varying vec2 vTextureCoord;
$COMMON
// Lift confident hair near the head into the person mask (row 0 = picture top, same layout as uPerson).
void main() {
    float person = texture2D(uPerson, vTextureCoord).r;
    vec2 p = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y);
    vec2 q = vec2(dot(uHairX0.xyz, vec3(p, 1.0)), dot(uHairX1.xyz, vec3(p, 1.0)));
    float hair = texture2D(uSemantic, vec2(q.x, 1.0 - q.y)).r * step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    // Head prior: distance from the face oval centre in oval radii, so background objects misread as hair stay blurred.
    float prior = 1.0 - smoothstep(2.0, 2.8, sqrt(ellipseE(p, uFaceOval, uFaceAxis.xy, max(uFaceAxis.z, 0.0001))));
    gl_FragColor = vec4(max(person, prior * uHair.x * smoothstep(uHair.y, uHair.z, hair)), 0.0, 0.0, 1.0);
}
"""

    /**
     * 背景虚化（仅 Android）的混合：按人像置信度在原帧与归一化背景模糊图之间取舍。uMask 为 LUMINANCE 遮罩，第 0 行是画面顶部；
     * uBackground = (强度, 置信度下限, 置信度上限, 细节保留 β)，强度为 0 时严格直通。见 BackgroundBlurPass。
     * 细节保留：1–3px 飞发、碎发在 216×384 遮罩上的置信度只有 0.07–0.40，低于下限会被整根虚化掉。遮罩略有置信度
     * （≥0.02–0.10）、且亮度与干净背景模糊图相差 0.15–0.30 以上、又在头部附近的像素按 β 保留原样，keep = max(遮罩 keep, β·fg)。
     * 只看亮度差、阈值从 0.15 起，低对比的纹理背景（与自身模糊图的亮度差多在 0.15 以内）不会被误留。
     * 头部先验（评审 2026-11-21 #1）：fg 再乘 1 − smoothstep(2.0, 2.8, d)，d 是到脸椭圆中心的距离（以椭圆半径为单位，写法同
     * BG_HAIR_FRAGMENT，uFaceOval = (cx, cy, rx, ry)、uFaceAxis = (u.x, u.y, aspect, 未用)，帧纹理坐标同主合成 uOval）。
     * 真实人像遮罩在整个轮廓外有一圈 0.02–0.3 的低置信带、且逐帧抖动：不加先验时肩、臂、躯干旁的条纹、文字等高对比背景会
     * 整圈保持清晰并随遮罩闪烁；先验把细节保留限制在头发所在的头部附近（头顶飞发约在 2 个半径内）。没有人脸时调用方传
     * β = 0，细节保留整体不生效。代价：头部附近 0–8px 的高对比背景细纹理仍会保持清晰并随遮罩抖动（门禁 G8 量化）。
     * 遮罩为 0 处 fg 恒为 0，背景照常全虚化。定义放在 COMMON 之后（要用其中的 luma / ellipseE，Kotlin 常量不能前向引用）。
     */
    const val BG_MIX_FRAGMENT = """$PRECISION
uniform sampler2D uSampler;
uniform sampler2D uBlurred;
uniform sampler2D uMask;
uniform vec4 uBackground;
uniform vec4 uFaceOval;
uniform vec4 uFaceAxis;
varying vec2 vTextureCoord;
$COMMON
// Keep the person sharp, blur only unpremultiplied background pixels (zero foreground halo bleed).
void main() {
  vec4 color = texture2D(uSampler, vTextureCoord);
  float person = texture2D(uMask, vec2(vTextureCoord.x, 1.0 - vTextureCoord.y)).r;
  vec4 blurSample = texture2D(uBlurred, vTextureCoord);
  vec3 blurred = blurSample.a > 0.02 ? clamp(blurSample.rgb / blurSample.a, 0.0, 1.0) : color.rgb;
  // Detail keep near the head only: low-confidence pixels whose luma stands out from the clean blurred background (flyaway hair).
  float head = 1.0 - smoothstep(2.0, 2.8, sqrt(ellipseE(vTextureCoord, uFaceOval, uFaceAxis.xy, max(uFaceAxis.z, 0.0001))));
  float fg = smoothstep(0.15, 0.30, abs(luma(color.rgb - blurred))) * smoothstep(0.02, 0.10, person) * head;
  float keep = max(smoothstep(uBackground.y, uBackground.z, person), uBackground.w * fg);
  gl_FragColor = vec4(mix(color.rgb, mix(blurred, color.rgb, keep), uBackground.x), color.a);
}
"""
}
