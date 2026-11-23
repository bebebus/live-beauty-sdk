// 在无头 Chrome 的 WebGL1（ANGLE/SwiftShader，严格 GLSL ES 1.00）里编译、链接并运行 VideoEffectsShaders 原文。
// 只负责取样和回读，所有比较与容差都在 glsl_gate.py 里。INPUT 由 glsl_gate.py 拼在本文件前面：
//   INPUT.shaders   展开后的常量原文
//   INPUT.programs  生产程序 [名字, 顶点常量, 片元常量]（glsl_gate.py 已核对与 VideoEffectsFilter.kt / BeautyPrePass.kt 一致）
//   INPUT.controls  负对照：每项一份合法源码和只差一处的非法源码
//   INPUT.vectors   beauty_vectors.json 的 skin_gate / composite / masks / prepass / end_to_end 段
//   INPUT.geometryUniforms  uniform 名 → 值：glsl_gate.py 按 VideoEffectsFilter.kt 的打包代码拆 end_to_end.state 得到
//   INPUT.guided    导向滤波常量（BeautyQualityTuning.kt：能量系数、σ0 区间 /255、P1 抽头间距 downTap、GuidedSpread.DEFAULT）
//   INPUT.curve     导向系数曲线取样点 [磨皮档, s 相对 √eps 的倍数, s 的绝对值]（glsl_gate.py 的 GUIDED_COEFF_CURVE）
//   INPUT.proCases / INPUT.naturalCases  beautyColorPro 的逐例探针（glsl_gate.py 定义，比较也在那边）
//   INPUT.hairEdge  毛发边界 G1–G8：场景（RGBA base64）、档位、回读下标与背景虚化场景参数（glsl_gate.py 生成）；
//                   头发遮罩 A3–A7 的整帧场景也在这里（皮肤遮罩带 A 通道，可左右两半不同，uFrame.w 可指定）
//   INPUT.hairACases  头发遮罩 A1 / A2 的 beautyColorPro 逐例探针（dMask.w = 非头发权重）
//   INPUT.hairAlpha / INPUT.bgHair  头发遮罩 HA1（HAIR_ALPHA_FRAGMENT）与 BH1（BG_HAIR_FRAGMENT + 背景虚化）的场景参数
'use strict';
const out = {errors: [], env: null, programs: {}, controls: {}, results: {}};
try {
  run();
} catch (error) {
  out.errors.push(String((error && error.stack) || error));
}
document.getElementById('out').textContent = JSON.stringify(out);

function run() {
  const S = INPUT.shaders, V = INPUT.vectors;
  const canvas = document.createElement('canvas');
  canvas.width = 1; canvas.height = 1;
  const gl = canvas.getContext('webgl', {antialias: false, alpha: true, premultipliedAlpha: false, depth: false,
    stencil: false, preserveDrawingBuffer: false});
  if (!gl) throw new Error('无法创建 WebGL1 上下文');
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  out.env = {
    version: gl.getParameter(gl.VERSION),
    glsl: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
  };

  // ---- 编译与链接
  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return {shader, ok: !!gl.getShaderParameter(shader, gl.COMPILE_STATUS), log: (gl.getShaderInfoLog(shader) || '').trim()};
  }
  function link(vertexSource, fragmentSource) {
    const v = compile(gl.VERTEX_SHADER, vertexSource), f = compile(gl.FRAGMENT_SHADER, fragmentSource);
    const report = {vertex: v.ok, fragment: f.ok, link: false, log: [v.log, f.log].filter(Boolean).join(' | ')};
    if (!v.ok || !f.ok) return {report, program: null};
    const program = gl.createProgram();
    gl.attachShader(program, v.shader); gl.attachShader(program, f.shader);
    gl.linkProgram(program);
    report.link = !!gl.getProgramParameter(program, gl.LINK_STATUS);
    const log = (gl.getProgramInfoLog(program) || '').trim();
    if (log) report.log = [report.log, log].filter(Boolean).join(' | ');
    if (!report.link) return {report, program: null};
    report.uniforms = [];
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const u = gl.getActiveUniform(program, i);
      report.uniforms.push([u.name, u.type, u.size]);
    }
    return {report, program};
  }
  // 强制 mediump：把固定开头整段换成一行 precision mediump float;（模拟没有片元 highp 的 GPU）。
  const mediump = (source) => source.split(S.PRECISION).join('precision mediump float;\n');
  const P = {};
  for (const [name, vertex, fragment] of INPUT.programs) {
    const high = link(S[vertex], S[fragment]);
    out.programs[name] = high.report;
    P[name] = high.program;
    out.programs[name + '@mediump'] = link(S[vertex], mediump(S[fragment])).report;
  }
  const debug = {
    // s = skinWeight × 255 拆成 (floor(s)/255, fract(s))，8 位回读后仍有约 1/65025 的分辨率。
    skin: S.PRECISION + `
uniform vec4 dI;
` + S.COMMON + `
void main() {
    float s = skinWeight(dI.rgb) * 255.0;
    gl_FragColor = vec4(floor(s) / 255.0, fract(s), 0.0, 1.0);
}
`,
    composite: S.PRECISION + `
uniform vec4 dI; uniform vec4 dMean; uniform vec4 dAround; uniform vec4 dMask;
uniform vec4 dParams; uniform vec4 dDetails; uniform vec4 dTone;
` + S.COMMON + S.COMPOSITE + `
void main() {
    gl_FragColor = vec4(beautyColor(dI.rgb, dMean.rgb, dMean.a, dAround.rgb, dMask.x, dMask.y,
        dParams, dDetails, dTone), 1.0);
}
`,
    // dGuided = (b.rgb, a)，dFine / dWide = 带通的两个低通，dMask = (face, excl, hiPlane, nonHair)，dGrade 同 uGrade。
    compositePro: S.PRECISION + `
uniform vec4 dI; uniform vec4 dGuided; uniform vec4 dFine; uniform vec4 dWide; uniform vec4 dMask;
uniform vec4 dParams; uniform vec4 dDetails; uniform vec4 dTone; uniform vec4 dGrade;
` + S.COMMON + S.COMPOSITE_PRO + `
void main() {
    gl_FragColor = vec4(beautyColorPro(dI.rgb, dGuided.rgb, dGuided.a, dFine.rgb, dWide.rgb, dMask.x, dMask.y, dMask.z,
        dMask.w, dParams, dDetails, dTone, dGrade), 1.0);
}
`,
    masks: S.PRECISION + `
uniform vec4 dUv; uniform vec2 dAxis; uniform vec4 dOval; uniform vec4 dMouth; uniform vec4 dEye0; uniform vec4 dEye1;
` + S.COMMON + S.MASKS + `
void main() {
    vec2 m = faceMasks(dUv.xy, dUv.z, dUv.w, dOval, dAxis, dMouth, dEye0, dEye1);
    gl_FragColor = vec4(m, 0.0, 1.0);
}
`,
    // 纯几何值直接调 COMMON 的 ellipseE，羽化区间取规格第 2 节。
    ellipses: S.PRECISION + `
uniform vec4 dUv; uniform vec2 dAxis; uniform vec4 dOval; uniform vec4 dMouth; uniform vec4 dEye0; uniform vec4 dEye1;
` + S.COMMON + `
void main() {
    gl_FragColor = vec4(1.0 - smoothstep(0.50, 1.0, ellipseE(dUv.xy, dOval, dAxis, dUv.z)),
        1.0 - smoothstep(0.55, 1.0, ellipseE(dUv.xy, dEye0, dAxis, dUv.z)),
        1.0 - smoothstep(0.55, 1.0, ellipseE(dUv.xy, dEye1, dAxis, dUv.z)),
        1.0 - smoothstep(0.65, 1.0, ellipseE(dUv.xy, dMouth, dAxis, dUv.z)));
}
`,
  };
  for (const name of Object.keys(debug)) {
    const built = link(S.VERTEX, debug[name]);
    out.programs['debug:' + name] = built.report;
    P['debug:' + name] = built.program;
  }

  // ---- 负对照：合法版本必须通过、非法版本必须被拒，证明拒绝来自那一处构造。
  for (const c of INPUT.controls) {
    const good = link(c.vertex, c.good).report, bad = link(c.vertex, c.bad).report;
    out.controls[c.name] = {
      good: good.vertex && good.fragment && good.link,
      bad: bad.vertex && bad.fragment && bad.link,
      log: bad.log.slice(0, 200),
    };
  }

  const missing = Object.keys(P).filter((k) => !P[k]);
  if (missing.length) { out.errors.push('以下程序未能链接，跳过像素检查：' + missing.join(', ')); return; }

  // ---- 公共 GL 工具
  function texture(w, h, data) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return t;
  }
  function framebuffer(t) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('FBO 不完整');
    return f;
  }
  // 与 BeautyPrePass / VideoEffectsFilter 相同的顶点数据：x, y, s, t 交错，TRIANGLE_STRIP。
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1]), gl.STATIC_DRAW);
  function use(program) {
    gl.useProgram(program);
    const arrays = [['aPosition', 0], ['aTextureCoord', 8]]
      .map(([name, offset]) => [gl.getAttribLocation(program, name), offset]).filter(([at]) => at >= 0);
    for (const [at, offset] of arrays) { gl.vertexAttribPointer(at, 2, gl.FLOAT, false, 16, offset); gl.enableVertexAttribArray(at); }
    return {
      at: (name) => gl.getUniformLocation(program, name),
      done: () => arrays.forEach(([at]) => gl.disableVertexAttribArray(at)),
    };
  }
  function read(x, y, w, h) {
    const bytes = new Uint8Array(w * h * 4);
    gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    let hex = '';
    for (let i = 0; i < bytes.length; i++) hex += (bytes[i] < 16 ? '0' : '') + bytes[i].toString(16);
    return hex;
  }
  function checkError(where) {
    const e = gl.getError();
    if (e !== gl.NO_ERROR) out.errors.push(where + '：glError 0x' + e.toString(16));
  }
  // 逐例调试程序：每个用例画到网格中的一个像素，最后一次回读（行优先，第 0 行在底部）。
  const GRID = 64;
  function grid(program, cases, setUniforms) {
    const rows = Math.ceil(cases.length / GRID);
    const target = texture(GRID, rows, null), frame = framebuffer(target);
    const p = use(program);
    cases.forEach((c, i) => {
      gl.viewport(i % GRID, Math.floor(i / GRID), 1, 1);
      setUniforms(p.at, c);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    p.done();
    const hex = read(0, 0, GRID, rows);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return hex;
  }

  // ---- skin_gate 段：COMMON 的 skinWeight 逐例。
  out.results.skin = grid(P['debug:skin'], V.skin_gate.cases, (at, c) => {
    gl.uniform4f(at('dI'), c.rgb8[0] / 255, c.rgb8[1] / 255, c.rgb8[2] / 255, 1.0);
  });
  checkError('skin_gate');

  // ---- composite 段：beautyColor 逐例，processing 恒为 1。
  const sets = V.composite.paramSets;
  out.results.composite = grid(P['debug:composite'], V.composite.cases, (at, c) => {
    const p = sets[c.params];
    gl.uniform4f(at('dI'), c.I[0], c.I[1], c.I[2], 1.0);
    gl.uniform4f(at('dMean'), c.mean[0], c.mean[1], c.mean[2], c.var);
    gl.uniform4f(at('dAround'), c.avg4[0], c.avg4[1], c.avg4[2], 1.0);
    gl.uniform4f(at('dMask'), c.face, c.excl, 0.0, 0.0);
    gl.uniform4f(at('dParams'), 1.0, p.smoothing, p.brightness, p.style);
    gl.uniform4f(at('dDetails'), p.skinOnly ? 1.0 : 0.0, p.sharpness, p.saturation, p.filterIntensity);
    gl.uniform4f(at('dTone'), p.whiten, p.rosy, 0.0, 0.0);
  });
  checkError('composite');

  // ---- masks 段：faceMasks 与各椭圆的纯几何值。
  const maskUniforms = (at, c) => {
    const g = V.masks.geometries[c.geometry];
    gl.uniform4f(at('dUv'), c.uv[0], c.uv[1], g.aspect, c.presence);
    gl.uniform2f(at('dAxis'), g.axis[0], g.axis[1]);
    gl.uniform4fv(at('dOval'), g.oval);
    gl.uniform4fv(at('dMouth'), g.mouth);
    gl.uniform4fv(at('dEye0'), g.eyeMasks.slice(0, 4));
    gl.uniform4fv(at('dEye1'), g.eyeMasks.slice(4, 8));
  };
  out.results.masks = grid(P['debug:masks'], V.masks.cases, maskUniforms);
  out.results.ellipses = grid(P['debug:ellipses'], V.masks.cases, maskUniforms);
  checkError('masks');

  // ---- prepass 段：64x96 源图跑生产 5 通道。向量第 0 行是顶部，上传时倒序（纹理 t = 0 在底部）。
  const W = V.prepass.width, H = V.prepass.height, SW = Math.ceil(W / 4), SH = Math.ceil(H / 4);
  const pixels = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) {
    const row = V.prepass.rgb8[H - 1 - r];
    for (let x = 0; x < W; x++) {
      for (let i = 0; i < 3; i++) pixels[(r * W + x) * 4 + i] = row[x * 3 + i];
      pixels[(r * W + x) * 4 + 3] = 255;
    }
  }
  const source = texture(W, H, pixels);
  const T = [0, 1, 2].map(() => texture(SW, SH, null)), F = T.map(framebuffer);
  function pass(program, frame, first, second, setUniforms) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, frame);
    const p = use(program);
    if (second) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, second); }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, first);
    gl.uniform1i(p.at('uSampler'), 0);
    setUniforms(p.at);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
  }
  // 与 BeautyPrePass.render 相同：一次 viewport 覆盖 5 个通道，返回 T2 = (mean.rgb, var)。
  function prepass() {
    gl.viewport(0, 0, SW, SH);
    pass(P.down, F[0], source, null, (at) => gl.uniform2f(at('uTexel'), 1 / W, 1 / H));
    pass(P.blurRgb, F[1], T[0], null, (at) => gl.uniform2f(at('uDir'), 1 / SW, 0));
    pass(P.blurDiff, F[2], T[1], T[0], (at) => { gl.uniform2f(at('uDir'), 0, 1 / SH); gl.uniform1i(at('uOrig'), 1); });
    pass(P.blurAlpha, F[1], T[2], null, (at) => gl.uniform2f(at('uDir'), 1 / SW, 0));
    pass(P.blurAlpha, F[2], T[1], null, (at) => gl.uniform2f(at('uDir'), 0, 1 / SH));
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    return T[2];
  }
  prepass();
  checkError('prepass');
  gl.bindFramebuffer(gl.FRAMEBUFFER, F[0]); out.results.prepassT0 = read(0, 0, SW, SH);
  gl.bindFramebuffer(gl.FRAMEBUFFER, F[2]); out.results.prepassT2 = read(0, 0, SW, SH);
  // 全分辨率 GL_LINEAR 放大（原画程序采 T2），用于探针与 clamp-to-edge 检查。
  const full = texture(W, H, null), fullFrame = framebuffer(full);
  function original(input) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.original);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, input);
    gl.uniform1i(p.at('uSampler'), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.bindTexture(gl.TEXTURE_2D, null);
    return read(0, 0, W, H);
  }
  out.results.prepassUp = original(T[2]);
  out.results.original = original(source);
  checkError('original');

  // ---- end_to_end 段：生产合成程序，无变形。几何 uniform 取自 INPUT.geometryUniforms（VideoEffectsFilter.kt 的
  // 打包代码作用在 41 维状态上的结果），其余 uniform 与 drawFilter 一致。
  for (const name of Object.keys(INPUT.geometryUniforms)) {
    if (gl.getUniformLocation(P.composite, name) === null) out.errors.push('合成程序没有活跃 uniform ' + name);
  }
  function composite(c) {
    // drawFilter：先在私有 FBO 跑预处理，再恢复输出 FBO 与 viewport。
    let blur = null;
    if (c.prepass || (c.processing && c.params.smoothing > 0)) blur = prepass();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.composite), at = p.at;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, blur); gl.uniform1i(at('uBlur'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(at('uSampler'), 0);
    for (const [name, values] of Object.entries(INPUT.geometryUniforms)) gl.uniform4fv(at(name), values);
    const q = c.params;
    gl.uniform4f(at('uFrame'), 1 / W, 1 / H, W / H, c.presence);
    gl.uniform4f(at('uParams'), c.processing ? 1 : 0, blur ? q.smoothing : 0, q.brightness, q.style);
    gl.uniform4f(at('uDetails'), q.skinOnly ? 1 : 0, q.sharpness, q.saturation, q.filterIntensity);
    gl.uniform4f(at('uTone'), q.whiten, q.rosy, c.slim, c.eye);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    checkError('end_to_end ' + c.name);
    return read(0, 0, W, H);
  }
  out.results.endToEnd = {};
  for (const c of INPUT.endToEnd) out.results.endToEnd[c.name] = composite(c);

  // ---- denoise 段：生产降噪程序（TemporalDenoisePass）。历史 = 源图上半（画面顶部）偏移 6/255（噪点量级）、
  // 下半偏移 80/255（运动量级），偏移方向朝向 128，差值不被截断。
  const historyPixels = new Uint8Array(pixels);
  for (let r = 0; r < H; r++) {
    const delta = r >= H / 2 ? INPUT.denoise.noise : INPUT.denoise.motion; // 纹理第 0 行在底部
    for (let x = 0; x < W; x++) for (let i = 0; i < 3; i++) {
      const k = (r * W + x) * 4 + i;
      historyPixels[k] = pixels[k] < 128 ? pixels[k] + delta : pixels[k] - delta;
    }
  }
  const history = texture(W, H, historyPixels);
  function denoise(weight) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.denoise), at = p.at;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, history); gl.uniform1i(at('uHistory'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4f(at('uDenoise'), weight, INPUT.denoise.low, INPUT.denoise.high, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    checkError('denoise ' + weight);
    return read(0, 0, W, H);
  }
  out.results.denoiseOff = denoise(0);
  out.results.denoiseOn = denoise(INPUT.denoise.weight);

  // ---- LUT 纹理：512x512、8x8 格、每格 64x64：格内 x = 红、y = 绿，格序 = 蓝，数据第 0 行即纹理 t = 0
  // （与 Android GLUtils.texImage2D 上传位图一致）。查表在主合成 FRAGMENT_PRO 内，见下方 compositePro 段。
  function lutTexture(map) {
    const data = new Uint8Array(512 * 512 * 4);
    for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) {
      const level = [x % 64, y % 64, Math.floor(y / 64) * 8 + Math.floor(x / 64)].map((v) => Math.round(v * 255 / 63));
      const k = (y * 512 + x) * 4;
      for (let i = 0; i < 3; i++) data[k + i] = map(level[i]);
      data[k + 3] = 255;
    }
    return texture(512, 512, data);
  }
  const identityLut = lutTexture((v) => v), invertLut = lutTexture((v) => 255 - v);
  // ---- body 段：生产美体程序（BodyWarpPass）。胯中点 (0.5, 0.3)，身体方向朝下（度量空间 (0, -1)），躯干 0.3、半宽 0.12。
  function body(a, b, c) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.body), at = p.at;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4fv(at('uBodyA'), a); gl.uniform4fv(at('uBodyB'), b); gl.uniform4fv(at('uBodyC'), c);
    gl.uniform1f(at('uAspect'), W / H);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.bindTexture(gl.TEXTURE_2D, null);
    checkError('body');
    return read(0, 0, W, H);
  }
  const hipBody = [0.5, 0.3, 0, -1];
  out.results.bodyNone = body([0.5, 0.3, 0, 0], [0.3, 0.12, 0.18, 0.25], [0.15, 0.3, 0, 0]);
  out.results.bodyNeutral = body(hipBody, [0.3, 0.12, 0, 0], [0, 0, 0, 0]);
  out.results.bodyNeck = body(hipBody, [0.3, 0.12, 0, 0], [0, 0.3, 0, 0]);
  out.results.bodyShoulders = body(hipBody, [0.3, 0.12, 0, 0], [0.15, 0, 0, 0]);
  out.results.bodyLegs = body([0.5, 0.5, 0, -1.25], [0.3, 0.12, 0, 0], [0, 0, 0, 0]);

  // ---- background 段：生产背景虚化程序（BackgroundBlurPass）。遮罩是 LUMINANCE、第 0 行为画面顶部。
  function maskTexture(rows) {
    const data = new Uint8Array(W * H);
    for (let r = 0; r < H; r++) data.fill(rows(r), r * W, (r + 1) * W);
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, W, H, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return t;
  }
  const flatPixels = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) flatPixels.set([40, 160, 90, 255], i * 4);
  const flat = texture(W, H, flatPixels);
  function background(blurred, mask, strength) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.bgMix), at = p.at;
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, mask); gl.uniform1i(at('uMask'), 2);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, blurred); gl.uniform1i(at('uBlurred'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4f(at('uBackground'), strength, 0.3, 0.7, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    for (const unit of [gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
    checkError('background ' + strength);
    return read(0, 0, W, H);
  }
  const allBackground = maskTexture(() => 0), allPerson = maskTexture(() => 255);
  out.results.bgOff = background(flat, allBackground, 0);
  out.results.bgPerson = background(flat, allPerson, 1);
  out.results.bgHalf = background(flat, maskTexture((r) => (r < H / 2 ? 255 : 0)), 1);
  // 高斯：纯色小图模糊后应不变（权重和为 1）。
  const BW = Math.floor(W / 4), BH = Math.floor(H / 4);
  const smallFlat = new Uint8Array(BW * BH * 4);
  for (let i = 0; i < BW * BH; i++) smallFlat.set([40, 160, 90, 255], i * 4);
  const smallIn = texture(BW, BH, smallFlat), smallOut = texture(BW, BH, null), smallFrame = framebuffer(smallOut);
  gl.bindFramebuffer(gl.FRAMEBUFFER, smallFrame); gl.viewport(0, 0, BW, BH);
  {
    const p = use(P.bgBlur), at = p.at;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, smallIn); gl.uniform1i(at('uSampler'), 0);
    gl.uniform2f(at('uStep'), 3 / BW, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  checkError('bgBlur');
  out.results.bgBlurFlat = read(0, 0, BW, BH);

  // 预乘背景消黑边验证：左半深黑人像 (10,10,10) + 右半亮蓝背景 (80,170,235)。
  const splitBgPixels = new Uint8Array(W * H * 4);
  const splitBgMaskData = new Uint8Array(W * H);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    const isPerson = x < W / 2;
    splitBgPixels.set(isPerson ? [10, 10, 10, 255] : [80, 170, 235, 255], (r * W + x) * 4);
    splitBgMaskData[r * W + x] = isPerson ? 255 : 0;
  }
  const splitBgSrc = texture(W, H, splitBgPixels);
  const splitBgMask = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, splitBgMask);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, W, H, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, splitBgMaskData);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.bindTexture(gl.TEXTURE_2D, null);
  const bgS0 = texture(BW, BH, null), bgF0 = framebuffer(bgS0);
  const bgS1 = texture(BW, BH, null), bgF1 = framebuffer(bgS1);
  gl.bindFramebuffer(gl.FRAMEBUFFER, bgF0); gl.viewport(0, 0, BW, BH);
  {
    const p = use(P.bgPremask), at = p.at;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, splitBgMask); gl.uniform1i(at('uMask'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, splitBgSrc); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4f(at('uBackground'), 1.0, 0.42, 0.68, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
  }
  for (const spread of [1.5, 3.0]) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, bgF1); gl.viewport(0, 0, BW, BH);
    const ph = use(P.bgBlur);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, bgS0); gl.uniform1i(ph.at('uSampler'), 0);
    gl.uniform2f(ph.at('uStep'), spread / BW, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    ph.done();
    gl.bindFramebuffer(gl.FRAMEBUFFER, bgF0); gl.viewport(0, 0, BW, BH);
    const pv = use(P.bgBlur);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, bgS1); gl.uniform1i(pv.at('uSampler'), 0);
    gl.uniform2f(pv.at('uStep'), 0, spread / BH);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    pv.done();
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
  {
    const pm = use(P.bgMix), at = pm.at;
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, splitBgMask); gl.uniform1i(at('uMask'), 2);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, bgS0); gl.uniform1i(at('uBlurred'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, splitBgSrc); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4f(at('uBackground'), 1.0, 0.42, 0.68, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    pm.done();
    for (const unit of [gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
  }
  checkError('bgNoHalo');
  out.results.bgNoHalo = read(0, 0, W, H);

  // ---- guidedFilter + skinMask 段（1/3 分辨率导向滤波、全分辨率重构 + 468 点多边形语义遮罩）。
  const GW = Math.ceil(W / 3), GH = Math.ceil(H / 3);
  const gT = [0, 1, 2].map(() => texture(GW, GH, null)), gF = gT.map(framebuffer);
  // 与 GuidedFilterPrePass.render 相同：spread = GuidedSpread.DEFAULT，eps = BeautyQualityTuning.guidedEps(smoothing)。
  function guidedEps(smoothing) {
    const g = INPUT.guided, t = Math.max(0, Math.min(1, smoothing));
    const sigma = g.energy * Math.sqrt(3) * (g.lo + (g.hi - g.lo) * t) / 255;
    return sigma * sigma;
  }
  // 1/3 分辨率缓冲：b = {gw, gh, gT, gF}，缺省为上面 W×H 的那一组。
  function guidedPrepass(inTex, smoothing, w = W, h = H, b = {gw: GW, gh: GH, gT, gF}) {
    const dx = INPUT.guided.spread / b.gw, dy = INPUT.guided.spread / b.gh;
    const eps = guidedEps(smoothing);
    gl.viewport(0, 0, b.gw, b.gh);
    // P1 DOWN：抽头间距 BeautyQualityTuning.DOWN_TAP_SCALE 原始像素（门禁已核对 GuidedFilterPrePass 用的就是它）。
    pass(P.down, b.gF[0], inTex, null, (at) => gl.uniform2f(at('uTexel'), INPUT.guided.downTap / w, INPUT.guided.downTap / h));
    pass(P.blurRgb, b.gF[1], b.gT[0], null, (at) => gl.uniform2f(at('uDir'), dx, 0));
    pass(P.guidedDiff, b.gF[2], b.gT[1], b.gT[0], (at) => { gl.uniform2f(at('uDir'), 0, dy); gl.uniform1i(at('uOrig'), 1); });
    pass(P.blurAlpha, b.gF[1], b.gT[2], null, (at) => gl.uniform2f(at('uDir'), dx, 0));
    pass(P.guidedCoeff, b.gF[0], b.gT[1], null, (at) => { gl.uniform2f(at('uDir'), 0, dy); gl.uniform1f(at('uGuidedEps'), eps); });
    pass(P.blurRgb, b.gF[1], b.gT[0], null, (at) => gl.uniform2f(at('uDir'), dx, 0));
    pass(P.blurRgb, b.gF[2], b.gT[1], null, (at) => gl.uniform2f(at('uDir'), 0, dy));
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    return b.gT[2];
  }
  // 与 GuidedFilterPrePass.renderWeighted 相同的头发加权路径：P1 → P2'（按非头发权重预乘横向模糊）→ P3'（归一化均值与能量）
  // → P4 → P5'（头发取恒等系数）→ P6/P7；权重纹理（皮肤遮罩，A = 非头发）绑纹理单元 2。
  function guidedPrepassWeighted(inTex, smoothing, w, h, b, weightTex) {
    const dx = INPUT.guided.spread / b.gw, dy = INPUT.guided.spread / b.gh;
    const eps = guidedEps(smoothing);
    gl.viewport(0, 0, b.gw, b.gh);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, weightTex);
    pass(P.down, b.gF[0], inTex, null, (at) => gl.uniform2f(at('uTexel'), INPUT.guided.downTap / w, INPUT.guided.downTap / h));
    pass(P.guidedWeightBlur, b.gF[1], b.gT[0], null, (at) => { gl.uniform2f(at('uDir'), dx, 0); gl.uniform1i(at('uWeight'), 2); });
    pass(P.guidedWeightDiff, b.gF[2], b.gT[1], b.gT[0], (at) => { gl.uniform2f(at('uDir'), 0, dy); gl.uniform1i(at('uOrig'), 1); });
    pass(P.blurAlpha, b.gF[1], b.gT[2], null, (at) => gl.uniform2f(at('uDir'), dx, 0));
    pass(P.guidedWeightCoeff, b.gF[0], b.gT[1], null, (at) => {
      gl.uniform2f(at('uDir'), 0, dy); gl.uniform1f(at('uGuidedEps'), eps); gl.uniform1i(at('uWeight'), 2);
    });
    pass(P.blurRgb, b.gF[1], b.gT[0], null, (at) => gl.uniform2f(at('uDir'), dx, 0));
    pass(P.blurRgb, b.gF[2], b.gT[1], null, (at) => gl.uniform2f(at('uDir'), 0, dy));
    for (const unit of [gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
    return b.gT[2];
  }
  // 无多边形遮罩时的占位：与生产回退一致，A = 255（非头发权重为 1），RGB 为 0。
  const emptyMaskTex = texture(4, 4, new Uint8Array(4 * 4 * 4).map((_, i) => (i % 4 === 3 ? 255 : 0)));
  // ---- compositePro 整帧：与 drawFilter 第 4 段一致（unit 0 源图、1 导向系数、2 皮肤遮罩、3 LUT），几何 uniform 取打包值。
  const PRO_NEUTRAL = {input: null, blur: undefined, flags: 16, smoothing: 0, brightness: 0, skinTexture: 0, radiance: 0,
    sharpness: 0, sat: 1, filterInt: 0, whiten: 0, rosy: 0, slim: 0, eye: 0, presence: 0, grade: [0, 0.005, 0, 0], lut: null};
  function compositeProFrame(options) {
    const o = Object.assign({}, PRO_NEUTRAL, options);
    const input = o.input || source;
    const blur = o.blur !== undefined ? o.blur : (o.smoothing > 0 ? guidedPrepass(input, o.smoothing) : null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.compositePro), at = p.at;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, o.lut || emptyMaskTex); gl.uniform1i(at('uLut'), 3);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, emptyMaskTex); gl.uniform1i(at('uSkinMask'), 2);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, blur); gl.uniform1i(at('uBlur'), 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, input); gl.uniform1i(at('uSampler'), 0);
    for (const [name, values] of Object.entries(INPUT.geometryUniforms)) {
      const loc = at(name);
      if (loc !== null) gl.uniform4fv(loc, values);
    }
    gl.uniform4f(at('uFrame'), 1 / W, 1 / H, W / H, o.presence);
    gl.uniform4f(at('uParams'), o.flags, blur ? o.smoothing : 0, o.brightness, o.skinTexture);
    gl.uniform4f(at('uDetails'), o.radiance, o.sharpness, o.sat, o.filterInt);
    gl.uniform4f(at('uTone'), o.whiten, o.rosy, o.slim, o.eye);
    gl.uniform4f(at('uGrade'), o.grade[0], o.grade[1], o.grade[2], o.grade[3]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    for (const unit of [gl.TEXTURE3, gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
    checkError('compositePro');
    return read(0, 0, W, H);
  }
  // 导向滤波阶跃边 / 斑块：不处理肤色（flags 0）、无遮罩，只看全分辨率重构。
  const runGuidedFull = (inTex, smoothing) => compositeProFrame({input: inTex, flags: 0, smoothing});
  const stepData = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    stepData.set(x < W / 2 ? [25, 22, 22, 255] : [215, 175, 155, 255], (r * W + x) * 4);
  }
  const stepTex = texture(W, H, stepData);
  out.results.guidedStepFull = runGuidedFull(stepTex, 0.55);
  out.results.guidedStepFull09 = runGuidedFull(stepTex, 0.9);
  const blemishData = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    const inSpot = Math.abs(x - W / 2) <= 2 && Math.abs(r - H / 2) <= 2;
    blemishData.set(inSpot ? [185, 140, 122, 255] : [215, 175, 155, 255], (r * W + x) * 4);
  }
  out.results.guidedBlemishFull = runGuidedFull(texture(W, H, blemishData), 0.9);
  // 导向系数曲线（单元级）：阶跃边与斑块只看 a 的两端，近二值映射在两端反而更好，拦不住；这里直接看过渡段。
  // 4×4 色块的 alpha 恒为 s（blur5 权重和为 1，模糊后仍是 s），单跑 guidedCoeff，回读中心像素的 a。
  // 取样点由 glsl_gate.py 的 GUIDED_COEFF_CURVE 给出：[磨皮档, s 相对 s0 = √eps 的倍数, 或 s 的绝对值]，
  // eps = guidedEps(磨皮档)（常量解析自 BeautyQualityTuning.kt）。按字节上传，回传量化后的 s 供 glsl_gate.py 折算。
  {
    const coeffFrame = framebuffer(texture(4, 4, null));
    out.results.guidedCoeffCurve = INPUT.curve.map(([smoothing, factor, absolute]) => {
      const eps = guidedEps(smoothing), s = absolute === null ? factor * Math.sqrt(eps) : absolute;
      const s8 = Math.round(s * 255), data = new Uint8Array(4 * 4 * 4);
      for (let i = 0; i < 16; i++) data.set([215, 175, 155, s8], i * 4);
      gl.viewport(0, 0, 4, 4);
      pass(P.guidedCoeff, coeffFrame, texture(4, 4, data), null, (at) => {
        gl.uniform2f(at('uDir'), 0, 1 / 4); gl.uniform1f(at('uGuidedEps'), eps);
      });
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {eps, s8, rgba: read(1, 1, 1, 1)};
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    checkError('guidedCoeffCurve');
  }

  // 多边形 SkinMaskPass 渲染验证（R=skin, G=excl, B=hiPlane + H/V 羽化，在 WxH 遮罩尺寸下对齐真机 180x320 面部像素占比）。
  {
    const MW = W, MH = H;
    const mT0 = texture(MW, MH, null), mF0 = framebuffer(mT0);
    const mT1 = texture(MW, MH, null), mF1 = framebuffer(mT1);
    const rects = [
      // 全脸皮肤底块
      [0.15, 0.15, 0.85, 0.85, 1.0, 0.0, 0.10],
      // 眼裂挖空保护块 (v in 0.58..0.72 -> 翻转后 r≈0.35*MH)
      [0.36, 0.56, 0.64, 0.74, 0.0, 1.0, 0.0],
      // 鼻梁骨相高光块 (v in 0.28..0.42 -> 翻转后 r≈0.65*MH)
      [0.40, 0.26, 0.60, 0.44, 1.0, 0.0, 0.96],
    ];
    const verts = [], idxs = [];
    rects.forEach(([u0, v0, u1, v1, wr, wg, wb], k) => {
      const b = k * 4;
      verts.push(
        u0 * 2 - 1, v0 * 2 - 1, wr, wg, wb,
        u1 * 2 - 1, v0 * 2 - 1, wr, wg, wb,
        u0 * 2 - 1, v1 * 2 - 1, wr, wg, wb,
        u1 * 2 - 1, v1 * 2 - 1, wr, wg, wb
      );
      idxs.push(b, b + 1, b + 3, b, b + 3, b + 2);
    });
    const mvb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, mvb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    const mib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idxs), gl.STATIC_DRAW);
    gl.bindFramebuffer(gl.FRAMEBUFFER, mF0); gl.viewport(0, 0, MW, MH);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(P.skinMask);
    const aPos = gl.getAttribLocation(P.skinMask, 'aPosition');
    const aW = gl.getAttribLocation(P.skinMask, 'aMaskWeights');
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 20, 0); gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aW, 3, gl.FLOAT, false, 20, 8); gl.enableVertexAttribArray(aW);
    gl.drawElements(gl.TRIANGLES, idxs.length, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(aPos); gl.disableVertexAttribArray(aW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    pass(P.blurRgb, mF1, mT0, null, (at) => gl.uniform2f(at('uDir'), 0.85 / MW, 0));
    pass(P.blurRgb, mF0, mT1, null, (at) => gl.uniform2f(at('uDir'), 0, 0.85 / MH));
    gl.bindFramebuffer(gl.FRAMEBUFFER, mF0);
    checkError('skinMask');
    out.results.skinMaskPoly = read(0, 0, MW, MH);
  }

  // ---- face 段：生产更多脸型程序（FaceShapePass）。脸横向 u = (1, 0)，额头方向 v = (0, 1)；中心为纹理坐标，半径为度量。
  function face(chin, forehead, nose, retouch = [0, 0, 0], input = source) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const p = use(P.face), at = p.at;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, input); gl.uniform1i(at('uSampler'), 0);
    gl.uniform4f(at('uChin'), 0.5, 0.35, 0.12, chin);
    gl.uniform4f(at('uForehead'), 0.5, 0.7, 0.15, forehead);
    gl.uniform4f(at('uNose'), 0.5, 0.5, 0.08, nose);
    gl.uniform4f(at('uFaceAxis'), 1, 0, W / H, 0);
    // 双眼 (0.4, 0.6) / (0.6, 0.6)，横半径 0.06、竖半径 0.03（度量）；嘴 (0.5, 0.35)，0.08 × 0.04。
    gl.uniform4f(at('uEye0'), 0.4, 0.6, 0.06, 0.03);
    gl.uniform4f(at('uEye1'), 0.6, 0.6, 0.06, 0.03);
    gl.uniform4f(at('uMouth'), 0.5, 0.35, 0.08, 0.04);
    gl.uniform4f(at('uRetouch'), retouch[0], retouch[1], retouch[2], retouch[3] || 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    p.done();
    gl.bindTexture(gl.TEXTURE_2D, null);
    checkError('face');
    return read(0, 0, W, H);
  }
  out.results.faceOff = face(0, 0, 0);
  out.results.faceChin = face(0.25, 0, 0);
  out.results.faceForehead = face(0, 0.2, 0);
  out.results.faceNose = face(0, 0, 0.3);
  out.results.faceDark = face(0, 0, 0, [1, 0, 0]);
  out.results.faceBright = face(0, 0, 0, [0, 1, 0]);
  out.results.faceTeeth = face(0, 0, 0, [0, 0, 1]);
  // 白牙选择性：左半是饱和的唇红，右半是偏黄的牙色。
  const mouthPixels = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) mouthPixels.set(x < W / 2 ? [200, 40, 50, 255] : [200, 190, 150, 255], (r * W + x) * 4);
  out.results.faceTeethSplit = face(0, 0, 0, [0, 0, 1], texture(W, H, mouthPixels));
  // 法令纹：纯色皮肤上画一条沿左侧法令纹的暗线（几何同 FACE_FRAGMENT：鼻翼外 1.05 半径 → 嘴角外）。
  const aspect = W / H, a0 = [0.5 - 1.05 * 0.08, 0.5 / aspect], b0 = [0.5 - (0.08 / 1.2 + 0.15 * 0.08), 0.35 / aspect];
  const foldPixels = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    const m = [(x + 0.5) / W, (r + 0.5) / H / aspect]; // 纹理第 0 行在底部
    const ab = [b0[0] - a0[0], b0[1] - a0[1]];
    const t = Math.max(0, Math.min(1, ((m[0] - a0[0]) * ab[0] + (m[1] - a0[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1])));
    const d = Math.hypot(m[0] - a0[0] - ab[0] * t, m[1] - a0[1] - ab[1] * t);
    foldPixels.set(d < 0.006 ? [120, 90, 80, 255] : [200, 160, 140, 255], (r * W + x) * 4);
  }
  out.results.faceFold = face(0, 0, 0, [0, 0, 0, 1], texture(W, H, foldPixels));
  out.results.faceFoldInput = Array.from(foldPixels, (b) => (b < 16 ? '0' : '') + b.toString(16)).join('');

  // ---- compositePro 段：直通、合成内 LUT、回退预处理，以及 beautyColorPro 的逐例探针（用例在 glsl_gate.py）。
  out.results.proOff = compositeProFrame({flags: -1, smoothing: 0.8, brightness: 0.3, skinTexture: 0.5, radiance: 0.5,
    sharpness: 0.4, sat: 1.2, filterInt: 0.5, whiten: 0.7, rosy: 0.5, slim: 0.12, eye: 0.25, presence: 1,
    grade: [1, 0.005, 0, 0], lut: invertLut});
  out.results.proNeutral = compositeProFrame({presence: 1});
  out.results.lutOff = compositeProFrame({lut: invertLut, grade: [0, 0.005, 0, 0]});
  out.results.lutIdentity = compositeProFrame({lut: identityLut, grade: [1, 0.005, 0, 0]});
  out.results.lutInvert = compositeProFrame({lut: invertLut, grade: [1, 0.005, 0, 0]});
  // 导向滤波故障回退 BeautyPrePass：uBlur 绑 (mean, var)。coring 门限取 1.0 关掉质感回填与锐化，只看重构本身。
  const fallbackBlur = prepass();
  out.results.fallbackPrepass = compositeProFrame({blur: fallbackBlur, flags: 0, smoothing: 0.8, grade: [0, 1.0, 1, 0]});
  out.results.fallbackPrepassLegacy = compositeProFrame({blur: prepass(), flags: 0, smoothing: 0.8, grade: [0, 1.0, 0, 0]});
  const proUniforms = (at, c) => {
    gl.uniform4f(at('dI'), c.I[0], c.I[1], c.I[2], 1.0);
    gl.uniform4fv(at('dGuided'), c.guided);
    gl.uniform4f(at('dFine'), c.fine[0], c.fine[1], c.fine[2], 1.0);
    gl.uniform4f(at('dWide'), c.wide[0], c.wide[1], c.wide[2], 1.0);
    gl.uniform4f(at('dMask'), c.mask[0], c.mask[1], c.mask[2], c.mask[3]);
    gl.uniform4fv(at('dParams'), c.params);
    gl.uniform4fv(at('dDetails'), c.details);
    gl.uniform4fv(at('dTone'), c.tone);
    gl.uniform4fv(at('dGrade'), c.grade);
  };
  out.results.proCases = grid(P['debug:compositePro'], INPUT.proCases, proUniforms);
  out.results.proNatural = grid(P['debug:compositePro'], INPUT.naturalCases, proUniforms);
  out.results.proHairA = grid(P['debug:compositePro'], INPUT.hairACases, proUniforms);
  checkError('compositeProCases');

  // ---- meshWarp 段（M3 2D 三角网格微整）：5x5 索引网格（外圈与缓冲环固定零位移），验证恒等直通与局部变形+背景零弯曲。
  function runMeshWarp(shiftCenter) {
    const N = 5;
    const verts = [];
    for (let j = 0; j < N; j++) {
      const sy = j / (N - 1);
      for (let i = 0; i < N; i++) {
        const sx = i / (N - 1);
        let dx = sx, dy = sy;
        if (shiftCenter && i === 2 && j === 2) { dx += 0.08; dy += 0.04; }
        verts.push(dx * 2 - 1, dy * 2 - 1, sx, sy);
      }
    }
    const idxs = [];
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
        idxs.push(a, b, d, a, d, c);
      }
    }
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idxs), gl.STATIC_DRAW);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    const prog = P.meshWarp;
    gl.useProgram(prog);
    const aPos = gl.getAttribLocation(prog, 'aPosition');
    const aUv = gl.getAttribLocation(prog, 'aTextureCoord');
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0); gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8); gl.enableVertexAttribArray(aUv);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source);
    gl.uniform1i(gl.getUniformLocation(prog, 'uSampler'), 0);
    gl.drawElements(gl.TRIANGLES, idxs.length, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(aPos); gl.disableVertexAttribArray(aUv);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    checkError('meshWarp ' + shiftCenter);
    return read(0, 0, W, H);
  }
  out.results.meshIdentity = runMeshWarp(false);
  out.results.meshWarped = runMeshWarp(true);

  // ---- makeup 段（M4 实时局部美妆）：验证 SoftLight/Multiply/ScreenLift、口红丝绒/水光、美瞳闭眼门控与眼神光。
  const makeupQuad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, makeupQuad);
  // x, y, screenU, screenV, makeupU, makeupV（6 floats / vertex，与 FaceMakeupUv.packVertices 一致）
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 0, 0, 0, 0,
     1, -1, 1, 0, 1, 0,
    -1,  1, 0, 1, 0, 1,
     1,  1, 1, 1, 1, 1,
  ]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);

  function solidTex(rgba) {
    const data = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) data.set(rgba, i * 4);
    return texture(W, H, data);
  }
  const skinBaseTex = solidTex([200, 160, 140, 255]);
  const eyeBaseTex = solidTex([45, 38, 36, 255]);
  const fullMaskTex = solidTex([255, 255, 255, 255]);
  // 唇带测试贴图：左半 B=0（非唇区/口腔），右半 B=255（唇带）。
  const lipMaskData = new Uint8Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    lipMaskData.set([0, 0, x >= W / 2 ? 255 : 0, 255], (r * W + x) * 4);
  }
  const lipMaskTex = texture(W, H, lipMaskData);
  const irisAtlasTex = solidTex([210, 145, 70, 220]);

  function runMakeup(program, baseTex, atlasTex, setUniforms) {
    // 先将底图原样画入 fullFrame（与 MakeupPass 先 copy 底图再 GL_BLEND 叠加一致）。
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullFrame); gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    {
      const cp = use(P.original);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, baseTex);
      gl.uniform1i(cp.at('uSampler'), 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      cp.done();
    }
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
    gl.bindBuffer(gl.ARRAY_BUFFER, makeupQuad);
    gl.useProgram(program);
    const aPos = gl.getAttribLocation(program, 'aPosition');
    const aScreen = gl.getAttribLocation(program, 'aScreenUv');
    const aMakeup = gl.getAttribLocation(program, 'aMakeupUv');
    if (aPos >= 0) { gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 24, 0); gl.enableVertexAttribArray(aPos); }
    if (aScreen >= 0) { gl.vertexAttribPointer(aScreen, 2, gl.FLOAT, false, 24, 8); gl.enableVertexAttribArray(aScreen); }
    if (aMakeup >= 0) { gl.vertexAttribPointer(aMakeup, 2, gl.FLOAT, false, 24, 16); gl.enableVertexAttribArray(aMakeup); }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, baseTex);
    gl.uniform1i(gl.getUniformLocation(program, 'uBase'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, atlasTex);
    gl.uniform1i(gl.getUniformLocation(program, 'uAtlas'), 1);
    setUniforms((name) => gl.getUniformLocation(program, name));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (aPos >= 0) gl.disableVertexAttribArray(aPos);
    if (aScreen >= 0) gl.disableVertexAttribArray(aScreen);
    if (aMakeup >= 0) gl.disableVertexAttribArray(aMakeup);
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    checkError('makeup');
    return read(0, 0, W, H);
  }

  out.results.makeupLayerOff = runMakeup(P.makeupLayer, skinBaseTex, fullMaskTex, (at) => {
    gl.uniform4f(at('uColor'), 0.96, 0.54, 0.58, 0.0);
    gl.uniform4f(at('uChannel'), 1, 0, 0, 0);
    gl.uniform4f(at('uMode'), 0, 0, 0, 0);
  });
  out.results.makeupLayerBlush = runMakeup(P.makeupLayer, skinBaseTex, fullMaskTex, (at) => {
    gl.uniform4f(at('uColor'), 0.96, 0.54, 0.58, 0.55);
    gl.uniform4f(at('uChannel'), 1, 0, 0, 0);
    gl.uniform4f(at('uMode'), 0, 0, 0, 0);
  });
  out.results.makeupLayerContour = runMakeup(P.makeupLayer, skinBaseTex, fullMaskTex, (at) => {
    gl.uniform4f(at('uColor'), 0.34, 0.25, 0.20, 0.45);
    gl.uniform4f(at('uChannel'), 1, 0, 0, 0);
    gl.uniform4f(at('uMode'), 1, 0, 0, 0);
  });
  out.results.makeupLayerHighlight = runMakeup(P.makeupLayer, skinBaseTex, fullMaskTex, (at) => {
    gl.uniform4f(at('uColor'), 1.0, 0.96, 0.92, 0.50);
    gl.uniform4f(at('uChannel'), 0, 1, 0, 0);
    gl.uniform4f(at('uMode'), 2, 0, 0, 0);
  });
  out.results.makeupLipVelvet = runMakeup(P.makeupLip, skinBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.73, 0.32, 0.33, 0.72);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipGlossy = runMakeup(P.makeupLip, skinBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.73, 0.32, 0.33, 0.72);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  // 深紫暗沉裸唇去乌显色验证 + L1–L5 物理透色柔焦唇防胶布/防发粉探针。
  const darkLipBaseTex = solidTex([105, 68, 72, 255]);
  out.results.makeupLipDarkBase = runMakeup(P.makeupLip, darkLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.86, 0.36, 0.27, 0.85);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  // L1: 闭合唇缝暗影 (45,25,27) 与口腔暗腔 (22,14,15) 在高强度水光下绝不抬亮发粉。
  const seamBaseTex = solidTex([45, 25, 27, 255]);
  const cavityBaseTex = solidTex([22, 14, 15, 255]);
  out.results.makeupLipSeamGlossy = runMakeup(P.makeupLip, seamBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.86, 0.36, 0.27, 0.85);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipCavityGlossy = runMakeup(P.makeupLip, cavityBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.73, 0.32, 0.33, 0.85);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  // L2: 白色露齿边缘 (222,216,206) 涂车厘子红绝不染粉。
  const teethBaseTex = solidTex([222, 216, 206, 255]);
  out.results.makeupLipTeethGuard = runMakeup(P.makeupLip, teethBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.70, 0.09, 0.11, 0.85);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  // L3: 外缘羽化区 (feather≈0.45) 嘴周暖黄肤 (195,156,120) vs 真实裸唇 (145,96,95) 自适应门控对比。
  const rimMaskTex = solidTex([0, 0, 115, 255]);
  const warmSkinBaseTex = solidTex([195, 156, 120, 255]);
  const bareLipBaseTex = solidTex([145, 96, 95, 255]);
  out.results.makeupLipRimSkin = runMakeup(P.makeupLip, warmSkinBaseTex, rimMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.70, 0.09, 0.11, 0.75);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipRimLip = runMakeup(P.makeupLip, bareLipBaseTex, rimMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.70, 0.09, 0.11, 0.75);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  // L4: 水光模式下 4 款代表色号（正红/橘汽/奶茶/梅子）色相区分度验证（杜绝全部发粉）。
  out.results.makeupLipGlossyCherry = runMakeup(P.makeupLip, bareLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.70, 0.09, 0.11, 0.65);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipGlossyOrange = runMakeup(P.makeupLip, bareLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.88, 0.30, 0.10, 0.65);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipGlossyMilkTea = runMakeup(P.makeupLip, bareLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.64, 0.39, 0.27, 0.65);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipGlossyBerry = runMakeup(P.makeupLip, bareLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.54, 0.12, 0.26, 0.65);
    gl.uniform4f(at('uLipStyle'), 1.0, 1 / W, 1 / H, 0);
  });
  // L5: 上唇背光暗面 (105,66,68) vs 下唇受光腹面 (145,96,95) 3D 立体明暗差保留。
  const upperLipBaseTex = solidTex([105, 66, 68, 255]);
  out.results.makeupLipUpperShadow = runMakeup(P.makeupLip, upperLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.73, 0.32, 0.33, 0.65);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  out.results.makeupLipLowerLit = runMakeup(P.makeupLip, bareLipBaseTex, lipMaskTex, (at) => {
    gl.uniform4f(at('uLipColor'), 0.73, 0.32, 0.33, 0.65);
    gl.uniform4f(at('uLipStyle'), 0.0, 1 / W, 1 / H, 0);
  });
  const splitAtlasData = new Uint8Array(64 * 64 * 4);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    const border = x <= 1 || x >= 62 || y === 0 || y === 31 || y === 32 || y === 63;
    const rgb = border ? [0, 0, 0] : (y < 32 ? [255, 0, 0] : [0, 255, 0]);
    splitAtlasData.set([rgb[0], rgb[1], rgb[2], 255], (y * 64 + x) * 4);
  }
  const splitAtlasTex = texture(64, 64, splitAtlasData);
  out.results.makeupSplitEye = runMakeup(P.makeupLayer, skinBaseTex, splitAtlasTex, (at) => {
    gl.uniform4f(at('uColor'), 0.25, 0.18, 0.15, 0.85);
    gl.uniform4f(at('uChannel'), 1, 0, 0, 0);
    gl.uniform4f(at('uMode'), 1, 1, 0, 0);
  });
  // E1–E4: 眼影皮肤门控 (uMode.z = 1) —— 真实眼皮自然显色，眼白/暗部眼白/瞳孔睫毛/中性背景零染色。
  const eyelidSkinBaseTex = solidTex([200, 160, 140, 255]);
  const scleraBaseTex = solidTex([210, 206, 204, 255]);
  const shadowScleraBaseTex = solidTex([165, 162, 164, 255]);
  const darkIrisBaseTex = solidTex([38, 30, 28, 255]);
  const greyBgBaseTex = solidTex([190, 192, 196, 255]);
  const setEyeshadowUniforms = (at) => {
    gl.uniform4f(at('uColor'), 0.50, 0.22, 0.28, 0.65);
    gl.uniform4f(at('uChannel'), 1, 0, 0, 0);
    gl.uniform4f(at('uMode'), 1, 0, 1, 0);
  };
  out.results.makeupEyeshadowSkin = runMakeup(P.makeupLayer, eyelidSkinBaseTex, fullMaskTex, setEyeshadowUniforms);
  out.results.makeupEyeshadowSclera = runMakeup(P.makeupLayer, scleraBaseTex, fullMaskTex, setEyeshadowUniforms);
  out.results.makeupEyeshadowShadowSclera = runMakeup(P.makeupLayer, shadowScleraBaseTex, fullMaskTex, setEyeshadowUniforms);
  out.results.makeupEyeshadowDarkIris = runMakeup(P.makeupLayer, darkIrisBaseTex, fullMaskTex, setEyeshadowUniforms);
  out.results.makeupEyeshadowGreyBg = runMakeup(P.makeupLayer, greyBgBaseTex, fullMaskTex, setEyeshadowUniforms);
  out.results.makeupIrisClosed = runMakeup(P.makeupIris, eyeBaseTex, irisAtlasTex, (at) => {
    gl.uniform4f(at('uIris'), 0.5, 0.5, 0.25, W / H);
    gl.uniform4f(at('uIrisTile'), 0.0, 0.0, 0.08, 0.0);
    gl.uniform4f(at('uIrisStrength'), 0.8, 0.85, 0.0, 0.0);
  });
  out.results.makeupIrisOpen = runMakeup(P.makeupIris, eyeBaseTex, irisAtlasTex, (at) => {
    gl.uniform4f(at('uIris'), 0.5, 0.5, 0.25, W / H);
    gl.uniform4f(at('uIrisTile'), 0.0, 0.0, 0.30, 0.0);
    gl.uniform4f(at('uIrisStrength'), 0.8, 0.85, 0.0, 0.0);
  });

  hairEdge(INPUT.hairEdge);
  out.results.hairAlpha = hairAlpha(INPUT.hairAlpha);
  out.results.bgHair = bgHair(INPUT.bgHair);
  out.results.beardBorder = runBeardBorderProbes();
  out.results.v11Upgrade = runV11UpgradeProbes();

  // ---- 毛发边界段（G1–G8）：场景、指标与阈值都在 glsl_gate.py，这里只跑生产程序并按 base64 回读。
  function hairEdge(input) {
    const fromB64 = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    const toB64 = (bytes) => {
      let text = '';
      for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      return btoa(text);
    };
    const readBytes = (x, y, w, h) => {
      const bytes = new Uint8Array(w * h * 4);
      gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      return bytes;
    };
    const solid = (w, h, rgba) => {
      const data = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) data.set(rgba, i * 4);
      return data;
    };
    out.results.hairEdge = {frames: {}, bg: {}};

    // G1–G7：按 GuidedFilterPrePass.render + drawFilter 第 4 段跑（多边形遮罩路径，presence = 1，uGrade = 室内光）。
    // 场景按行原样上传、原样回读，与设计组探针一致；这些度量与上下方向无关。皮肤遮罩是与场景同尺寸的常量纹理
    // （scene.mask 给 8 个数时左半 / 右半各 4 个，头发遮罩 A3 / A4 用）；scene.maskA 给出时改为 scene.maskSize 尺寸（帧的 1/4，
    // 同 SkinMaskPass 输出）、RGB 取 scene.mask 前 3 个、A 按列取 scene.maskA（G1 矩阵的头发遮罩）。scene.frameW 缺省为 5
    // （多边形遮罩、presence = 1）。scene.looks 走原导向滤波；scene.weighted 里的档位另按头发加权路径跑一遍（权重 = 皮肤遮罩
    // 的 A），帧名后缀 |w。只回传 input.picks[scene.pick] 列出的像素（行优先序号；null 为整帧）的 RGB。
    const rgbAt = (bytes, pick, count) => {
      const indices = pick || Array.from({length: count}, (_, i) => i), rgb = new Uint8Array(indices.length * 3);
      indices.forEach((k, n) => rgb.set(bytes.subarray(k * 4, k * 4 + 3), n * 3));
      return rgb;
    };
    const sized = {}, masks = {};
    for (const scene of input.scenes) {
      const w = scene.w, h = scene.h, key = w + 'x' + h;
      if (!sized[key]) {
        const gw = Math.ceil(w / 3), gh = Math.ceil(h / 3), t = [0, 1, 2].map(() => texture(gw, gh, null));
        sized[key] = {gw, gh, gT: t, gF: t.map(framebuffer), full: framebuffer(texture(w, h, null))};
      }
      const b = sized[key], maskKey = scene.maskA ? 'A:' + scene.name : key + ':' + scene.mask.join(',');
      if (!masks[maskKey]) {
        if (scene.maskA) {
          const [mw, mh] = scene.maskSize, data = new Uint8Array(mw * mh * 4);
          for (let r = 0; r < mh; r++) for (let x = 0; x < mw; x++) data.set([scene.mask[0], scene.mask[1], scene.mask[2], scene.maskA[x]], (r * mw + x) * 4);
          masks[maskKey] = texture(mw, mh, data);
        } else {
          const data = solid(w, h, scene.mask.slice(0, 4));
          if (scene.mask.length === 8) for (let r = 0; r < h; r++) for (let x = w / 2; x < w; x++) data.set(scene.mask.slice(4), (r * w + x) * 4);
          masks[maskKey] = texture(w, h, data);
        }
      }
      const source = texture(w, h, fromB64(scene.rgba));
      const runs = scene.looks.map((n) => [n, false]).concat((scene.weighted || []).map((n) => [n, true]));
      for (const [name, weighted] of runs) {
        const look = input.looks[name], smoothing = look.params[1];
        const blur = smoothing > 0 ? (weighted ? guidedPrepassWeighted(source, smoothing, w, h, b, masks[maskKey]) :
          guidedPrepass(source, smoothing, w, h, b)) : null;
        gl.bindFramebuffer(gl.FRAMEBUFFER, b.full); gl.viewport(0, 0, w, h);
        const p = use(P.compositePro), at = p.at;
        gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, emptyMaskTex); gl.uniform1i(at('uLut'), 3);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, masks[maskKey]); gl.uniform1i(at('uSkinMask'), 2);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, blur || source); gl.uniform1i(at('uBlur'), 1);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(at('uSampler'), 0);
        gl.uniform4f(at('uFrame'), 1 / w, 1 / h, w / h, scene.frameW === undefined ? 5.0 : scene.frameW);
        gl.uniform4f(at('uParams'), look.params[0], blur ? smoothing : 0, look.params[2], look.params[3]);
        gl.uniform4fv(at('uDetails'), look.details);
        gl.uniform4fv(at('uTone'), look.tone);
        gl.uniform4f(at('uGrade'), 0, 0.005, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        p.done();
        for (const unit of [gl.TEXTURE3, gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
        out.results.hairEdge.frames[scene.name + '|' + name + (weighted ? '|w' : '')] =
          toB64(rgbAt(readBytes(0, 0, w, h), input.picks[scene.pick], w * h));
      }
      gl.deleteTexture(source);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    checkError('hairEdge frames');

    // G8：按 bg/p_bg.py 栅格化 720×1280 合成场景（4×4 超采样，线段由 glsl_gate.py 给出），模拟人像遮罩，
    // 跑生产 BG_PREMASK → BG_BLUR ×4 → BG_MIX（BackgroundBlurPass.render）。只回传裁剪区里 g.pick 列出的取样点
    // （下标按裁剪区行优先、第 0 行在画面顶部）的 R 通道。
    const g = input.bg, BW = g.width, BH = g.height, SS = g.ss, SW = BW * SS, SH = BH * SS, N = BW * BH;
    const strand = new Uint8Array(SW * SH);
    for (const [ax, ay, bx, by, width] of g.strands) {
      const vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy, half = width / 2;
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - half - 1) * SS)), i1 = Math.min(SW, Math.ceil((Math.max(ax, bx) + half + 1) * SS));
      const j0 = Math.max(0, Math.floor((Math.min(ay, by) - half - 1) * SS)), j1 = Math.min(SH, Math.ceil((Math.max(ay, by) + half + 1) * SS));
      for (let j = j0; j < j1; j++) {
        const yy = (j + 0.5) / SS;
        for (let i = i0; i < i1; i++) {
          const xx = (i + 0.5) / SS, t = Math.min(Math.max(((xx - ax) * vx + (yy - ay) * vy) / l2, 0), 1);
          if (Math.hypot(xx - (ax + t * vx), yy - (ay + t * vy)) <= half) strand[j * SW + i] = 1;
        }
      }
    }
    // 每像素 16 个子样本的计数：人像 / 发团（含细丝）/ 脸 / 衣服与脖子；n = 无细丝，s = 有细丝。
    const count = {n: [0, 1, 2, 3].map(() => new Uint8Array(N)), s: [0, 1, 2, 3].map(() => new Uint8Array(N))};
    const [hx, hy, hrx, hry] = g.head, [fx, fy, frx, fry] = g.face;
    for (let y = 0; y < BH; y++) for (let j = 0; j < SS; j++) {
      const yy = (y * SS + j + 0.5) / SS, ty = (yy - hy) / hry, uy = (yy - fy) / fry;
      for (let x = 0; x < BW; x++) for (let i = 0; i < SS; i++) {
        const xx = (x * SS + i + 0.5) / SS, tx = (xx - hx) / hrx, ux = (xx - fx) / frx, k = y * BW + x;
        const hair = tx * tx + ty * ty <= 1.0, face = ux * ux + uy * uy <= 1.0;
        const body = (yy >= 860.0 && Math.abs(xx - hx) <= 160.0 + (yy - 860.0) * 0.6) ||
          (yy >= 780.0 && yy < 880.0 && Math.abs(xx - hx) <= 70.0);
        const s = strand[(y * SS + j) * SW + x * SS + i] === 1;
        const c = count.n;
        if (hair || face || body) c[0][k]++;
        if (hair && !face) c[1][k]++;
        if (face) c[2][k]++;
        if (body && !hair && !face) c[3][k]++;
        const d = count.s;
        if (hair || face || body || s) d[0][k]++;
        if ((hair && !face) || s) d[1][k]++;
        if (face && !s) d[2][k]++;
        if (body && !hair && !face && !s) d[3][k]++;
      }
    }
    // GL_LINEAR 单次取样式的缩放（像素中心对齐、边缘钳位），同 bg/common.py 的 resize_bilinear。
    function resize(src, w, h, ow, oh) {
      const outImg = new Float64Array(ow * oh);
      const xs = [], ys = [];
      for (let i = 0; i < ow; i++) xs.push(Math.min(Math.max((i + 0.5) * w / ow - 0.5, 0), w - 1));
      for (let j = 0; j < oh; j++) ys.push(Math.min(Math.max((j + 0.5) * h / oh - 0.5, 0), h - 1));
      for (let j = 0; j < oh; j++) {
        const y0 = Math.floor(ys[j]), y1 = Math.min(y0 + 1, h - 1), fy_ = ys[j] - y0;
        for (let i = 0; i < ow; i++) {
          const x0 = Math.floor(xs[i]), x1 = Math.min(x0 + 1, w - 1), fx_ = xs[i] - x0;
          const top = src[y0 * w + x0] * (1 - fx_) + src[y0 * w + x1] * fx_;
          const bottom = src[y1 * w + x0] * (1 - fx_) + src[y1 * w + x1] * fx_;
          outImg[j * ow + i] = top * (1 - fy_) + bottom * fy_;
        }
      }
      return outImg;
    }
    // 先水平再竖直、边缘复制，同 bg/common.py 的 gauss_blur；核由 glsl_gate.py 给出。
    function blurPlane(src, w, h, k) {
      const r = (k.length - 1) / 2, tmp = new Float64Array(w * h), res = new Float64Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let i = 0; i < k.length; i++) sum += k[i] * src[y * w + Math.min(Math.max(x + i - r, 0), w - 1)];
        tmp[y * w + x] = sum;
      }
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let i = 0; i < k.length; i++) sum += k[i] * tmp[Math.min(Math.max(y + i - r, 0), h - 1) * w + x];
        res[y * w + x] = sum;
      }
      return res;
    }
    // 模拟人像遮罩：覆盖率 → 216×384（模型输入）→ 256² → PSF → 216×384 → u8（第 0 行在画面顶部）。
    const [mw, mh, model] = g.mask;
    function personMask(c) {
      const alpha = Float64Array.from(c[0], (v) => v / 16);
      const back = resize(blurPlane(resize(resize(alpha, BW, BH, mw, mh), mw, mh, model, model), model, model, g.psf),
        model, model, mw, mh);
      return Uint8Array.from(back, (v) => Math.min(Math.max(Math.floor(Math.min(Math.max(v, 0), 1) * 255 + 0.5), 0), 255));
    }
    function lumTexture(w, h, data) {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, w, h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return t;
    }
    // np.round（四舍六入五成双）后钳到 0..255。
    const rint = (v) => {
      let r = Math.round(v);
      if (r - v === 0.5 && r % 2 !== 0) r -= 1;
      return Math.min(Math.max(r, 0), 255);
    };
    const [cx0, cy0, cw, ch] = g.crop;
    // 背景：[名字, 低, 高] 为 6px 棋盘；[名字, 低, 高, 'text'] 为文字（8×12 字格里按格号哈希画 1px 横竖笔画，墨 = 低、纸 = 高）。
    function backgroundAt(spec, x, y) {
      const [, lo, hi, kind] = spec;
      if (kind !== 'text') return ((Math.floor(x / 6) + Math.floor(y / 6)) % 2 === 0) ? lo : hi;
      const u = x % 8, v = y % 12, h = (Math.imul(Math.floor(x / 8), 73856093) ^ Math.imul(Math.floor(y / 12), 19349663)) >>> 0;
      const ink = (u >= 1 && u <= 5 && v >= 2 && v <= 10) && (
        (u === 1 && (h & 1)) || (u === 5 && (h & 2)) || (u === 3 && (h & 32)) ||
        (v === 2 && (h & 4)) || (v === 6 && (h & 8)) || (v === 10 && (h & 16)));
      return ink ? lo : hi;
    }
    // 画面：背景·(1 − 人像) + 发团 0.12 + 脸 0.70 + 衣服 0.30，钳到 [0, 1]；纹理第 0 行在底部。
    function picture(c, checker) {
      const gray = new Uint8Array(N), rgba = new Uint8Array(N * 4);
      for (let y = 0; y < BH; y++) for (let x = 0; x < BW; x++) {
        const k = y * BW + x, bg = backgroundAt(checker, x, y);
        const v = bg * (1 - c[0][k] / 16) + g.tones[0] * (c[1][k] / 16) + g.tones[1] * (c[2][k] / 16) + g.tones[2] * (c[3][k] / 16);
        gray[k] = rint(Math.min(Math.max(v, 0), 1) * 255);
        rgba.set([gray[k], gray[k], gray[k], 255], ((BH - 1 - y) * BW + x) * 4);
      }
      const crop = Uint8Array.from(g.pick, (k) => gray[(cy0 + Math.floor(k / cw)) * BW + cx0 + k % cw]);
      return {texture: texture(BW, BH, rgba), crop: toB64(crop)};
    }
    // 与 BackgroundBlurPass.render 相同：1/4 预乘 → 横竖高斯两轮（步长 1.5、3）→ 全分辨率混合；uBackground 取自 BackgroundBlurPass.kt。
    const sw = Math.floor(BW / 4), sh = Math.floor(BH / 4);
    const small = [0, 1].map(() => texture(sw, sh, null)), smallF = small.map(framebuffer);
    const mixed = texture(BW, BH, null), mixedF = framebuffer(mixed);
    // opts.detail / opts.oval / opts.axis 覆盖 β、脸椭圆与轴向（不给时同生产：有人脸、β = DETAIL_KEEP）；opts.full 时回读整帧 R
    // （行优先、第 0 行在顶部）。
    function blurBackground(source, mask, opts = {}) {
      const [strength, low, high, detailKeep] = g.uniforms;
      const detail = opts.detail === undefined ? detailKeep : opts.detail, oval = opts.oval || g.oval, axis = opts.axis || g.axis;
      gl.bindFramebuffer(gl.FRAMEBUFFER, smallF[0]); gl.viewport(0, 0, sw, sh);
      let p = use(P.bgPremask);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mask); gl.uniform1i(p.at('uMask'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(p.at('uSampler'), 0);
      gl.uniform4f(p.at('uBackground'), strength, low, high, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const spread of [1.5, 3.0]) {
        for (const [from, to, step] of [[0, 1, [spread / sw, 0]], [1, 0, [0, spread / sh]]]) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, smallF[to]);
          p = use(P.bgBlur);
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, small[from]); gl.uniform1i(p.at('uSampler'), 0);
          gl.uniform2f(p.at('uStep'), step[0], step[1]);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          p.done();
        }
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, mixedF); gl.viewport(0, 0, BW, BH);
      p = use(P.bgMix);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, mask); gl.uniform1i(p.at('uMask'), 2);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, small[0]); gl.uniform1i(p.at('uBlurred'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(p.at('uSampler'), 0);
      // 有人脸：β = DETAIL_KEEP，脸椭圆供头部先验（BackgroundBlurPass.render 的 face）。
      gl.uniform4f(p.at('uBackground'), strength, low, high, detail);
      gl.uniform4fv(p.at('uFaceOval'), oval); gl.uniform4fv(p.at('uFaceAxis'), axis);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const unit of [gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
      if (opts.full) {
        const all = readBytes(0, 0, BW, BH), r = new Uint8Array(N);
        for (let y = 0; y < BH; y++) for (let x = 0; x < BW; x++) r[y * BW + x] = all[((BH - 1 - y) * BW + x) * 4];
        return r;
      }
      // 回读第 0 行在底部，翻成画面方向再取样。
      const bytes = readBytes(cx0, BH - (cy0 + ch), cw, ch);
      return toB64(Uint8Array.from(g.pick, (k) => bytes[((ch - 1 - Math.floor(k / cw)) * cw + k % cw) * 4]));
    }
    const maskS = lumTexture(mw, mh, personMask(count.s)), maskN = lumTexture(mw, mh, personMask(count.n));
    const zero = lumTexture(mw, mh, new Uint8Array(mw * mh));
    const u8 = (v) => Math.min(Math.max(Math.floor(Math.min(Math.max(v, 0), 1) * 255 + 0.5), 0), 255);
    const gaussKernel = (sig) => {
      const r = Math.ceil(3 * sig), k = [];
      for (let i = -r; i <= r; i++) k.push(Math.exp(-0.5 * (i / sig) ** 2));
      const sum = k.reduce((a, b) => a + b, 0);
      return k.map((v) => v / sum);
    };
    // 第二批合并道（评审 #19）：头发遮罩 = 发团（含细丝、不含脸）覆盖率 → 216×384 → 高斯 σ（遮罩像素，模拟 hair_segmenter 的软边）
    // → u8，跑生产 BG_HAIR（shown = 1、单位仿射、同一脸椭圆），R 平面即 BackgroundBlurPass 改绑的合并遮罩。
    function hairTex(c, sig) {
      const m = blurPlane(resize(Float64Array.from(c[1], (v) => v / 16), BW, BH, mw, mh), mw, mh, gaussKernel(sig));
      return lumTexture(mw, mh, Uint8Array.from(m, u8));
    }
    const merged = [0, 1].map(() => texture(mw, mh, null)), mergedF = merged.map(framebuffer);
    function mergeHair(slot, person, hair) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, mergedF[slot]); gl.viewport(0, 0, mw, mh);
      const p = use(P.bgHair), at = p.at;
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, hair); gl.uniform1i(at('uSemantic'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, person); gl.uniform1i(at('uPerson'), 0);
      gl.uniform4f(at('uHair'), 1, g.hairUniforms[0], g.hairUniforms[1], 0);
      gl.uniform4fv(at('uHairX0'), [1, 0, 0, 0]); gl.uniform4fv(at('uHairX1'), [0, 1, 0, 0]);
      gl.uniform4fv(at('uFaceOval'), g.oval); gl.uniform4fv(at('uFaceAxis'), g.axis);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const unit of [gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
      const bytes = readBytes(0, 0, mw, mh);
      return {texture: merged[slot], plane: Uint8Array.from({length: mw * mh}, (_, k) => bytes[k * 4])};
    }
    const hairSigmas = g.hairSigmas || [];
    const mergedN = hairSigmas.map((sig) => { const t = hairTex(count.n, sig), r = mergeHair(0, maskN, t); gl.deleteTexture(t); return r; });
    out.results.hairEdge.mergedPlanes = mergedN.map((r) => toB64(r.plane));
    for (const checker of g.checkers) {
      const withStrands = picture(count.s, checker), plain = picture(count.n, checker);
      const crops = {in_s: withStrands.crop, in_n: plain.crop};
      // out：带飞发画面 + 带飞发遮罩；cross：无飞发画面 + 同一遮罩；edge：无飞发画面 + 无飞发遮罩；full：全虚化参照。
      crops.out = blurBackground(withStrands.texture, maskS);
      crops.cross = blurBackground(plain.texture, maskS);
      crops.edge = blurBackground(plain.texture, maskN);
      crops.full = blurBackground(plain.texture, zero);
      // β 扫描（评审复核）：β = detailKeep(true, k)（glsl_gate.py 按 BackgroundBlurPass.kt 的写法算好），另跑一次没有人脸
      // （β 0、NO_FACE 椭圆与无人脸轴向，同 BackgroundBlurPass.render 的 face = null 分支）。
      const sweep = g.detailSweep;
      if (sweep && checker[0] === sweep.checker) {
        sweep.betas.forEach((beta, i) => { crops['outB' + i] = blurBackground(withStrands.texture, maskS, {detail: beta}); });
        crops.outNoFace = blurBackground(withStrands.texture, maskS, {detail: 0, oval: sweep.noFace, axis: sweep.noFaceAxis});
      }
      // 合并道各 σ：edgeH 用无飞发合并遮罩，outH / crossH 用带飞发合并遮罩（slot 1，每个 σ 现合现用）。
      hairSigmas.forEach((sig, i) => {
        const t = hairTex(count.s, sig), mS = mergeHair(1, maskS, t);
        gl.deleteTexture(t);
        crops['outH' + i] = blurBackground(withStrands.texture, mS.texture);
        crops['crossH' + i] = blurBackground(plain.texture, mS.texture);
        const tn = hairTex(count.n, sig), mN = mergeHair(1, maskN, tn);
        gl.deleteTexture(tn);
        crops['edgeH' + i] = blurBackground(plain.texture, mN.texture);
      });
      out.results.hairEdge.bg[checker[0]] = crops;
      gl.deleteTexture(withStrands.texture); gl.deleteTexture(plain.texture);
    }
    // G8 时域（评审 #1）：静态画面（带飞发）+ 逐帧抖动的软人像遮罩，统计真背景（覆盖率为 0）里相邻两帧 |ΔR| > 8 的像素数 / 帧。
    // 遮罩模拟真实 selfie 输出：覆盖率按帧做亚像素平移 → 216×384 → 高斯 coreSigma（比静态 G8 的 PSF 宽）→ 再叠一层
    // haloGain·高斯 haloSigma 的低置信光晕（逐帧整体强弱浮动）→ 逐像素噪声。variants：prod = 有人脸、β = DETAIL_KEEP（生产）；
    // old = β 0（细节保留关闭，即 1.24.x 与无人脸）；noprior = β = DETAIL_KEEP 但脸椭圆巨大、头部先验处处为 1（修复前的 D3）。
    // 同时统计头部先验区外（d ≥ 2.8 个脸半径）的部分。
    const tg = g.temporal;
    if (tg) {
      let seed = tg.seed >>> 0;
      const rand = () => {
        seed = (seed + 0x6D2B79F5) >>> 0;
        let t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
      const alpha = Float64Array.from(count.s[0], (v) => v / 16), core = gaussKernel(tg.coreSigma), wide = gaussKernel(tg.haloSigma);
      function shifted(sx, sy) {
        const res = new Float64Array(mw * mh);
        for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) {
          const x = Math.min(Math.max((i + 0.5 + sx) * BW / mw - 0.5, 0), BW - 1), y = Math.min(Math.max((j + 0.5 + sy) * BH / mh - 0.5, 0), BH - 1);
          const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, BW - 1), y1 = Math.min(y0 + 1, BH - 1), fx_ = x - x0, fy_ = y - y0;
          res[j * mw + i] = (alpha[y0 * BW + x0] * (1 - fx_) + alpha[y0 * BW + x1] * fx_) * (1 - fy_) +
            (alpha[y1 * BW + x0] * (1 - fx_) + alpha[y1 * BW + x1] * fx_) * fy_;
        }
        return res;
      }
      const frameMasks = [];
      for (let f = 0; f < tg.frames; f++) {
        const m = blurPlane(shifted((rand() - 0.5) * 2 * tg.shift, (rand() - 0.5) * 2 * tg.shift), mw, mh, core);
        const halo = blurPlane(m, mw, mh, wide), gain = tg.haloGain * (0.75 + 0.5 * rand());
        frameMasks.push(lumTexture(mw, mh, Uint8Array.from(m, (v, k) => u8(Math.max(v, gain * halo[k]) + tg.noise * gauss()))));
      }
      const outside = new Uint8Array(N), [cx, cy, rx, ry] = g.oval, aspect = g.axis[2];
      for (let y = 0; y < BH; y++) for (let x = 0; x < BW; x++) {
        const a = Math.min(Math.max(((x + 0.5) / BW - cx) / rx, -4), 4), b = Math.min(Math.max((1 - (y + 0.5) / BH - cy) / aspect / ry, -4), 4);
        outside[y * BW + x] = Math.sqrt(a * a + b * b) >= 2.8 ? 1 : 0;
      }
      const huge = [0.5, 0.5, 100, 100], result = {};
      for (const spec of tg.backgrounds) {
        const pic = picture(count.s, spec);
        const row = {};
        for (const [name, opts] of [['prod', {}], ['old', {detail: 0}], ['noprior', {oval: huge}]]) {
          let prev = null, all = 0, far = 0;
          for (const mask of frameMasks) {
            const cur = blurBackground(pic.texture, mask, Object.assign({full: true}, opts));
            if (prev) for (let k = 0; k < N; k++) {
              if (count.s[0][k] === 0 && Math.abs(cur[k] - prev[k]) > 8) { all++; if (outside[k]) far++; }
            }
            prev = cur;
          }
          row[name] = [all / (tg.frames - 1), far / (tg.frames - 1)];
        }
        result[spec[0]] = row;
        gl.deleteTexture(pic.texture);
      }
      frameMasks.forEach((t) => gl.deleteTexture(t));
      out.results.hairEdge.temporal = result;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    checkError('hairEdge background');
  }

  // LUMINANCE 遮罩纹理（数据第 0 行即纹理 t = 0；头发 / 人像遮罩约定第 0 行是画面顶部）。
  function luminance(w, h, data) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, w, h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return t;
  }
  function bytesToB64(bytes) {
    let text = '';
    for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(text);
  }

  // ---- 头发遮罩 HA1：生产 HAIR_ALPHA 程序，按 SemanticAlphaPass.render 的顺序画进带 RGB 图样的目标（SkinMaskPass 输出的
  // 布局，纹理 t = 1 是画面顶部）：colorMask(F, F, F, T)、关混合 → 全屏四边形 → 可选网格（开背景面剔除）→ 恢复 colorMask。
  function hairAlpha(input) {
    const N = input.size, [lo, hi, bias] = input.uniforms;
    const pattern = new Uint8Array(N * N * 4);
    for (let t = 0; t < N; t++) for (let x = 0; x < N; x++) pattern.set([(x * 4) & 255, (t * 4) & 255, 77, 255], (t * N + x) * 4);
    // 头发图样（第 0 行是画面顶部）：c128 / c230 / c255 常数，top 上半，left 左半，blob 中心半径 0.05 的圆。
    const shapes = {
      c128: () => 128, c230: () => 230, c255: () => 255,
      top: (x, r) => (r < N / 2 ? 255 : 0), left: (x) => (x < N / 2 ? 255 : 0),
      blob: (x, r) => (Math.hypot((x + 0.5) / N - 0.5, (r + 0.5) / N - 0.5) <= 0.05 ? 255 : 0),
    };
    const hairTex = {};
    for (const [name, fn] of Object.entries(shapes)) {
      const data = new Uint8Array(N * N);
      for (let r = 0; r < N; r++) for (let x = 0; x < N; x++) data[r * N + x] = fn(x, r);
      hairTex[name] = luminance(N, N, data);
    }
    // 网格：同 meshWarp 用例的 5×5 网格（dst NDC、src 纹理坐标），中心顶点的落点移动 (+0.08, +0.04)，外圈不动。
    const meshVerts = [], meshIdx = [];
    for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) {
      const sx = i / 4, sy = j / 4, shift = i === 2 && j === 2;
      meshVerts.push((sx + (shift ? 0.08 : 0)) * 2 - 1, (sy + (shift ? 0.04 : 0)) * 2 - 1, sx, sy);
    }
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
      const a = j * 5 + i, b = a + 1, c = a + 5, d = c + 1;
      meshIdx.push(a, b, d, a, d, c);
    }
    const meshVb = gl.createBuffer(), meshIb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, meshVb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshVerts), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshIb); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(meshIdx), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const result = {cases: {}};
    for (const c of input.cases) {
      const target = texture(N, N, pattern), frame = framebuffer(target);
      gl.viewport(0, 0, N, N);
      gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE);
      gl.colorMask(false, false, false, true);
      const p = use(P.hairAlpha), at = p.at;
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, hairTex[c.hair]); gl.uniform1i(at('uSemantic'), 0);
      gl.uniform4f(at('uHair'), c.strength, lo, hi, bias);
      gl.uniform4fv(at('uHairX0'), c.x0); gl.uniform4fv(at('uHairX1'), c.x1);
      gl.uniform4fv(at('uFaceOval'), c.oval); gl.uniform4fv(at('uFaceAxis'), c.axis);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      if (c.mesh) {
        const aPos = gl.getAttribLocation(P.hairAlpha, 'aPosition'), aUv = gl.getAttribLocation(P.hairAlpha, 'aTextureCoord');
        gl.bindBuffer(gl.ARRAY_BUFFER, meshVb); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshIb);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0); gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8); gl.enableVertexAttribArray(aUv);
        gl.enable(gl.CULL_FACE); gl.frontFace(gl.CCW); gl.cullFace(gl.BACK);
        gl.drawElements(gl.TRIANGLES, meshIdx.length, gl.UNSIGNED_SHORT, 0);
        gl.disable(gl.CULL_FACE);
        gl.disableVertexAttribArray(aPos); gl.disableVertexAttribArray(aUv);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      }
      gl.colorMask(true, true, true, true);
      gl.bindTexture(gl.TEXTURE_2D, null);
      checkError('hairAlpha ' + c.name);
      result.cases[c.name] = read(0, 0, N, N);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(frame); gl.deleteTexture(target);
    }
    // 之后再画一道普通程序：RGB 必须写得进去（colorMask 已恢复）。
    {
      const solidTex = texture(4, 4, new Uint8Array(4 * 4 * 4).map((_, i) => [90, 160, 210, 255][i % 4]));
      const target = texture(4, 4, null), frame = framebuffer(target);
      gl.viewport(0, 0, 4, 4);
      const p = use(P.original);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, solidTex); gl.uniform1i(p.at('uSampler'), 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      gl.bindTexture(gl.TEXTURE_2D, null);
      const px = new Uint8Array(4);
      gl.readPixels(2, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      result.after = [px[0], px[1], px[2]];
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    checkError('hairAlpha');
    return result;
  }

  // ---- 头发遮罩 BH1：生产 BG_HAIR 合并道（遮罩分辨率、独立 RGBA8 FBO，同 BackgroundBlurPass.mergeHair），回读 R 平面
  // （第 0 行是画面顶部）；再按 BackgroundBlurPass.render 把合并后的遮罩（或原人像遮罩）接上生产背景虚化，只回传飞发芯的 RGB。
  function bgHair(input) {
    const [MW, MH] = input.mask, [FW, FH] = input.frame, [hairLow, hairHigh] = input.hairUniforms;
    const fromB64 = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    const merged = texture(MW, MH, null), mergedF = framebuffer(merged);
    const identity = [[1, 0, 0, 0], [0, 1, 0, 0]];
    function merge(person, hair, shown, x, oval, axis) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, mergedF); gl.viewport(0, 0, MW, MH);
      const p = use(P.bgHair), at = p.at;
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, hair); gl.uniform1i(at('uSemantic'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, person); gl.uniform1i(at('uPerson'), 0);
      gl.uniform4f(at('uHair'), shown, hairLow, hairHigh, 0);
      gl.uniform4fv(at('uHairX0'), x[0]); gl.uniform4fv(at('uHairX1'), x[1]);
      gl.uniform4fv(at('uFaceOval'), oval); gl.uniform4fv(at('uFaceAxis'), axis);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const unit of [gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
      const bytes = new Uint8Array(MW * MH * 4);
      gl.readPixels(0, 0, MW, MH, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return bytesToB64(Uint8Array.from({length: MW * MH}, (_, k) => bytes[k * 4]));
    }
    const plane = (fn) => {
      const data = new Uint8Array(MW * MH);
      for (let r = 0; r < MH; r++) for (let x = 0; x < MW; x++) data[r * MW + x] = fn(x, r);
      return luminance(MW, MH, data);
    };
    const all = plane(() => 255), none = plane(() => 0), topHalf = plane((x, r) => (r < MH / 2 ? 255 : 0));
    const big = [0.5, 0.5, 10, 10], axis = [1, 0, MW / MH, 0];
    const result = {planes: {}, picks: {}};
    // (a) shown = 0 直通；(b) 方向；(c) 仿射 y 平移 −0.25；(d) 头部先验（小椭圆在画面上 1/4 处，处处是头发）。
    result.planes.pass = merge(plane((x, r) => (x + 3 * r) % 256), all, 0, identity, big, axis);
    result.planes.top = merge(none, topHalf, 1, identity, big, axis);
    result.planes.shift = merge(none, topHalf, 1, [[1, 0, 0, 0], [0, 1, -0.25, 0]], big, axis);
    result.planes.prior = merge(none, all, 1, identity, input.priorOval, axis);
    // (e) 场景：白墙前的低对比浅金发飞发。画面按像素中心栅格化（第 0 行在画面顶部，上传时倒序）。
    const person = luminance(MW, MH, fromB64(input.person)), hairMask = luminance(MW, MH, fromB64(input.hair));
    result.planes.scene = merge(person, hairMask, 1, identity, input.oval, input.axis);
    const inside = (e, x, y) => ((x - e[0]) / e[2]) ** 2 + ((y - e[1]) / e[3]) ** 2 <= 1;
    const [columns, strandW, y0, y1] = input.strands;
    function picture(withStrands) {
      const rgba = new Uint8Array(FW * FH * 4), picked = new Uint8Array(input.pick.length * 3);
      const color = (x, y) => {
        const px = x + 0.5, py = y + 0.5;
        if (withStrands && y >= y0 && y < y1 && columns.some((c) => x >= c && x < c + strandW)) return input.blonde;
        if (inside(input.face, px, py)) return input.skin;
        if (inside(input.head, px, py)) return input.dark;
        if (py >= input.body[0] && Math.abs(px - input.head[0]) <= input.body[1]) return input.shirt;
        return input.wall;
      };
      for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) rgba.set([...color(x, y), 255], ((FH - 1 - y) * FW + x) * 4);
      input.pick.forEach((k, n) => picked.set(color(k % FW, Math.floor(k / FW)), n * 3));
      return {texture: texture(FW, FH, rgba), picked: bytesToB64(picked)};
    }
    // 与 BackgroundBlurPass.render 相同：1/4 预乘 → 横竖高斯两轮（步长 1.5、3）→ 全分辨率混合。
    const sw = Math.floor(FW / 4), sh = Math.floor(FH / 4);
    const small = [0, 1].map(() => texture(sw, sh, null)), smallF = small.map(framebuffer);
    const mixedF = framebuffer(texture(FW, FH, null));
    function blurBackground(source, mask) {
      const [strength, low, high, detail] = input.uniforms;
      gl.bindFramebuffer(gl.FRAMEBUFFER, smallF[0]); gl.viewport(0, 0, sw, sh);
      let p = use(P.bgPremask);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mask); gl.uniform1i(p.at('uMask'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(p.at('uSampler'), 0);
      gl.uniform4f(p.at('uBackground'), strength, low, high, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const spread of [1.5, 3.0]) {
        for (const [from, to, step] of [[0, 1, [spread / sw, 0]], [1, 0, [0, spread / sh]]]) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, smallF[to]);
          p = use(P.bgBlur);
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, small[from]); gl.uniform1i(p.at('uSampler'), 0);
          gl.uniform2f(p.at('uStep'), step[0], step[1]);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          p.done();
        }
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, mixedF); gl.viewport(0, 0, FW, FH);
      p = use(P.bgMix);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, mask); gl.uniform1i(p.at('uMask'), 2);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, small[0]); gl.uniform1i(p.at('uBlurred'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source); gl.uniform1i(p.at('uSampler'), 0);
      gl.uniform4f(p.at('uBackground'), strength, low, high, detail);
      gl.uniform4fv(p.at('uFaceOval'), input.oval); gl.uniform4fv(p.at('uFaceAxis'), input.axis);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const unit of [gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D, null); }
      const bytes = new Uint8Array(FW * FH * 4);
      gl.readPixels(0, 0, FW, FH, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const picked = new Uint8Array(input.pick.length * 3);
      input.pick.forEach((k, n) => {
        const i = ((FH - 1 - Math.floor(k / FW)) * FW + k % FW) * 4;
        picked.set(bytes.subarray(i, i + 3), n * 3);
      });
      return bytesToB64(picked);
    }
    const withStrands = picture(true), plain = picture(false);
    result.picks.inS = withStrands.picked; result.picks.inN = plain.picked;
    // 不合并（第一批，hair == null）：原人像遮罩；合并后：merged 里仍是上面 (e) 的 BG_HAIR 输出（shown = 1）。
    result.picks.beforeS = blurBackground(withStrands.texture, person);
    result.picks.beforeN = blurBackground(plain.texture, person);
    result.picks.afterS = blurBackground(withStrands.texture, merged);
    result.picks.afterN = blurBackground(plain.texture, merged);
    checkError('bgHair');
    return result;
  }

  // ---- 胡须与皮肤边界防粘皮 + 室内冷光/暗影肤色防斑驳探针（B1，对应 COMPOSITE_PRO bleedGuard 与 polyGate）
  function runBeardBorderProbes() {
    const SW = 192, SH = 256;
    const compT = texture(SW, SH, null), compF = framebuffer(compT);
    const bGw = Math.ceil(SW / 3), bGh = Math.ceil(SH / 3);
    const bSmallT = [0, 1, 2].map(() => texture(bGw, bGh, null));
    const bSmallF = bSmallT.map(framebuffer);
    const bBox = { gw: bGw, gh: bGh, gT: bSmallT, gF: bSmallF };
    const faceOneMask = texture(1, 1, new Uint8Array([255, 0, 77, 255]));
    const emptyLut = texture(8, 8, new Uint8Array(8 * 8 * 4));

    function readFbo(fbo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      const buf = new Uint8Array(SW * SH * 4);
      gl.readPixels(0, 0, SW, SH, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return buf;
    }

    const lumaAt = (buf, x, y) => {
      const i = (y * SW + x) * 4;
      return 0.2126 * buf[i] + 0.7152 * buf[i + 1] + 0.0722 * buf[i + 2];
    };

    function renderNaturalFull(img, smoothing) {
      const srcT = texture(SW, SH, img);
      const blurT = guidedPrepass(srcT, smoothing, SW, SH, bBox);
      gl.bindFramebuffer(gl.FRAMEBUFFER, compF); gl.viewport(0, 0, SW, SH);
      const p = use(P.compositePro), at = p.at;
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, emptyLut); gl.uniform1i(at('uLut'), 3);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, faceOneMask); gl.uniform1i(at('uSkinMask'), 2);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, blurT); gl.uniform1i(at('uBlur'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcT); gl.uniform1i(at('uSampler'), 0);
      gl.uniform4f(at('uFrame'), 1 / SW, 1 / SH, SW / SH, 5.0);
      gl.uniform4f(at('uParams'), 16.0, smoothing, 0.0, 0.5);
      gl.uniform4f(at('uDetails'), 0.20, 0.30, 1.0, 0.0);
      gl.uniform4f(at('uTone'), 0.30, 0.15, 0.0, 0.0);
      gl.uniform4f(at('uGrade'), 0.0, 0.005, 0.0, 0.0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      for (const u of [gl.TEXTURE3, gl.TEXTURE2, gl.TEXTURE1, gl.TEXTURE0]) { gl.activeTexture(u); gl.bindTexture(gl.TEXTURE_2D, null); }
      const outBytes = readFbo(compF);
      gl.deleteTexture(srcT);
      return outBytes;
    }

    function runBeardPatch(beardRgb) {
      const skinRgb = [200, 155, 132];
      const img = new Uint8Array(SW * SH * 4);
      for (let y = 0; y < SH; y++) {
        for (let x = 0; x < SW; x++) {
          const inBeard = x >= 48 && x < 144 && y >= 72 && y < 184;
          const strandPhase = ((Math.floor(x / 2) * 5 + Math.floor(y / 4) * 3) % 3) - 1;
          const ripple = inBeard ? strandPhase * 11 : 0;
          const base = inBeard ? beardRgb : skinRgb;
          const k = (y * SW + x) * 4;
          img[k] = Math.max(0, Math.min(255, base[0] + ripple));
          img[k + 1] = Math.max(0, Math.min(255, base[1] + ripple));
          img[k + 2] = Math.max(0, Math.min(255, base[2] + ripple));
          img[k + 3] = 255;
        }
      }
      const outBytes = renderNaturalFull(img, 1.0);
      // 统计胡须与亮肤色交界内侧 4px 带（x∈[49,53), y∈[104,152)）的有符号均值亮度抬升 ΔY（拦截导向均值跨界粘皮）。
      let borderLift = 0, borderN = 0;
      for (let y = 104; y < 152; y++) {
        for (let x = 49; x < 53; x++) {
          borderLift += lumaAt(outBytes, x, y) - lumaAt(img, x, y);
          borderN++;
        }
      }
      return { borderLift: borderLift / borderN };
    }

    // 室内暗影噪点皮肤（[72,54,46] ± 6/255 传感器噪点）与手机屏幕冷白光脸颊（[212,202,200] ± 6/255）必须正常平滑且无斑驳断层。
    function runNoisySkinPatch(baseRgb) {
      const img = new Uint8Array(SW * SH * 4);
      for (let y = 0; y < SH; y++) {
        for (let x = 0; x < SW; x++) {
          const n = (((x * 17 + y * 31) % 5) - 2) * 3;
          const k = (y * SW + x) * 4;
          img[k] = baseRgb[0] + n; img[k + 1] = baseRgb[1] + n; img[k + 2] = baseRgb[2] + n; img[k + 3] = 255;
        }
      }
      const outBytes = renderNaturalFull(img, 0.75);
      const valsIn = [], valsOut = [];
      for (let y = 96; y < 160; y++) {
        for (let x = 64; x < 128; x++) {
          valsIn.push(lumaAt(img, x, y));
          valsOut.push(lumaAt(outBytes, x, y));
        }
      }
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length);
      return std(valsOut, mean(valsOut)) / Math.max(std(valsIn, mean(valsIn)), 1e-6);
    }

    const beardBlack = runBeardPatch([42, 36, 34]);
    const beardBrown = runBeardPatch([66, 48, 38]);
    const shadowSmoothRatio = runNoisySkinPatch([72, 54, 46]);
    const coolLedSmoothRatio = runNoisySkinPatch([212, 202, 200]);

    for (const f of bSmallF) gl.deleteFramebuffer(f);
    for (const t of bSmallT) gl.deleteTexture(t);
    gl.deleteFramebuffer(compF); gl.deleteTexture(compT);
    gl.deleteTexture(faceOneMask); gl.deleteTexture(emptyLut);
    checkError('beardBorder');
    return {
      blackBorderLift: beardBlack.borderLift,
      brownBorderLift: beardBrown.borderLift,
      shadowSmoothRatio,
      coolLedSmoothRatio,
    };
  }

  // ---- v1.1 双频奶油肌匀肤（G9）与 468 点解剖子网格精修（R1–R4）像素探针
  function runV11UpgradeProbes() {
    const luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // 1. G9: GUIDED_COEFF_FRAGMENT 双频奶油肌匀光（uEvenness=0 vs 0.42）
    function evalGuidedCoeff(fineRgb, macroRgb, energyA, evenness) {
      const tFine = texture(4, 4, (() => {
        const b = new Uint8Array(4 * 4 * 4);
        for (let i = 0; i < 16; i++) { b[i * 4] = fineRgb[0]; b[i * 4 + 1] = fineRgb[1]; b[i * 4 + 2] = fineRgb[2]; b[i * 4 + 3] = energyA; }
        return b;
      })());
      const tMacro = texture(4, 4, (() => {
        const b = new Uint8Array(4 * 4 * 4);
        for (let i = 0; i < 16; i++) { b[i * 4] = macroRgb[0]; b[i * 4 + 1] = macroRgb[1]; b[i * 4 + 2] = macroRgb[2]; b[i * 4 + 3] = 255; }
        return b;
      })());
      const dstT = texture(4, 4, null), dstF = framebuffer(dstT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstF); gl.viewport(0, 0, 4, 4);
      const p = use(P.guidedCoeff), at = p.at;
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, tMacro);
      if (at('uMacro') !== null) gl.uniform1i(at('uMacro'), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tFine);
      gl.uniform1i(at('uSampler'), 0);
      gl.uniform2f(at('uDir'), 0.0, 0.25);
      gl.uniform1f(at('uGuidedEps'), 0.0443);
      if (at('uEvenness') !== null) gl.uniform1f(at('uEvenness'), evenness);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      p.done();
      const px = new Uint8Array(4 * 4 * 4);
      gl.readPixels(0, 0, 4, 4, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
      gl.deleteFramebuffer(dstF); gl.deleteTexture(dstT); gl.deleteTexture(tFine); gl.deleteTexture(tMacro);
      // 重构平坦皮肤 q = a * fine + b * 255
      const a = px[3] / 255.0;
      return {
        raw: [px[0], px[1], px[2], px[3]],
        reconY: luma(a * fineRgb[0] + px[0], a * fineRgb[1] + px[1], a * fineRgb[2] + px[2]),
      };
    }

    const blotchFine = [192, 146, 124], blotchMacro = [206, 158, 135];
    const gZero = evalGuidedCoeff(blotchFine, blotchMacro, 8, 0.0);
    const gZeroNoMacro = evalGuidedCoeff(blotchFine, [0, 0, 0], 8, 0.0);
    let evenZeroDiff = 0;
    for (let c = 0; c < 4; c++) evenZeroDiff = Math.max(evenZeroDiff, Math.abs(gZero.raw[c] - gZeroNoMacro.raw[c]));
    const gEven = evalGuidedCoeff(blotchFine, blotchMacro, 8, 0.42);
    const targetGap = luma(...blotchMacro) - gZero.reconY;
    const blotchCloseRatio = (gEven.reconY - gZero.reconY) / Math.max(targetGap, 1e-4);

    // 48/255 立体骨相大光比：boneKeep 门控保护阴影不被抹平
    const boneFine = [150, 112, 95], boneMacro = [205, 160, 138];
    const bZero = evalGuidedCoeff(boneFine, boneMacro, 8, 0.0);
    const bEven = evalGuidedCoeff(boneFine, boneMacro, 8, 0.42);
    const boneShiftAbs = Math.abs(bEven.reconY - bZero.reconY);

    // 2. R1 & R2: 468 点解剖子网格 RETOUCH_SKIN_FRAGMENT（卧蚕权重 0 + 泪沟遮瑕 + 弧形法令纹）
    const RW = 64, RH = 64;
    function runRetouchMesh(prog, baseImg, verts, indices, setUniforms) {
      const srcT = texture(RW, RH, baseImg);
      const dstT = texture(RW, RH, new Uint8Array(baseImg)); // 原位已有底图（模拟 FaceMeshWarpPass FBO）
      const dstF = framebuffer(dstT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstF); gl.viewport(0, 0, RW, RH);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
      gl.useProgram(prog);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
      const pLoc = gl.getAttribLocation(prog, 'aPosition');
      const sLoc = gl.getAttribLocation(prog, 'aSrcCoord');
      const aLoc = gl.getAttribLocation(prog, 'aRetouchAttr');
      if (pLoc >= 0) { gl.enableVertexAttribArray(pLoc); gl.vertexAttribPointer(pLoc, 2, gl.FLOAT, false, 32, 0); }
      if (sLoc >= 0) { gl.enableVertexAttribArray(sLoc); gl.vertexAttribPointer(sLoc, 2, gl.FLOAT, false, 32, 8); }
      if (aLoc >= 0) { gl.enableVertexAttribArray(aLoc); gl.vertexAttribPointer(aLoc, 4, gl.FLOAT, false, 32, 16); }
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcT);
      gl.uniform1i(gl.getUniformLocation(prog, 'uSampler'), 0);
      setUniforms((name) => gl.getUniformLocation(prog, name));
      gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);
      if (pLoc >= 0) gl.disableVertexAttribArray(pLoc);
      if (sLoc >= 0) gl.disableVertexAttribArray(sLoc);
      if (aLoc >= 0) gl.disableVertexAttribArray(aLoc);
      gl.bindBuffer(gl.ARRAY_BUFFER, null); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      gl.disable(gl.BLEND);
      const outPx = new Uint8Array(RW * RH * 4);
      gl.readPixels(0, 0, RW, RH, gl.RGBA, gl.UNSIGNED_BYTE, outPx);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.deleteBuffer(vbo); gl.deleteBuffer(ibo);
      gl.deleteFramebuffer(dstF); gl.deleteTexture(dstT); gl.deleteTexture(srcT);
      return outPx;
    }

    // R1 合成场景：上半 y∈[36,56) 为卧蚕带（attr.x=0），中间 y∈[20,36) 为暗青泪沟（142,102,94），下半 y∈[0,20) 为干净苹果肌（202,158,136）
    const r1Base = new Uint8Array(RW * RH * 4);
    for (let y = 0; y < RH; y++) {
      const rgb = y >= 36 ? [210, 168, 146] : (y >= 20 ? [142, 102, 94] : [202, 158, 136]);
      for (let x = 0; x < RW; x++) {
        const k = (y * RW + x) * 4;
        r1Base[k] = rgb[0]; r1Base[k + 1] = rgb[1]; r1Base[k + 2] = rgb[2]; r1Base[k + 3] = 255;
      }
    }
    // 仅在泪沟子网格 y∈[18,36) (v∈[0.28125, 0.5625]) 赋 tearWeight=1、向下采样偏移 dy=-0.25；卧蚕区 y>=38 顶点权重严格 0
    const r1Verts = [
      // 卧蚕带矩形 (v=0.60..0.90)：tearWeight = 0 -> 必须 discard 零修改
      -0.8, 0.20, 0.1, 0.60, 0.0, 0.0, 0.0, -0.25,
       0.8, 0.20, 0.9, 0.60, 0.0, 0.0, 0.0, -0.25,
      -0.8, 0.80, 0.1, 0.90, 0.0, 0.0, 0.0, -0.25,
       0.8, 0.80, 0.9, 0.90, 0.0, 0.0, 0.0, -0.25,
      // 泪沟带矩形 (v=0.32..0.55)：tearWeight = 1.0，向下采干净脸颊
      -0.8, -0.36, 0.1, 0.32, 1.0, 0.0, 0.0, -0.22,
       0.8, -0.36, 0.9, 0.32, 1.0, 0.0, 0.0, -0.22,
      -0.8,  0.10, 0.1, 0.55, 1.0, 0.0, 0.0, -0.22,
       0.8,  0.10, 0.9, 0.55, 1.0, 0.0, 0.0, -0.22,
    ];
    const r1Out = runRetouchMesh(P.retouchSkin, r1Base, r1Verts, [0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7], (at) => {
      gl.uniform4f(at('uRetouchStrength'), 1.0, 0.0, 0.0, 0.0);
    });
    let aegyoMaxDiff = 0;
    for (let y = 40; y < 54; y++) {
      for (let x = 16; x < 48; x++) {
        const k = (y * RW + x) * 4;
        for (let c = 0; c < 3; c++) aegyoMaxDiff = Math.max(aegyoMaxDiff, Math.abs(r1Out[k + c] - r1Base[k + c]));
      }
    }
    const tearIdx = (26 * RW + 32) * 4;
    const tearLift = luma(r1Out[tearIdx], r1Out[tearIdx + 1], r1Out[tearIdx + 2]) - luma(r1Base[tearIdx], r1Base[tearIdx + 1], r1Base[tearIdx + 2]);

    // R1b 暗肤色/暗光且无黑眼圈（泪沟与苹果肌同色 [68,48,38]）：禁止误涂亮橘色遮瑕条（"反向熊猫眼"）
    const r1EvenDark = new Uint8Array(RW * RH * 4);
    for (let i = 0; i < RW * RH; i++) {
      r1EvenDark[i * 4] = 68; r1EvenDark[i * 4 + 1] = 48; r1EvenDark[i * 4 + 2] = 38; r1EvenDark[i * 4 + 3] = 255;
    }
    const r1EvenOut = runRetouchMesh(P.retouchSkin, r1EvenDark, r1Verts.slice(32), [0, 1, 2, 2, 1, 3], (at) => {
      gl.uniform4f(at('uRetouchStrength'), 1.0, 0.0, 0.0, 0.0);
    });
    let evenDarkMaxDiff = 0;
    for (let c = 0; c < 3; c++) evenDarkMaxDiff = Math.max(evenDarkMaxDiff, Math.abs(r1EvenOut[tearIdx + c] - r1EvenDark[tearIdx + c]));

    // R2 弧形法令纹：中心暗沟 x∈[29,35) 为 (148,110,92)，两侧皮肤为 (204,160,138)
    const r2Base = new Uint8Array(RW * RH * 4);
    for (let y = 0; y < RH; y++) {
      for (let x = 0; x < RW; x++) {
        const rgb = (x >= 29 && x < 35) ? [148, 110, 92] : (x >= 46 ? [222, 180, 158] : [204, 160, 138]);
        const k = (y * RW + x) * 4;
        r2Base[k] = rgb[0]; r2Base[k + 1] = rgb[1]; r2Base[k + 2] = rgb[2]; r2Base[k + 3] = 255;
      }
    }
    const r2Verts = [
      -0.6, -0.6, 0.2, 0.2, 0.0, 1.0, 0.16, 0.0,
       0.8, -0.6, 0.9, 0.2, 0.0, 1.0, 0.16, 0.0,
      -0.6,  0.6, 0.2, 0.8, 0.0, 1.0, 0.16, 0.0,
       0.8,  0.6, 0.9, 0.8, 0.0, 1.0, 0.16, 0.0,
    ];
    const r2Out = runRetouchMesh(P.retouchSkin, r2Base, r2Verts, [0, 1, 2, 2, 1, 3], (at) => {
      gl.uniform4f(at('uRetouchStrength'), 0.0, 1.0, 0.0, 0.0);
    });
    const foldIdx = (32 * RW + 32) * 4;
    const foldLift = luma(r2Out[foldIdx], r2Out[foldIdx + 1], r2Out[foldIdx + 2]) - luma(r2Base[foldIdx], r2Base[foldIdx + 1], r2Base[foldIdx + 2]);
    const brightIdx = (32 * RW + 52) * 4;
    let foldBrightMaxDiff = 0;
    for (let c = 0; c < 3; c++) foldBrightMaxDiff = Math.max(foldBrightMaxDiff, Math.abs(r2Out[brightIdx + c] - r2Base[brightIdx + c]));

    // R3 内唇口腔子网格白牙：外围水光浅粉唇釉 (232,198,204) 低饱和高亮，中心内唇子网格内为偏黄牙色 (202,188,144)
    const r3Base = new Uint8Array(RW * RH * 4);
    for (let y = 0; y < RH; y++) {
      for (let x = 0; x < RW; x++) {
        const inCavity = x >= 20 && x < 44 && y >= 22 && y < 42;
        const rgb = inCavity ? [202, 188, 144] : [232, 198, 204];
        const k = (y * RW + x) * 4;
        r3Base[k] = rgb[0]; r3Base[k + 1] = rgb[1]; r3Base[k + 2] = rgb[2]; r3Base[k + 3] = 255;
      }
    }
    // 严格只在内唇孔洞 [20..44, 22..42] (NDC [-0.375..0.375, -0.3125..0.3125]) 光栅化
    const r3Verts = [
      -0.375, -0.3125, 0.3125, 0.34375, 1.0, 0.0, 0.0, 0.0,
       0.375, -0.3125, 0.6875, 0.34375, 1.0, 0.0, 0.0, 0.0,
      -0.375,  0.3125, 0.3125, 0.65625, 1.0, 0.0, 0.0, 0.0,
       0.375,  0.3125, 0.6875, 0.65625, 1.0, 0.0, 0.0, 0.0,
    ];
    const r3Out = runRetouchMesh(P.retouchEyeTeeth, r3Base, r3Verts, [0, 1, 2, 2, 1, 3], (at) => {
      gl.uniform4f(at('uRetouchMode'), 1.0, 1.0, 0.0, 0.0);
    });
    let lipGlossMaxDiff = 0;
    for (const [lx, ly] of [[12, 32], [52, 32], [32, 14], [32, 50]]) {
      const k = (ly * RW + lx) * 4;
      for (let c = 0; c < 3; c++) lipGlossMaxDiff = Math.max(lipGlossMaxDiff, Math.abs(r3Out[k + c] - r3Base[k + c]));
    }
    const tIdx = (32 * RW + 32) * 4;
    const teethLumaLift = luma(r3Out[tIdx], r3Out[tIdx + 1], r3Out[tIdx + 2]) - luma(r3Base[tIdx], r3Base[tIdx + 1], r3Base[tIdx + 2]);
    const teethYellowDrop = (r3Base[tIdx] - r3Base[tIdx + 2]) - (r3Out[tIdx] - r3Out[tIdx + 2]);

    // R4 眼裂子网格亮眼：中心眼裂内巩膜 (196,192,188)，外部眼睑皮肤 (206,162,140)
    const r4Base = new Uint8Array(RW * RH * 4);
    for (let y = 0; y < RH; y++) {
      for (let x = 0; x < RW; x++) {
        const inEye = x >= 20 && x < 44 && y >= 24 && y < 40;
        const rgb = inEye ? [196, 192, 188] : [206, 162, 140];
        const k = (y * RW + x) * 4;
        r4Base[k] = rgb[0]; r4Base[k + 1] = rgb[1]; r4Base[k + 2] = rgb[2]; r4Base[k + 3] = 255;
      }
    }
    const r4Out = runRetouchMesh(P.retouchEyeTeeth, r4Base, r3Verts, [0, 1, 2, 2, 1, 3], (at) => {
      gl.uniform4f(at('uRetouchMode'), 1.0, 0.0, 0.0, 0.0);
    });
    let eyeOutsideMaxDiff = 0;
    for (const [ex, ey] of [[10, 32], [54, 32], [32, 10], [32, 54]]) {
      const k = (ey * RW + ex) * 4;
      for (let c = 0; c < 3; c++) eyeOutsideMaxDiff = Math.max(eyeOutsideMaxDiff, Math.abs(r4Out[k + c] - r4Base[k + c]));
    }
    const sIdx = (32 * RW + 32) * 4;
    const scleraLift = luma(r4Out[sIdx], r4Out[sIdx + 1], r4Out[sIdx + 2]) - luma(r4Base[sIdx], r4Base[sIdx + 1], r4Base[sIdx + 2]);

    checkError('v11Upgrade');
    return {
      evenZeroDiff,
      blotchCloseRatio,
      boneShiftAbs,
      aegyoMaxDiff,
      tearLift,
      evenDarkMaxDiff,
      foldLift,
      foldBrightMaxDiff,
      lipGlossMaxDiff,
      teethLumaLift,
      teethYellowDrop,
      eyeOutsideMaxDiff,
      scleraLift,
    };
  }
}
