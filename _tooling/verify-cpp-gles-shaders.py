#!/usr/bin/env python3
"""Verify all C++ GLES shaders in core/src/rhi/gles/gles_shaders.cpp compile and link
under strict GLSL ES 1.00 (both highp and forced mediump fragment precision) via headless Chrome WebGL1.
"""
import html as hmod
import json
import os
import re
import subprocess
import sys
import tempfile

CHROME = os.environ.get(
    "CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)

PREC = (
    "#ifdef GL_ES\n"
    "#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
    "precision highp float;\n"
    "#else\n"
    "precision mediump float;\n"
    "#endif\n"
    "#endif\n"
)
BLUR_COMMON = PREC + (
    "uniform sampler2D uTex0;\n"
    "varying vec2 vC; varying vec2 vA1; varying vec2 vB1; varying vec2 vA2; varying vec2 vB2;\n"
    "vec4 blur5() {\n"
    "  return texture2D(uTex0, vC) * 0.2042 + "
    "(texture2D(uTex0, vA1) + texture2D(uTex0, vB1)) * 0.3040 + "
    "(texture2D(uTex0, vA2) + texture2D(uTex0, vB2)) * 0.0939;\n"
    "}\n"
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify-cpp-gles-shaders.py <gles_shaders.cpp>", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as fh:
        src = fh.read()

    shaders = {}
    for m in re.finditer(
        r'(?:constexpr\s+)?const\s+char\*\s+(\w+)\s*=\s*(LB_GLSL_PREC\s+|LB_GLSL_BLUR_COMMON\s+)?R"\((.*?)\)"\s*;',
        src,
        re.S,
    ):
        name, prefix, body = m.group(1), m.group(2) or "", m.group(3)
        p = PREC if "LB_GLSL_PREC" in prefix else (BLUR_COMMON if "LB_GLSL_BLUR_COMMON" in prefix else "")
        shaders[name] = p + body

    pairs = [
        (m.group(1), m.group(2), m.group(3))
        for m in re.finditer(r"case\s+ShaderId::(\w+):\s*return\s*\{(\w+),\s*(\w+)\};", src)
    ]
    if not pairs:
        print("FAIL  no ShaderId pairs found in gles_shaders.cpp", file=sys.stderr)
        return 1

    missing = [(n, v, f) for n, v, f in pairs if v not in shaders or f not in shaders]
    if missing:
        print(f"FAIL  unresolved shader symbols: {missing}", file=sys.stderr)
        return 1

    if not os.path.exists(CHROME):
        print(f"SKIP  Chrome not found at {CHROME}")
        return 0

    doc = f"""<!doctype html><body><pre id="res"></pre><script>
const shaders = {json.dumps(shaders)};
const pairs = {json.dumps(pairs)};
const c = document.createElement('canvas');
const gl = c.getContext('webgl');
const out = [];
function runMode(forceMediump) {{
  for (const [name, vsName, fsName] of pairs) {{
    let vs = shaders[vsName];
    let fs = shaders[fsName];
    if (forceMediump) {{
      fs = fs.replace(
        '#ifdef GL_FRAGMENT_PRECISION_HIGH\\nprecision highp float;\\n#else\\nprecision mediump float;\\n#endif',
        'precision mediump float;'
      );
    }}
    const s1 = gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(s1, vs); gl.compileShader(s1);
    const s2 = gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(s2, fs); gl.compileShader(s2);
    const p = gl.createProgram(); gl.attachShader(p, s1); gl.attachShader(p, s2); gl.linkProgram(p);
    const ok1 = gl.getShaderParameter(s1, gl.COMPILE_STATUS);
    const ok2 = gl.getShaderParameter(s2, gl.COMPILE_STATUS);
    const okp = gl.getProgramParameter(p, gl.LINK_STATUS);
    const nUni = okp ? gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) : -1;
    out.push({{
      mode: forceMediump ? 'mediump' : 'highp',
      name, ok: !!(ok1 && ok2 && okp), activeUniforms: nUni,
      vsLog: gl.getShaderInfoLog(s1), fsLog: gl.getShaderInfoLog(s2), pLog: gl.getProgramInfoLog(p)
    }});
  }}
}}
runMode(false);
runMode(true);
document.getElementById('res').textContent = JSON.stringify(out);
</script></body>"""

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tf:
        tf.write(doc)
        html_path = tf.name

    try:
        res = subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--disable-gpu-sandbox",
                "--dump-dom",
                "file://" + html_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        os.unlink(html_path)

    m = re.search(r'<pre id="res">(.*?)</pre>', res.stdout, re.S)
    if not m:
        print("FAIL  headless WebGL1 probe produced no output", file=sys.stderr)
        return 1

    results = json.loads(hmod.unescape(m.group(1)))
    fails = [r for r in results if not r["ok"]]
    if fails:
        for f in fails:
            print(f"FAIL  [{f['mode']}] {f['name']}: vs={f['vsLog']} fs={f['fsLog']} link={f['pLog']}", file=sys.stderr)
        return 1

    comp_high = [r for r in results if r["mode"] == "highp" and r["name"] == "kCompositePro"]
    if not comp_high or comp_high[0]["activeUniforms"] > 15:
        print(f"FAIL  kCompositePro uniform budget exceeded: {comp_high}", file=sys.stderr)
        return 1

    print(f"PASS  all {len(pairs)} C++ GLES shaders linked in both highp and mediump ({len(results)}/{len(pairs) * 2}, kCompositePro={comp_high[0]['activeUniforms']} uniforms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
