#!/usr/bin/env bash
# 在无头 Chrome WebGL1（ANGLE/SwiftShader）上验证 Android 41 个美颜 GLSL 原文、接线变异门禁与黄金向量，
# 并校验 10 套内置 64^3 3D LUT PNG 资产。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KOTLIN_DIR="$ROOT_DIR/android/live-beauty/src/main/kotlin/io/bebebus/livebeauty"
SHADERS_KT="${1:-$KOTLIN_DIR/VideoEffectsShaders.kt}"
WIRING_DIR="${2:-$KOTLIN_DIR}"
VECTORS="$ROOT_DIR/reference/beauty_vectors.json"

if [[ "$(uname -s)" == Darwin && -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  GLSL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/livebeauty-glsl.XXXXXX")"
  trap 'rm -rf "$GLSL_TMP"' EXIT
  python3 -I -B "$ROOT_DIR/_tooling/glsl/glsl_gate.py" "$SHADERS_KT" "$VECTORS" "$GLSL_TMP" "$WIRING_DIR"
  python3 -I -B "$ROOT_DIR/_tooling/verify-cpp-gles-shaders.py" "$ROOT_DIR/core/src/rhi/gles/gles_shaders.cpp"
else
  echo "SKIP  未检测到 macOS Google Chrome，跳过无头 WebGL1 光栅探针；继续校验 3D LUT 资产。"
fi
python3 -I -B "$ROOT_DIR/reference/gen_builtin_luts.py" --verify
