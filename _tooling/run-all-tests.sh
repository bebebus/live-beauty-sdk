#!/usr/bin/env bash
# Bebebus Live Beauty SDK — 一键全量回归门禁
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=================================================================="
echo " [1/7] Open-Source Sanitization Gate (_tooling/sanitize-check.py)"
echo "=================================================================="
python3 _tooling/sanitize-check.py

echo "=================================================================="
echo " [2/7] Bundled MediaPipe Models SHA256 Gate (_tooling/verify-models.py)"
echo "=================================================================="
python3 _tooling/verify-models.py

echo "=================================================================="
echo " [3/7] Canonical 468/548 Face Topology Gate (reference/gen_face_topology.py)"
echo "=================================================================="
python3 reference/gen_face_topology.py --verify

echo "=================================================================="
echo " [4/7] 44 GLSL Shaders + 28 Mutants + 10 3D LUTs Gate (_tooling/verify-glsl.sh)"
echo "=================================================================="
if [[ "${SKIP_GLSL_BROWSER_GATE:-0}" == "1" ]]; then
  echo "SKIP  SKIP_GLSL_BROWSER_GATE=1，仅执行内置 3D LUT 像素校验"
  python3 reference/gen_builtin_luts.py --verify
else
  bash _tooling/verify-glsl.sh
fi

echo "=================================================================="
echo " [5/7] iOS/macOS CoreImage Kernel & Smoother Gate (ios/Tests/verify_macos_ci.m)"
echo "=================================================================="
if [[ "$(uname -s)" == "Darwin" ]]; then
  CI_BIN="$(mktemp -t livebeauty_ci).bin"
  trap 'rm -f "$CI_BIN"' EXIT
  clang -fobjc-arc \
    -framework Foundation \
    -framework CoreImage \
    -framework CoreGraphics \
    -framework CoreMedia \
    -framework CoreVideo \
    -framework QuartzCore \
    ios/Classes/FLTFaceGeometry.m \
    ios/Classes/FLTFaceGeometrySmoother.m \
    ios/Classes/FLTFaceLandmarkTracker.m \
    ios/Classes/FLTVideoEffects.m \
    ios/Tests/verify_macos_ci.m \
    -o "$CI_BIN"
  "$CI_BIN"
  rm -f "$CI_BIN"
  trap - EXIT
else
  echo "SKIP  非 macOS 宿主，跳过 CoreImage 原生编译门禁"
fi

echo "=================================================================="
echo " [6/7] Android Standalone SDK Unit Tests & Release AAR (:live-beauty)"
echo "=================================================================="
if command -v java >/dev/null 2>&1 && [[ -n "${ANDROID_HOME:-}${ANDROID_SDK_ROOT:-}" ]]; then
  (cd android && ./gradlew --no-daemon :live-beauty:testDebugUnitTest :live-beauty:testReleaseUnitTest :live-beauty:assembleRelease)
elif [[ -n "${LIVE_BEAUTY_FLUTTER_IMAGE:-}" ]]; then
  docker run --rm --platform linux/amd64 \
    -e GRADLE_USER_HOME=/tmp/gradle-cache \
    ${LIVE_BEAUTY_GRADLE_CACHE:+-v "$LIVE_BEAUTY_GRADLE_CACHE:/tmp/gradle-cache"} \
    -v "$ROOT:/work" \
    -w /work/android \
    "$LIVE_BEAUTY_FLUTTER_IMAGE" \
    ./gradlew --no-daemon :live-beauty:testDebugUnitTest :live-beauty:testReleaseUnitTest :live-beauty:assembleRelease
else
  echo "ERROR 未检测到 ANDROID_HOME/ANDROID_SDK_ROOT，且未指定 LIVE_BEAUTY_FLUTTER_IMAGE 容器镜像" >&2
  exit 1
fi

echo "=================================================================="
echo " [7/7] Flutter Wrapper Static Analysis & Unit Tests (flutter/)"
echo "=================================================================="
if command -v flutter >/dev/null 2>&1; then
  (cd flutter && flutter pub get && flutter analyze --no-pub && flutter test --no-pub)
elif [[ -n "${LIVE_BEAUTY_FLUTTER_IMAGE:-}" ]]; then
  DOCKER_ARCH="linux/amd64"
  case "$(uname -m)" in arm64|aarch64) DOCKER_ARCH="linux/arm64" ;; esac
  docker run --rm --platform "$DOCKER_ARCH" \
    -e PUB_CACHE=/tmp/pub-cache \
    -e HOME=/tmp \
    ${LIVE_BEAUTY_PUB_CACHE:+-v "$LIVE_BEAUTY_PUB_CACHE:/tmp/pub-cache"} \
    -v "$ROOT:/work" \
    -w /work/flutter \
    "$LIVE_BEAUTY_FLUTTER_IMAGE" \
    bash -lc "flutter pub get && flutter analyze --no-pub && flutter test --no-pub"
else
  echo "ERROR 未检测到 flutter SDK，且未指定 LIVE_BEAUTY_FLUTTER_IMAGE 容器镜像" >&2
  exit 1
fi

echo ""
echo "ALL 7 LIVE BEAUTY SDK GATES PASSED!"
