#!/usr/bin/env python3
"""开源发布脱敏门禁：扫描全仓文本文件，严禁残留原业务仓库名、内部域名、本机路径或推流耦合包名。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(
    r"(cartoon|video\.dh\.app|1Bit-Software|rtmp_publisher|rtmp_broadcaster|com\.pedro|/Users/fitz|/Users/chen)",
    re.IGNORECASE,
)
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".task", ".tflite", ".onnx", ".jar", ".keystore", ".so", ".a"}
SKIP_DIRS = {".git", ".gradle", "build", ".dart_tool", "Pods", "DerivedData", ".cxx", ".externalNativeBuild"}

violations = []
checked = 0
for p in sorted(ROOT.rglob("*")):
    if any(part in SKIP_DIRS for part in p.parts):
        continue
    if not p.is_file() or p.suffix.lower() in BINARY_EXTS:
        continue
    if p.resolve() == Path(__file__).resolve():
        continue
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    checked += 1
    for lineno, line in enumerate(text.splitlines(), 1):
        m = FORBIDDEN.search(line)
        if m:
            violations.append(f"{p.relative_to(ROOT)}:{lineno}: forbidden token '{m.group(0)}' -> {line.strip()[:120]}")

if violations:
    print("FAIL  脱敏门禁拦截到内部或耦合标识残留：", file=sys.stderr)
    for v in violations[:40]:
        print("  " + v, file=sys.stderr)
    raise SystemExit(1)

print(f"PASS  开源脱敏门禁通过（已扫描 {checked} 个文本文件，0 处敏感/耦合残留）")
