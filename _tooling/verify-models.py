#!/usr/bin/env python3
"""校验随 SDK 内置分发的 4 个 MediaPipe Apache-2.0 离线模型 SHA256 指纹。"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    (
        "Face Landmarker float16/1 (Android)",
        ROOT / "android/live-beauty/src/main/assets/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    (
        "Face Landmarker float16/1 (iOS)",
        ROOT / "ios/Resources/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    (
        "Pose Landmarker lite float16/1",
        ROOT / "android/live-beauty/src/main/assets/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    (
        "Selfie Segmenter float16/1",
        ROOT / "android/live-beauty/src/main/assets/selfie_segmenter.tflite",
        "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
    ),
    (
        "Hair Segmenter float32/1",
        ROOT / "android/live-beauty/src/main/assets/hair_segmenter.tflite",
        "2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84",
    ),
]

for label, path, expected in MODELS:
    if not path.is_file():
        raise SystemExit(f"[FAIL] 缺失模型文件：{path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"[FAIL] {label} SHA256 不匹配：expected={expected}, actual={actual}")
    print(f"PASS  {label} SHA256 校验通过")
