"""由参考模型 beauty_ref.py 生成两端共享的数值向量 beauty_vectors.json。

  /usr/bin/python3 -I gen_vectors.py [输出路径]      # 缺省写到本目录的 beauty_vectors.json

输出是确定的：同一 numpy 版本下重复运行字节一致（随机量只用 Python random.Random 固定种子）。
所有期望值都由写进 JSON 的（已舍入）输入重新计算，消费者直接读输入即可复现。
各段含义与坐标约定见 README.md 与 JSON 的 meta 段。
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beauty_ref as B  # noqa: E402
from beauty_ref import np  # noqa: E402

ASPECT = 720.0 / 1280.0
USED = [132, 361, 172, 397, 33, 133, 362, 263, 1, 234, 454, 10, 152, 61, 291, 0, 17]

# MediaPipe canonical_face_model.obj（mediapipe/modules/face_geometry/data，Apache-2.0，
# 许可文本见 ../licenses/MediaPipe-LICENSE.txt）中本模型用到的 17 个顶点，单位 cm，y 向上。
CANONICAL = {
    0: (0.000000, -3.406404, 5.979507), 1: (0.000000, -1.126865, 7.475604), 10: (0.000000, 8.261778, 4.481535),
    17: (0.000000, -5.365123, 5.535441), 33: (-4.445859, 2.663991, 3.173422), 61: (-2.456206, -4.342621, 4.283884),
    132: (-7.270895, -2.890917, -2.252455), 133: (-1.856432, 2.585245, 3.757904), 152: (0.000000, -9.403378, 4.264492),
    172: (-5.940524, -6.223629, -0.631468), 234: (-7.664182, 0.673132, -2.435867), 263: (4.445859, 2.663991, 3.173422),
    291: (2.456206, -4.342621, 4.283884), 361: (7.270895, -2.890917, -2.252455), 362: (1.856432, 2.585245, 3.757904),
    397: (5.940524, -6.223629, -0.631468), 454: (7.664182, 0.673132, -2.435867),
}


def r6(x):
    """舍入到 6 位小数；-0.0 归一为 0.0。"""
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = float(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r6(v) for v in x]
    return round(float(x), 6) + 0.0


def project(yaw=0.0, pitch=0.0, roll=0.0, face_px=375.0, center=(360.0, 600.0), width=720, height=1280,
            dist_cm=45.0):
    """透视投影标准人脸：返回 {索引: (x, y)} 像素坐标（x 向右、y 向下）；正脸时 234–454 宽 face_px 像素。"""
    ya, pa, ra = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(ya), 0, math.sin(ya)], [0, 1, 0], [-math.sin(ya), 0, math.cos(ya)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pa), -math.sin(pa)], [0, math.sin(pa), math.cos(pa)]])
    rz = np.array([[math.cos(ra), -math.sin(ra), 0], [math.sin(ra), math.cos(ra), 0], [0, 0, 1]])
    rot = rz @ rx @ ry
    f = face_px * dist_cm / (CANONICAL[454][0] - CANONICAL[234][0])
    out = {}
    for i, v in CANONICAL.items():
        p = rot @ np.array(v)
        z = dist_cm - p[2]
        out[i] = (center[0] + f * p[0] / z, center[1] - f * p[1] / z)
    return out


def landmarks(mirror=False, width=720, height=1280, **kw):
    """用到的 17 个点的 MediaPipe 归一化坐标（y 向下），已舍入到 6 位小数。"""
    px = project(width=width, height=height, **kw)
    return {i: [r6((1.0 - x / width) if mirror else x / width), r6(y / height)] for i, (x, y) in px.items()}


def to_points(lm):
    """展开成 478 点数组；未用到的索引填 (0.5, 0.5)，几何不读它们。字符串 'NaN'/'Infinity' 转成非有限浮点。"""
    def num(v):
        return float(v.lower().replace("infinity", "inf")) if isinstance(v, str) else float(v)
    pts = [(0.5, 0.5)] * 478
    for i, (x, y) in lm.items():
        pts[int(i)] = (num(x), num(y))
    return pts


def geo_json(g):
    return None if g is None else {k: r6(v) for k, v in g.items()}


def pixel_uv(x, row, w, h):
    return [(x + 0.5) / w, 1.0 - (row + 0.5) / h]


def rgb8(c):
    return [int(round(float(v) * 255.0)) for v in c]


out = {}

# ---------------------------------------------------------------- constants / kernel
out["constants"] = B.constants()
taps9 = B.gauss_kernel()
assert max(abs(a - b) for a, b in zip(taps9, B.IOS_CONV9)) < 1e-6, "iOS CIConvolution9 权重须与 9 抽头核一致"
pairs = B.linear_pairs()
for (_, w, o), (so, sw) in zip(pairs, B.SHADER_PAIRS):
    assert abs(w - sw) < 1e-4 and abs(o - so) < 5e-4, "Android 线性采样常量须与离散核一致"
out["kernel"] = dict(
    note="sigma 2、±4 抽头截断后归一化（有效 sigma 1.85）。参考模型用 taps9 离散核；Android 用 shaderPairs 的线性采样"
         "（中心 + 两对，偏移乘 uDir），iOS 用 iosConvolution9 作为 CIConvolution9Horizontal/Vertical 的 inputWeights。",
    sigma=B.SIGMA, taps9=r6(taps9),
    linearPairs=[dict(name=n, weight=r6(w), offset=r6(o)) for n, w, o in pairs],
    shaderPairs=[dict(offset=o, weight=w) for o, w in B.SHADER_PAIRS],
    iosConvolution9=list(B.IOS_CONV9))

# ---------------------------------------------------------------- skin_gate
rng = random.Random(11)
gate_rgb = [[194, 150, 130], [115, 82, 68], [246, 237, 228], [41, 36, 32], [58, 49, 42], [232, 239, 249], [217, 199, 168],
            [140, 97, 56], [26, 23, 23], [128, 128, 128], [184, 77, 82], [255, 255, 255], [0, 0, 0], [201, 171, 151],
            [82, 58, 49], [230, 153, 166]]
for _ in range(16):
    gain = rng.uniform(0.3, 1.2)
    gate_rgb.append([min(255, max(0, int(round(c * gain + rng.gauss(0.0, 12.0))))) for c in (192, 156, 133)])
# 四条过渡带各取 3 个点：该因子约为 0.2 / 0.5 / 0.8、其余三个因子为 1，任一门限常量漂移 0.01 都会远超 0.004。
# 上面的肤色样本只落在 cb 上沿与 cr 下沿，cb 下沿（偏黄）与 cr 上沿（偏红）全靠这里覆盖。
gate_grid = np.stack(np.meshgrid(*[np.arange(0, 256, 5)] * 3, indexing="ij"), -1).reshape(-1, 3)
gate_factors = np.stack(B.skin_factors(gate_grid / 255.0))
for band in range(4):
    others = np.prod(np.delete(gate_factors, band, axis=0), axis=0)
    for t in (0.2, 0.5, 0.8):
        score = np.where(others >= 1.0, np.abs(gate_factors[band] - t), np.inf)
        pick = int(np.argmin(score))
        assert score[pick] < 0.05, (band, t, score[pick])
        gate_rgb.append([int(c) for c in gate_grid[pick]])
for band in range(4):
    assert sum(0.1 < float(B.skin_factors(np.array(c) / 255.0)[band]) < 0.9 for c in gate_rgb) >= 3, band
out["skin_gate"] = dict(
    note="I = rgb8 / 255，expected = skinWeight(I)。后 12 例分别落在 cb 下沿、cb 上沿、cr 下沿、cr 上沿四条过渡带中段。",
    cases=[dict(rgb8=c, expected=r6(B.skin_weight(np.array(c) / 255.0))) for c in gate_rgb])

# ---------------------------------------------------------------- geometry
GEO_FRAMES = {"portrait": (720, 1280), "landscape": (1280, 720)}
geo_poses = [
    ("frontal_375", dict()),
    ("yaw15", dict(yaw=15)),
    ("yaw17_partial_warp", dict(yaw=17)),
    ("yaw25_warp_off", dict(yaw=25)),
    ("yaw-25_pitch10_warp_off", dict(yaw=-25, pitch=10)),
    ("yaw15_roll10", dict(yaw=15, roll=10)),
    ("yaw-12_pitch10_roll-20", dict(yaw=-12, pitch=10, roll=-20)),
    ("roll35_yaw5", dict(yaw=5, roll=35)),
    ("mirrored_yaw10_roll5", dict(yaw=10, roll=5, mirror=True)),
    ("mirrored_yaw17_partial_warp", dict(yaw=17, mirror=True)),
    ("small_face_300_offcentre", dict(face_px=300, center=(250, 420), yaw=5)),
    ("big_face_forehead_out_of_frame", dict(face_px=520, center=(360, 270))),
    ("landscape_frontal", dict(frame="landscape", face_px=300, center=(640, 380))),
    ("yaw35_rejected", dict(yaw=35)),
    ("yaw40_rejected", dict(yaw=40)),
    ("cheek_out_of_frame_rejected", dict(center=(120, 600))),
    ("span_too_small_rejected", dict(face_px=60)),
]


def edited(base, **changes):
    lm = {i: list(v) for i, v in base.items()}
    for key, fn in changes.items():
        lm[int(key[1:])] = fn(lm)
    return lm


front = landmarks()
geo_edits = [
    # 遮罩点越过 [-0.25, 1.25]
    ("landmark10_y_below_minus_0.25_rejected", edited(front, p10=lambda lm: [lm[10][0], -0.26]), "mask_point"),
    ("landmark234_x_above_1.25_rejected", edited(front, p234=lambda lm: [1.26, lm[234][1]]), "mask_point"),
    # 非有限（JSON 里以字符串表示）
    ("landmark61_nan_rejected", edited(front, p61=lambda lm: ["NaN", lm[61][1]]), "mask_point"),
    ("landmark1_infinite_rejected", edited(front, p1=lambda lm: [lm[1][0], "Infinity"]), "warp_point"),
    # 退化保护
    ("face_width_below_0.05_rejected", edited(front, p454=lambda lm: [r6(lm[234][0] + 0.04), lm[234][1]]), "face_width"),
    ("oval_rx_below_0.03_rejected", edited(front, p454=lambda lm: [r6(lm[234][0] + 0.052), lm[234][1]]), "oval_radius"),
    ("oval_ry_below_0.03_rejected", edited(front, p10=lambda lm: [lm[152][0], r6(lm[152][1] - 0.04 * ASPECT)]), "oval_radius"),
    ("mouth_rx_below_0.005_rejected", edited(front, p291=lambda lm: [r6(lm[61][0] + 0.005), lm[61][1]]), "mouth_radius"),
    ("eye_width_below_0.015_rejected", edited(front, p133=lambda lm: [r6(lm[33][0] + 0.01), lm[33][1]]), "eye_width"),
]

geometry_cases = []
GEOS = {}
for name, kw in geo_poses:
    kw = dict(kw)
    fw, fh = GEO_FRAMES[kw.pop("frame", "portrait")]
    kw.setdefault("center", (fw / 2.0, fh * 600.0 / 1280.0))
    lm = landmarks(width=fw, height=fh, **kw)
    geometry_cases.append((name, fw / fh, lm, None))
for name, lm, reason in geo_edits:
    geometry_cases.append((name, ASPECT, lm, reason))

out_geo = []
for name, aspect, lm, want in geometry_cases:
    pts = to_points(lm)
    g, reason = B.geometry_with_reason(pts, aspect)
    if want is not None:
        assert g is None and reason == want, (name, reason, want)
    if name.endswith("_rejected"):
        assert g is None, name
    else:
        assert g is not None, (name, reason)
    GEOS[name] = (g, aspect)
    r = B.yaw_ratio(pts, aspect) if reason not in ("warp_point", "input", "span") else None
    out_geo.append(dict(name=name, aspect=r6(aspect), landmarks={str(i): lm[i] for i in USED},
                        yawRatio=r6(r), rejectReason=reason, expected=geo_json(g)))

# 规格第 2 节的门限在标准人脸上的落点
ww = {n: GEOS[n][0]["warpWeight"] for n in ("frontal_375", "yaw15", "yaw17_partial_warp", "yaw25_warp_off")}
assert ww["frontal_375"] == 1.0 and ww["yaw15"] == 1.0 and 0.0 < ww["yaw17_partial_warp"] < 1.0, ww
assert ww["yaw25_warp_off"] == 0.0 and 0.0 < GEOS["mirrored_yaw17_partial_warp"][0]["warpWeight"] < 1.0
assert min(front[10]) > 0 and landmarks(face_px=520, center=(360, 270))[10][1] < 0, "大脸用例的额头应在画面外"
out["geometry"] = dict(
    note="输入是 MediaPipe 归一化图像坐标（y 向下），只列出用到的 17 个索引；消费者建 478 点数组，其余索引填任意有限值。"
         "字符串 'NaN'/'Infinity' 表示非有限值。expected 为 null 表示几何无效（全有或全无）；rejectReason 与 yawRatio 只供调试。"
         "期望值的中心是纹理坐标（y 向上，不除 aspect），半径是度量值，axis 是度量空间单位向量，v = (-u.y, u.x) 指向额头。",
    cases=out_geo)

# ---------------------------------------------------------------- masks
mask_geos = ["frontal_375", "yaw15_roll10", "mirrored_yaw10_roll5", "yaw25_warp_off", "landscape_frontal"]
mask_cases = []
for name in mask_geos:
    g, aspect = GEOS[name]
    g = geo_json(g)
    ox, oy, orx, ory = g["oval"]
    em, mo = g["eyeMasks"], g["mouth"]
    probes = [(ox, oy), (em[0], em[1]), (em[4], em[5]), (mo[0], mo[1]), (0.01, 0.99), (0.99, 0.01), (0.5, 0.02),
              (ox, oy + 0.9 * ory * aspect), (ox, oy - 0.95 * ory * aspect), (ox + 0.9 * orx, oy), (ox - 1.0 * orx, oy),
              (em[0] + 0.8 * em[2], em[1]), (em[0], em[1] - 0.85 * em[3] * aspect),
              (mo[0] + 0.9 * mo[2], mo[1]), (mo[0], mo[1] + 0.85 * mo[3] * aspect)]
    rs = random.Random(name)
    probes += [(ox + rs.uniform(-1.2, 1.2) * orx, oy + rs.uniform(-1.2, 1.2) * ory * aspect) for _ in range(15)]
    probes = [r6(p) for p in probes]
    for presence in (1.0, 0.4, 0.0):
        sel = probes if presence > 0 else probes[:4]
        m = B.face_masks(np.array(sel), g, aspect, presence)
        for i, p in enumerate(sel):
            mask_cases.append(dict(geometry=name, presence=presence, uv=p, expected={k: r6(v[i]) for k, v in m.items()}))
out["masks"] = dict(
    note="uv 是纹理坐标（y 向上），即逆向变形后的采样坐标；geometries 给出 uniform（与 geometry 段同名用例的 expected 一致）。"
         "presence == 0 时 face = excl = 0，各端可以不算遮罩；oval/eye0/eye1/mouth 仍给出纯几何值。",
    geometries={n: dict(aspect=r6(GEOS[n][1]), **{k: geo_json(GEOS[n][0])[k] for k in ("oval", "axis", "mouth", "eyeMasks")})
                for n in mask_geos},
    cases=mask_cases)

# ---------------------------------------------------------------- composite
P_SETS = {
    "neutral": dict(),
    "smoothing_1": dict(smoothing=1.0), "smoothing_0.5": dict(smoothing=0.5),
    "smoothing_0.6_all": dict(smoothing=0.6, skinOnly=False),
    "sharpness_1": dict(sharpness=1.0), "sharpness_0.4": dict(sharpness=0.4),
    "sharp_1_smooth_1": dict(sharpness=1.0, smoothing=1.0),
    "whiten_1": dict(whiten=1.0), "whiten_0.4": dict(whiten=0.4),
    "rosy_1": dict(rosy=1.0), "rosy_0.5": dict(rosy=0.5),
    "brightness_1": dict(brightness=1.0), "brightness_0.3": dict(brightness=0.3),
    "saturation_0": dict(saturation=0.0), "saturation_1.5": dict(saturation=1.5), "saturation_2": dict(saturation=2.0),
    "filter_warm": dict(filter="warm", filterIntensity=1.0), "filter_cool_0.7": dict(filter="cool", filterIntensity=0.7),
    "filter_mono": dict(filter="monochrome", filterIntensity=1.0), "filter_sepia_0.5": dict(filter="sepia", filterIntensity=0.5),
    "filter_rose": dict(filter="rose", filterIntensity=1.0),
    "whiten_rosy": dict(whiten=0.6, rosy=0.8),
    "sharp_whiten_sat": dict(sharpness=1.0, whiten=1.0, saturation=1.5),
    "preset_natural": dict(smoothing=0.15, brightness=0.10, whiten=0.10, sharpness=0.15),
    "preset_soft": dict(smoothing=0.30, brightness=0.10, whiten=0.20, rosy=0.15, sharpness=0.10, saturation=0.95,
                        filter="rose", filterIntensity=0.35),
    "preset_vivid": dict(smoothing=0.10, brightness=0.05, whiten=0.05, rosy=0.10, sharpness=0.30, saturation=1.2,
                         filter="warm", filterIntensity=0.3),
    "preset_cinema": dict(smoothing=0.10, brightness=0.05, sharpness=0.2, saturation=0.85, filter="sepia", filterIntensity=0.35),
    "everything": dict(smoothing=0.9, sharpness=0.8, whiten=0.8, rosy=0.7, brightness=0.6, saturation=1.3, filter="warm",
                       filterIntensity=0.6),
    "max_all": dict(smoothing=1.0, sharpness=1.0, whiten=1.0, rosy=1.0, brightness=1.0, saturation=2.0, filter="sepia",
                    filterIntensity=1.0),
}
FULL_P = {n: dict(B.DEFAULT_PARAMS, **p) for n, p in P_SETS.items()}
# 只给「锐化后 clamp」区分用例用的参数组：不进入下面按肤色遍历的循环（不改变既有用例与随机序列）。
CLAMP_P_SETS = {
    "sharp_mono": dict(sharpness=1.0, filter="monochrome", filterIntensity=1.0),
    "sharp_sat2": dict(sharpness=1.0, saturation=2.0),
    "sharp_sat0": dict(sharpness=1.0, saturation=0.0),
    "sharp_sepia": dict(sharpness=1.0, filter="sepia", filterIntensity=1.0),
}
FULL_P.update({n: dict(B.DEFAULT_PARAMS, **p) for n, p in CLAMP_P_SETS.items()})
for p in FULL_P.values():
    p["style"] = B.STYLES[p["filter"]]
TONES = {"ita_light": [192, 156, 133], "ita_dark": [108, 75, 59], "mst10": [41, 36, 32], "pale_cool": [229, 233, 240],
         "black_hair": [26, 23, 23], "white": [255, 255, 255], "black": [0, 0, 0], "blue": [51, 77, 153],
         "lips": [184, 77, 82], "grey": [128, 128, 128]}
MASK_SETS = {"in_face": (1.0, 0.0, 1.0), "no_face": (0.0, 0.0, 0.0), "feather": (0.35, 0.0, 1.0), "eye_excluded": (0.1, 0.9, 1.0)}
MASKED_SETS = ("smoothing_1", "whiten_1", "rosy_1", "preset_soft", "everything")
comp_cases = []
rs = random.Random(5)
pre_clamp_hits = 0
# 「锐化后 clamp」区分用例：去掉该 clamp 时输出至少要变 4/255，远超合成容差 1.5/255，实现漏掉 clamp 必被发现。
CLAMP_SENSITIVE_MIN = 4.0 / 255
clamp_sensitive = []


def comp_case(name, tone, pset, I, mean, var, a4, face, excl, presence, clamp_sensitive_case=False):
    global pre_clamp_hits
    I, mean, a4, var = r6(I), r6(mean), r6(a4), r6(var)
    exp, dbg = B.composite(np.array(I), np.array(mean), var, np.array(a4), face, excl, FULL_P[pset], debug=True)
    if clamp_sensitive_case:
        unclamped = B.composite(np.array(I), np.array(mean), var, np.array(a4), face, excl, FULL_P[pset],
                                sharpen_clamp=False)
        delta = float(np.max(np.abs(exp - unclamped)))
        assert delta > CLAMP_SENSITIVE_MIN, (name, delta * 255)
        clamp_sensitive.append((name, delta))
    if pset == "neutral":
        assert np.array_equal(exp, np.array(I)), "中性参数必须精确直通"
    if FULL_P[pset]["sharpness"] > 0 and (np.any(dbg["sharpened"] > 1.0) or np.any(dbg["sharpened"] < 0.0)):
        pre_clamp_hits += 1
    comp_cases.append(dict(name=name, tone=tone, params=pset, I=I, mean=mean, var=var, avg4=a4,
                           face=face, excl=excl, presence=presence,
                           debug=dict(gate=r6(dbg["gate"]), mSkin=r6(dbg["mSkin"]), k=r6(dbg["k"])),
                           expected=r6(exp), expected_rgb8=rgb8(exp)))


for tn, t8 in TONES.items():
    I = [c / 255.0 for c in t8]
    for pn in P_SETS:
        for mn, (face, excl, presence) in MASK_SETS.items():
            if mn in ("feather", "eye_excluded") and pn not in MASKED_SETS:
                continue
            mean = [min(1.0, max(0.0, c + rs.uniform(-0.06, 0.06))) for c in I]
            d = rs.choice([-0.05, -0.015, -0.006, 0.0, 0.006, 0.02, 0.06])
            a4 = [min(1.0, max(0.0, c + d)) for c in I]
            var = rs.choice([0.0, 0.02, 0.08, 0.15, 0.4, 1.0])
            comp_case("%s/%s/%s" % (tn, pn, mn), tn, pn, I, mean, var, a4, face, excl, presence)
# 锐化把颜色推出 [0, 1]，验证「锐化后、美白前 clamp」
for tn, I, a4 in (("hot_orange", [0.99, 0.5, 0.3], [0.93, 0.44, 0.24]), ("deep_shadow", [0.02, 0.05, 0.2], [0.08, 0.11, 0.26]),
                  ("white", [1.0, 1.0, 1.0], [0.94, 0.94, 0.94])):
    for mn in ("in_face", "no_face"):
        face, excl, presence = MASK_SETS[mn]
        comp_case("clamp_before_whiten/%s/%s" % (tn, mn), tn, "sharp_whiten_sat", I, I, 0.0, a4, face, excl, presence)
# 上面几例在后续步骤里大多又被最终 clamp 吃掉，去掉「锐化后 clamp」只差不到 1/255。下面的用例让越界通道经
# 跨通道混合（单色 L601、饱和度 0/2、sepia 矩阵、美白 + 饱和度）流进其它通道，去掉该 clamp 时差 > 4/255。
# 均为非肤色、无人脸（锐化增益 0.8，越界 ±0.08）；最后一例在人脸内（mSkin 0.85，增益降为 0.39）。
CLAMP_CASES = [
    ("yellow_over", [0.97, 0.97, 0.3], [0.79, 0.79, 0.12], "sharp_mono", "no_face"),
    ("cyan_red_under", [0.02, 0.6, 0.6], [0.2, 0.78, 0.78], "sharp_mono", "no_face"),
    ("green_over", [0.6, 0.98, 0.6], [0.42, 0.8, 0.42], "sharp_sat2", "no_face"),
    ("magenta_green_under", [0.6, 0.03, 0.6], [0.78, 0.21, 0.78], "sharp_sat0", "no_face"),
    ("white_over", [0.98, 0.98, 0.98], [0.8, 0.8, 0.8], "sharp_sepia", "no_face"),
    ("purple_green_under", [0.3, 0.03, 0.3], [0.48, 0.21, 0.48], "sharp_sepia", "no_face"),
    ("yellow_over", [0.97, 0.97, 0.3], [0.79, 0.79, 0.12], "sharp_whiten_sat", "no_face"),
    ("magenta_green_under", [0.6, 0.03, 0.6], [0.78, 0.21, 0.78], "sharp_whiten_sat", "no_face"),
    ("white_over", [0.98, 0.98, 0.98], [0.8, 0.8, 0.8], "sharp_sepia", "in_face"),
]
for tn, I, a4, pn, mn in CLAMP_CASES:
    face, excl, presence = MASK_SETS[mn]
    comp_case("clamp_after_sharpen/%s/%s/%s" % (tn, pn, mn), tn, pn, I, I, 0.0, a4, face, excl, presence,
              clamp_sensitive_case=True)
assert pre_clamp_hits >= 4 + len(CLAMP_CASES), pre_clamp_hits
assert len(clamp_sensitive) == len(CLAMP_CASES)
out["composite"] = dict(
    note="逐像素合成阶段（规格 4.2 的 beautyColor），假定 processing 为真；中性直通由调用方另测。"
         "输入 I/mean/avg4 为 RGB 0..1，var 为方差标量，face/excl 为遮罩，presence 只作说明（合成只看 face/excl）。"
         "params 引用 paramSets 中的完整参数（含 style 编号）。expected 是精确浮点结果，expected_rgb8 = round(x*255)；"
         "debug 给出 gate、mSkin、k 便于定位。neutral 用例必须原样返回 I。"
         "clamp_after_sharpen/* 用例专门区分「锐化后、美白前 clamp(color, 0, 1)」：去掉该 clamp 时输出至少变 4/255"
         "（生成器断言），各用例的差值见 meta.counts.composite_clamp_sensitive_min_rgb8。",
    paramSets=FULL_P, cases=comp_cases)

# ---------------------------------------------------------------- prepass
PW, PH = 64, 96
rs = random.Random(21)
src = np.ones((PH, PW, 3)) * (np.array([192, 156, 133]) / 255.0)
src[:, :20] = np.array([0.10, 0.09, 0.09])                                                   # 硬边（头发）
yy, xx = np.mgrid[0:PH, 0:PW]
src -= (0.10 * np.exp(-(((xx - 44) ** 2 + (yy - 30) ** 2) / (2 * 3.0 ** 2))))[..., None]    # 斑点
src -= (0.12 * np.clip((yy - 60) / 30.0, 0, 1))[..., None] * (xx >= 20)[..., None]          # 渐变
noise = np.array([rs.gauss(0.0, 0.02) for _ in range(PH * PW * 3)]).reshape(PH, PW, 3)
src8 = np.round(np.clip(src + noise, 0, 1) * 255).astype(int)
img = src8 / 255.0
pf, pq = B.prepass_float(img), B.prepass_rgba8(img)
PROBES = [(0, 0), (1, 2), (17, 40), (20, 41), (23, 47), (44, 30), (40, 28), (63, 95), (62, 3), (30, 80), (21, 70),
          (10, 10), (63, 0), (0, 95), (19, 60), (48, 64)]


def probe_vals(mean_q, var_q):
    mu, vu = B.upsample4(mean_q, PH, PW), B.upsample4(var_q, PH, PW)
    return [dict(x=x, row=y, uv=r6(pixel_uv(x, y, PW, PH)), meanUp=r6(mu[y, x]), varUp=r6(vu[y, x])) for x, y in PROBES]


assert all(abs(a - b) < 2.0 / 255 for a, b in zip(np.ravel(pf[1]), np.ravel(pq[1])))
out["prepass"] = dict(
    note="源图 rgb8[row] 为该行 W 个 RGB 三元组展平（第 0 行是画面顶部）。1/4 分辨率数组为 [row][col]（第 0 行也是顶部），"
         "尺寸 ceil(w/4) x ceil(h/4)。T0 = 4x4 盒式均值；mean = 9 抽头可分离高斯（先横后纵，clamp-to-edge）；"
         "v0 = min((dot(T0 - mean, L709) * 7.07)^2, 1)；var = v0 的同一高斯。float 为全浮点管线（iOS RGBAh 近似），"
         "rgba8 为 Android 5 通道 RGBA8 管线（每通道输出量化到 8 位，P3 用未量化 mean 算 v0）。"
         "probes 是放大后（GL_LINEAR / 4x 线性仿射）在全分辨率像素中心的值，x 为列、row 为行（自上而下），uv 为纹理坐标。"
         "iOS 中 Core Image 第 0 行是底部，交换原始缓冲时要翻转行。",
    width=PW, height=PH, rgb8=src8.reshape(PH, PW * 3).tolist(),
    float=dict(T0=r6(pf[0]), mean=r6(pf[1]), v0=r6(pf[2]), var=r6(pf[3]), probes=probe_vals(pf[1], pf[3])),
    rgba8=dict(T0=r6(pq[0]), mean=r6(pq[1]), v0=r6(pq[2]), var=r6(pq[3]), probes=probe_vals(pq[1], pq[3])))

# ---------------------------------------------------------------- smoother
def geo_vec(**kw):
    g = B.full_geometry(to_points(landmarks(**kw)), ASPECT)
    return None if g is None else r6(B.geometry_to_vector(g))


smoother_seqs = []


def run_seq(name, events, note):
    """events：(now, sampleTs 或 None, target 或 None) 或字符串 'reset'。相同 (sampleTs, target) 只存一次。"""
    sm = B.FinalSmoother(ASPECT)
    samples, steps = [], []
    for ev in events:
        if ev == "reset":
            sm.reset()
            steps.append(dict(op="reset"))
            continue
        now, ts, tgt = ev
        idx = None
        if tgt is not None:
            tgt = r6(tgt)
            key = dict(sampleTs=ts, target=tgt)
            if not samples or samples[-1] != key:
                samples.append(key)
            idx = len(samples) - 1
        x, pres = sm.update(tgt, ts, now)
        steps.append(dict(op="update", now=now, sample=idx, expectedState=r6(x), expectedPresence=r6(pres)))
    smoother_seqs.append(dict(name=name, note=note, samples=samples, steps=steps))


rs = random.Random(31)
JIT_MASK = [0.0 if k in ("axis", "none") else 1.0 for k in B.STATE_KINDS]


def jit(v, s=0.0015):
    return [a + rs.gauss(0.0, s) * m for a, m in zip(v, JIT_MASK)]


base = geo_vec()
# 1) 静止抖动：每 100 ms 一个样本、晚到 67 ms，渲染每 33/34 ms 一帧
ev, cur = [], None
for i in range(40):
    now = round(i * 33.333)
    if i % 3 == 2:
        cur = (now - 67, jit(base))
    ev.append((now, cur[0] if cur else None, cur[1] if cur else None))
run_seq("static_jitter", ev, "前两帧无目标；presence 150 ms 渐入；抖动多数落在死区内")
# 2) 快速移动后停止
ev, cur = [], None
for i in range(45):
    now = round(i * 33.333)
    if i % 3 == 0:
        ts = now - 67 if i else 0
        t = max(0.0, min(0.4, (ts - 300) / 1000.0))
        cur = (ts, geo_vec(center=(250 + 562.0 * t, 600)))
    ev.append((now, cur[0], cur[1]))
run_seq("fast_move_then_stop", ev, "1.5 脸宽/秒移动 0.4 s；输出在样本间线性滑行，无过冲")
# 3) 掉帧 100/300/700 ms，随后在别处重新捕获；再加 dt == 0 与 dt > 100 ms 的调用
ev, t = [], 0


def push(n, vec, step=33):
    global t
    for _ in range(n):
        ev.append((t, t - 34 if vec is not None else None, vec))
        t += step


push(8, base); push(3, None); push(6, base); push(9, None); push(8, base); push(22, None)
push(6, geo_vec(center=(420, 640)))
ev.append((t - 33, t - 60, base))                              # dt == 0：原样返回上次结果
ev.append((t + 400, t + 380, geo_vec(center=(430, 640))))      # dt 截到 100 ms
run_seq("dropouts_and_dt_edges", ev, "保持 150 ms、250 ms 渐出、presence 到 0 清空几何、清空后的首个样本直接对齐；dt==0 原样返回；dt 截到 100 ms")
# 4) reset
ev = [(0, 0, base), (33, 0, base), (67, 60, jit(base)), "reset", (100, 90, geo_vec(center=(300, 500))),
      (133, 90, geo_vec(center=(300, 500)))]
run_seq("reset", ev, "reset() 立即清空状态与 presence（不渐出）")
# 5) 转头：warpWeight 随侧脸量 1 → 部分 → 0 → 1
yaws = [0, 8, 14, 17, 20, 25, 25, 25, 18, 10, 0, 0, 0]
ev = []
for i in range(len(yaws) * 3):
    now = round(i * 33.333)
    j = i // 3
    ev.append((now, j * 100, geo_vec(yaw=yaws[j])))
run_seq("head_turn_warp_weight", ev, "侧脸量变化时 warpWeight（index 40）与几何一起滤波和插值")
# 6) 几何不变、只有 warpWeight 跳变：偏差 s 不含 warpWeight，滤波按最低截止频率缓变
ww_only = []
for i, w in enumerate([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]):
    v = list(base)
    v[B.STATE_WARP_INDEX] = w
    ww_only.append(v)
ev = [(round(i * 33.333), (i // 3) * 100, ww_only[i // 3]) for i in range(len(ww_only) * 3)]
run_seq("warp_weight_only", ev, "warpWeight 不参与偏差：只有它变化时 s = 0，fc = 0.6 Hz")
assert all(len(s["samples"][0]["target"]) == B.STATE_SIZE for s in smoother_seqs)
out["smoother"] = dict(
    note="FinalSmoother 逐步回放。samples 是本序列的样本表（sampleTs + 41 维 target），step.sample 为样本下标或 null（无目标）。"
         "op=update 调用 update(target?, sampleTs, now)；op=reset 调用 reset()。expectedState 为 41 维输出或 null，"
         "expectedPresence 为 presence。新样本以 sampleTs 变化识别。float32 实现容差 1e-4。",
    aspect=r6(ASPECT), stateKinds=B.STATE_KINDS, rxIndex=B.STATE_RX_INDEX, axisIndex=B.STATE_AXIS_INDEX,
    warpWeightIndex=B.STATE_WARP_INDEX, sequences=smoother_seqs)

# ---------------------------------------------------------------- end_to_end
E2E_ASPECT = PW / PH
e2e_lm = landmarks(width=PW, height=PH, face_px=40, center=(42, 50))
e2e_geo, why = B.geometry_with_reason(to_points(e2e_lm), E2E_ASPECT)
assert e2e_geo is not None, why
e2e_geo = geo_json(e2e_geo)


def uv_to_px(u, v):
    return min(PW - 1, max(0, int(u * PW))), min(PH - 1, max(0, int((1.0 - v) * PH)))


g_ox, g_oy, g_orx, g_ory = e2e_geo["oval"]
E2E_PROBES = [(0, 0), (63, 0), (0, 95), (63, 95), (19, 10), (20, 10), (21, 40), (18, 70), (44, 30), (43, 31), (46, 28),
              (30, 75), (50, 90), (40, 62), (55, 10), (10, 50),
              uv_to_px(g_ox, g_oy), uv_to_px(*e2e_geo["eyeMasks"][0:2]), uv_to_px(*e2e_geo["eyeMasks"][4:6]),
              uv_to_px(*e2e_geo["mouth"][0:2]), uv_to_px(g_ox + 0.95 * g_orx, g_oy), uv_to_px(g_ox - 0.8 * g_orx, g_oy),
              uv_to_px(g_ox, g_oy + 0.9 * g_ory * E2E_ASPECT), uv_to_px(g_ox, g_oy - 0.9 * g_ory * E2E_ASPECT)]
assert len(set(E2E_PROBES)) == len(E2E_PROBES) >= 20, E2E_PROBES
E2E_CASES = [
    ("neutral_passthrough", dict(), 1.0),
    ("smooth_face_skin_only", dict(smoothing=0.8), 1.0),
    ("smooth_all_sharp_no_face", dict(smoothing=0.6, skinOnly=False, sharpness=0.7), 0.0),
    ("tone_face", dict(whiten=0.7, rosy=0.6, brightness=0.4, saturation=1.1), 0.7),
    ("preset_soft", P_SETS["preset_soft"], 1.0),
    ("everything", dict(smoothing=1.0, sharpness=1.0, whiten=1.0, rosy=1.0, brightness=1.0, saturation=1.3,
                        filter="sepia", filterIntensity=0.5), 0.6),
]
e2e_out, e2e_maxdiff = [], 0
for name, p, presence in E2E_CASES:
    full = dict(B.DEFAULT_PARAMS, **p)
    full["style"] = B.STYLES[full["filter"]]
    geo = e2e_geo if presence > 0 else None
    res_f = B.process_frame(img, full, geo, presence, pipeline="float")
    res_q = B.process_frame(img, full, geo, presence, pipeline="rgba8")
    a = np.round(res_f * 255).astype(int)
    b = np.round(res_q * 255).astype(int)
    e2e_maxdiff = max(e2e_maxdiff, int(np.abs(a - b).max()))
    probe_rgb8 = [a[y, x].tolist() for x, y in E2E_PROBES]
    assert all(abs(int(a[y, x, c]) - int(b[y, x, c])) <= 2 for x, y in E2E_PROBES for c in range(3)), name
    if name == "neutral_passthrough":
        assert np.array_equal(a, src8)
    e2e_out.append(dict(name=name, params=full, presence=presence, useGeometry=geo is not None, expected_rgb8=probe_rgb8))
out["end_to_end"] = dict(
    note="整帧管线（预处理仅 smoothing > 0 时执行 → 遮罩 → 合成），源图为 prepass.rgb8（64x96，aspect = 64/96），无变形"
         "（faceSlim = eyeEnlarge = 0，采样坐标就是像素中心 uv）。useGeometry 为真时用 geometry 的遮罩 uniform 与给定 presence，"
         "否则 presence = 0、face = excl = 0。期望值取浮点管线；Android RGBA8 管线在全部探针上与之相差不超过 2/255，"
         "全图最大差见 rgba8MaxDiffRgb8。容差 3/255；neutral_passthrough 须逐字节等于源图。"
         "state 是同一几何的 41 维平滑器状态（布局见 meta.conventions.state），GLSL 门禁按 VideoEffectsFilter.kt 的"
         "打包代码把它拆成 uniform，再与按 meta.uniforms.android 由 geometry 各字段拼出的值逐项比较。",
    width=PW, height=PH, aspect=r6(E2E_ASPECT), source="prepass.rgb8",
    landmarks={str(i): e2e_lm[i] for i in USED}, geometry=e2e_geo,
    state=r6(B.geometry_to_vector(e2e_geo)),
    probes=[dict(x=x, row=y, uv=r6(pixel_uv(x, y, PW, PH))) for x, y in E2E_PROBES],
    tolerance_rgb8=3, rgba8MaxDiffRgb8=e2e_maxdiff, cases=e2e_out)

# ---------------------------------------------------------------- ios_selftest
st_params = dict(B.DEFAULT_PARAMS, smoothing=1.0, sharpness=1.0, brightness=1.0, skinOnly=False)
st_img = np.ones((8, 8, 3)) * (np.array([170, 115, 85]) / 255.0)
st = [rgb8(B.process_frame(st_img, st_params, pipeline=pl)[4, 4]) for pl in ("float", "rgba8")]
assert st[0] == st[1] == [191, 141, 109], st
out["ios_selftest"] = dict(
    note="iOS 首次使用的内核自检：8x8 纯色帧，渲染到预填哨兵值的缓冲，中心像素须在 ±1 内且 alpha = 255。"
         "纯色帧上 mean = I、var = 0、细节为 0，只有 lift(x, 0.5) 可见；skinOnly 不影响结果。",
    width=8, height=8, rgb8=[170, 115, 85], params=dict(st_params, style=0),
    expected_center_rgb8=st[0], expected_alpha=255, tolerance_rgb8=1)

# ---------------------------------------------------------------- meta（放最前）
meta = dict(
    generator="reference/gen_vectors.py（参考模型 beauty_ref.py，规格 v1）",
    spec="docs/plans/2026-11-11-live-beauty-capture-optimization-plan.md",
    regenerate="cd reference && BEAUTY_PYDEPS=<numpy 目录> /usr/bin/python3 -I gen_vectors.py",
    numpy=np.__version__,
    conventions=dict(
        colour="伽马编码 0..1 浮点；rgb8 = round(x * 255)。所有计算都在伽马编码值上进行。",
        landmarks="geometry 段输入是 MediaPipe 归一化图像坐标，y 向下；进入几何前转纹理坐标 (x, 1 - y)。",
        texture="纹理坐标归一化，y 向上：像素 (x, row) 的中心 uv = ((x + 0.5) / W, 1 - (row + 0.5) / H)，row 0 是画面顶部。",
        metric="度量空间 p' = (x, y / aspect)，aspect = W / H。中心（oval/mouth/eyeMasks 的 cx, cy 与 cheeks/eyes）是纹理坐标，"
               "半径（rx, ry, eyes.z）是度量值（x 单位），axis 是度量空间单位向量 u，v = (-u.y, u.x) 指向额头。",
        mask="E(uv, c, u)：d = (uv.x - c.x, (uv.y - c.y) / aspect)；a = clamp(dot(d, u) / max(c.z, 1e-4), -4, 4)；"
             "b = clamp(dot(d, v) / max(c.w, 1e-4), -4, 4)；E = a*a + b*b。",
        state="平滑器状态 41 维：cheeks[0..15]、eyes[16..21]、oval[22..25]（rx 在 24）、axis[26..27]、mouth[28..31]、"
              "eyeMasks[32..39]、warpWeight[40]（不参与偏差）。",
        pixels="prepass/end_to_end 的数组与探针 row 自上而下；Core Image 第 0 行是底部，iOS 测试交换原始缓冲时要翻转行。",
    ),
    uniforms=dict(
        android=dict(
            uFrame="(1/w, 1/h, aspect, presence)",
            uParams="(processing, smoothing, brightness, style)；style：none 0、warm 1、cool 2、monochrome 3、sepia 4、rose 5",
            uDetails="(skinOnly ? 1 : 0, sharpness, saturation, filterIntensity)",
            uTone="(whiten, rosy, faceSlim*0.12*presence*warpWeight, eyeEnlarge*0.25*presence*warpWeight)",
            uCheeks="cheeks[16] 按 4 个 vec4 原样上传（纹理坐标）",
            uEyes="vec4(cx, cy, 0.9*眼角距, axis 分量)：uEyes[0] = (eyes[0..2], axis.x)，uEyes[1] = (eyes[3..5], axis.y)",
            uOval="oval 原样（纹理中心 + 度量半径）", uMouth="mouth 原样",
            uEyeMask="eyeMasks[0..3]、eyeMasks[4..7] 原样",
            note="presence 与 warpWeight 取平滑器输出（state[40]）；采样坐标 = vTextureCoord 逆向变形后的值。"),
        ios=dict(
            pixelSpace="像素空间 = 度量空间 × w（各向同性）：中心 × (w, h)，半径与眼宽 × w，axis 不变；destCoord() 就是像素坐标，"
                       "CI 原点在左下，y 向上，与纹理坐标一致（uv = destCoord / (w, h)）。",
            oval="CIVector(cx*w, cy*h, rx*w, ry*w)", mouth="同 oval", eyeMask0="eyeMasks[0..3] 同 oval 换算",
            eyeMask1="eyeMasks[4..7] 同 oval 换算", axis="CIVector(u.x, u.y)",
            eye0="CIVector(cx*w, cy*h, 0.9*眼角距*w)（既有 warp 换算）", c0_c3="CIVector(ox*w, oy*h, tx*w, ty*h)（既有）",
            warp="(faceSlim*0.12*presence*warpWeight, eyeEnlarge*0.25*presence*warpWeight)", size="(w, h)",
            p0="(smoothing, skinOnly ? 1 : 0, sharpness, saturation)", p1="(whiten, rosy, brightness, filterIntensity)",
            style="同 Android 编号"),
    ),
    tolerance=dict(skin_gate=0.004, geometry=2e-5, masks=0.01, composite=r6(1.5 / 255),
                   composite_note="expected 为精确浮点结果；8 位回读另加 0.5 LSB，fp32 计算误差 < 0.1 LSB，真 mediump（fp16）可达 0.65 LSB，"
                                  "不在共享向量覆盖范围内",
                   prepass_float=r6(1.0 / 255), prepass_rgba8=r6(2.0 / 255), smoother=1e-4,
                   end_to_end_rgb8=3, ios_selftest_rgb8=1),
    counts=dict(skin_gate=len(out["skin_gate"]["cases"]), geometry=len(out["geometry"]["cases"]),
                geometry_null=sum(1 for c in out["geometry"]["cases"] if c["expected"] is None),
                masks=len(mask_cases), composite=len(comp_cases), composite_pre_clamp=pre_clamp_hits,
                composite_clamp_sensitive=len(clamp_sensitive),
                composite_clamp_sensitive_min_rgb8=round(min(d for _, d in clamp_sensitive) * 255, 2),
                smoother_sequences=len(smoother_seqs), smoother_steps=sum(len(s["steps"]) for s in smoother_seqs),
                end_to_end_cases=len(e2e_out), end_to_end_probes=len(E2E_PROBES)),
)
out = dict(meta=meta, **out)


def emit(o, depth=0):
    """前两层对象与对象数组逐行输出，便于 diff；其余紧凑。"""
    def compact(v):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(o, dict) and depth < 2:
        return "{\n" + ",\n".join(compact(k) + ":" + emit(v, depth + 1) for k, v in o.items()) + "\n}"
    if isinstance(o, list) and depth <= 2 and o and all(isinstance(e, dict) for e in o):
        return "[\n" + ",\n".join(compact(e) for e in o) + "\n]"
    return compact(o)


path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "beauty_vectors.json")
with open(path, "w", encoding="utf-8") as f:
    f.write(emit(out) + "\n")
c = meta["counts"]
print("wrote %s (%.0f KB): %d gate, %d geometry (%d null), %d mask, %d composite, %d smoother sequences (%d steps), "
      "prepass %dx%d, end_to_end %d x %d probes" % (
          path, os.path.getsize(path) / 1024, c["skin_gate"], c["geometry"], c["geometry_null"], c["masks"], c["composite"],
          c["smoother_sequences"], c["smoother_steps"], PW, PH, c["end_to_end_cases"], c["end_to_end_probes"]))
