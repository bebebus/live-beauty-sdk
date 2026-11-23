#!/usr/bin/env python3
"""程序化生成 Android 实时美妆引擎所需的 5 张通道打包 512×512 RGBA PNG 纹理贴图，并校验规范与体积上限。

严格对齐随包 MediaPipe face_landmarker.task 的 468 点规范 UV 坐标（u 左→右，v 上→下，PNG 第 0 行对应 v=0 即额头，
与 Android GLUtils.texImage2D 的 t=0 方向完全一致）。仅依赖 Python 标准库，输出完全确定性。

贴图目录：android/live-beauty/src/main/assets/makeup/
  1. blush_atlas.png       — R=苹果肌腮红, G=微醺横扫腮红, B=眼下纯欲腮红, A=255
  2. contour_highlight.png — R=鼻影/颌线修容阴影, G=鼻梁/颧骨/眉骨/唇峰高光, B=唇带边缘清晰羽化, A=255
  3. eyebrow_atlas.png     — 上下分屏高清眉区贴图（上半右眉眼 [0.18..0.48, 0.21..0.45]，下半左眉眼 [0.52..0.82, 0.21..0.45]）：R=自然野生眉, G=精致弯眉, B=卧蚕下缘微阴影, A=255
  4. eye_makeup.png        — 上下分屏高清眼妆贴图（同局部放大 UV）：R=眼影主晕染, G=眼影外眼角深层晕染, B=贴睑精细微扬眼线, A=卧蚕提亮高光
  5. iris_atlas.png        — 2×2 径向美瞳图集（256×256×4：hazel / chocolate / mist_grey / sakura_pink），中心瞳孔全透明

用法：
  python3 -I gen_makeup_assets.py          # 生成 5 张贴图并校验
  python3 -I gen_makeup_assets.py --verify # 仅校验现有贴图尺寸、内容安全边界与总体积 (< 800 KiB)
"""
import argparse
import math
import os
import struct
import sys
import zipfile
import zlib

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'ios', 'Resources', 'face_landmarker.task')
OUT_DIR = os.path.join(ROOT, 'android', 'live-beauty', 'src', 'main', 'assets', 'makeup')
SIZE = 512
MAX_TOTAL_BYTES = 800 * 1024


def _read_varint(buf, i):
    val, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def _parse_proto(buf):
    i, n = 0, len(buf)
    fields = {}
    while i < n:
        tag, i = _read_varint(buf, i)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:
            v, i = _read_varint(buf, i)
            fields.setdefault(fnum, []).append(v)
        elif wtype == 5:
            v = struct.unpack_from('<f', buf, i)[0]
            i += 4
            fields.setdefault(fnum, []).append(v)
        elif wtype == 2:
            ln, i = _read_varint(buf, i)
            v = buf[i:i + ln]
            i += ln
            fields.setdefault(fnum, []).append(v)
        elif wtype == 1:
            v = struct.unpack_from('<d', buf, i)[0]
            i += 8
            fields.setdefault(fnum, []).append(v)
    return fields


def load_canonical_uvs():
    with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
        pb = zf.read('geometry_pipeline_metadata_landmarks.binarypb')
    top = _parse_proto(pb)
    mesh = _parse_proto(top[1][0])
    vb = mesh[3]
    return [(vb[i * 5 + 3], vb[i * 5 + 4]) for i in range(468)]


def _png_chunk(tag, payload):
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack('>I', len(payload)) + tag + payload + struct.pack('>I', crc)


def encode_rgba_png(w, h, rgba_bytes):
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    raw = bytearray()
    stride = w * 4
    prev = bytes(stride)
    for y in range(h):
        row = rgba_bytes[y * stride:(y + 1) * stride]
        # 标准 PNG Sub(1) 与 Up(2) 自适应无损行差分滤波，像素逐字节零损失且压缩率提升 2~3 倍。
        sub = bytearray(stride)
        up = bytearray(stride)
        s_cost = u_cost = 0
        for i in range(stride):
            sv = (row[i] - (row[i - 4] if i >= 4 else 0)) & 0xFF
            uv = (row[i] - prev[i]) & 0xFF
            sub[i] = sv
            up[i] = uv
            s_cost += sv if sv < 128 else 256 - sv
            u_cost += uv if uv < 128 else 256 - uv
        if s_cost <= u_cost:
            raw.append(1)
            raw.extend(sub)
        else:
            raw.append(2)
            raw.extend(up)
        prev = row
    idat = zlib.compress(bytes(raw), level=9)
    return sig + _png_chunk(b'IHDR', ihdr) + _png_chunk(b'IDAT', idat) + _png_chunk(b'IEND', b'')


def decode_rgba_png(data):
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'Not a PNG'
    i = 8
    w = h = 0
    idat = bytearray()
    while i < len(data):
        ln = struct.unpack_from('>I', data, i)[0]
        tag = data[i + 4:i + 8]
        payload = data[i + 8:i + 8 + ln]
        i += 12 + ln
        if tag == b'IHDR':
            w, h, bd, ct = struct.unpack_from('>IIBB', payload, 0)
            assert bd == 8 and ct == 6, f'Expected 8-bit RGBA, got bd={bd} ct={ct}'
        elif tag == b'IDAT':
            idat.extend(payload)
    raw = zlib.decompress(bytes(idat))
    out = bytearray(w * h * 4)
    stride = w * 4
    for y in range(h):
        ftype = raw[y * (stride + 1)]
        filt = raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
        base = y * stride
        prev_base = (y - 1) * stride
        if ftype == 0:
            out[base:base + stride] = filt
        elif ftype == 1:
            for k in range(stride):
                a = out[base + k - 4] if k >= 4 else 0
                out[base + k] = (filt[k] + a) & 0xFF
        elif ftype == 2:
            for k in range(stride):
                b = out[prev_base + k] if y > 0 else 0
                out[base + k] = (filt[k] + b) & 0xFF
        else:
            raise AssertionError(f'Unsupported PNG filter type: {ftype}')
    return w, h, bytes(out)


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def gauss2d(u, v, cu, cv, su, sv, angle=0.0):
    du = u - cu
    dv = v - cv
    if angle != 0.0:
        ca = math.cos(angle)
        sa = math.sin(angle)
        du, dv = du * ca + dv * sa, -du * sa + dv * ca
    return math.exp(-0.5 * ((du / su) ** 2 + (dv / sv) ** 2))


def dist_seg(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = clamp01(((px - ax) * abx + (py - ay) * aby) / denom)
    return math.hypot(px - (ax + abx * t), py - (ay + aby * t)), t


def dist_polyline(px, py, pts):
    best_d = 1e9
    best_u = 0.0
    total = max(1, len(pts) - 1)
    for k in range(len(pts) - 1):
        d, t = dist_seg(px, py, pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1])
        if d < best_d:
            best_d = d
            best_u = (k + t) / total
    return best_d, best_u


def point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def poly_edge_dist(px, py, poly):
    best = 1e9
    n = len(poly)
    for i in range(n):
        d, _ = dist_seg(px, py, poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
        if d < best:
            best = d
    return best


def _to_byte(x):
    return max(0, min(255, int(round(clamp01(x) * 255.0))))


def build_blush_atlas(uvs):
    """R=苹果肌腮红, G=微醺横扫腮红, B=眼下纯欲腮红, A=255。"""
    # 关键锚点：苹果肌中心（左 118/50，右 347/280）、眼下（119/120，348/349）、鼻梁（197）。
    lc_apple = ((uvs[118][0] + uvs[50][0]) * 0.5, (uvs[118][1] + uvs[50][1]) * 0.5 + 0.015)
    rc_apple = ((uvs[347][0] + uvs[280][0]) * 0.5, (uvs[347][1] + uvs[280][1]) * 0.5 + 0.015)
    lc_under = ((uvs[119][0] + uvs[118][0]) * 0.5, (uvs[119][1] + uvs[118][1]) * 0.5 - 0.008)
    rc_under = ((uvs[348][0] + uvs[347][0]) * 0.5, (uvs[348][1] + uvs[347][1]) * 0.5 - 0.008)
    drunk_line = [
        (uvs[117][0] - 0.01, uvs[117][1] + 0.012),
        (uvs[119][0], uvs[119][1] + 0.012),
        (uvs[197][0], uvs[197][1] + 0.010),
        (uvs[348][0], uvs[348][1] + 0.012),
        (uvs[346][0] + 0.01, uvs[346][1] + 0.012),
    ]
    buf = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        v = (y + 0.5) / SIZE
        for x in range(SIZE):
            u = (x + 0.5) / SIZE
            apple = max(
                gauss2d(u, v, lc_apple[0], lc_apple[1], 0.088, 0.068, -0.20),
                gauss2d(u, v, rc_apple[0], rc_apple[1], 0.088, 0.068, 0.20),
            )
            d_drunk, t_drunk = dist_polyline(u, v, drunk_line)
            # 微醺腮红两端丰满、鼻梁略轻，避免死红。
            bridge_dip = 1.0 - 0.35 * math.exp(-((t_drunk - 0.5) / 0.14) ** 2)
            drunk = math.exp(-0.5 * (d_drunk / 0.048) ** 2) * bridge_dip
            under = max(
                gauss2d(u, v, lc_under[0], lc_under[1], 0.072, 0.038, -0.10),
                gauss2d(u, v, rc_under[0], rc_under[1], 0.072, 0.038, 0.10),
            )
            k = (y * SIZE + x) * 4
            buf[k] = _to_byte(apple)
            buf[k + 1] = _to_byte(drunk)
            buf[k + 2] = _to_byte(under)
            buf[k + 3] = 255
    return bytes(buf)


def build_contour_highlight_atlas(uvs):
    """R=鼻影与颌线修容, G=鼻梁/苹果肌/眉骨/唇峰高光, B=唇带边缘羽化遮罩, A=255。"""
    nose_left = [uvs[i] for i in (193, 122, 196, 174, 217, 198)]
    nose_right = [uvs[i] for i in (417, 351, 419, 399, 437, 420)]
    jaw_left = [uvs[i] for i in (132, 58, 172, 136, 150, 149)]
    jaw_right = [uvs[i] for i in (361, 288, 397, 365, 379, 378)]
    nose_bridge = [uvs[i] for i in (168, 6, 197, 195, 5, 4)]
    brow_hl_l = [uvs[i] for i in (53, 52, 65)]
    brow_hl_r = [uvs[i] for i in (283, 282, 295)]
    outer_lip = [uvs[i] for i in (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)]
    inner_lip = [uvs[i] for i in (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)]

    buf = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        v = (y + 0.5) / SIZE
        for x in range(SIZE):
            u = (x + 0.5) / SIZE
            # R: 鼻侧影 + 柔和下颌/颧下修容
            dnl, _ = dist_polyline(u, v, nose_left)
            dnr, _ = dist_polyline(u, v, nose_right)
            djl, _ = dist_polyline(u, v, jaw_left)
            djr, _ = dist_polyline(u, v, jaw_right)
            nose_sh = max(math.exp(-0.5 * (dnl / 0.022) ** 2), math.exp(-0.5 * (dnr / 0.022) ** 2))
            jaw_sh = max(math.exp(-0.5 * (djl / 0.052) ** 2), math.exp(-0.5 * (djr / 0.052) ** 2)) * 0.72
            contour = clamp01(max(nose_sh * 0.88, jaw_sh))

            # G: 鼻梁高光 + 颧骨 C 区高光 + 眉骨 + 唇峰/下巴微光
            dnb, tnb = dist_polyline(u, v, nose_bridge)
            bridge_hl = math.exp(-0.5 * (dnb / 0.017) ** 2) * (0.65 + 0.35 * math.sin(tnb * math.pi))
            cheek_hl = max(
                gauss2d(u, v, uvs[117][0], uvs[117][1], 0.055, 0.032, -0.25),
                gauss2d(u, v, uvs[346][0], uvs[346][1], 0.055, 0.032, 0.25),
            ) * 0.75
            dbl, _ = dist_polyline(u, v, brow_hl_l)
            dbr, _ = dist_polyline(u, v, brow_hl_r)
            brow_hl = max(math.exp(-0.5 * (dbl / 0.020) ** 2), math.exp(-0.5 * (dbr / 0.020) ** 2)) * 0.55
            cupid_hl = gauss2d(u, v, uvs[0][0], uvs[0][1] - 0.012, 0.030, 0.012) * 0.55
            chin_hl = gauss2d(u, v, uvs[152][0], uvs[152][1] - 0.045, 0.042, 0.024) * 0.45
            highlight = clamp01(max(bridge_hl, cheek_hl, brow_hl, cupid_hl, chin_hl))

            # B: 柔焦唇带遮罩（仅用于 120 三角形唇带；宽外缘过渡 + 唇心到唇缘自然渐隐 + 宽嘴角柔化，杜绝硬边胶布感，内唇严格归零防涂牙）。
            in_outer = point_in_poly(u, v, outer_lip)
            in_inner = point_in_poly(u, v, inner_lip)
            if in_outer and not in_inner:
                d_out = poly_edge_dist(u, v, outer_lip)
                d_in = poly_edge_dist(u, v, inner_lip)
                # 上唇外缘略宽（0.018）防人中上溢，下唇外缘 0.015 自然柔焦；内唇湿线 0.0055 避免张嘴硬切边。
                mid_lip_v = 0.5 * (uvs[13][1] + uvs[14][1])
                outer_fade = smoothstep(0.001, 0.018 if v < mid_lip_v else 0.015, d_out)
                inner_fade = smoothstep(0.0005, 0.0055, d_in)
                # 唇心饱满、外唇线轻柔晕染（自然柔焦咬唇分布）。
                belly_grad = 0.80 + 0.20 * (1.0 - smoothstep(0.005, 0.026, d_in))
                # 嘴角宽幅渐隐，消除张嘴/微笑时的硬尖角。
                corner_l = math.hypot(u - uvs[61][0], v - uvs[61][1])
                corner_r = math.hypot(u - uvs[291][0], v - uvs[291][1])
                corner_fade = smoothstep(0.008, 0.044, min(corner_l, corner_r))
                lip_feather = clamp01(outer_fade * inner_fade * belly_grad * corner_fade)
            else:
                lip_feather = 0.0

            k = (y * SIZE + x) * 4
            buf[k] = _to_byte(contour)
            buf[k + 1] = _to_byte(highlight)
            buf[k + 2] = _to_byte(lip_feather)
            buf[k + 3] = 255
    return bytes(buf)


def _split_eye_uv(x, y):
    """将 512×512 贴图像素映射回规范人脸 UV（匹配 MAKEUP_LAYER_FRAGMENT uMode.y = 1.0）。
    上半屏 y in [0..255] -> 右眉眼 [0.18..0.48, 0.21..0.45]；
    下半屏 y in [256..511] -> 左眉眼 [0.52..0.82, 0.21..0.45]。
    返回 (u, v, is_left, on_border)。
    """
    half = SIZE // 2
    is_left = y >= half
    ly = y - half if is_left else y
    on_border = (x <= 1 or x >= SIZE - 2 or ly <= 1 or ly >= half - 2)
    eu = (x + 0.5) / SIZE
    ev = (ly + 0.5) / half
    u = (0.52 if is_left else 0.18) + eu * 0.30
    v = 0.21 + ev * 0.24
    return u, v, is_left, on_border


def build_eyebrow_atlas(uvs):
    """上下分屏高清局部放大眉妆与卧蚕阴影贴图：R=自然野生眉, G=精致弯眉, B=卧蚕下缘微阴影, A=255。"""
    def brow_center_curve(upper_ids, lower_ids, arch_lift=0.0):
        pts = []
        n = len(upper_ids)
        for k in range(n):
            u0, v0 = uvs[upper_ids[k]]
            u1, v1 = uvs[lower_ids[k]]
            t = k / (n - 1)
            # 弯眉在眉峰（t≈0.58）处精致上扬。
            lift = arch_lift * math.exp(-((t - 0.58) / 0.23) ** 2)
            pts.append(((u0 + u1) * 0.5, (v0 + v1) * 0.5 - lift))
        return pts

    # 从眉头（内侧）到眉尾（外侧）排序：t=0 眉头雾感羽化，t=1 眉尾利落收尖。
    r_nat = brow_center_curve([107, 66, 105, 63, 70], [55, 65, 52, 53, 46], arch_lift=0.0)
    l_nat = brow_center_curve([336, 296, 334, 293, 300], [285, 295, 282, 283, 276], arch_lift=0.0)
    r_arc = brow_center_curve([107, 66, 105, 63, 70], [55, 65, 52, 53, 46], arch_lift=0.012)
    l_arc = brow_center_curve([336, 296, 334, 293, 300], [285, 295, 282, 283, 276], arch_lift=0.012)

    # 卧蚕下缘微阴影（位于下眼睑下方 ~0.023 处）。
    r_aegyo_sh = [(uvs[i][0], uvs[i][1] + 0.023) for i in (7, 163, 144, 145, 153, 154)]
    l_aegyo_sh = [(uvs[i][0], uvs[i][1] + 0.023) for i in (249, 390, 373, 374, 380, 381)]

    def eval_brow(u, v, curve, is_left, base_w, sharp):
        d, t = dist_polyline(u, v, curve)
        taper = smoothstep(0.0, 0.20, t) * (1.0 - 0.68 * smoothstep(0.62, 1.0, t))
        width = base_w * (0.48 + 0.52 * (1.0 - smoothstep(0.56, 1.0, t)))
        val = math.exp(-0.5 * (d / max(width, 1e-4)) ** (2.8 if sharp else 2.2)) * taper
        # 高清局部放大 UV 下的真实斜向毛流笔触调制（左右眉对称生长方向）。
        stroke_u = (u - 0.5) * (-1.0 if is_left else 1.0)
        hair1 = math.sin(stroke_u * 620.0 + v * 260.0)
        hair2 = math.sin(stroke_u * 1050.0 + v * 410.0 + 1.1)
        strokes = 0.82 + 0.12 * hair1 + 0.06 * hair2
        return clamp01(val * strokes)

    buf = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        for x in range(SIZE):
            u, v, is_left, on_border = _split_eye_uv(x, y)
            if on_border:
                nat = arc = aegyo_sh = 0.0
            else:
                nat = eval_brow(u, v, l_nat if is_left else r_nat, is_left, 0.0165, sharp=False)
                arc = eval_brow(u, v, l_arc if is_left else r_arc, is_left, 0.0128, sharp=True)
                da, ta = dist_polyline(u, v, l_aegyo_sh if is_left else r_aegyo_sh)
                aegyo_sh = math.exp(-0.5 * (da / 0.0075) ** 2) * math.sin(clamp01(ta) * math.pi)
            k = (y * SIZE + x) * 4
            buf[k] = _to_byte(nat)
            buf[k + 1] = _to_byte(arc)
            buf[k + 2] = _to_byte(aegyo_sh)
            buf[k + 3] = 255
    return bytes(buf)


def build_eye_makeup_atlas(uvs):
    """上下分屏高清局部放大眼妆贴图：R=眼影主晕染, G=眼影深层晕染, B=贴睑微扬眼线, A=卧蚕提亮高光（严格避开眼裂内部）。"""
    r_eye_poly = [uvs[i] for i in (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)]
    l_eye_poly = [uvs[i] for i in (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)]

    # 上眼睑精细贴根曲线 + 外眼角秀场猫眼微扬尾翼（高清放大 UV 下锐利收尖）。
    r_wing_tip = (uvs[33][0] - 0.024, uvs[33][1] - 0.010)
    r_wing_mid = (uvs[33][0] - 0.011, uvs[33][1] - 0.004)
    l_wing_tip = (uvs[263][0] + 0.024, uvs[263][1] - 0.010)
    l_wing_mid = (uvs[263][0] + 0.011, uvs[263][1] - 0.004)
    r_lash = [r_wing_tip, r_wing_mid] + [(uvs[i][0], uvs[i][1] - 0.0012) for i in (33, 246, 161, 160, 159, 158, 157, 173)]
    l_lash = [l_wing_tip, l_wing_mid] + [(uvs[i][0], uvs[i][1] - 0.0012) for i in (263, 466, 388, 387, 386, 385, 384, 398)]

    # 眼影主晕染中心（紧贴自然上睑折痕）与深晕染中心（克制外眼角加深，不甩出太阳穴长尾）。
    r_sh_base = [(uvs[i][0] - 0.002, uvs[i][1] - 0.0115) for i in (33, 246, 161, 159, 157)]
    l_sh_base = [(uvs[i][0] + 0.002, uvs[i][1] - 0.0115) for i in (263, 466, 388, 386, 384)]
    r_sh_deep = [(uvs[33][0] - 0.005, uvs[33][1] - 0.006), (uvs[246][0], uvs[246][1] - 0.0055), (uvs[160][0], uvs[160][1] - 0.005)]
    l_sh_deep = [(uvs[263][0] + 0.005, uvs[263][1] - 0.006), (uvs[466][0], uvs[466][1] - 0.0055), (uvs[387][0], uvs[387][1] - 0.005)]
    r_eye_mid_v = 0.5 * (uvs[33][1] + uvs[133][1])
    l_eye_mid_v = 0.5 * (uvs[263][1] + uvs[362][1])
    r_brow_bot_v = min(uvs[i][1] for i in (53, 52, 65)) + 0.008
    l_brow_bot_v = min(uvs[i][1] for i in (283, 282, 295)) + 0.008

    # 卧蚕提亮带（紧贴下眼睑下方 ~0.011 处）。
    r_aegyo_hi = [(uvs[i][0], uvs[i][1] + 0.011) for i in (7, 163, 144, 145, 153, 154)]
    l_aegyo_hi = [(uvs[i][0], uvs[i][1] + 0.011) for i in (249, 390, 373, 374, 380, 381)]

    buf = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        for x in range(SIZE):
            u, v, is_left, on_border = _split_eye_uv(x, y)
            eye_poly = l_eye_poly if is_left else r_eye_poly
            if on_border or point_in_poly(u, v, eye_poly):
                sh_base = sh_deep = liner = aegyo_hi = 0.0
            else:
                # 眼裂外缘柔和距离羽化 + 上睑限定 + 眉下渐隐，彻底消除 1px 硬切眼洞剪影与糊眉/下渗黑眼圈。
                d_eye = poly_edge_dist(u, v, eye_poly)
                eye_edge_fade = smoothstep(0.0005, 0.0075, d_eye)
                eye_mid_v = l_eye_mid_v if is_left else r_eye_mid_v
                brow_bot_v = l_brow_bot_v if is_left else r_brow_bot_v
                upper_only = 1.0 - smoothstep(eye_mid_v - 0.003, eye_mid_v + 0.010, v)
                brow_guard = smoothstep(brow_bot_v - 0.006, brow_bot_v + 0.022, v)
                sh_guard = eye_edge_fade * upper_only * brow_guard

                d, t = dist_polyline(u, v, l_sh_base if is_left else r_sh_base)
                sh_base = math.exp(-0.5 * (d / 0.0165) ** 2) * (1.0 - 0.35 * smoothstep(0.65, 1.0, t)) * sh_guard

                dd, _ = dist_polyline(u, v, l_sh_deep if is_left else r_sh_deep)
                sh_deep = math.exp(-0.5 * (dd / 0.010) ** 2) * sh_guard

                # 高清局部放大空间下的锐利贴睑眼线（3.3× 横向分辨率，尾翼锋利不发糊）
                dl, tl = dist_polyline(u, v, l_lash if is_left else r_lash)
                lw = 0.0038 * (0.32 + 0.68 * smoothstep(0.0, 0.18, tl)) * (1.0 - 0.58 * smoothstep(0.66, 1.0, tl))
                liner = math.exp(-0.5 * (dl / max(lw, 1e-4)) ** 3.2) * smoothstep(0.0, 0.04, tl)

                da, ta = dist_polyline(u, v, l_aegyo_hi if is_left else r_aegyo_hi)
                aegyo_hi = math.exp(-0.5 * (da / 0.0085) ** 2) * math.sin(clamp01(ta) * math.pi)

            k = (y * SIZE + x) * 4
            buf[k] = _to_byte(sh_base)
            buf[k + 1] = _to_byte(sh_deep)
            buf[k + 2] = _to_byte(liner)
            buf[k + 3] = _to_byte(aegyo_hi)
    return bytes(buf)


def build_iris_atlas():
    """2×2 径向美瞳图集（256×256 RGBA，每格 128×128），中心瞳孔 r < 0.34 严格全透明。"""
    # (tileX, tileY, inner_rgb, mid_rgb, limbal_rgb, limbal_alpha)
    styles = [
        # (0,0) hazel 琥珀榛果
        (0, 0, (0.86, 0.62, 0.30), (0.64, 0.42, 0.20), (0.20, 0.13, 0.08), 0.92),
        # (1,0) chocolate 融瞳深巧
        (1, 0, (0.52, 0.35, 0.22), (0.38, 0.24, 0.15), (0.14, 0.09, 0.06), 0.95),
        # (0,1) mist_grey 混血雾灰
        (0, 1, (0.76, 0.72, 0.60), (0.56, 0.62, 0.68), (0.22, 0.25, 0.30), 0.90),
        # (1,1) sakura_pink 甜美粉棕
        (1, 1, (0.88, 0.60, 0.64), (0.66, 0.42, 0.46), (0.24, 0.14, 0.16), 0.90),
    ]
    tile = SIZE // 2
    buf = bytearray(SIZE * SIZE * 4)
    for tx, ty, c_in, c_mid, c_out, limbal_a in styles:
        for ly in range(tile):
            ny = ((ly + 0.5) / tile) * 2.0 - 1.0
            for lx in range(tile):
                nx = ((lx + 0.5) / tile) * 2.0 - 1.0
                r = math.hypot(nx, ny)
                theta = math.atan2(ny, nx)
                # 中心瞳孔严格透明 (r <= 0.33 -> alpha = 0)，外缘软羽化至 r = 0.98 归零。
                pupil_mask = smoothstep(0.34, 0.48, r)
                outer_mask = 1.0 - smoothstep(0.84, 0.98, r)
                base_alpha = pupil_mask * outer_mask

                # 径向虹膜纤维纹理（多谐波确定性叠加，真实自然无伪影）。
                fibers = (
                    0.50
                    + 0.22 * math.sin(theta * 24.0 + r * 6.0)
                    + 0.16 * math.sin(theta * 41.0 - r * 9.0 + 1.3)
                    + 0.12 * math.cos(theta * 63.0 + r * 4.0 + 2.1)
                )
                radial_t = clamp01((r - 0.38) / 0.46)
                limbal = smoothstep(0.70, 0.88, r)

                # 内圈星芒 → 中段主色 → 外圈锁边环
                r_col = c_in[0] * (1.0 - radial_t) + c_mid[0] * radial_t
                g_col = c_in[1] * (1.0 - radial_t) + c_mid[1] * radial_t
                b_col = c_in[2] * (1.0 - radial_t) + c_mid[2] * radial_t
                fiber_gain = 0.78 + 0.42 * fibers
                r_col = clamp01(r_col * fiber_gain)
                g_col = clamp01(g_col * fiber_gain)
                b_col = clamp01(b_col * fiber_gain)
                r_col = r_col * (1.0 - limbal) + c_out[0] * limbal
                g_col = g_col * (1.0 - limbal) + c_out[1] * limbal
                b_col = b_col * (1.0 - limbal) + c_out[2] * limbal

                alpha = base_alpha * (0.76 + 0.24 * fibers) * (1.0 - limbal + limbal * limbal_a)
                if r <= 0.33 or r >= 0.99:
                    alpha = 0.0

                gx = tx * tile + lx
                gy = ty * tile + ly
                k = (gy * SIZE + gx) * 4
                buf[k] = _to_byte(r_col)
                buf[k + 1] = _to_byte(g_col)
                buf[k + 2] = _to_byte(b_col)
                buf[k + 3] = _to_byte(alpha)
    return bytes(buf)


def verify_all(uvs):
    expected_files = [
        'blush_atlas.png',
        'contour_highlight.png',
        'eyebrow_atlas.png',
        'eye_makeup.png',
        'iris_atlas.png',
    ]
    total = 0
    for name in expected_files:
        path = os.path.join(OUT_DIR, name)
        if not os.path.isfile(path):
            print(f'FAIL  缺少美妆资产 {path}')
            return 1
        data = open(path, 'rb').read()
        total += len(data)
        w, h, rgba = decode_rgba_png(data)
        if (w, h) != (SIZE, SIZE):
            print(f'FAIL  {name} 尺寸应为 {SIZE}x{SIZE}，实际 {w}x{h}')
            return 1
        # 检查非空有效通道
        nonzero = sum(1 for i in range(0, len(rgba), 4) if rgba[i] > 10 or rgba[i + 1] > 10 or rgba[i + 2] > 10)
        if nonzero < 200:
            print(f'FAIL  {name} 有效像素过少：{nonzero}')
            return 1
        if name == 'contour_highlight.png':
            # 内唇中点 (13+14)/2 处 B 通道（唇带遮罩）必须严格为 0（防涂牙）。
            mu = 0.5 * (uvs[13][0] + uvs[14][0])
            mv = 0.5 * (uvs[13][1] + uvs[14][1])
            px, py = int(mu * SIZE), int(mv * SIZE)
            b_val = rgba[(py * SIZE + px) * 4 + 2]
            if b_val != 0:
                print(f'FAIL  contour_highlight.png 内唇口腔中心 B 通道非 0：{b_val}')
                return 1
        if name in ('eyebrow_atlas.png', 'eye_makeup.png'):
            half = SIZE // 2
            top_nz = sum(1 for i in range(0, half * SIZE * 4, 4) if rgba[i] > 10 or rgba[i + 1] > 10 or rgba[i + 2] > 10)
            bot_nz = sum(1 for i in range(half * SIZE * 4, len(rgba), 4) if rgba[i] > 10 or rgba[i + 1] > 10 or rgba[i + 2] > 10)
            if top_nz < 400 or bot_nz < 400:
                print(f'FAIL  {name} 上下分屏局部放大贴图有效像素不足：top={top_nz}, bot={bot_nz}')
                return 1
            for seam_y in (half - 1, half):
                seam_leak = any(
                    rgba[(seam_y * SIZE + sx) * 4 + ch] != 0
                    for sx in range(SIZE)
                    for ch in ((0, 1, 2) if name == 'eyebrow_atlas.png' else (0, 1, 2, 3))
                )
                if seam_leak:
                    print(f'FAIL  {name} 分屏接缝行 y={seam_y} 存在非零像素串边')
                    return 1
        if name == 'iris_atlas.png':
            # 4 个美瞳格中心瞳孔必须严格全透明 (A == 0)。
            for ty in (0, 1):
                for tx in (0, 1):
                    cx = tx * (SIZE // 2) + SIZE // 4
                    cy = ty * (SIZE // 2) + SIZE // 4
                    a_val = rgba[(cy * SIZE + cx) * 4 + 3]
                    if a_val != 0:
                        print(f'FAIL  iris_atlas.png 格 ({tx},{ty}) 瞳孔中心 alpha 非 0：{a_val}')
                        return 1
    if total > MAX_TOTAL_BYTES:
        print(f'FAIL  美妆贴图总体积超限：{total} > {MAX_TOTAL_BYTES}')
        return 1
    print(f'PASS  5 张 {SIZE}×{SIZE} 美妆贴图校验全部通过（总计 {total / 1024:.1f} KiB ≤ {MAX_TOTAL_BYTES // 1024} KiB，内唇/瞳孔安全透明）')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    uvs = load_canonical_uvs()
    if args.verify:
        return verify_all(uvs)

    os.makedirs(OUT_DIR, exist_ok=True)
    builders = {
        'blush_atlas.png': build_blush_atlas(uvs),
        'contour_highlight.png': build_contour_highlight_atlas(uvs),
        'eyebrow_atlas.png': build_eyebrow_atlas(uvs),
        'eye_makeup.png': build_eye_makeup_atlas(uvs),
        'iris_atlas.png': build_iris_atlas(),
    }
    for name, rgba in builders.items():
        png = encode_rgba_png(SIZE, SIZE, rgba)
        path = os.path.join(OUT_DIR, name)
        with open(path, 'wb') as f:
            f.write(png)
        print(f'OK    {name}: {len(png) / 1024:.1f} KiB')
    return verify_all(uvs)


if __name__ == '__main__':
    sys.exit(main())
