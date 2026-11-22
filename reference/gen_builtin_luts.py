#!/usr/bin/env python3
"""程序化生成并校验 Android 内置 10 套风格 64^3 3D LUT（512×512 RGB PNG）。

美白与去黄只在着色器（COMPOSITE_PRO）里做一次，LUT 只管氛围：内置配方不再提亮、去黄肤色，影调曲线在肤色上只生效
1 - skin_curve_keep，避免与着色器叠加后肤色发灰、背景发雾。

格式与主合成 FRAGMENT_PRO 的查表（LUT_SAMPLE，纹理单元 3）完全一致：
- 512×512，8×8 格，每格 64×64；
- 格内 x = 红 (0..63)、y = 绿 (0..63)，格序 row * 8 + col = 蓝 (0..63)；
- 图片第 0 行在顶部（与 GLUtils.texImage2D 上传位图一致）。

零外部依赖（仅用 Python 标准库），输出字节完全确定。

用法：
  python3 -I reference/gen_builtin_luts.py            # 生成全部 10 套 PNG
  python3 -I reference/gen_builtin_luts.py --verify   # 校验资产尺寸、平滑性、肤色保护、背景提亮与体积
"""
import argparse
import hashlib
import math
import os
import struct
import sys
import zlib

sys.dont_write_bytecode = True

SIZE = 512
GRID = 64
TILES = 8
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024

OUT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        'android',
        'live-beauty',
        'src',
        'main',
        'assets',
        'luts',
    )
)


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a + (b - a) * t


def luma709(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def skin_weight(r, g, b):
    """宽柔 YCbCr 肤色权重（较渲染门限更宽更平滑，避免 LUT 立方内部出现色带跳变）。"""
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    w = (
        smoothstep(0.24, 0.35, cb)
        * (1.0 - smoothstep(0.47, 0.58, cb))
        * smoothstep(0.49, 0.56, cr)
        * (1.0 - smoothstep(0.71, 0.82, cr))
    )
    # 极黑与极白不按肤色处理，保持黑位与白平衡纯净。
    y = luma709(r, g, b)
    return w * smoothstep(0.08, 0.22, y) * (1.0 - smoothstep(0.92, 1.0, y))


def tone_curve(v, lift_black, gamma, contrast, hi_roll):
    """摄影人像平滑影调曲线：黑位控制 + 伽马曝光 + 中灰 S 曲线 + 高光柔滚降。"""
    x = clamp(v, 0.0, 1.0)
    if gamma != 1.0 and x > 0.0:
        x = x ** gamma
    if contrast != 0.0:
        # 经典平滑 S 曲线，端点 0/1 不动。
        s = x * x * (3.0 - 2.0 * x)
        x = mix(x, s, contrast)
    if hi_roll > 0.0:
        # 柔和压缩最亮高光，防止打光脸颊死白过曝。
        x = x / (1.0 + hi_roll * x * x) * (1.0 + hi_roll)
    if lift_black != 0.0:
        # 轻抬或轻压暗部，只在阴影区平缓生效。
        x = x + lift_black * (1.0 - x) * (1.0 - x)
    return clamp(x, 0.0, 1.0)


def apply_recipe(r, g, b, cfg):
    """通用调色引擎：环境氛围分层，肤色只做轻微校色。"""
    sw = skin_weight(r, g, b)
    bg = 1.0 - sw

    # 1. 影调曲线；肤色上按 skin_curve_keep 保留原值（明暗与饱和度交给着色器，LUT 不再把脸压灰或抬雾）。
    lift_black = cfg.get('lift_black', 0.0)
    gamma = cfg.get('gamma', 1.0)
    contrast = cfg.get('contrast', 0.0)
    hi_roll = cfg.get('hi_roll', 0.0)
    keep = cfg.get('skin_curve_keep', 0.0) * sw
    r = mix(tone_curve(r, lift_black, gamma, contrast, hi_roll), r, keep)
    g = mix(tone_curve(g, lift_black, gamma, contrast, hi_roll), g, keep)
    b = mix(tone_curve(b, lift_black, gamma, contrast, hi_roll), b, keep)

    y = luma709(r, g, b)
    sh = (1.0 - y) * (1.0 - y) * smoothstep(0.0, 0.12, y)
    mid = 4.0 * y * (1.0 - y)
    hi = y * y * (1.0 - smoothstep(0.94, 1.0, y))

    # 2. 肤色校色。skin_lift / skin_deyellow 内置配方不再使用（美白与去黄由着色器做一次），保留给外部配方。
    skin_lift = cfg.get('skin_lift', 0.0)
    if skin_lift != 0.0:
        gain = 1.0 + skin_lift * sw * (1.0 - y * 0.45)
        r *= gain
        g *= gain
        b *= gain

    skin_deyellow = cfg.get('skin_deyellow', 0.0)
    if skin_deyellow != 0.0:
        yellow = clamp(0.5 * (r + g) - b, 0.0, 0.35)
        k = skin_deyellow * sw * yellow * 2.0
        r += 0.010 * k
        g -= 0.018 * k
        b += 0.072 * k

    skin_shift = cfg.get('skin_shift', (0.0, 0.0, 0.0))
    if skin_shift != (0.0, 0.0, 0.0):
        k = sw * mix(0.45, 1.0, mid)
        r += skin_shift[0] * k
        g += skin_shift[1] * k
        b += skin_shift[2] * k

    # 3. 氛围分层调色（阴影/中间调/高光 Split Toning，肤色区自动减权防止脸泛绿泛蓝）
    skin_tone_damp = cfg.get('skin_tone_damp', 0.32)
    tone_w = mix(1.0, skin_tone_damp, sw)
    sh_tint = cfg.get('shadow_tint', (0.0, 0.0, 0.0))
    mid_tint = cfg.get('mid_tint', (0.0, 0.0, 0.0))
    hi_tint = cfg.get('highlight_tint', (0.0, 0.0, 0.0))
    r += (sh_tint[0] * sh + mid_tint[0] * mid + hi_tint[0] * hi) * tone_w
    g += (sh_tint[1] * sh + mid_tint[1] * mid + hi_tint[1] * hi) * tone_w
    b += (sh_tint[2] * sh + mid_tint[2] * mid + hi_tint[2] * hi) * tone_w

    # 4. 肤色与背景分域饱和度（压背景杂色、保肤色干净与唇颊气色）
    r = clamp(r, 0.0, 1.0)
    g = clamp(g, 0.0, 1.0)
    b = clamp(b, 0.0, 1.0)
    y2 = luma709(r, g, b)
    sat_skin = cfg.get('sat_skin', 1.0)
    sat_bg = cfg.get('sat_bg', 1.0)
    sat = mix(sat_bg, sat_skin, sw)
    # 唇红/暖粉高红通道额外微提鲜活度（只给风格需要的预设）
    lip_boost = cfg.get('lip_boost', 0.0)
    if lip_boost != 0.0:
        redness = smoothstep(0.10, 0.28, r - max(g, b)) * (1.0 - sw * 0.5)
        sat += lip_boost * redness
    r = mix(y2, r, sat)
    g = mix(y2, g, sat)
    b = mix(y2, b, sat)

    # 5. 柔光微通透（可选轻度 Soft-Light 立体感）
    soft_glow = cfg.get('soft_glow', 0.0)
    if soft_glow > 0.0:
        r = clamp(r, 0.0, 1.0)
        g = clamp(g, 0.0, 1.0)
        b = clamp(b, 0.0, 1.0)
        ref = clamp(y2 * 0.90 + 0.10, 0.0, 1.0)
        sr = r * (r + 2.0 * ref * (1.0 - r))
        sg = g * (g + 2.0 * ref * (1.0 - g))
        sb = b * (b + 2.0 * ref * (1.0 - b))
        w = soft_glow * (0.45 + 0.55 * sw)
        r = mix(r, sr, w)
        g = mix(g, sg, w)
        b = mix(b, sb, w)

    # 6. 极浅氛围暖冷平衡（bg 权重大于肤色）
    bg_cool = cfg.get('bg_cool', 0.0)
    if bg_cool != 0.0:
        k = bg_cool * bg * mid
        r -= 0.025 * k
        b += 0.032 * k

    return clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)


# 10 套内置风格 3D LUT（顺序与 Dart BroadcastBuiltinLut 枚举一一对应）。约束（verify_all 守护）：不提亮、不去黄肤色，
# 肤色蓝移 <= 0.004、肤色饱和度 >= 1.0、黑位抬升 <= 0.012，影调曲线在肤色上只生效 40%（skin_curve_keep = 0.6）。
RECIPES = {
    'natural_studio': dict(
        gamma=1.0,
        contrast=0.06,
        hi_roll=0.05,
        lift_black=0.0,
        skin_shift=(0.008, 0.002, 0.004),
        shadow_tint=(0.0, 0.0, 0.0),
        highlight_tint=(0.006, 0.004, 0.002),
        sat_skin=1.02,
        sat_bg=1.0,
        soft_glow=0.04,
        skin_curve_keep=0.6,
    ),
    'clear_white': dict(
        gamma=0.93,
        contrast=0.08,
        hi_roll=0.08,
        lift_black=0.012,
        skin_shift=(0.012, 0.002, 0.004),
        shadow_tint=(-0.014, 0.002, 0.024),
        highlight_tint=(0.004, 0.006, 0.014),
        sat_skin=1.0,
        sat_bg=0.88,
        soft_glow=0.14,
        bg_cool=0.5,
        skin_curve_keep=0.6,
    ),
    'peach_milk': dict(
        gamma=0.94,
        contrast=-0.04,
        hi_roll=0.14,
        lift_black=0.012,
        skin_shift=(0.026, 0.006, 0.004),
        shadow_tint=(0.010, 0.004, 0.016),
        highlight_tint=(0.022, 0.010, 0.008),
        sat_skin=1.00,
        sat_bg=0.90,
        soft_glow=0.16,
        skin_curve_keep=0.6,
    ),
    'pure_pink': dict(
        gamma=0.94,
        contrast=0.02,
        hi_roll=0.12,
        lift_black=0.012,
        skin_shift=(0.028, -0.006, 0.004),
        shadow_tint=(0.012, -0.006, 0.028),
        highlight_tint=(0.028, 0.004, 0.020),
        sat_skin=1.02,
        sat_bg=0.88,
        lip_boost=0.14,
        soft_glow=0.15,
        skin_curve_keep=0.6,
    ),
    'korean_glow': dict(
        gamma=0.93,
        contrast=0.10,
        hi_roll=0.10,
        lift_black=0.010,
        skin_shift=(0.016, 0.003, 0.004),
        shadow_tint=(-0.018, 0.006, 0.026),
        highlight_tint=(0.010, 0.008, 0.012),
        sat_skin=1.0,
        sat_bg=0.92,
        lip_boost=0.12,
        soft_glow=0.20,
        bg_cool=0.45,
        skin_curve_keep=0.6,
    ),
    'japanese_airy': dict(
        gamma=0.92,
        contrast=-0.06,
        hi_roll=0.15,
        lift_black=0.012,
        skin_shift=(0.008, 0.004, 0.004),
        shadow_tint=(-0.016, 0.010, 0.030),
        highlight_tint=(0.014, 0.012, 0.006),
        sat_skin=1.0,
        sat_bg=0.78,
        soft_glow=0.12,
        bg_cool=0.7,
        skin_curve_keep=0.6,
    ),
    'soft_film': dict(
        gamma=0.99,
        contrast=0.14,
        hi_roll=0.16,
        lift_black=0.012,
        skin_shift=(0.014, 0.004, -0.004),
        shadow_tint=(-0.014, 0.014, 0.018),
        highlight_tint=(0.026, 0.014, -0.010),
        skin_tone_damp=0.25,
        sat_skin=1.0,
        sat_bg=0.84,
        soft_glow=0.08,
        skin_curve_keep=0.6,
    ),
    'night_vibe': dict(
        gamma=0.96,
        contrast=0.16,
        hi_roll=0.10,
        lift_black=-0.012,
        skin_shift=(0.022, 0.006, 0.004),
        shadow_tint=(-0.012, -0.004, 0.034),
        highlight_tint=(0.024, 0.010, 0.008),
        skin_tone_damp=0.22,
        sat_skin=1.04,
        sat_bg=0.92,
        lip_boost=0.16,
        soft_glow=0.14,
        skin_curve_keep=0.6,
    ),
    'champagne_glam': dict(
        gamma=0.95,
        contrast=0.15,
        hi_roll=0.10,
        lift_black=0.006,
        skin_shift=(0.022, 0.008, 0.002),
        shadow_tint=(0.006, 0.002, 0.014),
        highlight_tint=(0.030, 0.018, -0.004),
        sat_skin=1.04,
        sat_bg=0.94,
        lip_boost=0.18,
        soft_glow=0.18,
        skin_curve_keep=0.6,
    ),
    'sweet_berry': dict(
        gamma=0.94,
        contrast=0.11,
        hi_roll=0.08,
        lift_black=0.010,
        skin_shift=(0.024, -0.004, 0.004),
        shadow_tint=(0.008, -0.004, 0.020),
        highlight_tint=(0.020, 0.006, 0.012),
        sat_skin=1.06,
        sat_bg=1.06,
        lip_boost=0.20,
        soft_glow=0.12,
        skin_curve_keep=0.6,
    ),
}


def png_chunk(kind, data):
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', crc)


def encode_rgb_png(width, height, rows_rgb):
    """把每行 bytes(width*3) 列表编码为标准 8-bit RGB PNG（使用 Sub 过滤提高连续 LUT 压缩率）。"""
    raw = bytearray()
    for row in rows_rgb:
        raw.append(1)  # PNG Filter Type 1: Sub (非常适合平滑渐变 LUT)
        prev0 = prev1 = prev2 = 0
        for i in range(0, len(row), 3):
            r, g, b = row[i], row[i + 1], row[i + 2]
            raw.append((r - prev0) & 0xFF)
            raw.append((g - prev1) & 0xFF)
            raw.append((b - prev2) & 0xFF)
            prev0, prev1, prev2 = r, g, b
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    return (
        b'\x89PNG\r\n\x1a\n'
        + png_chunk(b'IHDR', ihdr)
        + png_chunk(b'IDAT', idat)
        + png_chunk(b'IEND', b'')
    )


def decode_rgb_png(blob):
    """轻量 PNG 解码校验器（支持 color_type=2 8-bit、filter 0/1）。"""
    if not blob.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('非法 PNG 魔数')
    pos = 8
    width = height = bit_depth = color_type = None
    idat_parts = []
    while pos < len(blob):
        length = struct.unpack('>I', blob[pos:pos + 4])[0]
        kind = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + length]
        crc = struct.unpack('>I', blob[pos + 8 + length:pos + 12 + length])[0]
        if (zlib.crc32(kind + data) & 0xFFFFFFFF) != crc:
            raise ValueError('PNG CRC 校验失败: %s' % kind)
        pos += 12 + length
        if kind == b'IHDR':
            width, height, bit_depth, color_type = struct.unpack('>IIBB', data[:10])
        elif kind == b'IDAT':
            idat_parts.append(data)
        elif kind == b'IEND':
            break
    if (width, height, bit_depth, color_type) != (SIZE, SIZE, 8, 2):
        raise ValueError('尺寸或格式不符: %s' % ((width, height, bit_depth, color_type),))
    raw = zlib.decompress(b''.join(idat_parts))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError('解压像素字节长度异常')
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        ftype = raw[start]
        src = raw[start + 1:start + 1 + stride]
        if ftype == 0:
            rows.append(bytes(src))
        elif ftype == 1:
            out = bytearray(stride)
            for i in range(stride):
                left = out[i - 3] if i >= 3 else 0
                out[i] = (src[i] + left) & 0xFF
            rows.append(bytes(out))
        else:
            raise ValueError('不支持的 PNG filter: %d' % ftype)
    return rows


def build_lut_rows(cfg):
    inv = 1.0 / (GRID - 1)
    # 预先按蓝格行计算，保持 8x8 格布局：row = b // 8, col = b % 8。
    rows = [bytearray(SIZE * 3) for _ in range(SIZE)]
    for b_idx in range(GRID):
        tile_row = b_idx // TILES
        tile_col = b_idx % TILES
        b_in = b_idx * inv
        y_base = tile_row * GRID
        x_base = tile_col * GRID
        for g_idx in range(GRID):
            g_in = g_idx * inv
            row_buf = rows[y_base + g_idx]
            offset = x_base * 3
            for r_idx in range(GRID):
                r_out, g_out, b_out = apply_recipe(r_idx * inv, g_in, b_in, cfg)
                row_buf[offset] = int(r_out * 255.0 + 0.5)
                row_buf[offset + 1] = int(g_out * 255.0 + 0.5)
                row_buf[offset + 2] = int(b_out * 255.0 + 0.5)
                offset += 3
    return rows


def lookup_rgb(rows, r, g, b):
    """在解码后的 512×512 LUT 行中做最近格点采样，用于肤色探针校验。"""
    ri = int(round(clamp(r) * 63))
    gi = int(round(clamp(g) * 63))
    bi = int(round(clamp(b) * 63))
    y = (bi // TILES) * GRID + gi
    x = (bi % TILES) * GRID + ri
    off = x * 3
    row = rows[y]
    return row[off] / 255.0, row[off + 1] / 255.0, row[off + 2] / 255.0


def sample_lut(rows, r, g, b):
    """与 FRAGMENT_PRO 相同的查表：格内红绿双线性、相邻蓝格线性，返回 0..1 浮点。"""
    def node(ri, gi, bi):
        off = ((bi % TILES) * GRID + ri) * 3
        row = rows[(bi // TILES) * GRID + gi]
        return row[off] / 255.0, row[off + 1] / 255.0, row[off + 2] / 255.0

    def tile(bi, rf, gf):
        r0, g0 = int(math.floor(rf)), int(math.floor(gf))
        r1, g1 = min(r0 + 1, GRID - 1), min(g0 + 1, GRID - 1)
        tr, tg = rf - r0, gf - g0
        c00, c10, c01, c11 = node(r0, g0, bi), node(r1, g0, bi), node(r0, g1, bi), node(r1, g1, bi)
        return tuple(mix(mix(c00[i], c10[i], tr), mix(c01[i], c11[i], tr), tg) for i in range(3))

    rf, gf, level = clamp(r) * (GRID - 1), clamp(g) * (GRID - 1), clamp(b) * (GRID - 1)
    low, high = tile(int(math.floor(level)), rf, gf), tile(int(math.ceil(level)), rf, gf)
    t = level - math.floor(level)
    return tuple(mix(low[i], high[i], t) for i in range(3))


def saturation(rgb):
    hi, lo = max(rgb), min(rgb)
    return (hi - lo) / hi if hi > 0.0 else 0.0


# 5 个典型人像肤色探针（亮白亚洲、暖黄室内、自然小麦、暖棕深肤、红润苹果肌）
SKIN_PROBES = (
    ('fair_asian', (0.86, 0.75, 0.67)),
    ('warm_indoor', (0.78, 0.63, 0.48)),
    ('medium_tan', (0.68, 0.52, 0.39)),
    ('deep_warm', (0.48, 0.34, 0.25)),
    ('rosy_cheek', (0.82, 0.64, 0.58)),
)
# 中性背景探针（与 GLSL 门禁「自然」档的背景一致）：暗部与灰墙，单位 /255。
NEUTRAL_DARK = (31, 31, 36)
NEUTRAL_GREY = (89, 84, 76)
# 肤色：饱和度比下限、亮度增量上限（美白交给着色器，LUT 只能轻微改肤色）。
MIN_SKIN_SAT_RATIO = 0.85
MAX_SKIN_DY = 0.06
# 背景：natural_studio 对中性暗部与灰墙各通道最多提亮 3/255，所有 LUT 的暗部最多提亮 24/255（氛围类风格的设计值）。
MAX_NATURAL_LIFT = 3
MAX_BLACK_LIFT = 24


def lift255(rows, rgb8):
    """中性探针经 LUT 后各通道相对输入的最大提亮（/255，四舍五入）。"""
    out = sample_lut(rows, *(v / 255.0 for v in rgb8))
    return int(round(max(out[i] * 255.0 - rgb8[i] for i in range(3))))


def verify_all():
    if not os.path.isdir(OUT_DIR):
        raise SystemExit('缺少 LUT 目录：%s' % OUT_DIR)
    total_bytes = 0
    for name in RECIPES:
        path = os.path.join(OUT_DIR, '%s.png' % name)
        if not os.path.isfile(path):
            raise SystemExit('缺少内置 LUT：%s' % path)
        blob = open(path, 'rb').read()
        size = len(blob)
        total_bytes += size
        if size <= 0 or size > MAX_FILE_BYTES:
            raise SystemExit('%s 体积超限：%d bytes' % (name, size))
        rows = decode_rgb_png(blob)
        # 1. 检查单调平滑度（沿红轴相邻格点突变不超过 18/255，杜绝色带断层）
        max_jump = 0
        for b_idx in range(GRID):
            y_base = (b_idx // TILES) * GRID
            x_base = (b_idx % TILES) * GRID
            for g_idx in range(0, GRID, 4):
                row = rows[y_base + g_idx]
                for r_idx in range(GRID - 1):
                    p0 = (x_base + r_idx) * 3
                    p1 = p0 + 3
                    jump = max(abs(row[p1] - row[p0]), abs(row[p1 + 1] - row[p0 + 1]), abs(row[p1 + 2] - row[p0 + 2]))
                    if jump > max_jump:
                        max_jump = jump
        if max_jump > 18:
            raise SystemExit('%s 存在色带突变：相邻步进差 %d/255' % (name, max_jump))
        # 2. 检查肤色保护：绝不偏绿（g <= r），不死白（平均亮度不全部爆 0.99）
        for probe_name, rgb in SKIN_PROBES:
            ro, go, bo = lookup_rgb(rows, *rgb)
            if go > ro + 0.01:
                raise SystemExit('%s 在肤色探针 %s 发生偏绿：in=%s out=(%.3f,%.3f,%.3f)' % (name, probe_name, rgb, ro, go, bo))
            if ro > 0.995 and go > 0.995 and bo > 0.995:
                raise SystemExit('%s 在肤色探针 %s 高光死白过曝' % (name, probe_name))
        # 3. 美白与去黄只在着色器做一次：配方不得再提亮、去黄肤色。
        cfg = RECIPES[name]
        if cfg.get('skin_lift', 0.0) or cfg.get('skin_deyellow', 0.0):
            raise SystemExit('%s 配方仍在提亮或去黄肤色（skin_lift / skin_deyellow 必须为 0）' % name)
        # 4. 肤色饱和度不掉、亮度不过提（按着色器同样的三线性插值查表）。
        min_ratio, max_dy = 9.0, -9.0
        for probe_name, rgb in SKIN_PROBES:
            out = sample_lut(rows, *rgb)
            ratio = saturation(out) / saturation(rgb)
            dy = luma709(*out) - luma709(*rgb)
            min_ratio, max_dy = min(min_ratio, ratio), max(max_dy, dy)
            if ratio < MIN_SKIN_SAT_RATIO:
                raise SystemExit('%s 在肤色探针 %s 饱和度比 %.3f < %.2f（肤色发灰）' % (name, probe_name, ratio, MIN_SKIN_SAT_RATIO))
            if dy > MAX_SKIN_DY:
                raise SystemExit('%s 在肤色探针 %s 亮度 %+.3f > +%.2f（与着色器美白叠加过亮）' % (name, probe_name, dy, MAX_SKIN_DY))
        # 5. 背景不起雾：暗部抬升有上限，natural_studio 对中性暗部与灰墙几乎不动。
        black = lift255(rows, NEUTRAL_DARK)
        if black > MAX_BLACK_LIFT:
            raise SystemExit('%s 中性暗部 %s 提亮 %+d/255 > %d' % (name, NEUTRAL_DARK, black, MAX_BLACK_LIFT))
        if name == 'natural_studio':
            for probe in (NEUTRAL_DARK, NEUTRAL_GREY):
                lift = lift255(rows, probe)
                if lift > MAX_NATURAL_LIFT:
                    raise SystemExit('%s 中性背景 %s 提亮 %+d/255 > %d' % (name, probe, lift, MAX_NATURAL_LIFT))
        digest = hashlib.sha256(blob).hexdigest()[:12]
        print('PASS  %-16s 512x512 RGB PNG  %6.1f KiB  maxJump=%2d  skinSat>=%.2f  skinDY<=%+.3f  black%+3d  sha256=%s' % (
            name, size / 1024.0, max_jump, min_ratio, max_dy, black, digest))
    if total_bytes > MAX_TOTAL_BYTES:
        raise SystemExit('内置 LUT 总大小超限：%.1f KiB' % (total_bytes / 1024.0))
    print('全部 %d 套内置 3D LUT 校验通过，总计 %.1f KiB。' % (len(RECIPES), total_bytes / 1024.0))


def generate_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, cfg in RECIPES.items():
        rows = build_lut_rows(cfg)
        blob = encode_rgb_png(SIZE, SIZE, rows)
        path = os.path.join(OUT_DIR, '%s.png' % name)
        with open(path, 'wb') as handle:
            handle.write(blob)
        print('GEN   %-16s -> %s (%.1f KiB)' % (name, os.path.relpath(path), len(blob) / 1024.0))
    verify_all()


def main(argv):
    parser = argparse.ArgumentParser(description='生成/校验 Android 内置 64^3 3D LUT 资产')
    parser.add_argument('--verify', action='store_true', help='仅校验现有 PNG 资产，不覆盖生成')
    args = parser.parse_args(argv[1:])
    if args.verify:
        verify_all()
    else:
        generate_all()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
