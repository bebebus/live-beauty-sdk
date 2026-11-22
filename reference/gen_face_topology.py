#!/usr/bin/env python3
"""从随包 Apache-2.0 MediaPipe face_landmarker.task 中提取规范 468 点拓扑、UV、3D 坐标与 Procrustes 稳定点，
生成并校验 FaceMeshTopology.kt。

仅依赖 Python 标准库，输出完全确定性。
用法：
  python3 -I gen_face_topology.py          # 生成 FaceMeshTopology.kt 并校验
  python3 -I gen_face_topology.py --verify # 仅校验现有 FaceMeshTopology.kt 与模型完全一致
"""
import argparse
import collections
import math
import os
import struct
import sys
import zipfile

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'ios', 'Resources', 'face_landmarker.task')
OUT_KT = os.path.join(
    ROOT, 'android', 'live-beauty', 'src', 'main', 'kotlin', 'io', 'bebebus', 'livebeauty', 'FaceMeshTopology.kt'
)

# MediaPipe 规范脸部轮廓 36 点（从额头正中 10 起，在 GL 纹理坐标 y 向上空间内严格顺时针）。
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109,
]
OUTER_STOPS = [0, 5, 9, 14, 18, 23, 27, 32]
RIGHT_EYE_RING = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246}
LEFT_EYE_RING = {263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}
OUTER_LIP_POLY = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_RING = {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95}
# 规范 3D 坐标定点化比例（规范单位约厘米，范围 ±10，int16 足够）；FaceMeshFilter 与 FaceMeshDeformer 解码使用。
XYZ_SCALE = 3000
# Procrustes 稳定点必须包含的锚点（鼻尖、鼻根、额头、双眼内外眼角）。
REQUIRED_BASIS = {4, 6, 10, 33, 263, 133, 362}


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
        else:
            raise ValueError(f'Unknown protobuf wire type {wtype}')
    return fields


def _area2(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _point_in_poly(px, py, poly_pts):
    inside = False
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def extract_canonical():
    with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
        pb = zf.read('geometry_pipeline_metadata_landmarks.binarypb')
    top = _parse_proto(pb)
    mesh = _parse_proto(top[1][0])
    vb = mesh[3]
    ib = mesh[4]
    assert len(vb) == 468 * 5, f'Expected 2340 floats, got {len(vb)}'
    assert len(ib) == 898 * 3, f'Expected 2694 indices, got {len(ib)}'

    uvs = [(vb[i * 5 + 3], vb[i * 5 + 4]) for i in range(468)]
    uv_gl = [(u, 1.0 - v) for u, v in uvs]
    # 每顶点 5 个 float 的前 3 个为规范 3D 坐标（x 向右、y 向上、z 朝向相机）。
    xyz = [(vb[i * 5], vb[i * 5 + 1], vb[i * 5 + 2]) for i in range(468)]
    assert len(xyz) == 468
    assert all(abs(c) < 32767 / XYZ_SCALE for p in xyz for c in p), 'Canonical xyz out of int16 range'
    # 顶层字段 2：Procrustes 加权稳定点（字段 1 landmark_id，字段 2 weight），MediaPipe 几何管线求头部姿态时用。
    basis = []
    for entry in top.get(2, []):
        f = _parse_proto(entry)
        basis.append((f[1][0], f[2][0]))
    assert len(basis) == 33, f'Expected 33 procrustes points, got {len(basis)}'
    assert REQUIRED_BASIS <= {i for i, _ in basis}, 'Procrustes basis misses required anchors'
    assert all(0 <= i < 468 and w > 0 for i, w in basis)
    tris = [(ib[i * 3], ib[i * 3 + 1], ib[i * 3 + 2]) for i in range(898)]

    # 校验全部 898 个规范三角形在 GL 纹理空间（y 向上）均为正面积 CCW，且唯一外边界即 36 点 FACE_OVAL。
    for t in tris:
        assert _area2(uv_gl[t[0]], uv_gl[t[1]], uv_gl[t[2]]) > 0, f'Non-CCW canonical tri {t}'
    ec = collections.Counter()
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            ec[(min(p, q), max(p, q))] += 1
    boundary = {e for e, cnt in ec.items() if cnt == 1}
    expected_oval = {(min(FACE_OVAL[k], FACE_OVAL[(k + 1) % 36]), max(FACE_OVAL[k], FACE_OVAL[(k + 1) % 36])) for k in range(36)}
    assert boundary == expected_oval, f'Boundary mismatch: {len(boundary)}'

    # 提取眼裂内部子网与严格镂空内唇的唇带子网（供 M4 美妆物理防穿帮复用）。
    right_eye_tris = [t for t in tris if set(t) <= RIGHT_EYE_RING]
    left_eye_tris = [t for t in tris if set(t) <= LEFT_EYE_RING]
    inner_mouth_tris = [t for t in tris if set(t) <= INNER_LIP_RING]
    outer_poly_pts = [uvs[i] for i in OUTER_LIP_POLY]
    lip_band_tris = []
    for t in tris:
        if set(t) <= INNER_LIP_RING:
            continue
        cx = (uvs[t[0]][0] + uvs[t[1]][0] + uvs[t[2]][0]) / 3.0
        cy = (uvs[t[0]][1] + uvs[t[1]][1] + uvs[t[2]][1]) / 3.0
        if _point_in_poly(cx, cy, outer_poly_pts):
            lip_band_tris.append(t)

    assert len(right_eye_tris) == 14
    assert len(left_eye_tris) == 14
    assert len(inner_mouth_tris) == 18
    assert len(lip_band_tris) == 120

    # 验证 1086 三角形（898 脸部 + 72 内过渡环 + 72 外缓冲环 + 44 屏幕外框）完全水密无裂缝。
    cx = sum(uv_gl[i][0] for i in FACE_OVAL) / 36.0
    cy = sum(uv_gl[i][1] for i in FACE_OVAL) / 36.0
    pts = list(uv_gl)
    for idx in FACE_OVAL:
        px, py = uv_gl[idx]
        dx, dy = px - cx, py - cy
        ln = math.hypot(dx, dy)
        pts.append((px + dx / ln * 0.11, py + dy / ln * 0.11))
    for idx in FACE_OVAL:
        px, py = uv_gl[idx]
        dx, dy = px - cx, py - cy
        ln = math.hypot(dx, dy)
        pts.append((px + dx / ln * 0.25, py + dy / ln * 0.25))
    for dx, dy in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
        pts.append((cx + dx * 2.6, cy + dy * 2.6))

    buf_tris = []
    for k in range(36):
        k2 = (k + 1) % 36
        o0, o1 = FACE_OVAL[k], FACE_OVAL[k2]
        m0, m1 = 468 + k, 468 + k2
        t1, t2 = (o0, m1, m0), (o0, o1, m1)
        assert _area2(pts[t1[0]], pts[t1[1]], pts[t1[2]]) > 0
        assert _area2(pts[t2[0]], pts[t2[1]], pts[t2[2]]) > 0
        buf_tris.extend((t1, t2))
    for k in range(36):
        k2 = (k + 1) % 36
        m0, m1 = 468 + k, 468 + k2
        b0, b1 = 504 + k, 504 + k2
        t1, t2 = (m0, b1, b0), (m0, m1, b1)
        assert _area2(pts[t1[0]], pts[t1[1]], pts[t1[2]]) > 0
        assert _area2(pts[t2[0]], pts[t2[1]], pts[t2[2]]) > 0
        buf_tris.extend((t1, t2))

    outer_tris = []
    for j in range(8):
        j2 = (j + 1) % 8
        s0 = OUTER_STOPS[j]
        s1 = OUTER_STOPS[j2] if j < 7 else 36
        f0, f1 = 540 + j, 540 + j2
        for k in range(s0, s1):
            b0 = 504 + (k % 36)
            b1 = 504 + ((k + 1) % 36)
            t = (b0, b1, f0)
            assert _area2(pts[t[0]], pts[t[1]], pts[t[2]]) > 0
            outer_tris.append(t)
        bs1 = 504 + (s1 % 36)
        t = (bs1, f1, f0)
        assert _area2(pts[t[0]], pts[t[1]], pts[t[2]]) > 0
        outer_tris.append(t)

    all_tris = tris + buf_tris + outer_tris
    assert len(all_tris) == 1086
    ec2 = collections.Counter()
    for a, b, c in all_tris:
        for p, q in ((a, b), (b, c), (c, a)):
            ec2[(min(p, q), max(p, q))] += 1
    assert all(cnt <= 2 for cnt in ec2.values()), 'Non-manifold edge found'
    outer_edges = {(min(540 + j, 540 + (j + 1) % 8), max(540 + j, 540 + (j + 1) % 8)) for j in range(8)}
    assert {e for e, cnt in ec2.items() if cnt == 1} == outer_edges

    return uvs, tris, right_eye_tris, left_eye_tris, lip_band_tris, xyz, basis


def _encode_shorts_b64_like(values):
    """每项 0..511 编成 2 个可打印 ASCII 字符（32 进制偏移 48..79），避免 JVM 64KB <clinit> 字节码上限。"""
    out = []
    for v in values:
        assert 0 <= v < 1024
        out.append(chr(48 + (v >> 5)))
        out.append(chr(48 + (v & 31)))
    return ''.join(out)


def _encode_uvs_hex(uvs):
    """468 对 (u, v) 量化到 0..65535（误差 < 1e-5），编成紧凑 16 进制串。"""
    parts = []
    for u, v in uvs:
        qu = max(0, min(65535, round(u * 65535.0)))
        qv = max(0, min(65535, round(v * 65535.0)))
        parts.append(f'{qu:04x}{qv:04x}')
    return ''.join(parts)


def _encode_xyz_hex(xyz):
    """468 × 3 个规范坐标按 1/XYZ_SCALE 定点化为 int16（误差 < 2e-4 规范单位），以 16 位补码编成 4 位 16 进制。"""
    parts = []
    for p in xyz:
        for c in p:
            q = round(c * XYZ_SCALE)
            assert -32768 <= q <= 32767
            parts.append(f'{q & 0xFFFF:04x}')
    return ''.join(parts)


def _chunk_kotlin_string(s, width=108):
    chunks = [s[i:i + width] for i in range(0, len(s), width)]
    return ' +\n        '.join(f'"{c}"' for c in chunks)


def render_kotlin(uvs, tris, right_eye_tris, left_eye_tris, lip_band_tris, xyz, basis):
    flat_canonical = [idx for t in tris for idx in t]
    flat_re = [idx for t in right_eye_tris for idx in t]
    flat_le = [idx for t in left_eye_tris for idx in t]
    flat_lip = [idx for t in lip_band_tris for idx in t]

    tri_str = _encode_shorts_b64_like(flat_canonical)
    uv_str = _encode_uvs_hex(uvs)
    xyz_str = _encode_xyz_hex(xyz)
    basis_idx = [i for i, _ in basis]
    # float32 原值按 9 位有效数字输出，Kotlin 解析回同一个 float。
    basis_w = [f'{w:.9g}f' for _, w in basis]

    def fmt_short_list(vals, per_line=18):
        lines = []
        for i in range(0, len(vals), per_line):
            lines.append('        ' + ', '.join(str(x) for x in vals[i:i + per_line]) + ',')
        return '\n'.join(lines)

    return f'''package io.bebebus.livebeauty

/**
 * MediaPipe 468 点规范人脸网格拓扑、零位移缓冲环与外框锚点缝合索引（仅 Android，纯 Kotlin 无 GL 依赖）。
 *
 * 数据源自随包 Apache-2.0 MediaPipe `face_landmarker.task` 内置规范模型，由
 * `reference/gen_face_topology.py` 确定性生成并验证：
 * - 顶点 `0..467`：468 个规范人脸关键点（含眼裂与内唇封闭三角化，共 898 个全正绕序三角形）；
 * - 顶点 `468..503`：36 个内层 C1 柔化过渡环顶点（与 [FACE_OVAL] 缝合为 72 个内过渡三角形，消除下颌线硬折痕）；
 * - 顶点 `504..539`：36 个零位移外轮廓缓冲环顶点（与内过渡环缝合为 72 个外缓冲三角形）；
 * - 顶点 `540..547`：8 个零位移屏幕外锚点（与外缓冲环缝合为 44 个恒等背景三角形，保证背景墙面/门框直线零弯曲）。
 * 总计 548 个顶点、1086 个水密无裂缝三角形。
 *
 * 另附规范 3D 坐标 [CANONICAL_XYZ] 与 MediaPipe 几何管线的 33 个 Procrustes 加权稳定点
 * （[PROCRUSTES_INDICES] / [PROCRUSTES_WEIGHTS]），供 [FaceMeshFilter] 拟合弱透视姿态、[FaceMeshDeformer] 算三角形可见度。
 */
internal object FaceMeshTopology {{
    const val FACE_VERTEX_COUNT = 468
    const val OVAL_VERTEX_COUNT = 36
    const val OUTER_VERTEX_COUNT = 8
    const val MID_BUFFER_OFFSET = FACE_VERTEX_COUNT
    const val BUFFER_OFFSET = MID_BUFFER_OFFSET + OVAL_VERTEX_COUNT
    const val OUTER_OFFSET = BUFFER_OFFSET + OVAL_VERTEX_COUNT
    const val TOTAL_VERTEX_COUNT = OUTER_OFFSET + OUTER_VERTEX_COUNT

    const val CANONICAL_TRIANGLE_COUNT = 898
    const val INNER_BUFFER_TRIANGLE_COUNT = 72
    const val BUFFER_TRIANGLE_COUNT = 144
    const val DEFORMABLE_TRIANGLE_COUNT = CANONICAL_TRIANGLE_COUNT + BUFFER_TRIANGLE_COUNT
    const val OUTER_TRIANGLE_COUNT = 44
    const val TOTAL_TRIANGLE_COUNT = DEFORMABLE_TRIANGLE_COUNT + OUTER_TRIANGLE_COUNT

    /** 36 点脸部外轮廓（从额头正中 10 起，在 GL 纹理坐标 y 向上空间严格顺时针）。 */
    val FACE_OVAL: IntArray = intArrayOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109,
    )

    /** 缓冲环 36 点与 8 个外框方向锚点的扇形缝合分段起点。 */
    val OUTER_ANCHOR_STOPS: IntArray = intArrayOf(0, 5, 9, 14, 18, 23, 27, 32)

    /** 8 个外框方向单位向量（在面轴 (u, v) 坐标系下顺时针排布，与 [FACE_OVAL] 角度一一对齐）。 */
    val OUTER_DIRECTIONS: FloatArray = floatArrayOf(
        0f, 1f,
        1f, 1f,
        1f, 0f,
        1f, -1f,
        0f, -1f,
        -1f, -1f,
        -1f, 0f,
        -1f, 1f,
    )

    /** 右眼（画面左侧）14 个眼裂内部三角形索引（M3 完整眼部网格 + M4 美瞳物理眼皮裁切）。 */
    val RIGHT_EYE_INDICES: ShortArray = shortArrayOf(
{fmt_short_list(flat_re)}
    )

    /** 左眼（画面右侧）14 个眼裂内部三角形索引（M3 完整眼部网格 + M4 美瞳物理眼皮裁切）。 */
    val LEFT_EYE_INDICES: ShortArray = shortArrayOf(
{fmt_short_list(flat_le)}
    )

    /** 120 个纯唇带三角形索引（严格位于外唇环与内唇环之间，口腔与牙齿零三角形，M4 物理防涂牙）。 */
    val LIP_BAND_INDICES: ShortArray = shortArrayOf(
{fmt_short_list(flat_lip)}
    )

    /** 898 个 MediaPipe 规范人脸三角形索引（2694 个 short）。 */
    val CANONICAL_INDICES: ShortArray = decodeIndices(CANONICAL_TRIS_PACKED)

    /** 468 个关键点的规范模板 UV（936 个 float，u 左→右、v 上→下，[0, 1]）。 */
    val CANONICAL_UV: FloatArray = decodeUvs(CANONICAL_UV_HEX)

    /** 完整 1086 个水密屏幕三角形索引（898 人脸 + 144 双环 C1 缓冲过渡 + 44 外框恒等背景 = 3258 个 short）。 */
    val FULL_INDICES: ShortArray = buildFullIndices()

    /**
     * 468 个关键点的规范 3D 坐标（1404 个 float，x 向右、y 向上、z 朝向相机，单位约厘米；定点化误差 < 2e-4）。
     * 规范脸宽 |p454 − p234| ≈ 15.33。
     */
    val CANONICAL_XYZ: FloatArray = decodeXyz(CANONICAL_XYZ_HEX)

    /** MediaPipe 几何管线的 33 个 Procrustes 稳定点（原样，含随下颌运动的 136/365，是否剔除由使用方决定）。 */
    val PROCRUSTES_INDICES: IntArray = intArrayOf(
{fmt_short_list(basis_idx, 17)}
    )

    /** 与 [PROCRUSTES_INDICES] 一一对应的权重（原样，未归一化）。 */
    val PROCRUSTES_WEIGHTS: FloatArray = floatArrayOf(
{fmt_short_list(basis_w, 6)}
    )

    private fun decodeIndices(encoded: String): ShortArray {{
        val n = encoded.length / 2
        val out = ShortArray(n)
        for (i in 0 until n) {{
            val hi = encoded[i * 2].code - 48
            val lo = encoded[i * 2 + 1].code - 48
            out[i] = ((hi shl 5) or lo).toShort()
        }}
        return out
    }}

    private fun decodeUvs(hex: String): FloatArray {{
        val n = hex.length / 4
        val out = FloatArray(n)
        for (i in 0 until n) {{
            val v = hex.substring(i * 4, i * 4 + 4).toInt(16)
            out[i] = v / 65535f
        }}
        return out
    }}

    private fun decodeXyz(hex: String): FloatArray {{
        val n = hex.length / 4
        val out = FloatArray(n)
        for (i in 0 until n) {{
            val raw = hex.substring(i * 4, i * 4 + 4).toInt(16)
            out[i] = (if (raw >= 32768) raw - 65536 else raw) / {XYZ_SCALE}f
        }}
        return out
    }}

    private fun buildFullIndices(): ShortArray {{
        val out = ShortArray(TOTAL_TRIANGLE_COUNT * 3)
        CANONICAL_INDICES.copyInto(out)
        var pos = CANONICAL_INDICES.size
        // 72 个脸廓 → 内层 C1 柔化过渡环三角形（CCW 正绕序）。
        for (k in 0 until OVAL_VERTEX_COUNT) {{
            val k2 = (k + 1) % OVAL_VERTEX_COUNT
            val o0 = FACE_OVAL[k].toShort()
            val o1 = FACE_OVAL[k2].toShort()
            val m0 = (MID_BUFFER_OFFSET + k).toShort()
            val m1 = (MID_BUFFER_OFFSET + k2).toShort()
            out[pos++] = o0; out[pos++] = m1; out[pos++] = m0
            out[pos++] = o0; out[pos++] = o1; out[pos++] = m1
        }}
        // 72 个内层过渡环 → 零位移外缓冲环三角形（CCW 正绕序）。
        for (k in 0 until OVAL_VERTEX_COUNT) {{
            val k2 = (k + 1) % OVAL_VERTEX_COUNT
            val m0 = (MID_BUFFER_OFFSET + k).toShort()
            val m1 = (MID_BUFFER_OFFSET + k2).toShort()
            val b0 = (BUFFER_OFFSET + k).toShort()
            val b1 = (BUFFER_OFFSET + k2).toShort()
            out[pos++] = m0; out[pos++] = b1; out[pos++] = b0
            out[pos++] = m0; out[pos++] = m1; out[pos++] = b1
        }}
        // 44 个零位移外缓冲环 → 屏幕外锚点恒等背景三角形（CCW 正绕序）。
        for (j in 0 until OUTER_VERTEX_COUNT) {{
            val j2 = (j + 1) % OUTER_VERTEX_COUNT
            val s0 = OUTER_ANCHOR_STOPS[j]
            val s1 = if (j < OUTER_VERTEX_COUNT - 1) OUTER_ANCHOR_STOPS[j2] else OVAL_VERTEX_COUNT
            val f0 = (OUTER_OFFSET + j).toShort()
            val f1 = (OUTER_OFFSET + j2).toShort()
            for (k in s0 until s1) {{
                val b0 = (BUFFER_OFFSET + (k % OVAL_VERTEX_COUNT)).toShort()
                val b1 = (BUFFER_OFFSET + ((k + 1) % OVAL_VERTEX_COUNT)).toShort()
                out[pos++] = b0; out[pos++] = b1; out[pos++] = f0
            }}
            val bs1 = (BUFFER_OFFSET + (s1 % OVAL_VERTEX_COUNT)).toShort()
            out[pos++] = bs1; out[pos++] = f1; out[pos++] = f0
        }}
        return out
    }}

    private const val CANONICAL_TRIS_PACKED: String =
        {_chunk_kotlin_string(tri_str)}

    private const val CANONICAL_UV_HEX: String =
        {_chunk_kotlin_string(uv_str)}

    private const val CANONICAL_XYZ_HEX: String =
        {_chunk_kotlin_string(xyz_str)}
}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    extracted = extract_canonical()
    expected = render_kotlin(*extracted)

    if args.verify:
        if not os.path.isfile(OUT_KT):
            print(f'FAIL  缺少 {OUT_KT}')
            return 1
        actual = open(OUT_KT, encoding='utf-8').read()
        if actual != expected:
            print(f'FAIL  {OUT_KT} 与 face_landmarker.task 规范拓扑不一致，请重跑 gen_face_topology.py')
            return 1
        print('PASS  FaceMeshTopology.kt 与 MediaPipe 规范 468 点拓扑（1086 水密三角形 + 120 唇带 + 28 眼裂）、'
              '3D 坐标与 33 个 Procrustes 稳定点完全一致')
        return 0

    with open(OUT_KT, 'w', encoding='utf-8') as f:
        f.write(expected)
    print(f'OK    已生成 {OUT_KT} ({len(expected)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
