"""直播美颜参考模型：规格 v1 第 2–4 节的唯一数学真值。

规格：docs/plans/2026-11-11-live-beauty-capture-optimization-plan.md（冲突时以规格为准）。
来源：数值评审（crit-numeric）的 beauty_ref.py，只保留 FINAL 常量与生成共享向量所需的函数。
Android（Kotlin/GLSL）与 iOS（Objective-C/CIKL）按本文件逐行移植，共享向量由 gen_vectors.py 生成。

约定
- 颜色：伽马编码 0..1 浮点；rgb8 = round(x * 255)。
- 图像数组：float64 [H, W, 3]，第 0 行是画面顶部。
- 纹理坐标：归一化，y 向上，uv = ((col + 0.5) / W, 1 - (row + 0.5) / H)。
- 度量空间：p' = (x, y / aspect)，aspect = W / H。
- MediaPipe 关键点：归一化图像坐标，y 向下；进入几何前先转纹理坐标 (x, 1 - y)。

运行：/usr/bin/python3 -I beauty_ref.py。numpy 只是开发依赖，见 README.md。
"""
import math
import os
import sys

# 只有设置了 BEAUTY_PYDEPS 才把该目录加入 sys.path；未设置时依赖当前解释器能正常 import 的 numpy。
_PYDEPS = os.environ.get("BEAUTY_PYDEPS", "")
if _PYDEPS:
    _PYDEPS = os.path.abspath(os.path.expanduser(_PYDEPS))
    if not os.path.isdir(_PYDEPS):
        sys.exit("BEAUTY_PYDEPS 指向的目录不存在：%s（见 README.md「运行与开发依赖」）" % _PYDEPS)
    if _PYDEPS not in sys.path:
        sys.path.insert(0, _PYDEPS)
try:
    import numpy as np  # noqa: E402
except ImportError:
    sys.exit("缺少 numpy：按 README.md「运行与开发依赖」安装到某个目录，并用环境变量 BEAUTY_PYDEPS 指向它")

# ----------------------------------------------------------------------------
# 常量（规格 v1）
# ----------------------------------------------------------------------------
L709 = (0.2126, 0.7152, 0.0722)        # 亮度、饱和度、方差与锐化细节
L601 = (0.299, 0.587, 0.114)           # 仅 monochrome 滤镜（沿用 Android 现有公式）

# 第 2 节几何
WARP_POINTS = (132, 361, 172, 397, 33, 133, 362, 263, 1)   # 既有塑形点，要求 [0, 1]
MASK_POINTS = (234, 454, 10, 152, 61, 291, 0, 17)          # 新增遮罩点，允许 [-0.25, 1.25]
MASK_POINT_MIN, MASK_POINT_MAX = -0.25, 1.25
SPAN_MIN, SPAN_MAX = 0.1, 0.85
EYE_WIDTH_MIN, EYE_WIDTH_SPAN = 0.015, 0.5
YAW_MAX = 0.45                          # r ≤ 0.45 几何有效（约 28°）
WARP_FADE = (0.24, 0.30)                # warpWeight = 1 - smoothstep(0.24, 0.30, r)
FACE_WIDTH_MIN = 0.05                   # |p454' - p234'|
OVAL_SX, OVAL_SY, OVAL_R_MIN = 1.12, 1.15, 0.03
MOUTH_SX, MOUTH_SY, MOUTH_RY_MIN, MOUTH_RX_MIN = 1.20, 1.25, 0.25, 0.005
EYE_RX, EYE_RY, EYE_DY = 0.72, 0.36, 0.04
EYE_WARP_SCALE = 0.9                    # 既有 eyes.z = 0.9 × 眼角距

# 第 2 节遮罩
E_CLAMP, E_RADIUS_MIN = 4.0, 1e-4
OVAL_FEATHER, EYE_FEATHER, MOUTH_FEATHER = (0.50, 1.0), (0.55, 1.0), (0.65, 1.0)

# 第 4.2 节合成
SKIN_FLOOR, EXCL_DIM = 0.85, 0.55
THETA, K_MAX = 0.15, 0.9
CORE, SHARP_CLAMP, SHARP_GAIN, SHARP_SKIN = (0.008, 0.03), 0.1, 0.8, 0.6
WHITEN_S, WHITEN_KEEP, WHITEN_FLOOR = 0.5, 0.5, 0.15
ROSY_D, ROSY_DIR = 0.03, (1.402, -0.714, 0.0)
BRIGHT_A = 0.5
# 滤镜编号与公式以 Android VideoEffectsFilter 现有着色器为准。
STYLES = {"none": 0, "warm": 1, "cool": 2, "monochrome": 3, "sepia": 4, "rose": 5}
WARM, COOL, ROSE = (0.045, 0.008, -0.025), (-0.025, 0.0, 0.045), (0.045, -0.005, 0.025)
SEPIA = ((0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131))

# 第 4.1 节预处理
SIGMA, TAPS, DELTA = 2.0, 4, 7.07
SHADER_PAIRS = ((0.0, 0.2042), (1.4072, 0.3040), (3.2939, 0.0939))   # Android 线性采样（偏移, 权重）
IOS_CONV9 = (0.027631, 0.066282, 0.123832, 0.180174, 0.204164, 0.180174, 0.123832, 0.066282, 0.027631)

# 第 3 节平滑器
SMOOTHER = dict(fc_min=0.6, k=60.0, dead=0.012, t_in_ms=150.0, hold_ms=150.0, t_out_ms=250.0,
                interp_min_ms=66.0, interp_max_ms=150.0, interp_init_ms=100.0,
                dt_clamp_ms=100.0, te_clamp_ms=(33.0, 500.0), rx_min=1e-3)

DEFAULT_PARAMS = dict(smoothing=0.0, skinOnly=True, sharpness=0.0, whiten=0.0, rosy=0.0,
                      brightness=0.0, saturation=1.0, filter="none", filterIntensity=0.0)


def constants():
    """共享向量 constants 段：各端测试可断言自己用的是同一组数。"""
    return dict(
        luma709=list(L709), luma601_monochrome=list(L601),
        geometry=dict(warpPoints=list(WARP_POINTS), maskPoints=list(MASK_POINTS),
                      maskPointRange=[MASK_POINT_MIN, MASK_POINT_MAX], span=[SPAN_MIN, SPAN_MAX],
                      eyeWidthMin=EYE_WIDTH_MIN, eyeWidthMaxOfSpan=EYE_WIDTH_SPAN, eyeWarpScale=EYE_WARP_SCALE,
                      yawMax=YAW_MAX, warpFade=list(WARP_FADE), faceWidthMin=FACE_WIDTH_MIN,
                      ovalScale=[OVAL_SX, OVAL_SY], ovalRadiusMin=OVAL_R_MIN,
                      mouthScale=[MOUTH_SX, MOUTH_SY], mouthRyMinOfRx=MOUTH_RY_MIN, mouthRxMin=MOUTH_RX_MIN,
                      eyeMask=dict(rx=EYE_RX, ry=EYE_RY, dy=EYE_DY)),
        masks=dict(eClamp=E_CLAMP, radiusMin=E_RADIUS_MIN, ovalFeather=list(OVAL_FEATHER),
                   eyeFeather=list(EYE_FEATHER), mouthFeather=list(MOUTH_FEATHER)),
        composite=dict(skinFloor=SKIN_FLOOR, exclDim=EXCL_DIM, theta=THETA, kMax=K_MAX,
                       sharpCore=list(CORE), sharpClamp=SHARP_CLAMP, sharpGain=SHARP_GAIN, sharpSkin=SHARP_SKIN,
                       whitenS=WHITEN_S, whitenKeep=WHITEN_KEEP, whitenFloor=WHITEN_FLOOR,
                       rosyD=ROSY_D, rosyDir=list(ROSY_DIR), brightA=BRIGHT_A,
                       styles=STYLES, warm=list(WARM), cool=list(COOL), rose=list(ROSE), sepia=[list(r) for r in SEPIA]),
        prepass=dict(sigma=SIGMA, taps=TAPS, delta=DELTA, quarter="ceil(w/4) x ceil(h/4)"),
        smoother=dict(SMOOTHER, te_clamp_ms=list(SMOOTHER["te_clamp_ms"])),
    )


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a * (1.0 - t) + b * t


def _smoothstep1(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------
# 第 4.2 节：肤色门限（既有 YCbCr 平滑门限，常量不变）
# ----------------------------------------------------------------------------
def skin_factors(rgb):
    """肤色门限的四个因子：cb 下沿、cb 上沿、cr 下沿、cr 上沿（skin_weight 是它们按此顺序的乘积）。"""
    rgb = np.asarray(rgb, dtype=np.float64)
    cb = 0.5 + rgb @ np.array([-0.168736, -0.331264, 0.5])
    cr = 0.5 + rgb @ np.array([0.5, -0.418688, -0.081312])
    return (smoothstep(0.26, 0.34, cb), 1.0 - smoothstep(0.48, 0.56, cb),
            smoothstep(0.50, 0.55, cr), 1.0 - smoothstep(0.72, 0.80, cr))


def skin_weight(rgb):
    cb_low, cb_high, cr_low, cr_high = skin_factors(rgb)
    return cb_low * cb_high * cr_low * cr_high


# ----------------------------------------------------------------------------
# 第 2 节：共享几何（纯标量，便于两端逐行移植）
# ----------------------------------------------------------------------------
STATE_SIZE = 41
STATE_KINDS = (["x", "y"] * 8 +                      # cheeks 4 ×（origin.xy, target.xy）     [0..15]
               ["x", "y", "x"] * 2 +                 # eyes 2 ×（cx, cy, 0.9 × 眼角距）        [16..21]
               ["x", "y", "x", "x"] +                # oval cx, cy, rx, ry                     [22..25]
               ["axis", "axis"] +                    # axis u.x, u.y                           [26..27]
               ["x", "y", "x", "x"] +                # mouth cx, cy, rx, ry                    [28..31]
               ["x", "y", "x", "x"] * 2 +            # eyeMasks 2 ×（cx, cy, rx, ry）          [32..39]
               ["none"])                             # warpWeight，不参与偏差                  [40]
STATE_RX_INDEX = 24
STATE_AXIS_INDEX = 26
STATE_WARP_INDEX = 40


def _finite(x, y):
    return math.isfinite(x) and math.isfinite(y)


def yaw_ratio(points, aspect):
    """r = 既有侧脸量 ÷ span（即现行判据 yaw > span × 0.3 里的比值）。塑形点或 span 无效时返回 None。"""
    if len(points) < 468 or not math.isfinite(aspect) or aspect <= 0:
        return None
    for i in WARP_POINTS:
        x, y = points[i]
        if not (_finite(x, y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
    left, right, nose = _tex(points, 132), _tex(points, 361), _tex(points, 1)
    span = _dist(left, right, aspect)
    if not (SPAN_MIN <= span <= SPAN_MAX):
        return None
    cx, cy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    yaw = abs((nose[0] - cx) * (right[0] - left[0]) +
              (nose[1] - cy) * (right[1] - left[1]) / (aspect * aspect)) / span
    return yaw / span


def _tex(points, i):
    return (points[i][0], 1.0 - points[i][1])


def _dist(a, b, aspect):
    return math.hypot(a[0] - b[0], (a[1] - b[1]) / aspect)


def full_geometry(points, aspect):
    """points：至少 468 个 MediaPipe 归一化图像坐标 (x, y)，y 向下。返回 FaceGeometry 字典或 None（全有或全无）。

    - cheeks[16]、eyes[6]：既有塑形 uniform，纹理坐标；eyes.z = 0.9 × 眼角距（度量）。
    - oval/mouth/eyeMasks：(cx, cy, rx, ry)，中心是纹理坐标（y 向上、不除 aspect），半径是度量值（x 单位）。
    - axis：度量空间单位向量 u，v = (-u.y, u.x) 指向额头。
    - warpWeight = 1 - smoothstep(0.24, 0.30, r)，只乘在塑形强度上。"""
    return geometry_with_reason(points, aspect)[0]


def geometry_with_reason(points, aspect):
    """同 full_geometry，另返回拒绝原因（有效时为 None），供向量自检与调试。"""
    if len(points) < 468 or not math.isfinite(aspect) or aspect <= 0:
        return None, "input"
    for i in WARP_POINTS:
        x, y = points[i]
        if not (_finite(x, y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None, "warp_point"
    for i in MASK_POINTS:
        x, y = points[i]
        if not (_finite(x, y) and MASK_POINT_MIN <= x <= MASK_POINT_MAX and MASK_POINT_MIN <= y <= MASK_POINT_MAX):
            return None, "mask_point"
    left, right = _tex(points, 132), _tex(points, 361)
    span = _dist(left, right, aspect)
    if not (SPAN_MIN <= span <= SPAN_MAX):
        return None, "span"
    r = yaw_ratio(points, aspect)
    if r > YAW_MAX:
        return None, "yaw"
    warp_weight = 1.0 - _smoothstep1(WARP_FADE[0], WARP_FADE[1], r)
    cheeks = []
    for a_i, b_i in ((132, 361), (172, 397)):
        a, b = _tex(points, a_i), _tex(points, b_i)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        cheeks += [a[0], a[1], mx, my, b[0], b[1], mx, my]
    eyes = []
    for a_i, b_i in ((33, 133), (362, 263)):
        a, b = _tex(points, a_i), _tex(points, b_i)
        width = _dist(a, b, aspect)
        if width < EYE_WIDTH_MIN or width > span * EYE_WIDTH_SPAN:
            return None, "eye_width"
        eyes += [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, width * EYE_WARP_SCALE]

    def m(i):  # 度量空间
        x, y = _tex(points, i)
        return (x, y / aspect)

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def mid(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    p234, p454, p10, p152 = m(234), m(454), m(10), m(152)
    w = sub(p454, p234)
    wl = math.hypot(w[0], w[1])
    if wl < FACE_WIDTH_MIN:
        return None, "face_width"
    u = (w[0] / wl, w[1] / wl)
    v = (-u[1], u[0])
    if dot(v, sub(p10, p152)) < 0:   # v 必须指向额头（152 → 10）
        u = (-u[0], -u[1])
        v = (-u[1], u[0])
    if not (_finite(u[0], u[1]) and abs(math.hypot(u[0], u[1]) - 1.0) < 1e-6):
        return None, "axis"

    def centre(side_a, side_b, mid_a, mid_b):
        """u 分量取侧向点对中点与中线点对中点的平均，v 分量只取中线点对中点；返回纹理坐标。"""
        sm, mm = mid(side_a, side_b), mid(mid_a, mid_b)
        cu = 0.5 * (dot(sm, u) + dot(mm, u))
        cv = dot(mm, v)
        return (cu * u[0] + cv * v[0], (cu * u[1] + cv * v[1]) * aspect)

    ox, oy = centre(p234, p454, p10, p152)
    orx = 0.5 * wl * OVAL_SX
    ory = 0.5 * abs(dot(sub(p10, p152), v)) * OVAL_SY
    if orx < OVAL_R_MIN or ory < OVAL_R_MIN:
        return None, "oval_radius"
    p61, p291, p0, p17 = m(61), m(291), m(0), m(17)
    mx, my = centre(p61, p291, p0, p17)
    mrx = 0.5 * abs(dot(sub(p61, p291), u)) * MOUTH_SX
    if mrx < MOUTH_RX_MIN:
        return None, "mouth_radius"
    mry = max(0.5 * abs(dot(sub(p0, p17), v)) * MOUTH_SY, MOUTH_RY_MIN * mrx)
    eye_masks = []
    for a_i, b_i in ((33, 133), (362, 263)):
        pa, pb = m(a_i), m(b_i)
        width = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        c = mid(pa, pb)
        cx, cym = c[0] + EYE_DY * width * v[0], c[1] + EYE_DY * width * v[1]
        eye_masks += [cx, cym * aspect, EYE_RX * width, EYE_RY * width]
    return dict(cheeks=cheeks, eyes=eyes, oval=[ox, oy, orx, ory], axis=[u[0], u[1]],
                mouth=[mx, my, mrx, mry], eyeMasks=eye_masks, warpWeight=warp_weight), None


def geometry_to_vector(geo):
    return np.array(geo["cheeks"] + geo["eyes"] + geo["oval"] + geo["axis"] + geo["mouth"] +
                    geo["eyeMasks"] + [geo["warpWeight"]], dtype=np.float64)


def vector_to_geometry(x):
    x = [float(v) for v in x]
    return dict(cheeks=x[0:16], eyes=x[16:22], oval=x[22:26], axis=x[26:28], mouth=x[28:32],
                eyeMasks=x[32:40], warpWeight=x[40])


# ----------------------------------------------------------------------------
# 第 2 节：遮罩（与着色器 / CI 内核逐式一致；uv 为逆向变形后的采样坐标）
# ----------------------------------------------------------------------------
def ellipse_e(uv, c4, axis, aspect):
    uv = np.asarray(uv, dtype=np.float64)
    dx = uv[..., 0] - c4[0]
    dy = (uv[..., 1] - c4[1]) / aspect
    a = np.clip((dx * axis[0] + dy * axis[1]) / max(c4[2], E_RADIUS_MIN), -E_CLAMP, E_CLAMP)
    b = np.clip((-dx * axis[1] + dy * axis[0]) / max(c4[3], E_RADIUS_MIN), -E_CLAMP, E_CLAMP)
    return a * a + b * b


def face_masks(uv, geo, aspect, presence):
    """返回 dict(oval, eye0, eye1, mouth, face, excl)。presence == 0 时 face = excl = 0（各端可跳过遮罩计算）。"""
    axis = geo["axis"]
    em = geo["eyeMasks"]
    oval = 1.0 - smoothstep(OVAL_FEATHER[0], OVAL_FEATHER[1], ellipse_e(uv, geo["oval"], axis, aspect))
    e0 = 1.0 - smoothstep(EYE_FEATHER[0], EYE_FEATHER[1], ellipse_e(uv, em[0:4], axis, aspect))
    e1 = 1.0 - smoothstep(EYE_FEATHER[0], EYE_FEATHER[1], ellipse_e(uv, em[4:8], axis, aspect))
    mouth = 1.0 - smoothstep(MOUTH_FEATHER[0], MOUTH_FEATHER[1], ellipse_e(uv, geo["mouth"], axis, aspect))
    keep = (1.0 - e0) * (1.0 - e1) * (1.0 - mouth)
    return dict(oval=oval, eye0=e0, eye1=e1, mouth=mouth,
                face=presence * oval * keep, excl=presence * oval * (1.0 - keep))


def pixel_uv(width, height):
    """全图像素中心的纹理坐标 [H, W, 2]，第 0 行是顶部（uv.y 接近 1）。"""
    xs = (np.arange(width) + 0.5) / width
    ys = 1.0 - (np.arange(height) + 0.5) / height
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


# ----------------------------------------------------------------------------
# 第 4.1 节：预处理（1/4 分辨率均值与方差）
# ----------------------------------------------------------------------------
def q8(a):
    """RGBA8 纹理写入：clamp 后四舍五入到 8 位。"""
    return np.round(np.clip(a, 0.0, 1.0) * 255.0) / 255.0


def quarter_size(width, height):
    return int(math.ceil(width / 4.0)), int(math.ceil(height / 4.0))


def box4(img):
    """T0：每个 1/4 像素取 4×4 原始像素均值（两端都等价于 ±1 原始像素处 4 个双线性抽头）；越界按复制延拓。"""
    h, w = img.shape[:2]
    sw, sh = quarter_size(w, h)
    p = np.pad(img, ((0, sh * 4 - h), (0, sw * 4 - w), (0, 0)), mode="edge")
    return p.reshape(sh, 4, sw, 4, -1).mean(axis=(1, 3))


def gauss_kernel(sigma=SIGMA, taps=TAPS):
    i = np.arange(-taps, taps + 1)
    g = np.exp(-(i * i) / (2.0 * sigma * sigma))
    return g / g.sum()


def linear_pairs(sigma=SIGMA, taps=TAPS):
    """9 抽头离散核的线性采样表示：[(名称, 权重, 偏移)]。"""
    g = gauss_kernel(sigma, taps)[taps:]
    out = [("center", float(g[0]), 0.0)]
    for a in range(1, taps + 1, 2):
        w = g[a] + g[a + 1]
        out.append(("pair(%d,%d)" % (a, a + 1), float(w), float((a * g[a] + (a + 1) * g[a + 1]) / w)))
    return out


def blur_1d(img, axis, kernel=None):
    """一维可分离高斯，边缘 clamp-to-edge。axis 1 = 横向，0 = 纵向。"""
    k = gauss_kernel() if kernel is None else kernel
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    n = img.shape[axis]
    acc = np.zeros_like(img)
    for j, wgt in enumerate(k):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(j, j + n)
        acc = acc + wgt * p[tuple(sl)]
    return acc


def variance_seed(t0, mean):
    """v0 = min((dot(T0 - mean, L) × 7.07)², 1)。"""
    d = (t0 - mean) @ np.array(L709)
    return np.minimum((d * DELTA) ** 2, 1.0)


def prepass_float(img):
    """浮点管线（iOS RGBAh 的近似）：返回 1/4 分辨率 (T0, mean, v0, var)。"""
    t0 = box4(img)
    mean = blur_1d(blur_1d(t0, 1), 0)
    v0 = variance_seed(t0, mean)
    var = blur_1d(blur_1d(v0, 1), 0)
    return t0, mean, v0, var


def prepass_rgba8(img):
    """Android 5 通道 RGBA8 管线：每个通道的输出都写入 8 位纹理。
    P1 DOWN → T0；P2 横向模糊 rgb → T1；P3 纵向模糊得 mean，并用未量化的 mean 算 v0，输出 (mean, v0)；
    P4/P5 只模糊 alpha（rgb 原样带过）。返回 1/4 分辨率 (T0, mean, v0, var)。"""
    t0 = q8(box4(img))
    t1 = q8(blur_1d(t0, 1))
    mean_f = blur_1d(t1, 0)
    mean = q8(mean_f)
    v0 = q8(variance_seed(t0, mean_f))
    var = q8(blur_1d(q8(blur_1d(v0, 1)), 0))
    return t0, mean, v0, var


def upsample4(q, height, width):
    """线性采样放大：全分辨率像素中心对应 1/4 坐标 (x + 0.5) / 4 - 0.5，越界 clamp（GL_LINEAR + CLAMP_TO_EDGE，
    iOS clamp 后 4× 仿射，在 4 的倍数尺寸上两者相同）。"""
    sh, sw = q.shape[:2]
    fy = (np.arange(height) + 0.5) / 4.0 - 0.5
    fx = (np.arange(width) + 0.5) / 4.0 - 0.5
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    ty, tx = fy - y0, fx - x0
    y0c, y1c = np.clip(y0, 0, sh - 1), np.clip(y0 + 1, 0, sh - 1)
    x0c, x1c = np.clip(x0, 0, sw - 1), np.clip(x0 + 1, 0, sw - 1)
    if q.ndim == 3:
        ty, tx = ty[:, None, None], tx[None, :, None]
    else:
        ty, tx = ty[:, None], tx[None, :]
    top = q[y0c][:, x0c] * (1 - tx) + q[y0c][:, x1c] * tx
    bot = q[y1c][:, x0c] * (1 - tx) + q[y1c][:, x1c] * tx
    return top * (1 - ty) + bot * ty


def avg4(img):
    """上下左右 ±1 像素均值，边缘 clamp。"""
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


# ----------------------------------------------------------------------------
# 第 4.2 节：合成（逐像素；数组参数在前导维上广播：I/mean/a4 [..., 3]，var/face/excl [...]）
# ----------------------------------------------------------------------------
def lift(x, a):
    return x * (1.0 + a) / (1.0 + a * x)


def style_of(params):
    return STYLES[params["filter"]]


def filter_color(color, style):
    """Android 现有 5 种滤镜（style 1 warm、2 cool、3 monochrome、4 sepia、5 rose）；0 原样返回。"""
    if style == 5:
        return color + np.array(ROSE)
    if style == 4:
        return color @ np.array(SEPIA).T
    if style == 3:
        return np.repeat((color @ np.array(L601))[..., None], 3, axis=-1)
    if style == 2:
        return color + np.array(COOL)
    if style == 1:
        return color + np.array(WARM)
    return color


def is_neutral(params):
    p = dict(DEFAULT_PARAMS, **params)
    return (p["smoothing"] == 0 and p["sharpness"] == 0 and p["whiten"] == 0 and p["rosy"] == 0 and
            p["brightness"] == 0 and p["saturation"] == 1 and (p["filter"] == "none" or p["filterIntensity"] == 0))


def composite(I, mean, var, a4, face, excl, params, debug=False, sharpen_clamp=True):
    """规格 4.2 的 beautyColor()，假定 processing 为真（直通由调用方判断）。

    sharpen_clamp=False 去掉「锐化后、美白前」的 clamp(color, 0, 1)，只供 gen_vectors.py 自检向量的区分度，
    不是规格行为。"""
    L = np.array(L709)
    P = dict(DEFAULT_PARAMS, **params)
    I = np.asarray(I, dtype=np.float64)
    face = np.asarray(face, dtype=np.float64)
    excl = np.asarray(excl, dtype=np.float64)
    gate = skin_weight(I)
    m_skin = face * mix(SKIN_FLOOR, 1.0, gate) + (1.0 - face) * gate * (1.0 - EXCL_DIM * excl)
    color = I.copy()
    k = np.zeros_like(m_skin)
    if P["smoothing"] > 0:
        r = np.asarray(var, dtype=np.float64) / THETA
        k = P["smoothing"] * K_MAX * (m_skin if P["skinOnly"] else 1.0) / (1.0 + r * r)
        color = mix(I, np.asarray(mean, dtype=np.float64), k[..., None])
    if P["sharpness"] > 0:
        detail = (I - np.asarray(a4, dtype=np.float64)) @ L
        detail = detail * smoothstep(CORE[0], CORE[1], np.abs(detail))
        gain = P["sharpness"] * SHARP_GAIN * (1.0 - SHARP_SKIN * m_skin) * (1.0 - k)
        color = color + (np.clip(detail, -SHARP_CLAMP, SHARP_CLAMP) * gain)[..., None]
    sharpened = color
    if sharpen_clamp:
        color = np.clip(color, 0.0, 1.0)
    if P["whiten"] > 0:
        y = np.clip(color @ L, 0.0, 1.0)
        w = np.minimum(color * (1.0 + WHITEN_S * P["whiten"] * (1.0 - mix(color, y[..., None], WHITEN_KEEP))), 1.0)
        color = mix(color, w, mix(WHITEN_FLOOR, 1.0, m_skin)[..., None])
    if P["rosy"] > 0:
        d = ROSY_D * P["rosy"] * m_skin
        color = color + d[..., None] * np.array(ROSY_DIR)
    if P["brightness"] > 0:
        color = lift(color, BRIGHT_A * P["brightness"])
    color = mix((color @ L)[..., None], color, P["saturation"])
    color = mix(color, np.clip(filter_color(color, style_of(P)), 0.0, 1.0), P["filterIntensity"])
    out = np.clip(color, 0.0, 1.0)
    if debug:
        return out, dict(gate=gate, mSkin=m_skin, k=k, sharpened=sharpened)
    return out


def process_frame(img, params, geo=None, presence=0.0, pipeline="float"):
    """无变形的整帧管线：预处理（仅 smoothing > 0）→ 遮罩 → 合成。pipeline 'float' | 'rgba8'。
    中性参数且无塑形时严格直通，返回原图。"""
    P = dict(DEFAULT_PARAMS, **params)
    if is_neutral(P):
        return img.copy()
    h, w = img.shape[:2]
    if P["smoothing"] > 0:
        _, mean_q, _, var_q = prepass_rgba8(img) if pipeline == "rgba8" else prepass_float(img)
        mean, var = upsample4(mean_q, h, w), upsample4(var_q, h, w)
    else:
        mean, var = img, np.zeros((h, w))
    if geo is not None and presence > 0:
        m = face_masks(pixel_uv(w, h), geo, w / h, presence)
        face, excl = m["face"], m["excl"]
    else:
        face, excl = np.zeros((h, w)), np.zeros((h, w))
    return composite(img, mean, var, avg4(img), face, excl, P)


# ----------------------------------------------------------------------------
# 第 3 节：时间平滑（FaceGeometrySmoother 的规范实现）
# ----------------------------------------------------------------------------
class FinalSmoother:
    """状态向量 41 维（布局见 STATE_KINDS）。update(target?, sampleTimestampMs, nowMs) → (state 或 None, presence)。
    - 自适应低通只在新样本（采样时间戳变化）到达时执行一次，te 为相邻样本时间戳间隔；
    - 偏差按单位统一：x 类 ×1，纹理 y 类 ÷aspect，axis × oval.rx，warpWeight 不参与；再除以 oval.rx；
    - 每帧在 prev → f 之间按实测样本间隔线性插值，消除 10 Hz 台阶；
    - presence：有目标时 150 ms 渐入；目标消失后先保持 150 ms，再 250 ms 渐出，到 0 清空几何。"""

    def __init__(self, aspect):
        self.aspect = aspect
        self.reset()

    def reset(self):
        self.f = None           # 滤波后的样本
        self.prev = None        # 上个样本到达时的输出
        self.ts = None          # 上个已消费样本的采样时间戳
        self.t_arr = 0.0        # 该样本到达时的渲染时间
        self.T = SMOOTHER["interp_init_ms"]
        self.x = None           # 上次输出
        self.presence = 0.0
        self.missing = 0.0
        self.last = None

    def _weights(self, rx):
        return np.array([1.0 if k == "x" else 1.0 / self.aspect if k == "y" else rx if k == "axis" else 0.0
                         for k in STATE_KINDS])

    def update(self, target, sample_ms, now_ms):
        S = SMOOTHER
        if self.last is None:
            dt = 0.0
        else:
            dt = min(max(now_ms - self.last, 0.0), S["dt_clamp_ms"])
            if dt == 0.0:
                return self._out()
        self.last = now_ms
        if target is not None:
            target = np.asarray(target, dtype=np.float64)
            self.missing = 0.0
            if self.f is None:
                self.f, self.prev, self.x = target.copy(), target.copy(), target.copy()
                self.ts, self.t_arr, self.T = sample_ms, now_ms, S["interp_init_ms"]
            elif sample_ms != self.ts:
                te = min(max(sample_ms - self.ts, S["te_clamp_ms"][0]), S["te_clamp_ms"][1])
                self.prev = self.x.copy()
                rx = max(self.f[STATE_RX_INDEX], S["rx_min"])
                s = float(np.max(np.abs(target - self.f) * self._weights(rx))) / rx
                fc = S["fc_min"] + S["k"] * max(0.0, s - S["dead"])
                alpha = 1.0 - math.exp(-2.0 * math.pi * fc * te / 1000.0)
                self.f = self.f + alpha * (target - self.f)
                self.T = min(max(0.5 * self.T + 0.5 * te, S["interp_min_ms"]), S["interp_max_ms"])
                self.ts, self.t_arr = sample_ms, now_ms
            self.presence = min(1.0, self.presence + dt / S["t_in_ms"])
        else:
            self.missing += dt
            if self.missing > S["hold_ms"]:
                self.presence = max(0.0, self.presence - dt / S["t_out_ms"])
            if self.presence <= 0.0:
                self.f = self.prev = self.x = None
                self.ts = None
                self.missing = 0.0
        if self.f is not None:
            g = min(1.0, max(0.0, (now_ms - self.t_arr + dt) / self.T))
            self.x = self.prev + g * (self.f - self.prev)
            i = STATE_AXIS_INDEX
            n = math.hypot(self.x[i], self.x[i + 1])
            if n > 1e-6:
                self.x[i] /= n
                self.x[i + 1] /= n
        return self._out()

    def _out(self):
        return (None if self.x is None else self.x.copy()), self.presence


if __name__ == "__main__":
    print("9 抽头 sigma=2 的线性采样表示：")
    for row in linear_pairs():
        print("  %-10s 权重 %.4f 偏移 %.4f" % row)
    k = gauss_kernel()
    i = np.arange(-TAPS, TAPS + 1)
    print("  截断后有效 sigma：%.4f（1/4 像素）" % math.sqrt(float((k * i * i).sum())))
