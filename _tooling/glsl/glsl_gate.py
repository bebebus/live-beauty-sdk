#!/usr/bin/env python3
"""Android GLSL 宿主机门禁（规格 v1 第 8 节、Task 3），只用 Python 标准库。

从 VideoEffectsShaders.kt 提取原文 → 静态核对 Kotlin 接线（VideoEffectsFilter.kt、BeautyPrePass.kt 的
程序配对必须与 PROGRAMS 一致；按 VideoEffectsFilter.kt 的打包代码把 end_to_end.state 拆成几何 uniform，
必须等于由 geometry 各字段按 meta.uniforms.android 拼出的值；BLUR_VERTEX / BLUR 的线性采样常量必须等于
kernel.shaderPairs；导向滤波常量（含 P1 抽头间距 DOWN_TAP_SCALE，并核对 GuidedFilterPrePass 用的就是它）从
BeautyQualityTuning.kt 解析，背景虚化阈值与细节保留 DETAIL_KEEP 从 BackgroundBlurPass.kt 解析，主合成的
uLut / uGrade 接线从 VideoEffectsFilter.kt 核对）→ 无头 Chrome WebGL1（ANGLE/SwiftShader，严格 GLSL ES 1.00）
编译并链接全部生产程序（highp 与强制 mediump 两种）→ 负对照必须被拒 → 合成程序活跃 uniform 向量（含采样器）
不超过 16、主合成 compositePro 恰为 15 → 用共享向量 beauty_vectors.json 跑 skin_gate / composite / masks / prepass / end_to_end 五段并按
meta.tolerance 比较（end_to_end 的几何 uniform 用上面从 Kotlin 打包代码得到的值）；中性配置与
processing = 0 必须逐字节直通 → 主合成 compositePro 的数值探针、合成内 LUT、回退预处理、导向滤波阶跃边与
导向系数曲线（GUIDED_COEFF_FRAGMENT 单元级，守软映射与自适应边缘锁）→ 毛发边界探针 G1–G8（发缘光晕与边宽、
色渗、1px 发丝相位、脸外发团纹理、中等瑕疵与小黑点代价对冻结基线 blemish_baseline.json、时域闪烁、胡茬对比、
背景虚化飞发保留与未虚化光环）。G1–G8 的场景按 numpy 随机数复刻设计组探针、逐字节相同，在进程池里生成。
2026-11-21 评审修复追加：G1 矩阵（4 发色 × 3 肤色 × def/full，无头发遮罩与带头发遮罩 + 导向滤波头发加权两条路径，
对 1.26.0+46 与 1.24.x 的冻结值）、头发加权路径在没有头发时的代价（G5w / G6w / G7w）、G7 满档、G8 头部先验
（时域闪烁、肩部光环）与合并道变体（头发遮罩高斯软化 σ1.25 / 1.95 / 3.63）、BH1 发丝软剖面与墙面 keep、HA1 脸内偏置位置。
G8 还有细节保留 β 扫描（check_bg_detail_sweep）：β = detailKeep(true, k) 随「人脸近期可见」系数 k 渐变，k = 1 与生产、k = 0 与
无人脸逐字节相同，相邻档连续、逐像素单调、飞发保留率随 k 单调；BackgroundBlurPass.kt 的 β 写法（detailKeep、faceWeight 缺省 1f）
静态核对。
→ 头发遮罩探针（第二批）：A1–A7 主合成读 uSkinMask.a（非头发权重）、HA1 SemanticAlphaPass 的 HAIR_ALPHA（只写 A、阈值与
脸内偏置及其位置、方向、仿射、网格）、BH1 背景虚化头发合并道 BG_HAIR（直通、方向、头部先验、低对比金发飞发保留率、墙面）；
SemanticAlphaPass.kt / BackgroundBlurPass.kt 的头发阈值常量与只写 A、finally 恢复 colorMask 也静态核对。
→ VideoEffectsFilter.kt 的头发接线静态核对（check_hair_wiring，13 条，各条含义见该函数 docstring）：presence（uFrame.w 经
SkinMaskPass.framePresence(on, polyMask, presence)）、track（头发完成时间不并入画面延后）、semantic（A 道只在 hairShown > 0f
时画）、upload（上传后紧跟 hairGate.uploaded）、affine（运动补偿 HairMaskGate.compensate 的两个时刻与输出）、oval（脸椭圆与
显示位姿按 end_to_end 向量等于 uOval 后接轴向）、enable（启停接 hairEnabled，无脸停推理，「人脸近期可见」接 timeline.latest、
timeline.missSince 与本帧人脸快照的在途信号 snapshot.inFlight）、cpu（开虚化时间隔下限与 p50 超限停用）、engine（外部引擎停用头发
分割）、reset（切镜头复位头发分割）、weight（导向滤波头发加权只接本帧 A 道画成功的皮肤遮罩）、face（背景虚化人脸椭圆与头发无关、
细节保留系数接 hairGate.faceKeep）、neck（A 通道按天鹅颈生效程度渐变）；另有 1 条自检：门禁自带的接线变异 HAIR_WIRING_MUTANTS
（27 个）每个都必须被对应核对拦下，
变异锚点在去注释后的源码里找不到也算 FAIL（改生产写法时同步这里）。

耗时（Intel 8 核 macOS，单独运行）：约 30 s。2026-11-21 加入毛发边界段之前约 12 s，新增部分是场景生成约 5 s、
Chrome 内约 5 s（含 720×1280 背景虚化场景）、指标约 1.5 s；头发遮罩探针再加约 6 s（Chrome 内约 3 s、指标约 1.6 s）。
评审修复追加了 G1 矩阵、加权路径代价、G8 时域（3 种 × 2 背景 × 8 帧整帧虚化与回读）与合并道变体：本机负载 20–30 时单独运行
由 64.5 s 增到 72.1 s，与其他门禁并行时 130–200 s（空闲机器上未复测）；之后头发接线核对增至 13 条加自检（本机负载 18 时
单独运行 47.8 s），G8 β 扫描再加 10 次低对比棋盘的裁剪区虚化：本机负载约 10 时前后相继单独运行，56.1 s → 57.8 s。

用法：python3 -I glsl_gate.py <VideoEffectsShaders.kt> <beauty_vectors.json> <工作目录> [Kotlin 源码目录]
Kotlin 源码目录缺省为 VideoEffectsShaders.kt 所在目录，须含 VideoEffectsFilter.kt、BeautyPrePass.kt、
FaceGeometrySmoother.kt、BeautyQualityTuning.kt 与 WIRING_FILES（含 SemanticAlphaPass.kt）。环境变量 CHROME_BIN 可指定 Chrome 可执行文件，
缺省为 /Applications/Google Chrome.app；GLSL_GATE_BLEMISH_OUT 指定路径时，把本次实测的 G5 / G5b 保留率表写成
blemish_baseline.json 的 keep 格式（只在有意重定基线时用）。
"""
import base64
import concurrent.futures
import html
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
import zlib

# -I 会忽略 PYTHONDONTWRITEBYTECODE；不在仓库里留下 __pycache__。
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_shaders import ShaderSourceError, extract  # noqa: E402

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
GRID = 64
# GLES2 保证的 gl_MaxFragmentUniformVectors 下限。
MAX_UNIFORM_VECTORS = 16
# 主合成 compositePro 的活跃 uniform 向量必须恰为 15（4 个采样器 + 11 个 vec4）：头发遮罩走 uSkinMask.a，不占新 uniform，
# 最后 1 个余量不能被悄悄占掉。
COMPOSITE_PRO_VECTORS = 15
# 生产程序：与 VideoEffectsFilter.kt、BeautyPrePass.kt 等的 createProgram 调用一一对应（check_wiring 静态核对）。
PROGRAMS = [
    ('compositePro', 'VERTEX', 'FRAGMENT_PRO'),
    ('original', 'VERTEX', 'ORIGINAL'),
    ('down', 'DOWN_VERTEX', 'DOWN_FRAGMENT'),
    ('blurRgb', 'BLUR_VERTEX', 'BLUR_RGB_FRAGMENT'),
    ('blurDiff', 'BLUR_VERTEX', 'BLUR_DIFF_FRAGMENT'),
    ('blurAlpha', 'BLUR_VERTEX', 'BLUR_ALPHA_FRAGMENT'),
    # 美体变形（仅 Android，BodyWarpPass.kt）。
    ('body', 'VERTEX', 'BODY_FRAGMENT'),
    # 暗光时域降噪（仅 Android，TemporalDenoisePass.kt）。
    ('denoise', 'VERTEX', 'DENOISE_FRAGMENT'),
    # 1/3 分辨率快速导向滤波磨皮预处理（仅 Android，GuidedFilterPrePass.kt）。
    ('guidedDiff', 'BLUR_VERTEX', 'GUIDED_DIFF_FRAGMENT'),
    ('guidedCoeff', 'BLUR_VERTEX', 'GUIDED_COEFF_FRAGMENT'),
    # 导向滤波的头发加权路径（GuidedFilterPrePass.renderWeighted，传入头发权重时代替 P2 / P3 / P5）。
    ('guidedWeightBlur', 'BLUR_VERTEX', 'GUIDED_WEIGHT_BLUR_FRAGMENT'),
    ('guidedWeightDiff', 'BLUR_VERTEX', 'GUIDED_WEIGHT_DIFF_FRAGMENT'),
    ('guidedWeightCoeff', 'BLUR_VERTEX', 'GUIDED_WEIGHT_COEFF_FRAGMENT'),
    # 468 点多边形光栅化精准面部皮肤遮罩（仅 Android，SkinMaskPass.kt）。
    ('skinMask', 'SKIN_MASK_VERTEX', 'SKIN_MASK_FRAGMENT'),
    # 背景虚化（仅 Android，BackgroundBlurPass.kt；先按人像遮罩预乘再高斯模糊消黑边光晕）；画面对齐帧复制（FrameDelay.kt）用 ORIGINAL。
    ('bgPremask', 'VERTEX', 'BG_PREMASK_FRAGMENT'),
    ('bgBlur', 'VERTEX', 'BG_BLUR_FRAGMENT'),
    ('bgMix', 'VERTEX', 'BG_MIX_FRAGMENT'),
    # 更多脸型（仅 Android，FaceShapePass.kt）。
    ('face', 'VERTEX', 'FACE_FRAGMENT'),
    # 478 点 2D 三角网格微整（仅 Android，FaceMeshWarpPass.kt）。
    ('meshWarp', 'MESH_WARP_VERTEX', 'MESH_WARP_FRAGMENT'),
    # 实时局部子网格美妆（仅 Android，MakeupPass.kt）。
    ('makeupLayer', 'MAKEUP_VERTEX', 'MAKEUP_LAYER_FRAGMENT'),
    ('makeupLip', 'MAKEUP_VERTEX', 'MAKEUP_LIP_FRAGMENT'),
    ('makeupIris', 'MAKEUP_VERTEX', 'MAKEUP_IRIS_FRAGMENT'),
    # 468 点解剖子网格局部精修（仅 Android，FaceRetouchMeshPass.kt）。
    ('retouchSkin', 'RETOUCH_VERTEX', 'RETOUCH_SKIN_FRAGMENT'),
    ('retouchEyeTeeth', 'RETOUCH_VERTEX', 'RETOUCH_EYE_TEETH_FRAGMENT'),
    # 头发遮罩写皮肤遮罩 A 通道（仅 Android，SemanticAlphaPass.kt）与背景虚化的头发合并道（BackgroundBlurPass.kt）。
    ('hairAlpha', 'SEMANTIC_VERTEX', 'HAIR_ALPHA_FRAGMENT'),
    ('bgHair', 'VERTEX', 'BG_HAIR_FRAGMENT'),
]
# 兼容基线程序：保留用于校验共享向量 beauty_vectors.json（与 iOS Core Image 规格 v1 保持逐像素同步），不在生产重复编译。
REFERENCE_PROGRAMS = [
    ('composite', 'VERTEX', 'FRAGMENT'),
]
ALL_PROGRAMS = PROGRAMS + REFERENCE_PROGRAMS
# 只作为片段被引用、自身不是完整着色器的常量。LUT 查表在 FRAGMENT_PRO 内（LUT_SAMPLE），LutPass 只管纹理。
FRAGMENTS = ('PRECISION', 'COMMON', 'WARP', 'MASKS', 'COMPOSITE', 'COMPOSITE_PRO', 'LUT_SAMPLE', 'BLUR', 'GUIDED_COEFF')
# GL 类型 → 占用的 uniform 向量数（mat 按列计）。
GL_MAT2, GL_MAT3, GL_MAT4 = 0x8B5A, 0x8B5B, 0x8B5C
ROWS = {GL_MAT2: 2, GL_MAT3: 3, GL_MAT4: 4}

CONTROL_VERTEX = 'attribute vec4 aPosition;\nvoid main() { gl_Position = aPosition; }\n'
CONTROL_HEAD = 'precision mediump float;\n'
# 负对照：good 与 bad 只差一处，bad 必须被拒（编译或链接失败）、good 必须通过。
CONTROLS = [
    ('implicit_int_to_float',
     CONTROL_HEAD + 'void main() { float x = 1.0; gl_FragColor = vec4(x); }\n',
     CONTROL_HEAD + 'void main() { float x = 1; gl_FragColor = vec4(x); }\n'),
    ('float_suffix',
     CONTROL_HEAD + 'void main() { gl_FragColor = vec4(1.0); }\n',
     CONTROL_HEAD + 'void main() { gl_FragColor = vec4(1.0f); }\n'),
    ('array_constructor',
     CONTROL_HEAD + 'void main() { float w[3]; w[0] = 0.2; w[1] = 0.3; w[2] = 0.5; gl_FragColor = vec4(w[1]); }\n',
     CONTROL_HEAD + 'void main() { float w[3] = float[3](0.2, 0.3, 0.5); gl_FragColor = vec4(w[1]); }\n'),
    ('missing_default_precision',
     CONTROL_HEAD + 'void main() { float x = 1.0; gl_FragColor = vec4(x); }\n',
     'void main() { float x = 1.0; gl_FragColor = vec4(x); }\n'),
    ('uniform_loop_bound',
     CONTROL_HEAD + 'uniform int uCount;\nvoid main() { float s = 0.0; for (int i = 0; i < 4; i++) s += 0.25;'
     ' gl_FragColor = vec4(s * float(uCount)); }\n',
     CONTROL_HEAD + 'uniform int uCount;\nvoid main() { float s = 0.0; for (int i = 0; i < uCount; i++) s += 0.25;'
     ' gl_FragColor = vec4(s); }\n'),
]
# 两阶段同名 uniform 精度不一致（顶点默认 highp、片元 mediump）必须链接失败。
STAGE_VERTEX = 'attribute vec4 aPosition;\nuniform vec2 uStep;\nvarying vec2 vStep;\n' \
               'void main() { gl_Position = aPosition; vStep = uStep; }\n'
STAGE_CONTROL = ('uniform_precision_mismatch',
                 CONTROL_HEAD + 'varying vec2 vStep;\nvoid main() { gl_FragColor = vec4(vStep, 0.0, 1.0); }\n',
                 CONTROL_HEAD + 'uniform vec2 uStep;\nvarying vec2 vStep;\n'
                 'void main() { gl_FragColor = vec4(vStep + uStep, 0.0, 1.0); }\n')


class Report:
    def __init__(self):
        self.failures = []

    def line(self, ok, text):
        print(('PASS  ' if ok else 'FAIL  ') + text)
        if not ok:
            self.failures.append(text)


def uniform_names(source):
    code = re.sub(r'//[^\n]*|/\*.*?\*/', '', source, flags=re.S)
    return set(re.findall(r'\buniform\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+(\w+)', code))


def static_checks(shaders, report):
    known = set(FRAGMENTS) | {name for _, v, f in ALL_PROGRAMS for name in (v, f)}
    unknown = sorted(set(shaders) - known)
    report.line(not unknown, '常量全部纳入门禁%s' % ('' if not unknown else '：未覆盖 ' + ', '.join(unknown)))
    missing = sorted(known - set(shaders))
    report.line(not missing, '生产程序与片段常量齐全%s' % ('' if not missing else '：缺少 ' + ', '.join(missing)))
    if missing:
        return False
    for name, vertex, fragment in ALL_PROGRAMS:
        report.line(shaders[fragment].startswith(shaders['PRECISION']), '%s 以固定 PRECISION 开头' % fragment)
        shared = sorted(uniform_names(shaders[vertex]) & uniform_names(shaders[fragment]))
        report.line(not shared, '%s：同一 uniform 不在两个阶段声明%s' % (name, '' if not shared else '（' + ', '.join(shared) + '）'))
    return True


# ---- Kotlin 接线的静态核对：PROGRAMS 与几何 uniform 打包都从生产 Kotlin 原文解析，不靠手抄。
WIRING_FILES = ('VideoEffectsFilter.kt', 'GuidedFilterPrePass.kt', 'SkinMaskPass.kt', 'BeautyPrePass.kt', 'BodyWarpPass.kt', 'TemporalDenoisePass.kt', 'LutPass.kt', 'BackgroundBlurPass.kt', 'FaceShapePass.kt', 'FaceRetouchMeshPass.kt', 'FaceMeshWarpPass.kt', 'MakeupPass.kt', 'FrameDelay.kt', 'SemanticAlphaPass.kt')
# GlUtil.createProgram(VideoEffectsShaders.X, VideoEffectsShaders.Y) 与 BeautyPrePass 的 Program(…, …)。
PROGRAM_CALL = re.compile(r'\b(?:createProgram|Program)\(\s*VideoEffectsShaders\.(\w+)\s*,\s*VideoEffectsShaders\.(\w+)\s*\)')
# 期望值：meta.uniforms.android 的约定（README「Uniform 换算」）由 end_to_end.geometry 各字段拼出。
GEOMETRY_UNIFORMS = {
    'uCheeks[0]': lambda g: g['cheeks'],
    'uEyes[0]': lambda g: g['eyes'][0:3] + [g['axis'][0]] + g['eyes'][3:6] + [g['axis'][1]],
    'uOval': lambda g: g['oval'],
    'uMouth': lambda g: g['mouth'],
    'uEyeMask[0]': lambda g: g['eyeMasks'],
}


class WiringError(Exception):
    pass


def kotlin_code(path):
    """读 Kotlin 原文并去掉注释（这些文件的字符串里没有 // 或 /*）。"""
    with open(path, encoding='utf-8') as handle:
        return re.sub(r'//[^\n]*|/\*.*?\*/', '', handle.read(), flags=re.S)


def program_pairs(code):
    """返回 (配对列表, 已解释的 VideoEffectsShaders 引用数)。

    认两种写法：两个参数都是 VideoEffectsShaders 常量的 createProgram / Program 调用；以及固定顶点的单参数
    辅助函数 fun f(x: String) = …createProgram(VideoEffectsShaders.V, x)… 与它的调用 f(VideoEffectsShaders.F)。"""
    pairs = PROGRAM_CALL.findall(code)
    explained = 2 * len(pairs)
    for helper, param, vertex in re.findall(
            r'\bfun (\w+)\((\w+): String\)[^\n]*\n?[^\n]*?createProgram\(\s*VideoEffectsShaders\.(\w+)\s*,\s*\2\s*\)', code):
        explained += 1
        for fragment in re.findall(r'(?<!fun )\b%s\(\s*VideoEffectsShaders\.(\w+)\s*\)' % helper, code):
            pairs.append((vertex, fragment))
            explained += 1
    return pairs, explained


def check_program_pairs(kotlin_dir, report):
    pairs, refs, explained = [], 0, 0
    for name in WIRING_FILES:
        code = kotlin_code(os.path.join(kotlin_dir, name))
        found_pairs, found_explained = program_pairs(code)
        pairs += found_pairs
        explained += found_explained
        refs += len(re.findall(r'\bVideoEffectsShaders\.\w+', code))
    found = sorted(set(pairs))
    expected = sorted({(vertex, fragment) for _, vertex, fragment in PROGRAMS})
    # 每个 VideoEffectsShaders 引用都必须落在一次可解析的配对里，换了写法也不会被漏掉。
    stray = refs - explained
    report.line(found == expected and stray == 0, '%s 的程序配对与门禁 PROGRAMS 一致%s' % (
        ' / '.join(WIRING_FILES), '' if found == expected and stray == 0 else
        '：Kotlin %s，门禁 %s，未解析的 VideoEffectsShaders 引用 %d 处' % (found, expected, stray)))


def arithmetic(expr, constants, i=None):
    """求 Kotlin 下标表达式（整数、i、+ - *、括号、FaceGeometrySmoother 常量）。"""
    text = re.sub(r'\b(?:FaceGeometrySmoother\.)?([A-Z][A-Z0-9_]*)\b',
                  lambda m: str(constants[m.group(1)]) if m.group(1) in constants else m.group(0), expr)
    if not re.fullmatch(r'[\d\s+\-*()i]+', text) or (i is None and 'i' in text):
        raise WiringError('无法求值的下标表达式：%s' % expr)
    return eval(text, {'__builtins__': {}}, {'i': i})  # noqa: S307 —— 上面已限定为整数算术


def geometry_packing(kotlin_dir, state):
    """按 VideoEffectsFilter.kt 的打包代码（glUniform4fv 与临时数组的循环）把平滑器状态拆成 uniform。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'VideoEffectsFilter.kt'))
    constants = {k: int(v) for k, v in re.findall(
        r'const val (\w+) = (\d+)\b', kotlin_code(os.path.join(kotlin_dir, 'FaceGeometrySmoother.kt')))}
    # glGetUniformLocation(id, "…") 或包一层的 uniform("…")。
    locations = dict(re.findall(r'(\w+) = (?:glGetUniformLocation\(\w+, |\w+\()"(\w+(?:\[0\])?)"\)', code))
    loops = re.findall(r'for \(i in 0 until (\d+)\) \{([^{}]*)\}', code)
    packed = {}
    for var, count, source, offset in re.findall(r'glUniform4fv\((\w+), (\d+), (\w+), (\d+)\)', code):
        if var not in locations:
            raise WiringError('glUniform4fv(%s, …) 找不到对应的 glGetUniformLocation' % var)
        size, offset = 4 * int(count), int(offset)
        if source == 'geometry':
            values = list(state)
        else:
            declared = re.search(r'\bval %s = FloatArray\((\d+)\)' % source, code)
            body = [b for _, b in loops if re.search(r'\b%s\b' % source, b)]
            if not declared or len(body) != 1:
                raise WiringError('临时数组 %s 的声明或填充循环无法解析' % source)
            values = [None] * int(declared.group(1))
            bound = int([n for n, b in loops if b == body[0]][0])
            statements = [s.strip() for s in re.split(r'[;\n]', body[0]) if s.strip()]
            for i in range(bound):
                for statement in statements:
                    copy = re.fullmatch(r'geometry\.copyInto\(%s, (.+?), (.+?), (.+?)\)' % source, statement)
                    single = re.fullmatch(r'%s\[(.+?)\] = geometry\[(.+?)\]' % source, statement)
                    if copy:
                        dest, start, end = (arithmetic(e, constants, i) for e in copy.groups())
                        values[dest:dest + end - start] = state[start:end]
                    elif single:
                        values[arithmetic(single.group(1), constants, i)] = state[arithmetic(single.group(2), constants, i)]
                    else:
                        raise WiringError('无法解析的打包语句：%s' % statement)
        chunk = values[offset:offset + size]
        if len(chunk) != size or any(v is None for v in chunk):
            raise WiringError('%s 打包越界或有未填充分量' % locations[var])
        packed[locations[var]] = chunk
    return packed


def check_geometry_packing(kotlin_dir, vectors, report):
    """返回 uniform 名 → 值（供 end_to_end 使用）；解析失败或与约定不符时返回 None。"""
    data = vectors['end_to_end']
    try:
        packed = geometry_packing(kotlin_dir, data['state'])
    except (WiringError, IndexError, KeyError, ValueError) as error:
        report.line(False, 'VideoEffectsFilter.kt 几何 uniform 打包无法解析（改了打包代码请同步 glsl_gate.py）：%s' % error)
        return None
    report.line(sorted(packed) == sorted(GEOMETRY_UNIFORMS), 'VideoEffectsFilter.kt 打包的几何 uniform：%s' % ', '.join(
        '%s[%d]' % (name, len(values) // 4) for name, values in sorted(packed.items())))
    for name, expected in GEOMETRY_UNIFORMS.items():
        want, got = expected(data['geometry']), packed.get(name)
        ok = got is not None and len(got) == len(want) and all(abs(a - b) <= 1e-9 for a, b in zip(got, want))
        report.line(ok, '打包 %s 等于 meta.uniforms.android 约定%s' % (name, '' if ok else '：得到 %s，应为 %s' % (got, want)))
    return packed if sorted(packed) == sorted(GEOMETRY_UNIFORMS) else None


def check_kernel(shaders, vectors, report):
    """BLUR_VERTEX 的偏移与 BLUR 的权重必须逐个等于 kernel.shaderPairs（参考模型的线性采样常量）。"""
    pairs = vectors['kernel']['shaderPairs']
    offsets = {(side, int(k)): (sign, float(value)) for side, k, sign, value in re.findall(
        r'\bv([AB])(\d) = aTextureCoord ([+-]) uDir \* ([0-9.]+);', shaders['BLUR_VERTEX'])}
    want_offsets = {(side, k): (sign, pairs[k]['offset']) for k in range(1, len(pairs))
                    for side, sign in (('A', '+'), ('B', '-'))}
    centre = [float(w) for w in re.findall(r'texture2D\(uSampler, vC\) \* ([0-9.]+)', shaders['BLUR'])]
    weights = {int(k): float(w) for k, w in re.findall(
        r'\(texture2D\(uSampler, vA(\d)\) \+ texture2D\(uSampler, vB\1\)\) \* ([0-9.]+)', shaders['BLUR'])}
    ok = (pairs[0]['offset'] == 0 and re.search(r'\bvC = aTextureCoord;', shaders['BLUR_VERTEX']) is not None and
          offsets == want_offsets and centre == [pairs[0]['weight']] and
          weights == {k: pairs[k]['weight'] for k in range(1, len(pairs))})
    report.line(ok, 'BLUR_VERTEX / BLUR 线性采样常量等于 kernel.shaderPairs%s' % (
        '' if ok else '：偏移 %s，中心 %s，成对权重 %s；应为 %s' % (sorted(offsets.items()), centre, weights, pairs)))
    # 头发加权路径 P2' 按权重预乘后自己展开 5 个抽头，权重必须与 blur5 相同。
    text = shaders['GUIDED_WEIGHT_BLUR_FRAGMENT']
    w_centre = [float(w) for w in re.findall(r'weighted\(vC\) \* ([0-9.]+)', text)]
    w_pairs = {int(k): float(w) for k, w in re.findall(r'\(weighted\(vA(\d)\) \+ weighted\(vB\1\)\) \* ([0-9.]+)', text)}
    w_ok = w_centre == centre and w_pairs == weights and len(w_pairs) == len(pairs) - 1
    report.line(w_ok, 'GUIDED_WEIGHT_BLUR_FRAGMENT 的抽头权重与 BLUR 的 blur5 相同：中心 %s、成对 %s' % (w_centre, w_pairs))


def processing(params):
    # VideoEffectsFilter.drawFilter 的 processing 判定（end_to_end 无变形、无美体 / 美妆 / LUT / 背景虚化、无负载降级），
    # 只保留共享向量里有的字段。drawFilter 的判定改了（例如把 skinTexture、radiance 或降级渐变计入）必须同步这里。
    return (params['smoothing'] > 0 or params['brightness'] > 0 or params['sharpness'] > 0 or
            params['whiten'] > 0 or params['rosy'] > 0 or params['saturation'] != 1 or
            params['filter'] != 'none' and params['filterIntensity'] > 0)


def end_to_end_cases(vectors):
    cases = []
    for case in vectors['end_to_end']['cases']:
        on = processing(case['params'])
        cases.append({'name': case['name'], 'params': case['params'], 'processing': on,
                      'presence': case['presence'] if on and case['useGeometry'] else 0.0,
                      'slim': 0.0, 'eye': 0.0})
    # processing = 0 时必须按原坐标严格直通：即使预处理、遮罩与塑形 uniform 都是非零值。
    loudest = vectors['end_to_end']['cases'][-1]['params']
    cases.append({'name': 'processing_off_with_everything', 'params': loudest, 'processing': False,
                  'prepass': True, 'presence': 1.0, 'slim': 0.12, 'eye': 0.25})
    return cases


def guided_constants(kotlin_dir, shaders, report):
    """从 BeautyQualityTuning.kt 读导向滤波能量系数、σ0 区间、P1 DOWN 抽头间距与 GuidedSpread.DEFAULT，不手抄；
    同时核对 GUIDED_DIFF_FRAGMENT 里的能量系数字面量与 ENERGY_SCALE 一致、GuidedFilterPrePass 的 P1 用的是
    DOWN_TAP_SCALE。解析失败返回 None。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'BeautyQualityTuning.kt'))
    found = dict(re.findall(r'const val (ENERGY_SCALE|SIGMA0_LO|SIGMA0_HI|DOWN_TAP_SCALE|DEFAULT) = ([0-9.]+)f', code))
    if sorted(found) != ['DEFAULT', 'DOWN_TAP_SCALE', 'ENERGY_SCALE', 'SIGMA0_HI', 'SIGMA0_LO']:
        report.line(False, 'BeautyQualityTuning.kt 的 ENERGY_SCALE / SIGMA0_LO / SIGMA0_HI / DOWN_TAP_SCALE / '
                           'GuidedSpread.DEFAULT 无法解析')
        return None
    guided = {'energy': float(found['ENERGY_SCALE']), 'lo': float(found['SIGMA0_LO']), 'hi': float(found['SIGMA0_HI']),
              'spread': float(found['DEFAULT']), 'downTap': float(found['DOWN_TAP_SCALE'])}
    literal = [re.findall(r'clamp\(length\(diff\) \* ([0-9.]+), 0\.0, 1\.0\)', shaders[name])
               for name in ('GUIDED_DIFF_FRAGMENT', 'GUIDED_WEIGHT_DIFF_FRAGMENT')]
    ok = all(len(found) == 1 and float(found[0]) == guided['energy'] for found in literal)
    report.line(ok, 'GUIDED_DIFF_FRAGMENT / GUIDED_WEIGHT_DIFF_FRAGMENT 能量系数 %s 等于 BeautyQualityTuning.ENERGY_SCALE %s' % (
        literal, guided['energy']))
    # 头发加权路径读皮肤遮罩 A 的换算必须与主合成的 nonHair 相同（权重与「哪里不磨皮」是同一张遮罩、同一条曲线）。
    mapping = 'clamp(texture2D(uWeight, %s).a * 1.04 - 0.02, 0.0, 1.0)'
    same = ('clamp(sem.a * 1.04 - 0.02, 0.0, 1.0)' in shaders['FRAGMENT_PRO'] and
            mapping % 'uv' in shaders['GUIDED_WEIGHT_BLUR_FRAGMENT'] and mapping % 'vC' in shaders['GUIDED_WEIGHT_COEFF_FRAGMENT'])
    report.line(same, 'GUIDED_WEIGHT_BLUR / GUIDED_WEIGHT_COEFF 读 A 的非头发换算与 FRAGMENT_PRO 的 nonHair 相同（a·1.04 − 0.02）')
    prepass = kotlin_code(os.path.join(kotlin_dir, 'GuidedFilterPrePass.kt'))
    report.line(re.search(r'BeautyQualityTuning\.guidedEps\(', prepass) is not None,
                'GuidedFilterPrePass.kt 的 uGuidedEps 取自 BeautyQualityTuning.guidedEps')
    # 门禁按 DOWN_TAP_SCALE 跑导向滤波探针，生产 P1 若另写字面量（例如退回 1f / width），探针就不再代表生产。
    tap = re.search(r'glUniform2f\(it\.texel, BeautyQualityTuning\.DOWN_TAP_SCALE / width, '
                    r'BeautyQualityTuning\.DOWN_TAP_SCALE / height\)', prepass) is not None
    report.line(tap, 'GuidedFilterPrePass.kt 的 P1 uTexel = BeautyQualityTuning.DOWN_TAP_SCALE / (width, height)'
                     '（DOWN_TAP_SCALE = %s）' % guided['downTap'])
    # 头发加权路径的接线（gate.js 的 guidedPrepassWeighted 照此复刻）：权重纹理绑单元 2，P2' / P5' 的 uWeight 取 2；两条路径的 P1
    # 都用 DOWN_TAP_SCALE；不传权重（0）时走原路径。
    weighted = (len(re.findall(r'glUniform2f\(it\.texel, BeautyQualityTuning\.DOWN_TAP_SCALE / width, '
                               r'BeautyQualityTuning\.DOWN_TAP_SCALE / height\)', prepass)) == 2 and
                re.search(r'glActiveTexture\(GL_TEXTURE2\)\s*\n\s*glBindTexture\(GL_TEXTURE_2D, hairWeight\)', prepass) is not None and
                len(re.findall(r'glUniform1i\(it\.weight, 2\)', prepass)) == 2 and
                re.search(r'\bhairWeight: Int = 0\b', prepass) is not None and
                re.search(r'if \(hairWeight != 0 && weightHealthy\)', prepass) is not None)
    report.line(weighted, 'GuidedFilterPrePass.kt 的头发加权路径：权重纹理绑单元 2、P2\'/P5\' 的 uWeight = 2、P1 同用 DOWN_TAP_SCALE、'
                          'hairWeight 缺省 0 走原路径=%s' % weighted)
    # 着色器里 coring 门限的下限（uGrade 未设置时也生效）等于 BeautyQualityTuning.CORE_BASE。
    base = re.findall(r'const val CORE_BASE = ([0-9.]+)f', code)
    floor = re.findall(r'float core = max\(grade\.y, ([0-9.]+)\);', shaders['COMPOSITE_PRO'])
    floor_ok = len(base) == 1 and len(floor) == 1 and float(base[0]) == float(floor[0])
    report.line(floor_ok, 'COMPOSITE_PRO 的 coring 门限下限 %s 等于 BeautyQualityTuning.CORE_BASE %s' % (floor, base))
    return guided if ok and floor_ok else None


def check_grade_wiring(kotlin_dir, report):
    """主合成新增的 uLut（纹理单元 3）与 uGrade（LUT 强度、coring 门限、回退预处理标记）必须在 VideoEffectsFilter.kt 接上。
    门禁在 gate.js 里按同样的约定设置这两个 uniform，Kotlin 改了接线写法请同步这里。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'VideoEffectsFilter.kt'))
    lut = re.findall(r'(\w+) = (?:glGetUniformLocation\(\w+, |\w+\()"uLut"\)', code)
    grade = re.findall(r'(\w+) = (?:glGetUniformLocation\(\w+, |\w+\()"uGrade"\)', code)
    report.line(len(lut) == 1 and len(grade) == 1,
                'VideoEffectsFilter.kt 取 uLut / uGrade 的 location：%s / %s' % (lut or '缺少', grade or '缺少'))
    if len(lut) != 1 or len(grade) != 1:
        return
    unit = re.search(r'glUniform1i\(%s, 3\)' % lut[0], code) is not None
    bound = len(re.findall(r'glActiveTexture\(GL_TEXTURE3\)', code)) >= 2  # 合成前绑定 + disableResources 解绑
    report.line(unit and bound, 'uLut 接纹理单元 3（glUniform1i(%s, 3)=%s，GL_TEXTURE3 绑定与解绑=%s）' % (lut[0], unit, bound))
    packed = re.search(r'glUniform4f\(%s,' % grade[0], code) is not None
    no_array = re.search(r'glUniform4fv\(%s,' % grade[0], code) is None
    core = re.search(r'BeautyQualityTuning\.coreThreshold\(', code) is not None
    report.line(packed and no_array and core, 'uGrade 用 glUniform4f 设置（不新增 glUniform4fv）且 coring 门限取自 '
                'BeautyQualityTuning.coreThreshold：glUniform4f=%s，无 4fv=%s，coreThreshold=%s' % (packed, no_array, core))
    texture = re.search(r'BitrateEffectsTuning\.texture\(', code) is not None
    report.line(texture, 'uParams.w（质感回填）经 BitrateEffectsTuning.texture 随码率压力收敛=%s' % texture)


def bracketed(code, start):
    """code[start] 是 '(' 或 '{'：返回到与它配对的右括号为止的内部文本（不含两端），不配对返回 None。"""
    opening = code[start]
    closing = {'(': ')', '{': '}'}[opening]
    depth = 0
    for i in range(start, len(code)):
        if code[i] == opening:
            depth += 1
        elif code[i] == closing:
            depth -= 1
            if depth == 0:
                return code[start + 1:i]
    return None


def top_level_args(text):
    """按顶层逗号拆实参（括号内的逗号不拆）。"""
    args, depth, current = [], 0, ''
    for c in text:
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        if c == ',' and depth == 0:
            args.append(current.strip())
            current = ''
        else:
            current += c
    args.append(current.strip())
    return args


def call_args(code, pattern):
    """pattern 匹配到 '(' 为止的每一次调用 → 实参列表。"""
    return [top_level_args(bracketed(code, m.end() - 1) or '') for m in re.finditer(pattern, code)]


def enclosing_if_conditions(code, pos):
    """pos 处由内向外各层代码块中，以 `if (…) {` 开头的那些块的条件（else 块、try 块、lambda 等不算）。"""
    conditions, depth = [], 0
    for i in range(pos - 1, -1, -1):
        if code[i] == '}':
            depth += 1
        elif code[i] == '{':
            if depth > 0:
                depth -= 1
                continue
            header = code[max(code.rfind(ch, 0, i) for ch in '{};') + 1:i]
            at = header.rfind('if (')
            if at >= 0:
                condition = bracketed(header, at + 3)
                if condition is not None and not header[at + 3 + len(condition) + 2:].strip():
                    conditions.append(condition)
    return conditions


def function_body(code, name):
    """`fun name(…)` 的块体（花括号内文本，不含两端）；没有或不是块体返回 None。"""
    m = re.search(r'\bfun %s\(' % re.escape(name), code)
    if not m:
        return None
    params = bracketed(code, m.end() - 1)
    if params is None:
        return None
    head = re.compile(r'\s*(?::\s*[\w<>?,. ]+?)?\s*\{').match(code, m.end() + len(params) + 1)
    return bracketed(code, head.end() - 1) if head else None


def int_constant(code, name):
    """`const val name = 整数` 的值；没有返回 None。"""
    m = re.search(r'\bconst val %s = (\d+)\b' % re.escape(name), code)
    return int(m.group(1)) if m else None


def index_value(expr, constants):
    """下标表达式（整数、+ -、`X.NAME` 或 `NAME` 常量）求值；不认得的名字抛 WiringError。"""
    def name(m):
        if m.group(1) not in constants:
            raise WiringError('下标表达式里不认得的常量：%s' % m.group(0))
        return str(constants[m.group(1)])
    text = re.sub(r'\b(?:\w+\.)?([A-Z][A-Z0-9_]*)\b', name, expr)
    if not re.fullmatch(r'[\d\s+\-()]+', text):
        raise WiringError('无法求值的下标表达式：%s' % expr)
    return eval(text, {'__builtins__': {}}, {})  # noqa: S307 —— 上面已限定为整数加减


def pack_hair_oval(body, state, constants):
    """按 packHairOval 的语句把几何状态拆成 8 个数（'aspect' 记为字符串）；有不认得的语句抛 WiringError。"""
    values = [None] * 8
    for statement in (s.strip() for s in re.split(r'[;\n]', body or '')):
        if not statement or statement == 'return out':
            continue
        copy = re.fullmatch(r'state\.copyInto\(out, (.+?), (.+?), (.+?)\)', statement)
        single = re.fullmatch(r'out\[(\d+)\] = state\[(.+?)\]', statement)
        literal = re.fullmatch(r'out\[(\d+)\] = (aspect|0f)', statement)
        if copy:
            dest, start, end = (index_value(e, constants) for e in copy.groups())
            values[dest:dest + end - start] = state[start:end]
        elif single:
            values[int(single.group(1))] = state[index_value(single.group(2), constants)]
        elif literal:
            values[int(literal.group(1))] = 'aspect' if literal.group(2) == 'aspect' else 0.0
        else:
            raise WiringError('packHairOval 里无法解析的语句：%s' % statement)
    if len(values) != 8 or any(v is None for v in values):
        raise WiringError('packHairOval 打包越界或有未填充分量')
    return values


def same_values(got, want):
    return len(got) == len(want) and all(
        (a == b) if isinstance(a, str) or isinstance(b, str) else abs(a - b) <= 1e-9 for a, b in zip(got, want))


def hair_wiring_lines(code, context):
    """VideoEffectsFilter.kt（已去注释）的头发接线核对：返回 [(键, 是否通过, 说明)]，见 check_hair_wiring。"""
    lines = []

    def line(key, ok, text):
        lines.append((key, bool(ok), text))

    # presence：uFrame.w 经 SkinMaskPass.framePresence(on, polyMask, presence)。
    frame = re.findall(r'(\w+) = (?:glGetUniformLocation\(\w+, |\w+\()"uFrame"\)', code)
    presence = call_args(code, r'\bSkinMaskPass\.framePresence\(')
    named = re.findall(r'\bval (\w+) = SkinMaskPass\.framePresence\(', code)
    uploaded = [args for args in call_args(code, r'\bglUniform4f\(') if len(frame) == 1 and args[0] == frame[0]]
    via = (len(presence) == 1 and len(presence[0]) == 3 and presence[0][1] == 'polyMask' and len(named) == 1 and
           len(uploaded) == 1 and uploaded[0][-1] == named[0])
    literal = re.search(r'\+\s*4(?:\.0)?f\b', code) is not None
    assigned = sorted(a.strip() for a in re.findall(r'\bpolyMask = ([^\n]+)', code))
    after_render = re.search(r'\bskinMaskTexture = skinMaskPass\.render\([^()]*\)\s*\n\s*polyMask = skinMaskTexture != 0\s*\n',
                             code) is not None
    poly = assigned == ['false', 'skinMaskTexture != 0'] and after_render
    line('presence', via and not literal and poly,
         'VideoEffectsFilter.kt 的 uFrame.w 经 SkinMaskPass.framePresence(on, polyMask, presence)=%s（实参 %s），'
         '不再自写 presence + 4f=%s，polyMask 只按 skinMaskPass.render 的结果置位=%s（赋值 %s）' % (
             via, presence, not literal, poly, assigned))
    # track：头发完成时间不并入画面延后。
    track = re.search(r'\bval trackCompleted = when \{', code)
    track_body = bracketed(code, track.end() - 1) if track else None
    tracking = re.search(r'\bval tracking = ([^\n]+)', code)
    push = call_args(code, r'\bframeDelay\.push\(')
    apart = (track_body is not None and 'hair' not in track_body.lower() and tracking is not None and
             'hair' not in tracking.group(1).lower() and len(push) >= 1 and
             all('trackCompleted' in args and not any('hair' in a.lower() for a in args) for args in push))
    line('track', apart, '头发完成时间不并入画面延后：trackCompleted / tracking / frameDelay.push 里没有 hair=%s' % apart)
    # semantic：A 道只在 hairShown > 0f 时画。
    calls = [m.start() for m in re.finditer(r'\bsemanticPass\.render\(', code)]
    conditions = enclosing_if_conditions(code, calls[0]) if len(calls) == 1 else []
    guarded = len(calls) == 1 and any(
        '||' not in c and 'hairShown > 0f' in (p.strip() for p in c.split('&&')) for c in conditions)
    line('semantic', guarded, 'semanticPass.render( %d 处，处在 hairShown > 0f 的条件内=%s（外层 if 条件：%s）' % (
        len(calls), guarded, conditions))

    # upload：上传成功后紧跟 hairGate.uploaded（同一个 try 块里，先上传后记采样时间），实参与 hairGate.update 对应。
    update = re.findall(r'\bval (\w+) = hairGate\.update\(', code)
    update_args = call_args(code, r'\bhairGate\.update\(')
    pair = re.findall(r'\btry\s*\{\s*hairTexture\.upload\((\w+)\)\s*(?:;|\n)\s*hairGate\.uploaded\((\w+)\)\s*\}', code)
    counts = (len(re.findall(r'\bhairTexture\.upload\(', code)), len(re.findall(r'\bhairGate\.uploaded\(', code)))
    ok = (len(update) == 1 and len(update_args) == 1 and len(update_args[0]) >= 3 and counts == (1, 1) and len(pair) == 1 and
          pair[0] == (update[0], update_args[0][2]))
    line('upload', ok, 'hairTexture.upload 之后紧跟 hairGate.uploaded（同一 try 块，实参为 hairGate.update 的返回值与遮罩采样时间）'
         '=%s（upload / uploaded 各 %d / %d 处，配对 %s，update %s）' % (ok, counts[0], counts[1], pair, update_args))

    # affine：运动补偿两个时刻的来源——遮罩时刻取已上传遮罩（hairGate.stamp），显示时刻取本帧 at(displayTime) 的几何与同一个
    # displayTime；输出就是 A 道与合并道用的那块仿射。
    compensate = call_args(code, r'\bHairMaskGate\.compensate\(')
    at = re.findall(r'\bval (\w+) = timeline\.at\((\w+)\)', code)
    semantic = call_args(code, r'\bsemanticPass\.render\(')
    holder = call_args(code, r'\bHairBackgroundInput\(')
    args = compensate[0] if len(compensate) == 1 else []
    ok = (len(args) == 8 and len(at) == 1 and args[:4] == ['timeline', 'hairGate.stamp', at[0][0], at[0][1]] and
          len(semantic) == 1 and len(semantic[0]) >= 4 and semantic[0][3] == args[7] and
          len(holder) == 1 and len(holder[0]) == 4 and holder[0][2] == args[7] and
          re.search(r'\bHairMaskTuning\.affine\(', function_body(code, 'drawFilter') or 'HairMaskTuning.affine(') is None)
    line('affine', ok, '运动补偿 HairMaskGate.compensate(timeline, hairGate.stamp, <at(displayTime) 的几何>, displayTime, …) '
         '且输出即 A 道 / 合并道的仿射=%s（实参 %s，at %s，A 道仿射 %s，合并道 %s）' % (
             ok, args, at, semantic[0][3] if semantic and len(semantic[0]) >= 4 else None, holder))

    # oval：packHairOval 的打包与 compensate 的显示位姿区间，都按 end_to_end 向量解析，必须等于 GEOMETRY_UNIFORMS['uOval'] 后接轴向
    # （与 FaceTimeline.poseAt 同一布局），不写死下标。
    state, geometry, constants = context['state'], context['geometry'], dict(context['constants'])
    want = GEOMETRY_UNIFORMS['uOval'](geometry) + list(geometry['axis'])
    try:
        constants['POSE_INDEX'] = int_constant(code, 'POSE_INDEX')
        packed = pack_hair_oval(function_body(code, 'packHairOval'), state, constants)
        body = function_body(code, 'compensate') or ''
        copies = re.findall(r'\b\w+\.copyInto\((\w+), 0, ([^,()]+), ([^,()]+)\)', body)
        if len(copies) != 1:
            raise WiringError('compensate 里显示位姿的 copyInto 有 %d 处' % len(copies))
        pose = state[index_value(copies[0][1], constants):index_value(copies[0][2], constants)]
        ok = (same_values(packed, want + ['aspect', 0.0]) and same_values(pose, want) and
              constants['POSE_INDEX'] == constants.get('OVAL_INDEX'))
        detail = 'packHairOval %s，显示位姿 %s，应为 %s（+ aspect, 0）；POSE_INDEX %s，FaceTimeline.OVAL_INDEX %s' % (
            packed, pose, want, constants['POSE_INDEX'], constants.get('OVAL_INDEX'))
    except (WiringError, IndexError, TypeError, ValueError) as error:
        ok, detail = False, '无法解析：%s' % error
    line('oval', ok, 'A 道 / 合并道的脸椭圆与运动补偿的显示位姿按 end_to_end 向量等于 uOval 后接轴向=%s（%s）' % (ok, detail))

    # enable：头发分割的启停接 hairEnabled；画面里近期有脸（timeline.latest）在 TrackingGrace 之外并进启停，「人脸近期可见」进强度，
    # 且接丢脸信号 timeline.missSince（只在检测器确实报告无脸时渐出，人脸推理慢时不随采样节奏锯齿）与人脸任务在途信号（停摆兜底
    # 只在没有人脸任务在途时生效）：初值 false 的变量，只在人脸跟踪块里、本帧人脸快照之后取 snapshot.inFlight，并且紧跟在
    # faceCompleted = snapshot.completed 之后、同一层（挪进切代际的分支里就只在换代那一帧取，运行时退回锯齿）。
    block = re.search(r'\bhairSegmenter\?\.let \{ detector ->', code)
    block_body = bracketed(code, code.index('{', block.start())) if block else None
    enable = call_args(block_body or '', r'\bdetector\.state\.enable\(')
    gate = call_args(code, r'\bval hairEnabled = HairMaskGate\.enabled\(')
    faces = [bracketed(code, code.index('{', m.start())) for m in re.finditer(r'\btracker\?\.let \{ detector ->', code)]
    faces = [b for b in faces if b is not None and re.search(r'\bval snapshot = detector\.state\.snapshot\(now\)', b)]
    args = update_args[0] if len(update_args) == 1 and len(update_args[0]) == 9 else [None] * 9
    flight = args[7] if args[7] and re.fullmatch(r'[A-Za-z_]\w*', args[7]) else None
    assigns = [a.strip() for a in re.findall(r'(?<![\w.])(?:var |val )?%s = ([^\n;]+)' % flight, code)] if flight else []
    flight_ok = (flight is not None and re.search(r'\bvar %s = false\n' % flight, code) is not None and
                 assigns == ['false', 'snapshot.inFlight'] and len(faces) == 1 and
                 re.search(r'\bval snapshot = detector\.state\.snapshot\(now\)\n[\s\S]*\n(\s*)faceCompleted = snapshot\.completed\n'
                           r'\1%s = snapshot\.inFlight\n' % flight, faces[0]) is not None)
    ok = (len(enable) == 1 and len(enable[0]) == 1 and 'hairEnabled' in (p.strip() for p in enable[0][0].split('&&')) and
          '||' not in enable[0][0] and len(gate) == 1 and len(gate[0]) == 6 and gate[0][4:] == ['timeline.latest', 'now'] and
          args[5:7] == ['timeline.latest', 'timeline.missSince'] and args[8] == 'now' and flight_ok)
    line('enable', ok, '头发分割启停 detector.state.enable(…) 含 hairEnabled，HairMaskGate.enabled 在宽限之外接 timeline.latest（无脸停推理），'
         'hairGate.update 接 timeline.latest、timeline.missSince（人脸近期可见，确认丢脸才渐出）与人脸跟踪块里本帧快照之后取的 '
         'snapshot.inFlight（停摆兜底只在不在途时生效）=%s（enable %s，enabled %s，update %s，在途信号 %s 的赋值 %s）' % (
             ok, enable, gate, update_args, flight, assigns))

    # cpu：两个 CPU 护栏——开虚化时间隔下限（segRunning = segEnabled 接进 interval）与 p50 超限停用；间隔用于取票与延迟回读。
    p50 = re.findall(r'\bval (\w+) = detector\.inferenceP50Ms\(\)', code)
    interval = call_args(code, r'\bHairMaskTuning\.interval\(')
    named_interval = re.findall(r'\bval (\w+) = HairMaskTuning\.interval\(', code)
    stop = re.findall(r'\bif \(!HairMaskTuning\.healthy\((\w+)\)\) failHair\(\)', code)
    due = re.findall(r'\bstagedHair = detector\.state\.due\(now, width, height, (\w+)\)', code)
    begin = re.findall(r'\bhairTicket = detector\.state\.begin\(now, width, height, (\w+)\)', code)
    ok = (re.search(r'\bsegRunning = segEnabled\b', code) is not None and len(p50) == 1 and len(interval) == 1 and
          interval[0] == [p50[0], 'segRunning'] and len(named_interval) == 1 and
          len(re.findall(r'\bHairMaskTuning\.healthy\(', code)) == 1 and stop == [p50[0]] and
          due == named_interval and begin == named_interval)
    line('cpu', ok, 'CPU 护栏：segRunning = segEnabled 接进 HairMaskTuning.interval(p50, segRunning)，取票与延迟回读都用这个间隔，'
         'if (!HairMaskTuning.healthy(p50)) failHair() 恰一处=%s（interval %s，healthy %s，due %s，begin %s）' % (
             ok, interval, stop, due, begin))

    # engine：外部引擎路径停用头发分割（引擎期间不推理、回到自研管线时按代际复位、从 0 渐入）。
    engine = function_body(code, 'drawEngine')
    ok = engine is not None and re.search(r'\bhairSegmenter\?\.state\?\.enable\(false\)', engine) is not None
    line('engine', ok, 'drawEngine 停用头发分割（hairSegmenter?.state?.enable(false)）=%s' % ok)

    # reset：切镜头（resetFaceTracking）一并作废头发分割的历史（NOTICE 隐私段）。
    reset = function_body(code, 'resetFaceTracking')
    ok = reset is not None and re.search(r'\bhairSegmenter\?\.state\?\.reset\(\)', reset) is not None
    line('reset', ok, 'resetFaceTracking 一并复位头发分割（hairSegmenter?.state?.reset()）=%s' % ok)

    # weight：导向滤波的头发加权只接「本帧 A 道画成功」的那张皮肤遮罩（主合成 uSkinMask 同一张），否则为 0（原路径，逐字节等于第一批）。
    guided = call_args(code, r'\bguidedFilterPass\.render\(')
    weight = guided[0][5] if len(guided) == 1 and len(guided[0]) == 6 and re.fullmatch(r'\w+', guided[0][5]) else None
    bound = re.findall(r'\bglActiveTexture\(GL_TEXTURE2\)\s*\n\s*glBindTexture\(GL_TEXTURE_2D, if \((\w+) != 0\) \1 else fallbackTexture\)',
                       code)
    assigns = re.findall(r'(?<![\w.])(?:var |val )?%s = ([^\n;]+)' % weight, code) if weight else []
    after = None
    if len(calls) == 1 and weight is not None:
        opening = calls[0] + len('semanticPass.render')
        inner = bracketed(code, opening)
        if inner is not None:
            after = re.compile(r'\s*\n\s*%s = (\w+)\s*\n' % weight).match(code, opening + len(inner) + 2)
    ok = (weight is not None and len(bound) == 1 and re.search(r'\bvar %s = 0\n' % weight, code) is not None and
          [a.strip() for a in assigns] == ['0', bound[0]] and after is not None and after.group(1) == bound[0])
    line('weight', ok, '导向滤波头发加权 guidedFilterPass.render(…, %s) 只在 semanticPass.render 成功后取 %s（主合成 uSkinMask），'
         '其余为 0=%s（赋值 %s）' % (weight, bound, ok, assigns))

    # face：背景虚化的人脸椭圆（头部先验，第一批 D3 的细节保留）只看有没有脸、不看头发；头发合并道只在强度 > 0 且有脸时传，
    # 两者同一块椭圆缓冲（变形前 rawGeometry）。细节保留的强度系数接本帧已推进的 hairGate.faceKeep（人脸近期可见，不随头发开关变化）。
    background = call_args(code, r'\bbgPass\.render\(')
    bargs = background[0] if len(background) == 1 and len(background[0]) == 7 else [None] * 7
    gate_at = [m.start() for m in re.finditer(r'\bhairGate\.update\(', code)]
    render_at = [m.start() for m in re.finditer(r'\bbgPass\.render\(', code)]
    weight = bargs[6] == 'hairGate.faceKeep' and len(gate_at) == 1 and len(render_at) == 1 and gate_at[0] < render_at[0]
    face = re.findall(r'\bval %s = if \(([^()\n]*)\) packHairOval\((\w+), aspect, (\w+)\) else null' % bargs[5], code) \
        if bargs[5] else []
    hair = re.findall(r'\bval %s = if \(([^\n]*)\) hairBackground\.also' % bargs[4], code) if bargs[4] else []
    ok = (len(face) == 1 and len(at) == 1 and face[0][0] == '%s != null' % at[0][0] and face[0][1] == at[0][0] and
          len(holder) == 1 and len(holder[0]) == 4 and holder[0][3] == face[0][2] and len(hair) == 1 and
          {'hairLevel > 0f', '%s != null' % bargs[5]} <= {p.strip() for p in hair[0].split('&&')} and '||' not in hair[0] and weight)
    line('face', ok, 'bgPass.render(…, 头发, 人脸椭圆, 细节保留系数)：人脸椭圆只在 rawGeometry 非空时打包、与头发无关，头发合并道要 '
         'hairLevel > 0f 且有脸，共用一块椭圆缓冲，系数为本帧 hairGate.update 之后的 hairGate.faceKeep=%s（实参 %s，人脸 %s，'
         '头发条件 %s）' % (ok, bargs, face, hair))

    # neck：A 通道强度按天鹅颈生效程度（随人体存在度）渐变。
    shown = call_args(code, r'\bhairGate\.shown\(')
    ok = shown == [['HairMaskGate.neckWeight(neckLifted, bodyPresence)', 'engine = false']]
    line('neck', ok, 'A 通道强度 hairGate.shown(HairMaskGate.neckWeight(neckLifted, bodyPresence), engine = false)=%s（%s）' % (
        ok, shown))
    return lines


# 门禁自带的接线变异：(名字, 应拦下它的核对, 原文, 改成)。原文按去注释后的 VideoEffectsFilter.kt 匹配，必须恰好出现一次；
# 改了生产写法时请同步这里（变异锚点找不到也算 FAIL）。
HAIR_WIRING_MUTANTS = [
    ('framePresence 按纹理非 0', 'presence', 'SkinMaskPass.framePresence(on, polyMask, presence)',
     'SkinMaskPass.framePresence(on, skinMaskTexture != 0, presence)'),
    ('头发完成时间并入 trackCompleted', 'track', '            faceTracking -> faceCompleted\n',
     '            faceTracking -> minOf(faceCompleted, hairSegmenter?.state?.snapshot(now)?.completed ?: faceCompleted)\n'),
    ('A 道不看强度', 'semantic', 'if (hairShown > 0f && semanticHealthy && geometry != null) {',
     'if (semanticHealthy && geometry != null) {'),
    ('上传后不记采样时间（第二批整体失效）', 'upload', 'try { hairTexture.upload(hairUpload); hairGate.uploaded(hairMaskStamp) }',
     'try { hairTexture.upload(hairUpload) }'),
    ('先记采样时间再上传', 'upload', 'try { hairTexture.upload(hairUpload); hairGate.uploaded(hairMaskStamp) }',
     'try { hairGate.uploaded(hairMaskStamp); hairTexture.upload(hairUpload) }'),
    ('遮罩时刻取本帧快照而不是已上传的遮罩', 'affine', 'HairMaskGate.compensate(timeline, hairGate.stamp,',
     'HairMaskGate.compensate(timeline, hairMaskStamp,'),
    ('显示位姿退回 poseAt(displayTime)', 'affine',
     'HairMaskGate.compensate(timeline, hairGate.stamp, rawGeometry, displayTime, aspect, hairPoseMask, hairPoseDisplay,\n'
     '                hairAffine)',
     'HairMaskTuning.affine(if (timeline.poseAt(hairGate.stamp, hairPoseMask)) hairPoseMask else null,\n'
     '                if (timeline.poseAt(displayTime, hairPoseDisplay)) hairPoseDisplay else null, aspect, hairAffine)'),
    ('脸椭圆轴向取错分量', 'oval', 'out[4] = state[FaceGeometrySmoother.AXIS_INDEX]',
     'out[4] = state[FaceGeometrySmoother.AXIS_INDEX + 1]'),
    ('显示位姿取错区间', 'oval', 'const val POSE_INDEX = 22', 'const val POSE_INDEX = 28'),
    ('启停不看 hairEnabled', 'enable', 'detector.state.enable(hairEnabled && hairHealthy)', 'detector.state.enable(hairHealthy)'),
    ('无脸照常推理', 'enable', 'tier, fades.body,\n                timeline.latest, now)', 'tier, fades.body,\n                now, now)'),
    ('强度不乘人脸近期可见', 'enable', 'displayTime, fades.body,\n            timeline.latest, timeline.missSince, faceInFlight, now)',
     'displayTime, fades.body,\n            now, timeline.missSince, faceInFlight, now)'),
    ('人脸近期可见不接丢脸信号（只看样本年龄）', 'enable', 'timeline.latest, timeline.missSince, faceInFlight, now)',
     'timeline.latest, Long.MIN_VALUE, faceInFlight, now)'),
    ('人脸近期可见不接在途信号（推理慢时停摆兜底随采样节奏锯齿）', 'enable', 'timeline.missSince, faceInFlight, now)',
     'timeline.missSince, false, now)'),
    ('在途信号不取自人脸快照（恒为在途，停摆兜底失效）', 'enable', 'faceInFlight = snapshot.inFlight\n', 'faceInFlight = true\n'),
    ('在途信号只在切代际时取（运行时退回锯齿）', 'enable',
     ('faceCompleted = snapshot.completed\n                faceInFlight = snapshot.inFlight\n',
      '                    lastCompletedSeen = Long.MIN_VALUE\n                }\n                snapshot.geometry?.let {'),
     ('faceCompleted = snapshot.completed\n',
      '                    lastCompletedSeen = Long.MIN_VALUE\n                    faceInFlight = snapshot.inFlight\n'
      '                }\n                snapshot.geometry?.let {')),
    ('开虚化不提间隔下限', 'cpu', 'HairMaskTuning.interval(p50, segRunning)', 'HairMaskTuning.interval(p50, false)'),
    ('p50 超限不停用', 'cpu', 'if (!HairMaskTuning.healthy(p50)) failHair()', ''),
    ('segRunning 不接人像分割', 'cpu', 'segRunning = segEnabled', 'segRunning = false'),
    ('外部引擎不停头发分割', 'engine', 'hairSegmenter?.state?.enable(false) \n        setDeferred(false); clearStaged()',
     'setDeferred(false); clearStaged()'),
    ('切镜头不复位头发分割', 'reset', '; hairSegmenter?.state?.reset() }', ' }'),
    ('导向滤波不接头发加权', 'weight', 'value.smoothing, guidedSpread.value, hairWeight)', 'value.smoothing, guidedSpread.value)'),
    ('导向滤波在 A 道之前就取头发权重', 'weight', 'var hairWeight = 0\n', 'var hairWeight = skinMaskTexture\n'),
    ('人脸椭圆只在有头发时传', 'face', 'val face = if (rawGeometry != null) packHairOval(',
     'val face = if (hairLevel > 0f && rawGeometry != null) packHairOval('),
    ('背景虚化不传人脸椭圆', 'face', 'source = bgPass.render(source, width, height, background, hair, face, hairGate.faceKeep)',
     'source = bgPass.render(source, width, height, background, hair)'),
    ('细节保留不随人脸近期可见渐变（有脸即满）', 'face', 'background, hair, face, hairGate.faceKeep)', 'background, hair, face)'),
    ('细节保留系数取头发强度（头发关闭时 D3 失效）', 'face', 'background, hair, face, hairGate.faceKeep)',
     'background, hair, face, hairLevel)'),
    ('天鹅颈硬开关', 'neck', 'HairMaskGate.neckWeight(neckLifted, bodyPresence)', 'HairMaskGate.neckWeight(neckLifted, 1f)'),
]


def check_hair_wiring(kotlin_dir, vectors, report):
    """头发遮罩（2026-11-21 第二批，计划 §4.4；评审修复 #12 / #14 / #17 / #21 与 SH 组新接口）在 VideoEffectsFilter.kt 的接线，静态核对：
    - presence：uFrame.w 取多边形遮罩分支（presence + 4f）必须经 SkinMaskPass.framePresence，第 2 个实参是 polyMask，且 polyMask 只在
      skinMaskPass.render 之后按它的返回值置位。不能按皮肤遮罩纹理非 0 判断：无脸或多边形遮罩故障时，renderEmpty 也会出一张只承载
      头发 A 通道的纹理（RGB 为 0），椭圆回退路径会读到 face = 0，磨皮就没了。
    - track：头发完成时间不并入画面延后：trackCompleted、tracking 与 frameDelay.push 的实参里都不出现 hair（推理慢不拉长画面延后）。
    - semantic：semanticPass.render( 只有一处，且处在条件含 `hairShown > 0f` 合取项（不含 ||）的 if 里：强度为 0 时不跑 A 道，A 恒为 1，
      画面与第一批逐字节相同。
    - upload：hairTexture.upload 与 hairGate.uploaded 在同一个 try 块里、先上传后记采样时间，实参是 hairGate.update 的返回值与遮罩采样
      时间。漏掉 uploaded 时 hairGate.stamp 恒为 MIN、hairLevel 恒为 0，第二批在生产中整体不生效，而联合测试是手抄的数据流测不到。
    - affine：运动补偿只经 HairMaskGate.compensate（JVM 有用例），遮罩时刻取已上传遮罩的 hairGate.stamp，显示时刻取本帧
      timeline.at(displayTime) 的几何与同一个 displayTime；输出就是 A 道与合并道（HairBackgroundInput）用的那块仿射。
    - oval：packHairOval 的打包与 compensate 里显示位姿的 copyInto 区间，按 end_to_end 向量求值后必须等于 GEOMETRY_UNIFORMS['uOval']
      后接轴向（与主合成 uOval、FaceTimeline.poseAt 同一布局；POSE_INDEX 等于 FaceTimeline 的 OVAL_INDEX），不写死下标。
    - enable：头发分割的启停 detector.state.enable(…) 含 hairEnabled；HairMaskGate.enabled 的「画面里近期有脸」接 timeline.latest、
      在 TrackingGrace 之外（无脸时停推理，不空跑 3 s）；hairGate.update 接 timeline.latest 与 timeline.missSince（「人脸近期可见」
      只在检测器确实报告无脸时渐出、几何清空前到 0；只看样本年龄时人脸走 CPU、推理慢的机型上强度随采样节奏锯齿），第 8 个实参是
      人脸任务在途信号：初值 false 的变量，只在人脸跟踪块（tracker?.let { detector -> … }）里、本帧 detector.state.snapshot(now)
      之后取 snapshot.inFlight（与并入时间线的结果同一次加锁；停摆兜底只在不在途时生效，推理慢时 faceKeep 与 β 不随采样节奏锯齿）。
    - cpu：两个 CPU 护栏（治理器看不到 worker CPU）——1c 段 segRunning = segEnabled，HairMaskTuning.interval 恰一处、实参为
      (p50, segRunning)，取票与延迟回读都用它；if (!HairMaskTuning.healthy(p50)) failHair() 恰一处，p50 取自 inferenceP50Ms。
    - engine：drawEngine 体内有 hairSegmenter?.state?.enable(false)（联合测试的引擎分支是手抄的，守不住生产这一行）。
    - reset：resetFaceTracking 体内有 hairSegmenter?.state?.reset()（切镜头作废旧镜头的头发遮罩）。
    - weight：guidedFilterPass.render 的头发加权实参是一个初值 0 的变量，只在 semanticPass.render 成功返回后的下一句取主合成
      uSkinMask 那张皮肤遮罩；头发关闭、遮罩缺失或 A 道异常时为 0，导向滤波走原路径（逐字节等于第一批）。
    - face：bgPass.render 的人脸椭圆只在 rawGeometry 非空时打包、条件里没有头发（第一批 D3 的头部先验不随头发开关变化），头发合并道
      要 hairLevel > 0f 且有脸，两者共用一块椭圆缓冲；细节保留系数（第 7 个实参）是本帧 hairGate.update 之后的 hairGate.faceKeep
      （β 随「人脸近期可见」渐变，丢脸、重新检出时不单帧硬切；β 的写法由 background_constants 核对，G8 β 扫描守像素）。
    - neck：A 通道强度 hairGate.shown(HairMaskGate.neckWeight(neckLifted, bodyPresence), engine = false)（天鹅颈按人体存在度渐变）。
    每条核对另有门禁自带的变异（HAIR_WIRING_MUTANTS）：把生产写法改坏后对应核对必须 FAIL，否则本门禁失败。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'VideoEffectsFilter.kt'))
    constants = {k: int(v) for k, v in re.findall(
        r'const val (\w+) = (\d+)\b', kotlin_code(os.path.join(kotlin_dir, 'FaceGeometrySmoother.kt')))}
    timeline_path = os.path.join(kotlin_dir, 'FaceTimeline.kt')
    if os.path.isfile(timeline_path):
        timeline = kotlin_code(timeline_path)
        for name in ('OVAL_INDEX', 'POSE_SIZE'):
            value = int_constant(timeline, name)
            if value is not None:
                constants[name] = value
    data = vectors['end_to_end']
    context = {'state': data['state'], 'geometry': data['geometry'], 'constants': constants}
    results = hair_wiring_lines(code, context)
    for _, ok, text in results:
        report.line(ok, text)
    caught, missed = [], []
    for name, key, old, new in HAIR_WIRING_MUTANTS:
        # 一处替换写成字符串；挪动一行这类需要多处替换的变异写成等长元组，按顺序依次替换，每个锚点在当时的文本里须恰好出现一次。
        pairs = list(zip(old, new)) if isinstance(old, tuple) else [(old, new)]
        mutated, anchor = code, None
        for o, n in pairs:
            if mutated.count(o) != 1:
                anchor = mutated.count(o)
                break
            mutated = mutated.replace(o, n)
        if anchor is not None:
            missed.append('%s（锚点出现 %d 次，改了写法请同步 HAIR_WIRING_MUTANTS）' % (name, anchor))
            continue
        verdict = {k: ok for k, ok, _ in hair_wiring_lines(mutated, context)}
        (caught if verdict.get(key) is False else missed).append('%s→%s' % (name, key))
    report.line(not missed, '头发接线核对的自带变异 %d 个，被对应核对拦下 %d 个%s' % (
        len(HAIR_WIRING_MUTANTS), len(caught), '' if not missed else '；未拦下：' + '；'.join(missed)))


# 降噪段的历史偏移（rgb8）：噪点量级应被混合，运动量级应直通。
DENOISE_NOISE, DENOISE_MOTION = 6, 80


def denoise_constants(kotlin_dir):
    """从 TemporalDenoisePass.kt 读 MAX_WEIGHT / DIFF_LOW / DIFF_HIGH，不手抄。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'TemporalDenoisePass.kt'))
    found = dict(re.findall(r'const val (MAX_WEIGHT|DIFF_LOW|DIFF_HIGH) = ([0-9.]+)f', code))
    if sorted(found) != ['DIFF_HIGH', 'DIFF_LOW', 'MAX_WEIGHT']:
        return None
    return {'weight': float(found['MAX_WEIGHT']), 'low': float(found['DIFF_LOW']), 'high': float(found['DIFF_HIGH']),
            'noise': DENOISE_NOISE, 'motion': DENOISE_MOTION}


def check_denoise(vectors, constants, result, report):
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['denoiseOff'], width, height) == source, 'denoise 权重 0 逐字节直通')
    image = pixels(result['denoiseOn'], width, height)

    def smoothstep(e0, e1, x):
        t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
        return t * t * (3 - 2 * t)
    worst = 0.0
    for r in range(height):
        delta = constants['noise'] if r < height // 2 else constants['motion']  # 第 0 行在顶部
        weight = constants['weight'] * (1 - smoothstep(constants['low'], constants['high'], delta / 255.0))
        for x in range(width):
            for i in range(3):
                s = source[r][x][i]
                h = s + delta if s < 128 else s - delta
                worst = max(worst, abs(image[r][x][i] - (s + (h - s) * weight)))
            worst = max(worst, abs(image[r][x][3] - 255))
    report.line(worst <= 1.0, 'denoise 噪点区按权重 %.2f 混合、运动区直通，最大误差 %.2f/255（容差 1）' % (constants['weight'], worst))


def check_lut(vectors, result, report):
    """LUT 查表在主合成 FRAGMENT_PRO 内（uLut 纹理单元 3、uGrade.x = 强度），其余参数全中性。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['lutOff'], width, height) == source, 'compositePro 内 LUT 强度 0（绑着反相表）逐字节直通')
    for name, expect in (('lutIdentity', lambda v: v), ('lutInvert', lambda v: 255 - v)):
        image = pixels(result[name], width, height)
        worst = max(abs(image[r][x][i] - expect(source[r][x][i])) for r in range(height) for x in range(width) for i in range(3))
        alpha = all(pixel[3] == 255 for row in image for pixel in row)
        report.line(worst <= 2 and alpha, 'compositePro 内 %s 最大误差 %d/255（容差 2，64 级量化 + 线性插值），alpha 保持=%s' % (
            name, worst, alpha))


def check_body(vectors, result, report):
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['bodyNone'], width, height) == source, 'body 无身体（|a.zw| < 0.5）逐字节直通')
    neutral = pixels(result['bodyNeutral'], width, height)
    worst = max(abs(neutral[r][x][i] - source[r][x][i]) for r in range(height) for x in range(width) for i in range(3))
    report.line(worst <= 1, 'body 有身体、强度全 0 时近似直通，最大误差 %d/255（容差 1）' % worst)
    # 胯 uv.y = 0.3，躯干 0.3（度量），肩线 uv.y = 0.3 + 0.3 * W / H；第 0 行在顶部，uv.y = 1 - (r + 0.5) / H。
    shoulder = 0.3 + 0.3 * width / height
    for name, label in (('bodyNeck', '天鹅颈'), ('bodyShoulders', '瘦肩')):
        image = pixels(result[name], width, height)
        below = [r for r in range(height) if 1 - (r + 0.5) / height < (shoulder - 0.2 if name == 'bodyShoulders' else shoulder - 0.02)]
        untouched = all(image[r] == source[r] for r in below)
        moved = sum(image[r][x] != source[r][x] for r in range(height) for x in range(width))
        report.line(untouched and moved > 0, 'body %s：肩线以下远处逐字节不变=%s，其余区域有变形的像素 %d 个' % (label, untouched, moved))
    # 长腿横向人体走廊边界（2026-11-22）：胯中点 (0.5, 0.5)，半宽 0.12，走廊外（|x/W - 0.5| > 0.24）背景地板逐字节不变。
    legs_img = pixels(result['bodyLegs'], width, height)
    side_cols = [x for x in range(width) if abs((x + 0.5) / width - 0.5) > 0.24]
    legs_sides_untouched = all(legs_img[r][x] == source[r][x] for r in range(height) for x in side_cols)
    legs_center_moved = sum(legs_img[r][width // 2] != source[r][width // 2] for r in range(height // 2 + 2, height))
    report.line(legs_sides_untouched and legs_center_moved > 0,
                'body 长腿横向走廊：走廊外两侧背景地板逐字节不变=%s，中线腿部变形像素 %d 个' % (legs_sides_untouched, legs_center_moved))


def check_background(vectors, result, report):
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['bgOff'], width, height) == source, 'background 强度 0 逐字节直通')
    report.line(pixels(result['bgPerson'], width, height) == source, 'background 全是人（遮罩 255）逐字节直通')
    flat = (40, 160, 90, 255)
    image = pixels(result['bgHalf'], width, height)
    # 遮罩第 0 行是画面顶部：上半是人保持原样，下半是背景取模糊图（纯色）；中线附近是线性插值的过渡带，不检查。
    top = all(image[r] == source[r] for r in range(height // 2 - 2))
    bottom = max(abs(image[r][x][i] - flat[i]) for r in range(height // 2 + 2, height) for x in range(width) for i in range(4))
    report.line(top and bottom <= 1, 'background 遮罩方向：上半（人）逐字节不变=%s，下半（背景）与模糊图最大误差 %d/255（容差 1）' % (top, bottom))
    blurred = pixels(result['bgBlurFlat'], width // 4, height // 4)
    worst = max(abs(p[i] - flat[i]) for row in blurred for p in row for i in range(3))
    report.line(worst <= 1, 'background 高斯权重归一：纯色模糊后最大误差 %d/255（容差 1）' % worst)
    # 预乘背景消黑边光晕：左半深黑人像 (10,10,10) + 右半亮蓝背景 (80,170,235)，经 bgPremask → bgBlur → bgMix 后，
    # 右半贴近交界处背景不应被左侧深色人像拉黑（无黑边残影）。
    no_halo = pixels(result['bgNoHalo'], width, height)
    edge_bg = no_halo[height // 2][width // 2 + 2]
    report.line(edge_bg[2] >= 225 and edge_bg[0] >= 72,
                'background 预乘消前景黑边光晕：紧贴暗色人像的背景像素 %s（期望接近 80,170,235）' % (edge_bg[:3],))


# 导向系数曲线探针（gate.js 的 guidedCoeffCurve）：(标签, 磨皮档, s 相对 s0 = √eps 的倍数, s 的绝对值, 下限, 上限)。
# 生产映射 a = max(r^1.5 / (1 + r^1.5), smoothstep(lo, 0.30, s))，r = s² / eps，自适应锁下限 lo = clamp(√eps − 0.09, 0.12, 0.18)。
# eps(0.55)（lo = 0.121）下依次取 s0/2、s0、1.25·s0、2·s0（r = 1/4、1、1.56、4）与 s = 0.60，解析值
# 0.111 / 0.500 / 0.889 / 1.000 / 1.000：前两点在锁下面（s0 处锁值恰也是 0.5），只看软映射；后三点落在锁区。
# - 软映射的指数：区间只放过约 1.2–1.85，近二值 r^4（0.004）、r^2（0.059）、r^1（0.200）、硬阈值开关（0 / 1）在 s0/2 处出界；
#   r = 1 处与指数无关恒为 0.5，用来核对 eps 标定。
# - 锁的位置：1.25·s0 与 2·s0 只有锁从 0.12 附近起步才进得了区间；旧的固定锁 0.30–0.55 是 0.660 / 0.888，去掉锁是
#   0.661 / 0.889，都出界。
# - 锁随磨皮档自适应：eps(1.0)（lo = 0.18）下 s = 0.24 应为 0.49；固定锁 0.12–0.30 是 0.735，旧锁 0.30–0.55 是 0.406，都出界。
# 有意改映射时同步这里的区间。
GUIDED_COEFF_CURVE = (('s0/2', 0.55, 0.5, None, 0.07, 0.16), ('s0', 0.55, 1.0, None, 0.42, 0.58),
                      ('1.25·s0', 0.55, 1.25, None, 0.84, 0.94), ('2·s0', 0.55, 2.0, None, 0.99, 1.0),
                      ('0.60', 0.55, None, 0.60, 0.99, 1.0), ('0.24@1.0', 1.0, None, 0.24, 0.44, 0.56))


def guided_exponent(curve, alphas):
    """由 s0/2 与 s0 两点（都在边缘锁下面）折算映射指数 p：logit(a) = p·ln r，r = s² / eps。只用于失败时定位，不参与判定；
    任一点落在 0 或 1（开关式映射）时返回 None。"""
    if not all(0.0 < alphas[i] < 1.0 for i in (0, 1)) or curve[0]['s8'] <= 0:
        return None
    logits = [math.log(alphas[i] / (1.0 - alphas[i])) for i in (0, 1)]
    return (logits[1] - logits[0]) / (2.0 * math.log(curve[1]['s8'] / curve[0]['s8']))


def check_guided_and_skin_mask(vectors, result, report):
    """v4 商业级 1/3 快速导向滤波全分辨率重构 + 导向系数曲线 + 468 点多边形 SkinMaskPass 专项门禁。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    # 1. 快速导向滤波全分辨率线性重构 (q = meanA * I_full + meanB)：
    # 左半深黑发丝 (25,22,22)，右半白皙皮肤 (215,175,155)；即便导向系数在 1/3 分辨率求得，
    # 因高对比边缘处 meanA → 1, meanB → 0，全分辨率紧贴阶跃边左右第 1 像素仍严格保持原生锋利零光晕。
    mid_r = height // 2
    # 默认磨皮 0.55 严格；0.9 时软映射在硬边两侧多约 3–5/255 误差（有意的代价，换来系数不再近二值），
    # 阈值按实测 (31,27,26) / (208,170,151) 各留 3 级。系数映射本身由下面第 3 项的曲线探针守。
    for name, smoothing, hair_max, skin_min in (('guidedStepFull', 0.55, (30, 27, 27), (210, 170, 150)),
                                                ('guidedStepFull09', 0.9, (34, 30, 29), (205, 167, 148))):
        step = pixels(result[name], width, height)
        left_edge = step[mid_r][width // 2 - 1]
        right_edge = step[mid_r][width // 2]
        hair_clean = all(left_edge[i] <= hair_max[i] for i in range(3))
        skin_clean = all(right_edge[i] >= skin_min[i] for i in range(3))
        report.line(hair_clean and skin_clean,
                    'guidedFilter 磨皮 %.2f 硬边两侧：发丝边 (25,22,22)→%s（≤%s），皮肤边 (215,175,155)→%s（≥%s）' % (
                        smoothing, left_edge[:3], hair_max, right_edge[:3], skin_min))
    # 2. 皮肤中频痘印（磨皮 0.9）：周围皮肤 (215,175,155)，中心 5×5 暗斑 (185,140,122) 被拉向周围肤色。
    blemish = pixels(result['guidedBlemishFull'], width, height)
    center = blemish[mid_r][width // 2]
    report.line(center[0] >= 200 and center[1] >= 158,
                'guidedFilter 磨皮 0.90 皮肤斑块被抹平：(185,140,122) → %s' % (center[:3],))
    # 3. 导向系数曲线（GUIDED_COEFF_FRAGMENT 单元级）：过渡平缓、不是近二值开关，强边缘由边缘锁钉到 1。
    curve = result['guidedCoeffCurve']
    alphas = [pixels(point['rgba'], 1, 1)[0][0][3] / 255.0 for point in curve]
    inside = len(curve) == len(GUIDED_COEFF_CURVE) and all(
        lo <= a <= hi for a, (_, _, _, _, lo, hi) in zip(alphas, GUIDED_COEFF_CURVE))
    exponent = guided_exponent(curve, alphas) if len(curve) == len(GUIDED_COEFF_CURVE) else None
    report.line(inside, 'guidedCoeff 系数曲线（折算指数 %s）：%s' % (
        '%.2f' % exponent if exponent is not None else '无（a 落在 0 或 1）',
        '，'.join('a(%s, eps=%.4f, s=%d/255)=%.3f ∈ [%.2f, %.2f]' % (label, point['eps'], point['s8'], a, lo, hi)
                 for point, a, (label, _, _, _, lo, hi) in zip(curve, alphas, GUIDED_COEFF_CURVE))))
    # 4. 多边形语义皮肤遮罩 (SKIN_MASK_VERTEX / SKIN_MASK_FRAGMENT)：R=skin, G=excl, B=hiPlane, 背景归零。
    mw, mh = width, height
    poly = pixels(result['skinMaskPoly'], mw, mh)
    bg_px = poly[0][0]
    cheek_px = poly[mh // 2][mw // 3]
    eye_px = poly[int(mh * 0.35)][mw // 2]
    bridge_px = poly[int(mh * 0.65)][mw // 2]
    poly_ok = (bg_px == (0, 0, 0, 255) and cheek_px[0] >= 210 and cheek_px[1] <= 10 and
               eye_px[0] <= 25 and eye_px[1] >= 210 and bridge_px[0] >= 210 and bridge_px[2] >= 200)
    report.line(poly_ok,
                'skinMask 多边形三通道光栅化：背景=%s，脸颊=%s，眼区挖空=%s，鼻梁骨相高光=%s' % (
                    bg_px[:3], cheek_px[:3], eye_px[:3], bridge_px[:3]))


def check_face_shape(vectors, result, report):
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['faceOff'], width, height) == source, 'face 三项强度为 0 逐字节直通')
    aspect = width / height
    # 每项中心与半径同 gate.js；第 0 行在顶部，uv = ((x + 0.5) / W, 1 - (r + 0.5) / H)，度量 y = uv.y / aspect。
    for name, label, (cx, cy, radius) in (('faceChin', '下巴', (0.5, 0.35, 0.12)), ('faceForehead', '额头', (0.5, 0.7, 0.15)),
                                          ('faceNose', '瘦鼻', (0.5, 0.5, 0.08))):
        image = pixels(result[name], width, height)
        outside, moved = True, 0
        for r in range(height):
            for x in range(width):
                u, v = (x + 0.5) / width, 1 - (r + 0.5) / height
                d = ((u - cx) ** 2 + ((v - cy) / aspect) ** 2) ** 0.5
                if image[r][x] != source[r][x]:
                    moved += 1
                    # 线性采样的边界可能多影响 1 个像素；瘦鼻椭圆竖向 1.3 倍。
                    if d > radius * 1.35 + 2.0 / width:
                        outside = False
        report.line(outside and moved > 0, 'face %s：半径外逐字节不变=%s，半径内变形的像素 %d 个' % (label, outside, moved))
    # 局部修饰：只改眼睛 / 眼下 / 嘴的椭圆内。椭圆：(中心, 横半径, 竖半径, 竖向偏移)，度量单位；同 gate.js。
    eyes = [(0.4, 0.6), (0.6, 0.6)]
    for name, label, regions in (
            ('faceDark', '去黑眼圈', [(x, y, 0.06 * 0.9, 0.03 * 1.1, -1.6 * 0.03) for x, y in eyes]),
            ('faceBright', '亮眼', [(x, y, 0.06, 0.03, 0.0) for x, y in eyes]),
            ('faceTeeth', '白牙', [(0.5, 0.35, 0.08, 0.04, 0.0)])):
        image = pixels(result[name], width, height)
        outside, changed = True, 0
        for r in range(height):
            for x in range(width):
                if image[r][x] == source[r][x]:
                    continue
                changed += 1
                u, my = (x + 0.5) / width, (1 - (r + 0.5) / height) / aspect
                inside = any((((u - cx) / rx) ** 2 + ((my - cy / aspect - shift) / ry) ** 2) ** 0.5 <= 1.02
                             for cx, cy, rx, ry, shift in regions)
                outside = outside and inside
        report.line(outside and (changed > 0 or name == 'faceTeeth'),
                    'face %s：椭圆外逐字节不变=%s，椭圆内改动的像素 %d 个' % (label, outside, changed))
    split = pixels(result['faceTeethSplit'], width, height)
    lips = all(split[r][x] == (200, 40, 50, 255) for r in range(height) for x in range(width // 2))
    # 嘴中心 (0.5, 0.35) 右侧一点：牙色应变白（蓝通道上升、黄偏差变小）。
    r0, x0 = int((1 - 0.35) * height), width // 2 + 2
    tooth = split[r0][x0]
    whiter = tooth[2] > 150 and (tooth[0] - tooth[2]) < (200 - 150)
    report.line(lips and whiter, 'face 白牙选择性：唇红逐字节不变=%s，牙色 (200,190,150) → %s 变白=%s' % (lips, tooth[:3], whiter))
    # 法令纹：输入数据第 0 行在底部（与回读一致），用 pixels() 同样翻转后逐像素比较。
    before = pixels(result['faceFoldInput'], width, height)
    after = pixels(result['faceFold'], width, height)
    line = [(r, x) for r in range(height) for x in range(width) if before[r][x][:3] == (120, 90, 80)]
    lifted = sum(1 for r, x in line if after[r][x][0] > 120) if line else 0
    skin_same = all(after[r][x] == before[r][x] for r in range(height) for x in range(width) if before[r][x][:3] == (200, 160, 140))
    report.line(len(line) > 0 and lifted >= len(line) * 0.8 and skin_same,
                'face 法令纹：暗线 %d 个像素中 %d 个被提亮，周围皮肤逐字节不变=%s' % (len(line), lifted, skin_same))


def build_page(shaders, vectors, cases, geometry_uniforms, path, denoise, guided, hair_edge, hair_mask):
    subset = {key: vectors[key] for key in ('skin_gate', 'composite', 'masks', 'prepass', 'end_to_end')}
    controls = [{'name': n, 'vertex': CONTROL_VERTEX, 'good': g, 'bad': b} for n, g, b in CONTROLS]
    controls.append({'name': STAGE_CONTROL[0], 'vertex': STAGE_VERTEX, 'good': STAGE_CONTROL[1], 'bad': STAGE_CONTROL[2]})
    data = {'shaders': shaders, 'programs': ALL_PROGRAMS, 'controls': controls, 'vectors': subset, 'endToEnd': cases,
            'geometryUniforms': geometry_uniforms, 'denoise': denoise, 'guided': guided,
            'curve': [[smoothing, factor, absolute] for _, smoothing, factor, absolute, _, _ in GUIDED_COEFF_CURVE],
            'proCases': PRO_CASES, 'naturalCases': NATURAL_CASES, 'hairEdge': hair_edge,
            'hairACases': HAIR_A_CASES, 'hairAlpha': hair_mask['alpha'], 'bgHair': hair_mask['bg']}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate.js'), encoding='utf-8') as handle:
        script = handle.read()
    # </script> 不会出现在 JSON 里（只有着色器文本与数字），仍按惯例转义斜杠。
    payload = json.dumps(data, ensure_ascii=True).replace('</', '<\\/')
    page = ('<!doctype html><html><head><meta charset="utf-8"></head><body><pre id="out">PENDING</pre>\n'
            '<script>\nconst INPUT = %s;\n%s</script></body></html>\n' % (payload, script))
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(page)


CHROME_TIMEOUT = 180


def kill_group(process):
    """结束 Chrome 所在进程组（含它拉起的渲染/GPU 子进程）；进程组已不存在时忽略。"""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_chrome(chrome, page, work):
    """只等待 Chrome 主进程本身。

    Chrome 会拉起继承其 stdout/stderr 的后台进程（例如 GoogleUpdater），这些进程可能比 Chrome 活得久；
    若用管道捕获输出，就要等到所有持有管道写端的进程退出才会读到 EOF，门禁会随机卡到超时。
    因此输出重定向到工作目录里的文件，只 wait 主进程；Chrome 放在独立会话里，超时时整组杀掉。"""
    args = [chrome, '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
            '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--disable-default-apps',
            '--disable-background-networking', '--disable-component-update', '--disable-sync',
            '--disable-breakpad', '--disable-crash-reporter', '--disable-domain-reliability',
            '--metrics-recording-only', '--no-service-autorun', '--password-store=basic', '--use-mock-keychain',
            '--user-data-dir=' + os.path.join(work, 'chrome-profile'), '--virtual-time-budget=60000',
            '--dump-dom', 'file://' + page]
    out_path, err_path = os.path.join(work, 'chrome.stdout'), os.path.join(work, 'chrome.stderr')
    with open(out_path, 'wb') as out, open(err_path, 'wb') as err:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        try:
            returncode = process.wait(timeout=CHROME_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill_group(process)
            process.wait()
            return None, 'Chrome %d s 内未结束' % CHROME_TIMEOUT
        except BaseException:
            kill_group(process)
            process.wait()
            raise
    # 主进程已退出；清理它留在同一进程组里的子进程，避免残留进程继续占用工作目录。
    kill_group(process)
    with open(out_path, 'rb') as handle:
        dom = handle.read().decode('utf-8', 'replace')
    match = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    text = html.unescape(match.group(1)) if match else ''
    if not text or text == 'PENDING':
        with open(err_path, 'rb') as handle:
            tail = handle.read().decode('utf-8', 'replace').strip().splitlines()[-8:]
        return None, 'Chrome 未产出结果（退出码 %s）：\n%s' % (returncode, '\n'.join(tail))
    return json.loads(text), None


def pixels(hex_text, width, height):
    """回读字节 → [行][列] 的 RGBA 元组；回读第 0 行在底部，这里翻成第 0 行在顶部。"""
    raw = bytes.fromhex(hex_text)
    rows = [[tuple(raw[(r * width + x) * 4:(r * width + x) * 4 + 4]) for x in range(width)] for r in range(height)]
    return rows[::-1]


def cells(hex_text, count):
    raw = bytes.fromhex(hex_text)
    return [tuple(raw[i * 4:i * 4 + 4]) for i in range(count)]


def check_programs(result, report):
    for name, info in result['programs'].items():
        ok = info.get('vertex') and info.get('fragment') and info.get('link')
        detail = '' if ok else '：' + (info.get('log') or '无日志')[:400]
        report.line(bool(ok), '编译+链接 %s%s' % (name, detail))
    for name in ('composite', 'composite@mediump', 'compositePro', 'compositePro@mediump'):
        uniforms = result['programs'].get(name, {}).get('uniforms')
        if uniforms is None:
            continue
        vectors = sum(size * ROWS.get(kind, 1) for _, kind, size in uniforms)
        names = ', '.join('%s[%d]' % (n.replace('[0]', ''), s) if s > 1 else n for n, _, s in uniforms)
        report.line(vectors <= MAX_UNIFORM_VECTORS,
                    '%s 活跃 uniform 向量（含采样器）%d，上限 %d：%s' % (name, vectors, MAX_UNIFORM_VECTORS, names))
        if name.startswith('compositePro'):
            report.line(vectors == COMPOSITE_PRO_VECTORS, '%s 活跃 uniform 向量恰为 %d（头发遮罩走 uSkinMask.a，不占新 uniform）：%d' % (
                name, COMPOSITE_PRO_VECTORS, vectors))
    for name, control in result['controls'].items():
        report.line(control['good'] and not control['bad'],
                    '负对照 %s：合法版本%s，非法版本%s（%s）' % (
                        name, '通过' if control['good'] else '未通过', '被拒' if not control['bad'] else '竟然通过',
                        control['log'].replace('\n', ' ')[:120]))


def check_skin_gate(vectors, result, report):
    # 调试程序把 s = skinWeight × 255 写成 (floor(s)/255, fract(s))，回读后还原到约 1/65025 的分辨率。
    cases = vectors['skin_gate']['cases']
    tolerance = vectors['meta']['tolerance']['skin_gate']
    worst, worst_rgb = 0.0, None
    for case, value in zip(cases, cells(result['skin'], len(cases))):
        diff = abs((value[0] + value[1] / 255) / 255 - case['expected'])
        if diff > worst:
            worst, worst_rgb = diff, case['rgb8']
    report.line(worst <= tolerance, 'skin_gate %d 例，最大误差 %.5f（rgb8 %s），容差 %.3f' % (
        len(cases), worst, worst_rgb, tolerance))


def check_composite(vectors, result, report):
    cases = vectors['composite']['cases']
    got = cells(result['composite'], len(cases))
    tolerance = vectors['meta']['tolerance']['composite'] * 255
    worst, worst_name, over, neutral_bad = 0.0, '', 0, []
    for case, value in zip(cases, got):
        diff = max(abs(value[i] - case['expected'][i] * 255) for i in range(3))
        if diff > worst:
            worst, worst_name = diff, case['name']
        over += diff > tolerance
        if case['params'] == 'neutral' and list(value[:3]) != [round(x * 255) for x in case['I']]:
            neutral_bad.append(case['name'])
    report.line(over == 0, 'composite %d 例，最大误差 %.3f/255（%s），容差 %.3f/255，超差 %d 例' % (
        len(cases), worst, worst_name, tolerance, over))
    report.line(not neutral_bad, 'composite 中性参数原样返回 I%s' % (
        '' if not neutral_bad else '：' + ', '.join(neutral_bad[:5])))


def check_masks(vectors, result, report):
    cases = vectors['masks']['cases']
    tolerance = vectors['meta']['tolerance']['masks']
    masks = cells(result['masks'], len(cases))
    ellipses = cells(result['ellipses'], len(cases))
    worst = {'face': 0.0, 'excl': 0.0, 'oval': 0.0, 'eye0': 0.0, 'eye1': 0.0, 'mouth': 0.0}
    for case, mask, ellipse in zip(cases, masks, ellipses):
        expected = case['expected']
        for key, value in (('face', mask[0]), ('excl', mask[1]), ('oval', ellipse[0]), ('eye0', ellipse[1]),
                           ('eye1', ellipse[2]), ('mouth', ellipse[3])):
            worst[key] = max(worst[key], abs(value / 255 - expected[key]))
    report.line(max(worst.values()) <= tolerance, 'masks %d 例，最大误差 %s，容差 %.3f' % (
        len(cases), ' '.join('%s=%.4f' % item for item in worst.items()), tolerance))


def check_prepass(vectors, result, report):
    data = vectors['prepass']
    width, height = data['width'], data['height']
    sw, sh = (width + 3) // 4, (height + 3) // 4
    expected = data['rgba8']
    tolerance = vectors['meta']['tolerance']['prepass_rgba8'] * 255
    t0 = pixels(result['prepassT0'], sw, sh)
    t2 = pixels(result['prepassT2'], sw, sh)
    worst_t0 = max(abs(t0[r][x][i] - expected['T0'][r][x][i] * 255)
                   for r in range(sh) for x in range(sw) for i in range(3))
    worst_mean = max(abs(t2[r][x][i] - expected['mean'][r][x][i] * 255)
                     for r in range(sh) for x in range(sw) for i in range(3))
    worst_var = max(abs(t2[r][x][3] - expected['var'][r][x] * 255) for r in range(sh) for x in range(sw))
    report.line(max(worst_t0, worst_mean, worst_var) <= tolerance,
                'prepass %dx%d → %dx%d：T0 %.3f/255，mean %.3f/255，var %.3f/255，容差 %.3f/255' % (
                    width, height, sw, sh, worst_t0, worst_mean, worst_var, tolerance))
    up = pixels(result['prepassUp'], width, height)
    worst_probe, where = 0.0, None
    for probe in expected['probes']:
        value = up[probe['row']][probe['x']]
        diff = max([abs(value[i] - probe['meanUp'][i] * 255) for i in range(3)] + [abs(value[3] - probe['varUp'] * 255)])
        if diff > worst_probe:
            worst_probe, where = diff, (probe['x'], probe['row'])
    report.line(worst_probe <= tolerance, 'prepass 放大探针 %d 个（含四角 clamp-to-edge），最大误差 %.3f/255 于 %s' % (
        len(expected['probes']), worst_probe, where))


def check_end_to_end(vectors, cases, result, report):
    data = vectors['end_to_end']
    width, height = data['width'], data['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    tolerance = data['tolerance_rgb8']
    exact = pixels(result['original'], width, height) == source
    report.line(exact, '原画程序 ORIGINAL 逐字节直通')
    expected = {case['name']: case['expected_rgb8'] for case in data['cases']}
    for case in cases:
        image = pixels(result['endToEnd'][case['name']], width, height)
        if case['name'] in expected:
            worst = max(abs(image[p['row']][p['x']][i] - e[i])
                        for p, e in zip(data['probes'], expected[case['name']]) for i in range(3))
            alpha = all(pixel[3] == 255 for row in image for pixel in row)
            report.line(worst <= tolerance and alpha, 'end_to_end %-26s processing=%d presence=%.1f 探针 %d 个最大误差 %d/255（容差 %d），alpha 保持=%s' % (
                case['name'], case['processing'], case['presence'], len(data['probes']), worst, tolerance, alpha))
        if not case['processing']:
            diff = sum(image[r][x] != source[r][x] for r in range(height) for x in range(width))
            report.line(diff == 0, 'end_to_end %s 逐字节直通（不同像素 %d）' % (case['name'], diff))


def rgb01(r, g, b):
    return [r / 255, g / 255, b / 255]


def coeff(r, g, b, a):
    """导向系数纹理的一个像素：(b.rgb, a)，b = (1 - a) * mean。"""
    return [x / 255 * (1 - a) for x in (r, g, b)] + [a]


# uGrade = (LUT 强度, coring 门限, 回退预处理标记, 未用)：探针默认无 LUT、室内光门限 0.005。
GRADE_INDOOR = [0.0, 0.005, 0.0, 0.0]


def pro_case(name, rgb, guided, fine, wide, mask, params, details, tone):
    """beautyColorPro 的一条探针（调试程序直接喂参数）：rgb / fine / wide 为 0..255 的 (R, G, B)，
    guided = coeff(...) 或 None（平坦色块：b = I、a = 0，导向重构即原色），mask = (face, excl, hiPlane[, nonHair])，
    第 4 个分量（非头发权重，FRAGMENT_PRO 由 uSkinMask.a 换算）缺省为 1.0，即不接头发遮罩。"""
    return {'name': name, 'rgb8': list(rgb), 'I': rgb01(*rgb), 'guided': guided or rgb01(*rgb) + [0.0],
            'fine': rgb01(*(fine or rgb)), 'wide': rgb01(*(wide or rgb)), 'mask': list(mask) + [1.0] * (4 - len(mask)),
            'params': params, 'details': details, 'tone': tone, 'grade': GRADE_INDOOR}


# 顺序即 check_composite_pro 的下标。params = (style + 16 * skinOnly, smoothing, brightness, skinTexture)，
# details = (radiance, sharpness, saturation, filterIntensity)，tone = (whiten, rosy, 未用, 未用)。
PRO_CASES = [
    pro_case('blemish', (180, 130, 110), coeff(210, 165, 145, 0.08), None, None, (1, 0, 0.5), [16, 0.85, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('tex_low', (220, 175, 155), coeff(210, 165, 145, 0.05), (218, 173, 153), (210, 165, 145), (1, 0, 0.5), [16, 0.9, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('tex_high', (220, 175, 155), coeff(210, 165, 145, 0.05), (218, 173, 153), (210, 165, 145), (1, 0, 0.5), [16, 0.9, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('dark_noise', (121, 95, 82), coeff(120, 94, 81, 0.02), (121, 95, 82), (120, 94, 81), (1, 0, 0.5), [16, 0.9, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('eyelash', (20, 18, 18), coeff(190, 145, 125, 0.99), None, None, (0, 0, 0), [0, 0.9, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('yellow_skin', (200, 160, 118), None, None, None, (1, 0, 0.5), [16, 0, 0, 0], [0, 0, 1, 0], [0.8, 0, 0, 0]),
    pro_case('feature', (180, 55, 65), None, None, None, (0, 1, 0), [16, 0, 0, 0], [0.8, 0, 1, 0], [0.8, 0.8, 0, 0]),
    pro_case('sculpt_high', (180, 140, 120), None, None, None, (1, 0, 1.0), [16, 0, 0, 0], [0.8, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('bg_grey', (89, 84, 76), None, None, None, (0, 0, 0), [16, 0, 0.5, 0], [0.8, 0, 1, 0], [0.8, 0.8, 0, 0]),
    pro_case('bg_dark', (31, 31, 36), None, None, None, (0, 0, 0), [16, 0, 0.5, 0], [0.8, 0, 1, 0], [0.8, 0.8, 0, 0]),
    pro_case('medium_skin', (194, 143, 117), None, None, None, (1, 0, 0.3), [16, 0, 0, 0], [0, 0, 1, 0], [0.5, 0.3, 0, 0]),
    pro_case('shadow_skin', (115, 79, 64), None, None, None, (1, 0, 0.3), [16, 0, 0, 0], [0, 0, 1, 0], [0.5, 0.3, 0, 0]),
    pro_case('highlight', (240, 215, 200), None, None, None, (1, 0, 0.3), [16, 0, 1.0, 0], [0, 0, 1, 0], [1.0, 0, 0, 0]),
    pro_case('sculpt_contour', (194, 143, 117), None, None, None, (1, 0, 0.0), [16, 0, 0, 0], [0.6, 0, 1, 0], [0, 0, 0, 0]),
    pro_case('rose_black', (0, 0, 0), None, None, None, (0, 0, 0), [5, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]),
    pro_case('rose_white', (255, 255, 255), None, None, None, (0, 0, 0), [5, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]),
    pro_case('feature_sharpen', (120, 80, 70), None, (125, 85, 75), (110, 70, 60), (0, 1, 0), [16, 0, 0, 0], [0, 0.5, 1, 0], [0, 0, 0, 0]),
    pro_case('bg_noise', (100, 100, 100), None, (101, 101, 101), (100, 100, 100), (0, 0, 0), [16, 0, 0, 0], [0, 0.5, 1, 0], [0, 0, 0, 0]),
]
# 默认「自然」档（计划 R1）：磨皮 0.55、质感 0.50、美白 0.30、红润 0.15、立体 0.20、提亮 0、清晰 0.30、饱和度 1.00、只磨皮肤。
NATURAL = {'params': [16, 0.55, 0.0, 0.5], 'details': [0.20, 0.30, 1.0, 0.0], 'tone': [0.30, 0.15, 0.0, 0.0]}
NATURAL_SKINS = (('medium', (194, 143, 117)), ('fair', (219, 178, 158)), ('shadow', (115, 79, 64)))
NATURAL_BACKGROUNDS = (('grey', (89, 84, 76)), ('dark', (31, 31, 36)), ('white', (217, 214, 209)),
                       ('green', (64, 102, 51)), ('blue', (51, 77, 153)))
# 已知限制：类肤色背景（米色墙、暖光白墙）按肤色门限处理，「背景提亮 ≤3 级」只针对中性背景，这里只打印不判失败。
NATURAL_KNOWN = (('beige_wall', (204, 178, 153)),)
NATURAL_CASES = (
    [pro_case('skin_' + n, c, None, None, None, (1, 0, 0.3), NATURAL['params'], NATURAL['details'], NATURAL['tone'])
     for n, c in NATURAL_SKINS] +
    [pro_case('bg_' + n, c, None, None, None, (0, 0, 0), NATURAL['params'], NATURAL['details'], NATURAL['tone'])
     for n, c in NATURAL_BACKGROUNDS + NATURAL_KNOWN])


def luma8(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def saturation(c):
    return (max(c[:3]) - min(c[:3])) / max(c[:3]) if max(c[:3]) else 0.0


def yellow_hue(c):
    """(G - B) / (R - B)：肤色色相里偏黄的程度。"""
    return (c[1] - c[2]) / max(c[0] - c[2], 1)


def check_composite_pro(vectors, result, report):
    """Android 主合成（COMPOSITE_PRO / FRAGMENT_PRO）专项数值门禁：直通 + 18 条探针 + 「自然」档探针组。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['proOff'], width, height) == source,
                'compositePro processing=0（flags=-1，即使磨皮、LUT 等都非零）逐字节直通')
    report.line(pixels(result['proNeutral'], width, height) == source, 'compositePro 全中性参数逐字节直通')
    got = dict(zip((c['name'] for c in PRO_CASES), cells(result['proCases'], len(PRO_CASES))))
    src = {c['name']: tuple(c['rgb8']) for c in PRO_CASES}

    def dy(name):
        return luma8(got[name]) - luma8(src[name])
    # 磨皮：中频痘印被拉向周围肤色 (210,165,145)。
    out = got['blemish']
    report.line(out[0] >= 198 and out[1] >= 150, 'compositePro 中频痘印被抹平：(180,130,110) → %s' % (out[:3],))
    # 质感：带通回填随 skinTexture 增强，且仍高于导向均值 210。
    low, high = got['tex_low'], got['tex_high']
    report.line(high[0] > low[0] > 210, 'compositePro 带通质感回填随质感增强：均值 210，质感 0 → %s，质感 1 → %s' % (low[:3], high[:3]))
    out = got['dark_noise']
    report.line(abs(out[0] - 120) <= 1, 'compositePro 噪声级（1/255）频带被 coring 压掉：%s（均值 120）' % (out[:3],))
    out = got['eyelash']
    report.line(out[0] <= 24, 'compositePro 导向重构保边：睫毛 (20,18,18) 贴皮肤 → %s' % (out[:3],))
    # 去黄：偏黄肤色的色相回到常见肤色，饱和度不掉，亮度上升。
    out, base = got['yellow_skin'], src['yellow_skin']
    ok = (yellow_hue(out) <= yellow_hue(base) - 0.03 and saturation(out) >= 0.95 * saturation(base) and dy('yellow_skin') >= 10)
    report.line(ok, 'compositePro 偏黄肤色去黄提亮：(200,160,118) → %s，色相 %.3f→%.3f，饱和度 %.3f→%.3f，ΔY %+.1f' % (
        out[:3], yellow_hue(base), yellow_hue(out), saturation(base), saturation(out), dy('yellow_skin')))
    out = got['feature']
    report.line(out[:3] == (180, 55, 65), 'compositePro 五官保护区不被美白/红润/立体污染：(180,55,65) → %s' % (out[:3],))
    out = got['sculpt_high']
    report.line(192 < out[0] < 248, 'compositePro 立体光高光平面提亮不死白：(180,140,120) → %s' % (out[:3],))
    for name in ('bg_grey', 'bg_dark'):
        report.line(got[name][:3] == src[name], 'compositePro 背景 %s 在美白/红润/立体 0.8、提亮 0.5 下逐字节不变 → %s' % (
            src[name], got[name][:3]))
    for name in ('medium_skin', 'shadow_skin'):
        out, base = got[name], src[name]
        ratio = saturation(out) / saturation(base)
        ok = ratio >= 0.85 and 8 <= dy(name) <= 25 and out[2] < out[1]
        report.line(ok, 'compositePro %s 美白 0.5 红润 0.3：%s → %s，饱和度比 %.2f（≥0.85），ΔY %+.1f（8..25），不偏紫=%s' % (
            name, base, out[:3], ratio, dy(name), out[2] < out[1]))
    out, base = got['highlight'], src['highlight']
    report.line(abs(yellow_hue(out) - yellow_hue(base)) <= 0.03 and dy('highlight') >= 8,
                'compositePro 高光肤色提亮不偏色：%s → %s，色相 %.3f→%.3f，ΔY %+.1f' % (
                    base, out[:3], yellow_hue(base), yellow_hue(out), dy('highlight')))
    report.line(-10 <= dy('sculpt_contour') <= -3, 'compositePro 立体光轮廓侧轻压暗：%s → %s，ΔY %+.1f（-10..-3）' % (
        src['sculpt_contour'], got['sculpt_contour'][:3], dy('sculpt_contour')))
    for name in ('rose_black', 'rose_white'):
        report.line(got[name][:3] == src[name], 'compositePro rose 滤镜黑白场不变：%s → %s' % (src[name], got[name][:3]))
    report.line(dy('feature_sharpen') >= 8, 'compositePro 五官带通锐化：%s → %s，ΔY %+.1f' % (
        src['feature_sharpen'], got['feature_sharpen'][:3], dy('feature_sharpen')))
    report.line(got['bg_noise'][:3] == src['bg_noise'], 'compositePro 背景噪声级频带不被锐化：%s → %s' % (
        src['bg_noise'], got['bg_noise'][:3]))
    # 「自然」档：肤色饱和度损失 ≤15%，中性背景各通道提亮 ≤3。
    natural = dict(zip((c['name'] for c in NATURAL_CASES), cells(result['proNatural'], len(NATURAL_CASES))))
    for name, base in NATURAL_SKINS:
        out = natural['skin_' + name]
        loss = 1 - saturation(out) / saturation(base)
        report.line(loss <= 0.15, 'compositePro 自然档 %s 肤色 %s → %s，饱和度损失 %.1f%%（≤15%%），ΔY %+.1f' % (
            name, base, out[:3], loss * 100, luma8(out) - luma8(base)))
    for name, base in NATURAL_BACKGROUNDS:
        out = natural['bg_' + name]
        lift = max(out[i] - base[i] for i in range(3))
        report.line(lift <= 3, 'compositePro 自然档中性背景 %s %s → %s，最大提亮 %+d（≤3）' % (name, base, out[:3], lift))
    for name, base in NATURAL_KNOWN:
        out = natural['bg_' + name]
        print('INFO  compositePro 自然档类肤色背景 %s %s → %s，最大提亮 %+d（已知限制：按肤色处理，不计入「背景提亮 ≤3」口径）' % (
            name, base, out[:3], max(out[i] - base[i] for i in range(3))))


def check_fallback_prepass(vectors, result, report):
    """导向滤波故障回退 BeautyPrePass 时，uBlur 绑的是 (mean, var)。uGrade.z = 1 时着色器换算成 (b, a)：
    输出必须落在原图与均值之间（无「base * var + mean」式的提亮光晕），且确有像素被平滑。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    up = pixels(result['prepassUp'], width, height)

    def outside(image):
        worst, changed = 0, 0
        for r in range(height):
            for x in range(width):
                changed += image[r][x][:3] != source[r][x][:3]
                for i in range(3):
                    lo = min(source[r][x][i], up[r][x][i]) - 2
                    hi = max(source[r][x][i], up[r][x][i]) + 2
                    worst = max(worst, lo - image[r][x][i], image[r][x][i] - hi)
        return worst, changed
    worst, changed = outside(pixels(result['fallbackPrepass'], width, height))
    report.line(worst <= 0 and changed > 0, 'compositePro 回退预处理（uGrade.z=1，磨皮 0.8）落在原图与均值之间：越界 %d/255（容差 2），'
                '被平滑的像素 %d 个' % (max(worst, 0), changed))
    legacy, _ = outside(pixels(result['fallbackPrepassLegacy'], width, height))
    print('INFO  对照：同一回退输入按导向系数解读（uGrade.z=0，旧语义）时最大越界 %d/255' % max(legacy, 0))


def check_mesh_warp(vectors, result, report):
    """Android 478 点 2D 三角网格微整（M3 MESH_WARP_VERTEX / MESH_WARP_FRAGMENT）门禁。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    rgb8 = vectors['prepass']['rgb8']
    source = [[tuple(rgb8[r][x * 3:x * 3 + 3]) + (255,) for x in range(width)] for r in range(height)]
    report.line(pixels(result['meshIdentity'], width, height) == source, 'meshWarp 恒等网格逐字节直通')
    warped = pixels(result['meshWarped'], width, height)
    # 缓冲环外（u < 0.24 或 u > 0.76 或 v < 0.24 或 v > 0.76）必须逐字节不变（背景门框直线零弯曲）。
    bg_unchanged, inside_moved = True, 0
    for r in range(height):
        v = 1.0 - (r + 0.5) / height
        for x in range(width):
            u = (x + 0.5) / width
            outside = (u < 0.24 or u > 0.76 or v < 0.24 or v > 0.76)
            if warped[r][x] != source[r][x]:
                if outside:
                    bg_unchanged = False
                else:
                    inside_moved += 1
    report.line(bg_unchanged and inside_moved > 0,
                'meshWarp 零弯曲缓冲环：外围背景逐字节不变=%s，网格内部平滑变形像素 %d 个' % (bg_unchanged, inside_moved))


def check_makeup(vectors, result, report):
    """Android 实时局部三角网格美妆（M4 MAKEUP_LAYER / MAKEUP_LIP / MAKEUP_IRIS）门禁。"""
    width, height = vectors['end_to_end']['width'], vectors['end_to_end']['height']
    skin = (200, 160, 140, 255)
    eye_base = (45, 38, 36, 255)
    skin_img = [[skin for _ in range(width)] for _ in range(height)]
    eye_img = [[eye_base for _ in range(width)] for _ in range(height)]

    report.line(pixels(result['makeupLayerOff'], width, height) == skin_img,
                'makeupLayer 强度为 0 时 discard，底图逐字节不变')

    blush = pixels(result['makeupLayerBlush'], width, height)[height // 2][width // 2]
    contour = pixels(result['makeupLayerContour'], width, height)[height // 2][width // 2]
    highlight = pixels(result['makeupLayerHighlight'], width, height)[height // 2][width // 2]
    report.line(blush[0] >= skin[0] - 4 and (blush[0] - blush[1]) > (skin[0] - skin[1]) + 6 and blush[3] == 255,
                'makeupLayer SoftLight 腮红自然增红润且保 Alpha：(200,160,140) → %s' % (blush,))
    report.line(contour[0] < skin[0] - 20 and contour[1] < skin[1] - 20 and contour[2] < skin[2] - 20,
                'makeupLayer Multiply 修容自然压暗：(200,160,140) → %s' % (contour[:3],))
    report.line(highlight[0] > skin[0] + 5 and highlight[1] > skin[1] + 5 and highlight[2] > skin[2] + 5,
                'makeupLayer ScreenLift 高光自然提亮：(200,160,140) → %s' % (highlight[:3],))

    velvet = pixels(result['makeupLipVelvet'], width, height)
    glossy = pixels(result['makeupLipGlossy'], width, height)
    # 唇带羽化贴图：左半 B=0（模拟非唇带/口腔），必须逐字节不变；右半 B=255 为纯唇带。
    left_untouched = all(velvet[r][x] == skin and glossy[r][x] == skin
                         for r in range(height) for x in range(width // 2 - 2))
    # 第 0 行在画面顶部；探测右半纯唇带靠近唇腹中央 (x≈0.55*W, vMakeupUv.y≈0.728) 的下唇水光高光带。
    probe_r, probe_x = int((1.0 - 0.728) * height), int(width * 0.55)
    v_lip = velvet[probe_r][probe_x]
    g_lip = glossy[probe_r][probe_x]
    lip_tinted = (v_lip[0] - v_lip[1]) > (skin[0] - skin[1]) + 25
    glossy_brighter = (g_lip[0] + g_lip[1] + g_lip[2]) > (v_lip[0] + v_lip[1] + v_lip[2]) + 15
    report.line(left_untouched and lip_tinted and glossy_brighter,
                'makeupLip 零羽化区不变=%s，丝绒=%s，水光=%s（水光更亮=%s）' % (
                    left_untouched, v_lip[:3], g_lip[:3], glossy_brighter))

    # 深紫暗沉裸唇 (105,68,72) 去乌归一化显色：涂蜜桃珊瑚红后必须通透提亮而非暗黑发乌。
    dark_lip = pixels(result['makeupLipDarkBase'], width, height)[probe_r][probe_x]
    report.line(dark_lip[0] >= 165 and (dark_lip[0] - dark_lip[1]) >= 55,
                'makeupLip 深唇去乌归一化纯净显色：(105,68,72) → %s' % (dark_lip[:3],))

    # 上下分屏局部放大高清眉眼 UV (uMode.y = 1)：右眼框 (0.33, 0.33) 与左眼框 (0.67, 0.33) 分别命中上半 R 与下半 G，框外逐字节不变。
    split_eye = pixels(result['makeupSplitEye'], width, height)
    r_box_px = split_eye[int((1.0 - 0.33) * height)][int(0.33 * width)]
    l_box_px = split_eye[int((1.0 - 0.33) * height)][int(0.67 * width)]
    nose_px = split_eye[int((1.0 - 0.33) * height)][width // 2]
    jaw_px = split_eye[int((1.0 - 0.75) * height)][width // 2]
    split_ok = (r_box_px[0] < skin[0] - 20 and l_box_px == skin and nose_px == skin and jaw_px == skin)
    report.line(split_ok,
                'makeupLayer 上下分屏局部放大眉眼 UV：右眼命中=%s，左眼另一通道=%s，鼻梁/下颌框外不变=%s/%s' % (
                    r_box_px[:3], l_box_px[:3], nose_px == skin, jaw_px == skin))

    iris_closed = pixels(result['makeupIrisClosed'], width, height)
    report.line(iris_closed == eye_img, 'makeupIris 闭眼门控（openness=0.08 < 0.10）逐字节不变，绝不穿帮')
    iris_open = pixels(result['makeupIrisOpen'], width, height)
    corners_untouched = all(iris_open[r][x] == eye_base
                            for r, x in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    # 圆心 (0.5, 0.5)，radiusMetric = 0.25，aspect = W/H；主眼神光在 d = (-0.26, +0.28)。
    aspect = width / height
    glint_u = 0.5 - 0.26 * 0.25
    glint_v = 0.5 + 0.28 * 0.25 * aspect
    glint_x = int(glint_u * width)
    glint_r = int((1.0 - glint_v) * height)
    center_px = iris_open[height // 2][width // 2]
    glint_px = iris_open[glint_r][glint_x]
    lens_visible = center_px[0] > eye_base[0] + 25 and center_px[0] > center_px[2] + 15
    glint_bright = (glint_px[0] + glint_px[1] + glint_px[2]) > (center_px[0] + center_px[1] + center_px[2]) + 120
    report.line(corners_untouched and lens_visible and glint_bright,
                'makeupIris 睁眼美瞳+眼神光：圆外不变=%s，虹膜中心=%s，眼神光=%s' % (
                    corners_untouched, center_px[:3], glint_px[:3]))


# ---- 毛发边界探针 G1–G8（2026-11-21 第一批：导向系数自适应边缘锁、DOWN 2/3、region 乘 face、胡须门、背景虚化细节保留）。
# 场景与指标逐行移植自设计组探针（会话 scratchpad 的 hair2/shader/{lib,p_hair,p_lock,p_blemish,p_beard,p_flicker,
# p_final}.py 与 bg/{common,p_bg,p_bg2}.py），只用标准库：随机数按 numpy Generator（SeedSequence + PCG64 + 256 层
# ziggurat）复刻，生成的场景与探针逐字节相同，所以 G5 / G5b 能直接和冻结基线 blemish_baseline.json 比。
# 场景在本文件生成（进程池并行），gate.js 只跑生产程序并回读；指标与阈值都在这里。阈值与依据见计划 2026-11-21 §4.1。
M32, M64, M128 = 0xFFFFFFFF, (1 << 64) - 1, (1 << 128) - 1
PCG64_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def _seed_words(seed):
    """numpy SeedSequence(seed).generate_state(4, uint64)：熵按 uint32 拆分，混进 4 个字的池，再按 INIT_B 链导出。"""
    def hashmix(value, const):
        value = (value ^ const[0]) & M32
        const[0] = (const[0] * 0x931E8875) & M32
        value = (value * const[0]) & M32
        return (value ^ (value >> 16)) & M32

    def mix(x, y):
        value = (0xCA01F9DD * x - 0x4973F715 * y) & M32
        return (value ^ (value >> 16)) & M32
    entropy, n = [], int(seed)
    while n > 0:
        entropy.append(n & M32)
        n >>= 32
    entropy = entropy or [0]
    const = [0x43B0D7E5]
    pool = [hashmix(entropy[i] if i < len(entropy) else 0, const) for i in range(4)]
    for src in range(4):
        for dst in range(4):
            if src != dst:
                pool[dst] = mix(pool[dst], hashmix(pool[src], const))
    for src in range(4, len(entropy)):
        for dst in range(4):
            pool[dst] = mix(pool[dst], hashmix(entropy[src], const))
    words, chain = [], 0x8B51F9DD
    for i in range(8):
        value = (pool[i % 4] ^ chain) & M32
        chain = (chain * 0x58F38DED) & M32
        value = (value * chain) & M32
        words.append((value ^ (value >> 16)) & M32)
    return [words[2 * i] | (words[2 * i + 1] << 32) for i in range(4)]


def _ziggurat():
    """numpy 的 256 层正态 ziggurat 表（ki / wi / fi），按 Marsaglia–Tsang 递推。v 由 numpy 内置表的 wi[0] 反推，
    与内置表的相对差 ≤1e-14（ki 差 ≤18 / 2^52），抽到的正态数与 numpy 只差末位，量化成字节后逐字节相同。"""
    r, v, scale = 3.6541528853610088, 0.004928673233974652, 2.0 ** 52
    ki, wi, fi = [0] * 256, [0.0] * 256, [0.0] * 256
    dn = tn = r
    q = v / math.exp(-0.5 * dn * dn)
    ki[0], wi[0], wi[255] = int(dn / q * scale), q / scale, dn / scale
    fi[0], fi[255] = 1.0, math.exp(-0.5 * dn * dn)
    for i in range(254, 0, -1):
        dn = math.sqrt(-2.0 * math.log(v / dn + math.exp(-0.5 * dn * dn)))
        ki[i + 1] = int(dn / tn * scale)
        tn = dn
        fi[i] = math.exp(-0.5 * dn * dn)
        wi[i] = dn / scale
    return ki, wi, fi


ZIG_KI, ZIG_WI, ZIG_FI = _ziggurat()
ZIG_R, ZIG_INV_R = 3.6541528853610087963519472518, 0.27366123732975827203338247596


class NumpyRandom:
    """numpy.random.default_rng(seed) 用到的子集：normal / uniform / choice，抽样顺序与 numpy 一致
    （PCG64 XSL-RR；choice 走 32 位 Lemire，PCG64 的 32 位输出成对缓存）。"""

    def __init__(self, seed):
        words = _seed_words(seed)
        self.inc = ((((words[2] << 64) | words[3]) << 1) | 1) & M128
        state = (self.inc + ((words[0] << 64) | words[1])) & M128
        self.state = (state * PCG64_MULT + self.inc) & M128
        self.cached = None

    def next64(self):
        s = self.state = (self.state * PCG64_MULT + self.inc) & M128
        x, rot = ((s >> 64) ^ s) & M64, s >> 122
        return ((x >> rot) | (x << ((64 - rot) & 63))) & M64

    def next32(self):
        if self.cached is not None:
            value, self.cached = self.cached, None
            return value
        value = self.next64()
        self.cached = value >> 32
        return value & M32

    def double(self):
        return (self.next64() >> 11) * (1.0 / 9007199254740992.0)

    def _tail(self, r, idx, rabs):
        """ziggurat 慢路径（约 1%）：底层尾部或楔形区的拒绝采样，失败时重抽。"""
        x = -rabs * ZIG_WI[idx] if r & 1 else rabs * ZIG_WI[idx]
        if idx == 0:
            while True:
                xx = -ZIG_INV_R * math.log1p(-self.double())
                yy = -math.log1p(-self.double())
                if yy + yy > xx * xx:
                    return -(ZIG_R + xx) if (rabs >> 8) & 1 else ZIG_R + xx
        if (ZIG_FI[idx - 1] - ZIG_FI[idx]) * self.double() + ZIG_FI[idx] < math.exp(-0.5 * x * x):
            return x
        return self.normal(0.0, 1.0)

    def normal(self, loc, scale):
        return self.normals(1, loc, scale)[0]

    def normals(self, count, loc=0.0, scale=1.0):
        """count 个 loc + scale·z（numpy 的 normal(loc, scale, size) 按行优先展开）；快路径内联 PCG64 步进。"""
        out, s, inc, ki, wi = [], self.state, self.inc, ZIG_KI, ZIG_WI
        while len(out) < count:
            s = (s * PCG64_MULT + inc) & M128
            x, rot = ((s >> 64) ^ s) & M64, s >> 122
            x = ((x >> rot) | (x << ((64 - rot) & 63))) & M64
            idx, r = x & 0xFF, x >> 8
            rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
            if rabs < ki[idx]:
                z = rabs * wi[idx]
                out.append(loc + scale * (-z if r & 1 else z))
                continue
            self.state = s
            out.append(loc + scale * self._tail(r, idx, rabs))
            s = self.state
        self.state = s
        return out

    def uniform(self, low, high):
        return low + (high - low) * self.double()

    def choice(self, items):
        bound = len(items) - 1
        if bound == 0:
            return items[0]
        m = self.next32() * (bound + 1)
        if m & M32 < bound + 1:
            threshold = (M32 - bound) % (bound + 1)
            while m & M32 < threshold:
                m = self.next32() * (bound + 1)
        return items[m >> 32]


# 图像一律按通道存成扁平 float 列表（第 0 行在画面顶部），与探针的 numpy 运算顺序逐项一致。
def gauss_kernel(sigma):
    """lib.gauss_blur 的核：k = −(⌊3σ⌋+1)…⌊3σ⌋+1，归一化。"""
    n = int(3 * sigma) + 1
    weights = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-n, n + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def gauss_blur(plane, width, height, sigma):
    """lib.gauss_blur：先竖直再水平，边缘复制。"""
    kernel = gauss_kernel(sigma)
    pad = len(kernel) // 2
    rows = [plane[y * width:(y + 1) * width] for y in range(height)]
    rows = [rows[0]] * pad + rows + [rows[-1]] * pad
    out = []
    for y in range(height):
        acc = [kernel[0] * v for v in rows[y]]
        for j in range(1, len(kernel)):
            acc = [a + kernel[j] * v for a, v in zip(acc, rows[y + j])]
        row = [acc[0]] * pad + acc + [acc[-1]] * pad
        res = [kernel[0] * v for v in row[:width]]
        for j in range(1, len(kernel)):
            res = [a + kernel[j] * v for a, v in zip(res, row[j:j + width])]
        out.extend(res)
    return out


def fmean(values):
    return sum(values) / len(values)


def fstd(values):
    m = fmean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / len(values))


def rgba_bytes(planes):
    """np.clip(np.rint(rgb), 0, 255) 打包成 RGBA8（A = 255）；round() 与 rint 同为四舍六入五成双。"""
    out = bytearray(b'\xff' * (4 * len(planes[0])))
    for c in range(3):
        out[c::4] = bytes(0 if v < 0 else 255 if v > 255 else v for v in map(round, planes[c]))
    return bytes(out)


def optic_row(row):
    """[1,2,1] / 4 一行，边缘复制（相机软化）。"""
    p = [row[0]] + row + [row[-1]]
    return [0.25 * a + 0.5 * b + 0.25 * c for a, b, c in zip(p, p[1:], p[2:])]


def optic(plane, width, height):
    """lib.optic：[1,2,1]^2 / 16，先水平再竖直。"""
    rows = [optic_row(plane[y * width:(y + 1) * width]) for y in range(height)]
    rows = [rows[0]] + rows + [rows[-1]]
    return [0.25 * a + 0.5 * b + 0.25 * c for r0, r1, r2 in zip(rows, rows[1:], rows[2:]) for a, b, c in zip(r0, r1, r2)]


# 发缘 / 色渗 / 硬丝 / 发团（p_hair、p_lock）：120×24 行内恒定，第 60–62 列为边或丝（3 个相位）。
EDGE_W, EDGE_H, MASS_N = 120, 24, 96
HAIR_BROWN, HAIR_FOREHEAD = (110, 78, 55), (215, 175, 155)
HAIR_BLONDE, HAIR_TEAL = (205, 175, 125), (40, 120, 140)
HARD_STRANDS = (('lt', (140, 100, 68)), ('dk', (70, 48, 36)))


# 发缘类指标只看中间一行。
EDGE_ROW = [EDGE_H // 2 * EDGE_W + x for x in range(EDGE_W)]


def edge_scene(left, right, x0):
    return rgba_bytes([optic_row([float(left[c])] * x0 + [float(right[c])] * (EDGE_W - x0)) * EDGE_H for c in range(3)])


def hard_strand_scene(color, x0):
    """1px 硬丝（不经相机软化）落在额头皮肤上。"""
    return rgba_bytes([([float(HAIR_FOREHEAD[c])] * x0 + [float(color[c])] +
                        [float(HAIR_FOREHEAD[c])] * (EDGE_W - x0 - 1)) * EDGE_H for c in range(3)])


def hair_mass_scene(base, sigma, angle=0.3, seed=3):
    """脸外发团纹理：逐根 1px 乘性起伏（σ 为相对幅度），沿 angle 方向。"""
    rng = NumpyRandom(seed)
    columns = 400
    vals, vals2 = rng.normals(columns), rng.normals(columns)
    ca, sa = math.cos(angle), math.sin(angle)
    g = []
    for y in range(MASS_N):
        for x in range(MASS_N):
            u, v = x * ca - y * sa, x * sa + y * ca
            floor = math.floor(u + 150)
            idx, frac = int(floor) % columns, u + 150 - floor
            g.append(vals[idx] * (1 - frac) + vals[(idx + 1) % columns] * frac + 0.35 * vals2[idx] * math.sin(v / 9.0 + idx))
    sd = fstd(g)
    return rgba_bytes([[b * (1 + sigma * (x / sd)) for x in g] for b in base])


# 中等对比瑕疵（p_blemish）：224×168，56px 一格；Δ30/45/60 × 浅 / 中肤色，皮肤带 σ2 毛孔纹理 + σ1.5 噪声。
BLEM_CELL = 56
BLEM_W, BLEM_H = 4 * BLEM_CELL, 3 * BLEM_CELL
SKIN_LIGHT, SKIN_MED = (215, 175, 155), (190, 140, 115)
BLEM_DIRS = {'acne': (0.75, 1.1, 1.0), 'tiny': (1.0, 1.0, 1.0), 'circle': (0.85, 1.05, 0.95), 'fold': (0.95, 1.0, 1.0)}

# G1 矩阵（评审 2026-11-21 #0）：发色 棕 / 深棕 / 金 / 黑 × 额头肤色 浅 / 中 / 偏深，3 相位，def 与 full 两档，场景同 halo_fh
# （120×24 发缘，[1,2,1] 相机软化）。hx_* 没有头发遮罩（A 恒 255，即第一批、遮罩缺失或过期时的回落路径），hm_* 带头发遮罩
# （第二批：皮肤遮罩 A = 非头发，并走导向滤波的头发加权路径）。头发遮罩按生产形态模拟：头发覆盖率 → 216/720 分辨率
# （G1_MASK_SCALE）→ 高斯 σ1.95 遮罩像素（research 实测过渡带 3.2–9.3px 的中位）→ 在帧的 1/4 分辨率（SkinMaskPass 输出）按
# 脸外阈值 ALPHA_LOW / ALPHA_HIGH 换成 A。
G1_HAIRS = (('brown', (110, 78, 55)), ('dkbrown', (70, 48, 36)), ('blonde', (205, 175, 125)), ('black', (35, 28, 25)))
G1_SKINS = (('light', SKIN_LIGHT), ('med', SKIN_MED), ('tan', (165, 118, 92)))
G1_MASK_SCALE, G1_MASK_SIGMA = 0.3, 1.95
# 冻结基线：(old 光晕, old 边宽, cur 光晕, cur 边宽)，指标同 edge_metrics（3 相位平均，光晕 /255、边宽 px）。old = 1.24.2+44
# （0893d837 的着色器与当时的「自然」预设：磨皮 0.45、质感 0.5、立体 0.3、提亮 0.12、美白 0.35、红润 0.25、锐化 0.25、
# 饱和 1.05；full 只把磨皮改成 1.0），cur = 1.26.0+46（HEAD ace033f9，抽头 1 像素）。由设计组探针 hair2/shader/lib.py 在
# 无头 Chrome SwiftShader 上实测（脚本 hair2/fix/SH-exp/expG1base.py）；同一套探针跑当前着色器得到的数值与门禁 hx_* 逐项相同。
G1_FROZEN = {
    'brown|light': {'def': (0.84, 2.0, 5.78, 2.0), 'full': (5.97, 2.0, 16.13, 14.3)},
    'brown|med': {'def': (2.94, 2.0, 7.35, 3.7), 'full': (14.42, 16.7, 17.32, 21.3)},
    'brown|tan': {'def': (5.90, 12.0, 7.50, 12.7), 'full': (15.61, 25.0, 14.61, 22.7)},
    'dkbrown|light': {'def': (0.71, 2.0, 4.29, 2.0), 'full': (3.86, 2.0, 13.56, 3.3)},
    'dkbrown|med': {'def': (1.14, 2.0, 5.83, 2.0), 'full': (7.37, 2.0, 16.31, 14.7)},
    'dkbrown|tan': {'def': (2.76, 2.0, 7.23, 2.0), 'full': (13.51, 13.0, 17.86, 18.7)},
    'blonde|light': {'def': (0.62, 17.7, 0.65, 11.3), 'full': (1.26, 27.0, 1.43, 20.7)},
    'blonde|med': {'def': (4.07, 16.3, 5.86, 16.0), 'full': (8.96, 25.0, 9.91, 24.7)},
    'blonde|tan': {'def': (5.79, 12.0, 8.00, 15.0), 'full': (15.34, 25.0, 15.98, 24.7)},
    'black|light': {'def': (0.22, 2.0, 2.54, 2.0), 'full': (1.76, 2.0, 7.76, 2.0)},
    'black|med': {'def': (0.45, 2.0, 4.46, 2.0), 'full': (3.26, 2.0, 12.00, 2.3)},
    'black|tan': {'def': (0.57, 2.0, 5.26, 2.0), 'full': (5.29, 2.0, 14.80, 8.3)},
}
# 「不劣于 HEAD」守护的容差（两条路径、全部组合）：光晕 +0.3/255、边宽 +1px（只在 |ΔY| ≥ G1_DY_MIN 的组合上算边宽，
# 金发 × 浅肤 ΔY≈4，10–90% 边宽的分母太小没有意义）。
G1_HEAD_TOL, G1_WIDTH_TOL, G1_DY_MIN = 0.3, 1.0, 20.0
# 「不劣于 1.24.x」：旧版默认档 ≤3 的组合（评审 #0 的核验口径），带头发遮罩时默认档光晕 ≤ old + G1_OLD_TOL。
G1_OLD_COMBOS, G1_OLD_TOL = ('brown|light', 'brown|med', 'dkbrown|light', 'dkbrown|med', 'dkbrown|tan'), 0.3
# 带头发遮罩路径的「不劣于 HEAD」另有一个不可见下限：金发 × 浅肤（ΔY≈4）在 A 通道的软过渡里一侧调色、一侧不调，默认档光晕
# 0.65 → 1.05（只有 A 通道，见矩阵输出的「仅 A」列）/ 1.14（再加导向滤波头发加权），远低于标准 1 的 3/255。
G1_MASK_FLOOR = 1.5
BLEM_SCENES = [('blem_%s_d%d' % (name, delta), 100 + 1 + i * 3 + j, skin, delta)
               for i, (name, skin) in enumerate((('light', SKIN_LIGHT), ('med', SKIN_MED))) for j, delta in enumerate((30, 45, 60))]


def unit_luma(d):
    total = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
    return tuple(x / total for x in d)


def blemish_shapes():
    """每个瑕疵：m = 形状模糊后归一（像素下标 → 权重），core = m ≥ 0.7，ring = 局部参考皮肤环。"""
    def blurred(inside, x0, x1, y0, y1, sigma):
        # 形状远离图像边缘：只在包围盒外扩一个核半径的窗口里模糊，窗口外恒为 0，与整幅边缘复制的结果相同。
        pad = int(3 * sigma) + 1
        wx0, wx1, wy0, wy1 = max(x0 - pad, 0), min(x1 + pad, BLEM_W), max(y0 - pad, 0), min(y1 + pad, BLEM_H)
        ww, wh = wx1 - wx0, wy1 - wy0
        b = gauss_blur([1.0 if inside(x + 0.5, y + 0.5) else 0.0 for y in range(wy0, wy1) for x in range(wx0, wx1)],
                       ww, wh, sigma)
        return {(wy0 + j) * BLEM_W + wx0 + i: b[j * ww + i] for j in range(wh) for i in range(ww)}

    def ring(test):
        return [y * BLEM_W + x for y in range(BLEM_H) for x in range(BLEM_W) if test(x + 0.5, y + 0.5)]
    shapes = []
    for name, col, row, dia, sigma, kind in (('acne6', 0, 0, 6, 1.0, 'acne'), ('acne9', 1, 0, 9, 1.0, 'acne'),
                                             ('acne12', 2, 0, 12, 1.0, 'acne'), ('mole6', 3, 0, 6, 0.6, 'mole'),
                                             ('mole9', 0, 1, 9, 0.6, 'mole'), ('tiny2', 1, 1, 2, 0.5, 'tiny'),
                                             ('tiny3', 2, 1, 3, 0.5, 'tiny')):
        cx, cy = (col + 0.5) * BLEM_CELL, (row + 0.5) * BLEM_CELL
        rad = int(dia / 2) + 2
        g = blurred(lambda x, y: math.hypot(x - cx, y - cy) <= dia / 2, int(cx) - rad, int(cx) + rad + 1,
                    int(cy) - rad, int(cy) + rad + 1, sigma)
        top = max(max(g.values()), 1e-9)
        shapes.append({'name': name, 'kind': kind, 'm': {k: min(max(v / top, 0.0), 1.0) for k, v in g.items()},
                       'ring': ring(lambda x, y: dia / 2 + 6 <= math.hypot(x - cx, y - cy) <= dia / 2 + 12)})
    # 黑眼圈：12×36px 椭圆，σ3。
    cx, cy = BLEM_CELL, 2.5 * BLEM_CELL
    g = blurred(lambda x, y: ((x - cx) / 18.0) ** 2 + ((y - cy) / 6.0) ** 2 <= 1.0,
                int(cx) - 20, int(cx) + 21, int(cy) - 8, int(cy) + 9, 3.0)
    top = max(g.values())
    m = {k: v / top for k, v in g.items()}
    shapes.append({'name': 'circle', 'kind': 'circle', 'm': m, 'ring': [
        k for k in ring(lambda x, y: ((x - cx) / 30.0) ** 2 + ((y - cy) / 18.0) ** 2 <= 1.0 and abs(x - cx) < BLEM_CELL)
        if m.get(k, 0.0) < 0.02]})
    # 法令纹阴影：8px 宽斜线，σ2。
    fx, fy = 3 * BLEM_CELL, 2.5 * BLEM_CELL

    def across(x, y):
        return abs((x - fx) * 0.6 - (y - fy) * 0.8)

    def along(x, y):
        return abs((x - fx) * 0.8 + (y - fy) * 0.6)
    g = blurred(lambda x, y: across(x, y) <= 4.0 and along(x, y) <= 20, int(fx) - 22, int(fx) + 23, int(fy) - 22, int(fy) + 23, 2.0)
    top = max(g.values())
    shapes.append({'name': 'fold', 'kind': 'fold', 'm': {k: v / top for k, v in g.items()},
                   'ring': ring(lambda x, y: 10 <= across(x, y) <= 16 and along(x, y) <= 14 and x > 2 * BLEM_CELL + 4)})
    for shape in shapes:
        shape['core'] = sorted(k for k, v in shape['m'].items() if v >= 0.7)
    return shapes


def blemish_pixels(shapes):
    return sorted({k for shape in shapes for k in shape['ring'] + shape['core']})


def blemish_scene(shapes, seed, skin, delta):
    rng = NumpyRandom(seed)
    tex = gauss_blur(rng.normals(BLEM_W * BLEM_H), BLEM_W, BLEM_H, 0.8)
    sd = fstd(tex)
    planes = [[float(skin[c]) + v / sd * 2.0 for v in tex] for c in range(3)]
    mole = unit_luma(tuple(float(skin[c]) - (90.0, 60.0, 45.0)[c] for c in range(3)))
    for shape in shapes:
        d = mole if shape['kind'] == 'mole' else unit_luma(BLEM_DIRS[shape['kind']])
        for k, weight in shape['m'].items():
            dm = delta * weight
            for c in range(3):
                planes[c][k] -= dm * d[c]
    noise = rng.normals(BLEM_W * BLEM_H, 0.0, 1.5)
    return rgba_bytes([[a + b for a, b in zip(plane, noise)] for plane in planes])


# 胡茬（p_beard）：144×96，胡茬区 x∈[24,120)、y∈[20,76)；4× 超采样画短线，覆盖率 ≥0.6 为胡茬芯，上下各 10 行净皮肤作参照。
BEARD_W, BEARD_H, BEARD_SS = 144, 96, 4
BEARD_X0, BEARD_X1, BEARD_Y0, BEARD_Y1 = 24, 120, 20, 76
BEARD_HAIR = (('blk', (38, 32, 31)), ('brn', (72, 52, 42)))
# 种子按 p_beard.KINDS 顺序递增（stub_lo、stub_mid、stub_hi、beard、stub_soft、beard_soft 各 4 个场景）；门禁只用
# 稀疏茬（覆盖率约 4%，[1,2,1]^2 软化）与软茬（约 10%，σ1.0 软化）。
BEARD_KINDS = (('stub_lo', 0.04, None, 0), ('stub_soft', 0.10, 1.0, 16))
BEARD_SCENES = [('%s_%s_%s' % (kind, hname, sname), kind, frac, blur, seed + 1 + i * 2 + j, hair, skin)
                for kind, frac, blur, seed in BEARD_KINDS for i, (hname, hair) in enumerate(BEARD_HAIR)
                for j, (sname, skin) in enumerate((('med', SKIN_MED), ('light', SKIN_LIGHT)))]


# 胡茬区（四周各收 3px）与上下各 10 行净皮肤。
BEARD_REGION = [y * BEARD_W + x for y in range(BEARD_Y0 + 3, BEARD_Y1 - 3) for x in range(BEARD_X0 + 3, BEARD_X1 - 3)]
BEARD_BARE = [y * BEARD_W + x for y in list(range(4, 14)) + list(range(BEARD_H - 14, BEARD_H - 4)) for x in range(4, BEARD_W - 4)]
BEARD_PIXELS = BEARD_REGION + BEARD_BARE


def beard_coverage(seed, count):
    """p_beard.strands_cov：count 根短线（长 1.5–4px、宽 1/1.5/2px、近竖直），返回（覆盖率，逐像素发色系数）。"""
    rng = NumpyRandom(seed)
    ss, wide, tall = BEARD_SS, BEARD_W * BEARD_SS, BEARD_H * BEARD_SS
    tone = {}
    for _ in range(count):
        cx, cy = rng.uniform(BEARD_X0 + 2, BEARD_X1 - 2), rng.uniform(BEARD_Y0 + 2, BEARD_Y1 - 2)
        length = rng.uniform(1.5, 4.0)
        half = rng.choice((1.0, 1.5, 2.0)) / 2
        theta = math.pi / 2 + rng.normal(0, 0.6)
        dx, dy = math.cos(theta), math.sin(theta)
        ax, ay = cx - 0.5 * length * dx, cy - 0.5 * length * dy
        bx0 = int(max(0, (min(ax, ax + length * dx) - 3) * ss)); bx1 = int(min(wide, (max(ax, ax + length * dx) + 3) * ss))
        by0 = int(max(0, (min(ay, ay + length * dy) - 3) * ss)); by1 = int(min(tall, (max(ay, ay + length * dy) + 3) * ss))
        value = rng.normal(1.0, 0.15)
        # 与探针同用 hypot 判界；先用平方距离排除明显在内 / 在外的子像素。
        inner, outer = (half * (1 - 1e-9)) ** 2, (half * (1 + 1e-9)) ** 2
        for j in range(by0, by1):
            Y = (j + 0.5) / ss
            for i in range(bx0, bx1):
                X = (i + 0.5) / ss
                t = min(max(((X - ax) * dx + (Y - ay) * dy) / length, 0.0), 1.0)
                ex, ey = X - (ax + t * length * dx), Y - (ay + t * length * dy)
                e2 = ex * ex + ey * ey
                if e2 < inner or (e2 <= outer and math.hypot(ex, ey) <= half):
                    tone[j * wide + i] = value
    cover, tsum = [0.0] * (BEARD_W * BEARD_H), [0.0] * (BEARD_W * BEARD_H)
    for k, value in tone.items():
        p = (k // wide // ss) * BEARD_W + (k % wide) // ss
        cover[p] += 1.0
        tsum[p] += value
    return [c / (ss * ss) for c in cover], [t / c if c > 0 else 1.0 for t, c in zip(tsum, cover)]


def beard_scene(frac, blur, seed, hair, skin):
    area = (BEARD_X1 - BEARD_X0) * (BEARD_Y1 - BEARD_Y0)
    count = int(frac * area / (0.5 * (1.5 + 4.0) * 1.5 * 0.85) * 1.1)
    cover, tone = beard_coverage(seed, count)
    planes = []
    for s, h in zip(skin, hair):
        plane = [s * (1 - c) + min(max(h * t, 0.0), 255.0) * c for c, t in zip(cover, tone)]
        planes.append(optic(plane, BEARD_W, BEARD_H) if blur is None else gauss_blur(plane, BEARD_W, BEARD_H, blur))
    noise = NumpyRandom(seed + 99).normals(BEARD_W * BEARD_H, 0.0, 1.5)
    return rgba_bytes([[a + b for a, b in zip(plane, noise)] for plane in planes]), cover


# 时域闪烁（p_flicker）：64×48 静态皮肤 + 逐帧独立噪声 30 帧，不含时域降噪（最坏情况）。
FLICK_W, FLICK_H, FLICK_FRAMES = 64, 48, 30
FLICK_SCENES = (('plat_light_s2', SKIN_LIGHT, 2.0, 1, 2.0), ('dark15', (54, 36, 28), 1.5, 2, 3.0),
                ('dark11', (40, 26, 20), 1.2, 3, 3.0))


# 闪烁只统计四周各收 10px 的区域。
FLICK_PIXELS = [y * FLICK_W + x for y in range(10, FLICK_H - 10) for x in range(10, FLICK_W - 10)]


def flicker_frames(name, skin, texture, seed, sigma):
    rng = NumpyRandom(seed)
    t = gauss_blur(rng.normals(FLICK_W * FLICK_H), FLICK_W, FLICK_H, 0.8)
    sd = fstd(t)
    base = [[float(c) + v / sd * texture for v in t] for c in skin]
    rng = NumpyRandom(zlib.crc32(name.encode()))
    frames = []
    for _ in range(FLICK_FRAMES):
        z = rng.normals(FLICK_W * FLICK_H * 3, 0.0, sigma)
        frames.append(rgba_bytes([[b + z[i * 3 + c] for i, b in enumerate(base[c])] for c in range(3)]))
    return frames


def hair_scene_task(task):
    """进程池任务：生成一组场景，返回 [(场景名, 宽, 高, RGBA 字节, 附带数据)]。"""
    kind, args = task[0], task[1:]
    if kind == 'edges':
        out = []
        for phase in (0, 1, 2):
            x0 = 60 + phase
            out.append(('halo_fh_p%d' % phase, EDGE_W, EDGE_H, edge_scene(HAIR_BROWN, HAIR_FOREHEAD, x0), x0))
            out.append(('bleed_p%d' % phase, EDGE_W, EDGE_H, edge_scene(HAIR_BLONDE, HAIR_TEAL, x0), x0))
            for name, color in HARD_STRANDS:
                out.append(('hard_%s_p%d' % (name, phase), EDGE_W, EDGE_H, hard_strand_scene(color, x0), x0))
        for hname, hair in G1_HAIRS:
            for sname, skin in G1_SKINS:
                for phase in (0, 1, 2):
                    x0 = 60 + phase
                    data = edge_scene(hair, skin, x0)
                    out.append(('hx_%s_%s_p%d' % (hname, sname, phase), EDGE_W, EDGE_H, data, x0))
                    out.append(('hm_%s_%s_p%d' % (hname, sname, phase), EDGE_W, EDGE_H, data, x0))
        out.append(('mass_brown', MASS_N, MASS_N, hair_mass_scene(HAIR_BROWN, 0.15), None))
        return out
    if kind == 'blemish':
        shapes, name, seed, skin, delta = args
        return [(name, BLEM_W, BLEM_H, blemish_scene(shapes, seed, skin, delta), None)]
    if kind == 'beard':
        name, _, frac, blur, seed, hair, skin = args
        data, cover = beard_scene(frac, blur, seed, hair, skin)
        return [(name, BEARD_W, BEARD_H, data, cover)]
    name, skin, texture, seed, sigma = args
    return [('fl_%s_f%02d' % (name, f), FLICK_W, FLICK_H, data, None)
            for f, data in enumerate(flicker_frames(name, skin, texture, seed, sigma))]


# 档位（设计组口径）：def = 默认「自然」(NATURAL)；mid80 / full 只把磨皮改成 0.8 / 1.0；so55 / so100 只开磨皮（质感 0.5）。
def hair_look(smoothing, natural=True):
    if natural:
        return {'params': [16, smoothing] + NATURAL['params'][2:], 'details': NATURAL['details'], 'tone': NATURAL['tone']}
    return {'params': [16, smoothing, 0.0, 0.5], 'details': [0.0, 0.0, 1.0, 0.0], 'tone': [0.0, 0.0, 0.0, 0.0]}


HAIR_LOOKS = {'def': hair_look(0.55), 'mid80': hair_look(0.8), 'full': hair_look(1.0),
              'so55': hair_look(0.55, False), 'so100': hair_look(1.0, False)}


def mask_bytes(face, excl, hi):
    """SkinMaskPass 输出（与场景同尺寸的常量 RGBA）：R = face、G = excl、B = hiPlane、A = 255（非头发）。"""
    return [round(face * 255), round(excl * 255), round(hi * 255), 255]


# G8 背景虚化（bg/p_bg.py 的几何）：720×1280，深色发团 0.12 + 脸 0.70 + 衣服 0.30；顶部 15 根 1/2/3px 径向飞发（48px），
# 两侧各 25 根 1px 碎发；背景为 6px 棋盘。人像遮罩是模拟的：覆盖率 → 216×384 → 256² → PSF σ1 → 216×384 → u8。
BG_W, BG_H, BG_SS = 720, 1280, 4
BG_HEAD = (360.0, 560.0, 200.0, 250.0)
BG_FACE = (360.0, 640.0, 150.0, 190.0)
BG_TONES = (0.12, 0.70, 0.30)
# 裁剪区 x 60..660、y 230..1030：头部与两侧肩部（肩部光环取样到 y≈1006）。
BG_CROP = (60, 230, 600, 800)
BG_OVAL = [BG_FACE[0] / BG_W, 1.0 - BG_FACE[1] / BG_H, BG_FACE[2] / BG_W, BG_FACE[3] / BG_W]
# 背景：低对比棋盘 0.55/0.80（看飞发保留与未虚化光环）、高对比棋盘 0.20/0.95（看头部旁误保留）。
BG_CHECKERS = (('checker', 0.55, 0.80), ('hc', 0.20, 0.95))


def bg_ellipse_point(theta):
    """发团椭圆上 theta 方向的点与外法向。"""
    cx, cy, rx, ry = BG_HEAD
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 / math.sqrt((c / rx) ** 2 + (s / ry) ** 2)
    px, py = cx + t * c, cy + t * s
    nx, ny = px - cx, (py - cy) * (rx / ry) ** 2
    n = math.hypot(nx, ny)
    return px, py, nx / n, ny / n


def bg_strands():
    spec = [('iso', math.radians(deg), 48.0, (1.0, 2.0, 3.0)[i % 3]) for i, deg in enumerate(range(-160, -19, 10))]
    for base in (180.0, 0.0):
        spec += [('fringe', math.radians(base + k * 1.2), 24.0 + 6.0 * ((k * 7) % 3), 1.0) for k in range(-12, 13)]
    return spec


# G8 第二批合并道变体（评审 #19）：头发遮罩高斯软化 σ（遮罩像素），覆盖 research 实测过渡带 3.2 / 5 / 9.3px。
BG_HAIR_SIGMAS = (1.25, 1.95, 3.63)
# 时域闪烁上限：生产（有头部先验）相对细节保留关闭的倍数。实测高对比棋盘 8.8×、文字 6.4×（先验区内的已知代价），
# 上限 10× 只防以后放宽细节保留时悄悄变宽；修复前后的区分靠先验区外的逐像素相等（见 check_bg_temporal）。
BG_FLICKER_RATIO = 10.0
# G8 细节保留 β 随「人脸近期可见」系数 k 渐变（评审复核）：在这些 k 上扫 β = detailKeep(true, k)，低对比棋盘（飞发所在）。
BG_SWEEP = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
BG_SWEEP_CHECKER = 'checker'
# G8 时域（评审 #1）：8 帧抖动遮罩；背景为高对比棋盘与文字（墨 0.15 / 纸 0.85）。
BG_TEMPORAL = {'frames': 8, 'coreSigma': 2.0, 'haloSigma': 5.0, 'haloGain': 0.30, 'shift': 0.5, 'noise': 2.0 / 255,
               'seed': 7, 'backgrounds': [['hc', 0.20, 0.95], ['text', 0.15, 0.85, 'text']]}


def bg_input(constants, hair):
    segments = []
    for _, theta, length, width in bg_strands():
        px, py, nx, ny = bg_ellipse_point(theta)
        segments.append([px - 6 * nx, py - 6 * ny, px + length * nx, py + length * ny, width])
    # common.gauss_blur 的核（半径 ⌈3σ⌉），PSF σ1。
    psf = [math.exp(-0.5 * (k / 1.0) ** 2) for k in range(-3, 4)]
    return {'width': BG_W, 'height': BG_H, 'ss': BG_SS, 'head': BG_HEAD, 'face': BG_FACE, 'tones': BG_TONES,
            'strands': segments, 'checkers': BG_CHECKERS, 'mask': [216, 384, 256], 'psf': [w / sum(psf) for w in psf],
            'crop': BG_CROP, 'pick': bg_pixels(), 'uniforms': [1.0, constants['low'], constants['high'], constants['detail']],
            # 脸椭圆（帧纹理坐标 y 向上，半径以帧宽为单位，同主合成 uOval）供 BG_MIX 的头部先验。
            'oval': BG_OVAL, 'axis': [1.0, 0.0, BG_W / BG_H, 0.0],
            'hairUniforms': [hair['HAIR_LOW'], hair['HAIR_HIGH']], 'hairSigmas': list(BG_HAIR_SIGMAS), 'temporal': BG_TEMPORAL,
            # β 扫描（check_bg_detail_sweep）：低对比棋盘、带飞发画面与遮罩，β = detailKeep(true, k)，另跑一次没有人脸（β 0、NO_FACE）。
            'detailSweep': {'checker': BG_SWEEP_CHECKER, 'betas': [detail_keep(constants['detail'], k) for k in BG_SWEEP],
                            'noFace': constants['noFace'], 'noFaceAxis': constants['noFaceAxis']}}


def f32(value):
    """按 IEEE 单精度舍入（Kotlin Float 运算的结果）。"""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def detail_keep(detail, weight):
    """BackgroundBlurPass.detailKeep(face = true, weight) 的 Python 复刻（写法由 background_constants 静态核对）：
    weight > 0 时 f32(DETAIL_KEEP × f32(min(weight, 1)))，否则（含 NaN）0。"""
    return f32(detail * f32(min(weight, 1.0))) if weight > 0 else 0.0


def background_constants(kotlin_dir, report):
    """从 BackgroundBlurPass.kt 读 MASK_LOW / MASK_HIGH / DETAIL_KEEP 与无人脸时的脸椭圆，并核对 BG_MIX 的 β 写法、BG_PREMASK 传 0。
    β（uBackground.w）= detailKeep(face != null, faceWeight)：没有人脸为 0；有人脸时 DETAIL_KEEP × clamp(faceWeight, 0, 1)
    （「人脸近期可见」系数，评审复核：D3 不再随人脸有无单帧硬切），faceWeight 缺省 1f（系数 1 时 β = DETAIL_KEEP，与之前逐字节相同）。
    G8 按同一公式扫 β（check_bg_detail_sweep）。"""
    code = kotlin_code(os.path.join(kotlin_dir, 'BackgroundBlurPass.kt'))
    found = dict(re.findall(r'const val (MASK_LOW|MASK_HIGH|DETAIL_KEEP) = ([0-9.]+)f', code))
    no_face = re.search(r'\bval NO_FACE = floatArrayOf\(([^()]*)\)', code)
    no_axis = re.search(r'glUniform4f\(mixAt\.faceAxis, ([-0-9.f, ]+)\)', code)
    if sorted(found) != ['DETAIL_KEEP', 'MASK_HIGH', 'MASK_LOW'] or not no_face or not no_axis:
        report.line(False, 'BackgroundBlurPass.kt 的 MASK_LOW / MASK_HIGH / DETAIL_KEEP / NO_FACE / 无人脸 uFaceAxis 无法解析')
        return None
    # BG_MIX 的 β：有人脸时 DETAIL_KEEP × 系数，没有人脸（头部先验无处可依）时 0；脸椭圆按 face[0..6] 传 uFaceOval / uFaceAxis。
    beta = (re.search(r'\bfun detailKeep\(face: Boolean, faceWeight: Float\): Float =\s*'
                      r'if \(face && faceWeight > 0f\) DETAIL_KEEP \* minOf\(faceWeight, 1f\) else 0f\s*\n', code) is not None and
            re.search(r'\bfun render\([^()]*\bface: FloatArray\? = null, faceWeight: Float = 1f\): Int', code) is not None and
            re.findall(r'\bval detail = ([^\n]+)', code) == ['detailKeep(face != null, faceWeight)'] and
            len(re.findall(r'\bdetailKeep\(', code)) == 2)
    mix = (re.search(r'glUniform4f\(mixAt\.background, .*?, MASK_LOW, MASK_HIGH, detail\)', code) is not None and
           re.search(r'glUniform4f\(mixAt\.faceOval, face\[0\], face\[1\], face\[2\], face\[3\]\)', code) is not None and
           re.search(r'glUniform4f\(mixAt\.faceAxis, face\[4\], face\[5\], face\[6\], 0f\)', code) is not None)
    premask = re.search(r'glUniform4f\(copyAt\.background, .*?, MASK_LOW, MASK_HIGH, 0f\)', code) is not None
    report.line(beta and mix and premask,
                'BackgroundBlurPass.kt：BG_MIX 的 uBackground.w = detailKeep(face != null, faceWeight)，即没有人脸为 0、有人脸为 '
                'DETAIL_KEEP × clamp(faceWeight, 0, 1)、faceWeight 缺省 1f=%s；脸椭圆接 uFaceOval / uFaceAxis=%s；BG_PREMASK 传 0=%s' % (
                    beta, mix, premask))
    numbers = lambda text: [float(v.strip().rstrip('f')) for v in text.split(',')]
    return {'low': float(found['MASK_LOW']), 'high': float(found['MASK_HIGH']), 'detail': float(found['DETAIL_KEEP']),
            'noFace': numbers(no_face.group(1)), 'noFaceAxis': numbers(no_axis.group(1))}


def g1_hair_alpha(x0, lo, hi):
    """G1 矩阵 hm_* 的皮肤遮罩 A（帧的 1/4 分辨率，按列）：发缘左侧是头发，见 G1_MASK_SCALE / G1_MASK_SIGMA 的说明。"""
    hw = round(EDGE_W * G1_MASK_SCALE)
    cover = [fmean([1.0 if (m + (j + 0.5) / 16) / G1_MASK_SCALE < x0 else 0.0 for j in range(16)]) for m in range(hw)]
    hair = gauss_blur(cover, hw, 1, G1_MASK_SIGMA)
    mw, out = EDGE_W // 4, []
    for i in range(mw):
        u = min(max((i + 0.5) / mw * hw - 0.5, 0.0), hw - 1.0)
        i0 = int(math.floor(u))
        i1, f = min(i0 + 1, hw - 1), u - i0
        out.append(round(255 * (1 - smooth01(lo, hi, hair[i0] * (1 - f) + hair[i1] * f))))
    return out


def hair_edge_input(background, hair):
    """生成全部场景（进程池并行，本机约 5 s），返回 (INPUT.hairEdge, 供指标用的上下文)。"""
    shapes = blemish_shapes()
    tasks = [('edges',)] + [('blemish', shapes) + spec for spec in BLEM_SCENES] + \
            [('beard',) + spec for spec in BEARD_SCENES] + [('flicker',) + spec for spec in FLICK_SCENES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        groups = list(pool.map(hair_scene_task, tasks))
    looks = {name: ['so55', 'so100'] for name in ('halo_fh_p0', 'halo_fh_p1', 'halo_fh_p2')}
    looks.update({name: ['def', 'full'] for name in ('bleed_p0', 'bleed_p1', 'bleed_p2')})
    looks.update({name: ['def', 'mid80', 'full'] for name, *_ in BLEM_SCENES})
    # G7 也跑满档（评审 #20）；G1 矩阵两档。
    looks.update({name: ['def', 'full'] for name, *_ in BEARD_SCENES})
    looks.update({'%s_%s_%s_p%d' % (kind, h, k, p): ['def', 'full'] for kind in ('hx', 'hm') for h, _ in G1_HAIRS
                  for k, _ in G1_SKINS for p in (0, 1, 2)})
    # 头发加权路径另跑一遍的档位（帧名后缀 |w）：hm_* 是 G1 矩阵带头发遮罩的主场景；瑕疵、闪烁、胡茬场景在 A 恒 255（没有头发）
    # 时走加权路径，守「加权只改发缘、不改皮肤」的代价（G5 / G5b / G6 / G7 照原阈值再判一遍）。
    weighted = {name: ['def', 'full'] for name in looks if name.startswith('hm_')}
    weighted.update({name: ['def', 'mid80', 'full'] for name, *_ in BLEM_SCENES})
    weighted.update({name: ['def', 'full'] for name, *_ in BEARD_SCENES})
    masks = {'bleed': mask_bytes(0, 0, 0), 'mass': mask_bytes(0, 0, 0), 'stub': mask_bytes(1, 0, 0.25)}
    # 只回读指标用到的像素（整帧回读会让 --dump-dom 的输出到 20MB、多花十几秒）：下标为行优先像素序号，None 为整帧。
    picks = {'halo': EDGE_ROW, 'bleed': EDGE_ROW, 'hard': EDGE_ROW, 'mass': None, 'blem': blemish_pixels(shapes),
             'stub': BEARD_PIXELS, 'fl': FLICK_PIXELS, 'hx': EDGE_ROW, 'hm': EDGE_ROW}
    scenes, sources, extra, dims = [], {}, {}, {}
    for name, width, height, data, more in (item for group in groups for item in group):
        mask = next((m for key, m in masks.items() if name.startswith(key)), mask_bytes(1, 0, 0.4))
        scene = {'name': name, 'w': width, 'h': height, 'rgba': base64.b64encode(data).decode(),
                 'mask': mask, 'looks': looks.get(name, ['def']), 'pick': name.split('_')[0]}
        if name.startswith('hm_'):
            scene.update(maskA=g1_hair_alpha(more, hair['ALPHA_LOW'], hair['ALPHA_HIGH']), maskSize=[EDGE_W // 4, EDGE_H // 4])
        if name.startswith('fl_'):
            scene['weighted'] = ['def']
        elif name in weighted:
            scene['weighted'] = weighted[name]
        scenes.append(scene)
        sources[name] = data
        extra[name] = more
        dims[name] = (width, height)
    # 头发遮罩 A3–A7 的整帧场景复用上面的源图，走同一套帧流程（回读整帧）。
    scenes += hair_a_scenes(sources, dims)
    picks['hairA'] = None
    return ({'looks': HAIR_LOOKS, 'scenes': scenes, 'picks': picks, 'bg': bg_input(background, hair)},
            {'sources': sources, 'extra': extra, 'shapes': shapes, 'picks': picks, 'background': background})


def luma_at(data, index):
    i = 4 * index
    return 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]


def luma_plane(data):
    return [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in zip(data[0::4], data[1::4], data[2::4])]


def edge_metrics(src, out, x0):
    """p_hair.edge_metrics：中间行，边外 3..20px 的 max|ΔY|（扣掉同侧远场整体偏移）与输出 10–90% 边宽。"""
    y = EDGE_H // 2
    li = [luma_at(src, y * EDGE_W + x) for x in range(EDGE_W)]
    lo = [luma_at(out, y * EDGE_W + x) for x in range(EDGE_W)]
    d = [o - i for o, i in zip(lo, li)]
    dl, dr = fmean(d[4:x0 - 25]), fmean(d[x0 + 26:EDGE_W - 4])
    halo = max(max(abs(v - dl) for v in d[x0 - 20:x0 - 2]), max(abs(v - dr) for v in d[x0 + 3:x0 + 21]))
    lref, rref = fmean(lo[4:x0 - 25]), fmean(lo[x0 + 26:EDGE_W - 4])
    width = sum(1 for v in lo[x0 - 30:x0 + 30] if 0.1 < (v - rref) / (lref - rref) < 0.9)
    return halo, width


def bleed_width(src, out, x0):
    """p_hair.bleed_metrics：金发侧（离边 ≥1px，最近 40px）扣掉远场偏移后任一通道偏差 ≥3/255 的像素数。"""
    row = (EDGE_H // 2) * EDGE_W
    d = [[out[4 * (row + x) + c] - src[4 * (row + x) + c] for c in range(3)] for x in range(EDGE_W)]
    far = [fmean([d[x][c] for x in range(4, x0 - 25)]) for c in range(3)]
    return sum(1 for x in range(x0 - 40, x0 - 1) if max(abs(d[x][c] - far[c]) for c in range(3)) >= 3.0)


def strand_keep(src, out, x0):
    """p_hair.strand_keep：发丝芯与两侧皮肤（离丝 20–30px）的 RGB 距离，输出 / 输入。"""
    row = (EDGE_H // 2) * EDGE_W
    ref = list(range(x0 - 30, x0 - 20)) + list(range(x0 + 21, x0 + 31))

    def contrast(data):
        around = [fmean([data[4 * (row + x) + c] for x in ref]) for c in range(3)]
        return math.sqrt(sum((data[4 * (row + x0) + c] - around[c]) ** 2 for c in range(3)))
    return contrast(out) / contrast(src)


def highpass_std(data, size, lo, hi):
    """lib.hp_std：luma − σ4 高斯的 [lo, hi)² 区域标准差。"""
    plane = luma_plane(data)
    base = gauss_blur(plane, size, size, 4.0)
    return fstd([plane[y * size + x] - base[y * size + x] for y in range(lo, hi) for x in range(lo, hi)])


def blemish_keep(src, out, shape):
    """p_blemish.keep：（参考环均值 − 瑕疵芯均值）输出 / 输入，越低越好。"""
    def depth(data):
        return fmean([luma_at(data, k) for k in shape['ring']]) - fmean([luma_at(data, k) for k in shape['core']])
    return depth(out) / depth(src)


def beard_contrast(src, out, cover):
    """p_beard.metrics 的 ctr：（净皮肤均值 − 胡茬芯均值）输出 / 输入；芯 = 胡茬区内覆盖率 ≥0.6（不足时取最大值的一半）。"""
    top = max(cover[k] for k in BEARD_REGION)
    core = [k for k in BEARD_REGION if cover[k] >= (0.6 if top >= 0.6 else 0.5 * top)]

    def depth(data):
        return fmean([luma_at(data, k) for k in BEARD_BARE]) - fmean([luma_at(data, k) for k in core])
    return depth(out) / depth(src)


def flicker_ratio(sources, outputs):
    """p_flicker 的 rms：区域内输出逐像素帧差 RMS / 输入帧差 RMS。"""
    def rms(frames):
        lumas = [[luma_at(f, k) for k in FLICK_PIXELS] for f in frames]
        diffs = [(b - a) ** 2 for f0, f1 in zip(lumas, lumas[1:]) for a, b in zip(f0, f1)]
        return math.sqrt(fmean(diffs))
    return rms(outputs) / rms(sources)


def bg_flyaway_points():
    """p_bg 的飞发取样点：每根 iso 飞发离边 2–48px 每 2px 一组 3×3 邻域（裁剪区内的像素下标）→ [(粗细, [组, …])]。"""
    x0, y0, cw, _ = BG_CROP
    points = []
    for kind, theta, length, width in bg_strands():
        if kind == 'iso':
            px, py, nx, ny = bg_ellipse_point(theta)
            groups = []
            for dist in range(2, int(length) + 1, 2):
                xi, yi = int(round(px + dist * nx - x0 - 0.5)), int(round(py + dist * ny - y0 - 0.5))
                groups.append([(yi + dy) * cw + xi + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            points.append((width, groups))
    return points


def bg_profile_points(bases):
    """光环取样点：每个起点 (px, py, nx, ny) 沿外法向 1–40px、切向 ±6px（裁剪区内的像素下标）→ {离边距离: [下标, …]}。"""
    x0, y0, cw, _ = BG_CROP
    points = {}
    for px, py, nx, ny in bases:
        tx, ty = -ny, nx
        for dist in range(1, 41):
            for t in range(-6, 7):
                x, y = px + dist * nx + t * tx - x0, py + dist * ny + t * ty - y0
                points.setdefault(dist, []).append(int(math.floor(y)) * cw + int(math.floor(x)))
    return points


def bg_halo_points(angles):
    """p_bg 的光环取样点：发团椭圆上各方向的外法向（同 bg_profile_points）。"""
    return bg_profile_points([bg_ellipse_point(math.radians(deg)) for deg in angles])


BG_HALO_SIDES = ((30, 40, 50), (150, 140, 130))
# 肩部（评审 #1：细节保留曾作用于整个轮廓）：衣服轮廓 x = 360 ± (160 + 0.6·(y − 860))，取 y = 900 / 950 / 1000，外法向 (±1, −0.6)。
BG_SHOULDER_YS = (900.0, 950.0, 1000.0)


def bg_shoulder_bases(side):
    n = math.hypot(1.0, 0.6)
    return [(BG_HEAD[0] + side * (160.0 + 0.6 * (y - 860.0)), y, side / n, -0.6 / n) for y in BG_SHOULDER_YS]


BG_HEAD_SIDES = [[bg_ellipse_point(math.radians(deg)) for deg in angles] for angles in BG_HALO_SIDES]
BG_SHOULDER_SIDES = [bg_shoulder_bases(1), bg_shoulder_bases(-1)]


def bg_pixels():
    picked = {k for _, groups in bg_flyaway_points() for group in groups for k in group}
    picked.update(k for sides in (BG_HEAD_SIDES, BG_SHOULDER_SIDES) for bases in sides
                  for ks in bg_profile_points(bases).values() for k in ks)
    return sorted(picked)


def bg_flyaway_keep(crops):
    """p_bg 的 iso_{1,2,3}px：每组 3×3 邻域取输入差最大的点，保留率 = |带飞发输出 − 无飞发画面同遮罩输出| /
    |带飞发输入 − 无飞发输入|（输入差 ≥40 才计），逐根平均后按粗细平均。"""
    vis = {}
    for width, groups in bg_flyaway_points():
        vals = []
        for group in groups:
            best = None
            for k in group:
                din = abs(crops['in_s'][k] - crops['in_n'][k])
                if best is None or din > best[0]:
                    best = (din, abs(crops['out'][k] - crops['cross'][k]))
            if best[0] >= 40:
                vals.append(best[1] / best[0])
        if vals:
            vis.setdefault(width, []).append(fmean(vals))
    return {width: fmean(v) for width, v in vis.items()}


def bg_ratio_profile(crops, bases, edge='edge'):
    """离边各距离上 |输出 − 原画| / |全虚化 − 原画| 的均值（只统计全虚化与原画差 >12/255 的点）。"""
    profile = {}
    for dist, ks in bg_profile_points(bases).items():
        for k in ks:
            o, i, f = crops[edge][k], crops['in_n'][k], crops['full'][k]
            if abs(f - i) > 12:
                profile.setdefault(dist, []).append(abs(o - i) / abs(f - i))
    return {d: fmean(v) for d, v in profile.items()}


def bg_halo50(crops, sides=None, edge='edge'):
    """p_bg 的 halo50：发团外侧「虚化不到全虚化的一半」的最远距离（px），右侧 30/40/50° 与左侧 150/140/130° 各取一次再平均。
    只统计全虚化与原画差 >12/255 的点。sides 换成肩部起点时量肩部。"""
    sides = BG_HEAD_SIDES if sides is None else sides
    return fmean([max([d for d, v in bg_ratio_profile(crops, bases, edge).items() if v < 0.5] or [0]) for bases in sides])


def bg_halo_run(crops, sides, edge):
    """从边缘起「连续」虚化不到一半的长度（px）：1px 起逐距离看，遇到第一个 ratio ≥ 0.5 就停（评审 #19：合并道打开后头旁 |f − i|
    很小、比值噪声大，halo50 的「最远距离」会被孤立噪声点拉远）。两侧取大。"""
    runs = []
    for bases in sides:
        profile, run = bg_ratio_profile(crops, bases, edge), 0
        for d in range(1, 41):
            if d in profile and profile[d] >= 0.5:
                break
            run = d
        runs.append(run)
    return max(runs)


def bg_keep_reach(plane, sides, low, high):
    """合并遮罩（216×384 的 R，第 0 行在画面顶部）沿离边法向的 keep = smoothstep(low, high, R) 均值 > 0.5 的最远距离（px），
    按帧像素中心双线性取样（同 GL_LINEAR）；两侧取大。"""
    mw, mh = 216, 384

    def sample(fx, fy):
        u = min(max(fx / BG_W * mw - 0.5, 0.0), mw - 1.0)
        v = min(max(fy / BG_H * mh - 0.5, 0.0), mh - 1.0)
        x0, y0 = int(u), int(v)
        x1, y1, a, b = min(x0 + 1, mw - 1), min(y0 + 1, mh - 1), u - int(u), v - int(v)
        return ((plane[y0 * mw + x0] * (1 - a) + plane[y0 * mw + x1] * a) * (1 - b) +
                (plane[y1 * mw + x0] * (1 - a) + plane[y1 * mw + x1] * a) * b) / 255.0
    reach = []
    for bases in sides:
        profile = {}
        for px, py, nx, ny in bases:
            tx, ty = -ny, nx
            for dist in range(1, 41):
                for t in range(-6, 7):
                    x, y = px + dist * nx + t * tx, py + dist * ny + t * ty
                    profile.setdefault(dist, []).append(smooth01(low, high, sample(math.floor(x) + 0.5, math.floor(y) + 0.5)))
        reach.append(max([d for d, v in profile.items() if fmean(v) > 0.5] or [0]))
    return max(reach)


def unpick(rgb, pick, count):
    """gate.js 按下标回读的 RGB → 整帧 RGBA（未回读的像素为 0），指标仍按像素下标取值。"""
    out = bytearray(4 * count)
    for n, k in enumerate(range(count) if pick is None else pick):
        out[4 * k:4 * k + 3] = rgb[3 * n:3 * n + 3]
    return bytes(out)


def unpick_crop(values, pick):
    """gate.js 只回传取样点的值（bg_pixels 的顺序）→ 整个裁剪区（未取样处为 0）。"""
    out = bytearray(BG_CROP[2] * BG_CROP[3])
    for k, v in zip(pick, values):
        out[k] = v
    return out


def load_blemish_baseline():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blemish_baseline.json'), encoding='utf-8') as handle:
        return json.load(handle)['keep']


G5_FEATURES = ('acne6', 'acne9', 'acne12', 'mole6', 'mole9', 'circle', 'fold')


def check_hair_edge(result, context, report):
    """G1–G8（计划 2026-11-21 §4.1）。G1–G4、G7、G8 在修复前（HEAD）必须失败；G5、G5b、G6 是代价守护。"""
    sources, extra = context['sources'], context['extra']
    frames = {}
    for key, data in result['frames'].items():
        name = key.split('|')[0]
        frames[key] = unpick(base64.b64decode(data), context['picks'][name.split('_')[0]], len(sources[name]) // 4)

    def out(name, look, weighted=False):
        return frames['%s|%s%s' % (name, look, '|w' if weighted else '')]

    def per_phase(fn, group, look, weighted=False):
        return [fn(sources['%s_p%d' % (group, p)], out('%s_p%d' % (group, p), look, weighted), 60 + p) for p in (0, 1, 2)]
    m = {}
    for look in ('so55', 'so100'):
        edges = per_phase(edge_metrics, 'halo_fh', look)
        m['G1.halo@' + look] = fmean([e[0] for e in edges])
        m['G1.w@' + look] = fmean([e[1] for e in edges])
    for look in ('def', 'full'):
        m['G2.bleedW@' + look] = fmean(per_phase(bleed_width, 'bleed', look))
    for name, _ in HARD_STRANDS:
        keeps = per_phase(strand_keep, 'hard_' + name, 'def')
        m['G3.ph_' + name] = max(keeps) - min(keeps)
    m['G4.mass'] = highpass_std(out('mass_brown', 'def'), MASS_N, 16, 80) / highpass_std(sources['mass_brown'], MASS_N, 16, 80)
    baseline = load_blemish_baseline()
    worst, measured = {'G5': (-1.0, ''), 'G5b': (-1.0, ''), 'G5w': (-1.0, ''), 'G5bw': (-1.0, '')}, {}
    weighted_diff = 0.0  # 加权路径（A 恒 255）与原路径的保留率最大差
    for look in ('def', 'mid80', 'full'):
        for name, *_ in BLEM_SCENES:
            for shape in context['shapes']:
                key = 'blem|%s|%s|%s' % (look, name, shape['name'])
                measured[key] = blemish_keep(sources[name], out(name, look), shape)
                group = 'G5' if shape['name'] in G5_FEATURES else 'G5b'
                worst[group] = max(worst[group], (measured[key] - baseline[key], key))
                keep_w = blemish_keep(sources[name], out(name, look, True), shape)
                worst[group + 'w'] = max(worst[group + 'w'], (keep_w - baseline[key], key))
                weighted_diff = max(weighted_diff, abs(keep_w - measured[key]))
    if os.environ.get('GLSL_GATE_BLEMISH_OUT'):
        with open(os.environ['GLSL_GATE_BLEMISH_OUT'], 'w', encoding='utf-8') as handle:
            json.dump({'keep': measured}, handle, indent=1)
    for name, *_ in FLICK_SCENES:
        keys = ['fl_%s_f%02d' % (name, f) for f in range(FLICK_FRAMES)]
        m['G6.' + name] = flicker_ratio([sources[k] for k in keys], [out(k, 'def') for k in keys])
        m['G6w.' + name] = flicker_ratio([sources[k] for k in keys], [out(k, 'def', True) for k in keys])
    for kind in ('stub_lo', 'stub_soft'):
        for look in ('def', 'full'):
            for weighted in (False, True):
                m['G7%s.%s@%s' % ('w' if weighted else '', kind, look)] = fmean([
                    beard_contrast(sources[name], out(name, look, weighted), extra[name])
                    for name, k, *_ in BEARD_SCENES if k == kind])
        m['G7.' + kind] = m['G7.%s@def' % kind]
    pick = bg_pixels()
    crops = {kind: {key: unpick_crop(base64.b64decode(data), pick) for key, data in value.items()}
             for kind, value in result['bg'].items()}
    flyaway = bg_flyaway_keep(crops['checker'])
    checks = (
        (m['G1.halo@so55'] <= 3.0, 'G1 棕发|额头 3 相位，只开磨皮 0.55：边外 3–20px 最大 |ΔY| %.2f（≤3.0）' % m['G1.halo@so55']),
        (m['G1.halo@so100'] <= 7.0, 'G1 同上，磨皮 1.0：最大 |ΔY| %.2f（≤7.0）' % m['G1.halo@so100']),
        (m['G1.w@so100'] <= 3.0, 'G1 磨皮 1.0 的 10–90%% 边宽 %.1f px（≤3）' % m['G1.w@so100']),
        (m['G2.bleedW@def'] <= 5.0, 'G2 金发|青墙 face=0，默认档色偏宽度 %.1f px（≤5）' % m['G2.bleedW@def']),
        (m['G2.bleedW@full'] <= 13.0, 'G2 同上，满档 %.1f px（≤13）' % m['G2.bleedW@full']),
        (m['G3.ph_lt'] <= 0.03 and m['G3.ph_dk'] <= 0.03,
         'G3 1px 硬丝 3 相位保留率极差（默认档）：浅棕 %.3f / 深棕 %.3f（≤0.03）' % (m['G3.ph_lt'], m['G3.ph_dk'])),
        (m['G4.mass'] >= 0.80, 'G4 脸外棕发团 σ15%% face=0，默认档 4px 高通保留率 %.3f（≥0.80）' % m['G4.mass']),
        (worst['G5'][0] <= 0.05, 'G5 中等瑕疵（痘印 / 痣 / 黑眼圈 / 法令纹 × Δ30/45/60 × 2 肤色 × 默认 / 0.8 / 满档）保留率 − 冻结基线，'
                                 '最差 %+.3f（≤+0.05）于 %s' % worst['G5']),
        (worst['G5b'][0] <= 0.05, 'G5b 2–3px 小黑点保留率 − 冻结基线，最差 %+.3f（≤+0.05）于 %s' % worst['G5b']),
        (m['G6.plat_light_s2'] <= 0.52 and m['G6.dark15'] <= 0.66 and m['G6.dark11'] <= 0.88,
         'G6 30 帧帧差 RMS 比：浅肤 σ2 %.3f（≤0.52）/ 暗肤 Y0.15 σ3 %.3f（≤0.66）/ Y0.11 σ3 %.3f（≤0.88）' % (
             m['G6.plat_light_s2'], m['G6.dark15'], m['G6.dark11'])),
        (m['G7.stub_lo'] >= 0.92 and m['G7.stub_soft'] >= 0.95,
         'G7 默认档胡茬对比保留：稀疏 %.3f（≥0.92）/ 软茬 %.3f（≥0.95）' % (m['G7.stub_lo'], m['G7.stub_soft'])),
        # 满档（评审 #20）：1.24.2 为 0.642 / 0.686，1.26.0+46 为 0.684 / 0.687，第一批（REC）0.770 / 0.855。下限落在 cur 与 REC
        # 之间、给 REC 留约 0.04–0.05 余量，同时高于 1.24.x，「明显高于 1.26.0+46 且不低于 1.24.x」一并守住。
        (m['G7.stub_lo@full'] >= 0.73 and m['G7.stub_soft@full'] >= 0.80,
         'G7 满档胡茬对比保留：稀疏 %.3f（≥0.73）/ 软茬 %.3f（≥0.80）；1.24.2 0.642 / 0.686，1.26.0+46 0.684 / 0.687' % (
             m['G7.stub_lo@full'], m['G7.stub_soft@full'])),
        # 头发加权路径在没有头发（A 恒 255）时的代价：瑕疵、闪烁、胡茬照原阈值再判一遍。
        (worst['G5w'][0] <= 0.05 and worst['G5bw'][0] <= 0.05 and weighted_diff <= 0.01,
         'G5w 头发加权路径（A 恒 255）：中等瑕疵 − 冻结基线最差 %+.3f、小黑点 %+.3f（≤+0.05），与原路径保留率最大差 %.4f（≤0.01）' % (
             worst['G5w'][0], worst['G5bw'][0], weighted_diff)),
        (m['G6w.plat_light_s2'] <= 0.52 and m['G6w.dark15'] <= 0.66 and m['G6w.dark11'] <= 0.88,
         'G6w 头发加权路径（A 恒 255）30 帧帧差 RMS 比：%.3f / %.3f / %.3f（≤0.52 / 0.66 / 0.88）' % (
             m['G6w.plat_light_s2'], m['G6w.dark15'], m['G6w.dark11'])),
        (m['G7w.stub_lo@def'] >= 0.92 and m['G7w.stub_soft@def'] >= 0.95 and m['G7w.stub_lo@full'] >= 0.73 and
         m['G7w.stub_soft@full'] >= 0.80,
         'G7w 头发加权路径（A 恒 255）胡茬对比保留：默认档 %.3f / %.3f（≥0.92 / 0.95），满档 %.3f / %.3f（≥0.73 / 0.80）' % (
             m['G7w.stub_lo@def'], m['G7w.stub_soft@def'], m['G7w.stub_lo@full'], m['G7w.stub_soft@full'])),
        (flyaway[1.0] >= 0.30 and flyaway[3.0] >= 0.30,
         'G8 背景虚化（模拟遮罩 PSF σ1，棋盘 0.55/0.80，强度 1）飞发保留率：1px %.3f / 2px %.3f / 3px %.3f（1px、3px ≥0.30）' % (
             flyaway[1.0], flyaway[2.0], flyaway[3.0])),
    )
    for ok, text in checks:
        report.line(ok, text)
    check_g1_matrix(sources, out, per_phase, report)
    for kind, label, limit in (('checker', '低对比棋盘', 2.0), ('hc', '高对比棋盘 0.20/0.95', 8.0)):
        halo = bg_halo50(crops[kind])
        report.line(halo <= limit, 'G8 %s 未虚化光环 halo50 %.1f px（≤%g）' % (label, halo, limit))
    # 肩部（评审 #1：细节保留曾作用于整个轮廓）：从边缘起连续虚化不到一半的长度。不用 halo50 的「最远距离」：肩线长而直，
    # 全虚化参照把衣服的暗色带进背景，低对比棋盘上 ratio 在 0.5 上下拖出一长串，halo50 恒报 7px（与细节保留无关）。
    shoulders = [(label, bg_halo_run(crops[kind], BG_SHOULDER_SIDES, 'edge'), limit)
                 for kind, label, limit in (('checker', '低对比', 2), ('hc', '高对比', 8))]
    report.line(all(h <= limit for _, h, limit in shoulders), 'G8 肩部（衣服轮廓 y=900/950/1000 两侧）从边缘起连续未虚化：%s' % (
        '，'.join('%s %dpx（≤%d）' % row for row in shoulders)))
    check_bg_merged(result, crops, context['background'], report)
    check_bg_detail_sweep(crops[BG_SWEEP_CHECKER], pick, context['background'], report)
    check_bg_temporal(result['temporal'], report)


def check_bg_merged(result, crops, background, report):
    """G8 第二批合并道变体（评审 #19）：头发遮罩 = 发团覆盖率 + 高斯 σ（BG_HAIR_SIGMAS），走生产 BG_HAIR 后接背景虚化。
    - 低对比背景：细节保留管不到，未虚化带由合并遮罩本身决定，看合并遮罩 keep = smoothstep(MASK_LOW, MASK_HIGH, R) 沿离边法向
      均值 > 0.5 的最远距离，≤2px（成功标准 4）。不用 halo50：合并后头旁 |全虚化 − 原画| 很小，比值噪声大（σ=0 时误报 5px）。
    - 高对比背景：图像侧「从边缘起连续虚化不到一半」的长度 ≤8px。
    - 飞发保留率仍 ≥0.30（1px、3px）。"""
    low, high = background['low'], background['high']
    rows, ok = [], True
    for i, sigma in enumerate(BG_HAIR_SIGMAS):
        plane = base64.b64decode(result['mergedPlanes'][i])
        reach = bg_keep_reach(plane, BG_HEAD_SIDES, low, high)
        run = bg_halo_run(crops['hc'], BG_HEAD_SIDES, 'edgeH%d' % i)
        fly = bg_flyaway_keep({'in_s': crops['checker']['in_s'], 'in_n': crops['checker']['in_n'],
                               'out': crops['checker']['outH%d' % i], 'cross': crops['checker']['crossH%d' % i]})
        ok = ok and reach <= 2 and run <= 8 and fly[1.0] >= 0.30 and fly[3.0] >= 0.30
        rows.append('σ%.2f：低对比 keep>0.5 最远 %dpx（≤2）、高对比连续未虚化 %dpx（≤8）、飞发 1px %.3f / 3px %.3f（≥0.30）' % (
            sigma, reach, run, fly[1.0], fly[3.0]))
    report.line(ok, 'G8 合并道打开（头发遮罩高斯软化，阈值 HAIR_LOW/HIGH）：%s' % '；'.join(rows))


def check_bg_detail_sweep(crops, pick, background, report):
    """G8 细节保留 β 随「人脸近期可见」系数 k 渐变（评审复核：第一批 D3 曾随人脸有无单帧硬切）。低对比棋盘、带飞发画面与遮罩，
    β = detailKeep(true, k)（BG_SWEEP），在取样点上（飞发、头部与肩部剖面）：
    - k = 1 与生产（β = DETAIL_KEEP）逐字节相同；k = 0 与没有人脸（β 0、NO_FACE 椭圆与轴向）逐字节相同；
    - 连续：相邻两档之间每个像素的变化 ≤ 255 × (β 之差) + 1（BG_MIX 对 β 的 Lipschitz 界；遮罩 keep 先取 max，变化只会更小），
      且整段确有变化（最大差 ≥ 2 × 单档上限，否则「连续」无从区分台阶）；
    - 单调：每个像素随 k 单向变化（±1 舍入）；
    - 1px / 3px 飞发保留率随 k 单调不降。"""
    betas = [detail_keep(background['detail'], k) for k in BG_SWEEP]
    keys = ['outB%d' % i for i in range(len(BG_SWEEP))]
    if any(k not in crops for k in keys + ['outNoFace']):
        report.line(False, 'G8 β 扫描：结果缺失')
        return
    rows = [[crops[k][i] for i in pick] for k in keys]
    prod = [crops['out'][i] for i in pick]
    no_face = [crops['outNoFace'][i] for i in pick]
    same_one = rows[-1] == prod and betas[-1] == background['detail']
    same_zero = rows[0] == no_face and betas[0] == 0.0
    step_ok, worst = True, []
    for a in range(len(rows) - 1):
        limit = 255.0 * (betas[a + 1] - betas[a]) + 1
        diff = max(abs(x - y) for x, y in zip(rows[a], rows[a + 1]))
        worst.append(diff)
        step_ok = step_ok and diff <= limit
    span = max(abs(x - y) for x, y in zip(rows[0], rows[-1]))
    bound = 255.0 * max(b - a for a, b in zip(betas, betas[1:])) + 1
    monotone = True
    for j in range(len(pick)):
        seq = [r[j] for r in rows]
        up = all(b >= a - 1 for a, b in zip(seq, seq[1:]))
        down = all(b <= a + 1 for a, b in zip(seq, seq[1:]))
        monotone = monotone and (up or down)
    keeps = []
    for key in keys:
        fly = bg_flyaway_keep({'in_s': crops['in_s'], 'in_n': crops['in_n'], 'out': crops[key], 'cross': crops['cross']})
        keeps.append((fly[1.0], fly[3.0]))
    keep_ok = all(b[0] >= a[0] - 0.005 and b[1] >= a[1] - 0.005 for a, b in zip(keeps, keeps[1:]))
    ok = same_one and same_zero and step_ok and span >= 2 * bound and monotone and keep_ok
    report.line(ok, 'G8 细节保留 β 随人脸近期可见系数渐变：k=1 与生产逐字节相同=%s，k=0 与无人脸逐字节相同=%s；相邻档最大差 %s'
                     '（≤255·Δβ+1=%s，整段最大差 %d ≥ %.0f），逐像素单调=%s；飞发保留率 1px / 3px 随 k：%s（单调=%s）' % (
                         same_one, same_zero, worst, step_ok, span, 2 * bound, monotone,
                         ' '.join('%.3f/%.3f' % kv for kv in keeps), keep_ok))


def check_bg_temporal(temporal, report):
    """G8 时域（评审 #1）：静态画面 + 8 帧抖动的软人像遮罩（宽 PSF + 低置信光晕 + 噪声），真背景里相邻帧 |ΔY| > 8/255 的像素数 / 帧。
    prod = 生产（有人脸、头部先验）；old = 细节保留关闭（β = 0，即 1.24.x 与无人脸）；noprior = 修复前的 D3（先验处处为 1）。
    - 头部先验区外（d ≥ 2.8 个脸半径：手臂、躯干、下半段肩膀、背景里的其他人和物）：prod 与 old 逐像素相同，闪烁像素数必须相等；
      noprior 必须多出来（修复前失败，同一次运行给出对照）。
    - 先验区内（头部与靠近脖子的上半段肩膀）细节保留照常作用，闪烁是已知代价：全画面 prod 不超过 old 的 BG_FLICKER_RATIO 倍，
      防止以后放宽细节保留时悄悄变宽。"""
    rows, ok_far, ok_all, before = [], True, True, True
    for name, label in (('hc', '高对比棋盘'), ('text', '文字')):
        row = temporal[name]
        prod, old, noprior = row['prod'], row['old'], row['noprior']
        ok_far = ok_far and prod[1] == old[1]
        ok_all = ok_all and prod[0] <= BG_FLICKER_RATIO * old[0]
        before = before and noprior[1] > old[1]
        rows.append('%s：生产 %.0f（先验区外 %.0f）/ 细节保留关闭 %.0f（%.0f）/ 无先验 %.0f（%.0f）' % (
            label, prod[0], prod[1], old[0], old[1], noprior[0], noprior[1]))
    report.line(ok_far and ok_all and before,
                'G8 时域：真背景 |ΔY|>8/255 像素数/帧 %s；先验区外与细节保留关闭相同=%s（无先验时多出=%s），全画面 ≤ %g×关闭=%s' % (
                    '；'.join(rows), ok_far, before, BG_FLICKER_RATIO, ok_all))


def check_g1_matrix(sources, out, per_phase, report):
    """G1 矩阵（评审 2026-11-21 #0）：发色 × 肤色 × def / full。
    - 两条路径、全部组合：不劣于 HEAD（1.26.0+46 冻结值 + G1_HEAD_TOL；|ΔY| ≥ G1_DY_MIN 时边宽 + G1_WIDTH_TOL）。
    - 没有头发遮罩（hx_*，第一批 / 回落路径）：只有上面的守护，中等、偏深肤色配棕发、金发的光晕仍高于 1.24.x（已知代价，计划 §9）。
    - 带头发遮罩（hm_*，皮肤遮罩 A + 导向滤波头发加权）：成功标准 1——默认档光晕 ≤3/255、满档 ≤7/255、满档边宽 ≤3px；
      旧版默认档 ≤3 的组合（G1_OLD_COMBOS）不劣于 1.24.x（+ G1_OLD_TOL）。"""
    rows, head_bad, old_bad, std_bad = [], [], [], []
    for hname, hair in G1_HAIRS:
        for sname, skin in G1_SKINS:
            combo = '%s|%s' % (hname, sname)
            frozen = G1_FROZEN[combo]
            dy = abs(luma8(hair) - luma8(skin))
            cells = []
            for kind in ('hx', 'hm'):
                for look in ('def', 'full'):
                    edges = per_phase(edge_metrics, '%s_%s_%s' % (kind, hname, sname), look, kind == 'hm')
                    halo, width = fmean([e[0] for e in edges]), fmean([e[1] for e in edges])
                    old_halo, old_w, cur_halo, cur_w = frozen[look]
                    cells.append('%s %5.2f/%4.1f' % (look, halo, width))
                    floor = G1_MASK_FLOOR if kind == 'hm' else 0.0
                    if halo > max(cur_halo + G1_HEAD_TOL, floor) or (dy >= G1_DY_MIN and width > cur_w + G1_WIDTH_TOL):
                        head_bad.append('%s %s %s %.2f/%.1f（HEAD %.2f/%.1f）' % (kind, combo, look, halo, width, cur_halo, cur_w))
                    if kind == 'hm':
                        if look == 'def' and combo in G1_OLD_COMBOS and halo > old_halo + G1_OLD_TOL:
                            old_bad.append('%s def %.2f（1.24.x %.2f）' % (combo, halo, old_halo))
                        if halo > (3.0 if look == 'def' else 7.0) or (look == 'full' and dy >= G1_DY_MIN and width > 3.0):
                            std_bad.append('%s %s %.2f/%.1f' % (combo, look, halo, width))
            alone = [fmean([e[0] for e in per_phase(edge_metrics, 'hm_%s_%s' % (hname, sname), look)]) for look in ('def', 'full')]
            rows.append('%-13s ΔY%5.1f 无遮罩 %s | 带遮罩 %s（仅 A %.2f / %.2f）| 1.24.x def %.2f full %.2f，1.26.0+46 def %.2f full %.2f' % (
                combo, dy, '  '.join(cells[:2]), '  '.join(cells[2:]), alone[0], alone[1], frozen['def'][0], frozen['full'][0],
                frozen['def'][2], frozen['full'][2]))
    print('G1 矩阵（光晕 /255 / 10–90% 边宽 px，3 相位平均）：\n  ' + '\n  '.join(rows))
    report.line(not head_bad, 'G1 矩阵 4 发色 × 3 肤色 × def/full，无遮罩与带遮罩两条路径都不劣于 1.26.0+46（光晕 +%.1f，带遮罩另有'
                              '不可见下限 %.1f；|ΔY|≥%g 时边宽 +%gpx）%s' % (G1_HEAD_TOL, G1_MASK_FLOOR, G1_DY_MIN, G1_WIDTH_TOL,
                                                                         '' if not head_bad else '：' + '；'.join(head_bad)))
    report.line(not old_bad, 'G1 矩阵带头发遮罩、默认档：旧版 ≤3 的 %s 不劣于 1.24.x（+%.1f）%s' % (
        ' / '.join(G1_OLD_COMBOS), G1_OLD_TOL, '' if not old_bad else '：' + '；'.join(old_bad)))
    report.line(not std_bad, 'G1 矩阵带头发遮罩：默认档光晕 ≤3/255、满档 ≤7/255、满档边宽 ≤3px（|ΔY|≥%g）%s' % (
        G1_DY_MIN, '' if not std_bad else '：' + '；'.join(std_bad)))


# ---- 头发遮罩探针（2026-11-21 第二批，计划 §4.3）：A1–A7 主合成读 uSkinMask.a（非头发权重 nonHair）、HA1 SemanticAlphaPass
# 的 HAIR_ALPHA_FRAGMENT、BH1 背景虚化的 BG_HAIR_FRAGMENT 合并道。A1 / A2 走调试程序（beautyColorPro 逐例，dMask.w =
# nonHair）；A3–A7 与 G1–G7 同一套整帧流程（生产导向滤波 + FRAGMENT_PRO，皮肤遮罩带 A 通道）；HA1 / BH1 跑生产程序。
# 阈值常量从 SemanticAlphaPass.kt / BackgroundBlurPass.kt 解析。场景值与阈值是初值；BH1 的低对比场景按实测定阈值。

# A1 / A2：平坦棕发，face = 1，自然档、磨皮满档；第 4 个分量即 nonHair（A = 0 / 0.5 / 1）。
HAIR_FLAT = (110, 78, 55)
HAIR_A_CASES = [pro_case('hair_flat_A%03d' % round(a * 100), HAIR_FLAT, None, None, None, (1, 0, 0.3, a),
                         [16, 1.0, 0.0, 0.5], NATURAL['details'], NATURAL['tone']) for a in (0.0, 0.5, 1.0)]
# A3–A7 的整帧场景：(名字, 源场景, 皮肤遮罩 RGBA 字节, uFrame.w)。遮罩给 8 个数时左半 / 右半各 4 个，RGB = (face, excl,
# hiPlane)。源场景复用 G 段的 mass_brown（棕发团 σ15%）与 stub_lo_brn_med（稀疏棕胡茬、中等肤色），skin_noise 是皮肤对照。
HAIR_A_SPLIT = [255, 0, 77, 0, 255, 0, 77, 255]
HAIR_A_SCENES = (
    ('hairA_half', 'mass_brown', HAIR_A_SPLIT, 5.0),          # A3：左半 A = 0、右半 A = 255，多边形遮罩路径
    ('hairA_full', 'mass_brown', [255, 0, 77, 255], 5.0),     # A3 的参照：整帧 A = 255
    ('hairA_nopoly', 'mass_brown', HAIR_A_SPLIT, 0.0),        # A4：同 A3，但没有多边形遮罩（presence = 0）
    ('hairA_skin254', 'skin_noise', [255, 0, 77, 254], 5.0),  # A5：A = 254 与 255
    ('hairA_skin255', 'skin_noise', [255, 0, 77, 255], 5.0),
    ('hairA_stub', 'stub_lo_brn_med', [255, 0, 64, 0], 5.0),  # A6：胡茬，A = 0
    ('hairA_out', 'mass_brown', [0, 0, 0, 0], 5.0),           # A7：脸外棕发团（face = 0），A = 0；G4 是它的 A = 255 对照
)
HAIR_A_LOOKS = ['def', 'full']
# A3 / A4 的左半：先裁出 [0, 48) 列再算 σ4 高通（右半的处理不会漏进来），统计内框 [8, 40) 列 × [16, 80) 行。
HAIR_A_LEFT = (0, MASS_N // 2, (8, 40, 16, 80))


def skin_noise_scene():
    """A5 的皮肤对照（pipeline 探针 skinS6）：浅肤色加 σ0.8 模糊后归一到 σ6 的噪声，96×96。"""
    t = gauss_blur(NumpyRandom(5).normals(MASS_N * MASS_N), MASS_N, MASS_N, 0.8)
    sd = fstd(t)
    return rgba_bytes([[float(SKIN_LIGHT[c]) + v / sd * 6.0 for v in t] for c in range(3)])


def hair_a_scenes(sources, dims):
    """A3–A7 的场景（追加到 INPUT.hairEdge.scenes，回读整帧），同时把源图登记进 sources。"""
    sources['skin_noise'], dims['skin_noise'] = skin_noise_scene(), (MASS_N, MASS_N)
    scenes = []
    for name, base, mask, frame_w in HAIR_A_SCENES:
        (width, height), data = dims[base], sources[base]
        scenes.append({'name': name, 'w': width, 'h': height, 'rgba': base64.b64encode(data).decode(), 'mask': mask,
                       'frameW': frame_w, 'looks': HAIR_A_LOOKS, 'pick': 'hairA'})
        sources[name] = data
    return scenes


# HA1（HAIR_ALPHA_FRAGMENT）：64×64 目标（RGB 为图样、A = 255，SkinMaskPass 输出的布局：纹理 t = 1 是画面顶部）与 64×64
# 头发遮罩（LUMINANCE，第 0 行是画面顶部）。头发图样由 gate.js 按名字生成：c128 / c230 / c255 为常数，top 上半为头发，
# left 左半为头发，blob 中心半径 0.05 的圆为头发。椭圆 = (uFaceOval, uFaceAxis)，目标是正方形，aspect = 1。
HA_N = 64
HA_IDENTITY = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
HA_FACE = ([0.5, 0.5, 0.15, 0.15], [1.0, 0.0, 1.0, 0.0])    # 脸椭圆在中心：中心在脸内，四角在脸外
HA_AWAY = ([-4.0, -4.0, 0.01, 0.01], [1.0, 0.0, 1.0, 0.0])  # 画面里没有脸：处处没有脸内偏置
HA_CASES = (  # (名字, 头发图样, 强度, 仿射两行, 椭圆, 是否再画网格)
    ('out128', 'c128', 1.0, HA_IDENTITY, HA_FACE, False),
    ('in230', 'c230', 1.0, HA_IDENTITY, HA_FACE, False),
    ('zero', 'c255', 0.0, HA_IDENTITY, HA_FACE, False),
    ('top', 'top', 1.0, HA_IDENTITY, HA_AWAY, False),
    ('shiftR', 'left', 1.0, ([1.0, 0.0, 0.25, 0.0], [0.0, 1.0, 0.0, 0.0]), HA_AWAY, False),
    ('shiftL', 'left', 1.0, ([1.0, 0.0, -0.25, 0.0], [0.0, 1.0, 0.0, 0.0]), HA_AWAY, False),
    ('blob', 'blob', 1.0, HA_IDENTITY, HA_AWAY, False),
    ('blobMesh', 'blob', 1.0, HA_IDENTITY, HA_AWAY, True),
    # 脸内偏置的位置（评审 #8）：HA_FACE 上下左右对称，偏置按 (x, 1 − y) 或按 vSrc 算都看不出来。offFace 的椭圆只罩住画面
    # 上方（纹理 t = 0.75），拦 y 翻转；meshFace 的小椭圆（r = 0.04 ≤ 0.08）放在网格中心顶点的落点 (0.58, 0.54)，按落点 vDst
    # 判脸内才有偏置，拦「改用 vSrc」（落点到源点 (0.5, 0.5) 的距离 0.089，r 取 0.15 时 vSrc 也在椭圆内，区分不出来）。
    ('offFace', 'c128', 1.0, HA_IDENTITY, ([0.5, 0.75, 0.15, 0.15], [1.0, 0.0, 1.0, 0.0]), False),
    ('meshFace', 'c128', 1.0, HA_IDENTITY, ([0.58, 0.54, 0.04, 0.04], [1.0, 0.0, 1.0, 0.0]), True),
)


def hair_constants(kotlin_dir, report):
    """从 SemanticAlphaPass.kt 读 ALPHA_LOW / ALPHA_HIGH / FACE_BIAS、从 BackgroundBlurPass.kt 读 HAIR_LOW / HAIR_HIGH；
    核对 Kotlin 按这些常量设 uHair，且 SemanticAlphaPass 只写 A、在 finally 里恢复 colorMask 与混合。解析失败返回 None。"""
    alpha = kotlin_code(os.path.join(kotlin_dir, 'SemanticAlphaPass.kt'))
    bg = kotlin_code(os.path.join(kotlin_dir, 'BackgroundBlurPass.kt'))
    found = dict(re.findall(r'const val (ALPHA_LOW|ALPHA_HIGH|FACE_BIAS) = ([0-9.]+)f', alpha))
    found.update(re.findall(r'const val (HAIR_LOW|HAIR_HIGH) = ([0-9.]+)f', bg))
    if sorted(found) != ['ALPHA_HIGH', 'ALPHA_LOW', 'FACE_BIAS', 'HAIR_HIGH', 'HAIR_LOW']:
        report.line(False, 'SemanticAlphaPass.kt 的 ALPHA_LOW / ALPHA_HIGH / FACE_BIAS 或 BackgroundBlurPass.kt 的 '
                           'HAIR_LOW / HAIR_HIGH 无法解析')
        return None
    wired = (re.search(r'glUniform4f\(\w+, [^\n]*?, ALPHA_LOW, ALPHA_HIGH, FACE_BIAS\)', alpha) is not None,
             re.search(r'glUniform4f\(mergeAt\.hair, [^\n]*?, HAIR_LOW, HAIR_HIGH, 0f\)', bg) is not None)
    report.line(all(wired), 'uHair 接线：SemanticAlphaPass.kt 传 ALPHA_LOW / ALPHA_HIGH / FACE_BIAS=%s，BackgroundBlurPass.kt '
                            '的合并道传 HAIR_LOW / HAIR_HIGH=%s' % wired)
    # 只写 A：render 先关掉 RGB 写入，finally 里恢复全开并关混合（泄漏会让后面各道的 RGB 都写不进去）。
    masked = re.search(r'glColorMask\(false, false, false, true\)', alpha) is not None
    block = re.search(r'\bfinally \{(.*?)\n        \}', alpha, re.S)
    restored = block is not None and 'glColorMask(true, true, true, true)' in block.group(1) and \
        'glDisable(GL_BLEND)' in block.group(1)
    report.line(masked and restored, 'SemanticAlphaPass.kt 只写 A（glColorMask(false, false, false, true)=%s），'
                                     'finally 里恢复 colorMask 全开并关混合=%s' % (masked, restored))
    return {k: float(v) for k, v in found.items()}


def hair_alpha_input(hair):
    return {'size': HA_N, 'uniforms': [hair['ALPHA_LOW'], hair['ALPHA_HIGH'], hair['FACE_BIAS']],
            'cases': [{'name': n, 'hair': h, 'strength': s, 'x0': a[0], 'x1': a[1], 'oval': o[0], 'axis': o[1], 'mesh': m}
                      for n, h, s, a, o, m in HA_CASES]}


# BH1（BG_HAIR_FRAGMENT 合并后接生产背景虚化）：帧 432×768（竖屏），人像 / 头发遮罩 216×384（第 0 行是画面顶部，同生产）。
# 画面：白墙 + 深色发团 + 脸 + 衣服；头顶上方一条飞发带里是 12 根 2px 竖直浅金发（与白墙的亮度差约 0.107，低于细节保留
# D3 的 0.15，D3 管不到）。遮罩：人像在头、脸、衣服内为 1，飞发带里只有 0.1（BH_PERSON_BAND）；头发在发团（不含脸）为 1，
# 飞发带里只在每根发丝处有软剖面（峰值 BH_HAIR_PEAK、横向高斯 σ BH_STRAND_SIGMA 遮罩像素，评审 #19：原先整块矩形 0.6 把发丝
# 之间约 10px 的墙也并成人像、合并后保留率 1.000 是构造出来的），左下角另有一块先验区外（约 3.1 个脸半径）的假头发。
# 画面像素坐标 y 向下。
BH_W, BH_H, BH_MW, BH_MH = 432, 768, 216, 384
BH_WALL, BH_BLONDE, BH_DARK, BH_SHIRT = (236, 233, 228), (222, 205, 170), (60, 45, 38), (70, 90, 120)
BH_HEAD, BH_FACE = (216.0, 400.0, 130.0, 170.0), (216.0, 430.0, 90.0, 120.0)
BH_BODY = (600.0, 150.0)                      # 衣服：y ≥ 600 且 |x − 216| ≤ 150
BH_BAND = (140, 292, 188, 240)                # 飞发带 [x0, x1) × [y0, y1)
BH_STRANDS = (list(range(150, 283, 12)), 2, 196, 240)  # 浅金发：左缘列、宽、[起行, 止行)
BH_CORE_ROWS = range(200, 225)                # 保留率只统计飞发带里、离发团足够远的这几行
BH_FAKE = (10, 70, 690, 750)                  # 先验区外的假头发 [x0, x1) × [y0, y1)
BH_PERSON_BAND, BH_HAIR_PEAK, BH_STRAND_SIGMA = 26, 0.6, 1.25  # 人像 0.1；发丝处头发峰值 0.6、σ 1.25 遮罩像素
# 墙面：飞发带里发丝之间（每个空隙中间 4px）的合并遮罩 keep = smoothstep(MASK_LOW, MASK_HIGH, R) 均值上限——合并道只该把发丝
# 并进人像，不能把整条带连墙一起并进去（整块矩形的旧场景是 1.000）。
BH_WALL_KEEP = 0.10
# 飞发保留率阈值（按实测定，计划给的初值是 0.50 / 0.30）：合并后（shown = 1）实测 1.000（发丝处合并到 255，keep = 1；发丝
# 之间的墙不合并，见 BH_WALL_KEEP），不合并（第一批）0.098；帧坐标或取头发时不翻转的变异把飞发带错位到下巴一带，保留率掉到
# 0.90 以下（数值见评审修复的变异记录），所以合并后的下限取 0.90。
BH_KEEP_AFTER, BH_KEEP_BEFORE = 0.90, 0.30


def bh_inside(e, x, y):
    return ((x - e[0]) / e[2]) ** 2 + ((y - e[1]) / e[3]) ** 2 <= 1.0


def bh_masks():
    """按遮罩纹素中心（换算到帧像素）生成人像与头发遮罩（第 0 行是画面顶部）。"""
    person, hair = bytearray(BH_MW * BH_MH), bytearray(BH_MW * BH_MH)
    sx, sy = BH_W / BH_MW, BH_H / BH_MH
    for my in range(BH_MH):
        for mx in range(BH_MW):
            x, y = (mx + 0.5) * sx, (my + 0.5) * sy
            head, face = bh_inside(BH_HEAD, x, y), bh_inside(BH_FACE, x, y)
            body = y >= BH_BODY[0] and abs(x - BH_HEAD[0]) <= BH_BODY[1]
            band = BH_BAND[0] <= x < BH_BAND[1] and BH_BAND[2] <= y < BH_BAND[3]
            fake = BH_FAKE[0] <= x < BH_FAKE[1] and BH_FAKE[2] <= y < BH_FAKE[3]
            k = my * BH_MW + mx
            person[k] = 255 if head or face or body else BH_PERSON_BAND if band else 0
            hair[k] = 255 if (head and not face) or fake else round(255 * bh_strand_hair(x)) if band else 0
    return bytes(person), bytes(hair)


def bh_strand_hair(x):
    """飞发带里帧像素列 x 处（遮罩纹素中心）的头发置信度：每根发丝中心一个峰值 BH_HAIR_PEAK、横向高斯 σ BH_STRAND_SIGMA 遮罩像素。"""
    columns, width, _, _ = BH_STRANDS
    scale = BH_W / BH_MW
    return min(1.0, sum(BH_HAIR_PEAK * math.exp(-0.5 * ((x - (c + width / 2)) / scale / BH_STRAND_SIGMA) ** 2) for c in columns))


def bh_wall():
    """飞发带里发丝之间的墙（每个空隙中间 4px，帧像素 (x, y)，y 向下）。"""
    columns, width, _, _ = BH_STRANDS
    return [(c + width + 3 + i, y) for c in columns[:-1] for i in range(4) for y in BH_CORE_ROWS]


def bh_core():
    """飞发芯（帧像素下标，第 0 行在画面顶部）。"""
    columns, width, _, _ = BH_STRANDS
    return [y * BH_W + x0 + i for x0 in columns for i in range(width) for y in BH_CORE_ROWS]


def bg_hair_input(background, hair):
    person, hair_mask = bh_masks()
    aspect = BH_W / BH_H
    return {'frame': [BH_W, BH_H], 'mask': [BH_MW, BH_MH], 'person': base64.b64encode(person).decode(),
            'hair': base64.b64encode(hair_mask).decode(), 'wall': BH_WALL, 'blonde': BH_BLONDE, 'dark': BH_DARK,
            'skin': SKIN_LIGHT, 'shirt': BH_SHIRT, 'head': BH_HEAD, 'face': BH_FACE, 'body': BH_BODY,
            'strands': BH_STRANDS, 'pick': bh_core(),
            # 脸椭圆（帧纹理坐标，y 向上；半径为度量单位 = 帧宽）与轴向；探针 (d) 另用一个小椭圆。
            'oval': [BH_FACE[0] / BH_W, 1.0 - BH_FACE[1] / BH_H, BH_FACE[2] / BH_W, BH_FACE[3] / BH_W],
            'axis': [1.0, 0.0, aspect, 0.0], 'priorOval': [0.5, 0.75, 0.1, 0.1],
            'uniforms': [1.0, background['low'], background['high'], background['detail']],
            'hairUniforms': [hair['HAIR_LOW'], hair['HAIR_HIGH']]}


def smooth01(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def crop_highpass(data, width, height, x0, x1, inner):
    """先裁出 [x0, x1) 列再算 lib.hp_std（σ4 高通，边缘复制），inner = (列起, 列止, 行起, 行止)，相对裁切区。"""
    cw, plane = x1 - x0, luma_plane(data)
    crop = [plane[y * width + x] for y in range(height) for x in range(x0, x1)]
    base = gauss_blur(crop, cw, height, 4.0)
    ix0, ix1, iy0, iy1 = inner
    return fstd([crop[y * cw + x] - base[y * cw + x] for y in range(iy0, iy1) for x in range(ix0, ix1)])


def mean_dy(src, out, width, xs, ys):
    return fmean([luma_at(out, y * width + x) - luma_at(src, y * width + x) for y in ys for x in xs])


def rgb_luma(data, count):
    return [0.2126 * data[3 * k] + 0.7152 * data[3 * k + 1] + 0.0722 * data[3 * k + 2] for k in range(count)]


def check_hair_mask(result, context, report):
    """A1–A7、HA1、BH1（计划 2026-11-21 §4.3）。A1、A3、A4、A7 与 BH1 的「合并后保留率」在第一批（S）的着色器上必须失败。"""
    sources = context['sources']
    # A1 / A2：调试程序逐例。
    got = dict(zip((c['name'] for c in HAIR_A_CASES), cells(result['proHairA'], len(HAIR_A_CASES))))
    dys = [luma8(got['hair_flat_A%03d' % a]) - luma8(HAIR_FLAT) for a in (0, 50, 100)]
    report.line(got['hair_flat_A000'][:3] == HAIR_FLAT, 'A1 平坦棕发 %s，face=1，自然档磨皮满档，A=0：输出 %s 与输入逐字节相同' % (
        HAIR_FLAT, got['hair_flat_A000'][:3]))
    report.line(dys[0] < dys[1] < dys[2], 'A2 同上 A=0 / 0.5 / 1 的 ΔY %+.2f / %+.2f / %+.2f：A=0.5 严格介于两者之间' % tuple(dys))
    # A3–A7：整帧。
    edge = result['hairEdge']

    def frame(name, look):
        return base64.b64decode(edge['frames']['%s|%s' % (name, look)])

    def unpicked(name, look):
        return unpick(frame(name, look), None, len(sources[name]) // 4)
    x0, x1, inner = HAIR_A_LEFT
    mass = sources['mass_brown']
    left_src = crop_highpass(mass, MASS_N, MASS_N, x0, x1, inner)
    a3, a4, a5, a6, a7 = [], [], [], [], []
    for look in HAIR_A_LOOKS:
        half, full = unpicked('hairA_half', look), unpicked('hairA_full', look)
        keep = crop_highpass(half, MASS_N, MASS_N, x0, x1, inner) / left_src
        dy = mean_dy(mass, half, MASS_N, range(inner[0], inner[1]), range(inner[2], inner[3]))
        right = all(half[4 * (y * MASS_N + x):4 * (y * MASS_N + x) + 3] == full[4 * (y * MASS_N + x):4 * (y * MASS_N + x) + 3]
                    for y in range(MASS_N) for x in range(MASS_N // 2, MASS_N))
        a3.append((look, keep, dy, right))
        a4.append((look, crop_highpass(unpicked('hairA_nopoly', look), MASS_N, MASS_N, x0, x1, inner) / left_src))
        a5.append((look, frame('hairA_skin254', look) == frame('hairA_skin255', look)))
        stub, stub_out = sources['hairA_stub'], unpicked('hairA_stub', look)
        a6.append((look, fmean([luma_at(stub_out, k) - luma_at(stub, k) for k in BEARD_REGION])))
        a7.append((look, highpass_std(unpicked('hairA_out', look), MASS_N, 16, 80) / highpass_std(mass, MASS_N, 16, 80)))
    report.line(all(k >= 0.98 and abs(d) <= 1.0 and r for _, k, d, r in a3),
                'A3 发团整帧，皮肤遮罩左半 A=0、右半 A=255：%s（左半 4px 高通保留率 ≥0.98、|平均 ΔY| ≤1，右半与 A=255 整帧逐字节相同）' % (
                    '；'.join('%s 保留率 %.3f ΔY %+.2f 右半相同=%s' % row for row in a3)))
    report.line(all(k >= 0.98 for _, k in a4), 'A4 同 A3、无多边形遮罩（presence=0）：左半保留率 %s（≥0.98）' % (
        ' / '.join('%s %.3f' % row for row in a4)))
    report.line(all(same for _, same in a5), 'A5 皮肤遮罩 A=254 与 A=255 整帧逐字节相同：%s' % (
        ' / '.join('%s %s' % row for row in a5)))
    report.line(all(abs(d) <= 1.0 for _, d in a6), 'A6 胡茬 A=0：胡茬区平均 ΔY %s（|ΔY| ≤1）' % (
        ' / '.join('%s %+.2f' % row for row in a6)))
    report.line(all(k >= 0.98 for _, k in a7), 'A7 脸外棕发团 face=0、A=0：4px 高通保留率 %s（≥0.98；A=255 见 G4）' % (
        ' / '.join('%s %.3f' % row for row in a7)))
    check_hair_alpha(result['hairAlpha'], context['hairAlpha'], report)
    check_bg_hair(result['bgHair'], report, context['background'])


def check_hair_alpha(result, spec, report):
    """HA1：HAIR_ALPHA_FRAGMENT 在 colorMask(F, F, F, T) 下只写 A、阈值与脸内偏置、方向、仿射、网格、强度 0、之后 RGB 能写入。"""
    n = spec['size']
    lo, hi, bias = spec['uniforms']
    images = {name: pixels(hex_text, n, n)[::-1] for name, hex_text in result['cases'].items()}  # 第 0 行 = 纹理 t = 0
    pattern = [[(x * 4 & 255, t * 4 & 255, 77) for x in range(n)] for t in range(n)]
    rgb_kept = all(image[t][x][:3] == pattern[t][x] for image in images.values() for t in range(n) for x in range(n))
    report.line(rgb_kept, 'HA1 HAIR_ALPHA 在 colorMask(F,F,F,T) 下 %d 例目标 RGB 字节全部不变' % len(images))

    def alpha(name, x, t):
        return images[name][t][x][3]

    def expect(value, strength, b):
        return round(255 * (1 - strength * smooth01(lo + b, hi + b, value / 255)))
    corner, centre = (0, 0), (n // 2, n // 2)
    rows = (
        ('脸外 hair=128/255', alpha('out128', *corner), expect(128, 1, 0)),
        ('脸内 hair=128/255', alpha('out128', *centre), expect(128, 1, bias)),
        ('脸内 hair=230/255', alpha('in230', *centre), expect(230, 1, bias)),
        ('脸外 hair=230/255', alpha('in230', *corner), expect(230, 1, 0)),
    )
    report.line(all(abs(a - e) <= 1 for _, a, e in rows) and rows[1][2] == 255 and rows[2][2] == 0,
                'HA1 阈值 %.2f/%.2f、脸内 +%.2f：%s（±1）' % (lo, hi, bias, '，'.join('%s A=%d（应为 %d）' % row for row in rows)))
    zero = all(alpha('zero', x, t) == 255 for t in range(n) for x in range(n))
    report.line(zero, 'HA1 强度 0 时 A 处处为 255（hair=1）')
    half = n // 2
    top = all(alpha('top', x, t) == 0 for t in range(half + 2, n) for x in range(n)) and \
        all(alpha('top', x, t) == 255 for t in range(half - 2) for x in range(n))
    report.line(top, 'HA1 方向：头发遮罩上半（第 0 行是画面顶部）为头发 → 输出画面上半（纹理 t > 0.5）A=0、下半 A=255')
    q = n // 4
    shift_r = all(alpha('shiftR', x, t) == 0 for t in range(n) for x in range(q - 2)) and \
        all(alpha('shiftR', x, t) == 255 for t in range(n) for x in range(q + 2, n))
    shift_l = all(alpha('shiftL', x, t) == 255 for t in range(n) for x in range(q - 2)) and \
        all(alpha('shiftL', x, t) == 0 for t in range(n) for x in range(q + 2, 3 * q - 2)) and \
        all(alpha('shiftL', x, t) == 255 for t in range(n) for x in range(3 * q + 2, n))
    report.line(shift_r and shift_l, 'HA1 仿射平移：遮罩左半为头发，x 平移 +0.25 → 只剩 u<0.25 为头发=%s；−0.25 → u∈[0.25,0.75) '
                                     '为头发、移出遮罩的 u<0.25 按无头发=%s' % (shift_r, shift_l))
    # 网格：中心顶点的落点移到 (0.58, 0.54)，头发圆斑（源中心、半径 0.05）跟着走；中间 4 格之外逐字节同不画网格。
    moved = (round(0.58 * n - 0.5), round(0.54 * n - 0.5))
    outside = all(alpha('blobMesh', x, t) == alpha('blob', x, t) for t in range(n) for x in range(n)
                  if not (0.24 * n <= x + 0.5 <= 0.76 * n and 0.24 * n <= t + 0.5 <= 0.76 * n))
    follow = alpha('blob', *moved) == 255 and alpha('blobMesh', *moved) <= 8 and alpha('blob', *centre) == 0
    report.line(follow and outside, 'HA1 网格：中心落点 (0.58, 0.54) 处 A 不画网格 %d → 画网格 %d（跟着变形走），中间 4 格外逐字节不变=%s' % (
        alpha('blob', *moved), alpha('blobMesh', *moved), outside))
    # 脸内偏置的位置：在目标坐标 vDst 上按椭圆判（与 SkinMaskPass / FRAGMENT_PRO 的 coordinate 同向），不翻转、不用 vSrc。
    up, down = alpha('offFace', n // 2, round(0.75 * n - 0.5)), alpha('offFace', n // 2, round(0.25 * n - 0.5))
    mesh_face = alpha('meshFace', *moved)
    report.line(abs(up - 255) <= 1 and abs(down - expect(128, 1, 0)) <= 1 and mesh_face >= 254,
                'HA1 脸内偏置的位置：偏心椭圆（只罩画面上方）t=0.75 A=%d（应 255）、t=0.25 A=%d（应 %d）；网格落点处的小椭圆 '
                'A=%d（应 255，按 vDst 判）' % (up, down, expect(128, 1, 0), mesh_face))
    after = result['after']
    report.line(after == [90, 160, 210], 'HA1 之后再画普通程序（ORIGINAL），RGB 能写进去：%s（应为 [90, 160, 210]）' % after)


def check_bg_hair(result, report, background):
    """BH1：BG_HAIR 直通、方向、仿射、头部先验，以及合并后接生产背景虚化的低对比金发飞发保留率。"""
    planes = {name: base64.b64decode(data) for name, data in result['planes'].items()}  # R，第 0 行是画面顶部

    def at(name, x, row):
        return planes[name][row * BH_MW + x]
    person = bytes((x + 3 * r) % 256 for r in range(BH_MH) for x in range(BH_MW))
    report.line(planes['pass'] == person, 'BH1 shown=0：合并输出与人像遮罩逐字节相同（头发与先验都满）')
    top = all(at('top', x, r) == 255 for r in range(BH_MH // 2 - 6) for x in range(BH_MW)) and \
        all(at('top', x, r) == 0 for r in range(BH_MH // 2 + 6, BH_MH) for x in range(BH_MW))
    shift = all(at('shift', x, r) == 255 for r in range(BH_MH // 4 - 6) for x in range(BH_MW)) and \
        all(at('shift', x, r) == 0 for r in range(BH_MH // 4 + 6, BH_MH) for x in range(BH_MW))
    report.line(top and shift, 'BH1 方向：头发遮罩上半为头发 → 合并输出上半（第 0 行在顶部）被提升=%s；仿射 y 平移 −0.25 → 只剩'
                               '画面上 1/4=%s' % (top, shift))
    # 先验：小椭圆在 (0.5, 0.75)（画面上 1/4 处），处处是头发、人像为 0；椭圆中心被提升，下方与左侧（d > 2.8）不提升。
    centre, below, side = at('prior', 108, 96), at('prior', 108, 330), at('prior', 5, 96)
    report.line(centre == 255 and below == 0 and side == 0, 'BH1 头部先验：椭圆中心 %d（应 255），先验区外（d>2.8）的假头发：'
                                                             '下方 %d、侧面 %d（应 0）' % (centre, below, side))
    scene = planes['scene']
    columns = BH_STRANDS[0]
    band = scene[(214 // 2) * BH_MW + (columns[len(columns) // 2] + 1) // 2]  # 带中间那根发丝所在的纹素
    fake = scene[(720 // 2) * BH_MW + 40 // 2]
    count = len(bh_core())
    lumas = {key: rgb_luma(base64.b64decode(data), count) for key, data in result['picks'].items()}
    keeps = {}
    for shown in ('before', 'after'):
        vals = []
        for k in range(count):
            din = abs(lumas['inS'][k] - lumas['inN'][k])
            if din >= 10:
                vals.append(abs(lumas[shown + 'S'][k] - lumas[shown + 'N'][k]) / din)
        keeps[shown] = fmean(vals) if vals else 0.0
    report.line(band == 255 and fake == 0, 'BH1 场景合并遮罩：飞发带里的发丝 %d（人像 0.1、头发 0.6，应提升到 255），先验区外假头发 %d'
                                           '（应 0）' % (band, fake))
    # 发丝之间的墙：按帧像素中心双线性取合并遮罩（同 GL_LINEAR），keep 均值应接近 0（墙照常虚化）。
    def sample(fx, fy):
        u, v = min(max(fx * BH_MW / BH_W - 0.5, 0.0), BH_MW - 1.0), min(max(fy * BH_MH / BH_H - 0.5, 0.0), BH_MH - 1.0)
        x0, y0 = int(u), int(v)
        x1, y1, a, b = min(x0 + 1, BH_MW - 1), min(y0 + 1, BH_MH - 1), u - x0, v - y0
        return ((scene[y0 * BH_MW + x0] * (1 - a) + scene[y0 * BH_MW + x1] * a) * (1 - b) +
                (scene[y1 * BH_MW + x0] * (1 - a) + scene[y1 * BH_MW + x1] * a) * b) / 255.0
    wall = fmean([smooth01(background['low'], background['high'], sample(x + 0.5, y + 0.5)) for x, y in bh_wall()])
    report.line(wall <= BH_WALL_KEEP, 'BH1 飞发带里发丝之间的墙：合并遮罩 keep 均值 %.3f（≤%.2f，合并道不把墙并进人像）' % (wall, BH_WALL_KEEP))
    report.line(keeps['after'] >= BH_KEEP_AFTER and keeps['before'] < BH_KEEP_BEFORE,
                'BH1 白墙前低对比浅金发飞发（人像 0.1、头发 0.6）背景虚化保留率：合并后 %.3f（≥%.2f），不合并（shown=0，'
                '即第一批）%.3f（<%.2f）' % (keeps['after'], BH_KEEP_AFTER, keeps['before'], BH_KEEP_BEFORE))


def check_beard_border(kotlin_dir, result, report):
    """核对常规瘦脸不误唤醒 CPU 人像分割，以及胡须边界防粘皮 + 室内暗影/冷光肤色平滑连续不斑驳。"""
    vef = kotlin_code(os.path.join(kotlin_dir, 'VideoEffectsFilter.kt'))
    seg_only_blur = (re.search(r'private fun wantsBackground\(value: VideoEffectsConfig\) = segHealthy && healthy && value\.enabled &&\s*value\.backgroundBlur > 0f', vef) is not None and
                     'SilhouetteShieldPass' not in vef)
    report.line(seg_only_blur, 'VideoEffectsFilter.kt：人像分割仅在 backgroundBlur > 0f 时开启（常规瘦脸不触发低分分割抠图破洞与重影）=%s' % seg_only_blur)

    b = result
    border_ok = b['blackBorderLift'] <= 2.5 and b['brownBorderLift'] <= 3.0
    smooth_ok = b['shadowSmoothRatio'] <= 0.45 and b['coolLedSmoothRatio'] <= 0.65
    report.line(border_ok and smooth_ok,
                'B1 胡须边界防粘皮与室内肤色防斑驳：炭黑/暖棕胡须边带 ΔY %+.2f/%+.2f（≤2.5/3.0），'
                '暗影噪点肤平滑比 %.2f（≤0.45）、手机冷白光肤平滑比 %.2f（≤0.65）' % (
                    b['blackBorderLift'], b['brownBorderLift'], b['shadowSmoothRatio'], b['coolLedSmoothRatio']))


def check_v11_upgrade(kotlin_dir, result, report):
    """核对 v1.1 双频奶油肌匀肤（G9）与 468 点解剖子网格精修（R1 留卧蚕去黑眼圈、R2 弧形法令纹、R3 零染唇白牙、R4 眼裂亮眼）。"""
    prepass = kotlin_code(os.path.join(kotlin_dir, 'GuidedFilterPrePass.kt'))
    even_wired = re.search(r'BeautyQualityTuning\.macroEvenness\(', prepass) is not None
    report.line(even_wired, 'GuidedFilterPrePass.kt 的 uEvenness 取自 BeautyQualityTuning.macroEvenness=%s' % even_wired)

    v = result
    # G9：uEvenness=0 与单频逐字节一致；uEvenness=0.42 下 14/255 泪沟色块拉近 >= 30%，而 48/255 骨相大光比偏移 <= 2.5/255。
    g9_ok = v['evenZeroDiff'] == 0 and v['blotchCloseRatio'] >= 0.30 and v['boneShiftAbs'] <= 2.5
    report.line(g9_ok, 'G9 双频奶油肌匀肤：evenness=0 逐字节恒等（diff=%d），14/255 色块匀化率 %.1f%%（≥30%%），'
                       '48/255 立体骨相保护偏移 %.2f/255（≤2.5）' % (
                           v['evenZeroDiff'], v['blotchCloseRatio'] * 100.0, v['boneShiftAbs']))

    # R1：去黑眼圈留卧蚕（卧蚕区 Δ=0，泪沟暗带提亮 >= 10/255）。
    r1_ok = v['aegyoMaxDiff'] == 0 and v['tearLift'] >= 10.0
    report.line(r1_ok, 'R1 468 点去黑眼圈留卧蚕：卧蚕保护带最大像素差 %d/255（=0），泪沟凹陷带提亮 %+.1f/255（≥+10）' % (
        v['aegyoMaxDiff'], v['tearLift']))

    # R2：弧形鼻唇沟淡法令纹（暗沟提亮 >= 16/255，亮肤色不变 <= 1/255）。
    r2_ok = v['foldLift'] >= 16.0 and v['foldBrightMaxDiff'] <= 1
    report.line(r2_ok, 'R2 468 点弧形法令纹：弯曲暗沟提亮 %+.1f/255（≥+16），高亮皮肤最大差 %d/255（≤1）' % (
        v['foldLift'], v['foldBrightMaxDiff']))

    # R3：内唇孔洞子网格白牙（浅粉水光唇釉像素差 == 0，口腔内黄牙提亮且去黄）。
    r3_ok = v['lipGlossMaxDiff'] == 0 and v['teethLumaLift'] >= 8.0 and v['teethYellowDrop'] >= 12.0
    report.line(r3_ok, 'R3 468 点内唇白牙防染唇：水光浅粉唇釉最大差 %d/255（=0），黄牙提亮 %+.1f/255（≥+8）、去黄 -%.1f/255（≥12）' % (
        v['lipGlossMaxDiff'], v['teethLumaLift'], v['teethYellowDrop']))

    # R4：眼裂子网格亮眼（眼外皮肤像素差 == 0，巩膜提亮 >= 4/255）。
    r4_ok = v['eyeOutsideMaxDiff'] == 0 and v['scleraLift'] >= 4.0
    report.line(r4_ok, 'R4 468 点眼裂亮眼：眼睑外皮肤最大差 %d/255（=0），巩膜提亮 %+.1f/255（≥+4）' % (
        v['eyeOutsideMaxDiff'], v['scleraLift']))


def main(argv):
    if len(argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 64
    started = time.time()
    kotlin, vectors_path, work = argv[1:4]
    kotlin_dir = argv[4] if len(argv) == 5 else os.path.dirname(os.path.abspath(kotlin))
    chrome = os.environ.get('CHROME_BIN') or CHROME
    if not os.access(chrome, os.X_OK):
        print('FAIL  未找到 Chrome：%s。本门禁需要 Google Chrome（无头 WebGL1），不会跳过；'
              '请安装到 /Applications 或用 CHROME_BIN 指定。' % chrome)
        return 1
    report = Report()
    try:
        shaders = extract(kotlin)
    except ShaderSourceError as error:
        report.line(False, '提取 %s：%s' % (kotlin, error))
        return 1
    print('提取 %d 个常量：%s' % (len(shaders), ', '.join(shaders)))
    if not static_checks(shaders, report):
        return 1
    with open(vectors_path, encoding='utf-8') as handle:
        vectors = json.load(handle)
    check_kernel(shaders, vectors, report)
    missing = [name for name in WIRING_FILES + ('FaceGeometrySmoother.kt', 'BeautyQualityTuning.kt')
               if not os.path.isfile(os.path.join(kotlin_dir, name))]
    if missing:
        report.line(False, 'Kotlin 源码目录 %s 缺少 %s' % (kotlin_dir, ', '.join(missing)))
        return 1
    check_program_pairs(kotlin_dir, report)
    geometry_uniforms = check_geometry_packing(kotlin_dir, vectors, report)
    if geometry_uniforms is None:
        return 1
    denoise = denoise_constants(kotlin_dir)
    if denoise is None:
        report.line(False, 'TemporalDenoisePass.kt 的 MAX_WEIGHT / DIFF_LOW / DIFF_HIGH 无法解析')
        return 1
    guided = guided_constants(kotlin_dir, shaders, report)
    if guided is None:
        return 1
    check_grade_wiring(kotlin_dir, report)
    check_hair_wiring(kotlin_dir, vectors, report)
    background = background_constants(kotlin_dir, report)
    if background is None:
        return 1
    hair = hair_constants(kotlin_dir, report)
    if hair is None:
        return 1
    cases = end_to_end_cases(vectors)
    hair_edge, hair_context = hair_edge_input(background, hair)
    hair_mask = {'alpha': hair_alpha_input(hair), 'bg': bg_hair_input(background, hair)}
    hair_context['hairAlpha'] = hair_mask['alpha']
    page = os.path.join(work, 'gate.html')
    build_page(shaders, vectors, cases, geometry_uniforms, page, denoise, guided, hair_edge, hair_mask)
    result, error = run_chrome(chrome, page, work)
    if error:
        report.line(False, error)
        return 1
    env = result.get('env') or {}
    print('渲染器 %s；%s；%s' % (env.get('renderer'), env.get('version'), env.get('glsl')))
    report.line('SwiftShader' in (env.get('renderer') or '') and 'GLSL ES 1.0' in (env.get('glsl') or ''),
                '运行在 ANGLE/SwiftShader 的 WebGL1（GLSL ES 1.00）上')
    for message in result['errors']:
        report.line(False, message)
    check_programs(result, report)
    if 'endToEnd' in result.get('results', {}):
        check_skin_gate(vectors, result['results'], report)
        check_composite(vectors, result['results'], report)
        check_masks(vectors, result['results'], report)
        check_prepass(vectors, result['results'], report)
        check_end_to_end(vectors, cases, result['results'], report)
        check_denoise(vectors, denoise, result['results'], report)
        check_lut(vectors, result['results'], report)
        check_body(vectors, result['results'], report)
        check_background(vectors, result['results'], report)
        check_guided_and_skin_mask(vectors, result['results'], report)
        check_face_shape(vectors, result['results'], report)
        check_composite_pro(vectors, result['results'], report)
        check_fallback_prepass(vectors, result['results'], report)
        check_mesh_warp(vectors, result['results'], report)
        check_makeup(vectors, result['results'], report)
        check_hair_edge(result['results']['hairEdge'], hair_context, report)
        check_hair_mask(result['results'], hair_context, report)
        check_beard_border(kotlin_dir, result['results']['beardBorder'], report)
        check_v11_upgrade(kotlin_dir, result['results']['v11Upgrade'], report)
    elif not result['errors']:
        report.line(False, '没有像素结果')
    elapsed = time.time() - started
    if report.failures:
        print('Android GLSL 门禁失败：%d 项（%.1f s）' % (len(report.failures), elapsed))
        return 1
    print('Android GLSL 门禁通过（%.1f s）' % elapsed)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
