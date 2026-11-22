# 美颜参考模型与共享向量

本目录是直播采集美颜（规格 v1：`docs/plans/2026-11-11-live-beauty-capture-optimization-plan.md` 第 2–4 节）的**唯一数学真值**。Android GLSL、iOS Core Image 与两端的几何、平滑器都对照这里的向量做一致性测试；规格与本目录冲突时以规格为准，并按下文「修改常量的流程」同步。

## 来源

从 2026-11-11 数值评审的原型迁入（评审记录见 `docs/reviews/2026-11-11-live-beauty-capture-optimization-review.md`）。只保留 FINAL 常量和生成向量所需的函数，并落实了规格 v1 相对评审原型的差异：

- 第 2 节：侧脸门限放宽到 `r ≤ 0.45`，新增 `warpWeight = 1 − smoothstep(0.24, 0.30, r)`，状态向量 41 维；补上几何退化保护（`|p454'−p234'| < 0.05`、oval `rx/ry < 0.03`、mouth `rx < 0.005` → null）。
- 第 4.2 节：锐化之后、美白之前 `clamp(color, 0, 1)`；滤镜按 Android 现有着色器的 style 编号与公式。
- 第 4.1 节：1/4 尺寸取 `ceil(w/4) × ceil(h/4)`；Android RGBA8 管线按 5 通道逐通道量化（P3 用未量化的 mean 计算 v0）。
- 第 3 节：新样本以采样时间戳**变化**识别。

几何用例的关键点由 MediaPipe 标准人脸 `canonical_face_model.obj`（Apache-2.0，许可见 `../licenses/MediaPipe-LICENSE.txt`）中用到的 17 个顶点透视投影得到，顶点坐标内嵌在 `gen_vectors.py`。

## 文件

| 文件 | 用途 |
| --- | --- |
| `beauty_ref.py` | 参考模型：常量、肤色门限、`full_geometry()`、遮罩、预处理（浮点 / Android RGBA8）、合成 `composite()`、整帧 `process_frame()`、平滑器 `FinalSmoother` |
| `gen_vectors.py` | 由参考模型生成共享向量，并自检（中性直通、肤色门限四条过渡带均有用例、拒绝原因、warpWeight 落点、RGBA8 与浮点管线差、iOS 自检值、「锐化后 clamp」区分度） |
| `beauty_vectors.json` | 共享向量（约 550 KB，UTF-8） |

## 运行与开发依赖

只需系统自带的 `/usr/bin/python3` 与 numpy（**仅开发时**，运行时与 CI 的平台测试只读 JSON）。设置了环境变量 `BEAUTY_PYDEPS` 时，脚本开头把它指向的目录加入 `sys.path`（目录不存在直接报错退出）；未设置时只使用当前解释器能正常 import 的 numpy，找不到就提示安装后退出。

```bash
# 一次性安装 numpy 到任意目录（不污染系统 Python）
/usr/bin/python3 -m pip install --target ~/.cache/beauty-pydeps numpy==2.0.2

# 重新生成（在本目录执行）
cd reference
BEAUTY_PYDEPS=~/.cache/beauty-pydeps /usr/bin/python3 -I gen_vectors.py
git diff --stat beauty_vectors.json   # 未改常量时应无差异
```

- 必须用 `-I`（隔离模式）运行；脚本自己把本目录加入 `sys.path`。
- 输出确定：随机量只用 Python `random.Random` 固定种子，同一 numpy 版本下重复运行字节一致（当前文件用 numpy 2.0.2 生成，版本记在 `meta.numpy`）。换 numpy 版本可能只在第 6 位小数上有差异，远小于容差。
- `/usr/bin/python3 -I beauty_ref.py` 打印 9 抽头核的线性采样表示与有效 sigma。

## 坐标约定

- 颜色：伽马编码 0..1 浮点，全部计算在伽马编码值上进行；`rgb8 = round(x × 255)`。
- MediaPipe 关键点：归一化图像坐标，**y 向下**；进入几何前转纹理坐标 `(x, 1 − y)`。
- 纹理坐标：归一化，**y 向上**；像素 `(x, row)` 的中心 `uv = ((x + 0.5)/W, 1 − (row + 0.5)/H)`，row 0 是画面顶部。
- 度量空间：`p' = (x, y / aspect)`，`aspect = W / H`。
- 几何 uniform：中心（cheeks、eyes 的 xy，oval/mouth/eyeMasks 的 cx, cy）是纹理坐标；半径（rx, ry、eyes.z）是度量值（x 单位）；`axis` 是度量空间单位向量 u，`v = (−u.y, u.x)` 指向额头。
- 数组：`prepass`、`end_to_end` 的图像与探针 row 都自上而下。Core Image 第 0 行是底部，iOS 测试交换原始缓冲时要翻转行。

### Uniform 换算

- **Android**（纹理坐标直接上传）：`uFrame = (1/w, 1/h, aspect, presence)`；`uParams = (processing, smoothing, brightness, style)`；`uDetails = (skinOnly, sharpness, saturation, filterIntensity)`；`uTone = (whiten, rosy, faceSlim·0.12·presence·warpWeight, eyeEnlarge·0.25·presence·warpWeight)`；`uEyes[i] = (eyes[3i..3i+2], axis 分量)`（`uEyes[0].w = axis.x`，`uEyes[1].w = axis.y`）；`uOval`、`uMouth`、`uEyeMask[2]` 原样。presence 与 warpWeight 取平滑器输出。
- **iOS**（像素空间 = 度量空间 × w，各向同性）：中心 × (w, h)，半径与眼宽 × w，axis 不变；`destCoord()` 原点在左下、y 向上，与纹理坐标一致。`p0 = (smoothing, skinOnly, sharpness, saturation)`，`p1 = (whiten, rosy, brightness, filterIntensity)`，`warp = (slim·0.12·presence·warpWeight, eye·0.25·presence·warpWeight)`。
- style 编号两端相同：none 0、warm 1、cool 2、monochrome 3、sepia 4、rose 5。

## 各段含义

| 段 | 内容 | 容差 | 主要消费者 |
| --- | --- | --- | --- |
| `meta` | 约定、uniform 换算、容差、生成命令、用例计数 | — | 全部 |
| `constants` | 规格 v1 的全部常量（几何、遮罩、合成、预处理、平滑器）的快照 | — | 无测试直接读取，仅供查阅与 diff 审查；各端常量由下面的计算向量间接约束 |
| `kernel` | 9 抽头 sigma 2 核、线性采样对、Android 着色器常量 `shaderPairs`、iOS `CIConvolution9` 权重 | 精确 | GLSL 门禁静态比对 `shaderPairs` 与 `BLUR_VERTEX` / `BLUR` 原文的偏移和权重；iOS 像素测试读 `iosConvolution9` |
| `skin_gate` | rgb8 → 肤色门限；含 cb 下沿、cb 上沿、cr 下沿、cr 上沿四条过渡带中段各 3 例 | 0.004 | GLSL 门禁（`COMMON.skinWeight` 逐例）。iOS 目前不单独读取，只经 `composite` / `end_to_end` 间接覆盖 |
| `geometry` | 17 个关键点（y 向下）→ 全部 uniform（含 `eyeMasks`、`warpWeight`）或 null；含正脸、15°、17°（warpWeight 介于 0..1）、25°（有效但 warpWeight = 0）、35°/40° 无效、镜像、横屏、各类越界/非有限/退化 → null | 2e-5 | Android `FaceGeometryTest`、iOS `FLTFaceGeometry` |
| `masks` | 几何 uniform + uv（纹理坐标）+ presence → oval、eye0、eye1、mouth、face、excl | 0.01 | GLSL 门禁、iOS |
| `composite` | 逐像素合成：I/mean/var/avg4/face/excl + `paramSets` 中的参数 → 浮点 RGB 与 rgb8；含中性精确直通、单参数、组合、预设、极值、锐化越界后 clamp；`clamp_after_sharpen/*` 让越界通道经单色、饱和度、sepia、美白 + 饱和度的跨通道混合流进其它通道，去掉「锐化后 clamp」时输出至少变 4/255（生成器断言） | 1.5/255 | GLSL 门禁、iOS 像素测试 |
| `prepass` | 64×96 源图 → 1/4 分辨率 T0/mean/v0/var（浮点与 Android RGBA8 两套）及全分辨率探针 | 浮点 1/255，RGBA8 2/255 | GLSL 门禁（RGBA8）、iOS（浮点） |
| `smoother` | 6 个回放序列：静止抖动、快速移动后停止、100/300/700 ms 掉帧 + dt==0 + dt>100、reset、转头时 warpWeight 变化、只有 warpWeight 变化 | 1e-4（float32） | Android `FaceGeometrySmootherTest`、iOS `FLTFaceGeometrySmoother` |
| `end_to_end` | 用 `prepass` 源图、无变形，6 组参数（含中性直通）与一组几何 uniform（`geometry` 各字段，另附同一几何的 41 维平滑器状态 `state`），24 个探针的期望 rgb8 | 3/255（中性直通为 0） | GLSL 门禁、iOS 像素测试 |
| `ios_selftest` | 8×8 纯色 (170,115,85)，smoothing = sharpness = brightness = 1 → 中心 (191,141,109) | ±1，alpha 255 | iOS 首次使用自检 |

补充说明：

- `composite` 假定 processing 为真；中性配置的严格直通（iOS 返回原 sample、Android 输出原像素）由 `end_to_end.neutral_passthrough` 与各端自己的用例覆盖。
- 容差对应 highp/fp32 路径。真 mediump（无 `GL_FRAGMENT_PRECISION_HIGH`）是降级路径，不在共享向量覆盖范围内。
- 向量只用 4 的倍数尺寸；非 4 倍数尺寸两端的映射不同（Android 归一化坐标缩放 w/sw，iOS 原点对齐的精确 4 倍），不做跨端比较。
- 塑形（cheek/eye 逆向变形）沿用既有公式，不在本向量覆盖范围内；`end_to_end` 一律无变形。
- GLSL 门禁不手抄 Kotlin 接线：程序配对从 `VideoEffectsFilter.kt`、`BeautyPrePass.kt` 的 `createProgram` / `Program(...)` 调用解析，并须与门禁的程序表一致；几何 uniform 按 `VideoEffectsFilter.kt` 的打包代码（`glUniform4fv` 偏移、`eyeValues` 循环）从 `end_to_end.state` 拆出，须逐项等于上文「Uniform 换算」由 `geometry` 字段拼出的值，`end_to_end` 再用这组值渲染。改了这些写法而门禁报「无法解析」时，同步 `scripts/glsl/glsl_gate.py`。

## 修改常量的流程

1. 先改规格 v1（第 2–4 节）并说明理由与证据。
2. 改 `beauty_ref.py` 中的常量或公式（必要时同步 `gen_vectors.py` 的用例与自检）。
3. 重新生成 `beauty_vectors.json`，确认自检全部通过、文件不超过 600 KB，并检查 diff 只出现在预期的段。
4. 同步三端实现并跑测试：Android 隔离副本 `testReleaseUnitTest`、`make app-native-glsl-test`、`make app-native-effects-test`。
5. 三处（参考模型、向量、三端实现）放进同一次提交。
