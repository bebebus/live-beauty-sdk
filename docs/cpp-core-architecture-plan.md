# Bebebus Live Beauty SDK 统一 C++17 渲染与数学内核架构计划

- 版本目标：`v2.0.0`（演进自 `v1.1.4`）
- 架构路线：**C++17 统一数学核 + 轻量 RHI 渲染编排层 + GLES (Android) / Metal (iOS & macOS) 双原生后端**
- Shader 策略：**手写精调 GLSL (Android GLES 2.0/3.0) + 等价 MSL (Apple Metal)**，通过双端自动化 GPU 像素探针门禁锁定数值等价（零运行期转译器依赖、零包体膨胀）
- 落地顺序：先在 `live-beauty-sdk` 开源仓按 **M1 -> M2 -> M3** 分里程碑完成全量门禁验收，稳定后再作为独立依赖接回业务客户端。

---

## 1. 设计目标与边界约束

1. **单一真相源（Single Source of Truth）**：
   - 全部核心数学模型、拓扑常量、形变算法、滤波/时间线状态机、ROI 选择、美妆色板、美体几何、负载治理与 RenderGraph Pass 编排全部收口至 `core/` C++17 库。
   - 消除 Android (Kotlin) 与 iOS (Objective-C) 双端算法代差，使 iOS 端直接获得 Android `v1.1.4` 全部能力（15 轴 $C^\infty$ 微分同胚网格、高颅顶、双频奶油肌导向滤波、468 点解剖子网格精修、6 层 UV 物理美妆、`64³` 3D LUT、5 轴美体与背景虚化）。
2. **零外部重型 C++ 依赖**：
   - `core/` 仅依赖 C++17 标准库及平台原生图形库（`GLESv2`/`GLESv3` 或 `Metal.framework`），不引入 ANGLE、glslang、SPIRV-Cross 或重型第三方运行时，确保移动端二进制增量极小、构建简单透明。
3. **对外 API 向前兼容**：
   - Android 保持 `LiveBeautyProcessor`、`VideoEffectsConfig`、`LiveBeautyPresets`、`BeautyEngine` 公开契约不变；
   - iOS 保持 `FLTVideoEffects` 公开接口兼容，内部无缝切换至 C++17 + Metal 满血引擎；
   - Flutter 保持 `LiveBeautyConfig` / `LiveBeautyController` 契约不变，并将原标注“仅 Android”的能力平滑解锁至 iOS。
4. **严格的零回归双端像素门禁**：
   - Android GLSL 继续受 `_tooling/glsl/glsl_gate.py`（无头 Chrome WebGL1）44 段着色器与像素探针保护；
   - Apple Metal MSL 新增 macOS 真实 Metal GPU 离线像素门禁（`test_metal_pixel_gate`），对标检验 `G1–G9`、`R1–R4`、`L1–L6`、`E1–E4` 等价性。

---

## 2. 目录结构规划

```text
live-beauty-sdk/
├── core/
│   ├── CMakeLists.txt                       # 跨平台构建：HOST 测试 / Android NDK / Apple Metal
│   ├── include/livebeauty/
│   │   ├── types.h                          # Vec2/Vec3/Vec4/Rect/TextureDesc 等基础类型
│   │   ├── config.h                         # BeautyConfig (40+ 全量参数 + 5 套秀场预设)
│   │   ├── math/
│   │   │   ├── face_mesh_topology.h         # 468/548 顶点、1086 三角形与解剖子网格 (由 Python 单源生成)
│   │   │   ├── face_geometry.h              # 478 点人脸几何与遮罩特征
│   │   │   ├── face_mesh_deformer.h         # 15 轴 C∞ 微分同胚形变 + 高颅顶 + Boltzmann 对称 + 雅可比限幅
│   │   │   ├── face_mesh_filter.h           # Procrustes 刚体 + One-Euro 分区滤波 + LID/LIP/EYE 加速
│   │   │   ├── face_timeline.h              # 时间线插值/外推 + eyeMakeupKeep 速度防残影
│   │   │   ├── face_roi_selector.h          # 远景 1:1 ROI 迟滞裁剪与坐标反投影
│   │   │   ├── face_makeup_uv.h             # 6 层美妆校准色板与风格参数
│   │   │   ├── body_geometry.h              # 33 点姿态 5 轴美体几何
│   │   │   └── beauty_tuning.h              # 质感参数、弱网码率联动、三级温控降载、帧步进
│   │   ├── rhi/
│   │   │   ├── rhi_types.h                  # 纹理/FBO/缓冲/管线状态描述符
│   │   │   ├── rhi_device.h                 # 轻量 RHI 抽象接口
│   │   │   └── render_pipeline.h            # 跨后端统一渲染管线编排器 (Pass 图调度)
│   │   └── engine.h                         # 跨平台统一引擎入口 (LiveBeautyEngine)
│   ├── src/
│   │   ├── config.cpp
│   │   ├── math/                            # 纯 C++17 数学实现 (零图形 API 依赖)
│   │   ├── rhi/
│   │   │   ├── render_pipeline.cpp          # 统一 Pass 调度器
│   │   │   ├── gles/                        # OpenGL ES 2.0/3.0 后端 + 精调 GLSL + 双 PBO 回读
│   │   │   └── metal/                       # Metal 后端 + 等价精调 MSL + CVPixelBuffer 零拷贝
│   │   └── jni/                             # Android JNI 桥接层
│   └── tests/                               # CTest 单元测试 + Mock RHI 管线测试 + Metal GPU 像素门禁
├── android/                                 # 链接 core/ CMake NDK 库，保留薄 Kotlin 门面与 MediaPipe 异步调度
├── ios/                                     # 链接 core/ + Metal 后端，保留 ObjC 门面与 MediaPipe 478 点跟踪
└── flutter/                                 # 跨端 Dart 插件封装
```

---

## 3. 分里程碑实施计划

### 里程碑 M1：C++17 统一数学与状态机内核（纯 CPU，100% 数值对齐）
- **范围**：
  1. 搭建 `core/CMakeLists.txt` 与零外部依赖 CTest 套件；
  2. 扩展 `reference/gen_face_topology.py` 同时生成并校验 `FaceMeshTopology.kt` 与 `core/src/math/face_mesh_topology.cpp`；
  3. 完整移植并逐不变量验证以下纯数学模块：
     - `BeautyConfig` & 5 套预设（与 `VideoEffectsConfig` / `LiveBeautyPresets` 一致）；
     - `FaceMeshTopology`（468/548 顶点、1086 三角形、UV、规范 3D 坐标、内唇 18 三角形、眼裂 28 三角形、卧蚕保护集）；
     - `FaceMeshDeformer`（15 轴含高颅顶 `cranialTop`、2D 平顶纯剪切场、Boltzmann SoftMax 对称化、雅可比防翻转、外环锚点锁零）；
     - `FaceMeshFilter`（Procrustes 刚体分解 + 分区 One-Euro + `LID_RING` / `LIP_RING` / `EYE_REGION` 提速）；
     - `FaceTimeline`（插值对齐、`steadyEye`、`eyeMakeupKeep` 35ms 快退 / 160ms 渐入）；
     - `FaceRoiSelector`（`0.30/0.36` 迟滞、`1:1` 裁剪与精确反投影、单帧漏检回退）；
     - `FaceMakeupUv`（6 色校准口红、腮红/眉/眼影/美瞳色板）；
     - `FaceGeometry`、`BodyGeometry`（5 轴美体）、`BeautyQualityTuning`、`EffectsLoadGovernor`、`BitrateEffectsTuning`。
- **验收标准**：
  - `cmake --build` + `ctest --output-on-failure` 全绿；
  - 覆盖现有 Kotlin 关键几何/形变/时间线/ROI/滤波核心断言，所有不变量（1086 三角正面积、高颅顶下颌锚点零位移、静止抖动 `≤ 0.8 px@720`、`eyeMakeupKeep` 快退恢复等）100% 通过。

### 里程碑 M2：轻量 RHI、统一渲染编排层与 Android GLES/JNI 落地
- **范围**：
  1. 设计最小充分 RHI 抽象（`TextureHandle`、`RenderTarget`、`BufferHandle`、`ShaderProgram`、`DrawCommand`），严守无虚函数热点与零每帧堆分配；
  2. 实现 `RenderPipeline` 统一编排器，将 Pass 顺序、FBO 复用池、Uniform 打包与降载开关收口到 C++，并用 Mock RHI 单测锁定 Pass 拓扑；
  3. 实现 `GlesDevice`（GLES 2.0/3.0 后端）与精调 GLSL 着色器集（与现有 44 段 GLSL 数学逐项等价，守住 `FRAGMENT_PRO` 15 uniform 预算）及 ES3 双 PBO 异步回读保护；
  4. 接入 `android/live-beauty` 的 `externalNativeBuild` (CMake + NDK) 与 JNI 桥接，逐步将高频 CPU 数学计算（形变、滤波、时间线、ROI、美体）与 GL 管线切换至 C++ 内核；
  5. 验证 Android Debug/Release 单测、Release AAR（含 `arm64-v8a` / `armeabi-v7a` `.so`）与现有 7 道门禁全部通过。

### 里程碑 M3：iOS/macOS Metal 后端、等价 MSL 着色器与双端像素门禁
- **范围**：
  1. 实现 `MetalDevice`（ObjC++ / `MTLDevice` / `MTLCommandQueue` / `CVMetalTextureCache` 零拷贝）；
  2. 编写与 GLSL 数学逐项等价的精调 `beauty_shaders.metal`（覆盖背景虚化、5 轴美体、548 顶点微分同胚人脸形变、468 点解剖子网格精修、双频奶油肌导向滤波、6 层 UV 物理美妆、主合成 + `64³` 3D LUT、时域降噪）；
  3. 将 `ios/Classes/FLTVideoEffects` 升级对接 `core/` + Metal 后端（保留现有 ObjC 头文件 API 兼容），补齐 iOS 端内置 10 张 LUT 与 5 张美妆贴图资源；
  4. 编写 macOS 真实 Metal GPU 离线像素门禁 `test_metal_pixel_gate.mm`，复刻 `G1–G9`、`R1–R4`、`L1–L6`、`E1–E4` 全部像素探针并纳入 `_tooling/run-all-tests.sh` 与 GitHub Actions `macos-latest` CI；
  5. 更新 `ARCHITECTURE.md`、`README.md`、`CHANGELOG.md` 并完成全量回归验收。
