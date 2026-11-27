# Bebebus Live Beauty SDK

> 中文 | [English](./README.en.md)

[![CI](https://github.com/bebebus/live-beauty-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/bebebus/live-beauty-sdk/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Version-v3.9.4-1E88E5.svg)](./CHANGELOG.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Free Forever](https://img.shields.io/badge/Commercial_Use-Free_Forever-2E7D32.svg)](#开源许可永久免费承诺与赞助)
[![Live Demo: dh.video](https://img.shields.io/badge/在线体验-dh.video-FF4D6D?logo=googlechrome&logoColor=white)](https://dh.video)
[![Android API](https://img.shields.io/badge/Android-API_24%2B-3DDC84?logo=android&logoColor=white)](./android)
[![iOS](https://img.shields.io/badge/iOS-15.0%2B-000000?logo=apple&logoColor=white)](./ios)
[![Flutter](https://img.shields.io/badge/Flutter-3.22%2B-02569B?logo=flutter&logoColor=white)](./flutter)

**Bebebus Live Beauty SDK** 是一套面向实时互动直播、连麦、音视频通话与相机拍摄场景的高性能开源端侧 GPU 美颜引擎。底层由零第三方重型依赖的**统一 C++17 渲染与数学内核（`core/`）**驱动，覆盖 **Android 原生（OpenGL ES 2.0/3.0 + JNI）**、**iOS / macOS 原生（Apple Metal 满血引擎 + CoreImage 兼容层）**、**Web（WebGL 2.0 / Emscripten RHI）** 与 **Flutter 跨端插件**。

SDK 与推流 / RTC 传输层**彻底解耦**——不绑定任何特定推流器或私有云鉴权，无需联网申请 License，开箱内置离线 MediaPipe 视觉模型、10 款电影级 `64³` 3D LUT 与 5 套高精度 UV 美妆贴图，可直接嵌入 **WebRTC、Agora（声网）、Zego（即构）、Tencent TRTC、CameraX、MediaCodec、AVCaptureSession** 等任意音视频管线。

### 在线体验与实机测试站点

- **官方体验测试站点**：[https://dh.video](https://dh.video)
- 该站点已完整接入 **Bebebus Live Beauty SDK** 生产管线，可直接在线查看真实直播间美颜画质，或通过站点获取客户端进行 **478 点微分同胚微整、双频奶油肌磨皮、6 层 PBR 美妆、沙龙级染发、CAT02 自动白平衡与 `64³` 3D LUT** 真机开播体验。

---

## 核心特性

### 1. 478 点 $C^\infty$ 微分同胚微整 & 双人同屏零抖动追踪
- **零背景扭曲 · 零轮廓残影 · 左右脸严格对称**：基于 MediaPipe 478 点高密人脸关键点与外扩双环锚定边界（`548` 顶点 / `1086` 三角形），将背景位移物理锁死为 `0`；结合 **2D 平顶纯剪切轮廓场（Pure-Shear Plateau）**、**Boltzmann 加权软上确界对称宽度**、**窄侧大转头防塌陷保护**与**雅可比安全限幅 + 翻转二分校正**，彻底杜绝“门框拉弯、转头大小脸、侧脸双影”。
- **15 项精细面部与颅顶微整**：瘦脸（下巴独立抗拉长补偿，不压低发际线）、V 脸、窄脸、小脸、瘦颧骨、**高颅顶（蓬松头包脸 `cranialTop`）**、下巴长短、额头高低、大眼（外眦独立护角）、眼距、眼角、瘦鼻、长鼻、嘴型大小、微笑嘴角。
- **手抖零滞后 & 防果冻波纹（`FaceMeshFilter` / `FaceTimeline`）**：117 点严格刚体 Procrustes 基底 + `SE(2)` 极分解统一旋转/平移滤波 + 4 区域 One-Euro 滤波 + 区域均值滞后快速追赶，同时支持 **双人同屏独立美颜**（`maxFaces = 2`，`MultiFaceTracker` 重复脸 NMS + 最小代价关联，交叉走位不互跳 ID，零额外 1080p FBO）。

### 2. 双频奶油肌磨皮、肤色光学 & 468 点解剖子网格精修
- **绿通道感知核双频导向滤波 + 8 方向毛流保护**：`1/3` 保边导向滤波 + `1/9` 大核匀光 + `RGBA16F` 中间高精度管线（按硬件档位自动启用），采用绿通道主导感知距离抵抗室内传感器色噪，并通过 8 方向 `lineScore` 精准保留 `1 px` 眉眼/胡须毛流与 `2–4 px` 原生毛孔颗粒感；头发加权导向滤波（`kGuidedWeight*`）彻底消除刘海/发际线渗黑灰边。
- **冷瓷去黄美白、颈面无缝过渡 & 影棚级光学质感**：
  - 亮度归一化身体/颈部肤色门控（`neckTone` + `portraitPrior` 空间保护），保证锁骨、脖颈、胸口平滑提亮无暗块，同时防止暖色木墙/背景泛白；
  - **3D 真实顶点法线 6-DoF 影棚重打光（`relight`）**：内置 `natural`、`studio_softbox`、`butterfly`、`rembrandt` 4 套随头转动的三点布光；
  - **T 区去油控光（`oilControl`）**、**多尺度孤立痘印/红斑修复（`blemishRemove`）** 与 **可插拔 AI 皮肤残差钩子（`setAiResidualTexture`）**。
- **468 点解剖子网格原位精修（零额外 1080p FBO）**：去黑眼圈（YCbCr 暖化色度迁移且精准避开卧蚕带）、弧形法令纹、18 块内唇孔洞三角形白牙（开口门控，物理杜绝浅色水光唇釉被漂白）、28 块眼裂三角形精准亮眼。

### 3. 6 层 PBR 实时美妆 & 沙龙级发丝保留染发
- **6 层 UV 实时美妆**：丝绒/水光 6 色口红（含 6-DoF 视角镜面滑移、开口内唇几何镂空防染牙与暗唇提亮防发黑）、3 款腮红、自然/挑眉、4 色眼影、眼线、卧蚕、W3C 柔光 3D 鼻影/高光、眨眼门控虹膜美瞳与角膜 3D 视差双眼神光。
- **环境光自适应 & 语义遮挡防糊妆**：支持环境光色温自动融入妆面（`ambientMakeupBlend`）；语义遮挡（刘海、眼镜、手部、水杯、话筒）严格解耦在磨皮美白之后、美妆之前执行——既**绝不在脸部/胸口挖黑块**，又能 100% 精准避让上妆，并在下半脸重度遮挡时自动柔和淡出唇妆/腮红/修容。
- **沙龙级实时染发 & 天使环光泽（`hairDye` / `hairShine`）**：内置 6 款发色（`milk_tea_brown`、`haze_blue`、`ash_purple`、`rose_gold`、`cool_black`、`caramel_copper`），采用黑色素 S 曲线提浅 + 前景/背景颜色投影边界精细抠像 + Kajiya-Kay 双瓣天使环高光，发丝细节清晰且对额头肤色保持 `0/255` 零渗色。

### 4. 电影级氛围后期、CIE D65 CAT02 自动白平衡 & `64³` 3D LUT
- **10 套内置 `512×512`（`64³`）专业直播间 3D LUT**：`natural_studio`、`clear_white`、`peach_milk`、`pure_pink`、`korean_glow`、`japanese_airy`、`soft_film`、`night_vibe`、`champagne_glam`、`sweet_berry`，同时支持 `https://` 远端 LUT 磁盘缓存热切换。
- **CIE D65 CAT02 3×3 色适应白平衡（`colorTemp` / `colorTint` / `autoWhiteBalance`）**：支持手动冷暖/洋红绿精调，以及由人脸肤色 ROI 实时驱动的阻尼自动白平衡（消除室内暖黄钨丝灯与荧光绿偏色，中性日光零偏移），并在无 Bloom 时**零额外 Pass 融合进 `kCompositePro`**。
- **多尺度软膝泛光 & 不压暗胶片颗粒（`bloom` / `filmGrain`）**：复用现有 `1/4` 与 `1/9` 分辨率缓冲（零额外 FBO），高光柔和绽放、暗部纯净不发灰。

### 5. 6 类语义解析、人像虚化、无绿幕虚拟背景、绿幕抠像 & 5 轴护手美体
- **6 类 & 19 类语义解析支持**：原生支持 MediaPipe `selfie_multiclass_256x256`（背景/头发/身体皮肤/面部皮肤/衣物/配饰手部）与 CelebAMask-HQ / BiSeNetV2 19 类语义图输入。
- **全形态背景处理**：支持 **人像虚化（Portrait Bokeh）**、**无绿幕虚拟背景替换/虚化（`kVirtualImage` / `kVirtualBlur`）** 与 **影棚级 YCbCr 绿幕/蓝幕抠像（`kChromaKey` + 溢色消除 + 背景光包裹）**；内置双尺度 RGB 引导时序精细抠像（`kMatteRefine`）与双人前景解剖护盾，抑制边缘闪烁与黑边穿帮。
- **5 轴护手美体**：基于 33 点姿态估计支持细腰、丰胸、C¹ 积分长腿、瘦肩、下巴感知天鹅颈，并内置双手掌心护盾（瘦腰绝不拉弯腰侧手指）。

### 6. 3D 头部遮挡 AR 挂件、2D 贴纸、手势 SDF 粒子 & `.lbfx` 特效包
- **3D 头部遮挡 AR 挂件（`Ar3dSolver`）**：468 点 3D 头部深度壳自动遮挡镜腿/耳饰后半段，支持曲面眼镜、圆柱头冠与自定义 glTF 网格、二阶弹簧摆物理惯性及 52 通道 Blendshape 表情触发（`smile`、`mouth_open`、`brow_up`、`blink`、`cheek_puff`、`tongue_out`）。
- **人脸锚定 2D AR 贴纸 & 手势 SDF 粒子爆发**：支持最多 4 层 2D 贴纸随头平移缩放旋转；内置零纹理 GPU 批量解析 SDF 粒子系统（心形/五星/菱形/星芒），支持比心、点赞、剪刀手、张掌手势触发。
- **声明式 `.lbfx` 特效包（`EffectBundleParser`）**：零依赖 JSON 清单一键打包美颜参数、滤镜、染发、2D 贴纸、3D 挂件与表情/手势联动规则。

### 7. 直播工程级性能、硬件自动分档与稳定度保障
- **极致性能与带宽优化**：统一 C++17 CPU 核单帧仅 **~55 μs**（占 30fps 预算 `< 0.2%`）；支持 `LoadOp::kDontCare` 消除 Tile GPU 冗余加载、零拷贝 `outputRt` FBO 直出（`nativeProcessFrameToFbo`）、Android CameraX `GL_TEXTURE_EXTERNAL_OES` 一步直入（`processOes`）、iOS `CVMetalTextureCache` 零拷贝 `CVPixelBuffer` 与 Apple Vision ANE 异步跟踪（`FLTVisionTracker`）。
- **全链路工程防护**：`DeviceTierPolicy`（`Lite` / `Standard` / `Ultra`）SoC/GPU 自动分档、`FrameDelayRing`（0..4 帧零滞后对齐）、远景自适应 `256×256` 人脸 ROI、GLES 3.0 双 PBO 异步 DMA 回读、IMU 陀螺仪手抖防残影联动（`setCameraShaking`）、`GpuDelegatePolicy` 原生崩溃自动降级看门狗、三级帧率温控与弱网码率联动，以及实时遥测 `BeautyPerfStats`。

> 更多 **CameraX (OES) / WebRTC / Agora / Zego / Tencent TRTC** 完整接入配方见 [docs/integration-recipes.md](./docs/integration-recipes.md)。

---

## 仓库目录结构

```text
live-beauty-sdk/
├── core/                     # 统一 C++17 渲染与数学内核（零重型第三方依赖）
│   ├── include/livebeauty/   # 数学核、RHI 抽象（GLES / Metal / WebGL2）、LiveBeautyEngine、Ar3dSolver、EffectBundleParser
│   ├── src/                  # 纯 C++17 实现、RenderPipeline、35 套跨后端着色器（GLES/JNI、Metal/MSL、WebGL2）
│   └── tests/                # CTest 数学单测、Mock RHI 管线测试、CPU 微基准与 macOS 真机 Metal GPU 像素门禁
├── android/                  # Android 独立原生库（io.bebebus:live-beauty:3.9.4, minSdk 24, arm64-v8a/armeabi-v7a .so）
│   └── live-beauty/          # OpenGL ES 2.0/3.0 引擎、JNI 桥接、MediaPipe 异步跟踪器、内置模型/LUT/美妆与 37 套 JVM 单测
├── ios/                      # iOS 独立原生库（LiveBeautySDK.podspec 3.9.4, iOS 15.0+）
│   ├── Classes/              # FLTLiveBeautyMetalEngine（Metal 满血）+ FLTVisionTracker（ANE）+ FLTVideoEffects（CoreImage 兼容）
│   ├── Resources/            # 离线 face_landmarker.task 模型
│   └── Tests/                # macOS / iOS CoreImage 像素渲染与平滑器回归测试
├── flutter/                  # Flutter 跨端插件封装（live_beauty_sdk 3.9.4）
│   ├── lib/                  # LiveBeautyConfig（60+ 参数与 5 套预设）、LiveBeautyController、扩展类型与开源许可注册
│   └── test/                 # Dart 契约、白名单校验与 MethodChannel 单元测试
├── reference/                # 跨端数学参考实现（beauty_ref.py）、黄金向量（beauty_vectors.json）与资产确定性生成器
├── _tooling/                 # 44+35 着色器无头 WebGL1 门禁、模型 SHA256 校验、脱敏扫描与一键 8 门禁回归脚本
├── docs/                     # 接入配方（integration-recipes.md）与架构规划文档
├── ARCHITECTURE.md           # 渲染管线拓扑、统一 C++17 内核与微分同胚网格算法深度说明
└── CHANGELOG.md              # 完整版本演进记录（v1.0.0 → v3.9.4）
```

---

## 快速接入指南

### 1. Android 原生接入（WebRTC / Agora / Zego / TRTC / CameraX）

将 `android/live-beauty` 作为 Gradle 子模块引入（或通过 `maven-publish` 发布 / 直接使用 `live-beauty-release.aar`，坐标 `io.bebebus:live-beauty:3.9.4`）：

```kotlin
import io.bebebus.livebeauty.LiveBeautyPresets
import io.bebebus.livebeauty.LiveBeautyProcessor
import io.bebebus.livebeauty.LiveBeautySticker
import io.bebebus.livebeauty.StickerAnchor
import io.bebebus.livebeauty.VirtualBackgroundMode

// 1. 创建处理器（可在主线程创建，updateConfig 线程安全）
val beautyProcessor = LiveBeautyProcessor(appContext = context.applicationContext)
beautyProcessor.setPipelineOptions(maxFaces = 2, frameDelayFrames = 0)
beautyProcessor.updateConfig(
    LiveBeautyPresets.koreanWaterGlow.copy(
        faceSlim = 0.42f,
        vFace = 0.30f,
        skinTexture = 0.50f,
        autoWhiteBalance = 0.55f,
    )
)

// 2. 在宿主 RTC / 相机的 OpenGL ES 线程初始化
beautyProcessor.onGlCreated(context)

// 3. 每帧传入竖屏正立 RGBA GL_TEXTURE_2D（或直接用 processOes 传入 CameraX/WebRTC OES 纹理）
val outputTexId = beautyProcessor.process(
    inputTexture = cameraRgbaTex2D,
    width = frameWidth,
    height = frameHeight,
    timestampNs = surfaceTimestampNs,
)

// 可选：无绿幕虚拟背景、2D 人脸贴纸、陀螺仪手抖防残影联动与实时性能遥测
beautyProcessor.setVirtualBackgroundTexture(bgTex2D, aspect = 9f / 16f, mode = VirtualBackgroundMode.VIRTUAL_IMAGE)
beautyProcessor.setSticker(0, LiveBeautySticker(anchor = StickerAnchor.FOREHEAD_CROWN, glTextureId = crownTex2D))
beautyProcessor.setCameraShaking(isGyroShaking)
val stats = beautyProcessor.perfStats() // emaFrameMs / fps / loadLevel / activePassesMask

// 4. 切换前后置摄像头时重置跟踪平滑状态
beautyProcessor.resetTracking()

// 5. GL 上下文销毁前释放资源
beautyProcessor.onGlReleased(contextAlive = true)
```

### 2. iOS 原生接入（CocoaPods / Metal 满血引擎 & CoreImage 兼容层）

在 `Podfile` 中引入 Pod：

```ruby
pod 'LiveBeautySDK', :git => 'https://github.com/bebebus/live-beauty-sdk.git', :tag => 'v3.9.4'
```

**推荐：使用统一 C++17 + Apple Metal 满血引擎（`FLTLiveBeautyMetalEngine`，与 Android 100% 特性对齐，零拷贝 `CVPixelBuffer` / `MTLTexture`）：**

```objc
#import <LiveBeautySDK/FLTLiveBeautyMetalEngine.h>

// 1. 初始化 Metal 引擎并配置参数（任意线程安全更新，支持 50+ 参数键）
self.metalBeauty = [[FLTLiveBeautyMetalEngine alloc] init];
[self.metalBeauty setConfiguration:@{
    @"enabled": @YES,
    @"smoothing": @0.55,
    @"skinOnly": @YES,
    @"whiten": @0.32,
    @"rosy": @0.18,
    @"faceSlim": @0.38,
    @"eyeEnlarge": @0.28,
    @"autoWhiteBalance": @0.50,
    @"hairDye": @0.0,
}];

// 2. 在相机采集队列驱动 Apple Vision ANE 异步分割/姿态，并零拷贝渲染 CVPixelBuffer
[self.metalBeauty submitVisionPixelBuffer:pixelBuffer frameTimestampMs:tsMs];
CVPixelBufferRef outBuffer = [self.metalBeauty copyProcessedPixelBuffer:pixelBuffer
                                                       frameTimestampMs:tsMs
                                                                  nowMs:nowMs];
if (outBuffer) {
    // 推送给 RTC / 渲染预览
    CVPixelBufferRelease(outBuffer);
}
```

> 若宿主仅需轻量 `CoreImage` / `CMSampleBufferRef` 管线，仍可直接使用完全向后兼容的 `FLTVideoEffects`（`copyProcessedSample:` / `beautyImage:configuration:geometry:presence:`）。

### 3. Flutter 接入（`live_beauty_sdk`）

在 `pubspec.yaml` 中添加依赖：

```yaml
dependencies:
  live_beauty_sdk:
    git:
      url: https://github.com/bebebus/live-beauty-sdk.git
      path: flutter
      ref: v3.9.4
```

通过 `LiveBeautyController` 实时驱动原生共享引擎（宿主原生插件可通过 `LiveBeautyEngineRegistry.sharedProcessor` / `LiveBeautySdkPlugin.sharedEffects` 直接消费帧）：

```dart
import 'package:live_beauty_sdk/live_beauty_sdk.dart';

// 注册内置 MediaPipe / GPUPixel Apache-2.0 开源声明
registerLiveBeautyLicenses();

final controller = LiveBeautyController();
await controller.initialize(initialConfig: LiveBeautyConfig.naturalStudio);

// 实时热更新美颜 / 微整 / 美妆 / 染发 / 3D LUT 参数
await controller.setConfig(
  LiveBeautyConfig.clearColdWhite.copyWith(
    faceSlim: 0.40,
    lipstick: 0.50,
    lipstickColor: 'cherry_red',
    lipstickFinish: 'glossy',
    hairDye: 0.35,
    hairDyeColor: 'milk_tea_brown',
    lutUrl: LiveBeautyConfig.assetLutUrl('japanese_airy'),
    lutIntensity: 0.45,
  ),
);
```

---

## 内置秀场级预设一览

| 预设名 | 适用场景 | 核心风格特征 |
| --- | --- | --- |
| `original` | 原画对比 / 零效果直通 | 零拷贝跳过全部 GPU Pass |
| `naturalStudio` | 日常连麦 / 才艺直播（默认推荐） | 保留 45% 原生肤质微纹理、轻度瘦脸大眼、`natural_studio` 3D LUT |
| `clearColdWhite` | 带货 / 女团高显白直播间 | 冷白奶油瓷肌、水光豆沙唇釉、蜜桃苹果肌腮红、双眼神光 |
| `koreanWaterGlow` | 韩系蜜桃水光妆 | 立体柔光高光、40% 自然卧蚕、蜜桃粉眼影、琥珀棕美瞳 |
| `mensClean` | 男士清爽主播档 | 65% 高比例肤质与胡须毛流保留、消黑眼圈提神、自然骨相微收 |

---

## 一键本地验证与回归门禁

本仓库提供完整的 **8 道自动化质量门禁**，覆盖脱敏扫描、模型 SHA256、548 顶点拓扑确定性、无头 Chrome WebGL1 门禁（44 段 Android GLSL + 28 变异体 + 35 套 C++ GLES 着色器 `highp`/`mediump` 双精度链接 + 10 套 `64³` 3D LUT）、CoreImage 像素回归、Android 37 套 JVM 单测与 Release AAR、Flutter 静态分析与单测，以及 C++17 数学/RHI/Apple Metal 真机 GPU 像素门禁（含 7 场景定量画质基准）：

```bash
# 运行全部 8 道质量门禁
bash _tooling/run-all-tests.sh
```

更多底层算法与渲染图拓扑说明请参阅 [ARCHITECTURE.md](./ARCHITECTURE.md)，版本变更明细请参阅 [CHANGELOG.md](./CHANGELOG.md)。

---

## 开源许可、永久免费承诺与赞助

- **允许免费商用**：本项目基于 **[Apache License 2.0](./LICENSE)** 开源发布，可免费用于任意个人、开源或商业产品，无需申请商业授权或联网鉴权。
- **永久免费承诺**：本项目郑重承诺**永不收费，永远不会推出任何收费版本、付费解锁或收费的高级功能**——所有美颜、微整、美妆、染发、3D AR、虚拟背景与性能优化能力全部平等开源，持续免费维护。
- **自愿赞助支持**：如果本项目对你的产品或研发有所帮助，欢迎通过以下地址自愿赞助支持项目持续演进（EVM 链通用，如 ETH / USDT / USDC / BSC / Arbitrum / Polygon 等）：
  - `0x70Ce069e59B3a5B0e3d83a8500094b6666666666`

第三方开源致谢详见 [NOTICE](./NOTICE)：
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe)（Apache License 2.0）—— 端侧实时人脸/人体/分割视觉推理运行时与预训练模型。
- [PixPark GPUPixel](https://github.com/pixpark/gpupixel)（Apache License 2.0）—— 标准 468 点人脸 UV 拓扑参考。
