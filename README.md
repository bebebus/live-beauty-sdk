# Bebebus Live Beauty SDK

> 中文 | [English](./README.en.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Live Demo: dh.video](https://img.shields.io/badge/在线体验-dh.video-FF4D6D?logo=googlechrome&logoColor=white)](https://dh.video)
[![Android API](https://img.shields.io/badge/Android-API_24%2B-3DDC84?logo=android&logoColor=white)](./android)
[![iOS](https://img.shields.io/badge/iOS-15.0%2B-000000?logo=apple&logoColor=white)](./ios)
[![Flutter](https://img.shields.io/badge/Flutter-3.22%2B-02569B?logo=flutter&logoColor=white)](./flutter)

**Bebebus Live Beauty SDK** 是一套面向实时互动直播、连麦、音视频通话与相机拍摄场景的高性能开源端侧 GPU 美颜引擎，覆盖 **Android 原生（OpenGL ES 2.0）**、**iOS 原生（CoreImage + Metal）** 与 **Flutter 跨端插件**。

SDK 与推流 / RTC 传输层**彻底解耦**——不绑定任何特定推流器或私有云鉴权，无需联网申请 License，开箱内置离线 MediaPipe 视觉模型、10 款电影级 `64³` 3D LUT 与 5 套高精度 UV 美妆贴图，可直接嵌入 **WebRTC、Agora（声网）、Zego（即构）、Tencent TRTC、CameraX、MediaCodec、AVCaptureSession** 等任意音视频管线。

### 在线体验与实机测试站点

- **官方体验测试站点**：[https://dh.video](https://dh.video)
- 该站点已完整接入 **Bebebus Live Beauty SDK** 生产管线，可直接在线查看真实直播间美颜画质，或通过站点获取客户端进行 **478 点微分同胚微整、1080p 导向滤波磨皮、胡须边界毛流保护、6 层 UV 美妆与 3D LUT** 真机开播体验。

---

## 核心特性

- **零背景扭曲 · 零轮廓残影 · 左右脸严格对称（478 点 $C^\infty$ 微分同胚网格）**
  - 基于 MediaPipe 478 点高密人脸关键点与外扩双环锚定边界（`548` 顶点 / `1086` 三角形），将背景位移物理锁死为 `0`。
  - 采用 **2D 平顶纯剪切轮廓场（Pure-Shear Plateau）** + **Boltzmann 加权软上确界对称宽度** + **雅可比安全限幅同步松弛**，彻底解决传统液化美颜“门框拉弯、转头大小脸、侧脸轮廓双影”三大顽疾。
  - 支持 **15 项精细面部与颅顶微整**：瘦脸、V 脸、窄脸、小脸、瘦颧骨、**高颅顶（蓬松头包脸 `cranialTop`）**、下巴长短、额头高低、大眼、眼距、眼角、瘦鼻、长鼻、嘴型大小、微笑嘴角。
- **双频奶油肌导向滤波磨皮 & 468 点解剖子网格精修**
  - **1/3 保边导向滤波 + 1/9 大核匀光双频架构**：在 100% 保留 `2–4 px` 原生毛孔颗粒感与眉眼唇/男士胡须边缘毛流的同时，抚平 `18–40 px` 泪沟暗沉与面中色块，并通过骨相光比门控保留鼻梁与下颌立体阴影。
  - **468 点解剖子网格原位精修（零额外 1080p FBO）**：去黑眼圈精准避开卧蚕带、弧形法令纹紧贴大笑鼻唇沟、18 个内唇孔洞三角形白牙物理杜绝浅色水光唇釉被漂白、28 个眼裂三角形精准亮眼。
- **6 层 UV 实时美妆 & `64³` 3D LUT 调色**
  - 支持丝绒/水光镜面口红（6 色）、苹果肌/微醺/眼下腮红、自然/挑眉、4 色眼影、眼线、卧蚕、鼻影修容高光、虹膜遮罩美瞳与双眼神光。
  - 内置 **10 张 `512×512`（`64³`）专业直播间 3D LUT**（`natural_studio`、`clear_white`、`peach_milk`、`pure_pink`、`korean_glow`、`japanese_airy`、`soft_film`、`night_vibe`、`champagne_glam`、`sweet_berry`），同时支持 `https://` 远端 LUT 动态缓存热切换。
- **5 轴美体微整、人像背景虚化 & 无绿幕虚拟背景替换**
  - 基于 33 点姿态估计支持细腰、丰胸、长腿、瘦肩、天鹅颈；结合人像分割 + 头发语义飞发合并遮罩，同时支持 **人像虚化（Portrait Bokeh）** 与 **无绿幕虚拟背景图片替换（自动居中 Aspect-Fill / 虚拟背景柔化）**。
- **双人同屏独立美颜 & 发际线加权导向滤波 & 跨后端 `FrameDelayRing`**
  - 内置最优最小代价匹配多人脸关联器（`MultiFaceTracker`，支持 `maxFaces = 2` 双人连麦/合拍同屏独立 478 点微整、原位精修、合并皮肤遮罩、6 层美妆与 2D 贴纸，零额外 1080p FBO 且交叉走位不互跳 ID）。
  - 头发加权双频导向滤波（`kGuidedWeightBlur/Diff/Coeff`）在 GLES 与 Metal 双端对齐落地，彻底消除刘海/发际线紧邻白皙皮肤时的渗黑发灰光晕。
  - 更多 **CameraX (OES) / WebRTC / Agora / Zego / Tencent TRTC** 完整接入代码配方见 [docs/integration-recipes.md](./docs/integration-recipes.md)。
- **人脸锚定 2D AR 挂件贴纸 & CameraX OES 一步直入 & 实时遥测**
  - C++17 管线内置最多 **4 层同屏 2D AR 贴纸**（支持 `FOREHEAD_CROWN` 头饰 / `EYES_BRIDGE` 眼镜 / `NOSE_TIP` 鼻贴 / `CHEEKS` 颊贴 / `FULL_FACE` 面具），直接锁定变形后 478 点人脸姿态随头平移缩放旋转，零额外 1080p FBO 开销。
  - 支持 Android `GL_TEXTURE_EXTERNAL_OES` + `SurfaceTexture` 4×4 矩阵一步直入（`LiveBeautyProcessor.processOes(...)`），并提供实时性能遥测 `BeautyPerfStats`（帧耗时、EMA、FPS、负载档位、活跃 Pass 掩码）。
  - 统一 C++17 CPU 核心微基准（`livebeauty_bench`：时间线 + 478 点滤波 + 548 顶点 15 轴微分同胚形变 + 贴纸矩阵）单帧仅耗时 **~55 μs**（占 30fps 帧预算 `< 0.2%`）。
- **直播工程级稳定性保障**
  - **`FrameDelayRing` 零滞后音画/掩码对齐 + 远景自适应 ROI + GLES 3.0 双 PBO 异步回读**：半身/舞蹈远景自动切换 `256×256` 各向同性人脸 ROI（有效采样精度提升 `2.5–3.5×`），ES 3.0 上下文下启用双 PBO 硬件 DMA 零阻塞回读降低长播温升。
  - **GPU 崩溃看门狗（`GpuDelegatePolicy`）**：自动识别驱动级 TFLite GPU 原生崩溃并平滑降级 CPU，永不拖垮宿主进程。
  - **三级帧率温控与弱网码率联动（`BeautyLoadGovernor` & `BitrateEffectsTuning`）**：帧耗时超预算自动阶梯降载，弱网降码率时自动抑噪省码率。

---

## 仓库目录结构

```text
live-beauty-sdk/
├── core/                     # 统一 C++17 渲染与数学内核 (零第三方重型依赖，支持 Android GLES & Apple Metal 双后端)
│   ├── include/livebeauty/   # 数学核 (FaceMeshDeformer/Filter/Timeline/ROI/Body) + 轻量 RHI + LiveBeautyEngine
│   ├── src/                  # 纯 C++17 实现、RenderPipeline、GLES/JNI 后端与 Metal/MSL 后端
│   └── tests/                # CTest 数学单测、Mock RHI 管线测试与 macOS 真实 Metal GPU 像素门禁
├── android/                  # Android 独立原生库 (io.bebebus:live-beauty, minSdk 24, 含 arm64-v8a/armeabi-v7a .so)
│   └── live-beauty/          # OpenGL ES 2.0 渲染引擎、JNI 桥接、MediaPipe 异步跟踪器、内置模型/LUT/美妆资产与 36 套 JVM 单测
├── ios/                      # iOS 独立原生库 (LiveBeautySDK.podspec, iOS 15.0+, Metal C++17 满血引擎 + CoreImage 兼容层)
│   ├── Classes/              # FLTLiveBeautyMetalEngine (Metal 满血) + FLTVideoEffects (CoreImage 兼容) + MediaPipe 478 点跟踪
│   ├── Resources/            # 离线 face_landmarker.task 模型
│   └── Tests/                # macOS / iOS CoreImage 像素渲染与平滑器回归测试
├── flutter/                  # Flutter 跨端插件封装 (live_beauty_sdk)
│   ├── lib/                  # LiveBeautyConfig (40+ 参数与 5 套预设)、LiveBeautyController、LicenseRegistry 辅助注册
│   └── test/                 # Dart 契约、白名单校验与 MethodChannel 单元测试
├── reference/                # 跨端数学参考实现 (beauty_ref.py)、黄金向量 (beauty_vectors.json) 与资产确定性生成器
├── _tooling/                 # 44 着色器无头 WebGL1 门禁、模型 SHA256 校验、脱敏扫描与一键 8 门禁回归脚本
└── ARCHITECTURE.md           # 渲染管线拓扑、统一 C++17 内核与微分同胚网格算法深度说明
```

---

## 快速接入指南

### 1. Android 原生接入（WebRTC / Agora / Zego / TRTC / CameraX）

将 `android/live-beauty` 作为 Gradle 子模块引入（或直接使用构建产物 `live-beauty-release.aar`）：

```kotlin
import io.bebebus.livebeauty.LiveBeautyPresets
import io.bebebus.livebeauty.LiveBeautyProcessor

// 1. 创建处理器（可在主线程创建，updateConfig 线程安全）
val beautyProcessor = LiveBeautyProcessor(appContext = context.applicationContext)
beautyProcessor.updateConfig(
    LiveBeautyPresets.koreanWaterGlow.copy(
        faceSlim = 0.42f,
        vFace = 0.30f,
        skinTexture = 0.50f,
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

// 可选：无绿幕虚拟背景替换、2D 人脸贴纸与实时性能遥测
beautyProcessor.setVirtualBackgroundTexture(bgTex2D, aspect = 9f / 16f, mode = VirtualBackgroundMode.VIRTUAL_IMAGE)
beautyProcessor.setSticker(0, LiveBeautySticker(anchor = StickerAnchor.FOREHEAD_CROWN, glTextureId = crownTex2D))
val stats = beautyProcessor.perfStats() // emaFrameMs / fps / loadLevel / activePassesMask

// 4. 切换前后置摄像头时重置跟踪平滑状态
beautyProcessor.resetTracking()

// 5. GL 上下文销毁前释放资源
beautyProcessor.onGlReleased(contextAlive = true)
```

### 2. iOS 原生接入（CocoaPods / `AVCaptureSession` / RTC 自采集）

在 `Podfile` 中引入本地或远端 Pod：

```ruby
pod 'LiveBeautySDK', :git => 'https://github.com/bebebus/live-beauty-sdk.git'
```

在视频采集回调中直接处理 `CMSampleBufferRef`（或直接调用 `+[FLTVideoEffects beautyImage:configuration:geometry:presence:]` 处理 `CIImage`）：

```objc
#import <LiveBeautySDK/FLTVideoEffects.h>

// 1. 初始化并设置美颜参数（任意线程安全更新）
self.beautyEffects = [[FLTVideoEffects alloc] initWithFailure:^{
    NSLog(@"Beauty kernel fallback to pass-through");
} faceFailure:^{
    NSLog(@"Face tracker unavailable, skin/color still active");
}];

[self.beautyEffects setConfiguration:@{
    @"enabled": @YES,
    @"smoothing": @0.55,
    @"skinOnly": @YES,
    @"whiten": @0.32,
    @"rosy": @0.18,
    @"faceSlim": @0.38,
    @"eyeEnlarge": @0.28,
    @"filter": @"none",
    @"filterIntensity": @1.0,
}];

// 2. 在 AVCaptureVideoDataOutput 队列中输出美颜后的 CMSampleBufferRef
CMSampleBufferRef processed = [self.beautyEffects copyProcessedSample:sampleBuffer];
if (processed) {
    [rtcEngine pushExternalVideoFrame:processed];
    CFRelease(processed);
}
```

### 3. Flutter 接入（`live_beauty_sdk`）

在 `pubspec.yaml` 中添加依赖：

```yaml
dependencies:
  live_beauty_sdk:
    git:
      url: https://github.com/bebebus/live-beauty-sdk.git
      path: flutter
```

通过 `LiveBeautyController` 实时驱动原生共享引擎（宿主原生插件可通过 `LiveBeautyEngineRegistry.sharedProcessor` / `LiveBeautySdkPlugin.sharedEffects` 直接消费帧）：

```dart
import 'package:live_beauty_sdk/live_beauty_sdk.dart';

// 注册内置 MediaPipe / GPUPixel Apache-2.0 开源声明
registerLiveBeautyLicenses();

final controller = LiveBeautyController();
await controller.initialize(initialConfig: LiveBeautyConfig.naturalStudio);

// 实时无锁热更新 40+ 美颜/微整/美妆/3D LUT 参数
await controller.setConfig(
  LiveBeautyConfig.clearColdWhite.copyWith(
    faceSlim: 0.40,
    lipstick: 0.50,
    lipstickColor: 'cherry_red',
    lipstickFinish: 'glossy',
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

本仓库提供完整的跨端数学向量、41 段 GLSL 真浏览器编译/变异注入门禁、CoreImage 像素测试及 Android/Flutter 自动化单测：

```bash
# 运行全部 8 道质量门禁（脱敏检查 + 模型 SHA256 + 双端拓扑校验 + WebGL1 GLSL 门禁 + CoreImage 单测 + Android 单测/NDK AAR + Flutter 单测 + C++17/Metal GPU 像素门禁）
bash _tooling/run-all-tests.sh
```

更多底层算法细节请参阅 [ARCHITECTURE.md](./ARCHITECTURE.md)。

---

## 开源许可与致谢

本项目基于 **[Apache License 2.0](./LICENSE)** 开源发布，无需任何商业授权费用即可用于商业或开源产品。

第三方开源致谢详见 [NOTICE](./NOTICE)：
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe)（Apache License 2.0）—— 端侧实时人脸/人体/分割视觉推理运行时与预训练模型。
- [PixPark GPUPixel](https://github.com/pixpark/gpupixel)（Apache License 2.0）—— 标准 468 点人脸 UV 拓扑参考。
