# 主流音视频 / 相机引擎集成配方（Integration Recipes）

**Bebebus Live Beauty SDK v2.2.0** 与推流器及 RTC 传输层彻底解耦，Android 端只需传入标准的 `GL_TEXTURE_2D` 或 `GL_TEXTURE_EXTERNAL_OES` 纹理，即可无感接入任何实时音视频框架。

---

## 1. CameraX / Camera2（`SurfaceTexture` OES 纹理一步直入）

无需宿主自行编写 OES → 2D FBO 转换着色器，直接调用 `LiveBeautyProcessor.processOes(...)`：

```kotlin
val beauty = LiveBeautyProcessor(appContext = context.applicationContext)
beauty.updateConfig(LiveBeautyPresets.naturalStudio)
// 可选：开启双人同屏美颜 + 1 帧零滞后对齐
beauty.setPipelineOptions(maxFaces = 2, frameDelayFrames = 1)

// 在你的 GL 渲染线程初始化一次
beauty.onGlCreated(context)

// CameraX Preview.SurfaceProvider 的 SurfaceTexture 帧回调中：
val stMatrix = FloatArray(16)
surfaceTexture.updateTexImage()
surfaceTexture.getTransformMatrix(stMatrix)

val beautifiedTex2D = beauty.processOes(
    oesTexture = cameraOesTexId,
    stMatrix16 = stMatrix,
    width = frameWidth,
    height = frameHeight,
    timestampNs = surfaceTexture.timestamp,
)
// 将 beautifiedTex2D (GL_TEXTURE_2D) 直接绘制到预览 Surface 或 MediaCodec 输入 Surface
```

---

## 2. WebRTC Android（`VideoProcessor` / `SurfaceTextureHelper`）

```kotlin
class BeautyVideoProcessor(private val context: Context) : VideoProcessor {
    private var sink: VideoSink? = null
    private val beauty = LiveBeautyProcessor(appContext = context.applicationContext)
    private var glReady = false

    override fun onCapturerStarted(success: Boolean) {
        beauty.updateConfig(LiveBeautyPresets.clearColdWhite)
    }

    override fun onFrameCaptured(frame: VideoFrame) {
        val buf = frame.buffer as? VideoFrame.TextureBuffer ?: run {
            sink?.onFrame(frame); return
        }
        if (!glReady) { beauty.onGlCreated(context); glReady = true }
        val outTex = if (buf.type == VideoFrame.TextureBuffer.Type.OES) {
            beauty.processOes(buf.textureId, RendererCommon.convertMatrixFromAndroidGraphicsMatrix(buf.transformMatrix),
                buf.width, buf.height, frame.timestampNs)
        } else {
            beauty.process(buf.textureId, buf.width, buf.height, frame.timestampNs)
        }
        // 使用 outTex 包装新的 TextureBuffer(Type.RGB) 回调 sink?.onFrame(...)
    }

    override fun setSink(sink: VideoSink?) { this.sink = sink }
    override fun onCapturerStopped() { beauty.onGlReleased(contextAlive = false); glReady = false }
}
```

---

## 3. Agora（声网 RTC `IVideoFrameObserver` / 自采集纹理）

在 Agora `ExternalVideoFrame` 自采集或 `VideoFrameObserver` 纹理回调的 GL 线程中：

```kotlin
val outTex2D = beauty.process(
    inputTexture = cameraTex2D,
    width = width,
    height = height,
    timestampNs = timestampNs,
)
val agoraFrame = AgoraVideoFrame().apply {
    format = AgoraVideoFrame.FORMAT_TEXTURE_2D
    textureID = outTex2D
    stride = width
    height = height
    timeStamp = timestampNs / 1_000_000L
    eglContext14 = EGL14.eglGetCurrentContext()
}
rtcEngine.pushExternalVideoFrame(agoraFrame)
```

---

## 4. Zego（即构 Express `IZegoCustomVideoProcessHandler`）

```kotlin
override fun onCapturedUnprocessedTextureData(
    textureID: Int, width: Int, height: Int, referenceTimeMillisecond: Long, channel: ZegoPublishChannel
) {
    val outTex = beauty.process(textureID, width, height, referenceTimeMillisecond * 1_000_000L)
    expressEngine.sendCustomVideoProcessedTextureData(outTex, width, height, referenceTimeMillisecond, channel)
}
```

---

## 5. Tencent TRTC（`TRTCVideoFrameListener`）

```kotlin
override fun onProcessVideoFrame(srcFrame: TRTCCloudDef.TRTCVideoFrame, dstFrame: TRTCCloudDef.TRTCVideoFrame): Int {
    dstFrame.texture.textureId = beauty.process(
        inputTexture = srcFrame.texture.textureId,
        width = srcFrame.width,
        height = srcFrame.height,
        timestampNs = srcFrame.timestamp * 1_000_000L,
    )
    return 0
}
```

---

## 6. 高阶功能快捷开关（虚拟背景 / 2D AR 贴纸 / 双人同屏 / 性能遥测）

```kotlin
// 1. 无绿幕虚拟背景替换（自动居中 Aspect-Fill + 飞发保护）
beauty.setVirtualBackgroundTexture(bgTex2D, aspect = 9f / 16f, mode = VirtualBackgroundMode.VIRTUAL_IMAGE)

// 2. 最多 4 层人脸锚定 2D AR 挂件贴纸（头饰/眼镜/鼻贴/颊贴/面具）
beauty.setSticker(0, LiveBeautySticker(anchor = StickerAnchor.FOREHEAD_CROWN, glTextureId = crownTex))
beauty.setSticker(1, LiveBeautySticker(anchor = StickerAnchor.EYES_BRIDGE, glTextureId = glassesTex))

// 3. 双人连麦 / 合拍独立双脸美颜 + 零滞后帧对齐环
beauty.setPipelineOptions(maxFaces = 2, frameDelayFrames = 1)

// 4. 实时性能遥测面板
val stats = beauty.perfStats()
Log.d("Beauty", "ema=${stats.emaFrameMs}ms fps=${stats.fps} faces=${stats.activeFaces} load=${stats.loadLevel}")
```
