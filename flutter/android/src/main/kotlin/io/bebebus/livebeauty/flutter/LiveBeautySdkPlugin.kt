package io.bebebus.livebeauty.flutter

import android.content.Context
import android.os.Handler
import android.os.Looper
import io.bebebus.livebeauty.LiveBeautyProcessor
import io.bebebus.livebeauty.LiveBeautySticker
import io.bebebus.livebeauty.StickerAnchor
import io.bebebus.livebeauty.VideoEffectsConfig
import io.bebebus.livebeauty.VirtualBackgroundMode
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/**
 * Process-wide registry exposing the shared [LiveBeautyProcessor] to host RTC / camera
 * plugins (Agora, Zego, TRTC, WebRTC, CameraX, MediaCodec) inside the same Flutter app.
 */
object LiveBeautyEngineRegistry {
    @Volatile
    var sharedProcessor: LiveBeautyProcessor? = null
}

class LiveBeautySdkPlugin : FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    private lateinit var appContext: Context
    private lateinit var methodChannel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var eventSink: EventChannel.EventSink? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        methodChannel = MethodChannel(binding.binaryMessenger, "io.bebebus.livebeauty/controller")
        methodChannel.setMethodCallHandler(this)
        eventChannel = EventChannel(binding.binaryMessenger, "io.bebebus.livebeauty/events")
        eventChannel.setStreamHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        eventSink = null
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
    }

    private fun emitEvent(payload: Map<String, Any>) {
        mainHandler.post {
            eventSink?.success(payload)
        }
    }

    private fun ensureProcessor(): LiveBeautyProcessor {
        LiveBeautyEngineRegistry.sharedProcessor?.let { return it }
        val created = LiveBeautyProcessor(
            appContext = appContext,
            onFailure = { emitEvent(mapOf("type" to "engineFailed")) },
            onFaceFailure = { emitEvent(mapOf("type" to "faceTrackingFailed")) },
            onLoadReduced = { level -> emitEvent(mapOf("type" to "loadReduced", "level" to level)) },
            onLegsHidden = { hidden -> emitEvent(mapOf("type" to "legsHiddenChanged", "hidden" to hidden)) },
            onLowLight = { lowLight -> emitEvent(mapOf("type" to "lowLightChanged", "lowLight" to lowLight)) },
        )
        LiveBeautyEngineRegistry.sharedProcessor = created
        return created
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        try {
            when (call.method) {
                "initialize", "setConfig" -> {
                    val map = call.arguments as? Map<*, *>
                        ?: return result.error("invalid_args", "Expected configuration map.", null)
                    val config = VideoEffectsConfig.parse(map)
                    ensureProcessor().updateConfig(config)
                    result.success(null)
                }
                "resetTracking" -> {
                    LiveBeautyEngineRegistry.sharedProcessor?.resetTracking()
                    result.success(null)
                }
                "setPipelineOptions" -> {
                    val map = call.arguments as? Map<*, *>
                        ?: return result.error("invalid_args", "Expected options map.", null)
                    val maxFaces = (map["maxFaces"] as? Number)?.toInt() ?: 1
                    val frameDelayFrames = (map["frameDelayFrames"] as? Number)?.toInt() ?: 0
                    ensureProcessor().setPipelineOptions(maxFaces, frameDelayFrames)
                    result.success(null)
                }
                "setVirtualBackgroundTexture" -> {
                    val map = call.arguments as? Map<*, *>
                        ?: return result.error("invalid_args", "Expected virtual background map.", null)
                    val glTex = (map["glTextureId"] as? Number)?.toInt() ?: 0
                    val aspect = (map["aspect"] as? Number)?.toFloat() ?: 0f
                    val modeStr = map["mode"] as? String ?: "none"
                    val mode = VirtualBackgroundMode.entries.firstOrNull { it.wireName == modeStr }
                        ?: VirtualBackgroundMode.NONE
                    val strength = (map["strength"] as? Number)?.toFloat() ?: 1f
                    ensureProcessor().setVirtualBackgroundTexture(glTex, aspect, mode, strength)
                    result.success(null)
                }
                "setSticker" -> {
                    val map = call.arguments as? Map<*, *>
                        ?: return result.error("invalid_args", "Expected sticker payload.", null)
                    val slot = (map["slot"] as? Number)?.toInt() ?: 0
                    val stickerMap = map["sticker"] as? Map<*, *>
                    val sticker = stickerMap?.let { sm ->
                        val anchorId = (sm["anchor"] as? Number)?.toInt() ?: 0
                        val anchor = StickerAnchor.entries.firstOrNull { it.id == anchorId }
                            ?: StickerAnchor.FOREHEAD_CROWN
                        LiveBeautySticker(
                            enabled = sm["enabled"] as? Boolean ?: true,
                            anchor = anchor,
                            glTextureId = (sm["glTextureId"] as? Number)?.toInt() ?: 0,
                            scale = (sm["scale"] as? Number)?.toFloat() ?: 1f,
                            offsetX = (sm["offsetX"] as? Number)?.toFloat() ?: 0f,
                            offsetY = (sm["offsetY"] as? Number)?.toFloat() ?: 0f,
                            rotationRad = (sm["rotationRad"] as? Number)?.toFloat() ?: 0f,
                            opacity = (sm["opacity"] as? Number)?.toFloat() ?: 1f,
                            textureAspect = (sm["textureAspect"] as? Number)?.toFloat() ?: 1f,
                        )
                    }
                    ensureProcessor().setSticker(slot, sticker)
                    result.success(null)
                }
                "clearStickers" -> {
                    LiveBeautyEngineRegistry.sharedProcessor?.clearStickers()
                    result.success(null)
                }
                "getPerfStats" -> {
                    val st = LiveBeautyEngineRegistry.sharedProcessor?.perfStats()
                    result.success(
                        mapOf(
                            "lastFrameMs" to (st?.lastFrameMs ?: 0f),
                            "emaFrameMs" to (st?.emaFrameMs ?: 0f),
                            "fps" to (st?.fps ?: 0f),
                            "loadLevel" to (st?.loadLevel ?: 0),
                            "facePresence" to (st?.facePresence ?: 0f),
                            "hairShown" to (st?.hairShown ?: 0f),
                            "activePassesMask" to (st?.activePassesMask ?: 0),
                            "frameCount" to (st?.frameCount ?: 0L),
                            "activeFaces" to (st?.activeFaces ?: 0),
                        ),
                    )
                }
                "dispose" -> {
                    LiveBeautyEngineRegistry.sharedProcessor = null
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        } catch (e: IllegalArgumentException) {
            result.error("invalid_effects", e.message, null)
        } catch (e: IllegalStateException) {
            result.error("invalid_effects", e.message, null)
        }
    }
}
