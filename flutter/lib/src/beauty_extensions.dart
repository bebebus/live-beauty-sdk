/// Portrait background processing modes supported by the C++17 RHI pipeline.
enum LiveBeautyVirtualBackgroundMode {
  none('none'),
  blur('blur'),
  virtualImage('virtual_image'),
  virtualBlur('virtual_blur');

  const LiveBeautyVirtualBackgroundMode(this.wireName);

  final String wireName;
}

/// Face-anchored 2D AR sticker attachment points (aligned with C++ `StickerAnchor`).
enum LiveBeautyStickerAnchor {
  foreheadCrown(0),
  eyesBridge(1),
  noseTip(2),
  cheeks(3),
  fullFace(4);

  const LiveBeautyStickerAnchor(this.id);

  final int id;
}

/// Configuration for one of up to 4 simultaneous 2D AR face sticker layers.
class LiveBeautySticker {
  const LiveBeautySticker({
    this.enabled = true,
    this.anchor = LiveBeautyStickerAnchor.foreheadCrown,
    required this.glTextureId,
    this.scale = 1.0,
    this.offsetX = 0.0,
    this.offsetY = 0.0,
    this.rotationRad = 0.0,
    this.opacity = 1.0,
    this.textureAspect = 1.0,
  });

  final bool enabled;
  final LiveBeautyStickerAnchor anchor;
  final int glTextureId;
  final double scale;
  final double offsetX;
  final double offsetY;
  final double rotationRad;
  final double opacity;
  final double textureAspect;

  void validate() {
    if (scale < 0.1 || scale > 4.0) {
      throw RangeError('scale must be in [0.1, 4.0], got $scale');
    }
    if (opacity < 0.0 || opacity > 1.0) {
      throw RangeError('opacity must be in [0.0, 1.0], got $opacity');
    }
    if (textureAspect <= 0.05) {
      throw RangeError('textureAspect must be > 0.05, got $textureAspect');
    }
  }

  Map<String, Object> toMap() => <String, Object>{
        'enabled': enabled,
        'anchor': anchor.id,
        'glTextureId': glTextureId,
        'scale': scale,
        'offsetX': offsetX,
        'offsetY': offsetY,
        'rotationRad': rotationRad,
        'opacity': opacity,
        'textureAspect': textureAspect,
      };
}

/// Pipeline execution options for dual-face co-streaming and FrameDelayRing alignment.
class LiveBeautyPipelineOptions {
  const LiveBeautyPipelineOptions({
    this.maxFaces = 1,
    this.frameDelayFrames = 0,
  });

  final int maxFaces;
  final int frameDelayFrames;

  void validate() {
    if (maxFaces < 1 || maxFaces > 2) {
      throw RangeError('maxFaces must be 1 or 2, got $maxFaces');
    }
    if (frameDelayFrames < 0 || frameDelayFrames > 2) {
      throw RangeError('frameDelayFrames must be in [0, 2], got $frameDelayFrames');
    }
  }

  Map<String, Object> toMap() => <String, Object>{
        'maxFaces': maxFaces,
        'frameDelayFrames': frameDelayFrames,
      };
}

/// Real-time performance and tracking telemetry reported by the C++17 core.
class LiveBeautyPerfStats {
  const LiveBeautyPerfStats({
    this.lastFrameMs = 0.0,
    this.emaFrameMs = 0.0,
    this.fps = 0.0,
    this.loadLevel = 0,
    this.facePresence = 0.0,
    this.hairShown = 0.0,
    this.activePassesMask = 0,
    this.frameCount = 0,
    this.activeFaces = 0,
  });

  final double lastFrameMs;
  final double emaFrameMs;
  final double fps;
  final int loadLevel;
  final double facePresence;
  final double hairShown;
  final int activePassesMask;
  final int frameCount;
  final int activeFaces;

  factory LiveBeautyPerfStats.fromMap(Map<Object?, Object?>? raw) {
    if (raw == null) return const LiveBeautyPerfStats();
    double d(String k) => (raw[k] as num?)?.toDouble() ?? 0.0;
    int i(String k) => (raw[k] as num?)?.toInt() ?? 0;
    return LiveBeautyPerfStats(
      lastFrameMs: d('lastFrameMs'),
      emaFrameMs: d('emaFrameMs'),
      fps: d('fps'),
      loadLevel: i('loadLevel'),
      facePresence: d('facePresence'),
      hairShown: d('hairShown'),
      activePassesMask: i('activePassesMask'),
      frameCount: i('frameCount'),
      activeFaces: i('activeFaces'),
    );
  }
}
