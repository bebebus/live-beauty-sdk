import 'package:flutter/foundation.dart';

/// Built-in 1D/matrix color grading styles supported on both Android and iOS.
enum LiveBeautyColorFilter {
  none('none'),
  warm('warm'),
  cool('cool'),
  monochrome('monochrome'),
  sepia('sepia'),
  rose('rose');

  const LiveBeautyColorFilter(this.wireValue);

  final String wireValue;

  static LiveBeautyColorFilter fromWire(String value) {
    for (final LiveBeautyColorFilter candidate in values) {
      if (candidate.wireValue == value) {
        return candidate;
      }
    }
    throw ArgumentError.value(value, 'filter', 'Unsupported color filter.');
  }
}

/// Immutable cross-platform snapshot of all 40+ Live Beauty parameters.
///
/// Matches Android `io.bebebus.livebeauty.VideoEffectsConfig` and iOS
/// `FLTVideoEffects` dictionary keys 1-to-1.
@immutable
class LiveBeautyConfig {
  const LiveBeautyConfig({
    this.enabled = true,
    this.smoothing = 0.0,
    this.brightness = 0.0,
    this.filter = LiveBeautyColorFilter.none,
    this.skinOnly = false,
    this.sharpness = 0.0,
    this.saturation = 1.0,
    this.filterIntensity = 1.0,
    this.compareOriginal = false,
    this.faceSlim = 0.0,
    this.eyeEnlarge = 0.0,
    this.whiten = 0.0,
    this.rosy = 0.0,
    this.waistSlim = 0.0,
    this.bustEnhance = 0.0,
    this.legLengthen = 0.0,
    this.shoulderSlim = 0.0,
    this.neckLengthen = 0.0,
    this.chinShape = 0.0,
    this.foreheadShape = 0.0,
    this.noseSlim = 0.0,
    this.vFace = 0.0,
    this.faceNarrow = 0.0,
    this.faceSmall = 0.0,
    this.cheekboneSlim = 0.0,
    this.eyeDistance = 0.0,
    this.eyeCorner = 0.0,
    this.noseLength = 0.0,
    this.mouthSize = 0.0,
    this.mouthSmile = 0.0,
    this.darkCircles = 0.0,
    this.brightEyes = 0.0,
    this.teethWhiten = 0.0,
    this.nasolabial = 0.0,
    this.backgroundBlur = 0.0,
    this.lutUrl = '',
    this.lutIntensity = 1.0,
    this.skinTexture = 0.0,
    this.radiance = 0.0,
    this.lipstick = 0.0,
    this.lipstickColor = 'rose_bean',
    this.lipstickFinish = 'velvet',
    this.blush = 0.0,
    this.blushStyle = 'apple',
    this.blushColor = 'peach',
    this.eyebrow = 0.0,
    this.eyebrowStyle = 'natural',
    this.eyebrowColor = 'dark_brown',
    this.eyeshadow = 0.0,
    this.eyeshadowColor = 'earth',
    this.eyeliner = 0.0,
    this.aegyoSal = 0.0,
    this.contour = 0.0,
    this.highlight = 0.0,
    this.contactLens = 0.0,
    this.contactLensStyle = 'hazel',
    this.catchlight = 0.0,
  });

  final bool enabled;
  final double smoothing;
  final double brightness;
  final LiveBeautyColorFilter filter;
  final bool skinOnly;
  final double sharpness;
  final double saturation;
  final double filterIntensity;
  final bool compareOriginal;

  // 14-item 478-pt C-infinity diffeomorphic face reshape
  final double faceSlim;
  final double eyeEnlarge;
  final double vFace;
  final double faceNarrow;
  final double faceSmall;
  final double cheekboneSlim;
  final double chinShape;
  final double foreheadShape;
  final double eyeDistance;
  final double eyeCorner;
  final double noseSlim;
  final double noseLength;
  final double mouthSize;
  final double mouthSmile;

  // Skin tone, texture & lighting
  final double whiten;
  final double rosy;
  final double skinTexture;
  final double radiance;

  // Local retouching
  final double darkCircles;
  final double brightEyes;
  final double teethWhiten;
  final double nasolabial;

  // 3D LUT & portrait bokeh
  final String lutUrl;
  final double lutIntensity;
  final double backgroundBlur;

  // Body reshape
  final double waistSlim;
  final double bustEnhance;
  final double legLengthen;
  final double shoulderSlim;
  final double neckLengthen;

  // 6-layer real-time makeup
  final double lipstick;
  final String lipstickColor;
  final String lipstickFinish;
  final double blush;
  final String blushStyle;
  final String blushColor;
  final double eyebrow;
  final String eyebrowStyle;
  final String eyebrowColor;
  final double eyeshadow;
  final String eyeshadowColor;
  final double eyeliner;
  final double aegyoSal;
  final double contour;
  final double highlight;
  final double contactLens;
  final String contactLensStyle;
  final double catchlight;

  /// Official live demo & on-device experience testbed powered by this SDK.
  static const String demoSiteUrl = 'https://dh.video';

  static final RegExp assetLutPattern =
      RegExp(r'^asset://luts/[a-z0-9_]{1,32}\.png$');

  /// Built-in 512x512 (`64^3`) 3D LUT IDs bundled inside the SDK assets.
  static const List<String> builtInLutIds = <String>[
    'natural_studio',
    'clear_white',
    'peach_milk',
    'pure_pink',
    'korean_glow',
    'japanese_airy',
    'soft_film',
    'night_vibe',
    'champagne_glam',
    'sweet_berry',
  ];

  static const Set<String> lipstickColors = <String>{
    'rose_bean',
    'coral_peach',
    'cherry_red',
    'milk_tea',
    'berry_plum',
    'orange_soda',
  };

  static const Set<String> lipstickFinishes = <String>{'velvet', 'glossy'};

  static const Set<String> blushStyles = <String>{
    'apple',
    'drunk',
    'under_eye',
  };

  static const Set<String> blushColors = <String>{
    'peach',
    'rose',
    'apricot',
    'berry',
  };

  static const Set<String> eyebrowStyles = <String>{'natural', 'arched'};

  static const Set<String> eyebrowColors = <String>{
    'dark_brown',
    'soft_brown',
    'grey_black',
  };

  static const Set<String> eyeshadowColors = <String>{
    'earth',
    'peach_pink',
    'rose_wine',
    'champagne',
  };

  static const Set<String> contactLensStyles = <String>{
    'hazel',
    'chocolate',
    'mist_grey',
    'sakura_pink',
  };

  /// Builds an `asset://luts/<id>.png` URI for a bundled 3D LUT.
  static String assetLutUrl(String id) {
    final String url = 'asset://luts/$id.png';
    if (!isValidLutUrl(url)) {
      throw ArgumentError.value(id, 'id', 'Invalid built-in LUT identifier.');
    }
    return url;
  }

  /// Validates an empty string, `https://` LUT URL, or `asset://luts/<id>.png`.
  static bool isValidLutUrl(String url) {
    if (url.isEmpty) {
      return true;
    }
    if (url.startsWith('https://')) {
      if (url.length > 2048) {
        return false;
      }
      for (int i = 0; i < url.length; i++) {
        final int code = url.codeUnitAt(i);
        if (code <= 0x20 || code == 0x7F) {
          return false;
        }
      }
      return true;
    }
    return assetLutPattern.hasMatch(url);
  }

  // ---------------------------------------------------------------------------
  // Studio Presets
  // ---------------------------------------------------------------------------

  /// Zero-effect camera pass-through.
  static const LiveBeautyConfig original = LiveBeautyConfig(enabled: false);

  /// Natural studio skin & subtle micro-reshape (recommended default).
  static const LiveBeautyConfig naturalStudio = LiveBeautyConfig(
    enabled: true,
    smoothing: 0.52,
    brightness: 0.10,
    skinOnly: true,
    skinTexture: 0.45,
    whiten: 0.32,
    rosy: 0.16,
    radiance: 0.25,
    sharpness: 0.20,
    faceSlim: 0.35,
    vFace: 0.25,
    eyeEnlarge: 0.28,
    noseSlim: 0.20,
    darkCircles: 0.35,
    brightEyes: 0.30,
    teethWhiten: 0.25,
    lutUrl: 'asset://luts/natural_studio.png',
    lutIntensity: 0.45,
  );

  /// Cool porcelain skin with glossy rose-bean lip & soft catchlight.
  static const LiveBeautyConfig clearColdWhite = LiveBeautyConfig(
    enabled: true,
    smoothing: 0.68,
    brightness: 0.10,
    skinOnly: true,
    skinTexture: 0.38,
    whiten: 0.58,
    rosy: 0.22,
    radiance: 0.38,
    sharpness: 0.20,
    faceSlim: 0.52,
    vFace: 0.42,
    faceSmall: 0.20,
    eyeEnlarge: 0.42,
    noseSlim: 0.20,
    darkCircles: 0.35,
    brightEyes: 0.30,
    teethWhiten: 0.25,
    lipstick: 0.45,
    lipstickColor: 'rose_bean',
    lipstickFinish: 'glossy',
    blush: 0.30,
    blushStyle: 'apple',
    blushColor: 'peach',
    catchlight: 0.35,
    lutUrl: 'asset://luts/clear_white.png',
    lutIntensity: 0.55,
  );

  /// Korean water-glow peach makeup with aegyo-sal & hazel contact lens.
  static const LiveBeautyConfig koreanWaterGlow = LiveBeautyConfig(
    enabled: true,
    smoothing: 0.68,
    brightness: 0.10,
    skinOnly: true,
    skinTexture: 0.38,
    whiten: 0.58,
    rosy: 0.30,
    radiance: 0.52,
    sharpness: 0.20,
    faceSlim: 0.52,
    vFace: 0.42,
    faceSmall: 0.20,
    eyeEnlarge: 0.42,
    noseSlim: 0.20,
    darkCircles: 0.35,
    brightEyes: 0.30,
    teethWhiten: 0.25,
    lipstick: 0.45,
    lipstickColor: 'rose_bean',
    lipstickFinish: 'glossy',
    blush: 0.30,
    blushStyle: 'apple',
    blushColor: 'peach',
    aegyoSal: 0.40,
    eyeshadow: 0.32,
    eyeshadowColor: 'peach_pink',
    eyeliner: 0.28,
    contactLens: 0.38,
    contactLensStyle: 'hazel',
    catchlight: 0.48,
    lutUrl: 'asset://luts/korean_glow.png',
    lutIntensity: 0.50,
  );

  /// Men's clean studio preset preserving beard hair texture & jawline structure.
  static const LiveBeautyConfig mensClean = LiveBeautyConfig(
    enabled: true,
    smoothing: 0.38,
    brightness: 0.08,
    skinOnly: true,
    skinTexture: 0.65,
    whiten: 0.18,
    sharpness: 0.25,
    faceSlim: 0.20,
    cheekboneSlim: 0.15,
    noseSlim: 0.15,
    darkCircles: 0.42,
    brightEyes: 0.25,
    lutUrl: 'asset://luts/natural_studio.png',
    lutIntensity: 0.35,
  );

  bool get hasFaceReshape =>
      faceSlim > 0 ||
      eyeEnlarge > 0 ||
      vFace > 0 ||
      faceNarrow > 0 ||
      faceSmall > 0 ||
      cheekboneSlim > 0 ||
      chinShape != 0 ||
      foreheadShape != 0 ||
      eyeDistance != 0 ||
      eyeCorner != 0 ||
      noseSlim > 0 ||
      noseLength != 0 ||
      mouthSize != 0 ||
      mouthSmile > 0;

  bool get hasMakeup =>
      lipstick > 0 ||
      blush > 0 ||
      eyebrow > 0 ||
      eyeshadow > 0 ||
      eyeliner > 0 ||
      aegyoSal > 0 ||
      contour > 0 ||
      highlight > 0 ||
      contactLens > 0 ||
      catchlight > 0;

  bool get hasBodyReshape =>
      waistSlim > 0 ||
      bustEnhance > 0 ||
      legLengthen > 0 ||
      shoulderSlim > 0 ||
      neckLengthen > 0;

  /// Throws [RangeError] or [ArgumentError] if any parameter violates native bounds.
  void validate() {
    void unit(String name, double v, [double max = 1.0]) {
      if (!v.isFinite || v < 0.0 || v > max) {
        throw RangeError.value(v, name, 'Must be finite and in [0.0, $max]');
      }
    }

    void signed(String name, double v) {
      if (!v.isFinite || v < -1.0 || v > 1.0) {
        throw RangeError.value(v, name, 'Must be finite and in [-1.0, 1.0]');
      }
    }

    void allow(String name, String v, Set<String> allowed) {
      if (!allowed.contains(v)) {
        throw ArgumentError.value(v, name, 'Must be one of $allowed');
      }
    }

    unit('smoothing', smoothing);
    unit('brightness', brightness);
    unit('sharpness', sharpness);
    unit('saturation', saturation, 2.0);
    unit('filterIntensity', filterIntensity);
    unit('faceSlim', faceSlim);
    unit('eyeEnlarge', eyeEnlarge);
    unit('whiten', whiten);
    unit('rosy', rosy);
    unit('waistSlim', waistSlim);
    unit('bustEnhance', bustEnhance);
    unit('legLengthen', legLengthen);
    unit('shoulderSlim', shoulderSlim);
    unit('neckLengthen', neckLengthen);
    signed('chinShape', chinShape);
    signed('foreheadShape', foreheadShape);
    unit('noseSlim', noseSlim);
    unit('vFace', vFace);
    unit('faceNarrow', faceNarrow);
    unit('faceSmall', faceSmall);
    unit('cheekboneSlim', cheekboneSlim);
    signed('eyeDistance', eyeDistance);
    signed('eyeCorner', eyeCorner);
    signed('noseLength', noseLength);
    signed('mouthSize', mouthSize);
    unit('mouthSmile', mouthSmile);
    unit('darkCircles', darkCircles);
    unit('brightEyes', brightEyes);
    unit('teethWhiten', teethWhiten);
    unit('nasolabial', nasolabial);
    unit('backgroundBlur', backgroundBlur);
    unit('lutIntensity', lutIntensity);
    unit('skinTexture', skinTexture);
    unit('radiance', radiance);
    unit('lipstick', lipstick);
    unit('blush', blush);
    unit('eyebrow', eyebrow);
    unit('eyeshadow', eyeshadow);
    unit('eyeliner', eyeliner);
    unit('aegyoSal', aegyoSal);
    unit('contour', contour);
    unit('highlight', highlight);
    unit('contactLens', contactLens);
    unit('catchlight', catchlight);

    if (!isValidLutUrl(lutUrl)) {
      throw ArgumentError.value(lutUrl, 'lutUrl', 'Invalid LUT URL');
    }
    allow('lipstickColor', lipstickColor, lipstickColors);
    allow('lipstickFinish', lipstickFinish, lipstickFinishes);
    allow('blushStyle', blushStyle, blushStyles);
    allow('blushColor', blushColor, blushColors);
    allow('eyebrowStyle', eyebrowStyle, eyebrowStyles);
    allow('eyebrowColor', eyebrowColor, eyebrowColors);
    allow('eyeshadowColor', eyeshadowColor, eyeshadowColors);
    allow('contactLensStyle', contactLensStyle, contactLensStyles);
  }

  Map<String, Object> toMap() {
    return <String, Object>{
      'enabled': enabled,
      'smoothing': smoothing,
      'brightness': brightness,
      'filter': filter.wireValue,
      'skinOnly': skinOnly,
      'sharpness': sharpness,
      'saturation': saturation,
      'filterIntensity': filterIntensity,
      'compareOriginal': compareOriginal,
      'faceSlim': faceSlim,
      'eyeEnlarge': eyeEnlarge,
      'whiten': whiten,
      'rosy': rosy,
      'waistSlim': waistSlim,
      'bustEnhance': bustEnhance,
      'legLengthen': legLengthen,
      'shoulderSlim': shoulderSlim,
      'neckLengthen': neckLengthen,
      'chinShape': chinShape,
      'foreheadShape': foreheadShape,
      'noseSlim': noseSlim,
      'vFace': vFace,
      'faceNarrow': faceNarrow,
      'faceSmall': faceSmall,
      'cheekboneSlim': cheekboneSlim,
      'eyeDistance': eyeDistance,
      'eyeCorner': eyeCorner,
      'noseLength': noseLength,
      'mouthSize': mouthSize,
      'mouthSmile': mouthSmile,
      'darkCircles': darkCircles,
      'brightEyes': brightEyes,
      'teethWhiten': teethWhiten,
      'nasolabial': nasolabial,
      'backgroundBlur': backgroundBlur,
      'lutUrl': lutUrl,
      'lutIntensity': lutIntensity,
      'skinTexture': skinTexture,
      'radiance': radiance,
      'lipstick': lipstick,
      'lipstickColor': lipstickColor,
      'lipstickFinish': lipstickFinish,
      'blush': blush,
      'blushStyle': blushStyle,
      'blushColor': blushColor,
      'eyebrow': eyebrow,
      'eyebrowStyle': eyebrowStyle,
      'eyebrowColor': eyebrowColor,
      'eyeshadow': eyeshadow,
      'eyeshadowColor': eyeshadowColor,
      'eyeliner': eyeliner,
      'aegyoSal': aegyoSal,
      'contour': contour,
      'highlight': highlight,
      'contactLens': contactLens,
      'contactLensStyle': contactLensStyle,
      'catchlight': catchlight,
    };
  }

  factory LiveBeautyConfig.fromMap(Map<Object?, Object?> map) {
    double numOr(String key, [double? fallback]) {
      if (!map.containsKey(key) && fallback != null) {
        return fallback;
      }
      final Object? v = map[key];
      if (v is num) {
        return v.toDouble();
      }
      throw ArgumentError.value(v, key, 'Expected numeric value');
    }

    bool boolOr(String key, [bool? fallback]) {
      if (!map.containsKey(key) && fallback != null) {
        return fallback;
      }
      final Object? v = map[key];
      if (v is bool) {
        return v;
      }
      throw ArgumentError.value(v, key, 'Expected boolean value');
    }

    String strOr(String key, [String? fallback]) {
      if (!map.containsKey(key) && fallback != null) {
        return fallback;
      }
      final Object? v = map[key];
      if (v is String) {
        return v;
      }
      throw ArgumentError.value(v, key, 'Expected string value');
    }

    final LiveBeautyConfig parsed = LiveBeautyConfig(
      enabled: boolOr('enabled'),
      smoothing: numOr('smoothing'),
      brightness: numOr('brightness'),
      filter: LiveBeautyColorFilter.fromWire(strOr('filter')),
      skinOnly: boolOr('skinOnly', false),
      sharpness: numOr('sharpness', 0.0),
      saturation: numOr('saturation', 1.0),
      filterIntensity: numOr('filterIntensity', 1.0),
      compareOriginal: boolOr('compareOriginal', false),
      faceSlim: numOr('faceSlim', 0.0),
      eyeEnlarge: numOr('eyeEnlarge', 0.0),
      whiten: numOr('whiten', 0.0),
      rosy: numOr('rosy', 0.0),
      waistSlim: numOr('waistSlim', 0.0),
      bustEnhance: numOr('bustEnhance', 0.0),
      legLengthen: numOr('legLengthen', 0.0),
      shoulderSlim: numOr('shoulderSlim', 0.0),
      neckLengthen: numOr('neckLengthen', 0.0),
      chinShape: numOr('chinShape', 0.0),
      foreheadShape: numOr('foreheadShape', 0.0),
      noseSlim: numOr('noseSlim', 0.0),
      vFace: numOr('vFace', 0.0),
      faceNarrow: numOr('faceNarrow', 0.0),
      faceSmall: numOr('faceSmall', 0.0),
      cheekboneSlim: numOr('cheekboneSlim', 0.0),
      eyeDistance: numOr('eyeDistance', 0.0),
      eyeCorner: numOr('eyeCorner', 0.0),
      noseLength: numOr('noseLength', 0.0),
      mouthSize: numOr('mouthSize', 0.0),
      mouthSmile: numOr('mouthSmile', 0.0),
      darkCircles: numOr('darkCircles', 0.0),
      brightEyes: numOr('brightEyes', 0.0),
      teethWhiten: numOr('teethWhiten', 0.0),
      nasolabial: numOr('nasolabial', 0.0),
      backgroundBlur: numOr('backgroundBlur', 0.0),
      lutUrl: strOr('lutUrl', ''),
      lutIntensity: numOr('lutIntensity', 1.0),
      skinTexture: numOr('skinTexture', 0.0),
      radiance: numOr('radiance', 0.0),
      lipstick: numOr('lipstick', 0.0),
      lipstickColor: strOr('lipstickColor', 'rose_bean'),
      lipstickFinish: strOr('lipstickFinish', 'velvet'),
      blush: numOr('blush', 0.0),
      blushStyle: strOr('blushStyle', 'apple'),
      blushColor: strOr('blushColor', 'peach'),
      eyebrow: numOr('eyebrow', 0.0),
      eyebrowStyle: strOr('eyebrowStyle', 'natural'),
      eyebrowColor: strOr('eyebrowColor', 'dark_brown'),
      eyeshadow: numOr('eyeshadow', 0.0),
      eyeshadowColor: strOr('eyeshadowColor', 'earth'),
      eyeliner: numOr('eyeliner', 0.0),
      aegyoSal: numOr('aegyoSal', 0.0),
      contour: numOr('contour', 0.0),
      highlight: numOr('highlight', 0.0),
      contactLens: numOr('contactLens', 0.0),
      contactLensStyle: strOr('contactLensStyle', 'hazel'),
      catchlight: numOr('catchlight', 0.0),
    );
    parsed.validate();
    return parsed;
  }

  LiveBeautyConfig copyWith({
    bool? enabled,
    double? smoothing,
    double? brightness,
    LiveBeautyColorFilter? filter,
    bool? skinOnly,
    double? sharpness,
    double? saturation,
    double? filterIntensity,
    bool? compareOriginal,
    double? faceSlim,
    double? eyeEnlarge,
    double? whiten,
    double? rosy,
    double? waistSlim,
    double? bustEnhance,
    double? legLengthen,
    double? shoulderSlim,
    double? neckLengthen,
    double? chinShape,
    double? foreheadShape,
    double? noseSlim,
    double? vFace,
    double? faceNarrow,
    double? faceSmall,
    double? cheekboneSlim,
    double? eyeDistance,
    double? eyeCorner,
    double? noseLength,
    double? mouthSize,
    double? mouthSmile,
    double? darkCircles,
    double? brightEyes,
    double? teethWhiten,
    double? nasolabial,
    double? backgroundBlur,
    String? lutUrl,
    double? lutIntensity,
    double? skinTexture,
    double? radiance,
    double? lipstick,
    String? lipstickColor,
    String? lipstickFinish,
    double? blush,
    String? blushStyle,
    String? blushColor,
    double? eyebrow,
    String? eyebrowStyle,
    String? eyebrowColor,
    double? eyeshadow,
    String? eyeshadowColor,
    double? eyeliner,
    double? aegyoSal,
    double? contour,
    double? highlight,
    double? contactLens,
    String? contactLensStyle,
    double? catchlight,
  }) {
    return LiveBeautyConfig(
      enabled: enabled ?? this.enabled,
      smoothing: smoothing ?? this.smoothing,
      brightness: brightness ?? this.brightness,
      filter: filter ?? this.filter,
      skinOnly: skinOnly ?? this.skinOnly,
      sharpness: sharpness ?? this.sharpness,
      saturation: saturation ?? this.saturation,
      filterIntensity: filterIntensity ?? this.filterIntensity,
      compareOriginal: compareOriginal ?? this.compareOriginal,
      faceSlim: faceSlim ?? this.faceSlim,
      eyeEnlarge: eyeEnlarge ?? this.eyeEnlarge,
      whiten: whiten ?? this.whiten,
      rosy: rosy ?? this.rosy,
      waistSlim: waistSlim ?? this.waistSlim,
      bustEnhance: bustEnhance ?? this.bustEnhance,
      legLengthen: legLengthen ?? this.legLengthen,
      shoulderSlim: shoulderSlim ?? this.shoulderSlim,
      neckLengthen: neckLengthen ?? this.neckLengthen,
      chinShape: chinShape ?? this.chinShape,
      foreheadShape: foreheadShape ?? this.foreheadShape,
      noseSlim: noseSlim ?? this.noseSlim,
      vFace: vFace ?? this.vFace,
      faceNarrow: faceNarrow ?? this.faceNarrow,
      faceSmall: faceSmall ?? this.faceSmall,
      cheekboneSlim: cheekboneSlim ?? this.cheekboneSlim,
      eyeDistance: eyeDistance ?? this.eyeDistance,
      eyeCorner: eyeCorner ?? this.eyeCorner,
      noseLength: noseLength ?? this.noseLength,
      mouthSize: mouthSize ?? this.mouthSize,
      mouthSmile: mouthSmile ?? this.mouthSmile,
      darkCircles: darkCircles ?? this.darkCircles,
      brightEyes: brightEyes ?? this.brightEyes,
      teethWhiten: teethWhiten ?? this.teethWhiten,
      nasolabial: nasolabial ?? this.nasolabial,
      backgroundBlur: backgroundBlur ?? this.backgroundBlur,
      lutUrl: lutUrl ?? this.lutUrl,
      lutIntensity: lutIntensity ?? this.lutIntensity,
      skinTexture: skinTexture ?? this.skinTexture,
      radiance: radiance ?? this.radiance,
      lipstick: lipstick ?? this.lipstick,
      lipstickColor: lipstickColor ?? this.lipstickColor,
      lipstickFinish: lipstickFinish ?? this.lipstickFinish,
      blush: blush ?? this.blush,
      blushStyle: blushStyle ?? this.blushStyle,
      blushColor: blushColor ?? this.blushColor,
      eyebrow: eyebrow ?? this.eyebrow,
      eyebrowStyle: eyebrowStyle ?? this.eyebrowStyle,
      eyebrowColor: eyebrowColor ?? this.eyebrowColor,
      eyeshadow: eyeshadow ?? this.eyeshadow,
      eyeshadowColor: eyeshadowColor ?? this.eyeshadowColor,
      eyeliner: eyeliner ?? this.eyeliner,
      aegyoSal: aegyoSal ?? this.aegyoSal,
      contour: contour ?? this.contour,
      highlight: highlight ?? this.highlight,
      contactLens: contactLens ?? this.contactLens,
      contactLensStyle: contactLensStyle ?? this.contactLensStyle,
      catchlight: catchlight ?? this.catchlight,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is LiveBeautyConfig && mapEquals(toMap(), other.toMap()));

  @override
  int get hashCode => Object.hashAll(toMap().values);
}
