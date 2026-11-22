# Bebebus Live Beauty SDK 公共 API 保留规则（供集成方开启 R8 fullMode 时直接消费）。
-keep public class io.bebebus.livebeauty.LiveBeautyProcessor { public *; }
-keep public class io.bebebus.livebeauty.LiveBeautyPresets { public *; }
-keep public class io.bebebus.livebeauty.BeautyEngine { public *; }
-keep public class io.bebebus.livebeauty.VideoEffectsConfig { public *; }
-keep public class io.bebebus.livebeauty.VideoEffectsConfig$Companion { public *; }
-keep public class io.bebebus.livebeauty.VideoEffectsFilter { public *; }
-keep public class io.bebebus.livebeauty.BaseFilterRender { public *; }
-keep public class io.bebebus.livebeauty.GlUtil { public *; }

# MediaPipe 0.10.35 JNI 按类名/成员访问 Graph/Packet 及生成的任务类型；
# Graph 静态初始化的 Flogger 依赖调用栈形态，禁止对这些库内联/改名。
# 仅约束美颜引擎及其反射运行时，不关闭宿主 App 的全局 R8 优化。
-keep class com.google.mediapipe.** { *; }
-keep class com.google.common.flogger.** { *; }
-keep class com.google.protobuf.** { *; }
-keepattributes InnerClasses,EnclosingMethod,Signature

# 固定 0.10.35 构件未附带的编译期注解与可选分析/模板 Proto；
# FaceLandmarker / PoseLandmarker / ImageSegmenter 不调用这些 API。
-dontwarn com.google.auto.value.extension.memoized.Memoized
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate
