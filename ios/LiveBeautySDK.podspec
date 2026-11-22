Pod::Spec.new do |s|
  s.name             = 'LiveBeautySDK'
  s.version          = '1.1.2'
  s.summary          = 'Real-time GPU face reshape, guided skin smoothing, makeup, and 3D LUT beauty SDK for iOS.'
  s.description      = <<-DESC
Bebebus Live Beauty SDK provides real-time CoreImage & MediaPipe face tracking, skin-aware variance-guided smoothing, whitening, rosy tone, and facial reshape filters for live streaming and RTC pipelines on iOS. Live demo & experience testbed: https://dh.video
                       DESC
  s.homepage         = 'https://dh.video'
  s.license          = { :type => 'Apache-2.0', :file => '../LICENSE' }
  s.author           = { 'Bebebus Open Source' => 'opensource@bebebus.io' }
  s.source           = { :git => 'https://github.com/bebebus/live-beauty-sdk.git', :tag => "v#{s.version}" }
  s.source_files     = 'Classes/**/*.{h,m}'
  s.public_header_files = 'Classes/**/*.h'
  s.static_framework = true
  s.frameworks       = 'Foundation', 'CoreImage', 'CoreGraphics', 'CoreMedia', 'CoreVideo', 'QuartzCore'
  s.resource_bundles = { 'LiveBeautyModels' => ['Resources/face_landmarker.task'] }
  s.dependency 'MediaPipeTasksVision', '0.10.35'
  s.platform         = :ios, '15.0'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
end
