Pod::Spec.new do |s|
  s.name             = 'live_beauty_sdk'
  s.version          = '2.0.0'
  s.summary          = 'Real-time GPU live beauty SDK for Flutter (iOS CoreImage + MediaPipe 478-pt).'
  s.description      = <<-DESC
    Flutter wrapper for Bebebus Live Beauty SDK on iOS (CoreImage + MediaPipe 478-pt face geometry). Live demo & experience testbed: https://dh.video
  DESC
  s.homepage         = 'https://dh.video'
  s.license          = { :type => 'Apache-2.0', :file => '../../LICENSE' }
  s.author           = { 'Bebebus Open Source' => 'opensource@bebebus.io' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*.{h,m}', '../../ios/Classes/**/*.{h,m}'
  s.public_header_files = 'Classes/**/*.h', '../../ios/Classes/**/*.h'
  s.resource_bundles = {
    'LiveBeautyModels' => ['../../ios/Resources/face_landmarker.task']
  }
  s.dependency 'Flutter'
  s.dependency 'MediaPipeTasksVision', '0.10.21'
  s.platform = :ios, '15.0'
  s.frameworks = 'CoreImage', 'CoreMedia', 'CoreVideo', 'QuartzCore', 'Accelerate'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
end
