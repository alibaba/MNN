Pod::Spec.new do |s|
  s.name             = "ExyteChat"
  s.version          = "2.1.2"
  s.summary          = "Chat with fully customizable message cells and built-in media picker written with SwiftUI"

  s.homepage         = 'https://github.com/Yogayu/Chat.git'
  s.license          = 'MIT'
  s.author           = { 'Exyte' => 'info@exyte.com' }
  s.source           = { :git => 'https://github.com/Yogayu/Chat.git', :tag => s.version.to_s }
  s.social_media_url = 'http://exyte.com'

  s.ios.deployment_target = '16.0'
  
  s.requires_arc = true
  s.swift_version = "5.7"

  s.source_files = [
     'Sources/*.h',
     'Sources/*.swift',
     'Sources/**/*.swift'
  ]

  s.resources = "Sources/ExyteChat/Resources/**/*"

  s.dependency 'SwiftUIIntrospect'
  s.dependency 'ExyteMediaPicker'
  s.dependency 'FloatingButton'
  s.dependency 'ActivityIndicatorView'
  s.dependency 'ExytePopupView'

end
