Pod::Spec.new do |s|
  s.name             = "ExyteMediaPicker"
  s.version          = "2.2.3"
  s.summary          = "MediaPicker is a customizable photo/video picker for iOS written in pure SwiftUI"

  s.homepage         = 'https://github.com/exyte/MediaPicker.git'
  s.license          = 'MIT'
  s.author           = { 'Exyte' => 'info@exyte.com' }
  s.source           = { :git => 'https://github.com/exyte/MediaPicker.git', :tag => s.version.to_s }
  s.social_media_url = 'http://exyte.com'

  s.ios.deployment_target = '15.0'
  
  s.requires_arc = true
  s.swift_version = "5.2"

  s.source_files = [
     'Sources/*.h',
     'Sources/*.swift',
     'Sources/**/*.swift'
  ]

end
