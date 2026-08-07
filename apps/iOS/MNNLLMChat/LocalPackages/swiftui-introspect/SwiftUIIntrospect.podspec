Pod::Spec.new do |spec|
  spec.name = 'SwiftUIIntrospect'
  spec.version = ENV['LIB_VERSION']
  spec.license = { type: 'MIT' }
  spec.homepage = 'https://github.com/siteline/swiftui-introspect'
  spec.author = 'David Roman'
  spec.summary = 'Introspect underlying UIKit/AppKit components from SwiftUI.'
  spec.source = {
    git: 'https://github.com/siteline/swiftui-introspect.git',
    tag: spec.version
  }

  spec.source_files = 'Sources/**/*.swift'

  spec.swift_version = '5.7'
  spec.ios.deployment_target = '13.0'
  spec.tvos.deployment_target = '13.0'
  spec.osx.deployment_target = '10.15'
  spec.visionos.deployment_target = '1.0'
end
