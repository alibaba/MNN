Pod::Spec.new do |s|
  s.name         = "MNN"
  s.version      = "0.1.1"
  s.summary      = "MNN"

  s.description  = <<-DESC
                    MNN is a lightweight deep neural network inference framework. It loads models and do inference on devices.
                   DESC

  s.homepage     = "https://github.com/alibaba/MNN"
  s.license = {
    :type => 'Apache License, Version 2.0',
    :text => <<-LICENSE
                      Copyright © 2018, Alibaba Group Holding Limited

                      Licensed under the Apache License, Version 2.0 (the "License");
                      you may not use this file except in compliance with the License.
                      You may obtain a copy of the License at

                        http://www.apache.org/licenses/LICENSE-2.0

                      Unless required by applicable law or agreed to in writing, software
                      distributed under the License is distributed on an "AS IS" BASIS,
                      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                      See the License for the specific language governing permissions and
                      limitations under the License.
    LICENSE
  }

  s.author       = { "MNN" => "MNN@alibaba-inc.com" }
  s.platform     = :ios
  s.ios.deployment_target = '8.0'
  s.requires_arc = true

  s.source =  { :git => "git@github.com:alibaba/MNN.git", :branch => 'master' } 
  s.frameworks = 'Metal', 'Accelerate'
  s.library = 'c++'

  s.subspec 'core' do |a|
    a.source_files = \
    'include/*.{h,hpp}',\
    'schema/current/*.{h}',\
    '3rd_party/flatbuffers/include/flatbuffers/*.{h}',\
    'source/core/**/*.{h,c,m,mm,cc,hpp,cpp}',\
    'source/cv/**/*.{h,c,m,mm,cc,hpp,cpp}',\
    'source/math/**/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
    'source/shape/*.{h,c,m,mm,cc,hpp,cpp}',\
    'source/backend/cpu/*.{h,c,m,mm,cc,S,hpp,cpp}',\
    'source/backend/cpu/arm/*.{h,c,m,mm,cc,S,hpp,cpp}',\
    'source/backend/cpu/compute/*.{h,c,m,mm,cc,S,hpp,cpp}',\
    'express/**/*.{hpp,cpp}'
  end
  s.subspec 'armv7' do |a|
    a.source_files = 'source/backend/cpu/arm/arm32/*.{h,c,m,mm,cc,S,hpp,cpp}'
  end
  s.subspec 'aarch64' do |a|
    a.source_files = 'source/backend/cpu/arm/arm64/*.{h,c,m,mm,cc,S,hpp,cpp}'
  end
  s.subspec 'metal' do |a|
    a.source_files = 'source/backend/metal/**/*.{h,c,m,mm,cc,hpp,cpp,metal}'
  end

  s.default_subspecs = 'core', 'armv7', 'aarch64', 'metal'
  s.pod_target_xcconfig = {'METAL_LIBRARY_FILE_BASE' => 'mnn', 'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)" "$(PODS_TARGET_SRCROOT)/3rd_party/flatbuffers/include" ', 'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) MNN_CODEGEN_REGISTER=1 MNN_SUPPORT_TFLITE_QUAN=1'}
  s.user_target_xcconfig = { 'OTHER_LDFLAGS' => '-force_load $(BUILD_DIR)/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/MNN/libMNN.a'}
end
