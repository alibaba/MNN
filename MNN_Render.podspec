Pod::Spec.new do |s|
  s.name         = "MNN"
  s.version      = "2.2.0"
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

  #s.source =  { :git => "git@github.com:alibaba/MNN.git", :branch => 'master' }
  s.source = {:git => "/Users/zhang/Development/AliNNPrivate/",:branch=> 'head'}
  s.frameworks = 'Metal', 'Accelerate', 'CoreML'
  s.library = 'c++'
  s.source_files = \
  'include/MNN/*.{h,hpp}',\
  'include/MNN/expr/*.{h,hpp}',\
  'schema/current/*.{h}',\
  '3rd_party/flatbuffers/include/flatbuffers/*.{h}',\
  'source/internal/logging/*.{hpp,cpp}',\
  'source/internal/logging/ios/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/internal/logging/aliyun-log-c-sdk/src/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/core/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/common/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/utils/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/geometry/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/cv/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/math/**/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/shape/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/shape/render/*.{h,c,m,mm,cc,hpp,cpp}',\
  #'source/backend/arm82/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  #'source/backend/arm82/asm/**/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/render/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/bf16/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/arm/**/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/compute/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/metal/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/backend/metal/render/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/backend/coreml/backend/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/backend/coreml/execution/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/backend/coreml/mlmodel/src/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'express/**/*.{hpp,cpp}',\
  'tools/cv/include/**/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'tools/cv/source/imgproc/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'tools/cv/source/calib3d/*.{h,c,m,mm,cc,hpp,cpp,metal}'

  s.header_mappings_dir = 'include'
  s.subspec 'cv' do |sp|
    sp.source_files = 'tools/cv/include/**/*.hpp'
    sp.header_mappings_dir = 'tools/cv/include'
    sp.xcconfig = { 'ALWAYS_SEARCH_USER_PATHS' => 'NO' }
  end

  s.compiler_flags = '-arch arm64 -march=armv8.2-a+simd+fp16'
  s.pod_target_xcconfig = {'METAL_LIBRARY_FILE_BASE' => 'mnn', 'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/include" "$(PODS_TARGET_SRCROOT)/3rd_party/flatbuffers/include" "$(PODS_TARGET_SRCROOT)/source" "$(PODS_TARGET_SRCROOT)/3rd_party/half" "$(PODS_TARGET_SRCROOT)/source/backend/coreml/mlmodel/include" "$(PODS_TARGET_SRCROOT)/tools/cv/include"', 'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) MNN_CODEGEN_REGISTER=1 MNN_SUPPORT_TFLITE_QUAN=1 MNN_METAL_ENABLED=1 MNN_METAL_FULL_PRECISION=1 MNN_SUPPORT_RENDER=1 MNN_SUPPORT_BF16=1 MNN_COREML_ENABLED=1 USE_LZ4_FLAG=1 MNN_INTERNAL_ENABLED=1 MNN_USE_SPARSE_COMPUTE=1'}
  s.user_target_xcconfig = { 'OTHER_LDFLAGS' => '-force_load $(BUILD_DIR)/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/MNN/libMNN.a', 'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/include"' }
end
