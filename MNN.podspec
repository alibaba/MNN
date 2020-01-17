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

  #s.source =  { :git => "git@github.com:alibaba/MNN.git", :branch => 'master' }
  s.prepare_command = <<-CMD
                          schema/generate.sh
                          python source/backend/metal/MetalCodeGen.py source/backend/metal/ source/backend/metal/MetalOPRegister.mm
                      CMD
  s.source = {:git => "/Users/zhang/Development/AliNNPrivate/",:branch=> 'head'}
  s.frameworks = 'Metal', 'Accelerate'
  s.library = 'c++'
  s.source_files = \
  'include/MNN/*.{h,hpp}',\
  'include/MNN/expr/*.{h,hpp}',\
  'schema/current/*.{h}',\
  '3rd_party/flatbuffers/include/flatbuffers/*.{h}',\
  'source/core/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/cv/**/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/math/**/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'source/shape/*.{h,c,m,mm,cc,hpp,cpp}',\
  'source/backend/cpu/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/arm/**/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/cpu/compute/*.{h,c,m,mm,cc,S,hpp,cpp}',\
  'source/backend/metal/*.{h,c,m,mm,cc,hpp,cpp,metal}',\
  'express/**/*.{hpp,cpp}'
  s.header_mappings_dir = 'include'

  s.pod_target_xcconfig = {'METAL_LIBRARY_FILE_BASE' => 'mnn', 'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/include" "$(PODS_TARGET_SRCROOT)/3rd_party/flatbuffers/include" "$(PODS_TARGET_SRCROOT)/source" "$(PODS_TARGET_SRCROOT)/3rd_party/half"', 'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) MNN_CODEGEN_REGISTER=1 MNN_SUPPORT_TFLITE_QUAN=1'}
  s.user_target_xcconfig = { 'OTHER_LDFLAGS' => '-force_load $(BUILD_DIR)/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/MNN/libMNN.a', 'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/include"' }
end
