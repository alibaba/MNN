Pod::Spec.new do |s|
  s.name         = "MNN"
  s.version      = "0.2.1.7"
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
  s.frameworks = 'Metal', 'Accelerate'
  s.library = 'c++'
  s.source = {:http=>"https://github.com/alibaba/MNN/releases/download/#{s.version}/MNN-iOS-#{s.version}.zip"}
  s.vendored_frameworks = "MNN.framework"
end
