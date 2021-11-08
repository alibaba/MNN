/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef OptimizedComputer_hpp
#define OptimizedComputer_hpp

#include <cmath>
#include <vector>
#include "backend/cpu/CPUFixedPoint.hpp"

namespace MNN {
namespace Optimized {
void AveragePool(const uint8_t* input_data, const std::vector<int>& input_dims, int stride_width, int stride_height,
                 int pad_width, int pad_height, int filter_width, int filter_height, int mOutputActivationMin,
                 int mOutputActivationMax, uint8_t* output_data, const std::vector<int>& output_dims);

void Logistic(const uint8_t* input_data, const std::vector<int>& input_dims, int32_t inputZeroPoint,
              int32_t input_range_radius, int32_t input_multiplier, int input_left_shift, uint8_t* output_data,
              const std::vector<int>& output_dims);
} // namespace Optimized
} // namespace MNN
#endif /* OptimizedComputer_hpp */
