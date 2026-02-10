// sherpa-mnn/csrc/unbind.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/unbind.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

template <typename T /*= float*/>
std::vector<MNN::Express::VARP> Unbind(MNNAllocator *allocator, MNN::Express::VARP value,
                               int32_t dim) {
  std::vector<int> shape = value->getInfo()->dim;
  assert(dim >= 0);
  assert(dim < static_cast<int32_t>(shape.size()));
  int32_t n = static_cast<int32_t>(shape[dim]);
  if (n == 1) {
    std::vector<MNN::Express::VARP> ans;
    ans.push_back(Clone(allocator, value));
    return ans;
  }

  std::vector<int> ans_shape = shape;
  ans_shape[dim] = 1;  // // Unlike torch, we keep the dim to 1

  // allocator tensors
  std::vector<MNN::Express::VARP> ans;
  ans.reserve(n);
  for (int32_t i = 0; i != n; ++i) {
    MNN::Express::VARP t = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
    ans.push_back(std::move(t));
  }

  auto leading_size = static_cast<int32_t>(std::accumulate(
      shape.begin(), shape.begin() + dim, 1, std::multiplies<int>()));

  auto trailing_size = static_cast<int32_t>(std::accumulate(
      shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int>()));

  const T *src = value->readMap<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (int32_t k = 0; k != n; ++k) {
      T *dst = ans[k]->writeMap<T>() + i * trailing_size;
      std::copy(src, src + trailing_size, dst);
      src += trailing_size;
    }
  }

  return ans;
}

template std::vector<MNN::Express::VARP> Unbind<float>(MNNAllocator *allocator,
                                               MNN::Express::VARP value,
                                               int32_t dim);

template std::vector<MNN::Express::VARP> Unbind<int>(MNNAllocator *allocator,
                                                 MNN::Express::VARP value,
                                                 int32_t dim);

}  // namespace sherpa_mnn
