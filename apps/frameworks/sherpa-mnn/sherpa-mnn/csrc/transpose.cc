// sherpa-mnn/csrc/transpose.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/transpose.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace sherpa_mnn {

template <typename T /*=float*/>
MNN::Express::VARP Transpose01(MNNAllocator *allocator, MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  assert(shape.size() == 3);

  std::array<int, 3> ans_shape{shape[1], shape[0], shape[2]};
  MNN::Express::VARP ans = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());

  T *dst = ans->writeMap<T>();
  auto plane_offset = shape[1] * shape[2];

  for (int i = 0; i != ans_shape[0]; ++i) {
    const T *src = v->readMap<T>() + i * shape[2];
    for (int k = 0; k != ans_shape[1]; ++k) {
      std::copy(src, src + shape[2], dst);
      src += plane_offset;
      dst += shape[2];
    }
  }

  return ans;
}

template <typename T /*= float*/>
MNN::Express::VARP Transpose12(MNNAllocator *allocator, MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  assert(shape.size() == 3);

  std::array<int, 3> ans_shape{shape[0], shape[2], shape[1]};
  MNN::Express::VARP ans = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans->writeMap<T>();
  auto row_stride = shape[2];
  for (int b = 0; b != ans_shape[0]; ++b) {
    const T *src = v->readMap<T>() + b * shape[1] * shape[2];
    for (int i = 0; i != ans_shape[1]; ++i) {
      for (int k = 0; k != ans_shape[2]; ++k, ++dst) {
        *dst = (src + k * row_stride)[i];
      }
    }
  }

  return ans;
}

template MNN::Express::VARP Transpose01<float>(MNNAllocator *allocator,
                                       MNN::Express::VARP v);

template MNN::Express::VARP Transpose12<float>(MNNAllocator *allocator,
                                       MNN::Express::VARP v);

}  // namespace sherpa_mnn
