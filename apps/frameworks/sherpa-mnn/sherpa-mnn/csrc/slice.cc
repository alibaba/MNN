// sherpa-mnn/csrc/slice.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/slice.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace sherpa_mnn {

template <typename T /*=float*/>
MNN::Express::VARP Slice(MNNAllocator *allocator, MNN::Express::VARP v,
                 int32_t dim0_start, int32_t dim0_end, int32_t dim1_start,
                 int32_t dim1_end) {
  std::vector<int> shape = v->getInfo()->dim;
  assert(shape.size() == 3);

  assert(0 <= dim0_start);
  assert(dim0_start < dim0_end);
  assert(dim0_end <= shape[0]);

  assert(0 <= dim1_start);
  assert(dim1_start < dim1_end);
  assert(dim1_end <= shape[1]);

  std::array<int, 3> ans_shape{dim0_end - dim0_start, dim1_end - dim1_start,
                                   shape[2]};

  MNN::Express::VARP ans = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans->writeMap<T>();
  for (int32_t i = dim0_start; i != dim0_end; ++i) {
    const T *src = v->readMap<T>() + i * shape[1] * shape[2];
    const T *start = src + dim1_start * shape[2];
    const T *end = src + dim1_end * shape[2];

    std::copy(start, end, dst);
    dst += ans_shape[1] * ans_shape[2];
  }

  return ans;
}

template <typename T /*= float*/>
MNN::Express::VARP Slice(MNNAllocator *allocator, MNN::Express::VARP v,
                 int32_t dim0_start, int32_t dim0_end) {
  std::vector<int> shape = v->getInfo()->dim;
  assert(shape.size() == 2);

  assert(0 <= dim0_start);
  assert(dim0_start < dim0_end);
  assert(dim0_end <= shape[0]);

  const T *src = v->readMap<T>();

  std::array<int, 2> ans_shape{dim0_end - dim0_start, shape[1]};

  MNN::Express::VARP ans = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  const T *start = v->readMap<T>() + dim0_start * shape[1];
  const T *end = v->readMap<T>() + dim0_end * shape[1];
  T *dst = ans->writeMap<T>();
  std::copy(start, end, dst);

  return ans;
}

template MNN::Express::VARP Slice<float>(MNNAllocator *allocator, MNN::Express::VARP v,
                                 int32_t dim0_start, int32_t dim0_end,
                                 int32_t dim1_start, int32_t dim1_end);

template MNN::Express::VARP Slice<float>(MNNAllocator *allocator, MNN::Express::VARP v,
                                 int32_t dim0_start, int32_t dim0_end);

}  // namespace sherpa_mnn
