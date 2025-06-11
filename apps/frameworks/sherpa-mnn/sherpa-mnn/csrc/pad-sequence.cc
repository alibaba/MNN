// sherpa-mnn/csrc/pad-sequence.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/pad-sequence.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace sherpa_mnn {

MNN::Express::VARP PadSequence(MNNAllocator *allocator,
                       const std::vector<MNN::Express::VARP > &values,
                       float padding_value) {
  int32_t batch_size = static_cast<int32_t>(values.size());

  std::vector<int> shape0 =
      values[0]->getInfo()->dim;
  assert(shape0.size() == 2);

  auto feature_dim = shape0[1];
  auto max_T = shape0[0];

  for (int32_t i = 1; i != batch_size; ++i) {
    auto shape = values[i]->getInfo()->dim;

    assert(shape.size() == 2);
    assert(shape[1] == feature_dim);

    max_T = std::max(max_T, shape[0]);
  }
  std::array<int, 3> ans_shape{batch_size, max_T, feature_dim};

  MNN::Express::VARP ans = MNNUtilsCreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());
  float *dst = ans->writeMap<float>();
  std::fill(dst, dst + batch_size * max_T * feature_dim, padding_value);

  for (const auto v : values) {
    const float *src = v->readMap<float>();
    auto shape = v->getInfo()->dim;
    std::copy(src, src + shape[0] * shape[1], dst);
    dst += max_T * feature_dim;
  }

  return ans;

  // TODO(fangjun): Check that the returned value is correct.
}

}  // namespace sherpa_mnn
