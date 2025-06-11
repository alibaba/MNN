// sherpa-mnn/csrc/packed-sequence.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/packed-sequence.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>

#include "sherpa-mnn/csrc/slice.h"
#include "sherpa-mnn/csrc/transpose.h"

namespace sherpa_mnn {

static MNN::Express::VARP IndexSelect(MNNAllocator *allocator, MNN::Express::VARP value,
                              const std::vector<int32_t> &sorted_indexes) {
  auto shape = value->getInfo()->dim;
  assert(shape.size() == 3);
  std::array<int, 3> ans_shape{static_cast<int>(sorted_indexes.size()),
                                   shape[1], shape[2]};

  MNN::Express::VARP ans = MNNUtilsCreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());
  float *dst = ans->writeMap<float>();
  const float *src = value->readMap<float>();

  for (auto i : sorted_indexes) {
    const float *start = src + i * shape[1] * shape[2];
    std::copy(start, start + shape[1] * shape[2], dst);
    dst += shape[1] * shape[2];
  }
  return ans;
}

PackedSequence PackPaddedSequence(MNNAllocator *allocator,
                                  MNN::Express::VARP value, MNN::Express::VARP length) {
  std::vector<int> v_shape = value->getInfo()->dim;
  std::vector<int> l_shape = length->getInfo()->dim;

  assert(v_shape.size() == 3);
  assert(l_shape.size() == 1);
  assert(v_shape[0] == l_shape[0]);

  std::vector<int32_t> indexes(v_shape[0]);
  std::iota(indexes.begin(), indexes.end(), 0);

  const int *p_length = length->readMap<int>();
  // sort in descending order
  std::sort(indexes.begin(), indexes.end(), [p_length](int32_t i, int32_t j) {
    return p_length[i] > p_length[j];
  });

  int32_t n = static_cast<int32_t>(v_shape[0]);

  int max_T = p_length[indexes[0]];

  auto sum_T = std::accumulate(p_length, p_length + n, static_cast<int>(0));

  std::array<int, 2> data_shape{sum_T, v_shape[2]};

  MNN::Express::VARP data = MNNUtilsCreateTensor<float>(
      allocator, data_shape.data(), data_shape.size());
  float *dst = data->writeMap<float>();

  MNN::Express::VARP tensor = IndexSelect(allocator, value, indexes);
  tensor = Transpose01(allocator, tensor);

  // batch size at each time step
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(max_T);

  int prev_l = 0;
  for (int32_t i = 0; i != n; ++i) {
    auto cur_l = p_length[indexes[n - 1 - i]];
    assert(cur_l >= prev_l);
    if (cur_l == prev_l) {
      continue;
    }

    auto cur_batch_size = n - i;

    MNN::Express::VARP cur_batch =
        Slice(allocator, tensor, prev_l, cur_l, 0, cur_batch_size);
    auto count = cur_batch->getInfo()->size;
    const float *src = cur_batch->readMap<float>();
    std::copy(src, src + count, dst);
    dst += count;

    for (int32_t j = prev_l; j < cur_l; ++j) {
      batch_sizes.push_back(cur_batch_size);
    }

    prev_l = cur_l;
  }

  PackedSequence packed_seq;
  packed_seq.sorted_indexes = std::move(indexes);
  packed_seq.data = std::move(data);
  packed_seq.batch_sizes = std::move(batch_sizes);

  return packed_seq;
}

}  // namespace sherpa_mnn
