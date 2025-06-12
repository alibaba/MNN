// sherpa-mnn/csrc/packed-sequence-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/packed-sequence.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TEST(PackedSequence, Case1) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{5, 5, 4};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  MNN::Express::VARP length =
      MNNUtilsCreateTensor<int>(allocator, shape.data(), 1);
  int *p_length = length->writeMap<int>();
  p_length[0] = 1;
  p_length[1] = 2;
  p_length[2] = 3;
  p_length[3] = 5;
  p_length[4] = 2;

  auto packed_seq = PackPaddedSequence(allocator, &v, &length);
  fprintf(stderr, "sorted indexes: ");
  for (auto i : packed_seq.sorted_indexes) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");
  // output index:   0 1 2 3 4
  // sorted indexes: 3 2 1 4 0
  // length:         5 3 2 2 1
  Print3D(&v);
  Print2D(&packed_seq.data);
  fprintf(stderr, "batch sizes per time step: ");
  for (auto i : packed_seq.batch_sizes) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");

  // TODO(fangjun): Check that the return value is correct
}

}  // namespace sherpa_mnn
