// sherpa-mnn/csrc/slice-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/slice.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TEST(Slice, Slice3D) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{5, 5, 4};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  auto v1 = Slice(allocator, &v, 2, 4, 0, 2);
  auto v2 = Slice(allocator, &v, 1, 3, 1, 3);

  Print3D(&v);
  Print3D(&v1);
  Print3D(&v2);

  // TODO(fangjun): Check that the results are correct
}

TEST(Slice, Slice2D) {
  MNNAllocator* allocator;
  std::array<int, 2> shape{5, 8};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  std::iota(p, p + shape[0] * shape[1], 0);

  auto v1 = Slice(allocator, &v, 1, 3);
  auto v2 = Slice(allocator, &v, 0, 2);

  Print2D(&v);
  Print2D(&v1);
  Print2D(&v2);

  // TODO(fangjun): Check that the results are correct
}

}  // namespace sherpa_mnn
