// sherpa-mnn/csrc/pad-sequence-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/pad-sequence.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TEST(PadSequence, ThreeTensors) {
  MNNAllocator* allocator;

  std::array<int, 2> shape1{3, 5};
  MNN::Express::VARP v1 =
      MNNUtilsCreateTensor<float>(allocator, shape1.data(), shape1.size());
  float *p1 = v1->writeMap<float>();
  std::iota(p1, p1 + shape1[0] * shape1[1], 0);

  std::array<int, 2> shape2{4, 5};
  MNN::Express::VARP v2 =
      MNNUtilsCreateTensor<float>(allocator, shape2.data(), shape2.size());
  float *p2 = v2->writeMap<float>();
  std::iota(p2, p2 + shape2[0] * shape2[1], 0);

  std::array<int, 2> shape3{2, 5};
  MNN::Express::VARP v3 =
      MNNUtilsCreateTensor<float>(allocator, shape3.data(), shape3.size());
  float *p3 = v3->writeMap<float>();
  std::iota(p3, p3 + shape3[0] * shape3[1], 0);

  auto ans = PadSequence(allocator, {&v1, &v2, &v3}, -1);

  Print2D(&v1);
  Print2D(&v2);
  Print2D(&v3);
  Print3D(&ans);
}

}  // namespace sherpa_mnn
