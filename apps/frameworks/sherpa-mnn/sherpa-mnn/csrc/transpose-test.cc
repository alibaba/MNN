// sherpa-mnn/csrc/transpose-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/transpose.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TEST(Tranpose, Tranpose01) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{3, 2, 5};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  auto ans = Transpose01(allocator, &v);
  auto v2 = Transpose01(allocator, &ans);

  Print3D(&v);
  Print3D(&ans);
  Print3D(&v2);

  const float *q = v2->readMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    EXPECT_EQ(p[i], q[i]);
  }
}

TEST(Tranpose, Tranpose12) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{3, 2, 5};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  auto ans = Transpose12(allocator, &v);
  auto v2 = Transpose12(allocator, &ans);

  Print3D(&v);
  Print3D(&ans);
  Print3D(&v2);

  const float *q = v2->readMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    EXPECT_EQ(p[i], q[i]);
  }
}

}  // namespace sherpa_mnn
