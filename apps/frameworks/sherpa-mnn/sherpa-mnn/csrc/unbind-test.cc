// sherpa-mnn/csrc/unbind-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/unbind.h"

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/cat.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TEST(Ubind, Test1DTensors) {
  MNNAllocator* allocator;
  std::array<int, 1> shape{3};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);
  EXPECT_EQ(ans.size(), shape[0]);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    EXPECT_EQ(ans[i]->readMap<float>()[0], p[i]);
  }
  Print1D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print1D(&ans[i]);
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 0);
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test2DTensorsDim0) {
  MNNAllocator* allocator;
  std::array<int, 2> shape{3, 2};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);

  Print2D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print2D(&ans[i]);
  }

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    const float *pans = ans[i]->readMap<float>();
    for (int32_t k = 0; k != static_cast<int32_t>(shape[1]); ++k, ++p) {
      EXPECT_EQ(*p, pans[k]);
    }
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 0);
  Print2D(&v2);

  p = v->writeMap<float>();
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0] * shape[1]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test2DTensorsDim1) {
  MNNAllocator* allocator;
  std::array<int, 2> shape{3, 2};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 1);

  Print2D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[1]); ++i) {
    Print2D(&ans[i]);
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 1);
  Print2D(&v2);

  p = v->writeMap<float>();
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0] * shape[1]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim0) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{3, 2, 5};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print3D(&ans[i]);
  }

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    const float *pans = ans[i]->readMap<float>();
    for (int32_t k = 0; k != static_cast<int32_t>(shape[1] * shape[2]);
         ++k, ++p) {
      EXPECT_EQ(*p, pans[k]);
    }
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 0);
  Print3D(&v2);

  p = v->writeMap<float>();
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim1) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{3, 2, 5};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 1);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[1]); ++i) {
    Print3D(&ans[i]);
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 1);
  Print3D(&v2);

  p = v->writeMap<float>();
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim2) {
  MNNAllocator* allocator;
  std::array<int, 3> shape{3, 2, 5};
  MNN::Express::VARP v =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v->writeMap<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 2);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[2]); ++i) {
    Print3D(&ans[i]);
  }

  // For Cat
  std::vector<MNN::Express::VARP > vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  MNN::Express::VARP v2 = Cat(allocator, vec, 2);
  Print3D(&v2);

  p = v->writeMap<float>();
  const float *p2 = v2->readMap<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

}  // namespace sherpa_mnn
