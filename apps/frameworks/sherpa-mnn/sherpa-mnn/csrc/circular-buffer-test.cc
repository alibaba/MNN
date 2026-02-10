// sherpa-mnn/csrc/circular-buffer-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/circular-buffer.h"

#include <vector>

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

TEST(CircularBuffer, Push) {
  CircularBuffer buffer(10);
  EXPECT_EQ(buffer.Size(), 0);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 0);

  std::vector<float> a = {0, 1, 2, 3, 4, 5};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 6);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 6);

  auto c = buffer.Get(0, a.size());
  EXPECT_EQ(a.size(), c.size());
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a[i], c[i]);
  }

  std::vector<float> d = {-6, -7, -8, -9};
  buffer.Push(d.data(), d.size());

  c = buffer.Get(a.size(), d.size());
  EXPECT_EQ(d.size(), c.size());
  for (int32_t i = 0; i != d.size(); ++i) {
    EXPECT_EQ(d[i], c[i]);
  }
}

TEST(CircularBuffer, PushAndPop) {
  CircularBuffer buffer(5);
  std::vector<float> a = {0, 1, 2, 3};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 4);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 4);

  buffer.Pop(2);

  EXPECT_EQ(buffer.Size(), 2);
  EXPECT_EQ(buffer.Head(), 2);
  EXPECT_EQ(buffer.Tail(), 4);

  auto c = buffer.Get(2, 2);
  EXPECT_EQ(c.size(), 2);
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 3);

  a = {10, 20, 30};
  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 2);
  EXPECT_EQ(buffer.Tail(), 7);

  c = buffer.Get(2, 5);
  EXPECT_EQ(c.size(), 5);
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 3);
  EXPECT_EQ(c[2], 10);
  EXPECT_EQ(c[3], 20);
  EXPECT_EQ(c[4], 30);

  c = buffer.Get(3, 4);
  EXPECT_EQ(c.size(), 4);
  EXPECT_EQ(c[0], 3);
  EXPECT_EQ(c[1], 10);
  EXPECT_EQ(c[2], 20);
  EXPECT_EQ(c[3], 30);

  c = buffer.Get(4, 3);
  EXPECT_EQ(c.size(), 3);
  EXPECT_EQ(c[0], 10);
  EXPECT_EQ(c[1], 20);
  EXPECT_EQ(c[2], 30);

  buffer.Pop(4);
  EXPECT_EQ(buffer.Size(), 1);
  EXPECT_EQ(buffer.Head(), 6);
  EXPECT_EQ(buffer.Tail(), 7);

  c = buffer.Get(6, 1);
  EXPECT_EQ(c.size(), 1);
  EXPECT_EQ(c[0], 30);

  a = {100, 200, 300, 400};
  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);

  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 6);
  EXPECT_EQ(buffer.Tail(), 11);

  c = buffer.Get(6, 5);
  EXPECT_EQ(c.size(), 5);
  EXPECT_EQ(c[0], 30);
  EXPECT_EQ(c[1], 100);
  EXPECT_EQ(c[2], 200);
  EXPECT_EQ(c[3], 300);
  EXPECT_EQ(c[4], 400);

  buffer.Pop(3);
  EXPECT_EQ(buffer.Size(), 2);
  EXPECT_EQ(buffer.Head(), 9);
  EXPECT_EQ(buffer.Tail(), 11);

  c = buffer.Get(10, 1);
  EXPECT_EQ(c.size(), 1);
  EXPECT_EQ(c[0], 400);

  a = {1000, 2000, 3000};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 9);
  EXPECT_EQ(buffer.Tail(), 14);

  buffer.Pop(1);

  EXPECT_EQ(buffer.Size(), 4);
  EXPECT_EQ(buffer.Head(), 10);
  EXPECT_EQ(buffer.Tail(), 14);

  a = {4000};

  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 10);
  EXPECT_EQ(buffer.Tail(), 15);

  c = buffer.Get(13, 2);
  EXPECT_EQ(c.size(), 2);
  EXPECT_EQ(c[0], 3000);
  EXPECT_EQ(c[1], 4000);
}

}  // namespace sherpa_mnn
