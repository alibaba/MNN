// sherpa-mnn/csrc/speaker-embedding-manager-test.cc
//
// Copyright (c) 2024 Jingzhao Ou (jingzhao.ou@gmail.com)

#include "sherpa-mnn/csrc/speaker-embedding-manager.h"

#include "gtest/gtest.h"

namespace sherpa_mnn {

TEST(SpeakerEmbeddingManager, AddAndRemove) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v = {0.1, 0.1};
  bool status = manager.Add("first", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  // duplicate
  status = manager.Add("first", v.data());
  ASSERT_FALSE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  // non-duplicate
  v = {0.1, 0.9};
  status = manager.Add("second", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 2);

  // do not exist
  status = manager.Remove("third");
  ASSERT_FALSE(status);

  status = manager.Remove("first");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  v = {0.1, 0.1};
  status = manager.Add("first", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 2);

  status = manager.Remove("first");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  status = manager.Remove("second");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 0);
}

TEST(SpeakerEmbeddingManager, Search) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v1 = {0.1, 0.1};
  std::vector<float> v2 = {0.1, 0.9};
  std::vector<float> v3 = {0.9, 0.1};
  bool status = manager.Add("first", v1.data());
  ASSERT_TRUE(status);

  status = manager.Add("second", v2.data());
  ASSERT_TRUE(status);

  status = manager.Add("third", v3.data());
  ASSERT_TRUE(status);

  ASSERT_EQ(manager.NumSpeakers(), 3);

  std::vector<float> v = {15, 16};
  float threshold = 0.9;

  std::string name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "first");

  v = {2, 17};
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "second");

  v = {17, 2};
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "third");

  threshold = 0.9;
  v = {15, 16};
  status = manager.Remove("first");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  v = {17, 2};
  status = manager.Remove("third");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  v = {2, 17};
  status = manager.Remove("second");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  ASSERT_EQ(manager.NumSpeakers(), 0);
}

TEST(SpeakerEmbeddingManager, Verify) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v1 = {0.1, 0.1};
  std::vector<float> v2 = {0.1, 0.9};
  std::vector<float> v3 = {0.9, 0.1};
  bool status = manager.Add("first", v1.data());
  ASSERT_TRUE(status);

  status = manager.Add("second", v2.data());
  ASSERT_TRUE(status);

  status = manager.Add("third", v3.data());
  ASSERT_TRUE(status);

  std::vector<float> v = {15, 16};
  float threshold = 0.9;

  status = manager.Verify("first", v.data(), threshold);
  ASSERT_TRUE(status);

  v = {2, 17};
  status = manager.Verify("first", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("second", v.data(), threshold);
  ASSERT_TRUE(status);

  v = {17, 2};
  status = manager.Verify("first", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("second", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("third", v.data(), threshold);
  ASSERT_TRUE(status);

  status = manager.Verify("fourth", v.data(), threshold);
  ASSERT_FALSE(status);
}

}  // namespace sherpa_mnn
