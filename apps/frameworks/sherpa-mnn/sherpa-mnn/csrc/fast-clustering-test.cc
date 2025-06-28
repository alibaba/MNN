// sherpa-mnn/csrc/fast-clustering-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/fast-clustering.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_mnn {

TEST(FastClustering, TestTwoClusters) {
  std::vector<float> features = {
      // point 0
      0.1,
      0.1,
      // point 2
      0.4,
      -0.5,
      // point 3
      0.6,
      -0.7,
      // point 1
      0.2,
      0.3,
  };

  FastClusteringConfig config;
  config.num_clusters = 2;

  FastClustering clustering(config);
  auto labels = clustering.Cluster(features.data(), 4, 2);
  int32_t k = 0;
  for (auto i : labels) {
    std::cout << "point " << k << ": label " << i << "\n";
    ++k;
  }
}

TEST(FastClustering, TestClusteringWithThreshold) {
  std::vector<float> features = {
      // point 0
      0.1,
      0.1,
      // point 2
      0.4,
      -0.5,
      // point 3
      0.6,
      -0.7,
      // point 1
      0.2,
      0.3,
  };

  FastClusteringConfig config;
  config.threshold = 0.5;

  FastClustering clustering(config);
  auto labels = clustering.Cluster(features.data(), 4, 2);
  int32_t k = 0;
  for (auto i : labels) {
    std::cout << "point " << k << ": label " << i << "\n";
    ++k;
  }
}

}  // namespace sherpa_mnn
