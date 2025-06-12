// sherpa-mnn/csrc/fast-clustering.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_
#define SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_

#include <memory>
#include <vector>

#include "sherpa-mnn/csrc/fast-clustering-config.h"

namespace sherpa_mnn {

class FastClustering {
 public:
  explicit FastClustering(const FastClusteringConfig &config);
  ~FastClustering();

  /**
   * @param features Pointer to a 2-D feature matrix in row major. Each row
   *                 is a feature frame. It is changed in-place. We will
   *                 convert each feature frame to a normalized vector.
   *                 That is, the L2-norm of each vector will be equal to 1.
   *                 It uses cosine dissimilarity,
   *                 which is 1 - (cosine similarity)
   * @param num_rows Number of feature frames
   * @param num-cols The feature dimension.
   *
   * @return Return a vector of size num_rows. ans[i] contains the label
   *         for the i-th feature frame, i.e., the i-th row of the feature
   *         matrix.
   */
  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn
#endif  // SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_
