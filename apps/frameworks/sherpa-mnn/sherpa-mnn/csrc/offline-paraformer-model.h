// sherpa-mnn/csrc/offline-paraformer-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_H_

#include <memory>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

class OfflineParaformerModel {
 public:
  explicit OfflineParaformerModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParaformerModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineParaformerModel();

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int32_t.
   *
   * @return Return a vector containing:
   *  - log_probs: A 3-D tensor of shape (N, T', vocab_size)
   *  - token_num: A 1-D tensor of shape (N, T') containing number
   *               of valid tokens in each utterance. Its dtype is int.
   *  If it is a model supporting timestamps, then there are additional two
   *  outputs:
   *   - us_alphas
   *   - us_cif_peak
   */
  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                  MNN::Express::VARP features_length);

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const;

  /** It is lfr_m in config.yaml
   */
  int32_t LfrWindowSize() const;

  /** It is lfr_n in config.yaml
   */
  int32_t LfrWindowShift() const;

  /** Return negative mean for CMVN
   */
  const std::vector<float> &NegativeMean() const;

  /** Return inverse stddev for CMVN
   */
  const std::vector<float> &InverseStdDev() const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_H_
