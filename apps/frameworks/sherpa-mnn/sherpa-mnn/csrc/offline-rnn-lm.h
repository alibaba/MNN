// sherpa-mnn/csrc/offline-rnn-lm.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RNN_LM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RNN_LM_H_

#include <memory>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-lm-config.h"
#include "sherpa-mnn/csrc/offline-lm.h"

namespace sherpa_mnn {

class OfflineRnnLM : public OfflineLM {
 public:
  ~OfflineRnnLM() override;

  explicit OfflineRnnLM(const OfflineLMConfig &config);

  template <typename Manager>
  OfflineRnnLM(Manager *mgr, const OfflineLMConfig &config);

  /** Rescore a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param x_lens A 1-D tensor of shape (N,) with data type int64.
   *               It contains number of valid tokens in x before padding.
   * @return Return a 1-D tensor of shape (N,) containing the log likelihood
   *         of each utterance. Its data type is float32.
   *
   * Caution: It returns log likelihood, not negative log likelihood (nll).
   */
  MNN::Express::VARP Rescore(MNN::Express::VARP x, MNN::Express::VARP x_lens) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RNN_LM_H_
