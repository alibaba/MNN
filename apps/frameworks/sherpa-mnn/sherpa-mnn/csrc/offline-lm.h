// sherpa-mnn/csrc/offline-lm.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_LM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_LM_H_

#include <memory>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/hypothesis.h"
#include "sherpa-mnn/csrc/offline-lm-config.h"

namespace sherpa_mnn {

class OfflineLM {
 public:
  virtual ~OfflineLM() = default;

  static std::unique_ptr<OfflineLM> Create(const OfflineLMConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineLM> Create(Manager *mgr,
                                           const OfflineLMConfig &config);

  /** Rescore a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param x_lens A 1-D tensor of shape (N,) with data type int64.
   *               It contains number of valid tokens in x before padding.
   * @return Return a 1-D tensor of shape (N,) containing the negative log
   *         likelihood of each utterance. Its data type is float32.
   *
   * Caution: It returns negative log likelihood (nll), not log likelihood
   */
  virtual MNN::Express::VARP Rescore(MNN::Express::VARP x, MNN::Express::VARP x_lens) = 0;

  // This function updates hyp.lm_lob_prob of hyps.
  //
  // @param scale LM score
  // @param context_size Context size of the transducer decoder model
  // @param hyps It is changed in-place.
  void ComputeLMScore(float scale, int32_t context_size,
                      std::vector<Hypotheses> *hyps);
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_LM_H_
