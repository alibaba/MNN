// sherpa-mnn/csrc/online-lm.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_LM_H_
#define SHERPA_ONNX_CSRC_ONLINE_LM_H_

#include <memory>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/hypothesis.h"
#include "sherpa-mnn/csrc/online-lm-config.h"

namespace sherpa_mnn {

class OnlineLM {
 public:
  virtual ~OnlineLM() = default;

  static std::unique_ptr<OnlineLM> Create(const OnlineLMConfig &config);

  // init states for classic rescore
  virtual std::vector<MNN::Express::VARP> GetInitStates() = 0;

  // init states for shallow fusion
  virtual std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> GetInitStatesSF() = 0;

   /** ScoreToken a batch of sentences (shallow fusion).
   *
   * @param x A 2-D tensor of shape (N, 1) with data type int64.
   * @param states It contains the states for the LM model
   * @return Return a pair containing
   *          - log_prob of NN LM
   *          - updated states
   *
   */
  virtual std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ScoreToken(
      MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) = 0;

  /** This function updates hyp.lm_log_prob of hyps (classic rescore).
   *
   * @param scale LM score
   * @param context_size Context size of the transducer decoder model
   * @param hyps It is changed in-place.
   *
   */
  virtual void ComputeLMScore(float scale, int32_t context_size,
                      std::vector<Hypotheses> *hyps) = 0;

  /** This function updates lm_log_prob and nn_lm_scores of hyp (shallow fusion).
   *
   * @param scale LM score
   * @param hyps It is changed in-place.
   *
   */
  virtual void ComputeLMScoreSF(float scale, Hypothesis *hyp) = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_H_
