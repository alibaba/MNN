// sherpa-mnn/csrc/online-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/online-model-config.h"

namespace sherpa_mnn {

class OnlineCtcModel {
 public:
  virtual ~OnlineCtcModel() = default;

  static std::unique_ptr<OnlineCtcModel> Create(
      const OnlineModelConfig &config);

  template <typename Manager>
  static std::unique_ptr<OnlineCtcModel> Create(
      Manager *mgr, const OnlineModelConfig &config);

  // Return a list of tensors containing the initial states
  virtual std::vector<MNN::Express::VARP> GetInitStates() const = 0;

  /** Stack a list of individual states into a batch.
   *
   * It is the inverse operation of `UnStackStates`.
   *
   * @param states states[i] contains the state for the i-th utterance.
   * @return Return a single value representing the batched state.
   */
  virtual std::vector<MNN::Express::VARP> StackStates(
      std::vector<std::vector<MNN::Express::VARP>> states) const = 0;

  /** Unstack a batch state into a list of individual states.
   *
   * It is the inverse operation of `StackStates`.
   *
   * @param states A batched state.
   * @return ans[i] contains the state for the i-th utterance.
   */
  virtual std::vector<std::vector<MNN::Express::VARP>> UnStackStates(
      std::vector<MNN::Express::VARP> states) const = 0;

  /**
   *
   * @param x A 3-D tensor of shape (N, T, C). N has to be 1.
   * @param states  It is from GetInitStates() or returned from this method.
   *
   * @return Return a list of tensors
   *    - ans[0] contains log_probs, of shape (N, T, C)
   *    - ans[1:] contains next_states
   */
  virtual std::vector<MNN::Express::VARP> Forward(
      MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) const = 0;

  /** Return the vocabulary size of the model
   */
  virtual int32_t VocabSize() const = 0;

  /** Return an allocator for allocating memory
   */
  virtual MNNAllocator *Allocator() const = 0;

  // The model accepts this number of frames before subsampling as input
  virtual int32_t ChunkLength() const = 0;

  // Similar to frame_shift in feature extractor, after processing
  // ChunkLength() frames, we advance by ChunkShift() frames
  // before we process the next chunk.
  virtual int32_t ChunkShift() const = 0;

  // Return true if the model supports batch size > 1
  virtual bool SupportBatchProcessing() const { return true; }
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_
