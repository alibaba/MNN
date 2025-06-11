// sherpa-mnn/csrc/online-wenet-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/online-ctc-model.h"
#include "sherpa-mnn/csrc/online-model-config.h"

namespace sherpa_mnn {

class OnlineWenetCtcModel : public OnlineCtcModel {
 public:
  explicit OnlineWenetCtcModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineWenetCtcModel(Manager *mgr, const OnlineModelConfig &config);

  ~OnlineWenetCtcModel() override;

  // A list of 3 tensors:
  //  - attn_cache
  //  - conv_cache
  //  - offset
  std::vector<MNN::Express::VARP> GetInitStates() const override;

  std::vector<MNN::Express::VARP> StackStates(
      std::vector<std::vector<MNN::Express::VARP>> states) const override;

  std::vector<std::vector<MNN::Express::VARP>> UnStackStates(
      std::vector<MNN::Express::VARP> states) const override;

  /**
   *
   * @param x A 3-D tensor of shape (N, T, C). N has to be 1.
   * @param states  It is from GetInitStates() or returned from this method.
   *
   * @return Return a list of tensors
   *    - ans[0] contains log_probs, of shape (N, T, C)
   *    - ans[1:] contains next_states
   */
  std::vector<MNN::Express::VARP> Forward(
      MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) const override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const override;

  // The model accepts this number of frames before subsampling as input
  int32_t ChunkLength() const override;

  // Similar to frame_shift in feature extractor, after processing
  // ChunkLength() frames, we advance by ChunkShift() frames
  // before we process the next chunk.
  int32_t ChunkShift() const override;

  bool SupportBatchProcessing() const override { return false; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_
