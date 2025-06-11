// sherpa-mnn/csrc/online-transducer-nemo-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/online-model-config.h"

namespace sherpa_mnn {

// see
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L40
// Its decoder is stateful, not stateless.
class OnlineTransducerNeMoModel {
 public:
  explicit OnlineTransducerNeMoModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineTransducerNeMoModel(Manager *mgr, const OnlineModelConfig &config);

  ~OnlineTransducerNeMoModel();
  // A list of 3 tensors:
  //  - cache_last_channel
  //  - cache_last_time
  //  - cache_last_channel_len
  std::vector<MNN::Express::VARP> GetEncoderInitStates() const;

  // stack encoder states
  std::vector<MNN::Express::VARP> StackStates(
      std::vector<std::vector<MNN::Express::VARP>> states) const;

  // unstack encoder states
  std::vector<std::vector<MNN::Express::VARP>> UnStackStates(
      std::vector<MNN::Express::VARP> states) const;

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param states  It is from GetEncoderInitStates() or returned from this
   *                method.
   *
   * @return Return a tuple containing:
   *           - ans[0]: encoder_out, a tensor of shape (N, encoder_out_dim, T')
   *           - ans[1:]: contains next states
   */
  std::vector<MNN::Express::VARP> RunEncoder(
      MNN::Express::VARP features, std::vector<MNN::Express::VARP> states) const;  // NOLINT

  /** Run the decoder network.
   *
   * @param targets A int32 tensor of shape (batch_size, 1)
   * @param states The states for the decoder model.
   * @return Return a vector:
   *           - ans[0] is the decoder_out (a float tensor)
   *           - ans[1:] is the next states
   */
  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> RunDecoder(
      MNN::Express::VARP targets, std::vector<MNN::Express::VARP> states) const;

  std::vector<MNN::Express::VARP> GetDecoderInitStates() const;

  /** Run the joint network.
   *
   * @param encoder_out Output of the encoder network.
   * @param decoder_out Output of the decoder network.
   * @return Return a tensor of shape (N, 1, 1, vocab_size) containing logits.
   */
  MNN::Express::VARP RunJoiner(MNN::Express::VARP encoder_out, MNN::Express::VARP decoder_out) const;

  /** We send this number of feature frames to the encoder at a time. */
  int32_t ChunkSize() const;

  /** Number of input frames to discard after each call to RunEncoder.
   *
   * For instance, if we have 30 frames, chunk_size=8, chunk_shift=6.
   *
   * In the first call of RunEncoder, we use frames 0~7 since chunk_size is 8.
   * Then we discard frame 0~5 since chunk_shift is 6.
   * In the second call of RunEncoder, we use frames 6~13; and then we discard
   * frames 6~11.
   * In the third call of RunEncoder, we use frames 12~19; and then we discard
   * frames 12~16.
   *
   * Note: ChunkSize() - ChunkShift() == right context size
   */
  int32_t ChunkShift() const;

  /** Return the subsampling factor of the model.
   */
  int32_t SubsamplingFactor() const;

  int32_t VocabSize() const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string FeatureNormalizationMethod() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_
