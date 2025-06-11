// sherpa-mnn/csrc/offline-moonshine-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

// please see
// https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/moonshine/test.py
class OfflineMoonshineModel {
 public:
  explicit OfflineMoonshineModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineMoonshineModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineMoonshineModel();

  /** Run the preprocessor model.
   *
   * @param audio A float32 tensor of shape (batch_size, num_samples)
   *
   * @return Return a float32 tensor of shape (batch_size, T, dim) that
   *         can be used as the input of ForwardEncoder()
   */
  MNN::Express::VARP ForwardPreprocessor(MNN::Express::VARP audio) const;

  /** Run the encoder model.
   *
   * @param features A float32 tensor of shape (batch_size, T, dim)
   * @param features_len A int32 tensor of shape (batch_size,)
   * @returns A float32 tensor of shape (batch_size, T, dim).
   */
  MNN::Express::VARP ForwardEncoder(MNN::Express::VARP features, MNN::Express::VARP features_len) const;

  /** Run the uncached decoder.
   *
   * @param token A int32 tensor of shape (batch_size, num_tokens)
   * @param seq_len A int32 tensor of shape (batch_size,) containing number
   *                of predicted tokens so far
   * @param encoder_out A float32 tensor of shape (batch_size, T, dim)
   *
   * @returns Return a pair:
   *
   *          - logits, a float32 tensor of shape (batch_size, 1, dim)
   *          - states, a list of states
   */
  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ForwardUnCachedDecoder(
      MNN::Express::VARP token, MNN::Express::VARP seq_len, MNN::Express::VARP encoder_out) const;

  /** Run the cached decoder.
   *
   * @param token A int32 tensor of shape (batch_size, num_tokens)
   * @param seq_len A int32 tensor of shape (batch_size,) containing number
   *                of predicted tokens so far
   * @param encoder_out A float32 tensor of shape (batch_size, T, dim)
   * @param states A list of previous states
   *
   * @returns Return a pair:
   *          - logits, a float32 tensor of shape (batch_size, 1, dim)
   *          - states, a list of new states
   */
  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ForwardCachedDecoder(
      MNN::Express::VARP token, MNN::Express::VARP seq_len, MNN::Express::VARP encoder_out,
      std::vector<MNN::Express::VARP> states) const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_H_
