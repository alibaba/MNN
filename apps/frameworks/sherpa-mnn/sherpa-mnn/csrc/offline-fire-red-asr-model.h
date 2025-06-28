// sherpa-mnn/csrc/offline-fire-red-asr-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_H_

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-fire-red-asr-model-meta-data.h"
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

class OfflineFireRedAsrModel {
 public:
  explicit OfflineFireRedAsrModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineFireRedAsrModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineFireRedAsrModel();

  /** Run the encoder model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_len  A tensor of shape (N,) with dtype int64.
   *
   * @return Return a pair containing:
   *  - n_layer_cross_k: A 4-D tensor of shape
   *                     (num_decoder_layers, N, T, d_model)
   *  - n_layer_cross_v: A 4-D tensor of shape
   *                     (num_decoder_layers, N, T, d_model)
   */
  std::pair<MNN::Express::VARP, MNN::Express::VARP> ForwardEncoder(
      MNN::Express::VARP features, MNN::Express::VARP features_length) const;

  /** Run the decoder model.
   *
   * @param tokens A int64 tensor of shape (N, num_words)
   * @param n_layer_self_k_cache  A 5-D tensor of shape
   *                       (num_decoder_layers, N, max_len, num_head, head_dim).
   * @param n_layer_self_v_cache  A 5-D tensor of shape
   *                       (num_decoder_layers, N, max_len, num_head, head_dim).
   * @param n_layer_cross_k       A 5-D tensor of shape
   *                              (num_decoder_layers, N, T, d_model).
   * @param n_layer_cross_v       A 5-D tensor of shape
   *                              (num_decoder_layers, N, T, d_model).
   * @param offset A int64 tensor of shape (N,)
   *
   * @return Return a tuple containing 6 tensors:
   *
   *  - logits A 3-D tensor of shape (N, num_words, vocab_size)
   *  - out_n_layer_self_k_cache Same shape as n_layer_self_k_cache
   *  - out_n_layer_self_v_cache Same shape as n_layer_self_v_cache
   *  - out_n_layer_cross_k Same as n_layer_cross_k
   *  - out_n_layer_cross_v Same as n_layer_cross_v
   *  - out_offset Same as offset
   */
  std::tuple<MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP,
             MNN::Express::VARP>
  ForwardDecoder(MNN::Express::VARP tokens, MNN::Express::VARP n_layer_self_k_cache,
                 MNN::Express::VARP n_layer_self_v_cache, MNN::Express::VARP n_layer_cross_k,
                 MNN::Express::VARP n_layer_cross_v, MNN::Express::VARP offset) const;

  /** Return the initial self kv cache in a pair
   *  - n_layer_self_k_cache A 5-D tensor of shape
   *                       (num_decoder_layers, N, max_len, num_head, head_dim).
   *  - n_layer_self_v_cache A 5-D tensor of shape
   *                       (num_decoder_layers, N, max_len, num_head, head_dim).
   */
  std::pair<MNN::Express::VARP, MNN::Express::VARP> GetInitialSelfKVCache() const;

  const OfflineFireRedAsrModelMetaData& metaData() const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_H_
