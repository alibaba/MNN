// sherpa-mnn/csrc/offline-whisper-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_H_

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-model-config.h"
#include "sherpa-mnn/csrc/spoken-language-identification.h"

namespace sherpa_mnn {

class OfflineWhisperModel {
 public:
  explicit OfflineWhisperModel(const OfflineModelConfig &config);

  explicit OfflineWhisperModel(
      const SpokenLanguageIdentificationConfig &config);

  template <typename Manager>
  OfflineWhisperModel(Manager *mgr, const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWhisperModel(Manager *mgr,
                      const SpokenLanguageIdentificationConfig &config);

  ~OfflineWhisperModel();

  /** Run the encoder model.
   *
   * @param features  A tensor of shape (N, C, T). It is changed in-place.
   *                  C is 80 and T is 3000.
   *
   * @return Return a pair containing:
   *  - n_layer_cross_k: A 4-D tensor of shape
   *                     (n_text_layer, N, n_audio_ctx, n_text_state)
   *  - n_layer_cross_v: A 4-D tensor of shape
   *                     (n_text_layer, N, n_audio_ctx, n_text_state)
   */
  std::pair<MNN::Express::VARP, MNN::Express::VARP> ForwardEncoder(MNN::Express::VARP features) const;

  /** Run the decoder model.
   *
   * @param tokens A int64 tensor of shape (N, num_words)
   * @param n_layer_self_k_cache  A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   * @param n_layer_self_v_cache  A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
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

  int32_t DetectLanguage(MNN::Express::VARP &cross_k,   // NOLINT
                         MNN::Express::VARP &cross_v);  // NOLINT

  /** Return the initial self kv cache in a pair
   *  - n_layer_self_k_cache A 4-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   *  - n_layer_self_v_cache A 4-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   */
  std::pair<MNN::Express::VARP, MNN::Express::VARP> GetInitialSelfKVCache() const;
  const std::vector<int> &GetInitialTokens() const;
  const std::vector<int32_t> &GetAllLanguageIDs() const;
  const std::unordered_map<std::string, int32_t> &GetLang2ID() const;
  const std::unordered_map<int32_t, std::string> &GetID2Lang() const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

  int32_t NoTimeStampsToken() const;
  int32_t EOT() const;
  int32_t SOT() const;
  int32_t TextCtx() const;
  int32_t VocabSize() const;
  int32_t FeatureDim() const;
  int32_t Translate() const;
  bool IsMultiLingual() const;

  static void NormalizeFeatures(float *features, int32_t num_frames,
                                int32_t feat_dim);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_H_
