// sherpa-mnn/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-whisper-greedy-search-decoder.h"

#include <algorithm>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

void OfflineWhisperGreedySearchDecoder::SetConfig(
    const OfflineWhisperModelConfig &config) {
  config_ = config;
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoder::Decode(MNN::Express::VARP cross_k,
                                          MNN::Express::VARP cross_v,
                                          int32_t num_feature_frames) {
  auto memory_info =
      (MNNAllocator*)(nullptr);

  // For multilingual models, initial_tokens contains [sot, language, task]
  //   - language is English by default
  //   - task is transcribe by default
  //
  // For non-multilingual models, initial_tokens contains [sot]
  std::vector<int> initial_tokens = model_->GetInitialTokens();

  if (model_->IsMultiLingual()) {
    if (!config_.language.empty()) {
      const auto &lang2id = model_->GetLang2ID();

      if (!lang2id.count(config_.language)) {
        SHERPA_ONNX_LOGE("Invalid language: %s", config_.language.c_str());
        exit(-1);
      }

      int32_t lang_id = lang2id.at(config_.language);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    } else {
      int32_t lang_id = model_->DetectLanguage(cross_k, cross_v);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    }

    if (config_.task == "translate") {
      initial_tokens[2] = model_->Translate();
    } else if (config_.task != "transcribe") {
      // initial_tokens[2] is transcribe by default
      SHERPA_ONNX_LOGE(
          "Unsupported task: %s. Valid values are: transcribe, translate.",
          config_.task.c_str());
    }
  }

  initial_tokens.push_back(model_->NoTimeStampsToken());

  int32_t batch_size = 1;
  std::array<int, 2> token_shape{
      batch_size, static_cast<int>(initial_tokens.size())};

  MNN::Express::VARP tokens = MNNUtilsCreateTensor(
      memory_info, initial_tokens.data(), initial_tokens.size(),
      token_shape.data(), token_shape.size());

  std::array<int, 1> offset_shape{1};
  MNN::Express::VARP offset = MNNUtilsCreateTensor<int>(
      model_->Allocator(), offset_shape.data(), offset_shape.size());
  *(offset->writeMap<int>()) = 0;

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  auto decoder_out = model_->ForwardDecoder(
      std::move(tokens), std::move(self_kv_cache.first),
      std::move(self_kv_cache.second), std::move(cross_k), std::move(cross_v),
      std::move(offset));

  *(std::get<5>(decoder_out)->writeMap<int>()) =
      initial_tokens.size();

  const auto &logits = std::get<0>(decoder_out);
  const float *p_logits = logits->readMap<float>();

  auto logits_shape = logits->getInfo()->dim;
  int32_t vocab_size = logits_shape[2];

  const float *p_start = p_logits + (logits_shape[1] - 1) * vocab_size;

  int32_t max_token_id = static_cast<int32_t>(
      std::distance(p_start, std::max_element(p_start, p_start + vocab_size)));

  int32_t n_text_ctx = model_->TextCtx();

  std::vector<int32_t> predicted_tokens;

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100 * 6;
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, n_text_ctx / 2);

  for (int32_t i = 0; i < num_possible_tokens; ++i) {
    if (max_token_id == model_->EOT()) {
      break;
    }

    predicted_tokens.push_back(max_token_id);

    std::array<int, 2> token_shape{1, 1};
    MNN::Express::VARP tokens = MNNUtilsCreateTensor<int>(
        model_->Allocator(), token_shape.data(), token_shape.size());

    int *p_tokens = tokens->writeMap<int>();
    p_tokens[0] = max_token_id;

    decoder_out = model_->ForwardDecoder(std::move(tokens),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)));

    int *p_offset =
        std::get<5>(decoder_out)->writeMap<int>();

    *p_offset += 1;
    if (*p_offset >= n_text_ctx - 1) {
      break;
    }

    const auto &logits = std::get<0>(decoder_out);
    const float *p_logits = logits->readMap<float>();

    max_token_id = static_cast<int>(std::distance(
        p_logits, std::max_element(p_logits, p_logits + vocab_size)));
  }

  std::vector<OfflineWhisperDecoderResult> ans(1);

  const auto &id2lang = model_->GetID2Lang();
  if (id2lang.count(initial_tokens[1])) {
    ans[0].lang = id2lang.at(initial_tokens[1]);
  } else {
    ans[0].lang = "";
  }

  ans[0].tokens = std::move(predicted_tokens);

  return ans;
}

}  // namespace sherpa_mnn
