// sherpa-mnn/csrc/offline-fire-red-asr-greedy-search-decoder.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-fire-red-asr-greedy-search-decoder.h"

#include <algorithm>
#include <tuple>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

// Note: this functions works only for batch size == 1 at present
std::vector<OfflineFireRedAsrDecoderResult>
OfflineFireRedAsrGreedySearchDecoder::Decode(MNN::Express::VARP cross_k,
                                             MNN::Express::VARP cross_v) {
  const auto &meta_data = model_->metaData();

  auto memory_info =
      (MNNAllocator*)(nullptr);

  // For multilingual models, initial_tokens contains [sot, language, task]
  //   - language is English by default
  //   - task is transcribe by default
  //
  // For non-multilingual models, initial_tokens contains [sot]
  std::array<int, 2> token_shape = {1, 1};
  int token = meta_data.sos_id;

  int32_t batch_size = 1;

  MNN::Express::VARP tokens = MNNUtilsCreateTensor(
      memory_info, &token, 1, token_shape.data(), token_shape.size());

  std::array<int, 1> offset_shape{1};
  MNN::Express::VARP offset = MNNUtilsCreateTensor<int>(
      model_->Allocator(), offset_shape.data(), offset_shape.size());
  *(offset->writeMap<int>()) = 0;

  std::vector<OfflineFireRedAsrDecoderResult> ans(1);

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  std::tuple<MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP,
             MNN::Express::VARP>
      decoder_out = {MNN::Express::VARP{nullptr},
                     std::move(self_kv_cache.first),
                     std::move(self_kv_cache.second),
                     std::move(cross_k),
                     std::move(cross_v),
                     std::move(offset)};

  for (int32_t i = 0; i < meta_data.max_len; ++i) {
    decoder_out = model_->ForwardDecoder(View(tokens),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)));

    const auto &logits = std::get<0>(decoder_out);
    const float *p_logits = logits->readMap<float>();

    auto logits_shape = logits->getInfo()->dim;
    int32_t vocab_size = logits_shape[2];

    int32_t max_token_id = static_cast<int32_t>(std::distance(
        p_logits, std::max_element(p_logits, p_logits + vocab_size)));
    if (max_token_id == meta_data.eos_id) {
      break;
    }

    ans[0].tokens.push_back(max_token_id);

    token = max_token_id;

    // increment offset
    *(std::get<5>(decoder_out)->writeMap<int>()) += 1;
  }

  return ans;
}

}  // namespace sherpa_mnn
