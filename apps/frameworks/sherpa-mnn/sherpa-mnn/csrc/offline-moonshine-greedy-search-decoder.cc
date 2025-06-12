// sherpa-mnn/csrc/offline-moonshine-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-moonshine-greedy-search-decoder.h"

#include <algorithm>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

std::vector<OfflineMoonshineDecoderResult>
OfflineMoonshineGreedySearchDecoder::Decode(MNN::Express::VARP encoder_out) {
  auto encoder_out_shape = encoder_out->getInfo()->dim;
  if (encoder_out_shape[0] != 1) {
    SHERPA_ONNX_LOGE("Support only batch size == 1. Given: %d\n",
                     static_cast<int32_t>(encoder_out_shape[0]));
    return {};
  }

  auto memory_info =
      (MNNAllocator*)(nullptr);

  // encoder_out_shape[1] * 384 is the number of audio samples
  // 16000 is the sample rate
  //
  //
  // 384 is from the moonshine paper
  int32_t max_len =
      static_cast<int32_t>(encoder_out_shape[1] * 384 / 16000.0 * 6);

  int32_t sos = 1;
  int32_t eos = 2;
  int32_t seq_len = 1;

  std::vector<int32_t> tokens;

  std::array<int, 2> token_shape = {1, 1};
  int seq_len_shape = 1;

  MNN::Express::VARP token_tensor = MNNUtilsCreateTensor(
      memory_info, &sos, 1, token_shape.data(), token_shape.size());

  MNN::Express::VARP seq_len_tensor =
      MNNUtilsCreateTensor(memory_info, &seq_len, 1, &seq_len_shape, 1);

  MNN::Express::VARP logits{nullptr};
  std::vector<MNN::Express::VARP> states;

  std::tie(logits, states) = model_->ForwardUnCachedDecoder(
      std::move(token_tensor), std::move(seq_len_tensor), View(encoder_out));

  int32_t vocab_size = logits->getInfo()->dim[2];

  for (int32_t i = 0; i != max_len; ++i) {
    const float *p = logits->readMap<float>();

    int32_t max_token_id = static_cast<int32_t>(
        std::distance(p, std::max_element(p, p + vocab_size)));
    if (max_token_id == eos) {
      break;
    }
    tokens.push_back(max_token_id);

    seq_len += 1;

    token_tensor = MNNUtilsCreateTensor(
        memory_info, &tokens.back(), 1, token_shape.data(), token_shape.size());

    seq_len_tensor =
        MNNUtilsCreateTensor(memory_info, &seq_len, 1, &seq_len_shape, 1);

    // To fix the false alarm of clang-tidy
    // error: 'states' used after it was moved
    // [bugprone-use-after-move,-warnings-as-errors]
    // we use a tmp_states here
    std::vector<MNN::Express::VARP> tmp_states{std::move(states)};

    std::tie(logits, states) = model_->ForwardCachedDecoder(
        std::move(token_tensor), std::move(seq_len_tensor), View(encoder_out),
        std::move(tmp_states));
  }

  OfflineMoonshineDecoderResult ans;
  ans.tokens = std::move(tokens);

  return {ans};
}

}  // namespace sherpa_mnn
