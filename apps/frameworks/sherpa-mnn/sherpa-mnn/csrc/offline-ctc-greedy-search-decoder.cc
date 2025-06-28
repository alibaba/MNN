// sherpa-mnn/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    MNN::Express::VARP log_probs, MNN::Express::VARP log_probs_length) {
  std::vector<int> shape = log_probs->getInfo()->dim;
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);

  const int *p_log_probs_length = log_probs_length->readMap<int>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t b = 0; b != batch_size; ++b) {
    const float *p_log_probs =
        log_probs->readMap<float>() + b * num_frames * vocab_size;

    OfflineCtcDecoderResult r;
    int prev_id = -1;

    for (int32_t t = 0; t != static_cast<int32_t>(p_log_probs_length[b]); ++t) {
      auto y = static_cast<int>(std::distance(
          static_cast<const float *>(p_log_probs),
          std::max_element(
              static_cast<const float *>(p_log_probs),
              static_cast<const float *>(p_log_probs) + vocab_size)));
      p_log_probs += vocab_size;

      if (y != blank_id_ && y != prev_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t);
      }
      prev_id = y;
    }  // for (int32_t t = 0; ...)

    ans.push_back(std::move(r));
  }
  return ans;
}

}  // namespace sherpa_mnn
