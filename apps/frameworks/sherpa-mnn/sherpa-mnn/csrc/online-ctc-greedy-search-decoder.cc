// sherpa-mnn/csrc/online-ctc-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OnlineCtcGreedySearchDecoder::Decode(
    const float *log_probs, int32_t batch_size, int32_t num_frames,
    int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
    OnlineStream ** /*ss=nullptr*/, int32_t /*n = 0*/) {
  if (batch_size != results->size()) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, results.size(0): %d",
                     batch_size, static_cast<int32_t>(results->size()));
    exit(-1);
  }

  const float *p = log_probs;

  for (int32_t b = 0; b != batch_size; ++b) {
    auto &r = (*results)[b];

    int32_t prev_id = -1;

    for (int32_t t = 0; t != num_frames; ++t, p += vocab_size) {
      int32_t y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p),
          std::max_element(static_cast<const float *>(p),
                           static_cast<const float *>(p) + vocab_size)));

      if (y == blank_id_) {
        r.num_trailing_blanks += 1;
      } else {
        r.num_trailing_blanks = 0;
      }

      if (y != blank_id_ && y != prev_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t + r.frame_offset);
      }

      prev_id = y;
    }  // for (int32_t t = 0; t != num_frames; ++t) {
  }    // for (int32_t b = 0; b != batch_size; ++b)

  // Update frame_offset
  for (auto &r : *results) {
    r.frame_offset += num_frames;
  }
}

}  // namespace sherpa_mnn
