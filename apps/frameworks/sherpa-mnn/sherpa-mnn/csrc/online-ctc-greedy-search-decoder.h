// sherpa-mnn/csrc/online-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/online-ctc-decoder.h"

namespace sherpa_mnn {

class OnlineCtcGreedySearchDecoder : public OnlineCtcDecoder {
 public:
  explicit OnlineCtcGreedySearchDecoder(int32_t blank_id)
      : blank_id_(blank_id) {}

  void Decode(const float *log_probs, int32_t batch_size, int32_t num_frames,
              int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
              OnlineStream **ss = nullptr, int32_t n = 0) override;

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_
