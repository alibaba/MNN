// sherpa-mnn/csrc/offline-paraformer-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/offline-paraformer-decoder.h"

namespace sherpa_mnn {

class OfflineParaformerGreedySearchDecoder : public OfflineParaformerDecoder {
 public:
  explicit OfflineParaformerGreedySearchDecoder(int32_t eos_id)
      : eos_id_(eos_id) {}

  std::vector<OfflineParaformerDecoderResult> Decode(
      MNN::Express::VARP log_probs, MNN::Express::VARP token_num,
      MNN::Express::VARP us_cif_peak = MNN::Express::VARP(nullptr)) override;

 private:
  int32_t eos_id_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_GREEDY_SEARCH_DECODER_H_
