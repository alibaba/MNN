// sherpa-mnn/csrc/offline-fire-red-asr-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/offline-fire-red-asr-decoder.h"
#include "sherpa-mnn/csrc/offline-fire-red-asr-model.h"

namespace sherpa_mnn {

class OfflineFireRedAsrGreedySearchDecoder : public OfflineFireRedAsrDecoder {
 public:
  explicit OfflineFireRedAsrGreedySearchDecoder(OfflineFireRedAsrModel *model)
      : model_(model) {}

  std::vector<OfflineFireRedAsrDecoderResult> Decode(
      MNN::Express::VARP cross_k, MNN::Express::VARP cross_v) override;

 private:
  OfflineFireRedAsrModel *model_;  // not owned
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_
