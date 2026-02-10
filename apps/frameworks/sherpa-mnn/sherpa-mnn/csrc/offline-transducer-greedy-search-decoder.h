// sherpa-mnn/csrc/offline-transducer-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/offline-transducer-decoder.h"
#include "sherpa-mnn/csrc/offline-transducer-model.h"

namespace sherpa_mnn {

class OfflineTransducerGreedySearchDecoder : public OfflineTransducerDecoder {
 public:
  OfflineTransducerGreedySearchDecoder(OfflineTransducerModel *model,
                                       int32_t unk_id,
                                       float blank_penalty)
      : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      MNN::Express::VARP encoder_out, MNN::Express::VARP encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
