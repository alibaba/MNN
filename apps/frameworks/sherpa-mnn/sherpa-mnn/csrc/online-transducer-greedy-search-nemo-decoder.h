// sherpa-mnn/csrc/online-transducer-greedy-search-nemo-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/online-transducer-decoder.h"
#include "sherpa-mnn/csrc/online-transducer-nemo-model.h"

namespace sherpa_mnn {

class OnlineStream;

class OnlineTransducerGreedySearchNeMoDecoder {
 public:
  OnlineTransducerGreedySearchNeMoDecoder(OnlineTransducerNeMoModel *model,
                                          float blank_penalty)
      : model_(model), blank_penalty_(blank_penalty) {}

  // @param n number of elements in ss
  void Decode(MNN::Express::VARP encoder_out, OnlineStream **ss, int32_t n) const;

 private:
  OnlineTransducerNeMoModel *model_;  // Not owned
  float blank_penalty_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
