// sherpa-mnn/csrc/online-ctc-fst-decoder-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OnlineCtcFstDecoderConfig {
  // Path to H.fst, HL.fst or HLG.fst
  std::string graph;
  int32_t max_active = 3000;

  OnlineCtcFstDecoderConfig() = default;

  OnlineCtcFstDecoderConfig(const std::string &graph, int32_t max_active)
      : graph(graph), max_active(max_active) {}

  std::string ToString() const;

  void Register(ParseOptions *po);
  bool Validate() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_CONFIG_H_
