// sherpa-mnn/csrc/online-paraformer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OnlineParaformerModelConfig {
  std::string encoder;
  std::string decoder;

  OnlineParaformerModelConfig() = default;

  OnlineParaformerModelConfig(const std::string &encoder,
                              const std::string &decoder)
      : encoder(encoder), decoder(decoder) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_CONFIG_H_
