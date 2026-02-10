// sherpa-mnn/csrc/online-nemo-ctc-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OnlineNeMoCtcModelConfig {
  std::string model;

  OnlineNeMoCtcModelConfig() = default;

  explicit OnlineNeMoCtcModelConfig(const std::string &model) : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_
