// sherpa-mnn/csrc/offline-paraformer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineParaformerModelConfig {
  std::string model;

  OfflineParaformerModelConfig() = default;
  explicit OfflineParaformerModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_
