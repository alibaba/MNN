// sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineNemoEncDecCtcModelConfig {
  std::string model;

  OfflineNemoEncDecCtcModelConfig() = default;
  explicit OfflineNemoEncDecCtcModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_CONFIG_H_
