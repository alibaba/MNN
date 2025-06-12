// sherpa-mnn/csrc/offline-sense-voice-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineSenseVoiceModelConfig {
  std::string model;

  // "" or "auto" to let the model recognize the language
  // valid values:
  //  zh, en, ja, ko, yue, auto
  std::string language = "auto";

  // true to use inverse text normalization
  // false to not use inverse text normalization
  bool use_itn = false;

  OfflineSenseVoiceModelConfig() = default;
  explicit OfflineSenseVoiceModelConfig(const std::string &model,
                                        const std::string &language,
                                        bool use_itn)
      : model(model), language(language), use_itn(use_itn) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_
