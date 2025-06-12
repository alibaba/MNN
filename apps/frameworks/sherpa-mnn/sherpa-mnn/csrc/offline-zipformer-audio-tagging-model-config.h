// sherpa-mnn/csrc/offline-zipformer-audio-tagging-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineZipformerAudioTaggingModelConfig {
  std::string model;

  OfflineZipformerAudioTaggingModelConfig() = default;

  explicit OfflineZipformerAudioTaggingModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_CONFIG_H_
