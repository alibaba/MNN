// sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineSpeechDenoiserGtcrnModelConfig {
  std::string model;
  OfflineSpeechDenoiserGtcrnModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_CONFIG_H_
