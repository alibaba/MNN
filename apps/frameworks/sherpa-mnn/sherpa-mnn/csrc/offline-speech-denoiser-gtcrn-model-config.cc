// sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineSpeechDenoiserGtcrnModelConfig::Register(ParseOptions *po) {
  po->Register("speech-denoiser-gtcrn-model", &model,
               "Path to the gtcrn model for speech denoising");
}

bool OfflineSpeechDenoiserGtcrnModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-gtcrn-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("gtcrn model file '%s' does not exist", model.c_str());
    return false;
  }
  return true;
}

std::string OfflineSpeechDenoiserGtcrnModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserGtcrnModelConfig(";
  os << "model=\"" << model << "\")";
  return os.str();
}

}  // namespace sherpa_mnn
