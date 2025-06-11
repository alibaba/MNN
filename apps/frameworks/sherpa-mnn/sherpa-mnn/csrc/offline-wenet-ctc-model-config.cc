// sherpa-mnn/csrc/offline-wenet-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-wenet-ctc-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineWenetCtcModelConfig::Register(ParseOptions *po) {
  po->Register(
      "wenet-ctc-model", &model,
      "Path to model.onnx from WeNet. Please see "
      "https://github.com/k2-fsa/sherpa-mnn/pull/425 for available models");
}

bool OfflineWenetCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("WeNet model: '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineWenetCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineWenetCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
