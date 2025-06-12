// sherpa-mnn/csrc/offline-paraformer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-paraformer-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineParaformerModelConfig::Register(ParseOptions *po) {
  po->Register("paraformer", &model, "Path to model.onnx of paraformer.");
}

bool OfflineParaformerModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Paraformer model '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineParaformerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineParaformerModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
