// sherpa-mnn/csrc/offline-tdnn-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tdnn-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineTdnnModelConfig::Register(ParseOptions *po) {
  po->Register("tdnn-model", &model, "Path to onnx model");
}

bool OfflineTdnnModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("tdnn model file %s does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineTdnnModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTdnnModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
