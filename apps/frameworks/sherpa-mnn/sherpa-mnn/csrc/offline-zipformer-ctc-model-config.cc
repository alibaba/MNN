// sherpa-mnn/csrc/offline-zipformer-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-zipformer-ctc-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineZipformerCtcModelConfig::Register(ParseOptions *po) {
  po->Register("zipformer-ctc-model", &model, "Path to zipformer CTC model");
}

bool OfflineZipformerCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("zipformer CTC model file '%s' does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

std::string OfflineZipformerCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineZipformerCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
