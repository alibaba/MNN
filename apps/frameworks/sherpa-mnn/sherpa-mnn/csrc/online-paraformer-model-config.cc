// sherpa-mnn/csrc/online-paraformer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-paraformer-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OnlineParaformerModelConfig::Register(ParseOptions *po) {
  po->Register("paraformer-encoder", &encoder,
               "Path to encoder.onnx of paraformer.");
  po->Register("paraformer-decoder", &decoder,
               "Path to decoder.onnx of paraformer.");
}

bool OnlineParaformerModelConfig::Validate() const {
  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("Paraformer encoder '%s' does not exist", encoder.c_str());
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("Paraformer decoder '%s' does not exist", decoder.c_str());
    return false;
  }

  return true;
}

std::string OnlineParaformerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineParaformerModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
