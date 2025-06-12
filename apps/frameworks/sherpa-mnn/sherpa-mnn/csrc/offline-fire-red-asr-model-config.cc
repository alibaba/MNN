// sherpa-mnn/csrc/offline-fire-red-asr-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-fire-red-asr-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineFireRedAsrModelConfig::Register(ParseOptions *po) {
  po->Register("fire-red-asr-encoder", &encoder,
               "Path to onnx encoder of FireRedAsr");

  po->Register("fire-red-asr-decoder", &decoder,
               "Path to onnx decoder of FireRedAsr");
}

bool OfflineFireRedAsrModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --fire-red-asr-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("FireRedAsr encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --fire-red-asr-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("FireRedAsr decoder file '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  return true;
}

std::string OfflineFireRedAsrModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineFireRedAsrModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
