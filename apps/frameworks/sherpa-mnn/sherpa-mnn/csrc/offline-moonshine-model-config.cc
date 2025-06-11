// sherpa-mnn/csrc/offline-moonshine-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-moonshine-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineMoonshineModelConfig::Register(ParseOptions *po) {
  po->Register("moonshine-preprocessor", &preprocessor,
               "Path to onnx preprocessor of moonshine, e.g., preprocess.onnx");

  po->Register("moonshine-encoder", &encoder,
               "Path to onnx encoder of moonshine, e.g., encode.onnx");

  po->Register(
      "moonshine-uncached-decoder", &uncached_decoder,
      "Path to onnx uncached_decoder of moonshine, e.g., uncached_decode.onnx");

  po->Register(
      "moonshine-cached-decoder", &cached_decoder,
      "Path to onnx cached_decoder of moonshine, e.g., cached_decode.onnx");
}

bool OfflineMoonshineModelConfig::Validate() const {
  if (preprocessor.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moonshine-preprocessor");
    return false;
  }

  if (!FileExists(preprocessor)) {
    SHERPA_ONNX_LOGE("moonshine preprocessor file '%s' does not exist",
                     preprocessor.c_str());
    return false;
  }

  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moonshine-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("moonshine encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (uncached_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moonshine-uncached-decoder");
    return false;
  }

  if (!FileExists(uncached_decoder)) {
    SHERPA_ONNX_LOGE("moonshine uncached decoder file '%s' does not exist",
                     uncached_decoder.c_str());
    return false;
  }

  if (cached_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moonshine-cached-decoder");
    return false;
  }

  if (!FileExists(cached_decoder)) {
    SHERPA_ONNX_LOGE("moonshine cached decoder file '%s' does not exist",
                     cached_decoder.c_str());
    return false;
  }

  return true;
}

std::string OfflineMoonshineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineMoonshineModelConfig(";
  os << "preprocessor=\"" << preprocessor << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "uncached_decoder=\"" << uncached_decoder << "\", ";
  os << "cached_decoder=\"" << cached_decoder << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
