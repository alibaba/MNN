// sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineSpeakerSegmentationPyannoteModelConfig::Register(ParseOptions *po) {
  po->Register("pyannote-model", &model,
               "Path to model.onnx of the Pyannote segmentation model.");
}

bool OfflineSpeakerSegmentationPyannoteModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Pyannote segmentation model: '%s' does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationPyannoteModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationPyannoteModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
