// sherpa-mnn/csrc/offline-speaker-segmentation-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/offline-speaker-segmentation-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineSpeakerSegmentationModelConfig::Register(ParseOptions *po) {
  pyannote.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineSpeakerSegmentationModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!pyannote.model.empty()) {
    return pyannote.Validate();
  }

  if (pyannote.model.empty()) {
    SHERPA_ONNX_LOGE(
        "You have to provide at least one speaker segmentation model");
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationModelConfig(";
  os << "pyannote=" << pyannote.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
