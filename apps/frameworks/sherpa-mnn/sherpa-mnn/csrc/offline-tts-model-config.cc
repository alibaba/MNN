// sherpa-mnn/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-model-config.h"

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineTtsModelConfig::Register(ParseOptions *po) {
  vits.Register(po);
  matcha.Register(po);
  kokoro.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineTtsModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!vits.model.empty()) {
    return vits.Validate();
  }

  if (!matcha.acoustic_model.empty()) {
    return matcha.Validate();
  }

  return kokoro.Validate();
}

std::string OfflineTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsModelConfig(";
  os << "vits=" << vits.ToString() << ", ";
  os << "matcha=" << matcha.ToString() << ", ";
  os << "kokoro=" << kokoro.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
