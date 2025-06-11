// sherpa-mnn/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speech-denoiser-model-config.h"

#include <string>

namespace sherpa_mnn {

void OfflineSpeechDenoiserModelConfig::Register(ParseOptions *po) {
  gtcrn.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineSpeechDenoiserModelConfig::Validate() const {
  return gtcrn.Validate();
}

std::string OfflineSpeechDenoiserModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserModelConfig(";
  os << "gtcrn=" << gtcrn.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
