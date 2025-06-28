// sherpa-mnn/csrc/vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/vad-model-config.h"

#include <sstream>
#include <string>

namespace sherpa_mnn {

void VadModelConfig::Register(ParseOptions *po) {
  silero_vad.Register(po);

  po->Register("vad-sample-rate", &sample_rate,
               "Sample rate expected by the VAD model");

  po->Register("vad-num-threads", &num_threads,
               "Number of threads to run the VAD model");

  po->Register("vad-provider", &provider,
               "Specify a provider to run the VAD model. Supported values: "
               "cpu, cuda, coreml");

  po->Register("vad-debug", &debug,
               "true to display debug information when loading vad models");
}

bool VadModelConfig::Validate() const { return silero_vad.Validate(); }

std::string VadModelConfig::ToString() const {
  std::ostringstream os;

  os << "VadModelConfig(";
  os << "silero_vad=" << silero_vad.ToString() << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_mnn
