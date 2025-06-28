// sherpa-mnn/csrc/online-lm-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-lm-config.h"

#include <string>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OnlineLMConfig::Register(ParseOptions *po) {
  po->Register("lm", &model, "Path to LM model.");
  po->Register("lm-scale", &scale, "LM scale.");
  po->Register("lm-num-threads", &lm_num_threads,
               "Number of threads to run the neural network of LM model");
  po->Register("lm-provider", &lm_provider,
               "Specify a provider to LM model use: cpu, cuda, coreml");
  po->Register("lm-shallow-fusion", &shallow_fusion,
               "Boolean whether to use shallow fusion or rescore.");
}

bool OnlineLMConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("'%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OnlineLMConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineLMConfig(";
  os << "model=\"" << model << "\", ";
  os << "scale=" << scale << ", ";
  os << "shallow_fusion=" << (shallow_fusion ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_mnn
