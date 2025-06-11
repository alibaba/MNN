// sherpa-mnn/csrc/provider.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/provider.h"

#include <algorithm>
#include <cctype>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

Provider StringToProvider(std::string s) {
  std::transform(s.cbegin(), s.cend(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "cpu") {
    return Provider::kCPU;
  } else if (s == "cuda") {
    return Provider::kCUDA;
  } else if (s == "coreml") {
    return Provider::kCoreML;
  } else if (s == "xnnpack") {
    return Provider::kXnnpack;
  } else if (s == "nnapi") {
    return Provider::kNNAPI;
  } else if (s == "trt") {
    return Provider::kTRT;
  } else if (s == "directml") {
    return Provider::kDirectML;
  } else {
    SHERPA_ONNX_LOGE("Unsupported string: %s. Fallback to cpu", s.c_str());
    return Provider::kCPU;
  }
}

}  // namespace sherpa_mnn
