// sherpa-mnn/python/csrc/offline-speech-denoiser-gtcrn-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-speech-denoiser-gtcrn-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model-config.h"

namespace sherpa_mnn {

void PybindOfflineSpeechDenoiserGtcrnModelConfig(py::module *m) {
  using PyClass = OfflineSpeechDenoiserGtcrnModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiserGtcrnModelConfig")
      .def(py::init<const std::string &>(), py::arg("model") = "")
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
