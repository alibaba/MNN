// sherpa-mnn/python/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-speech-denoiser-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-mnn/python/csrc/offline-speech-denoiser-gtcrn-model-config.h"

namespace sherpa_mnn {

void PybindOfflineSpeechDenoiserModelConfig(py::module *m) {
  PybindOfflineSpeechDenoiserGtcrnModelConfig(m);

  using PyClass = OfflineSpeechDenoiserModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiserModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeechDenoiserGtcrnModelConfig &, int32_t,
                    bool, const std::string &>(),
           py::arg("gtcrn") = OfflineSpeechDenoiserGtcrnModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("gtcrn", &PyClass::gtcrn)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
