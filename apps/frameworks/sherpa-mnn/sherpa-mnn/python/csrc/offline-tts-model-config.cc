// sherpa-mnn/python/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-tts-model-config.h"
#include "sherpa-mnn/python/csrc/offline-tts-kokoro-model-config.h"
#include "sherpa-mnn/python/csrc/offline-tts-matcha-model-config.h"
#include "sherpa-mnn/python/csrc/offline-tts-vits-model-config.h"

namespace sherpa_mnn {

void PybindOfflineTtsModelConfig(py::module *m) {
  PybindOfflineTtsVitsModelConfig(m);
  PybindOfflineTtsMatchaModelConfig(m);
  PybindOfflineTtsKokoroModelConfig(m);

  using PyClass = OfflineTtsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsVitsModelConfig &,
                    const OfflineTtsMatchaModelConfig &,
                    const OfflineTtsKokoroModelConfig &, int32_t, bool,
                    const std::string &>(),
           py::arg("vits") = OfflineTtsVitsModelConfig{},
           py::arg("matcha") = OfflineTtsMatchaModelConfig{},
           py::arg("kokoro") = OfflineTtsKokoroModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("vits", &PyClass::vits)
      .def_readwrite("matcha", &PyClass::matcha)
      .def_readwrite("kokoro", &PyClass::kokoro)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
