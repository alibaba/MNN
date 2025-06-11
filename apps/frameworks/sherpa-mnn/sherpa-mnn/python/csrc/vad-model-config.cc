// sherpa-mnn/python/csrc/vad-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/vad-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/vad-model-config.h"
#include "sherpa-mnn/python/csrc/silero-vad-model-config.h"

namespace sherpa_mnn {

void PybindVadModelConfig(py::module *m) {
  PybindSileroVadModelConfig(m);

  using PyClass = VadModelConfig;
  py::class_<PyClass>(*m, "VadModelConfig")
      .def(py::init<>())
      .def(py::init<const SileroVadModelConfig &, int32_t, int32_t,
                    const std::string &, bool>(),
           py::arg("silero_vad"), py::arg("sample_rate") = 16000,
           py::arg("num_threads") = 1, py::arg("provider") = "cpu",
           py::arg("debug") = false)
      .def_readwrite("silero_vad", &PyClass::silero_vad)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
