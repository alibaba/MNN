// sherpa-mnn/python/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-whisper-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/python/csrc/offline-whisper-model-config.h"

namespace sherpa_mnn {

void PybindOfflineWhisperModelConfig(py::module *m) {
  using PyClass = OfflineWhisperModelConfig;
  py::class_<PyClass>(*m, "OfflineWhisperModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t>(),
           py::arg("encoder"), py::arg("decoder"), py::arg("language"),
           py::arg("task"), py::arg("tail_paddings") = -1)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("task", &PyClass::task)
      .def_readwrite("tail_paddings", &PyClass::tail_paddings)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
