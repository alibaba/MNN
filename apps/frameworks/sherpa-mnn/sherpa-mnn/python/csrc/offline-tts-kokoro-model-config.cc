// sherpa-mnn/python/csrc/offline-tts-kokoro-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-tts-kokoro-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-tts-kokoro-model-config.h"

namespace sherpa_mnn {

void PybindOfflineTtsKokoroModelConfig(py::module *m) {
  using PyClass = OfflineTtsKokoroModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsKokoroModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, float>(),
           py::arg("model"), py::arg("voices"), py::arg("tokens"),
           py::arg("lexicon") = "", py::arg("data_dir"),
           py::arg("dict_dir") = "", py::arg("length_scale") = 1.0)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("voices", &PyClass::voices)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("dict_dir", &PyClass::dict_dir)
      .def_readwrite("length_scale", &PyClass::length_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
