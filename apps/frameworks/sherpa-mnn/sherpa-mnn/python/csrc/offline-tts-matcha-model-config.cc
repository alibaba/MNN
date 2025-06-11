// sherpa-mnn/python/csrc/offline-tts-matcha-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-tts-matcha-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-tts-matcha-model-config.h"

namespace sherpa_mnn {

void PybindOfflineTtsMatchaModelConfig(py::module *m) {
  using PyClass = OfflineTtsMatchaModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsMatchaModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, float, float>(),
           py::arg("acoustic_model"), py::arg("vocoder"), py::arg("lexicon"),
           py::arg("tokens"), py::arg("data_dir") = "",
           py::arg("dict_dir") = "", py::arg("noise_scale") = 1.0,
           py::arg("length_scale") = 1.0)
      .def_readwrite("acoustic_model", &PyClass::acoustic_model)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("dict_dir", &PyClass::dict_dir)
      .def_readwrite("noise_scale", &PyClass::noise_scale)
      .def_readwrite("length_scale", &PyClass::length_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
