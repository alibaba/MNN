// sherpa-mnn/python/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-tts-vits-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-tts-vits-model-config.h"

namespace sherpa_mnn {

void PybindOfflineTtsVitsModelConfig(py::module *m) {
  using PyClass = OfflineTtsVitsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsVitsModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, float, float, float>(),
           py::arg("model"), py::arg("lexicon"), py::arg("tokens"),
           py::arg("data_dir") = "", py::arg("dict_dir") = "",
           py::arg("noise_scale") = 0.667, py::arg("noise_scale_w") = 0.8,
           py::arg("length_scale") = 1.0)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("dict_dir", &PyClass::dict_dir)
      .def_readwrite("noise_scale", &PyClass::noise_scale)
      .def_readwrite("noise_scale_w", &PyClass::noise_scale_w)
      .def_readwrite("length_scale", &PyClass::length_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
