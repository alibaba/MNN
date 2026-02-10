// sherpa-mnn/python/csrc/offline-moonshine-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-moonshine-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/python/csrc/offline-moonshine-model-config.h"

namespace sherpa_mnn {

void PybindOfflineMoonshineModelConfig(py::module *m) {
  using PyClass = OfflineMoonshineModelConfig;
  py::class_<PyClass>(*m, "OfflineMoonshineModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("preprocessor"), py::arg("encoder"),
           py::arg("uncached_decoder"), py::arg("cached_decoder"))
      .def_readwrite("preprocessor", &PyClass::preprocessor)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("uncached_decoder", &PyClass::uncached_decoder)
      .def_readwrite("cached_decoder", &PyClass::cached_decoder)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
