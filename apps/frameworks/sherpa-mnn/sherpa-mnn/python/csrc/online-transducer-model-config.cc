// sherpa-mnn/python/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-transducer-model-config.h"

#include <string>

#include "sherpa-mnn/python/csrc/online-transducer-model-config.h"

namespace sherpa_mnn {

void PybindOnlineTransducerModelConfig(py::module *m) {
  using PyClass = OnlineTransducerModelConfig;
  py::class_<PyClass>(*m, "OnlineTransducerModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("encoder"), py::arg("decoder"), py::arg("joiner"))
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("joiner", &PyClass::joiner)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
