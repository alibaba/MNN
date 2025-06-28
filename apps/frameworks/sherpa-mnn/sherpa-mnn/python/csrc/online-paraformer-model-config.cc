// sherpa-mnn/python/csrc/online-paraformer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-paraformer-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/online-paraformer-model-config.h"

namespace sherpa_mnn {

void PybindOnlineParaformerModelConfig(py::module *m) {
  using PyClass = OnlineParaformerModelConfig;
  py::class_<PyClass>(*m, "OnlineParaformerModelConfig")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("encoder"), py::arg("decoder"))
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
