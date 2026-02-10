// sherpa-mnn/python/csrc/offline-wenet-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-wenet-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/python/csrc/offline-wenet-ctc-model-config.h"

namespace sherpa_mnn {

void PybindOfflineWenetCtcModelConfig(py::module *m) {
  using PyClass = OfflineWenetCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineWenetCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
