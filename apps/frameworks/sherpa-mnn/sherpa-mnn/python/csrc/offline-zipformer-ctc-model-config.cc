// sherpa-mnn/python/csrc/offline-zipformer-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-zipformer-ctc-model-config.h"

#include <string>

#include "sherpa-mnn/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_mnn {

void PybindOfflineZipformerCtcModelConfig(py::module *m) {
  using PyClass = OfflineZipformerCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineZipformerCtcModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
