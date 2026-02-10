// sherpa-mnn/python/csrc/offline-nemo-enc-dec-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-nemo-enc-dec-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model-config.h"

namespace sherpa_mnn {

void PybindOfflineNemoEncDecCtcModelConfig(py::module *m) {
  using PyClass = OfflineNemoEncDecCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineNemoEncDecCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
