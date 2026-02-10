// sherpa-mnn/python/csrc/online-wenet-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-wenet-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/online-wenet-ctc-model-config.h"

namespace sherpa_mnn {

void PybindOnlineWenetCtcModelConfig(py::module *m) {
  using PyClass = OnlineWenetCtcModelConfig;
  py::class_<PyClass>(*m, "OnlineWenetCtcModelConfig")
      .def(py::init<const std::string &, int32_t, int32_t>(), py::arg("model"),
           py::arg("chunk_size") = 16, py::arg("num_left_chunks") = 4)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("chunk_size", &PyClass::chunk_size)
      .def_readwrite("num_left_chunks", &PyClass::num_left_chunks)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
