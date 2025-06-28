// sherpa-mnn/python/csrc/online-lm-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-lm-config.h"

#include <string>

#include "sherpa-mnn//csrc/online-lm-config.h"

namespace sherpa_mnn {

void PybindOnlineLMConfig(py::module *m) {
  using PyClass = OnlineLMConfig;
  py::class_<PyClass>(*m, "OnlineLMConfig")
      .def(py::init<const std::string &, float, int32_t,
           const std::string &, bool>(),
           py::arg("model") = "", py::arg("scale") = 0.5f,
           py::arg("lm_num_threads") = 1, py::arg("lm_provider") = "cpu",
           py::arg("shallow_fusion") = true)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("scale", &PyClass::scale)
      .def_readwrite("lm_provider", &PyClass::lm_provider)
      .def_readwrite("lm_num_threads", &PyClass::lm_num_threads)
      .def_readwrite("shallow_fusion", &PyClass::shallow_fusion)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_mnn
