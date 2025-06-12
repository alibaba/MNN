// sherpa-mnn/python/csrc/display.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/display.h"

#include "sherpa-mnn/csrc/display.h"

namespace sherpa_mnn {

void PybindDisplay(py::module *m) {
  using PyClass = Display;
  py::class_<PyClass>(*m, "Display")
      .def(py::init<int32_t>(), py::arg("max_word_per_line") = 60)
      .def("print", &PyClass::Print, py::arg("idx"), py::arg("s"));
}

}  // namespace sherpa_mnn
