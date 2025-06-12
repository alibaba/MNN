// sherpa-mnn/python/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/circular-buffer.h"

#include <vector>

#include "sherpa-mnn/csrc/circular-buffer.h"

namespace sherpa_mnn {

void PybindCircularBuffer(py::module *m) {
  using PyClass = CircularBuffer;
  py::class_<PyClass>(*m, "CircularBuffer")
      .def(py::init<int32_t>(), py::arg("capacity"))
      .def(
          "push",
          [](PyClass &self, const std::vector<float> &samples) {
            self.Push(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>())
      .def("get", &PyClass::Get, py::arg("start_index"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("pop", &PyClass::Pop, py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("size", &PyClass::Size)
      .def_property_readonly("head", &PyClass::Head)
      .def_property_readonly("tail", &PyClass::Tail);
}

}  // namespace sherpa_mnn
