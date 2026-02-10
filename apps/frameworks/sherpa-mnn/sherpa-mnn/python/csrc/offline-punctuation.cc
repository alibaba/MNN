// sherpa-mnn/python/csrc/offline-punctuation.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-punctuation.h"

#include <string>

#include "sherpa-mnn/csrc/offline-punctuation.h"

namespace sherpa_mnn {

static void PybindOfflinePunctuationModelConfig(py::module *m) {
  using PyClass = OfflinePunctuationModelConfig;
  py::class_<PyClass>(*m, "OfflinePunctuationModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t, bool, const std::string &>(),
           py::arg("ct_transformer"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("ct_transformer", &PyClass::ct_transformer)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindOfflinePunctuationConfig(py::module *m) {
  PybindOfflinePunctuationModelConfig(m);
  using PyClass = OfflinePunctuationConfig;

  py::class_<PyClass>(*m, "OfflinePunctuationConfig")
      .def(py::init<>())
      .def(py::init<const OfflinePunctuationModelConfig &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflinePunctuation(py::module *m) {
  PybindOfflinePunctuationConfig(m);
  using PyClass = OfflinePunctuation;

  py::class_<PyClass>(*m, "OfflinePunctuation")
      .def(py::init<const OfflinePunctuationConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_punctuation", &PyClass::AddPunctuation, py::arg("text"),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
