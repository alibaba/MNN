// sherpa-mnn/python/csrc/online-punctuation.cc
//
// Copyright (c) 2024

#include "sherpa-mnn/python/csrc/online-punctuation.h"

#include <string>

#include "sherpa-mnn/csrc/online-punctuation.h"

namespace sherpa_mnn {

static void PybindOnlinePunctuationModelConfig(py::module *m) {
  using PyClass = OnlinePunctuationModelConfig;
  py::class_<PyClass>(*m, "OnlinePunctuationModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, int32_t, bool,
                    const std::string &>(),
           py::arg("cnn_bilstm"), py::arg("bpe_vocab"),
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("cnn_bilstm", &PyClass::cnn_bilstm)
      .def_readwrite("bpe_vocab", &PyClass::bpe_vocab)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindOnlinePunctuationConfig(py::module *m) {
  PybindOnlinePunctuationModelConfig(m);
  using PyClass = OnlinePunctuationConfig;

  py::class_<PyClass>(*m, "OnlinePunctuationConfig")
      .def(py::init<>())
      .def(py::init<const OnlinePunctuationModelConfig &>(),
           py::arg("model_config"))
      .def_readwrite("model_config", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlinePunctuation(py::module *m) {
  PybindOnlinePunctuationConfig(m);
  using PyClass = OnlinePunctuation;

  py::class_<PyClass>(*m, "OnlinePunctuation")
      .def(py::init<const OnlinePunctuationConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_punctuation_with_case", &PyClass::AddPunctuationWithCase,
           py::arg("text"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
