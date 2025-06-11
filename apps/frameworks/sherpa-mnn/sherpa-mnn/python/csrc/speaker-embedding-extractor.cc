// sherpa-mnn/python/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/speaker-embedding-extractor.h"

#include <string>

#include "sherpa-mnn/csrc/speaker-embedding-extractor.h"

namespace sherpa_mnn {

static void PybindSpeakerEmbeddingExtractorConfig(py::module *m) {
  using PyClass = SpeakerEmbeddingExtractorConfig;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractorConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t, bool, const std::string &>(),
           py::arg("model"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindSpeakerEmbeddingExtractor(py::module *m) {
  PybindSpeakerEmbeddingExtractorConfig(m);

  using PyClass = SpeakerEmbeddingExtractor;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractor")
      .def(py::init<const SpeakerEmbeddingExtractorConfig &>(),
           py::arg("config"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dim", &PyClass::Dim)
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>())
      .def("compute", &PyClass::Compute,
           py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
