// sherpa-mnn/python/csrc/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/audio-tagging.h"

#include <string>

#include "sherpa-mnn/csrc/audio-tagging.h"

namespace sherpa_mnn {

static void PybindOfflineZipformerAudioTaggingModelConfig(py::module *m) {
  using PyClass = OfflineZipformerAudioTaggingModelConfig;
  py::class_<PyClass>(*m, "OfflineZipformerAudioTaggingModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindAudioTaggingModelConfig(py::module *m) {
  PybindOfflineZipformerAudioTaggingModelConfig(m);

  using PyClass = AudioTaggingModelConfig;

  py::class_<PyClass>(*m, "AudioTaggingModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineZipformerAudioTaggingModelConfig &,
                    const std::string &, int32_t, bool, const std::string &>(),
           py::arg("zipformer") = OfflineZipformerAudioTaggingModelConfig{},
           py::arg("ced") = "", py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("zipformer", &PyClass::zipformer)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindAudioTaggingConfig(py::module *m) {
  PybindAudioTaggingModelConfig(m);

  using PyClass = AudioTaggingConfig;

  py::class_<PyClass>(*m, "AudioTaggingConfig")
      .def(py::init<>())
      .def(py::init<const AudioTaggingModelConfig &, const std::string &,
                    int32_t>(),
           py::arg("model"), py::arg("labels"), py::arg("top_k") = 5)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("labels", &PyClass::labels)
      .def_readwrite("top_k", &PyClass::top_k)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindAudioEvent(py::module *m) {
  using PyClass = AudioEvent;

  py::class_<PyClass>(*m, "AudioEvent")
      .def_property_readonly(
          "name", [](const PyClass &self) -> std::string { return self.name; })
      .def_property_readonly(
          "index", [](const PyClass &self) -> int32_t { return self.index; })
      .def_property_readonly(
          "prob", [](const PyClass &self) -> float { return self.prob; })
      .def("__str__", &PyClass::ToString);
}

void PybindAudioTagging(py::module *m) {
  PybindAudioTaggingConfig(m);
  PybindAudioEvent(m);

  using PyClass = AudioTagging;

  py::class_<PyClass>(*m, "AudioTagging")
      .def(py::init<const AudioTaggingConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>())
      .def("compute", &PyClass::Compute, py::arg("s"), py::arg("top_k") = -1,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
