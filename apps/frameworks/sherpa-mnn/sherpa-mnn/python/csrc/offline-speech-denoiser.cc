// sherpa-mnn/python/csrc/offline-speech-denoiser.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-speech-denoiser.h"

#include <vector>

#include "sherpa-mnn/csrc/offline-speech-denoiser.h"
#include "sherpa-mnn/python/csrc/offline-speech-denoiser-model-config.h"

namespace sherpa_mnn {

void PybindOfflineSpeechDenoiserConfig(py::module *m) {
  PybindOfflineSpeechDenoiserModelConfig(m);

  using PyClass = OfflineSpeechDenoiserConfig;

  py::class_<PyClass>(*m, "OfflineSpeechDenoiserConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeechDenoiserModelConfig &>(),
           py::arg("model") = OfflineSpeechDenoiserModelConfig{})
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindDenoisedAudio(py::module *m) {
  using PyClass = DenoisedAudio;
  py::class_<PyClass>(*m, "DenoisedAudio")
      .def_property_readonly(
          "sample_rate", [](const PyClass &self) { return self.sample_rate; })
      .def_property_readonly("samples",
                             [](const PyClass &self) { return self.samples; });
}

void PybindOfflineSpeechDenoiser(py::module *m) {
  PybindOfflineSpeechDenoiserConfig(m);
  PybindDenoisedAudio(m);
  using PyClass = OfflineSpeechDenoiser;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiser")
      .def(py::init<const OfflineSpeechDenoiserConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "__call__",
          [](const PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "run",
          [](const PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::GetSampleRate);
}

}  // namespace sherpa_mnn
