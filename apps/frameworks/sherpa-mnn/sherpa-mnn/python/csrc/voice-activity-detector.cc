// sherpa-mnn/python/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/voice-activity-detector.h"

#include <vector>

#include "sherpa-mnn/csrc/voice-activity-detector.h"

namespace sherpa_mnn {

void PybindSpeechSegment(py::module *m) {
  using PyClass = SpeechSegment;
  py::class_<PyClass>(*m, "SpeechSegment")
      .def_property_readonly("start",
                             [](const PyClass &self) { return self.start; })
      .def_property_readonly("samples",
                             [](const PyClass &self) { return self.samples; });
}

void PybindVoiceActivityDetector(py::module *m) {
  PybindSpeechSegment(m);
  using PyClass = VoiceActivityDetector;
  py::class_<PyClass>(*m, "VoiceActivityDetector")
      .def(py::init<const VadModelConfig &, float>(), py::arg("config"),
           py::arg("buffer_size_in_seconds") = 60,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "accept_waveform",
          [](PyClass &self, const std::vector<float> &samples) {
            self.AcceptWaveform(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &PyClass::GetConfig)
      .def("empty", &PyClass::Empty, py::call_guard<py::gil_scoped_release>())
      .def("pop", &PyClass::Pop, py::call_guard<py::gil_scoped_release>())
      .def("is_speech_detected", &PyClass::IsSpeechDetected,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def("flush", &PyClass::Flush, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("front", &PyClass::Front);
}

}  // namespace sherpa_mnn
