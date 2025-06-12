// sherpa-mnn/python/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/silero-vad-model-config.h"

#include <memory>
#include <string>

#include "sherpa-mnn/csrc/silero-vad-model-config.h"

namespace sherpa_mnn {

void PybindSileroVadModelConfig(py::module *m) {
  using PyClass = SileroVadModelConfig;
  py::class_<PyClass>(*m, "SileroVadModelConfig")
      .def(py::init<>())
      .def(py::init([](const std::string &model, float threshold,
                       float min_silence_duration, float min_speech_duration,
                       int32_t window_size,
                       float max_speech_duration) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();

             ans->model = model;
             ans->threshold = threshold;
             ans->min_silence_duration = min_silence_duration;
             ans->min_speech_duration = min_speech_duration;
             ans->window_size = window_size;
             ans->max_speech_duration = max_speech_duration;

             return ans;
           }),
           py::arg("model"), py::arg("threshold") = 0.5,
           py::arg("min_silence_duration") = 0.5,
           py::arg("min_speech_duration") = 0.25, py::arg("window_size") = 512,
           py::arg("max_speech_duration") = 20)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("threshold", &PyClass::threshold)
      .def_readwrite("min_silence_duration", &PyClass::min_silence_duration)
      .def_readwrite("min_speech_duration", &PyClass::min_speech_duration)
      .def_readwrite("window_size", &PyClass::window_size)
      .def_readwrite("max_speech_duration", &PyClass::max_speech_duration)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
