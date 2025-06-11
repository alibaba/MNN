// sherpa-mnn/python/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-mnn/python/csrc/offline-tts.h"

#include <algorithm>
#include <string>

#include "sherpa-mnn/csrc/offline-tts.h"
#include "sherpa-mnn/python/csrc/offline-tts-model-config.h"

namespace sherpa_mnn {

static void PybindGeneratedAudio(py::module *m) {
  using PyClass = GeneratedAudio;
  py::class_<PyClass>(*m, "GeneratedAudio")
      .def(py::init<>())
      .def_readwrite("samples", &PyClass::samples)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def("__str__", [](PyClass &self) {
        std::ostringstream os;
        os << "GeneratedAudio(sample_rate=" << self.sample_rate << ", ";
        os << "num_samples=" << self.samples.size() << ")";
        return os.str();
      });
}

static void PybindOfflineTtsConfig(py::module *m) {
  PybindOfflineTtsModelConfig(m);

  using PyClass = OfflineTtsConfig;
  py::class_<PyClass>(*m, "OfflineTtsConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsModelConfig &, const std::string &,
                    const std::string &, int32_t, float>(),
           py::arg("model"), py::arg("rule_fsts") = "",
           py::arg("rule_fars") = "", py::arg("max_num_sentences") = 2,
           py::arg("silence_scale") = 0.2)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("max_num_sentences", &PyClass::max_num_sentences)
      .def_readwrite("silence_scale", &PyClass::silence_scale)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineTts(py::module *m) {
  PybindOfflineTtsConfig(m);
  PybindGeneratedAudio(m);

  using PyClass = OfflineTts;
  py::class_<PyClass>(*m, "OfflineTts")
      .def(py::init<const OfflineTtsConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers)
      .def(
          "generate",
          [](const PyClass &self, const std::string &text, int64_t sid,
             float speed,
             std::function<int32_t(py::array_t<float>, float)> callback)
              -> GeneratedAudio {
            if (!callback) {
              return self.Generate(text, sid, speed);
            }

            std::function<int32_t(const float *, int32_t, float)>
                callback_wrapper = [callback](const float *samples, int32_t n,
                                              float progress) {
                  // CAUTION(fangjun): we have to copy samples since it is
                  // freed once the call back returns.

                  pybind11::gil_scoped_acquire acquire;

                  pybind11::array_t<float> array(n);
                  py::buffer_info buf = array.request();
                  auto p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);
                  return callback(array, progress);
                };

            return self.Generate(text, sid, speed, callback_wrapper);
          },
          py::arg("text"), py::arg("sid") = 0, py::arg("speed") = 1.0,
          py::arg("callback") = py::none(),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
