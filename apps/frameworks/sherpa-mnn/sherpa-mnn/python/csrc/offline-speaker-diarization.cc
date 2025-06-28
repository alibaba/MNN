// sherpa-mnn/python/csrc/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-speaker-diarization.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-speaker-diarization.h"
#include "sherpa-mnn/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model-config.h"

namespace sherpa_mnn {

static void PybindOfflineSpeakerSegmentationPyannoteModelConfig(py::module *m) {
  using PyClass = OfflineSpeakerSegmentationPyannoteModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeakerSegmentationPyannoteModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

static void PybindOfflineSpeakerSegmentationModelConfig(py::module *m) {
  PybindOfflineSpeakerSegmentationPyannoteModelConfig(m);

  using PyClass = OfflineSpeakerSegmentationModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeakerSegmentationModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeakerSegmentationPyannoteModelConfig &,
                    int32_t, bool, const std::string &>(),
           py::arg("pyannote"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("pyannote", &PyClass::pyannote)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

static void PybindOfflineSpeakerDiarizationConfig(py::module *m) {
  PybindOfflineSpeakerSegmentationModelConfig(m);

  using PyClass = OfflineSpeakerDiarizationConfig;
  py::class_<PyClass>(*m, "OfflineSpeakerDiarizationConfig")
      .def(py::init<const OfflineSpeakerSegmentationModelConfig &,
                    const SpeakerEmbeddingExtractorConfig &,
                    const FastClusteringConfig &, float, float>(),
           py::arg("segmentation"), py::arg("embedding"), py::arg("clustering"),
           py::arg("min_duration_on") = 0.3, py::arg("min_duration_off") = 0.5)
      .def_readwrite("segmentation", &PyClass::segmentation)
      .def_readwrite("embedding", &PyClass::embedding)
      .def_readwrite("clustering", &PyClass::clustering)
      .def_readwrite("min_duration_on", &PyClass::min_duration_on)
      .def_readwrite("min_duration_off", &PyClass::min_duration_off)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

void PybindOfflineSpeakerDiarization(py::module *m) {
  PybindOfflineSpeakerDiarizationConfig(m);

  using PyClass = OfflineSpeakerDiarization;
  py::class_<PyClass>(*m, "OfflineSpeakerDiarization")
      .def(py::init<const OfflineSpeakerDiarizationConfig &>(),
           py::arg("config"))
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      .def("set_config", &PyClass::SetConfig, py::arg("config"))
      .def(
          "process",
          [](const PyClass &self, const std::vector<float> samples,
             std::function<int32_t(int32_t, int32_t)> callback) {
            if (!callback) {
              return self.Process(samples.data(), samples.size());
            }

            std::function<int32_t(int32_t, int32_t, void *)> callback_wrapper =
                [callback](int32_t processed_chunks, int32_t num_chunks,
                           void *) -> int32_t {
              callback(processed_chunks, num_chunks);
              return 0;
            };

            return self.Process(samples.data(), samples.size(),
                                callback_wrapper);
          },
          py::arg("samples"), py::arg("callback") = py::none());
}

}  // namespace sherpa_mnn
