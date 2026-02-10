// sherpa-mnn/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-mnn/python/csrc/offline-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-recognizer.h"

namespace sherpa_mnn {

static void PybindOfflineRecognizerConfig(py::module *m) {
  using PyClass = OfflineRecognizerConfig;
  py::class_<PyClass>(*m, "OfflineRecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &, const OfflineModelConfig &,
                    const OfflineLMConfig &, const OfflineCtcFstDecoderConfig &,
                    const std::string &, int32_t, const std::string &, float,
                    float, const std::string &, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OfflineLMConfig(),
           py::arg("ctc_fst_decoder_config") = OfflineCtcFstDecoderConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("max_active_paths") = 4, py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 1.5, py::arg("blank_penalty") = 0.0,
           py::arg("rule_fsts") = "", py::arg("rule_fars") = "")
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("ctc_fst_decoder_config", &PyClass::ctc_fst_decoder_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init<const OfflineRecognizerConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &hotwords) {
            return self.CreateStream(hotwords);
          },
          py::arg("hotwords"), py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](const PyClass &self, std::vector<OfflineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
