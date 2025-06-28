// sherpa-mnn/python/csrc/online-recongizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/online-recognizer.h"

namespace sherpa_mnn {

static void PybindOnlineRecognizerResult(py::module *m) {
  using PyClass = OnlineRecognizerResult;
  py::class_<PyClass>(*m, "OnlineRecognizerResult")
      .def_property_readonly(
          "text",
          [](PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.text.c_str(),
                                                self.text.size(), "ignore"));
          })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<std::string> { return self.tokens; })
      .def_property_readonly(
          "start_time", [](PyClass &self) -> float { return self.start_time; })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; })
      .def_property_readonly(
          "ys_probs",
          [](PyClass &self) -> std::vector<float> { return self.ys_probs; })
      .def_property_readonly(
          "lm_probs",
          [](PyClass &self) -> std::vector<float> { return self.lm_probs; })
      .def_property_readonly("context_scores",
                             [](PyClass &self) -> std::vector<float> {
                               return self.context_scores;
                             })
      .def_property_readonly(
          "segment", [](PyClass &self) -> int32_t { return self.segment; })
      .def_property_readonly(
          "words",
          [](PyClass &self) -> std::vector<int32_t> { return self.words; })
      .def_property_readonly(
          "is_final", [](PyClass &self) -> bool { return self.is_final; })
      .def("__str__", &PyClass::AsJsonString,
           py::call_guard<py::gil_scoped_release>())
      .def("as_json_string", &PyClass::AsJsonString,
           py::call_guard<py::gil_scoped_release>());
}

static void PybindOnlineRecognizerConfig(py::module *m) {
  using PyClass = OnlineRecognizerConfig;
  py::class_<PyClass>(*m, "OnlineRecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &, const OnlineModelConfig &,
                    const OnlineLMConfig &, const EndpointConfig &,
                    const OnlineCtcFstDecoderConfig &, bool,
                    const std::string &, int32_t, const std::string &, float,
                    float, float, const std::string &, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OnlineLMConfig(),
           py::arg("endpoint_config") = EndpointConfig(),
           py::arg("ctc_fst_decoder_config") = OnlineCtcFstDecoderConfig(),
           py::arg("enable_endpoint"), py::arg("decoding_method"),
           py::arg("max_active_paths") = 4, py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 0, py::arg("blank_penalty") = 0.0,
           py::arg("temperature_scale") = 2.0, py::arg("rule_fsts") = "",
           py::arg("rule_fars") = "")
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("endpoint_config", &PyClass::endpoint_config)
      .def_readwrite("ctc_fst_decoder_config", &PyClass::ctc_fst_decoder_config)
      .def_readwrite("enable_endpoint", &PyClass::enable_endpoint)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def_readwrite("temperature_scale", &PyClass::temperature_scale)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlineRecognizer(py::module *m) {
  PybindOnlineRecognizerResult(m);
  PybindOnlineRecognizerConfig(m);

  using PyClass = OnlineRecognizer;
  py::class_<PyClass>(*m, "OnlineRecognizer")
      .def(py::init<const OnlineRecognizerConfig &>(), py::arg("config"),
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
      .def("is_ready", &PyClass::IsReady,
           py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult,
           py::call_guard<py::gil_scoped_release>())
      .def("is_endpoint", &PyClass::IsEndpoint,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
