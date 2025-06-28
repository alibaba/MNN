// sherpa-mnn/python/csrc/keyword-spotter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/keyword-spotter.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/keyword-spotter.h"

namespace sherpa_mnn {

static void PybindKeywordResult(py::module *m) {
  using PyClass = KeywordResult;
  py::class_<PyClass>(*m, "KeywordResult")
      .def_property_readonly(
          "keyword",
          [](PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.keyword.c_str(),
                                                self.keyword.size(), "ignore"));
          })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<std::string> { return self.tokens; })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; });
}

static void PybindKeywordSpotterConfig(py::module *m) {
  using PyClass = KeywordSpotterConfig;
  py::class_<PyClass>(*m, "KeywordSpotterConfig")
      .def(py::init<const FeatureExtractorConfig &, const OnlineModelConfig &,
                    int32_t, int32_t, float, float, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("max_active_paths") = 4, py::arg("num_trailing_blanks") = 1,
           py::arg("keywords_score") = 1.0,
           py::arg("keywords_threshold") = 0.25, py::arg("keywords_file") = "")
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("num_trailing_blanks", &PyClass::num_trailing_blanks)
      .def_readwrite("keywords_score", &PyClass::keywords_score)
      .def_readwrite("keywords_threshold", &PyClass::keywords_threshold)
      .def_readwrite("keywords_file", &PyClass::keywords_file)
      .def("__str__", &PyClass::ToString);
}

void PybindKeywordSpotter(py::module *m) {
  PybindKeywordResult(m);
  PybindKeywordSpotterConfig(m);

  using PyClass = KeywordSpotter;
  py::class_<PyClass>(*m, "KeywordSpotter")
      .def(py::init<const KeywordSpotterConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &keywords) {
            return self.CreateStream(keywords);
          },
          py::arg("keywords"), py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
