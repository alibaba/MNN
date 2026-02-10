// sherpa-mnn/python/csrc/features.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/features.h"

#include "sherpa-mnn/csrc/features.h"

namespace sherpa_mnn {

static void PybindFeatureExtractorConfig(py::module *m) {
  using PyClass = FeatureExtractorConfig;
  py::class_<PyClass>(*m, "FeatureExtractorConfig")
      .def(py::init<int32_t, int32_t, float, float, float, bool, bool>(),
           py::arg("sampling_rate") = 16000,
           py::arg("feature_dim") = 80,
           py::arg("low_freq") = 20.0f,
           py::arg("high_freq") = -400.0f,
           py::arg("dither") = 0.0f,
           py::arg("normalize_samples") = true,
           py::arg("snip_edges") = false)
      .def_readwrite("sampling_rate", &PyClass::sampling_rate)
      .def_readwrite("feature_dim", &PyClass::feature_dim)
      .def_readwrite("low_freq", &PyClass::low_freq)
      .def_readwrite("high_freq", &PyClass::high_freq)
      .def_readwrite("dither", &PyClass::dither)
      .def_readwrite("normalize_samples", &PyClass::normalize_samples)
      .def_readwrite("snip_edges", &PyClass::snip_edges)
      .def("__str__", &PyClass::ToString);
}

void PybindFeatures(py::module *m) { PybindFeatureExtractorConfig(m); }

}  // namespace sherpa_mnn
