// sherpa-mnn/python/csrc/provider-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela A)


#include "sherpa-mnn/python/csrc/provider-config.h"

#include <string>

#include "sherpa-mnn/csrc/provider-config.h"
#include "sherpa-mnn/python/csrc/cuda-config.h"
#include "sherpa-mnn/python/csrc/tensorrt-config.h"

namespace sherpa_mnn {

void PybindProviderConfig(py::module *m) {
  PybindCudaConfig(m);
  PybindTensorrtConfig(m);

  using PyClass = ProviderConfig;
  py::class_<PyClass>(*m, "ProviderConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t>(),
           py::arg("provider") = "cpu",
           py::arg("device") = 0)
      .def(py::init<const TensorrtConfig &, const CudaConfig &,
          const std::string &, int32_t>(),
           py::arg("trt_config") = TensorrtConfig{},
           py::arg("cuda_config") = CudaConfig{},
           py::arg("provider") = "cpu",
           py::arg("device") = 0)
      .def_readwrite("trt_config", &PyClass::trt_config)
      .def_readwrite("cuda_config", &PyClass::cuda_config)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("device", &PyClass::device)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}
}  // namespace sherpa_mnn
