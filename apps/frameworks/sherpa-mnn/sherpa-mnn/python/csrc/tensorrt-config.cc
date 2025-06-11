// sherpa-mnn/python/csrc/tensorrt-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela A)

#include "sherpa-mnn/python/csrc/tensorrt-config.h"

#include <string>
#include <memory>
#include "sherpa-mnn/csrc/provider-config.h"

namespace sherpa_mnn {

void PybindTensorrtConfig(py::module *m) {
  using PyClass = TensorrtConfig;
  py::class_<PyClass>(*m, "TensorrtConfig")
        .def(py::init<>())
        .def(py::init([](int64_t trt_max_workspace_size,
                      int32_t trt_max_partition_iterations,
                      int32_t trt_min_subgraph_size,
                      bool trt_fp16_enable,
                      bool trt_detailed_build_log,
                      bool trt_engine_cache_enable,
                      bool trt_timing_cache_enable,
                      const std::string &trt_engine_cache_path,
                      const std::string &trt_timing_cache_path,
                      bool trt_dump_subgraphs) -> std::unique_ptr<PyClass> {
            auto ans = std::make_unique<PyClass>();

            ans->trt_max_workspace_size = trt_max_workspace_size;
            ans->trt_max_partition_iterations = trt_max_partition_iterations;
            ans->trt_min_subgraph_size = trt_min_subgraph_size;
            ans->trt_fp16_enable = trt_fp16_enable;
            ans->trt_detailed_build_log = trt_detailed_build_log;
            ans->trt_engine_cache_enable = trt_engine_cache_enable;
            ans->trt_timing_cache_enable = trt_timing_cache_enable;
            ans->trt_engine_cache_path = trt_engine_cache_path;
            ans->trt_timing_cache_path = trt_timing_cache_path;
            ans->trt_dump_subgraphs = trt_dump_subgraphs;

            return ans;
          }),
           py::arg("trt_max_workspace_size") = 2147483647,
           py::arg("trt_max_partition_iterations") = 10,
           py::arg("trt_min_subgraph_size") = 5,
           py::arg("trt_fp16_enable") = true,
           py::arg("trt_detailed_build_log") = false,
           py::arg("trt_engine_cache_enable") = true,
           py::arg("trt_timing_cache_enable") = true,
           py::arg("trt_engine_cache_path") = ".",
           py::arg("trt_timing_cache_path") = ".",
           py::arg("trt_dump_subgraphs") = false)

      .def_readwrite("trt_max_workspace_size",
          &PyClass::trt_max_workspace_size)
      .def_readwrite("trt_max_partition_iterations",
          &PyClass::trt_max_partition_iterations)
      .def_readwrite("trt_min_subgraph_size", &PyClass::trt_min_subgraph_size)
      .def_readwrite("trt_fp16_enable", &PyClass::trt_fp16_enable)
      .def_readwrite("trt_detailed_build_log",
          &PyClass::trt_detailed_build_log)
      .def_readwrite("trt_engine_cache_enable",
          &PyClass::trt_engine_cache_enable)
      .def_readwrite("trt_timing_cache_enable",
          &PyClass::trt_timing_cache_enable)
      .def_readwrite("trt_engine_cache_path", &PyClass::trt_engine_cache_path)
      .def_readwrite("trt_timing_cache_path", &PyClass::trt_timing_cache_path)
      .def_readwrite("trt_dump_subgraphs", &PyClass::trt_dump_subgraphs)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_mnn
