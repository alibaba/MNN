// sherpa-mnn/csrc/provider-config.h
//
// Copyright (c)  2024  Uniphore (Author: Manickavela)

#ifndef SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_
#define SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_

#include <string>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct CudaConfig {
  int32_t cudnn_conv_algo_search = 0;

  CudaConfig() = default;
  explicit CudaConfig(int32_t cudnn_conv_algo_search)
      : cudnn_conv_algo_search(cudnn_conv_algo_search) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct TensorrtConfig {
  int trt_max_workspace_size = 2147483647;
  int32_t trt_max_partition_iterations = 10;
  int32_t trt_min_subgraph_size = 5;
  bool trt_fp16_enable = true;
  bool trt_detailed_build_log = false;
  bool trt_engine_cache_enable = true;
  bool trt_timing_cache_enable = true;
  std::string trt_engine_cache_path = ".";
  std::string trt_timing_cache_path = ".";
  bool trt_dump_subgraphs = false;

  TensorrtConfig() = default;
  TensorrtConfig(int trt_max_workspace_size,
                 int32_t trt_max_partition_iterations,
                 int32_t trt_min_subgraph_size, bool trt_fp16_enable,
                 bool trt_detailed_build_log, bool trt_engine_cache_enable,
                 bool trt_timing_cache_enable,
                 const std::string &trt_engine_cache_path,
                 const std::string &trt_timing_cache_path,
                 bool trt_dump_subgraphs)
      : trt_max_workspace_size(trt_max_workspace_size),
        trt_max_partition_iterations(trt_max_partition_iterations),
        trt_min_subgraph_size(trt_min_subgraph_size),
        trt_fp16_enable(trt_fp16_enable),
        trt_detailed_build_log(trt_detailed_build_log),
        trt_engine_cache_enable(trt_engine_cache_enable),
        trt_timing_cache_enable(trt_timing_cache_enable),
        trt_engine_cache_path(trt_engine_cache_path),
        trt_timing_cache_path(trt_timing_cache_path),
        trt_dump_subgraphs(trt_dump_subgraphs) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct ProviderConfig {
  TensorrtConfig trt_config;
  CudaConfig cuda_config;
  std::string provider = "cpu";
  int32_t device = 0;
  // device only used for cuda and trt

  ProviderConfig() = default;
  ProviderConfig(const std::string &provider, int32_t device)
      : provider(provider), device(device) {}
  ProviderConfig(const TensorrtConfig &trt_config,
                 const CudaConfig &cuda_config, const std::string &provider,
                 int32_t device)
      : trt_config(trt_config),
        cuda_config(cuda_config),
        provider(provider),
        device(device) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_
