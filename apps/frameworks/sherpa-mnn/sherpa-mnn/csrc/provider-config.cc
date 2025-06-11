// sherpa-mnn/csrc/provider-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela)

#include "sherpa-mnn/csrc/provider-config.h"

#include <sstream>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void CudaConfig::Register(ParseOptions *po) {
  po->Register("cuda-cudnn-conv-algo-search", &cudnn_conv_algo_search,
               "CuDNN convolution algrorithm search");
}

bool CudaConfig::Validate() const {
  if (cudnn_conv_algo_search < 1 || cudnn_conv_algo_search > 3) {
    SHERPA_ONNX_LOGE(
        "cudnn_conv_algo_search: '%d' is not a valid option."
        "Options : [1,3]. Check OnnxRT docs",
        cudnn_conv_algo_search);
    return false;
  }
  return true;
}

std::string CudaConfig::ToString() const {
  std::ostringstream os;

  os << "CudaConfig(";
  os << "cudnn_conv_algo_search=" << cudnn_conv_algo_search << ")";

  return os.str();
}

void TensorrtConfig::Register(ParseOptions *po) {
  po->Register("trt-max-workspace-size", &trt_max_workspace_size,
               "Set TensorRT EP GPU memory usage limit.");
  po->Register("trt-max-partition-iterations", &trt_max_partition_iterations,
               "Limit partitioning iterations for model conversion.");
  po->Register("trt-min-subgraph-size", &trt_min_subgraph_size,
               "Set minimum size for subgraphs in partitioning.");
  po->Register("trt-fp16-enable", &trt_fp16_enable,
               "Enable FP16 precision for faster performance.");
  po->Register("trt-detailed-build-log", &trt_detailed_build_log,
               "Enable detailed logging of build steps.");
  po->Register("trt-engine-cache-enable", &trt_engine_cache_enable,
               "Enable caching of TensorRT engines.");
  po->Register("trt-timing-cache-enable", &trt_timing_cache_enable,
               "Enable use of timing cache to speed up builds.");
  po->Register("trt-engine-cache-path", &trt_engine_cache_path,
               "Set path to store cached TensorRT engines.");
  po->Register("trt-timing-cache-path", &trt_timing_cache_path,
               "Set path for storing timing cache.");
  po->Register("trt-dump-subgraphs", &trt_dump_subgraphs,
               "Dump optimized subgraphs for debugging.");
}

bool TensorrtConfig::Validate() const {
  if (trt_max_workspace_size < 0) {
    std::ostringstream os;
    os << "trt_max_workspace_size: " << trt_max_workspace_size
       << " is not valid.";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return false;
  }
  if (trt_max_partition_iterations < 0) {
    SHERPA_ONNX_LOGE("trt_max_partition_iterations: %d is not valid.",
                     trt_max_partition_iterations);
    return false;
  }
  if (trt_min_subgraph_size < 0) {
    SHERPA_ONNX_LOGE("trt_min_subgraph_size: %d is not valid.",
                     trt_min_subgraph_size);
    return false;
  }

  return true;
}

std::string TensorrtConfig::ToString() const {
  std::ostringstream os;

  os << "TensorrtConfig(";
  os << "trt_max_workspace_size=" << trt_max_workspace_size << ", ";
  os << "trt_max_partition_iterations=" << trt_max_partition_iterations << ", ";
  os << "trt_min_subgraph_size=" << trt_min_subgraph_size << ", ";
  os << "trt_fp16_enable=\"" << (trt_fp16_enable ? "True" : "False") << "\", ";
  os << "trt_detailed_build_log=\""
     << (trt_detailed_build_log ? "True" : "False") << "\", ";
  os << "trt_engine_cache_enable=\""
     << (trt_engine_cache_enable ? "True" : "False") << "\", ";
  os << "trt_engine_cache_path=\"" << trt_engine_cache_path.c_str() << "\", ";
  os << "trt_timing_cache_enable=\""
     << (trt_timing_cache_enable ? "True" : "False") << "\", ";
  os << "trt_timing_cache_path=\"" << trt_timing_cache_path.c_str() << "\",";
  os << "trt_dump_subgraphs=\"" << (trt_dump_subgraphs ? "True" : "False")
     << "\" )";
  return os.str();
}

void ProviderConfig::Register(ParseOptions *po) {
  cuda_config.Register(po);
  trt_config.Register(po);

  po->Register("device", &device, "GPU device index for CUDA and Trt EP");
  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool ProviderConfig::Validate() const {
  if (device < 0) {
    SHERPA_ONNX_LOGE("device: '%d' is invalid.", device);
    return false;
  }

  if (provider == "cuda" && !cuda_config.Validate()) {
    return false;
  }

  if (provider == "trt" && !trt_config.Validate()) {
    return false;
  }

  return true;
}

std::string ProviderConfig::ToString() const {
  std::ostringstream os;

  os << "ProviderConfig(";
  os << "device=" << device << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "cuda_config=" << cuda_config.ToString() << ", ";
  os << "trt_config=" << trt_config.ToString() << ")";
  return os.str();
}

}  // namespace sherpa_mnn
