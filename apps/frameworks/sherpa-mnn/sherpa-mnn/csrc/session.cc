// sherpa-mnn/csrc/session.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/session.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/provider.h"


#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
#include "dml_provider_factory.h"  // NOLINT
#endif

namespace sherpa_mnn {


MNNConfig GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config /*= nullptr*/) {
  MNN::ScheduleConfig config;
  config.numThread = num_threads;
  MNN::BackendConfig bnConfig;
  bnConfig.memory = MNN::BackendConfig::Memory_Low;
  config.backendConfig = &bnConfig;
  MNNConfig sess_opts;
  sess_opts.pManager.reset(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
  sess_opts.pConfig.rearrange = true;
  return sess_opts;
}

MNNConfig GetSessionOptions(const OnlineModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads,
                               config.provider_config.provider,
                               &config.provider_config);
}

MNNConfig GetSessionOptions(const OnlineModelConfig &config,
                                      const std::string &model_type) {
  /*
    Transducer models : Only encoder will run with tensorrt,
                        decoder and joiner will run with cuda
  */
  if (config.provider_config.provider == "trt" &&
      (model_type == "decoder" || model_type == "joiner")) {
    return GetSessionOptionsImpl(config.num_threads, "cuda",
                                 &config.provider_config);
  }
  return GetSessionOptionsImpl(config.num_threads,
                               config.provider_config.provider,
                               &config.provider_config);
}

MNNConfig GetSessionOptions(const OfflineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

MNNConfig GetSessionOptions(const OnlineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

MNNConfig GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str) {
  return GetSessionOptionsImpl(num_threads, provider_str);
}

}  // namespace sherpa_mnn
