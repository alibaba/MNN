// sherpa-mnn/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SESSION_H_
#define SHERPA_ONNX_CSRC_SESSION_H_

#include <string>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-lm-config.h"
#include "sherpa-mnn/csrc/online-lm-config.h"
#include "sherpa-mnn/csrc/online-model-config.h"

namespace sherpa_mnn {

MNNConfig GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config = nullptr);

MNNConfig GetSessionOptions(const OfflineLMConfig &config);
MNNConfig GetSessionOptions(const OnlineLMConfig &config);

MNNConfig GetSessionOptions(const OnlineModelConfig &config);

MNNConfig GetSessionOptions(const OnlineModelConfig &config,
                                      const std::string &model_type);

MNNConfig GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str);

template <typename T>
MNNConfig GetSessionOptions(const T &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SESSION_H_
