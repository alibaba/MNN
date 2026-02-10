// sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
#include <memory>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model-meta-data.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser.h"

namespace sherpa_mnn {

class OfflineSpeechDenoiserGtcrnModel {
 public:
  ~OfflineSpeechDenoiserGtcrnModel();
  explicit OfflineSpeechDenoiserGtcrnModel(
      const OfflineSpeechDenoiserModelConfig &config);

  template <typename Manager>
  OfflineSpeechDenoiserGtcrnModel(
      Manager *mgr, const OfflineSpeechDenoiserModelConfig &config);

  using States = std::vector<MNN::Express::VARP>;

  States GetInitStates() const;

  std::pair<MNN::Express::VARP, States> Run(MNN::Express::VARP x, States states) const;

  const OfflineSpeechDenoiserGtcrnModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
