// sherpa-mnn/csrc/offline-tts-kokoro-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_H_

#include <memory>
#include <string>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-tts-kokoro-model-meta-data.h"
#include "sherpa-mnn/csrc/offline-tts-model-config.h"

namespace sherpa_mnn {

class OfflineTtsKokoroModel {
 public:
  ~OfflineTtsKokoroModel();

  explicit OfflineTtsKokoroModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsKokoroModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // Return a float32 tensor containing the mel
  // of shape (batch_size, mel_dim, num_frames)
  MNN::Express::VARP Run(MNN::Express::VARP x, int sid = 0, float speed = 1.0) const;

  const OfflineTtsKokoroModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_H_
