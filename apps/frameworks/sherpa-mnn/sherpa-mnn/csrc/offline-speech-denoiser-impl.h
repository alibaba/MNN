// sherpa-mnn/csrc/offline-speaker-speech-denoiser-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_

#include <memory>

#include "sherpa-mnn/csrc/offline-speech-denoiser.h"

namespace sherpa_mnn {

class OfflineSpeechDenoiserImpl {
 public:
  virtual ~OfflineSpeechDenoiserImpl() = default;

  static std::unique_ptr<OfflineSpeechDenoiserImpl> Create(
      const OfflineSpeechDenoiserConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineSpeechDenoiserImpl> Create(
      Manager *mgr, const OfflineSpeechDenoiserConfig &config);

  virtual DenoisedAudio Run(const float *samples, int32_t n,
                            int32_t sample_rate) const = 0;

  virtual int32_t GetSampleRate() const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_
