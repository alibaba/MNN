// sherpa-mnn/csrc/offline-speech-denoiser.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct DenoisedAudio {
  std::vector<float> samples;
  int32_t sample_rate;
};

struct OfflineSpeechDenoiserConfig {
  OfflineSpeechDenoiserModelConfig model;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflineSpeechDenoiserImpl;

class OfflineSpeechDenoiser {
 public:
  explicit OfflineSpeechDenoiser(const OfflineSpeechDenoiserConfig &config);
  ~OfflineSpeechDenoiser();

  template <typename Manager>
  OfflineSpeechDenoiser(Manager *mgr,
                        const OfflineSpeechDenoiserConfig &config);

  /*
   * @param samples 1-D array of audio samples. Each sample is in the
   *                range [-1, 1].
   * @param n Number of samples
   * @param sample_rate Sample rate of the input samples
   *
   */
  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) const;

  /*
   * Return the sample rate of the denoised audio
   */
  int32_t GetSampleRate() const;

 private:
  std::unique_ptr<OfflineSpeechDenoiserImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_H_
