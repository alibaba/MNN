// sherpa-mnn/csrc/voice-activity-detector.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#include "sherpa-mnn/csrc/vad-model-config.h"

namespace sherpa_mnn {

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
};

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

  template <typename Manager>
  VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  bool Empty() const;
  void Pop();
  void Clear();
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  void Reset() const;

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush() const;

  const VadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
