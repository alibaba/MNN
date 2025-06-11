// sherpa-mnn/csrc/silero-vad-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct SileroVadModelConfig {
  std::string model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold = 0.5;

  float min_silence_duration = 0.5;  // in seconds

  float min_speech_duration = 0.25;  // in seconds

  // 512, 1024, 1536 samples for 16000 Hz
  // 256, 512, 768 samples for 800 Hz
  int32_t window_size = 512;  // in samples

  // If a speech segment is longer than this value, then we increase
  // the threshold to 0.9. After finishing detecting the segment,
  // the threshold value is reset to its original value.
  float max_speech_duration = 20;  // in seconds

  SileroVadModelConfig() = default;

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_
