// sherpa-mnn/csrc/audio-tagging-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/offline-zipformer-audio-tagging-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct AudioTaggingModelConfig {
  struct OfflineZipformerAudioTaggingModelConfig zipformer;
  std::string ced;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  AudioTaggingModelConfig() = default;

  AudioTaggingModelConfig(
      const OfflineZipformerAudioTaggingModelConfig &zipformer,
      const std::string &ced, int32_t num_threads, bool debug,
      const std::string &provider)
      : zipformer(zipformer),
        ced(ced),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_
