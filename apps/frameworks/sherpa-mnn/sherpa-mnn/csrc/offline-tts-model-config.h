// sherpa-mnn/csrc/offline-tts-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/offline-tts-kokoro-model-config.h"
#include "sherpa-mnn/csrc/offline-tts-matcha-model-config.h"
#include "sherpa-mnn/csrc/offline-tts-vits-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;
  OfflineTtsMatchaModelConfig matcha;
  OfflineTtsKokoroModelConfig kokoro;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OfflineTtsModelConfig() = default;

  OfflineTtsModelConfig(const OfflineTtsVitsModelConfig &vits,
                        const OfflineTtsMatchaModelConfig &matcha,
                        const OfflineTtsKokoroModelConfig &kokoro,
                        int32_t num_threads, bool debug,
                        const std::string &provider)
      : vits(vits),
        matcha(matcha),
        kokoro(kokoro),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
