// sherpa-mnn/csrc/offline-tts-matcha-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineTtsMatchaModelConfig {
  std::string acoustic_model;
  std::string vocoder;
  std::string lexicon;
  std::string tokens;

  // If data_dir is given, lexicon is ignored
  // data_dir is for piper-phonemizer, which uses espeak-ng
  std::string data_dir;

  // Used for Chinese TTS models using jieba
  std::string dict_dir;

  float noise_scale = 1;
  float length_scale = 1;

  OfflineTtsMatchaModelConfig() = default;

  OfflineTtsMatchaModelConfig(const std::string &acoustic_model,
                              const std::string &vocoder,
                              const std::string &lexicon,
                              const std::string &tokens,
                              const std::string &data_dir,
                              const std::string &dict_dir,
                              float noise_scale = 1.0, float length_scale = 1)
      : acoustic_model(acoustic_model),
        vocoder(vocoder),
        lexicon(lexicon),
        tokens(tokens),
        data_dir(data_dir),
        dict_dir(dict_dir),
        noise_scale(noise_scale),
        length_scale(length_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_CONFIG_H_
