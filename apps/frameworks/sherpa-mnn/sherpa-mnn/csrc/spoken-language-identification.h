// sherpa-mnn/csrc/spoken-language-identification.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_

#include <memory>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/offline-stream.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct SpokenLanguageIdentificationWhisperConfig {
  // Requires a multi-lingual whisper model.
  // That is, it supports only tiny, base, small, medium, large.
  // Note: It does NOT support tiny.en, base.en, small.en, medium.en
  std::string encoder;
  std::string decoder;

  // Number of tail padding frames.
  //
  // Since we remove the 30-second constraint, we need to add some paddings
  // at the end.
  //
  // Recommended values:
  //   - 50 for English models
  //   - 300 for multilingual models
  int32_t tail_paddings = -1;

  SpokenLanguageIdentificationWhisperConfig() = default;

  SpokenLanguageIdentificationWhisperConfig(const std::string &encoder,
                                            const std::string &decoder,
                                            int32_t tail_paddings)
      : encoder(encoder), decoder(decoder), tail_paddings(tail_paddings) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

struct SpokenLanguageIdentificationConfig {
  SpokenLanguageIdentificationWhisperConfig whisper;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  SpokenLanguageIdentificationConfig() = default;

  SpokenLanguageIdentificationConfig(
      const SpokenLanguageIdentificationWhisperConfig &whisper,
      int32_t num_threads, bool debug, const std::string &provider)
      : whisper(whisper),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class SpokenLanguageIdentificationImpl;

class SpokenLanguageIdentification {
 public:
  explicit SpokenLanguageIdentification(
      const SpokenLanguageIdentificationConfig &config);

#if __ANDROID_API__ >= 9
  SpokenLanguageIdentification(
      AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config);
#endif

  ~SpokenLanguageIdentification();

  // Create a stream to accept audio samples and compute features
  std::unique_ptr<OfflineStream> CreateStream() const;

  // Return a string containing the language, e.g., en, zh, de,
  // etc.
  // Note: en is for English, zh is for Chinese, de is for German, etc.
  std::string Compute(OfflineStream *s) const;

 private:
  std::unique_ptr<SpokenLanguageIdentificationImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_
