// sherpa-mnn/csrc/spoken-language-identification-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_

#include <memory>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/spoken-language-identification.h"

namespace sherpa_mnn {

class SpokenLanguageIdentificationImpl {
 public:
  virtual ~SpokenLanguageIdentificationImpl() = default;

  static std::unique_ptr<SpokenLanguageIdentificationImpl> Create(
      const SpokenLanguageIdentificationConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<SpokenLanguageIdentificationImpl> Create(
      AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config);
#endif

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual std::string Compute(OfflineStream *s) const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_
