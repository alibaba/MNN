// sherpa-mnn/csrc/offline-punctuation.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/offline-punctuation-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflinePunctuationConfig {
  OfflinePunctuationModelConfig model;

  OfflinePunctuationConfig() = default;

  explicit OfflinePunctuationConfig(const OfflinePunctuationModelConfig &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflinePunctuationImpl;

class OfflinePunctuation {
 public:
  explicit OfflinePunctuation(const OfflinePunctuationConfig &config);

#if __ANDROID_API__ >= 9
  OfflinePunctuation(AAssetManager *mgr,
                     const OfflinePunctuationConfig &config);
#endif

  ~OfflinePunctuation();

  // Add punctuation to the input text and return it.
  std::string AddPunctuation(const std::string &text) const;

 private:
  std::unique_ptr<OfflinePunctuationImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_H_
