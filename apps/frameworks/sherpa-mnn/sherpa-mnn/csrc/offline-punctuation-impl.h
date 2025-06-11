// sherpa-mnn/csrc/offline-punctuation-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_

#include <memory>
#include <string>
#include <vector>
#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/offline-punctuation.h"

namespace sherpa_mnn {

class OfflinePunctuationImpl {
 public:
  virtual ~OfflinePunctuationImpl() = default;

  static std::unique_ptr<OfflinePunctuationImpl> Create(
      const OfflinePunctuationConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OfflinePunctuationImpl> Create(
      AAssetManager *mgr, const OfflinePunctuationConfig &config);
#endif

  virtual std::string AddPunctuation(const std::string &text) const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_
