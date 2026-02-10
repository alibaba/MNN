// sherpa-mnn/csrc/online-punctuation-impl.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_IMPL_H_

#include <memory>
#include <string>
#include <vector>
#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/online-punctuation.h"

namespace sherpa_mnn {

class OnlinePunctuationImpl {
 public:
  virtual ~OnlinePunctuationImpl() = default;

  static std::unique_ptr<OnlinePunctuationImpl> Create(
      const OnlinePunctuationConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OnlinePunctuationImpl> Create(
      AAssetManager *mgr, const OnlinePunctuationConfig &config);
#endif

  virtual std::string AddPunctuationWithCase(const std::string &text) const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_IMPL_H_
