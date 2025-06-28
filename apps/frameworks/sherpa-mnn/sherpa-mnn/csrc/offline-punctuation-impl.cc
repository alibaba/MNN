// sherpa-mnn/csrc/offline-punctuation-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-punctuation-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-punctuation-ct-transformer-impl.h"

namespace sherpa_mnn {

std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    const OfflinePunctuationConfig &config) {
  if (!config.model.ct_transformer.empty()) {
    return std::make_unique<OfflinePunctuationCtTransformerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a punctuation model! Return a null pointer");
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    AAssetManager *mgr, const OfflinePunctuationConfig &config) {
  if (!config.model.ct_transformer.empty()) {
    return std::make_unique<OfflinePunctuationCtTransformerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a punctuation model! Return a null pointer");
  return nullptr;
}
#endif

}  // namespace sherpa_mnn
