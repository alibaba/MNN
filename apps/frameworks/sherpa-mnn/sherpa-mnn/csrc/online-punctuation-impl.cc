// sherpa-mnn/csrc/online-punctuation-impl.cc
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#include "sherpa-mnn/csrc/online-punctuation-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-punctuation-cnn-bilstm-impl.h"

namespace sherpa_mnn {

std::unique_ptr<OnlinePunctuationImpl> OnlinePunctuationImpl::Create(
    const OnlinePunctuationConfig &config) {
  if (!config.model.cnn_bilstm.empty() && !config.model.bpe_vocab.empty()) {
    return std::make_unique<OnlinePunctuationCNNBiLSTMImpl>(config);
  }

  SHERPA_ONNX_LOGE(
      "Please specify a punctuation model and bpe vocab! Return a null "
      "pointer");
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OnlinePunctuationImpl> OnlinePunctuationImpl::Create(
    AAssetManager *mgr, const OnlinePunctuationConfig &config) {
  if (!config.model.cnn_bilstm.empty() && !config.model.bpe_vocab.empty()) {
    return std::make_unique<OnlinePunctuationCNNBiLSTMImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE(
      "Please specify a punctuation model and bpe vocab! Return a null "
      "pointer");
  return nullptr;
}
#endif

}  // namespace sherpa_mnn
