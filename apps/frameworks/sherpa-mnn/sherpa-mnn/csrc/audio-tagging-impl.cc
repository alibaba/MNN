// sherpa-mnn/csrc/audio-tagging-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/audio-tagging-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/audio-tagging-ced-impl.h"
#include "sherpa-mnn/csrc/audio-tagging-zipformer-impl.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    const AudioTaggingConfig &config) {
  if (!config.model.zipformer.model.empty()) {
    return std::make_unique<AudioTaggingZipformerImpl>(config);
  } else if (!config.model.ced.empty()) {
    return std::make_unique<AudioTaggingCEDImpl>(config);
  }

  SHERPA_ONNX_LOGE(
      "Please specify an audio tagging model! Return a null pointer");
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    AAssetManager *mgr, const AudioTaggingConfig &config) {
  if (!config.model.zipformer.model.empty()) {
    return std::make_unique<AudioTaggingZipformerImpl>(mgr, config);
  } else if (!config.model.ced.empty()) {
    return std::make_unique<AudioTaggingCEDImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE(
      "Please specify an audio tagging model! Return a null pointer");
  return nullptr;
}
#endif

}  // namespace sherpa_mnn
