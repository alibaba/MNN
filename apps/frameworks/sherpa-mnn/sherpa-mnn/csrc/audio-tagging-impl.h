// sherpa-mnn/csrc/audio-tagging-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/audio-tagging.h"

namespace sherpa_mnn {

class AudioTaggingImpl {
 public:
  virtual ~AudioTaggingImpl() = default;

  static std::unique_ptr<AudioTaggingImpl> Create(
      const AudioTaggingConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<AudioTaggingImpl> Create(
      AAssetManager *mgr, const AudioTaggingConfig &config);
#endif

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual std::vector<AudioEvent> Compute(OfflineStream *s,
                                          int32_t top_k = -1) const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_
