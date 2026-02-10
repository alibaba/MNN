// sherpa-mnn/csrc/audio-tagging-label-file.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_

#include <istream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

namespace sherpa_mnn {

class AudioTaggingLabels {
 public:
  explicit AudioTaggingLabels(const std::string &filename);
#if __ANDROID_API__ >= 9
  AudioTaggingLabels(AAssetManager *mgr, const std::string &filename);
#endif

  // Return the event name for the given index.
  // The returned reference is valid as long as this object is alive
  const std::string &GetEventName(int32_t index) const;
  int32_t NumEventClasses() const { return names_.size(); }

 private:
  void Init(std::istream &is);

 private:
  std::vector<std::string> names_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_
