// sherpa-mnn/csrc/vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/vad-model.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/silero-vad-model.h"

namespace sherpa_mnn {

std::unique_ptr<VadModel> VadModel::Create(const VadModelConfig &config) {
  // TODO(fangjun): Support other VAD models.
  return std::make_unique<SileroVadModel>(config);
}

template <typename Manager>
std::unique_ptr<VadModel> VadModel::Create(Manager *mgr,
                                           const VadModelConfig &config) {
  // TODO(fangjun): Support other VAD models.
  return std::make_unique<SileroVadModel>(mgr, config);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<VadModel> VadModel::Create(
    AAssetManager *mgr, const VadModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<VadModel> VadModel::Create(
    NativeResourceManager *mgr, const VadModelConfig &config);
#endif
}  // namespace sherpa_mnn
