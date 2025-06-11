// sherpa-mnn/csrc/online-ctc-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-ctc-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-nemo-ctc-model.h"
#include "sherpa-mnn/csrc/online-wenet-ctc-model.h"
#include "sherpa-mnn/csrc/online-zipformer2-ctc-model.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    const OnlineModelConfig &config) {
  if (!config.wenet_ctc.model.empty()) {
    return std::make_unique<OnlineWenetCtcModel>(config);
  } else if (!config.zipformer2_ctc.model.empty()) {
    return std::make_unique<OnlineZipformer2CtcModel>(config);
  } else if (!config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineNeMoCtcModel>(config);
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }
}

template <typename Manager>
std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    Manager *mgr, const OnlineModelConfig &config) {
  if (!config.wenet_ctc.model.empty()) {
    return std::make_unique<OnlineWenetCtcModel>(mgr, config);
  } else if (!config.zipformer2_ctc.model.empty()) {
    return std::make_unique<OnlineZipformer2CtcModel>(mgr, config);
  } else if (!config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineNeMoCtcModel>(mgr, config);
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
