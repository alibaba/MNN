// sherpa-mnn/csrc/offline-speech-denoiser.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speech-denoiser.h"

#include "sherpa-mnn/csrc/offline-speech-denoiser-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_mnn {

void OfflineSpeechDenoiserConfig::Register(ParseOptions *po) {
  model.Register(po);
}

bool OfflineSpeechDenoiserConfig::Validate() const { return model.Validate(); }

std::string OfflineSpeechDenoiserConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserConfig(";
  os << "model=" << model.ToString() << ")";
  return os.str();
}

template <typename Manager>
OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    Manager *mgr, const OfflineSpeechDenoiserConfig &config)
    : impl_(OfflineSpeechDenoiserImpl::Create(mgr, config)) {}

OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    const OfflineSpeechDenoiserConfig &config)
    : impl_(OfflineSpeechDenoiserImpl::Create(config)) {}

OfflineSpeechDenoiser::~OfflineSpeechDenoiser() = default;

DenoisedAudio OfflineSpeechDenoiser::Run(const float *samples, int32_t n,
                                         int32_t sample_rate) const {
  return impl_->Run(samples, n, sample_rate);
}

int32_t OfflineSpeechDenoiser::GetSampleRate() const {
  return impl_->GetSampleRate();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    AAssetManager *mgr, const OfflineSpeechDenoiserConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserConfig &config);
#endif

}  // namespace sherpa_mnn
