// sherpa-mnn/csrc/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speaker-diarization.h"

#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/offline-speaker-diarization-impl.h"

namespace sherpa_mnn {

void OfflineSpeakerDiarizationConfig::Register(ParseOptions *po) {
  ParseOptions po_segmentation("segmentation", po);
  segmentation.Register(&po_segmentation);

  ParseOptions po_embedding("embedding", po);
  embedding.Register(&po_embedding);

  ParseOptions po_clustering("clustering", po);
  clustering.Register(&po_clustering);

  po->Register("min-duration-on", &min_duration_on,
               "if a segment is less than this value, then it is discarded. "
               "Set it to 0 so that no segment is discarded");

  po->Register("min-duration-off", &min_duration_off,
               "if the gap between to segments of the same speaker is less "
               "than this value, then these two segments are merged into a "
               "single segment. We do it recursively.");
}

bool OfflineSpeakerDiarizationConfig::Validate() const {
  if (!segmentation.Validate()) {
    return false;
  }

  if (!embedding.Validate()) {
    return false;
  }

  if (!clustering.Validate()) {
    return false;
  }

  if (min_duration_on < 0) {
    SHERPA_ONNX_LOGE("min_duration_on %.3f is negative", min_duration_on);
    return false;
  }

  if (min_duration_off < 0) {
    SHERPA_ONNX_LOGE("min_duration_off %.3f is negative", min_duration_off);
    return false;
  }

  return true;
}

std::string OfflineSpeakerDiarizationConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerDiarizationConfig(";
  os << "segmentation=" << segmentation.ToString() << ", ";
  os << "embedding=" << embedding.ToString() << ", ";
  os << "clustering=" << clustering.ToString() << ", ";
  os << "min_duration_on=" << min_duration_on << ", ";
  os << "min_duration_off=" << min_duration_off << ")";

  return os.str();
}

OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    const OfflineSpeakerDiarizationConfig &config)
    : impl_(OfflineSpeakerDiarizationImpl::Create(config)) {}

template <typename Manager>
OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
    : impl_(OfflineSpeakerDiarizationImpl::Create(mgr, config)) {}

OfflineSpeakerDiarization::~OfflineSpeakerDiarization() = default;

int32_t OfflineSpeakerDiarization::SampleRate() const {
  return impl_->SampleRate();
}

void OfflineSpeakerDiarization::SetConfig(
    const OfflineSpeakerDiarizationConfig &config) {
  impl_->SetConfig(config);
}

OfflineSpeakerDiarizationResult OfflineSpeakerDiarization::Process(
    const float *audio, int32_t n,
    OfflineSpeakerDiarizationProgressCallback callback /*= nullptr*/,
    void *callback_arg /*= nullptr*/) const {
  return impl_->Process(audio, n, std::move(callback), callback_arg);
}

#if __ANDROID_API__ >= 9
template OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    AAssetManager *mgr, const OfflineSpeakerDiarizationConfig &config);
#endif

#if __OHOS__
template OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    NativeResourceManager *mgr, const OfflineSpeakerDiarizationConfig &config);
#endif

}  // namespace sherpa_mnn
