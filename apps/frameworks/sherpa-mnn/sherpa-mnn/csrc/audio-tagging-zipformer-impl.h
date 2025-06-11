// sherpa-mnn/csrc/audio-tagging-zipformer-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_

#include <assert.h>

#include <memory>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/audio-tagging-impl.h"
#include "sherpa-mnn/csrc/audio-tagging-label-file.h"
#include "sherpa-mnn/csrc/audio-tagging.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/math.h"
#include "sherpa-mnn/csrc/offline-zipformer-audio-tagging-model.h"

namespace sherpa_mnn {

class AudioTaggingZipformerImpl : public AudioTaggingImpl {
 public:
  explicit AudioTaggingZipformerImpl(const AudioTaggingConfig &config)
      : config_(config), model_(config.model), labels_(config.labels) {
    if (model_.NumEventClasses() != labels_.NumEventClasses()) {
      SHERPA_ONNX_LOGE("number of classes: %d (model) != %d (label file)",
                       model_.NumEventClasses(), labels_.NumEventClasses());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  explicit AudioTaggingZipformerImpl(AAssetManager *mgr,
                                     const AudioTaggingConfig &config)
      : config_(config),
        model_(mgr, config.model),
        labels_(mgr, config.labels) {
    if (model_.NumEventClasses() != labels_.NumEventClasses()) {
      SHERPA_ONNX_LOGE("number of classes: %d (model) != %d (label file)",
                       model_.NumEventClasses(), labels_.NumEventClasses());
      exit(-1);
    }
  }
#endif

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>();
  }

  std::vector<AudioEvent> Compute(OfflineStream *s,
                                  int32_t top_k = -1) const override {
    if (top_k < 0) {
      top_k = config_.top_k;
    }

    int32_t num_event_classes = model_.NumEventClasses();

    if (top_k > num_event_classes) {
      top_k = num_event_classes;
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    // WARNING(fangjun): It is fixed to 80 for all models from icefall
    int32_t feat_dim = 80;
    std::vector<float> f = s->GetFrames();

    int32_t num_frames = f.size() / feat_dim;

    assert(feat_dim * num_frames == static_cast<int32_t>(f.size()));

    std::array<int, 3> shape = {1, num_frames, feat_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int x_length_scalar = num_frames;
    std::array<int, 1> x_length_shape = {1};
    MNN::Express::VARP x_length =
        MNNUtilsCreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());

    MNN::Express::VARP probs = model_.Forward(std::move(x), std::move(x_length));

    const float *p = probs->readMap<float>();

    std::vector<int32_t> top_k_indexes = TopkIndex(p, num_event_classes, top_k);

    std::vector<AudioEvent> ans(top_k);

    int32_t i = 0;

    for (int32_t index : top_k_indexes) {
      ans[i].name = labels_.GetEventName(index);
      ans[i].index = index;
      ans[i].prob = p[index];
      i += 1;
    }

    return ans;
  }

 private:
  AudioTaggingConfig config_;
  OfflineZipformerAudioTaggingModel model_;
  AudioTaggingLabels labels_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_
