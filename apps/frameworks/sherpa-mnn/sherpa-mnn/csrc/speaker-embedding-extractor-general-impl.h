// sherpa-mnn/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-mnn/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-mnn/csrc/speaker-embedding-extractor-model.h"

namespace sherpa_mnn {

class SpeakerEmbeddingExtractorGeneralImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorGeneralImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorGeneralImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.normalize_samples = meta_data.normalize_samples;

    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
    if (num_frames <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %{public}d",
          num_frames);
#else
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %d",
          num_frames);
#endif
      return {};
    }

    std::vector<float> features =
        s->GetFrames(s->GetNumProcessedFrames(), num_frames);

    s->GetNumProcessedFrames() += num_frames;

    int32_t feat_dim = features.size() / num_frames;

    const auto &meta_data = model_.GetMetaData();
    if (!meta_data.feature_normalize_type.empty()) {
      if (meta_data.feature_normalize_type == "global-mean") {
        SubtractGlobalMean(features.data(), num_frames, feat_dim);
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                         meta_data.feature_normalize_type.c_str());
#else
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                         meta_data.feature_normalize_type.c_str());
#endif
        exit(-1);
      }
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{1, num_frames, feat_dim};
    MNN::Express::VARP x =
        MNNUtilsCreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());
    MNN::Express::VARP embedding = model_.Compute(std::move(x));
    std::vector<int> embedding_shape =
        embedding->getInfo()->dim;

    std::vector<float> ans(embedding_shape[1]);
    std::copy(embedding->readMap<float>(),
              embedding->readMap<float>() + ans.size(), ans.begin());

    return ans;
  }

 private:
  void SubtractGlobalMean(float *p, int32_t num_frames,
                          int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    m = m.rowwise() - m.colwise().mean();
  }

 private:
  SpeakerEmbeddingExtractorModel model_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
