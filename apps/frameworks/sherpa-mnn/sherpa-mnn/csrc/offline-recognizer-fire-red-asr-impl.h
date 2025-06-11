// sherpa-mnn/csrc/offline-recognizer-fire-red-asr-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/offline-fire-red-asr-decoder.h"
#include "sherpa-mnn/csrc/offline-fire-red-asr-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/offline-fire-red-asr-model.h"
#include "sherpa-mnn/csrc/offline-model-config.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/transpose.h"

namespace sherpa_mnn {

static OfflineRecognitionResult Convert(
    const OfflineFireRedAsrDecoderResult &src, const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    if (!sym_table.Contains(i)) {
      continue;
    }

    const auto &s = sym_table[i];
    text += s;
    r.tokens.push_back(s);
  }

  r.text = text;

  return r;
}

class OfflineRecognizerFireRedAsrImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerFireRedAsrImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineFireRedAsrModel>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerFireRedAsrImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineFireRedAsrModel>(mgr,
                                                        config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineFireRedAsrGreedySearchDecoder>(model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for FireRedAsr. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &meta_data = model_->metaData();

    config_.feat_config.normalize_samples = false;
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // batch decoding is not implemented yet
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    ApplyCMVN(&f);

    int num_frames = f.size() / feat_dim;

    std::array<int, 3> shape{1, num_frames, feat_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int len_shape = 1;
    MNN::Express::VARP x_len =
        MNNUtilsCreateTensor(memory_info, &num_frames, 1, &len_shape, 1);

    auto cross_kv = model_->ForwardEncoder(std::move(x), std::move(x_len));

    auto results =
        decoder_->Decode(std::move(cross_kv.first), std::move(cross_kv.second));

    auto r = Convert(results[0], symbol_table_);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    s->SetResult(r);
  }

  void ApplyCMVN(std::vector<float> *v) const {
    const auto &meta_data = model_->metaData();
    const auto &mean = meta_data.mean;
    const auto &inv_stddev = meta_data.inv_stddev;
    int32_t feat_dim = static_cast<int32_t>(mean.size());
    int32_t num_frames = static_cast<int32_t>(v->size()) / feat_dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != feat_dim; ++k) {
        p[k] = (p[k] - mean[k]) * inv_stddev[k];
      }

      p += feat_dim;
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineFireRedAsrModel> model_;
  std::unique_ptr<OfflineFireRedAsrDecoder> decoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_
