// sherpa-mnn/csrc/offline-recognizer-paraformer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/offline-model-config.h"
#include "sherpa-mnn/csrc/offline-paraformer-decoder.h"
#include "sherpa-mnn/csrc/offline-paraformer-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/offline-paraformer-model.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/pad-sequence.h"
#include "sherpa-mnn/csrc/symbol-table.h"

namespace sherpa_mnn {

static OfflineRecognitionResult Convert(
    const OfflineParaformerDecoderResult &src, const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps = src.timestamps;

  std::string text;

  // When the current token ends with "@@" we set mergeable to true
  bool mergeable = false;

  for (int32_t i = 0; i != src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    r.tokens.push_back(sym);

    if ((sym.back() != '@') || (sym.size() > 2 && sym[sym.size() - 2] != '@')) {
      // sym does not end with "@@"
      const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
      if (p[0] < 0x80) {
        // an ascii
        if (mergeable) {
          mergeable = false;
          text.append(sym);
        } else {
          text.append(" ");
          text.append(sym);
        }
      } else {
        // not an ascii
        mergeable = false;

        if (i > 0) {
          const uint8_t p = reinterpret_cast<const uint8_t *>(
              sym_table[src.tokens[i - 1]].c_str())[0];
          if (p < 0x80) {
            // put a space between ascii and non-ascii
            text.append(" ");
          }
        }
        text.append(sym);
      }
    } else {
      // this sym ends with @@
      sym = std::string(sym.data(), sym.size() - 2);
      if (mergeable) {
        text.append(sym);
      } else {
        text.append(" ");
        text.append(sym);
        mergeable = true;
      }
    }
  }
  r.text = std::move(text);

  return r;
}

class OfflineRecognizerParaformerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParaformerImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModel>(config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      int32_t eos_id = symbol_table_["</s>"];
      decoder_ = std::make_unique<OfflineParaformerGreedySearchDecoder>(eos_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerParaformerImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModel>(mgr,
                                                        config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      int32_t eos_id = symbol_table_["</s>"];
      decoder_ = std::make_unique<OfflineParaformerGreedySearchDecoder>(eos_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    InitFeatConfig();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // 1. Apply LFR
    // 2. Apply CMVN
    //
    // Please refer to
    // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45555.pdf
    // for what LFR means
    //
    // "Lower Frame Rate Neural Network Acoustic Models"
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::vector<MNN::Express::VARP> features;
    features.reserve(n);

    int32_t feat_dim =
        config_.feat_config.feature_dim * model_->LfrWindowSize();

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int32_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> f = ss[i]->GetFrames();

      f = ApplyLFR(f);
      ApplyCMVN(&f);

      int32_t num_frames = f.size() / feat_dim;
      features_vec[i] = std::move(f);

      features_length_vec[i] = num_frames;

      std::array<int, 2> shape = {num_frames, feat_dim};

      MNN::Express::VARP x = MNNUtilsCreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<MNN::Express::VARP> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = features[i];
    }

    std::array<int, 1> features_length_shape = {n};
    MNN::Express::VARP x_length = MNNUtilsCreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    // Caution(fangjun): We cannot pad it with log(eps),
    // i.e., -23.025850929940457f
    MNN::Express::VARP x = PadSequence(model_->Allocator(), features_pointer, 0);

    std::vector<MNN::Express::VARP> t;
      t = model_->Forward(std::move(x), std::move(x_length));

    std::vector<OfflineParaformerDecoderResult> results;
    if (t.size() == 2) {
      results = decoder_->Decode(std::move(t[0]), std::move(t[1]));
    } else {
      results =
          decoder_->Decode(std::move(t[0]), std::move(t[1]), std::move(t[3]));
    }

    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_);
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig() {
    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    int32_t lfr_window_size = model_->LfrWindowSize();
    int32_t lfr_window_shift = model_->LfrWindowShift();
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void ApplyCMVN(std::vector<float> *v) const {
    const std::vector<float> &neg_mean = model_->NegativeMean();
    const std::vector<float> &inv_stddev = model_->InverseStdDev();

    int32_t dim = neg_mean.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + neg_mean[k]) * inv_stddev[k];
      }

      p += dim;
    }
  }

  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineParaformerModel> model_;
  std::unique_ptr<OfflineParaformerDecoder> decoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_
