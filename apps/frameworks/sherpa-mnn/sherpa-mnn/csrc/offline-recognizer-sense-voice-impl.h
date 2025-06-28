// sherpa-mnn/csrc/offline-recognizer-sense-voice-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/offline-model-config.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/offline-sense-voice-model.h"
#include "sherpa-mnn/csrc/pad-sequence.h"
#include "sherpa-mnn/csrc/symbol-table.h"

namespace sherpa_mnn {

static OfflineRecognitionResult ConvertSenseVoiceResult(
    const OfflineCtcDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;

  for (int32_t i = 4; i < src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;

  for (int32_t i = 4; i < src.timestamps.size(); ++i) {
    float time = frame_shift_s * (src.timestamps[i] - 4);
    r.timestamps.push_back(time);
  }

  r.words = std::move(src.words);

  // parse lang, emotion and event from tokens.
  if (src.tokens.size() >= 3) {
    r.lang = sym_table[src.tokens[0]];
    r.emotion = sym_table[src.tokens[1]];
    r.event = sym_table[src.tokens[2]];
  }

  return r;
}

class OfflineRecognizerSenseVoiceImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerSenseVoiceImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(config.model_config)) {
    const auto &meta_data = model_->metaData();
    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineCtcGreedySearchDecoder>(meta_data.blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerSenseVoiceImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(mgr,
                                                        config.model_config)) {
    const auto &meta_data = model_->metaData();
    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineCtcGreedySearchDecoder>(meta_data.blank_id);
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
    if (n == 1) {
      DecodeOneStream(ss[0]);
      return;
    }

    const auto &meta_data = model_->metaData();
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

    int32_t feat_dim = config_.feat_config.feature_dim * meta_data.window_size;

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

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_ONNX_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    std::vector<int32_t> language_array(n);
    std::fill(language_array.begin(), language_array.end(), language);

    std::vector<int32_t> text_norm_array(n);
    std::fill(text_norm_array.begin(), text_norm_array.end(),
              config_.model_config.sense_voice.use_itn
                  ? meta_data.with_itn_id
                  : meta_data.without_itn_id);

    MNN::Express::VARP language_tensor = MNNUtilsCreateTensor(
        memory_info, language_array.data(), n, features_length_shape.data(),
        features_length_shape.size());

    MNN::Express::VARP text_norm_tensor = MNNUtilsCreateTensor(
        memory_info, text_norm_array.data(), n, features_length_shape.data(),
        features_length_shape.size());

    MNN::Express::VARP logits{nullptr};
      logits = model_->Forward(std::move(x), std::move(x_length),
                               std::move(language_tensor),
                               std::move(text_norm_tensor));
    // decoder_->Decode() requires that logits_length is of dtype int64
    std::vector<int> features_length_vec_64;
    features_length_vec_64.reserve(n);
    for (auto i : features_length_vec) {
      i += 4;
      features_length_vec_64.push_back(i);
    }

    MNN::Express::VARP logits_length = MNNUtilsCreateTensor(
        memory_info, features_length_vec_64.data(), n,
        features_length_shape.data(), features_length_shape.size());

    auto results =
        decoder_->Decode(std::move(logits), std::move(logits_length));

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    for (int32_t i = 0; i != n; ++i) {
      auto r = ConvertSenseVoiceResult(results[i], symbol_table_,
                                       frame_shift_ms, subsampling_factor);
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeOneStream(OfflineStream *s) const {
    const auto &meta_data = model_->metaData();

    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t feat_dim = config_.feat_config.feature_dim * meta_data.window_size;
    std::vector<float> f = s->GetFrames();
    f = ApplyLFR(f);
    ApplyCMVN(&f);
    int32_t num_frames = f.size() / feat_dim;
    std::array<int, 3> shape = {1, num_frames, feat_dim};
    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int scale_shape = 1;

    MNN::Express::VARP x_length =
        MNNUtilsCreateTensor(memory_info, &num_frames, 1, &scale_shape, 1);

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_ONNX_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    int32_t text_norm = config_.model_config.sense_voice.use_itn
                            ? meta_data.with_itn_id
                            : meta_data.without_itn_id;

    MNN::Express::VARP language_tensor =
        MNNUtilsCreateTensor(memory_info, &language, 1, &scale_shape, 1);

    MNN::Express::VARP text_norm_tensor =
        MNNUtilsCreateTensor(memory_info, &text_norm, 1, &scale_shape, 1);

    MNN::Express::VARP logits{nullptr};
      logits = model_->Forward(std::move(x), std::move(x_length),
                               std::move(language_tensor),
                               std::move(text_norm_tensor));

    int new_num_frames = num_frames + 4;
    MNN::Express::VARP logits_length = MNNUtilsCreateTensor(
        memory_info, &new_num_frames, 1, &scale_shape, 1);

    auto results =
        decoder_->Decode(std::move(logits), std::move(logits_length));

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    auto r = ConvertSenseVoiceResult(results[0], symbol_table_, frame_shift_ms,
                                     subsampling_factor);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    s->SetResult(r);
  }

  void InitFeatConfig() {
    const auto &meta_data = model_->metaData();

    config_.feat_config.normalize_samples = meta_data.normalize_samples;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }
  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    const auto &meta_data = model_->metaData();

    int32_t lfr_window_size = meta_data.window_size;
    int32_t lfr_window_shift = meta_data.window_shift;
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
    const auto &meta_data = model_->metaData();

    const std::vector<float> &neg_mean = meta_data.neg_mean;
    const std::vector<float> &inv_stddev = meta_data.inv_stddev;

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
  std::unique_ptr<OfflineSenseVoiceModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
