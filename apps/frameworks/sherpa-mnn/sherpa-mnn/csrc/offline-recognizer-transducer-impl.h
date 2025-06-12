// sherpa-mnn/csrc/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/context-graph.h"
#include "sherpa-mnn/csrc/log.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/offline-transducer-decoder.h"
#include "sherpa-mnn/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/offline-transducer-model.h"
#include "sherpa-mnn/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa-mnn/csrc/pad-sequence.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_mnn {

static OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback,
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }
  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  return r;
}

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(config_.model_config)) {
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(config.lm_config);
      }

      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_, config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
  }

  template <typename Manager>
  explicit OfflineRecognizerTransducerImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(mgr,
                                                        config_.model_config)) {
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(mgr, config.lm_config);
      }

      if (!config_.model_config.bpe_vocab.empty()) {
        auto buf = ReadFile(mgr, config_.model_config.bpe_vocab);
        std::istringstream iss(std::string(buf.begin(), buf.end()));
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(iss);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_, config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &current, &current_scores)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are : %s",
                       hotwords.c_str());
    }

    int32_t num_default_hws = hotwords_.size();
    int32_t num_hws = current.size();

    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_hws,
                            config_.hotwords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_hws,
                            config_.hotwords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else {
      // Do nothing.
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);
    return std::make_unique<OfflineStream>(config_.feat_config, context_graph);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config,
                                           hotwords_graph_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<MNN::Express::VARP> features;

    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFrames();
      int32_t num_frames = f.size() / feat_dim;

      features_length_vec[i] = num_frames;
      features_vec[i] = std::move(f);

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

    MNN::Express::VARP x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    auto results =
        decoder_->Decode(std::move(t.first), std::move(t.second), ss, n);

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));

      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      exit(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  template <typename Manager>
  void InitHotwords(Manager *mgr) {
    // each line in hotwords_file contains space-separated words

    auto buf = ReadFile(mgr, config_.hotwords_file);

    std::istringstream is(std::string(buf.begin(), buf.end()));

    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      exit(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  std::unique_ptr<OfflineLM> lm_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
