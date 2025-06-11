// sherpa-mnn/csrc/online-recognizer-transducer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <algorithm>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-lm.h"
#include "sherpa-mnn/csrc/online-recognizer-impl.h"
#include "sherpa-mnn/csrc/online-recognizer.h"
#include "sherpa-mnn/csrc/online-transducer-decoder.h"
#include "sherpa-mnn/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/online-transducer-model.h"
#include "sherpa-mnn/csrc/online-transducer-modified-beam-search-decoder.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_mnn {

OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                               const SymbolTable &sym_table,
                               float frame_shift_ms, int32_t subsampling_factor,
                               int32_t segment, int32_t frames_since_start) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback
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

  r.ys_probs = std::move(src.ys_probs);
  r.lm_probs = std::move(src.lm_probs);
  r.context_scores = std::move(src.context_scores);

  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class OnlineRecognizerTransducerImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerImpl(const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    model_->SetFeatureDim(config.feat_config.feature_dim);

    if (config.decoding_method == "modified_beam_search") {
      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_buf.empty()) {
        InitHotwordsFromBufStr();
      } else if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      if (!config_.lm_config.model.empty()) {
        lm_ = OnlineLM::Create(config.lm_config);
      }

      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, config_.lm_config.shallow_fusion, unk_id_,
          config_.blank_penalty, config_.temperature_scale);

    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty,
          config_.temperature_scale);

    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }

  template <typename Manager>
  explicit OnlineRecognizerTransducerImpl(Manager *mgr,
                                          const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        model_(OnlineTransducerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    model_->SetFeatureDim(config.feat_config.feature_dim);

    if (config.decoding_method == "modified_beam_search") {
#if 0
      // TODO(fangjun): Implement it
      if (!config_.lm_config.model.empty()) {
        lm_ = OnlineLM::Create(mgr, config.lm_config);
      }
#endif

      if (!config_.model_config.bpe_vocab.empty()) {
        auto buf = ReadFile(mgr, config_.model_config.bpe_vocab);
        std::istringstream iss(std::string(buf.begin(), buf.end()));
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(iss);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }

      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, config_.lm_config.shallow_fusion, unk_id_,
          config_.blank_penalty, config_.temperature_scale);

    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty,
          config_.temperature_scale);

    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream =
        std::make_unique<OnlineStream>(config_.feat_config, hotwords_graph_);
    InitOnlineStream(stream.get());
    return stream;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
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
    auto stream =
        std::make_unique<OnlineStream>(config_.feat_config, context_graph);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  // Warmping up engine with wp: warm_up count and max-batch-size
  void WarmpUpRecognizer(int32_t warmup, int32_t mbs) const override {
    auto max_batch_size = mbs;
    if (warmup <= 0 || warmup > 100) {
      return;
    }
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();
    int32_t feature_dim = 80;
    std::vector<OnlineTransducerDecoderResult> results(max_batch_size);
    std::vector<float> features_vec(max_batch_size * chunk_size * feature_dim);
    std::vector<std::vector<MNN::Express::VARP>> states_vec(max_batch_size);

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{max_batch_size, chunk_size, feature_dim};

    for (int32_t i = 0; i != max_batch_size; ++i) {
      states_vec[i] = model_->GetEncoderInitStates();
      results[i] = decoder_->GetEmptyResult();
    }

    for (int32_t i = 0; i != warmup; ++i) {
      auto states = model_->StackStates(states_vec);
      MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, features_vec.data(),
                                              features_vec.size(),
                                              x_shape.data(), x_shape.size());
      auto x_copy = Clone(model_->Allocator(), x);
      auto pair = model_->RunEncoder(std::move(x), std::move(states),
                                     std::move(x_copy));
      decoder_->Decode(std::move(pair.first), &results);
    }
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<OnlineTransducerDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<MNN::Express::VARP>> states_vec(n);
    std::vector<int> all_processed_frames(n);
    bool has_context_graph = false;

    for (int32_t i = 0; i != n; ++i) {
      if (!has_context_graph && ss[i]->GetContextGraph()) {
        has_context_graph = true;
      }

      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      results[i] = std::move(ss[i]->GetResult());
      states_vec[i] = std::move(ss[i]->GetStates());
      all_processed_frames[i] = num_processed_frames;
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{n, chunk_size, feature_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    std::array<int, 1> processed_frames_shape{
        static_cast<int>(all_processed_frames.size())};

    MNN::Express::VARP processed_frames = MNNUtilsCreateTensor(
        memory_info, all_processed_frames.data(), all_processed_frames.size(),
        processed_frames_shape.data(), processed_frames_shape.size());

    auto states = model_->StackStates(states_vec);

    auto pair = model_->RunEncoder(std::move(x), std::move(states),
                                   std::move(processed_frames));

    if (has_context_graph) {
      decoder_->Decode(std::move(pair.first), ss, &results);
    } else {
      decoder_->Decode(std::move(pair.first), &results);
    }

    std::vector<std::vector<MNN::Express::VARP>> next_states =
        model_->UnStackStates(pair.second);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetStates(std::move(next_states[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineTransducerDecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r = Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                     s->GetCurrentSegment(), s->GetNumFramesSinceStart());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    int32_t context_size = model_->ContextSize();

    {
      // segment is incremented only when the last
      // result is not empty, contains non-blanks and longer than context_size)
      const auto &r = s->GetResult();
      if (!r.tokens.empty() && r.tokens.back() != 0 &&
          r.tokens.size() > context_size) {
        s->GetCurrentSegment() += 1;
      }
    }

    // reset encoder states
    // s->SetStates(model_->GetEncoderInitStates());

    auto r = decoder_->GetEmptyResult();
    auto last_result = s->GetResult();
    // if last result is not empty, then
    // preserve last tokens as the context for next result
    if (static_cast<int32_t>(last_result.tokens.size()) > context_size) {
      std::vector<int> context(last_result.tokens.end() - context_size,
                                   last_result.tokens.end());

      Hypotheses context_hyp({{context, 0}});
      r.hyps = std::move(context_hyp);
      r.tokens = std::move(context);
    }

    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != s->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = s->GetContextGraph()->Root();
      }
    }

    s->SetResult(r);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      exit(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
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

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  void InitHotwordsFromBufStr() {
    // each line in hotwords_file contains space-separated words

    std::istringstream iss(config_.hotwords_buf);
    if (!EncodeHotwords(iss, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  void InitOnlineStream(OnlineStream *stream) const {
    auto r = decoder_->GetEmptyResult();

    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != stream->GetContextGraph()) {
      // r.hyps has only one element.
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = stream->GetContextGraph()->Root();
      }
    }

    stream->SetResult(r);
    stream->SetStates(model_->GetEncoderInitStates());
  }

 private:
  OnlineRecognizerConfig config_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineLM> lm_;
  std::unique_ptr<OnlineTransducerDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
