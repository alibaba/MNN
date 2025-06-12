// sherpa-mnn/csrc/online-recognizer-ctc-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_CTC_IMPL_H_

#include <algorithm>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-ctc-decoder.h"
#include "sherpa-mnn/csrc/online-ctc-fst-decoder.h"
#include "sherpa-mnn/csrc/online-ctc-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/online-ctc-model.h"
#include "sherpa-mnn/csrc/online-recognizer-impl.h"
#include "sherpa-mnn/csrc/symbol-table.h"

namespace sherpa_mnn {

OnlineRecognizerResult ConvertCtc(const OnlineCtcDecoderResult &src,
                                  const SymbolTable &sym_table,
                                  float frame_shift_ms,
                                  int32_t subsampling_factor, int32_t segment,
                                  int32_t frames_since_start) {
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

  r.segment = segment;
  r.words = std::move(src.words);
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class OnlineRecognizerCtcImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerCtcImpl(const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        model_(OnlineCtcModel::Create(config.model_config)),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (!config.model_config.wenet_ctc.model.empty()) {
      // WeNet CTC models assume input samples are in the range
      // [-32768, 32767], so we set normalize_samples to false
      config_.feat_config.normalize_samples = false;
    }

    InitDecoder();
  }

  template <typename Manager>
  explicit OnlineRecognizerCtcImpl(Manager *mgr,
                                   const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        model_(OnlineCtcModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.wenet_ctc.model.empty()) {
      // WeNet CTC models assume input samples are in the range
      // [-32768, 32767], so we set normalize_samples to false
      config_.feat_config.normalize_samples = false;
    }

    InitDecoder();
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    stream->SetStates(model_->GetInitStates());
    stream->SetFasterDecoder(decoder_->CreateFasterDecoder());

    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkLength() <
           s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    if (n == 1 || !model_->SupportBatchProcessing()) {
      for (int32_t i = 0; i != n; ++i) {
        DecodeStream(ss[i]);
      }
      return;
    }

    // batch processing
    int32_t chunk_length = model_->ChunkLength();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<OnlineCtcDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_length * feat_dim);
    std::vector<std::vector<MNN::Express::VARP>> states_vec(n);
    std::vector<int> all_processed_frames(n);

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_length);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_length * feat_dim);

      results[i] = std::move(ss[i]->GetCtcResult());
      states_vec[i] = std::move(ss[i]->GetStates());
      all_processed_frames[i] = num_processed_frames;
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{n, chunk_length, feat_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    auto states = model_->StackStates(std::move(states_vec));
    int32_t num_states = states.size();
    auto out = model_->Forward(std::move(x), std::move(states));
    std::vector<MNN::Express::VARP> out_states;
    out_states.reserve(num_states);

    for (int32_t k = 1; k != num_states + 1; ++k) {
      out_states.push_back(std::move(out[k]));
    }

    std::vector<std::vector<MNN::Express::VARP>> next_states =
        model_->UnStackStates(std::move(out_states));

    std::vector<int> log_probs_shape =
        out[0]->getInfo()->dim;
    decoder_->Decode(out[0]->readMap<float>(), log_probs_shape[0],
                     log_probs_shape[1], log_probs_shape[2], &results, ss, n);

    for (int32_t k = 0; k != n; ++k) {
      ss[k]->SetCtcResult(results[k]);
      ss[k]->SetStates(std::move(next_states[k]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineCtcDecoderResult decoder_result = s->GetCtcResult();

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r =
        ConvertCtc(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                   s->GetCurrentSegment(), s->GetNumFramesSinceStart());
    r.text = ApplyInverseTextNormalization(r.text);
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
    int32_t trailing_silence_frames = s->GetCtcResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    // segment is incremented only when the last
    // result is not empty
    const auto &r = s->GetCtcResult();
    if (!r.tokens.empty()) {
      s->GetCurrentSegment() += 1;
    }

    // clear result
    s->SetCtcResult({});

    // clear states
    s->SetStates(model_->GetInitStates());

    s->GetFasterDecoderProcessedFrames() = 0;

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  void InitDecoder() {
    if (!sym_.Contains("<blk>") && !sym_.Contains("<eps>") &&
        !sym_.Contains("<blank>")) {
      SHERPA_ONNX_LOGE(
          "We expect that tokens.txt contains "
          "the symbol <blk> or <eps> or <blank> and its ID.");
      exit(-1);
    }

    int32_t blank_id = 0;
    if (sym_.Contains("<blk>")) {
      blank_id = sym_["<blk>"];
    } else if (sym_.Contains("<eps>")) {
      // for tdnn models of the yesno recipe from icefall
      blank_id = sym_["<eps>"];
    } else if (sym_.Contains("<blank>")) {
      // for WeNet CTC models
      blank_id = sym_["<blank>"];
    }

    if (!config_.ctc_fst_decoder_config.graph.empty()) {
      decoder_ = std::make_unique<OnlineCtcFstDecoder>(
          config_.ctc_fst_decoder_config, blank_id);
    } else if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineCtcGreedySearchDecoder>(blank_id);
    } else {
      SHERPA_ONNX_LOGE(
          "Unsupported decoding method: %s for streaming CTC models",
          config_.decoding_method.c_str());
      exit(-1);
    }
  }

  void DecodeStream(OnlineStream *s) const {
    int32_t chunk_length = model_->ChunkLength();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feat_dim = s->FeatureDim();

    const auto num_processed_frames = s->GetNumProcessedFrames();
    std::vector<float> frames =
        s->GetFrames(num_processed_frames, chunk_length);
    s->GetNumProcessedFrames() += chunk_shift;

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{1, chunk_length, feat_dim};
    MNN::Express::VARP x =
        MNNUtilsCreateTensor(memory_info, frames.data(), frames.size(),
                                 x_shape.data(), x_shape.size());
    auto out = model_->Forward(std::move(x), std::move(s->GetStates()));
    int32_t num_states = static_cast<int32_t>(out.size()) - 1;

    std::vector<MNN::Express::VARP> states;
    states.reserve(num_states);

    for (int32_t i = 0; i != num_states; ++i) {
      states.push_back(std::move(out[i + 1]));
    }
    s->SetStates(std::move(states));

    std::vector<OnlineCtcDecoderResult> results(1);
    results[0] = std::move(s->GetCtcResult());

    std::vector<int> log_probs_shape =
        out[0]->getInfo()->dim;
    decoder_->Decode(out[0]->readMap<float>(), log_probs_shape[0],
                     log_probs_shape[1], log_probs_shape[2], &results, &s, 1);
    s->SetCtcResult(results[0]);
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineCtcModel> model_;
  std::unique_ptr<OnlineCtcDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_CTC_IMPL_H_
