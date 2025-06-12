// sherpa-mnn/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-mnn/csrc/online-stream.h"

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/features.h"
#include "sherpa-mnn/csrc/transducer-keyword-decoder.h"

namespace sherpa_mnn {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(std::move(context_graph)) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() const { feat_extractor_.InputFinished(); }

  int32_t NumFramesReady() const {
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  int32_t GetNumFramesSinceStart() const { return start_frame_index_; }

  int32_t &GetCurrentSegment() { return segment_; }

  void SetResult(const OnlineTransducerDecoderResult &r) { result_ = r; }

  OnlineTransducerDecoderResult &GetResult() { return result_; }

  void SetKeywordResult(const TransducerKeywordResult &r) {
    keyword_result_ = r;
  }
  TransducerKeywordResult &GetKeywordResult(bool remove_duplicates) {
    if (remove_duplicates) {
      if (!prev_keyword_result_.timestamps.empty() &&
          !keyword_result_.timestamps.empty() &&
          keyword_result_.timestamps[0] <=
              prev_keyword_result_.timestamps.back()) {
        return empty_keyword_result_;
      } else {
        prev_keyword_result_ = keyword_result_;
      }
      return keyword_result_;
    } else {
      return keyword_result_;
    }
  }

  OnlineCtcDecoderResult &GetCtcResult() { return ctc_result_; }

  void SetCtcResult(const OnlineCtcDecoderResult &r) { ctc_result_ = r; }

  void SetParaformerResult(const OnlineParaformerDecoderResult &r) {
    paraformer_result_ = r;
  }

  OnlineParaformerDecoderResult &GetParaformerResult() {
    return paraformer_result_;
  }

  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<MNN::Express::VARP> states) {
    states_ = std::move(states);
  }

  std::vector<MNN::Express::VARP> &GetStates() { return states_; }

  void SetNeMoDecoderStates(std::vector<MNN::Express::VARP> decoder_states) {
    decoder_states_ = std::move(decoder_states);
  }

  std::vector<MNN::Express::VARP> &GetNeMoDecoderStates() { return decoder_states_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

  std::vector<float> &GetParaformerFeatCache() {
    return paraformer_feat_cache_;
  }

  std::vector<float> &GetParaformerEncoderOutCache() {
    return paraformer_encoder_out_cache_;
  }

  std::vector<float> &GetParaformerAlphaCache() {
    return paraformer_alpha_cache_;
  }

  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
    faster_decoder_ = std::move(decoder);
  }

  kaldi_decoder::FasterDecoder *GetFasterDecoder() const {
    return faster_decoder_.get();
  }

  int32_t &GetFasterDecoderProcessedFrames() {
    return faster_decoder_processed_frames_;
  }

 private:
  FeatureExtractor feat_extractor_;
  /// For contextual-biasing
  ContextGraphPtr context_graph_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  int32_t segment_ = 0;
  OnlineTransducerDecoderResult result_;
  TransducerKeywordResult prev_keyword_result_;
  TransducerKeywordResult keyword_result_;
  TransducerKeywordResult empty_keyword_result_;
  OnlineCtcDecoderResult ctc_result_;
  std::vector<MNN::Express::VARP> states_;  // states for transducer or ctc models
  std::vector<MNN::Express::VARP> decoder_states_;  // states for nemo transducer models
  std::vector<float> paraformer_feat_cache_;
  std::vector<float> paraformer_encoder_out_cache_;
  std::vector<float> paraformer_alpha_cache_;
  OnlineParaformerDecoderResult paraformer_result_;
  std::unique_ptr<kaldi_decoder::FasterDecoder> faster_decoder_;
  int32_t faster_decoder_processed_frames_ = 0;
};

OnlineStream::OnlineStream(const FeatureExtractorConfig &config /*= {}*/,
                           ContextGraphPtr context_graph /*= nullptr */)
    : impl_(std::make_unique<Impl>(config, std::move(context_graph))) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                  int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void OnlineStream::InputFinished() const { impl_->InputFinished(); }

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> OnlineStream::GetFrames(int32_t frame_index,
                                           int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void OnlineStream::Reset() { impl_->Reset(); }

int32_t OnlineStream::FeatureDim() const { return impl_->FeatureDim(); }

int32_t &OnlineStream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

int32_t OnlineStream::GetNumFramesSinceStart() const {
  return impl_->GetNumFramesSinceStart();
}

int32_t &OnlineStream::GetCurrentSegment() {
  return impl_->GetCurrentSegment();
}

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

OnlineTransducerDecoderResult &OnlineStream::GetResult() {
  return impl_->GetResult();
}

void OnlineStream::SetKeywordResult(const TransducerKeywordResult &r) {
  impl_->SetKeywordResult(r);
}

TransducerKeywordResult &OnlineStream::GetKeywordResult(
    bool remove_duplicates /*=false*/) {
  return impl_->GetKeywordResult(remove_duplicates);
}

OnlineCtcDecoderResult &OnlineStream::GetCtcResult() {
  return impl_->GetCtcResult();
}

void OnlineStream::SetCtcResult(const OnlineCtcDecoderResult &r) {
  impl_->SetCtcResult(r);
}

void OnlineStream::SetParaformerResult(const OnlineParaformerDecoderResult &r) {
  impl_->SetParaformerResult(r);
}

OnlineParaformerDecoderResult &OnlineStream::GetParaformerResult() {
  return impl_->GetParaformerResult();
}

void OnlineStream::SetStates(std::vector<MNN::Express::VARP> states) {
  impl_->SetStates(std::move(states));
}

std::vector<MNN::Express::VARP> &OnlineStream::GetStates() {
  return impl_->GetStates();
}

void OnlineStream::SetNeMoDecoderStates(
    std::vector<MNN::Express::VARP> decoder_states) {
  return impl_->SetNeMoDecoderStates(std::move(decoder_states));
}

std::vector<MNN::Express::VARP> &OnlineStream::GetNeMoDecoderStates() {
  return impl_->GetNeMoDecoderStates();
}

const ContextGraphPtr &OnlineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

void OnlineStream::SetFasterDecoder(
    std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
  impl_->SetFasterDecoder(std::move(decoder));
}

kaldi_decoder::FasterDecoder *OnlineStream::GetFasterDecoder() const {
  return impl_->GetFasterDecoder();
}

int32_t &OnlineStream::GetFasterDecoderProcessedFrames() {
  return impl_->GetFasterDecoderProcessedFrames();
}

std::vector<float> &OnlineStream::GetParaformerFeatCache() {
  return impl_->GetParaformerFeatCache();
}

std::vector<float> &OnlineStream::GetParaformerEncoderOutCache() {
  return impl_->GetParaformerEncoderOutCache();
}

std::vector<float> &OnlineStream::GetParaformerAlphaCache() {
  return impl_->GetParaformerAlphaCache();
}

}  // namespace sherpa_mnn
