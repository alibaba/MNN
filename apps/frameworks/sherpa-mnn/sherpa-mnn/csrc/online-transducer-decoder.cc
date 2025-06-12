// sherpa-mnn/csrc/online-transducer-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-transducer-decoder.h"

#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

OnlineTransducerDecoderResult::OnlineTransducerDecoderResult(
    const OnlineTransducerDecoderResult &other)
    : OnlineTransducerDecoderResult() {
  *this = other;
}

OnlineTransducerDecoderResult &OnlineTransducerDecoderResult::operator=(
    const OnlineTransducerDecoderResult &other) {
  if (this == &other) {
    return *this;
  }

  tokens = other.tokens;
  num_trailing_blanks = other.num_trailing_blanks;

  MNNAllocator* allocator;
  if (other.decoder_out.get() != nullptr) {
    decoder_out = Clone(allocator, other.decoder_out);
  }

  hyps = other.hyps;

  frame_offset = other.frame_offset;
  timestamps = other.timestamps;

  ys_probs = other.ys_probs;
  lm_probs = other.lm_probs;
  context_scores = other.context_scores;

  return *this;
}

OnlineTransducerDecoderResult::OnlineTransducerDecoderResult(
    OnlineTransducerDecoderResult &&other) noexcept
    : OnlineTransducerDecoderResult() {
  *this = std::move(other);
}

OnlineTransducerDecoderResult &OnlineTransducerDecoderResult::operator=(
    OnlineTransducerDecoderResult &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  tokens = std::move(other.tokens);
  num_trailing_blanks = other.num_trailing_blanks;
  decoder_out = std::move(other.decoder_out);
  hyps = std::move(other.hyps);

  frame_offset = other.frame_offset;
  timestamps = std::move(other.timestamps);

  ys_probs = std::move(other.ys_probs);
  lm_probs = std::move(other.lm_probs);
  context_scores = std::move(other.context_scores);

  return *this;
}

}  // namespace sherpa_mnn
