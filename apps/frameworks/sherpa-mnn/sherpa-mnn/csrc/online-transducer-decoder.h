// sherpa-mnn/csrc/online-transducer-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/hypothesis.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

struct OnlineTransducerDecoderResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  std::vector<float> ys_probs;
  std::vector<float> lm_probs;
  std::vector<float> context_scores;

  // Cache decoder_out for endpointing
  MNN::Express::VARP decoder_out;

  // used only in modified beam_search
  Hypotheses hyps;

  OnlineTransducerDecoderResult()
      : tokens{}, num_trailing_blanks(0), decoder_out{nullptr}, hyps{} {}

  OnlineTransducerDecoderResult(const OnlineTransducerDecoderResult &other);

  OnlineTransducerDecoderResult &operator=(
      const OnlineTransducerDecoderResult &other);

  OnlineTransducerDecoderResult(OnlineTransducerDecoderResult &&other) noexcept;

  OnlineTransducerDecoderResult &operator=(
      OnlineTransducerDecoderResult &&other) noexcept;
};

class OnlineStream;
class OnlineTransducerDecoder {
 public:
  virtual ~OnlineTransducerDecoder() = default;

  /* Return an empty result.
   *
   * To simplify the decoding code, we add `context_size` blanks
   * to the beginning of the decoding result, which will be
   * stripped by calling `StripPrecedingBlanks()`.
   */
  virtual OnlineTransducerDecoderResult GetEmptyResult() const = 0;

  /** Strip blanks added by `GetEmptyResult()`.
   *
   * @param r It is changed in-place.
   */
  virtual void StripLeadingBlanks(OnlineTransducerDecoderResult * /*r*/) const {
  }

  /** Run transducer beam search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param result  It is modified in-place.
   *
   * @note There is no need to pass encoder_out_length here since for the
   * online decoding case, each utterance has the same number of frames
   * and there are no paddings.
   */
  virtual void Decode(MNN::Express::VARP encoder_out,
                      std::vector<OnlineTransducerDecoderResult> *result) = 0;

  /** Run transducer beam search given the output from the encoder model.
   *
   * Note: Currently this interface is for contextual-biasing feature which
   *       needs a ContextGraph owned by the OnlineStream.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param ss  A list of OnlineStreams.
   * @param result  It is modified in-place.
   *
   * @note There is no need to pass encoder_out_length here since for the
   * online decoding case, each utterance has the same number of frames
   * and there are no paddings.
   */
  virtual void Decode(MNN::Express::VARP /*encoder_out*/, OnlineStream ** /*ss*/,
                      std::vector<OnlineTransducerDecoderResult> * /*result*/) {
    SHERPA_ONNX_LOGE(
        "This interface is for OnlineTransducerModifiedBeamSearchDecoder.");
    exit(-1);
  }

  // used for endpointing. We need to keep decoder_out after reset
  virtual void UpdateDecoderOut(OnlineTransducerDecoderResult * /*result*/) {}
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_
