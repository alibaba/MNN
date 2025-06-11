// sherpa-mnn/csrc/online-ctc-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_

#include <memory>
#include <vector>

#include "kaldi-decoder/csrc/faster-decoder.h"
#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

class OnlineStream;

struct OnlineCtcDecoderResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs
  std::vector<int> tokens;

  /// The decoded word IDs
  /// Note: tokens.size() is usually not equal to words.size()
  /// words is empty for greedy search decoding.
  /// it is not empty when an HLG graph or an HLG graph is used.
  std::vector<int32_t> words;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  /// Note: The index is after subsampling
  ///
  /// tokens.size() == timestamps.size()
  std::vector<int32_t> timestamps;

  int32_t num_trailing_blanks = 0;
};

class OnlineCtcDecoder {
 public:
  virtual ~OnlineCtcDecoder() = default;

  /** Run streaming CTC decoding given the output from the encoder model.
   *
   * @param log_probs A 3-D tensor of shape
   *                  (batch_size, num_frames, vocab_size) containing
   *                  lob_probs in row major.
   *
   * @param  results Input & Output parameters..
   */
  virtual void Decode(const float *log_probs, int32_t batch_size,
                      int32_t num_frames, int32_t vocab_size,
                      std::vector<OnlineCtcDecoderResult> *results,
                      OnlineStream **ss = nullptr, int32_t n = 0) = 0;

  virtual std::unique_ptr<kaldi_decoder::FasterDecoder> CreateFasterDecoder()
      const {
    return nullptr;
  }
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_
