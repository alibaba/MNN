// sherpa-mnn/csrc/offline-ctc-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

struct OfflineCtcDecoderResult {
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
};

class OfflineCtcDecoder {
 public:
  virtual ~OfflineCtcDecoder() = default;

  /** Run CTC decoding given the output from the encoder model.
   *
   * @param log_probs A 3-D tensor of shape (N, T, vocab_size) containing
   *                  lob_probs.
   * @param log_probs_length A 1-D tensor of shape (N,) containing number
   *                         of valid frames in log_probs before padding.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineCtcDecoderResult> Decode(
      MNN::Express::VARP log_probs, MNN::Express::VARP log_probs_length) = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_
