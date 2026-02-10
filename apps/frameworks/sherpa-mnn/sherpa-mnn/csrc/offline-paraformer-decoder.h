// sherpa-mnn/csrc/offline-paraformer-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_DECODER_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

struct OfflineParaformerDecoderResult {
  /// The decoded token IDs
  std::vector<int> tokens;

  // it contains the start time of each token in seconds
  //
  // len(timestamps) == len(tokens)
  std::vector<float> timestamps;
};

class OfflineParaformerDecoder {
 public:
  virtual ~OfflineParaformerDecoder() = default;

  /** Run beam search given the output from the paraformer model.
   *
   * @param log_probs A 3-D tensor of shape (N, T, vocab_size)
   * @param token_num A 1-D tensor of shape (N). token_num equals to T.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineParaformerDecoderResult> Decode(
      MNN::Express::VARP log_probs, MNN::Express::VARP token_num,
      MNN::Express::VARP us_cif_peak = MNN::Express::VARP(nullptr)) = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_DECODER_H_
