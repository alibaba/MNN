// sherpa-mnn/csrc/offline-ctc-fst-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_

#include <memory>
#include <vector>

#include "fst/fst.h"
#include "sherpa-mnn/csrc/offline-ctc-decoder.h"
#include "sherpa-mnn/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

class OfflineCtcFstDecoder : public OfflineCtcDecoder {
 public:
  explicit OfflineCtcFstDecoder(const OfflineCtcFstDecoderConfig &config);

  std::vector<OfflineCtcDecoderResult> Decode(
      MNN::Express::VARP log_probs, MNN::Express::VARP log_probs_length) override;

 private:
  OfflineCtcFstDecoderConfig config_;

  std::unique_ptr<fst::Fst<fst::StdArc>> fst_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_
