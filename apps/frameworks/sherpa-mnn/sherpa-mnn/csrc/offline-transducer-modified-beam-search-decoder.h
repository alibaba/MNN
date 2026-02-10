// sherpa-mnn/csrc/offline-transducer-modified-beam-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/offline-lm.h"
#include "sherpa-mnn/csrc/offline-transducer-decoder.h"
#include "sherpa-mnn/csrc/offline-transducer-model.h"

namespace sherpa_mnn {

class OfflineTransducerModifiedBeamSearchDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerModifiedBeamSearchDecoder(OfflineTransducerModel *model,
                                             OfflineLM *lm,
                                             int32_t max_active_paths,
                                             float lm_scale, int32_t unk_id,
                                             float blank_penalty)
      : model_(model),
        lm_(lm),
        max_active_paths_(max_active_paths),
        lm_scale_(lm_scale),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      MNN::Express::VARP encoder_out, MNN::Express::VARP encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
  OfflineLM *lm_;                  // Not owned; may be nullptr

  int32_t max_active_paths_;
  float lm_scale_;  // used only when lm_ is not nullptr
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
