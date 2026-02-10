// sherpa-mnn/csrc/online-transducer-modified_beam-search-decoder.h
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-mnn/csrc/online-lm.h"
#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/online-transducer-decoder.h"
#include "sherpa-mnn/csrc/online-transducer-model.h"

namespace sherpa_mnn {

class OnlineTransducerModifiedBeamSearchDecoder
    : public OnlineTransducerDecoder {
 public:
  OnlineTransducerModifiedBeamSearchDecoder(OnlineTransducerModel *model,
                                            OnlineLM *lm,
                                            int32_t max_active_paths,
                                            float lm_scale,
                                            bool shallow_fusion,
                                            int32_t unk_id,
                                            float blank_penalty,
                                            float temperature_scale)
      : model_(model),
        lm_(lm),
        max_active_paths_(max_active_paths),
        lm_scale_(lm_scale),
        shallow_fusion_(shallow_fusion),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty),
        temperature_scale_(temperature_scale) {}

  OnlineTransducerDecoderResult GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) const override;

  void Decode(MNN::Express::VARP encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void Decode(MNN::Express::VARP encoder_out, OnlineStream **ss,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void UpdateDecoderOut(OnlineTransducerDecoderResult *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  OnlineLM *lm_;                  // Not owned

  int32_t max_active_paths_;
  float lm_scale_;  // used only when lm_ is not nullptr
  bool shallow_fusion_;  // used only when lm_ is not nullptr
  int32_t unk_id_;
  float blank_penalty_;
  float temperature_scale_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
