// sherpa-mnn/csrc/online-paraformer-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/online-model-config.h"

namespace sherpa_mnn {

class OnlineParaformerModel {
 public:
  explicit OnlineParaformerModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineParaformerModel(Manager *mgr, const OnlineModelConfig &config);

  ~OnlineParaformerModel();

  std::vector<MNN::Express::VARP> ForwardEncoder(MNN::Express::VARP features,
                                         MNN::Express::VARP features_length) const;

  std::vector<MNN::Express::VARP> ForwardDecoder(MNN::Express::VARP encoder_out,
                                         MNN::Express::VARP encoder_out_length,
                                         MNN::Express::VARP acoustic_embedding,
                                         MNN::Express::VARP acoustic_embedding_length,
                                         std::vector<MNN::Express::VARP> states) const;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const;

  /** It is lfr_m in config.yaml
   */
  int32_t LfrWindowSize() const;

  /** It is lfr_n in config.yaml
   */
  int32_t LfrWindowShift() const;

  int32_t EncoderOutputSize() const;

  int32_t DecoderKernelSize() const;
  int32_t DecoderNumBlocks() const;

  /** Return negative mean for CMVN
   */
  const std::vector<float> &NegativeMean() const;

  /** Return inverse stddev for CMVN
   */
  const std::vector<float> &InverseStdDev() const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_MODEL_H_
