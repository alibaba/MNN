// sherpa-mnn/csrc/online-cnn-bilstm-model.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_H_
#include <memory>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/online-cnn-bilstm-model-meta-data.h"
#include "sherpa-mnn/csrc/online-punctuation-model-config.h"

namespace sherpa_mnn {

/** This class implements
 *  https://github.com/frankyoujian/Edge-Punct-Casing/blob/main/onnx_decode_sentence.py
 */
class OnlineCNNBiLSTMModel {
 public:
  explicit OnlineCNNBiLSTMModel(const OnlinePunctuationModelConfig &config);

#if __ANDROID_API__ >= 9
  OnlineCNNBiLSTMModel(AAssetManager *mgr,
                       const OnlinePunctuationModelConfig &config);
#endif

  ~OnlineCNNBiLSTMModel();

  /** Run the forward method of the model.
   *
   * @param token_ids  A tensor of shape (N, T) of dtype int32.
   * @param valid_ids  A tensor of shape (N, T) of dtype int32.
   * @param label_lens A tensor of shape (N) of dtype int32.
   *
   * @return Return a pair of tensors
   *  - case_logits:  A 2-D tensor of shape (T', num_cases).
   *  - punct_logits: A 2-D tensor of shape (T', num_puncts).
   */
  std::pair<MNN::Express::VARP, MNN::Express::VARP> Forward(MNN::Express::VARP token_ids,
                                            MNN::Express::VARP valid_ids,
                                            MNN::Express::VARP label_lens) const;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const;

  const OnlineCNNBiLSTMModelMetaData& metaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_H_
