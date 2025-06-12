// sherpa-mnn/csrc/offline-ctc-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

class OfflineCtcModel {
 public:
  virtual ~OfflineCtcModel() = default;

  static std::unique_ptr<OfflineCtcModel> Create(
      const OfflineModelConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineCtcModel> Create(
      Manager *mgr, const OfflineModelConfig &config);

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int.
   *
   * @return Return a vector containing:
   *  - log_probs: A 3-D tensor of shape (N, T', vocab_size).
   *  - log_probs_length A 1-D tensor of shape (N,). Its dtype is int
   */
  virtual std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                          MNN::Express::VARP features_length) = 0;

  /** Return the vocabulary size of the model
   */
  virtual int32_t VocabSize() const = 0;

  /** SubsamplingFactor of the model
   *
   * For NeMo Citrinet, the subsampling factor is usually 4.
   * For NeMo Conformer CTC, the subsampling factor is usually 8.
   */
  virtual int32_t SubsamplingFactor() const { return 1; }

  /** Return an allocator for allocating memory
   */
  virtual MNNAllocator *Allocator() const = 0;

  /** For some models, e.g., those from NeMo, they require some preprocessing
   * for the features.
   */
  virtual std::string FeatureNormalizationMethod() const { return {}; }

  // Return true if the model supports batch size > 1
  virtual bool SupportBatchProcessing() const { return true; }

  // return true for models from https://github.com/salute-developers/GigaAM
  // return false otherwise
  virtual bool IsGigaAM() const { return false; }
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_
