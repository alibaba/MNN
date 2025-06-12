// sherpa-mnn/csrc/offline-wenet-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_WENET_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WENET_CTC_MODEL_H_
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-ctc-model.h"
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

/** This class implements the CTC model from WeNet.
 *
 * See
 * https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/export-onnx.py
 * https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/test-onnx.py
 * https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/run.sh
 *
 */
class OfflineWenetCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineWenetCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWenetCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineWenetCtcModel() override;

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
  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                  MNN::Express::VARP features_length) override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** SubsamplingFactor of the model
   *
   * For Citrinet, the subsampling factor is usually 4.
   * For Conformer CTC, the subsampling factor is usually 8.
   */
  int32_t SubsamplingFactor() const override;

  /** Return an allocator for allocating memory
   */
  MNNAllocator *Allocator() const override;

  // WeNet CTC models do not support batch size > 1
  bool SupportBatchProcessing() const override { return false; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WENET_CTC_MODEL_H_
