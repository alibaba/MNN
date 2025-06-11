// sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_H_
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-ctc-model.h"
#include "sherpa-mnn/csrc/offline-model-config.h"

namespace sherpa_mnn {

/** This class implements the EncDecCTCModelBPE model from NeMo.
 *
 * See
 * https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_bpe_models.py
 * https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
 */
class OfflineNemoEncDecCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineNemoEncDecCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineNemoEncDecCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineNemoEncDecCtcModel() override;

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

  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string FeatureNormalizationMethod() const override;

  bool IsGigaAM() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

using OfflineNemoEncDecHybridRNNTCTCBPEModel = OfflineNemoEncDecCtcModel;

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_H_
