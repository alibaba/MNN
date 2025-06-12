// sherpa-mnn/csrc/offline-tts-vits-model.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_

#include <memory>
#include <string>

#include "MNNUtils.hpp"  // NOLINT
#include "sherpa-mnn/csrc/offline-tts-model-config.h"
#include "sherpa-mnn/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_mnn {

class OfflineTtsVitsModel {
 public:
  ~OfflineTtsVitsModel();

  explicit OfflineTtsVitsModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsVitsModel(Manager *mgr, const OfflineTtsModelConfig &config);

  /** Run the model.
   *
   * @param x A int64 tensor of shape (1, num_tokens)
  // @param sid Speaker ID. Used only for multi-speaker models, e.g., models
  //            trained using the VCTK dataset. It is not used for
  //            single-speaker models, e.g., models trained using the ljspeech
  //            dataset.
   * @return Return a float32 tensor containing audio samples. You can flatten
   *         it to a 1-D tensor.
   */
  MNN::Express::VARP Run(MNN::Express::VARP x, int sid = 0, float speed = 1.0);

  // This is for MeloTTS
  MNN::Express::VARP Run(MNN::Express::VARP x, MNN::Express::VARP tones, int sid = 0,
                 float speed = 1.0) const;

  const OfflineTtsVitsModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_
