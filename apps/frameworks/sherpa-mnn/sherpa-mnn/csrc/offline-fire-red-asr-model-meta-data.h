// sherpa-mnn/csrc/offline-fire-red-asr-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_mnn {

struct OfflineFireRedAsrModelMetaData {
  int32_t sos_id;
  int32_t eos_id;
  int32_t max_len;

  int32_t num_decoder_layers;
  int32_t num_head;
  int32_t head_dim;

  std::vector<float> mean;
  std::vector<float> inv_stddev;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_
