// sherpa-mnn/csrc/offline-ct-transformer-model-meta-data.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_mnn {

struct OfflineCtTransformerModelMetaData {
  std::unordered_map<std::string, int32_t> token2id;
  std::unordered_map<std::string, int32_t> punct2id;
  std::vector<std::string> id2punct;

  int32_t unk_id;
  int32_t dot_id;
  int32_t comma_id;
  int32_t quest_id;
  int32_t pause_id;
  int32_t underline_id;
  int32_t num_punctuations;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_META_DATA_H_
