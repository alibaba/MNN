// sherpa-mnn/csrc/online-cnn-bilstm-model-meta-data.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_

namespace sherpa_mnn {

struct OnlineCNNBiLSTMModelMetaData {
  int32_t comma_id = -1;
  int32_t period_id = -1;
  int32_t quest_id = -1;

  int32_t upper_id = -1;
  int32_t cap_id = -1;
  int32_t mix_case_id = -1;

  int32_t num_cases = -1;
  int32_t num_punctuations = -1;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_
