// sherpa-mnn/csrc/offline-tts-matcha-model-meta-data.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_mnn {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
struct OfflineTtsMatchaModelMetaData {
  int32_t sample_rate = 0;
  int32_t num_speakers = 0;
  int32_t version = 1;
  int32_t jieba = 0;
  int32_t has_espeak = 0;
  int32_t use_eos_bos = 0;
  int32_t pad_id = 0;

  std::string voice;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_META_DATA_H_
