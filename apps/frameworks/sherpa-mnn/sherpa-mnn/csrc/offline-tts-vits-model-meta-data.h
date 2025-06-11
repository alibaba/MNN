// sherpa-mnn/csrc/offline-tts-vits-model-meta-data.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_mnn {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
struct OfflineTtsVitsModelMetaData {
  int32_t sample_rate = 0;
  int32_t add_blank = 0;
  int32_t num_speakers = 0;

  bool is_piper = false;
  bool is_coqui = false;
  bool is_icefall = false;
  bool is_melo_tts = false;

  // for Chinese TTS models from
  // https://github.com/Plachtaa/VITS-fast-fine-tuning
  int32_t jieba = 0;

  // the following options are for models from coqui-ai/TTS
  int32_t blank_id = 0;
  int32_t bos_id = 0;
  int32_t eos_id = 0;
  int32_t use_eos_bos = 0;
  int32_t pad_id = 0;

  // for melo tts
  int32_t speaker_id = 0;
  int32_t version = 0;

  std::string punctuations;
  std::string language;
  std::string voice;
  std::string frontend;  // characters
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
