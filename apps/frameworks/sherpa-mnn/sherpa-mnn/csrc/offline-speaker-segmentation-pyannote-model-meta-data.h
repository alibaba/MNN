// sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model-meta-data.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_mnn {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
struct OfflineSpeakerSegmentationPyannoteModelMetaData {
  int32_t sample_rate = 0;
  int32_t window_size = 0;            // in samples
  int32_t window_shift = 0;           // in samples
  int32_t receptive_field_size = 0;   // in samples
  int32_t receptive_field_shift = 0;  // in samples
  int32_t num_speakers = 0;
  int32_t powerset_max_classes = 0;
  int32_t num_classes = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_META_DATA_H_
