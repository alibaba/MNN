// sherpa-mnn/csrc/online-wenet-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-wenet-ctc-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OnlineWenetCtcModelConfig::Register(ParseOptions *po) {
  po->Register("wenet-ctc-model", &model,
               "Path to CTC model.onnx from WeNet. Please see "
               "https://github.com/k2-fsa/sherpa-mnn/pull/425");
  po->Register("wenet-ctc-chunk-size", &chunk_size,
               "Chunk size after subsampling used for decoding.");
  po->Register("wenet-ctc-num-left-chunks", &num_left_chunks,
               "Number of left chunks after subsampling used for decoding.");
}

bool OnlineWenetCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("WeNet CTC model '%s' does not exist", model.c_str());
    return false;
  }

  if (chunk_size <= 0) {
    SHERPA_ONNX_LOGE(
        "Please specify a positive value for --wenet-ctc-chunk-size. Currently "
        "given: %d",
        chunk_size);
    return false;
  }

  if (num_left_chunks <= 0) {
    SHERPA_ONNX_LOGE(
        "Please specify a positive value for --wenet-ctc-num-left-chunks. "
        "Currently given: %d. Note that if you want to use -1, please consider "
        "using a non-streaming model.",
        num_left_chunks);
    return false;
  }

  return true;
}

std::string OnlineWenetCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineWenetCtcModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "chunk_size=" << chunk_size << ", ";
  os << "num_left_chunks=" << num_left_chunks << ")";

  return os.str();
}

}  // namespace sherpa_mnn
