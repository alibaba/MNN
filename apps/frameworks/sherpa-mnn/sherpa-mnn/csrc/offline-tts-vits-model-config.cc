// sherpa-mnn/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-vits-model-config.h"

#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineTtsVitsModelConfig::Register(ParseOptions *po) {
  po->Register("vits-model", &model, "Path to VITS model");
  po->Register("vits-lexicon", &lexicon, "Path to lexicon.txt for VITS models");
  po->Register("vits-tokens", &tokens, "Path to tokens.txt for VITS models");
  po->Register("vits-data-dir", &data_dir,
               "Path to the directory containing dict for espeak-ng. If it is "
               "given, --vits-lexicon is ignored.");
  po->Register("vits-dict-dir", &dict_dir,
               "Path to the directory containing dict for jieba. Used only for "
               "Chinese TTS models using jieba");
  po->Register("vits-noise-scale", &noise_scale, "noise_scale for VITS models");
  po->Register("vits-noise-scale-w", &noise_scale_w,
               "noise_scale_w for VITS models");
  po->Register("vits-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

bool OfflineTtsVitsModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--vits-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--vits-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    if (!FileExists(data_dir + "/phontab")) {
      SHERPA_ONNX_LOGE(
          "'%s/phontab' does not exist. Please check --vits-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phonindex")) {
      SHERPA_ONNX_LOGE(
          "'%s/phonindex' does not exist. Please check --vits-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phondata")) {
      SHERPA_ONNX_LOGE(
          "'%s/phondata' does not exist. Please check --vits-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/intonations")) {
      SHERPA_ONNX_LOGE(
          "'%s/intonations' does not exist. Please check --vits-data-dir",
          data_dir.c_str());
      return false;
    }
  }

  if (!dict_dir.empty()) {
    std::vector<std::string> required_files = {
        "jieba.dict.utf8", "hmm_model.utf8",  "user.dict.utf8",
        "idf.utf8",        "stop_words.utf8",
    };

    for (const auto &f : required_files) {
      if (!FileExists(dict_dir + "/" + f)) {
        SHERPA_ONNX_LOGE("'%s/%s' does not exist. Please check vits-dict-dir",
                         dict_dir.c_str(), f.c_str());
        return false;
      }
    }
  }
  return true;
}

std::string OfflineTtsVitsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsVitsModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "dict_dir=\"" << dict_dir << "\", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "noise_scale_w=" << noise_scale_w << ", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_mnn
