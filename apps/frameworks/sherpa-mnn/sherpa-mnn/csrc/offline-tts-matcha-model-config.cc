// sherpa-mnn/csrc/offline-tts-matcha-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-matcha-model-config.h"

#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineTtsMatchaModelConfig::Register(ParseOptions *po) {
  po->Register("matcha-acoustic-model", &acoustic_model,
               "Path to matcha acoustic model");
  po->Register("matcha-vocoder", &vocoder, "Path to matcha vocoder");
  po->Register("matcha-lexicon", &lexicon,
               "Path to lexicon.txt for Matcha models");
  po->Register("matcha-tokens", &tokens,
               "Path to tokens.txt for Matcha models");
  po->Register("matcha-data-dir", &data_dir,
               "Path to the directory containing dict for espeak-ng. If it is "
               "given, --matcha-lexicon is ignored.");
  po->Register("matcha-dict-dir", &dict_dir,
               "Path to the directory containing dict for jieba. Used only for "
               "Chinese TTS models using jieba");
  po->Register("matcha-noise-scale", &noise_scale,
               "noise_scale for Matcha models");
  po->Register("matcha-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

bool OfflineTtsMatchaModelConfig::Validate() const {
  if (acoustic_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-acoustic-model");
    return false;
  }

  if (!FileExists(acoustic_model)) {
    SHERPA_ONNX_LOGE("--matcha-acoustic-model: '%s' does not exist",
                     acoustic_model.c_str());
    return false;
  }

  if (vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-vocoder");
    return false;
  }

  if (!FileExists(vocoder)) {
    SHERPA_ONNX_LOGE("--matcha-vocoder: '%s' does not exist", vocoder.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--matcha-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    if (!FileExists(data_dir + "/phontab")) {
      SHERPA_ONNX_LOGE(
          "'%s/phontab' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phonindex")) {
      SHERPA_ONNX_LOGE(
          "'%s/phonindex' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phondata")) {
      SHERPA_ONNX_LOGE(
          "'%s/phondata' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/intonations")) {
      SHERPA_ONNX_LOGE(
          "'%s/intonations' does not exist. Please check --matcha-data-dir",
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
        SHERPA_ONNX_LOGE(
            "'%s/%s' does not exist. Please check --matcha-dict-dir",
            dict_dir.c_str(), f.c_str());
        return false;
      }
    }

    // we require that --matcha-lexicon is not empty
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide --matcha-lexicon");
      return false;
    }

    if (!FileExists(lexicon)) {
      SHERPA_ONNX_LOGE("--matcha-lexicon: '%s' does not exist",
                       lexicon.c_str());
      return false;
    }
  }

  return true;
}

std::string OfflineTtsMatchaModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsMatchaModelConfig(";
  os << "acoustic_model=\"" << acoustic_model << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "dict_dir=\"" << dict_dir << "\", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_mnn
