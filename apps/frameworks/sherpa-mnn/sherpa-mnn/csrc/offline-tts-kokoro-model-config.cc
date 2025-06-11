// sherpa-mnn/csrc/offline-tts-kokoro-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-kokoro-model-config.h"

#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

void OfflineTtsKokoroModelConfig::Register(ParseOptions *po) {
  po->Register("kokoro-model", &model, "Path to Kokoro model");
  po->Register("kokoro-voices", &voices,
               "Path to voices.bin for Kokoro models");
  po->Register("kokoro-tokens", &tokens,
               "Path to tokens.txt for Kokoro models");
  po->Register(
      "kokoro-lexicon", &lexicon,
      "Path to lexicon.txt for Kokoro models. Used only for Kokoro >= v1.0"
      "You can pass multiple files, separated by ','. Example: "
      "./lexicon-us-en.txt,./lexicon-zh.txt");
  po->Register("kokoro-data-dir", &data_dir,
               "Path to the directory containing dict for espeak-ng.");
  po->Register("kokoro-dict-dir", &dict_dir,
               "Path to the directory containing dict for jieba. "
               "Used only for Kokoro >= v1.0");
  po->Register("kokoro-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

bool OfflineTtsKokoroModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--kokoro-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--kokoro-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --kokoro-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (data_dir.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-data-dir");
    return false;
  }

  if (!FileExists(data_dir + "/phontab")) {
    SHERPA_ONNX_LOGE(
        "'%s/phontab' does not exist. Please check --kokoro-data-dir",
        data_dir.c_str());
    return false;
  }

  if (!FileExists(data_dir + "/phonindex")) {
    SHERPA_ONNX_LOGE(
        "'%s/phonindex' does not exist. Please check --kokoro-data-dir",
        data_dir.c_str());
    return false;
  }

  if (!FileExists(data_dir + "/phondata")) {
    SHERPA_ONNX_LOGE(
        "'%s/phondata' does not exist. Please check --kokoro-data-dir",
        data_dir.c_str());
    return false;
  }

  if (!FileExists(data_dir + "/intonations")) {
    SHERPA_ONNX_LOGE(
        "'%s/intonations' does not exist. Please check --kokoro-data-dir",
        data_dir.c_str());
    return false;
  }

  if (!dict_dir.empty()) {
    std::vector<std::string> required_files = {
        "jieba.dict.utf8", "hmm_model.utf8",  "user.dict.utf8",
        "idf.utf8",        "stop_words.utf8",
    };

    for (const auto &f : required_files) {
      if (!FileExists(dict_dir + "/" + f)) {
        SHERPA_ONNX_LOGE("'%s/%s' does not exist. Please check kokoro-dict-dir",
                         dict_dir.c_str(), f.c_str());
        return false;
      }
    }
  }

  return true;
}

std::string OfflineTtsKokoroModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsKokoroModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "voices=\"" << voices << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "dict_dir=\"" << dict_dir << "\", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_mnn
