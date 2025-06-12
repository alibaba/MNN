// sherpa-mnn/csrc/offline-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-recognizer.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-lm-config.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

void OfflineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  lm_config.Register(po);
  ctc_fst_decoder_config.Register(po);

  po->Register(
      "decoding-method", &decoding_method,
      "decoding method,"
      "Valid values: greedy_search, modified_beam_search. "
      "modified_beam_search is applicable only for transducer models.");

  po->Register("max-active-paths", &max_active_paths,
               "Used only when decoding_method is modified_beam_search");

  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");

  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, For example: "
      "HELLO WORLD"
      "你好世界");

  po->Register("hotwords-score", &hotwords_score,
               "The bonus score for each token in context word/phrase. "
               "Used only when decoding_method is modified_beam_search");

  po->Register(
      "rule-fsts", &rule_fsts,
      "If not empty, it specifies fsts for inverse text normalization. "
      "If there are multiple fsts, they are separated by a comma.");

  po->Register(
      "rule-fars", &rule_fars,
      "If not empty, it specifies fst archives for inverse text normalization. "
      "If there are multiple archives, they are separated by a comma.");
}

bool OfflineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths is less than 0! Given: %d",
                       max_active_paths);
      return false;
    }
    if (!lm_config.Validate()) {
      return false;
    }
  }

  if (!hotwords_file.empty() && decoding_method != "modified_beam_search") {
    SHERPA_ONNX_LOGE(
        "Please use --decoding-method=modified_beam_search if you"
        " provide --hotwords-file. Given --decoding-method='%s'",
        decoding_method.c_str());
    return false;
  }

  if (!ctc_fst_decoder_config.graph.empty() &&
      !ctc_fst_decoder_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in fst_decoder");
    return false;
  }

  if (!hotwords_file.empty() && !FileExists(hotwords_file)) {
    SHERPA_ONNX_LOGE("--hotwords-file: '%s' does not exist",
                     hotwords_file.c_str());
    return false;
  }

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule far '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  return model_config.Validate();
}

std::string OfflineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "ctc_fst_decoder_config=" << ctc_fst_decoder_config.ToString() << ", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "blank_penalty=" << blank_penalty << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\")";

  return os.str();
}

template <typename Manager>
OfflineRecognizer::OfflineRecognizer(Manager *mgr,
                                     const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(mgr, config)) {}

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(config)) {}

OfflineRecognizer::~OfflineRecognizer() = default;

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream(
    const std::string &hotwords) const {
  return impl_->CreateStream(hotwords);
}

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

void OfflineRecognizer::SetConfig(const OfflineRecognizerConfig &config) {
  impl_->SetConfig(config);
}

OfflineRecognizerConfig OfflineRecognizer::GetConfig() const {
  return impl_->GetConfig();
}

#if __ANDROID_API__ >= 9
template OfflineRecognizer::OfflineRecognizer(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizer::OfflineRecognizer(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_mnn
