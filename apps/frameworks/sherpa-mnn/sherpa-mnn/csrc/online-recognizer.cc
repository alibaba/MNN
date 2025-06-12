// sherpa-mnn/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-mnn/csrc/online-recognizer.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/online-recognizer-impl.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

namespace {

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <typename T>
std::string VecToString(const std::vector<T> &vec, int32_t precision = 6) {
  std::ostringstream oss;
  if (precision != 0) {
    oss << std::fixed << std::setprecision(precision);
  }
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << item;
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <>  // explicit specialization for T = std::string
std::string VecToString<std::string>(const std::vector<std::string> &vec,
                                     int32_t) {  // ignore 2nd arg
  std::ostringstream oss;
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << std::quoted(item);
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

}  // namespace

std::string OnlineRecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{ ";
  os << "\"text\": " << std::quoted(text) << ", ";
  os << "\"tokens\": " << VecToString(tokens) << ", ";
  os << "\"timestamps\": " << VecToString(timestamps, 2) << ", ";
  os << "\"ys_probs\": " << VecToString(ys_probs, 6) << ", ";
  os << "\"lm_probs\": " << VecToString(lm_probs, 6) << ", ";
  os << "\"context_scores\": " << VecToString(context_scores, 6) << ", ";
  os << "\"segment\": " << segment << ", ";
  os << "\"words\": " << VecToString(words, 0) << ", ";
  os << "\"start_time\": " << std::fixed << std::setprecision(2) << start_time
     << ", ";
  os << "\"is_final\": " << (is_final ? "true" : "false");
  os << "}";
  return os.str();
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);
  lm_config.Register(po);
  ctc_fst_decoder_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");
  po->Register("hotwords-score", &hotwords_score,
               "The bonus score for each token in context word/phrase. "
               "Used only when decoding_method is modified_beam_search");
  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, For example: "
      "HELLO WORLD"
      "你好世界");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search and modified_beam_search.");
  po->Register("temperature-scale", &temperature_scale,
               "Temperature scale for confidence computation in decoding.");
  po->Register(
      "rule-fsts", &rule_fsts,
      "If not empty, it specifies fsts for inverse text normalization. "
      "If there are multiple fsts, they are separated by a comma.");

  po->Register(
      "rule-fars", &rule_fars,
      "If not empty, it specifies fst archives for inverse text normalization. "
      "If there are multiple archives, they are separated by a comma.");
}

bool OnlineRecognizerConfig::Validate() const {
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
        " provide --hotwords-file. Given --decoding-method=%s",
        decoding_method.c_str());
    return false;
  }

  if (!ctc_fst_decoder_config.graph.empty() &&
      !ctc_fst_decoder_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in ctc_fst_decoder_config");
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

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "ctc_fst_decoder_config=" << ctc_fst_decoder_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "blank_penalty=" << blank_penalty << ", ";
  os << "temperature_scale=" << temperature_scale << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\")";

  return os.str();
}

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(config)) {}

template <typename Manager>
OnlineRecognizer::OnlineRecognizer(Manager *mgr,
                                   const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(mgr, config)) {}

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream(
    const std::string &hotwords) const {
  return impl_->CreateStream(hotwords);
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::WarmpUpRecognizer(int32_t warmup, int32_t mbs) const {
  if (warmup > 0) {
    impl_->WarmpUpRecognizer(warmup, mbs);
  }
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void OnlineRecognizer::Reset(OnlineStream *s) const { impl_->Reset(s); }

#if __ANDROID_API__ >= 9
template OnlineRecognizer::OnlineRecognizer(
    AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

#if __OHOS__
template OnlineRecognizer::OnlineRecognizer(
    NativeResourceManager *mgr, const OnlineRecognizerConfig &config);
#endif

}  // namespace sherpa_mnn
