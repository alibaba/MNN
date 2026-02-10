// sherpa-mnn/csrc/lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LEXICON_H_
#define SHERPA_ONNX_CSRC_LEXICON_H_

#include <cstdint>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sherpa-mnn/csrc/offline-tts-frontend.h"

namespace sherpa_mnn {

class Lexicon : public OfflineTtsFrontend {
 public:
  Lexicon() = default;  // for subclasses
                        //
  // Note: for models from piper, we won't use this class.
  Lexicon(const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations, const std::string &language,
          bool debug = false);

  template <typename Manager>
  Lexicon(Manager *mgr, const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations, const std::string &language,
          bool debug = false);

  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  std::vector<TokenIDs> ConvertTextToTokenIdsNotChinese(
      const std::string &text) const;

  std::vector<TokenIDs> ConvertTextToTokenIdsChinese(
      const std::string &text) const;

  void InitLanguage(const std::string &lang);
  void InitTokens(std::istream &is);
  void InitLexicon(std::istream &is);
  void InitPunctuations(const std::string &punctuations);

 private:
  enum class Language {
    kNotChinese,
    kChinese,
    kUnknown,
  };

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> punctuations_;
  std::unordered_map<std::string, int32_t> token2id_;
  Language language_ = Language::kUnknown;
  bool debug_ = false;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_LEXICON_H_
