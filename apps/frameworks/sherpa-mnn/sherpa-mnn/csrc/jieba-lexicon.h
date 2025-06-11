// sherpa-mnn/csrc/jieba-lexicon.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_
#define SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-mnn/csrc/offline-tts-frontend.h"

namespace sherpa_mnn {

class JiebaLexicon : public OfflineTtsFrontend {
 public:
  ~JiebaLexicon() override;

  JiebaLexicon(const std::string &lexicon, const std::string &tokens,
               const std::string &dict_dir, bool debug);

  template <typename Manager>
  JiebaLexicon(Manager *mgr, const std::string &lexicon,
               const std::string &tokens, const std::string &dict_dir,
               bool debug);

  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text,
      const std::string &unused_voice = "") const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_
