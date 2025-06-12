// sherpa-mnn/csrc/jieba-lexicon.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/jieba-lexicon.h"

#include <fstream>
#include <regex>  // NOLINT
#include <strstream>
#include <unordered_set>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "cppjieba/Jieba.hpp"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

static bool IsPunct(const std::string &s) {
  static const std::unordered_set<std::string> puncts = {
      ",",  ".",  "!",  "?", ":", "\"", "'", "，",
      "。", "！", "？", "“", "”", "‘",  "’",
  };
  return puncts.count(s);
}

class JiebaLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir, bool debug)
      : debug_(debug) {
    std::string dict = dict_dir + "/jieba.dict.utf8";
    std::string hmm = dict_dir + "/hmm_model.utf8";
    std::string user_dict = dict_dir + "/user.dict.utf8";
    std::string idf = dict_dir + "/idf.utf8";
    std::string stop_word = dict_dir + "/stop_words.utf8";

    AssertFileExists(dict);
    AssertFileExists(hmm);
    AssertFileExists(user_dict);
    AssertFileExists(idf);
    AssertFileExists(stop_word);

    jieba_ =
        std::make_unique<cppjieba::Jieba>(dict, hmm, user_dict, idf, stop_word);

    {
      std::ifstream is(tokens);
      InitTokens(is);
    }

    {
      std::ifstream is(lexicon);
      InitLexicon(is);
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir, bool debug)
      : debug_(debug) {
    std::string dict = dict_dir + "/jieba.dict.utf8";
    std::string hmm = dict_dir + "/hmm_model.utf8";
    std::string user_dict = dict_dir + "/user.dict.utf8";
    std::string idf = dict_dir + "/idf.utf8";
    std::string stop_word = dict_dir + "/stop_words.utf8";

    AssertFileExists(dict);
    AssertFileExists(hmm);
    AssertFileExists(user_dict);
    AssertFileExists(idf);
    AssertFileExists(stop_word);

    jieba_ =
        std::make_unique<cppjieba::Jieba>(dict, hmm, user_dict, idf, stop_word);

    {
      auto buf = ReadFile(mgr, tokens);
      std::istrstream is(buf.data(), buf.size());

      InitTokens(is);
    }

    {
      auto buf = ReadFile(mgr, lexicon);
      std::istrstream is(buf.data(), buf.size());
      InitLexicon(is);
    }
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &text) const {
    // see
    // https://github.com/Plachtaa/VITS-fast-fine-tuning/blob/main/text/mandarin.py#L244
    std::regex punct_re{"：|、|；"};
    std::string s = std::regex_replace(text, punct_re, "，");

    std::regex punct_re2("[.]");
    s = std::regex_replace(s, punct_re2, "。");

    std::regex punct_re3("[?]");
    s = std::regex_replace(s, punct_re3, "？");

    std::regex punct_re4("[!]");
    s = std::regex_replace(s, punct_re4, "！");

    std::vector<std::string> words;
    bool is_hmm = true;
    jieba_->Cut(text, words, is_hmm);

    if (debug_) {
#if __OHOS__
      SHERPA_ONNX_LOGE("input text:\n%{public}s", text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%{public}s", s.c_str());
#else
      SHERPA_ONNX_LOGE("input text:\n%s", text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%s", s.c_str());
#endif

      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("after jieba processing:\n%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("after jieba processing:\n%s", os.str().c_str());
#endif
    }

    // remove spaces after punctuations
    std::vector<std::string> words2 = std::move(words);
    words.reserve(words2.size());

    for (int32_t i = 0; i < words2.size(); ++i) {
      if (i == 0) {
        words.push_back(std::move(words2[i]));
      } else if (words2[i] == " ") {
        if (words.back() == " " || IsPunct(words.back())) {
          continue;
        } else {
          words.push_back(std::move(words2[i]));
        }
      } else if (IsPunct(words2[i])) {
        if (words.back() == " " || IsPunct(words.back())) {
          continue;
        } else {
          words.push_back(std::move(words2[i]));
        }
      } else {
        words.push_back(std::move(words2[i]));
      }
    }

    if (debug_) {
      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("after removing spaces after punctuations:\n%{public}s",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("after removing spaces after punctuations:\n%s",
                       os.str().c_str());
#endif
    }

    std::vector<TokenIDs> ans;
    std::vector<int> this_sentence;

    for (const auto &w : words) {
      auto ids = ConvertWordToIds(w);
      if (ids.empty()) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Ignore OOV '%{public}s'", w.c_str());
#else
        SHERPA_ONNX_LOGE("Ignore OOV '%s'", w.c_str());
#endif
        continue;
      }

      this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());

      if (IsPunct(w)) {
        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};
      }
    }  // for (const auto &w : words)

    if (!this_sentence.empty()) {
      ans.emplace_back(std::move(this_sentence));
    }

    return ans;
  }

 private:
  std::vector<int32_t> ConvertWordToIds(const std::string &w) const {
    if (word2ids_.count(w)) {
      return word2ids_.at(w);
    }

    if (token2id_.count(w)) {
      return {token2id_.at(w)};
    }

    std::vector<int32_t> ans;

    std::vector<std::string> words = SplitUtf8(w);
    for (const auto &word : words) {
      if (word2ids_.count(word)) {
        auto ids = ConvertWordToIds(word);
        ans.insert(ans.end(), ids.begin(), ids.end());
      }
    }

    return ans;
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);

    std::vector<std::pair<std::string, std::string>> puncts = {
        {",", "，"}, {".", "。"}, {"!", "！"}, {"?", "？"}, {":", "："},
        {"\"", "“"}, {"\"", "”"}, {"'", "‘"},  {"'", "’"},  {";", "；"},
    };

    for (const auto &p : puncts) {
      if (token2id_.count(p.first) && !token2id_.count(p.second)) {
        token2id_[p.second] = token2id_[p.first];
      }

      if (!token2id_.count(p.first) && token2id_.count(p.second)) {
        token2id_[p.first] = token2id_[p.second];
      }
    }

    if (!token2id_.count("、") && token2id_.count("，")) {
      token2id_["、"] = token2id_["，"];
    }

    if (!token2id_.count(";") && token2id_.count(",")) {
      token2id_[";"] = token2id_[","];
    }
  }

  void InitLexicon(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;
    std::string line;
    std::string phone;
    int32_t line_num = 0;

    while (std::getline(is, line)) {
      ++line_num;

      std::istringstream iss(line);

      token_list.clear();

      iss >> word;
      ToLowerCase(&word);

      if (word2ids_.count(word)) {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "Duplicated word: %{public}s at line %{public}d:%{public}s. Ignore "
            "it.",
            word.c_str(), line_num, line.c_str());
#else
        SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                         word.c_str(), line_num, line.c_str());
#endif
        continue;
      }

      while (iss >> phone) {
        token_list.push_back(std::move(phone));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);
      if (ids.empty()) {
        continue;
      }

      word2ids_.insert({std::move(word), std::move(ids)});
    }
  }

 private:
  // lexicon.txt is saved in word2ids_
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  std::unique_ptr<cppjieba::Jieba> jieba_;
  bool debug_ = false;
};

JiebaLexicon::~JiebaLexicon() = default;

JiebaLexicon::JiebaLexicon(const std::string &lexicon,
                           const std::string &tokens,
                           const std::string &dict_dir, bool debug)
    : impl_(std::make_unique<Impl>(lexicon, tokens, dict_dir, debug)) {}

template <typename Manager>
JiebaLexicon::JiebaLexicon(Manager *mgr, const std::string &lexicon,
                           const std::string &tokens,
                           const std::string &dict_dir, bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, dict_dir, debug)) {}

std::vector<TokenIDs> JiebaLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template JiebaLexicon::JiebaLexicon(AAssetManager *mgr,
                                    const std::string &lexicon,
                                    const std::string &tokens,
                                    const std::string &dict_dir, bool debug);
#endif

#if __OHOS__
template JiebaLexicon::JiebaLexicon(NativeResourceManager *mgr,
                                    const std::string &lexicon,
                                    const std::string &tokens,
                                    const std::string &dict_dir, bool debug);
#endif

}  // namespace sherpa_mnn
