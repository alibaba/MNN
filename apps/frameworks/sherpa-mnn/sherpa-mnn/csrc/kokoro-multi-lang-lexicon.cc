// sherpa-mnn/csrc/kokoro-multi-lang-lexicon.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/kokoro-multi-lang-lexicon.h"

#include <fstream>
#include <regex>  // NOLINT
#include <sstream>
#include <strstream>
#include <unordered_map>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include <codecvt>

#include "cppjieba/Jieba.hpp"
#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes);

class KokoroMultiLangLexicon::Impl {
 public:
  Impl(const std::string &tokens, const std::string &lexicon,
       const std::string &dict_dir, const std::string &data_dir,
       const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    InitTokens(tokens);

    InitLexicon(lexicon);

    InitJieba(dict_dir);

    InitEspeak(data_dir);  // See ./piper-phonemize-lexicon.cc
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &tokens, const std::string &lexicon,
       const std::string &dict_dir, const std::string &data_dir,
       const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    InitTokens(mgr, tokens);

    InitLexicon(mgr, lexicon);

    // we assume you have copied dict_dir and data_dir from assets to some path
    InitJieba(dict_dir);

    InitEspeak(data_dir);  // See ./piper-phonemize-lexicon.cc
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &_text) const {
    std::string text = ToLowerCase(_text);
    if (debug_) {
      SHERPA_ONNX_LOGE("After converting to lowercase:\n%s", text.c_str());
    }

    std::vector<std::pair<std::string, std::string>> replace_str_pairs = {
        {"，", ","}, {":", ","},  {"、", ","}, {"；", ";"},   {"：", ":"},
        {"。", "."}, {"？", "?"}, {"！", "!"}, {"\\s+", " "},
    };
    for (const auto &p : replace_str_pairs) {
      std::regex re(p.first);
      text = std::regex_replace(text, re, p.second);
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("After replacing punctuations and merging spaces:\n%s",
                       text.c_str());
    }

    // https://en.cppreference.com/w/cpp/regex
    // https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    std::string expr_chinese = "([\\u4e00-\\u9fff]+)";
    std::string expr_not_chinese = "([^\\u4e00-\\u9fff]+)";

    std::string expr_both = expr_chinese + "|" + expr_not_chinese;

    auto ws = ToWideString(text);
    std::wstring wexpr_both = ToWideString(expr_both);
    std::wregex we_both(wexpr_both);

    std::wstring wexpr_zh = ToWideString(expr_chinese);
    std::wregex we_zh(wexpr_zh);

    auto begin = std::wsregex_iterator(ws.begin(), ws.end(), we_both);
    auto end = std::wsregex_iterator();

    std::vector<TokenIDs> ans;

    for (std::wsregex_iterator i = begin; i != end; ++i) {
      std::wsmatch match = *i;
      std::wstring match_str = match.str();

      auto ms = ToString(match_str);
      uint8_t c = reinterpret_cast<const uint8_t *>(ms.data())[0];

      std::vector<std::vector<int32_t>> ids_vec;
      if (std::regex_match(match_str, we_zh)) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Chinese: %s", ms.c_str());
        }
        ids_vec = ConvertChineseToTokenIDs(ms);
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Non-Chinese: %s", ms.c_str());
        }

        ids_vec = ConvertEnglishToTokenIDs(ms, meta_data_.voice);
      }

      for (const auto &ids : ids_vec) {
        if (ids.size() > 10 + 2) {
          ans.emplace_back(ids);
        } else {
          if (ans.empty()) {
            ans.emplace_back(ids);
          } else {
            if (ans.back().tokens.size() + ids.size() < 50) {
              ans.back().tokens.back() = ids[1];
              ans.back().tokens.insert(ans.back().tokens.end(), ids.begin() + 2,
                                       ids.end());
            } else {
              ans.emplace_back(ids);
            }
          }
        }
      }
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v.tokens) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

 private:
  bool IsPunctuation(const std::string &text) const {
    if (text == ";" || text == ":" || text == "," || text == "." ||
        text == "!" || text == "?" || text == "—" || text == "…" ||
        text == "\"" || text == "(" || text == ")" || text == "“" ||
        text == "”") {
      return true;
    }

    return false;
  }

  std::vector<int32_t> ConvertWordToIds(const std::string &w) const {
    std::vector<int32_t> ans;
    if (word2ids_.count(w)) {
      ans = word2ids_.at(w);
      return ans;
    }

    std::vector<std::string> words = SplitUtf8(w);
    for (const auto &word : words) {
      if (word2ids_.count(word)) {
        auto ids = ConvertWordToIds(word);
        ans.insert(ans.end(), ids.begin(), ids.end());
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Skip OOV: '%s'", word.c_str());
        }
      }
    }

    return ans;
  }

  std::vector<std::vector<int32_t>> ConvertChineseToTokenIDs(
      const std::string &text) const {
    bool is_hmm = true;

    std::vector<std::string> words;
    jieba_->Cut(text, words, is_hmm);
    if (debug_) {
      std::ostringstream os;
      os << "After jieba processing:\n";

      std::string sep;
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    std::vector<std::vector<int32_t>> ans;
    std::vector<int32_t> this_sentence;
    int32_t max_len = meta_data_.max_token_len;

    this_sentence.push_back(0);
    for (const auto &w : words) {
      auto ids = ConvertWordToIds(w);
      if (this_sentence.size() + ids.size() > max_len - 2) {
        this_sentence.push_back(0);
        ans.push_back(std::move(this_sentence));

        this_sentence.push_back(0);
      }

      this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
    }

    if (this_sentence.size() > 1) {
      this_sentence.push_back(0);
      ans.push_back(std::move(this_sentence));
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

  std::vector<std::vector<int32_t>> ConvertEnglishToTokenIDs(
      const std::string &text, const std::string &voice) const {
    std::vector<std::string> words = SplitUtf8(text);
    if (debug_) {
      std::ostringstream os;
      os << "After splitting to words: ";
      std::string sep;
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    std::vector<std::vector<int32_t>> ans;
    int32_t max_len = meta_data_.max_token_len;
    std::vector<int32_t> this_sentence;

    int32_t space_id = token2id_.at(" ");

    this_sentence.push_back(0);

    for (const auto &word : words) {
      if (IsPunctuation(word)) {
        this_sentence.push_back(token2id_.at(word));

        if (this_sentence.size() > max_len - 2) {
          // this sentence is too long, split it
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
          continue;
        }

        if (word == "." || word == "!" || word == "?" || word == ";") {
          // Note: You can add more punctuations here to split the text
          // into sentences. We just use four here: .!?;
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }
      } else if (word2ids_.count(word)) {
        const auto &ids = word2ids_.at(word);
        if (this_sentence.size() + ids.size() + 3 > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }

        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
        this_sentence.push_back(space_id);
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Use espeak-ng to handle the OOV: '%s'",
                           word.c_str());
        }

        piper::eSpeakPhonemeConfig config;

        config.voice = voice;

        std::vector<std::vector<piper::Phoneme>> phonemes;

        CallPhonemizeEspeak(word, config, &phonemes);
        // Note phonemes[i] contains a vector of unicode codepoints;
        // we need to convert them to utf8

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

        std::vector<int32_t> ids;
        for (const auto &v : phonemes) {
          for (const auto p : v) {
            auto token = conv.to_bytes(p);
            if (token2id_.count(token)) {
              ids.push_back(token2id_.at(token));
            } else {
              if (debug_) {
                SHERPA_ONNX_LOGE("Skip OOV token '%s' from '%s'", token.c_str(),
                                 word.c_str());
              }
            }
          }
        }

        if (this_sentence.size() + ids.size() + 3 > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }

        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
        this_sentence.push_back(space_id);
      }
    }

    if (this_sentence.size() > 1) {
      this_sentence.push_back(0);
      ans.push_back(std::move(this_sentence));
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

  void InitTokens(const std::string &tokens) {
    std::ifstream is(tokens);
    InitTokens(is);
  }

  template <typename Manager>
  void InitTokens(Manager *mgr, const std::string &tokens) {
    auto buf = ReadFile(mgr, tokens);

    std::istrstream is(buf.data(), buf.size());
    InitTokens(is);
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);  // defined in ./symbol-table.cc
  }

  void InitLexicon(const std::string &lexicon) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      std::ifstream is(f);
      InitLexicon(is);
    }
  }

  template <typename Manager>
  void InitLexicon(Manager *mgr, const std::string &lexicon) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      auto buf = ReadFile(mgr, f);

      std::istrstream is(buf.data(), buf.size());
      InitLexicon(is);
    }
  }

  void InitLexicon(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;
    std::string token;

    std::string line;
    int32_t line_num = 0;
    int32_t num_warn = 0;
    while (std::getline(is, line)) {
      ++line_num;
      std::istringstream iss(line);

      token_list.clear();
      iss >> word;
      ToLowerCase(&word);

      if (word2ids_.count(word)) {
        num_warn += 1;
        if (num_warn < 10) {
          SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                           word.c_str(), line_num, line.c_str());
        }
        continue;
      }

      while (iss >> token) {
        token_list.push_back(std::move(token));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);

      if (ids.empty()) {
        SHERPA_ONNX_LOGE(
            "Invalid pronunciation for word '%s' at line %d:%s. Ignore it",
            word.c_str(), line_num, line.c_str());
        continue;
      }

      word2ids_.insert({std::move(word), std::move(ids)});
    }
  }

  void InitJieba(const std::string &dict_dir) {
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
  }

 private:
  OfflineTtsKokoroModelMetaData meta_data_;

  // word to token IDs
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  std::unique_ptr<cppjieba::Jieba> jieba_;
  bool debug_ = false;
};

KokoroMultiLangLexicon::~KokoroMultiLangLexicon() = default;

KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(tokens, lexicon, dict_dir, data_dir,
                                   meta_data, debug)) {}

template <typename Manager>
KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    Manager *mgr, const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(mgr, tokens, lexicon, dict_dir, data_dir,
                                   meta_data, debug)) {}

std::vector<TokenIDs> KokoroMultiLangLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug);
#endif

#if __OHOS__
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &lexicon, const std::string &dict_dir,
    const std::string &data_dir, const OfflineTtsKokoroModelMetaData &meta_data,
    bool debug);
#endif

}  // namespace sherpa_mnn
