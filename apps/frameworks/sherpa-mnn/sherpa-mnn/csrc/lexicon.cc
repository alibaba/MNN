// sherpa-mnn/csrc/lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/lexicon.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <strstream>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/symbol-table.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

static std::vector<std::string> ProcessHeteronyms(
    const std::vector<std::string> &words) {
  std::vector<std::string> ans;
  ans.reserve(words.size());

  int32_t num_words = static_cast<int32_t>(words.size());
  int32_t i = 0;
  int32_t prev = -1;
  while (i < num_words) {
    // start of a phrase #$|
    if ((i + 2 < num_words) && words[i] == "#" && words[i + 1] == "$" &&
        words[i + 2] == "|") {
      if (prev == -1) {
        prev = i + 3;
      }
      i = i + 3;
      continue;
    }

    // end of a phrase |$#
    if ((i + 2 < num_words) && words[i] == "|" && words[i + 1] == "$" &&
        words[i + 2] == "#") {
      if (prev != -1) {
        std::ostringstream os;
        for (int32_t k = prev; k < i; ++k) {
          if (words[k] != "|" && words[k] != "$" && words[k] != "#") {
            os << words[k];
          }
        }
        ans.push_back(os.str());

        prev = -1;
      }

      i += 3;
      continue;
    }

    if (prev == -1) {
      // not inside a phrase
      ans.push_back(words[i]);
    }

    ++i;
  }

  return ans;
}

std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto &s : tokens) {
    if (!token2id.count(s)) {
      return {};
    }
    int32_t id = token2id.at(s);
    ids.push_back(id);
  }

  return ids;
}

Lexicon::Lexicon(const std::string &lexicon, const std::string &tokens,
                 const std::string &punctuations, const std::string &language,
                 bool debug /*= false*/)
    : debug_(debug) {
  InitLanguage(language);

  {
    std::ifstream is(tokens);
    InitTokens(is);
  }

  {
    std::ifstream is(lexicon);
    InitLexicon(is);
  }

  InitPunctuations(punctuations);
}

template <typename Manager>
Lexicon::Lexicon(Manager *mgr, const std::string &lexicon,
                 const std::string &tokens, const std::string &punctuations,
                 const std::string &language, bool debug /*= false*/
                 )
    : debug_(debug) {
  InitLanguage(language);

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

  InitPunctuations(punctuations);
}

std::vector<TokenIDs> Lexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*voice*/ /*= ""*/) const {
  switch (language_) {
    case Language::kChinese:
      return ConvertTextToTokenIdsChinese(text);
    case Language::kNotChinese:
      return ConvertTextToTokenIdsNotChinese(text);
    default:
      SHERPA_ONNX_LOGE("Unknown language: %d", static_cast<int32_t>(language_));
      exit(-1);
  }

  return {};
}

std::vector<TokenIDs> Lexicon::ConvertTextToTokenIdsChinese(
    const std::string &_text) const {
  std::string text(_text);
  ToLowerCase(&text);

  std::vector<std::string> words = SplitUtf8(text);
  words = ProcessHeteronyms(words);

  if (debug_) {
    std::ostringstream os;

    os << "Input text in string: " << text << "\n";
    os << "Input text in bytes:";
    for (uint8_t c : text) {
      os << " 0x" << std::setfill('0') << std::setw(2) << std::right << std::hex
         << c;
    }
    os << "\n";
    os << "After splitting to words:";
    for (const auto &w : words) {
      os << " " << w;
    }
    os << "\n";

#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  std::vector<TokenIDs> ans;
  std::vector<int> this_sentence;

  int32_t blank = -1;
  if (token2id_.count(" ")) {
    blank = token2id_.at(" ");
  }

  int32_t sil = -1;
  int32_t eos = -1;
  if (token2id_.count("sil")) {
    sil = token2id_.at("sil");
    eos = token2id_.at("eos");
  }

  int32_t pad = -1;
  if (token2id_.count("#0")) {
    pad = token2id_.at("#0");
  }

  if (sil != -1) {
    this_sentence.push_back(sil);
  }

  for (const auto &w : words) {
    if (w == "." || w == ";" || w == "!" || w == "?" || w == "-" || w == ":" ||
        w == "。" || w == "；" || w == "！" || w == "？" || w == "：" ||
        w == "”" ||
        // not sentence break
        w == "," || w == "“" || w == "，" || w == "、") {
      if (punctuations_.count(w)) {
        if (token2id_.count(w)) {
          this_sentence.push_back(token2id_.at(w));
        } else if (pad != -1) {
          this_sentence.push_back(pad);
        } else if (sil != -1) {
          this_sentence.push_back(sil);
        }
      }

      if (w != "," && w != "“" && w != "，" && w != "、") {
        if (eos != -1) {
          this_sentence.push_back(eos);
        }
        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};

        if (sil != -1) {
          this_sentence.push_back(sil);
        }
      }
      continue;
    }

    if (!word2ids_.count(w)) {
      SHERPA_ONNX_LOGE("OOV %s. Ignore it!", w.c_str());
      continue;
    }

    const auto &token_ids = word2ids_.at(w);
    this_sentence.insert(this_sentence.end(), token_ids.begin(),
                         token_ids.end());
    if (blank != -1) {
      this_sentence.push_back(blank);
    }
  }

  if (sil != -1) {
    this_sentence.push_back(sil);
  }

  if (eos != -1) {
    this_sentence.push_back(eos);
  }
  ans.emplace_back(std::move(this_sentence));

  return ans;
}

std::vector<TokenIDs> Lexicon::ConvertTextToTokenIdsNotChinese(
    const std::string &_text) const {
  std::string text(_text);
  ToLowerCase(&text);

  std::vector<std::string> words = SplitUtf8(text);

  if (debug_) {
    std::ostringstream os;

    os << "Input text (lowercase) in string: " << text << "\n";
    os << "Input text in bytes:";
    for (uint8_t c : text) {
      os << " 0x" << std::setfill('0') << std::setw(2) << std::right << std::hex
         << c;
    }
    os << "\n";
    os << "After splitting to words:";
    for (const auto &w : words) {
      os << " " << w;
    }
    os << "\n";

#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  int32_t blank = token2id_.at(" ");

  std::vector<TokenIDs> ans;
  std::vector<int> this_sentence;

  for (const auto &w : words) {
    if (w == "." || w == ";" || w == "!" || w == "?" || w == "-" || w == ":" ||
        // not sentence break
        w == ",") {
      if (punctuations_.count(w)) {
        this_sentence.push_back(token2id_.at(w));
      }

      if (w != ",") {
        this_sentence.push_back(blank);
        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};
      }

      continue;
    }

    if (!word2ids_.count(w)) {
      SHERPA_ONNX_LOGE("OOV %s. Ignore it!", w.c_str());
      continue;
    }

    const auto &token_ids = word2ids_.at(w);
    this_sentence.insert(this_sentence.end(), token_ids.begin(),
                         token_ids.end());
    this_sentence.push_back(blank);
  }

  if (!this_sentence.empty()) {
    // remove the last blank
    this_sentence.resize(this_sentence.size() - 1);
  }

  if (!this_sentence.empty()) {
    ans.emplace_back(std::move(this_sentence));
  }

  return ans;
}

void Lexicon::InitTokens(std::istream &is) { token2id_ = ReadTokens(is); }

void Lexicon::InitLanguage(const std::string &_lang) {
  std::string lang(_lang);
  ToLowerCase(&lang);
  if (lang == "chinese") {
    language_ = Language::kChinese;
  } else if (!lang.empty()) {
    language_ = Language::kNotChinese;
  } else {
    SHERPA_ONNX_LOGE("Unknown language: %s", _lang.c_str());
    exit(-1);
  }
}

void Lexicon::InitLexicon(std::istream &is) {
  std::string word;
  std::vector<std::string> token_list;
  std::string line;
  std::string phone;

  while (std::getline(is, line)) {
    std::istringstream iss(line);

    token_list.clear();

    iss >> word;
    ToLowerCase(&word);

    if (word2ids_.count(word)) {
      SHERPA_ONNX_LOGE("Duplicated word: %s. Ignore it.", word.c_str());
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

void Lexicon::InitPunctuations(const std::string &punctuations) {
  std::vector<std::string> punctuation_list;
  SplitStringToVector(punctuations, " ", false, &punctuation_list);
  for (auto &s : punctuation_list) {
    punctuations_.insert(std::move(s));
  }
}

#if __ANDROID_API__ >= 9
template Lexicon::Lexicon(AAssetManager *mgr, const std::string &lexicon,
                          const std::string &tokens,
                          const std::string &punctuations,
                          const std::string &language, bool debug = false);
#endif

#if __OHOS__
template Lexicon::Lexicon(NativeResourceManager *mgr,
                          const std::string &lexicon, const std::string &tokens,
                          const std::string &punctuations,
                          const std::string &language, bool debug = false);
#endif

}  // namespace sherpa_mnn
