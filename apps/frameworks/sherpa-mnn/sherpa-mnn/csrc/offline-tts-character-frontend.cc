// sherpa-mnn/csrc/offline-tts-character-frontend.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <locale>
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
#include "sherpa-mnn/csrc/offline-tts-character-frontend.h"

namespace sherpa_mnn {

static std::unordered_map<char32_t, int32_t> ReadTokens(std::istream &is) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::unordered_map<char32_t, int32_t> token2id;

  std::string line;

  std::string sym;
  std::u32string s;
  int32_t id = 0;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error when reading tokens: %s", line.c_str());
      exit(-1);
    }

    // Form models from coqui-ai/TTS, we have saved the IDs of the following
    // symbols in OfflineTtsVitsModelMetaData, so it is safe to skip them here.
    if (sym == "<PAD>" || sym == "<EOS>" || sym == "<BOS>" || sym == "<BLNK>") {
      continue;
    }

    s = conv.from_bytes(sym);
    if (s.size() != 1) {
      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      exit(-1);
    }

    char32_t c = s[0];

    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      exit(-1);
    }

    token2id.insert({c, id});
  }

  return token2id;
}

OfflineTtsCharacterFrontend::OfflineTtsCharacterFrontend(
    const std::string &tokens, const OfflineTtsVitsModelMetaData &meta_data)
    : meta_data_(meta_data) {
  std::ifstream is(tokens);
  token2id_ = ReadTokens(is);
}

template <typename Manager>
OfflineTtsCharacterFrontend::OfflineTtsCharacterFrontend(
    Manager *mgr, const std::string &tokens,
    const OfflineTtsVitsModelMetaData &meta_data)
    : meta_data_(meta_data) {
  auto buf = ReadFile(mgr, tokens);
  std::istrstream is(buf.data(), buf.size());
  token2id_ = ReadTokens(is);
}

std::vector<TokenIDs> OfflineTtsCharacterFrontend::ConvertTextToTokenIds(
    const std::string &_text, const std::string & /*voice = ""*/) const {
  // see
  // https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py#L87
  int32_t use_eos_bos = meta_data_.use_eos_bos;
  int32_t bos_id = meta_data_.bos_id;
  int32_t eos_id = meta_data_.eos_id;
  int32_t blank_id = meta_data_.blank_id;
  int32_t add_blank = meta_data_.add_blank;

  std::string text(_text.size(), 0);
  std::transform(_text.begin(), _text.end(), text.begin(),
                 [](auto c) { return std::tolower(c); });

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string s = conv.from_bytes(text);

  std::vector<TokenIDs> ans;

  std::vector<int> this_sentence;
  if (add_blank) {
    if (use_eos_bos) {
      this_sentence.push_back(bos_id);
    }

    this_sentence.push_back(blank_id);

    for (char32_t c : s) {
      if (token2id_.count(c)) {
        this_sentence.push_back(token2id_.at(c));
        this_sentence.push_back(blank_id);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown character. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(c));
      }

      if (c == '.' || c == ':' || c == '?' || c == '!') {
        // end of a sentence
        if (use_eos_bos) {
          this_sentence.push_back(eos_id);
        }

        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};

        // re-initialize this_sentence
        if (use_eos_bos) {
          this_sentence.push_back(bos_id);
        }
        this_sentence.push_back(blank_id);
      }
    }

    if (use_eos_bos) {
      this_sentence.push_back(eos_id);
    }

    if (static_cast<int32_t>(this_sentence.size()) > 1 + use_eos_bos) {
      ans.emplace_back(std::move(this_sentence));
    }
  } else {
    // not adding blank
    if (use_eos_bos) {
      this_sentence.push_back(bos_id);
    }

    for (char32_t c : s) {
      if (token2id_.count(c)) {
        this_sentence.push_back(token2id_.at(c));
      }

      if (c == '.' || c == ':' || c == '?' || c == '!') {
        // end of a sentence
        if (use_eos_bos) {
          this_sentence.push_back(eos_id);
        }

        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};

        // re-initialize this_sentence
        if (use_eos_bos) {
          this_sentence.push_back(bos_id);
        }
      }
    }

    if (this_sentence.size() > 1) {
      ans.emplace_back(std::move(this_sentence));
    }
  }

  return ans;
}

#if __ANDROID_API__ >= 9
template OfflineTtsCharacterFrontend::OfflineTtsCharacterFrontend(
    AAssetManager *mgr, const std::string &tokens,
    const OfflineTtsVitsModelMetaData &meta_data);

#endif

#if __OHOS__
template OfflineTtsCharacterFrontend::OfflineTtsCharacterFrontend(
    NativeResourceManager *mgr, const std::string &tokens,
    const OfflineTtsVitsModelMetaData &meta_data);

#endif

}  // namespace sherpa_mnn
