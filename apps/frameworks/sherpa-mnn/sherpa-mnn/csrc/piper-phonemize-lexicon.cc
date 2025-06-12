// sherpa-mnn/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/piper-phonemize-lexicon.h"

#include <codecvt>
#include <fstream>
#include <locale>
#include <map>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes) {
  static std::mutex espeak_mutex;

  std::lock_guard<std::mutex> lock(espeak_mutex);

  // keep multi threads from calling into piper::phonemize_eSpeak
  piper::phonemize_eSpeak(text, config, *phonemes);
}

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

    s = conv.from_bytes(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }

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

// see the function "phonemes_to_ids" from
// https://github.com/rhasspy/piper/blob/master/notebooks/piper_inference_(ONNX).ipynb
static std::vector<int> PiperPhonemesToIdsVits(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes) {
  // see
  // https://github.com/rhasspy/piper-phonemize/blob/master/src/phoneme_ids.hpp#L17
  int32_t pad = token2id.at(U'_');
  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  std::vector<int> ans;
  ans.reserve(phonemes.size());

  ans.push_back(bos);
  for (auto p : phonemes) {
    if (token2id.count(p)) {
      ans.push_back(token2id.at(p));
      ans.push_back(pad);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }
  ans.push_back(eos);

  return ans;
}

static std::vector<int> PiperPhonemesToIdsMatcha(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, bool use_eos_bos) {
  std::vector<int> ans;
  ans.reserve(phonemes.size());

  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  if (use_eos_bos) {
    ans.push_back(bos);
  }

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      ans.push_back(token2id.at(p));
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }

  if (use_eos_bos) {
    ans.push_back(eos);
  }

  return ans;
}

static std::vector<std::vector<int>> PiperPhonemesToIdsKokoro(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, int32_t max_len) {
  std::vector<std::vector<int>> ans;

  std::vector<int> current;
  current.reserve(phonemes.size());

  current.push_back(0);

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      if (current.size() > max_len - 1) {
        current.push_back(0);
        ans.push_back(std::move(current));

        current.reserve(phonemes.size());
        current.push_back(0);
      }

      current.push_back(token2id.at(p));
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }

  current.push_back(0);
  ans.push_back(std::move(current));
  return ans;
}

static std::vector<int> CoquiPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes,
    const OfflineTtsVitsModelMetaData &vits_meta_data) {
  // see
  // https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py#L87
  int32_t use_eos_bos = vits_meta_data.use_eos_bos;
  int32_t bos_id = vits_meta_data.bos_id;
  int32_t eos_id = vits_meta_data.eos_id;
  int32_t blank_id = vits_meta_data.blank_id;
  int32_t add_blank = vits_meta_data.add_blank;
  int32_t comma_id = token2id.at(',');

  std::vector<int> ans;
  if (add_blank) {
    ans.reserve(phonemes.size() * 2 + 3);
  } else {
    ans.reserve(phonemes.size() + 2);
  }

  if (use_eos_bos) {
    ans.push_back(bos_id);
  }

  if (add_blank) {
    ans.push_back(blank_id);

    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
        ans.push_back(blank_id);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  } else {
    // not adding blank
    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  }

  // add a comma at the end of a sentence so that we can have a longer pause.
  ans.push_back(comma_id);

  if (use_eos_bos) {
    ans.push_back(eos_id);
  }

  return ans;
}

void InitEspeak(const std::string &data_dir) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [data_dir]() {
    int32_t result =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.c_str(), 0);
    if (result != 22050) {
      SHERPA_ONNX_LOGE(
          "Failed to initialize espeak-ng with data dir: %s. Return code is: "
          "%d",
          data_dir.c_str(), result);
      exit(-1);
    }
  });
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data)
    : vits_meta_data_(vits_meta_data) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data)
    : vits_meta_data_(vits_meta_data) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &voice /*= ""*/) const {
  if (is_matcha_) {
    return ConvertTextToTokenIdsMatcha(text, voice);
  } else if (is_kokoro_) {
    return ConvertTextToTokenIdsKokoro(text, voice);
  } else {
    return ConvertTextToTokenIdsVits(text, voice);
  }
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsMatcha(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  std::vector<int> phoneme_ids;

  for (const auto &p : phonemes) {
    phoneme_ids =
        PiperPhonemesToIdsMatcha(token2id_, p, matcha_meta_data_.use_eos_bos);
    ans.emplace_back(std::move(phoneme_ids));
  }

  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsKokoro(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  for (const auto &p : phonemes) {
    auto phoneme_ids =
        PiperPhonemesToIdsKokoro(token2id_, p, kokoro_meta_data_.max_token_len);

    for (auto &ids : phoneme_ids) {
      ans.emplace_back(std::move(ids));
    }
  }

  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsVits(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  std::vector<int> phoneme_ids;

  if (vits_meta_data_.is_piper || vits_meta_data_.is_icefall) {
    for (const auto &p : phonemes) {
      phoneme_ids = PiperPhonemesToIdsVits(token2id_, p);
      ans.emplace_back(std::move(phoneme_ids));
    }
  } else if (vits_meta_data_.is_coqui) {
    for (const auto &p : phonemes) {
      phoneme_ids = CoquiPhonemesToIds(token2id_, p, vits_meta_data_);
      ans.emplace_back(std::move(phoneme_ids));
    }

  } else {
    SHERPA_ONNX_LOGE("Unsupported model");
    exit(-1);
  }

  return ans;
}

#if __ANDROID_API__ >= 9
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);
#endif

#if __OHOS__
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);
#endif

}  // namespace sherpa_mnn
