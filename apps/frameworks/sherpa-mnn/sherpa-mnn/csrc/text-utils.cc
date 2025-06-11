// sherpa-mnn/csrc/text-utils.cc
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/text-utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <codecvt>
#include <cstdint>
#include <cwctype>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#endif

#include "sherpa-mnn/csrc/macros.h"

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.cc

namespace sherpa_mnn {

// copied from kaldi/src/util/text-util.cc
template <class T>
class NumberIstream {
 public:
  explicit NumberIstream(std::istream &i) : in_(i) {}

  NumberIstream &operator>>(T &x) {
    if (!in_.good()) return *this;
    in_ >> x;
    if (!in_.fail() && RemainderIsOnlySpaces()) return *this;
    return ParseOnFail(&x);
  }

 private:
  std::istream &in_;

  bool RemainderIsOnlySpaces() {
    if (in_.tellg() != std::istream::pos_type(-1)) {
      std::string rem;
      in_ >> rem;

      if (rem.find_first_not_of(' ') != std::string::npos) {
        // there is not only spaces
        return false;
      }
    }

    in_.clear();
    return true;
  }

  NumberIstream &ParseOnFail(T *x) {
    std::string str;
    in_.clear();
    in_.seekg(0);
    // If the stream is broken even before trying
    // to read from it or if there are many tokens,
    // it's pointless to try.
    if (!(in_ >> str) || !RemainderIsOnlySpaces()) {
      in_.setstate(std::ios_base::failbit);
      return *this;
    }

    std::unordered_map<std::string, T> inf_nan_map;
    // we'll keep just uppercase values.
    inf_nan_map["INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INF"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INFINITY"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["+NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-NAN"] = -std::numeric_limits<T>::quiet_NaN();
    // MSVC
    inf_nan_map["1.#INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-1.#INF"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["1.#QNAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-1.#QNAN"] = -std::numeric_limits<T>::quiet_NaN();

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    if (inf_nan_map.find(str) != inf_nan_map.end()) {
      *x = inf_nan_map[str];
    } else {
      in_.setstate(std::ios_base::failbit);
    }

    return *this;
  }
};

/// ConvertStringToReal converts a string into either float or double
/// and returns false if there was any kind of problem (i.e. the string
/// was not a floating point number or contained extra non-whitespace junk).
/// Be careful- this function will successfully read inf's or nan's.
template <typename T>
bool ConvertStringToReal(const std::string &str, T *out) {
  std::istringstream iss(str);

  NumberIstream<T> i(iss);

  i >> *out;

  if (iss.fail()) {
    // Number conversion failed.
    return false;
  }

  return true;
}

template bool ConvertStringToReal<float>(const std::string &str, float *out);

template bool ConvertStringToReal<double>(const std::string &str, double *out);

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

template <class F>
bool SplitStringToFloats(const std::string &full, const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out) {
  assert(out != nullptr);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); ++i) {
    // assume atof never fails
    F f = 0;
    if (!ConvertStringToReal(split[i], &f)) return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<float> *out);
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<double> *out);

static bool IsPunct(char c) { return c != '\'' && std::ispunct(c); }
static bool IsGermanUmlaut(const std::string &word) {
  // ä 0xC3 0xA4
  // ö 0xC3 0xB6
  // ü 0xC3 0xBC
  // Ä 0xC3 0x84
  // Ö 0xC3 0x96
  // Ü 0xC3 0x9C
  // ß 0xC3 0x9F

  if (word.size() != 2 || static_cast<uint8_t>(word[0]) != 0xc3) {
    return false;
  }

  auto c = static_cast<uint8_t>(word[1]);
  if (c == 0xa4 || c == 0xb6 || c == 0xbc || c == 0x84 || c == 0x96 ||
      c == 0x9c || c == 0x9f) {
    return true;
  }

  return false;
}

// see https://www.tandem.net/blog/spanish-accents
// https://www.compart.com/en/unicode/U+00DC
static bool IsSpanishDiacritic(const std::string &word) {
  // á 0xC3 0xA1
  // é 0xC3 0xA9
  // í 0xC3 0xAD
  // ó 0xC3 0xB3
  // ú 0xC3 0xBA
  // ü 0xC3 0xBC
  // ñ 0xC3 0xB1
  //
  // uppercase
  //
  // Á 0xC3 0x81
  // É 0xC3 0x89
  // Í 0xC3 0x8D
  // Ó 0xC3 0x93
  // Ú 0xC3 0x9A
  // Ü 0xC3 0x9C
  // Ñ 0xC3 0x91

  if (word.size() != 2 || static_cast<uint8_t>(word[0]) != 0xc3) {
    return false;
  }

  auto c = static_cast<uint8_t>(word[1]);
  if (c == 0xa1 || c == 0xa9 || c == 0xad || c == 0xb3 || c == 0xba ||
      c == 0xbc || c == 0xb1 || c == 0x81 || c == 0x89 || c == 0x8d ||
      c == 0x93 || c == 0x9a || c == 0x9c || c == 0x91) {
    return true;
  }

  return false;
}

// see https://www.busuu.com/en/french/accent-marks
static bool IsFrenchDiacritic(const std::string &word) {
  // acute accent
  // é 0xC3 0xA9
  //
  // grave accent
  // à 0xC3 0xA0
  // è 0xC3 0xA8
  // ù 0xC3 0xB9
  //
  // cedilla
  // ç 0xC3 0xA7
  //
  // circumflex
  // â 0xC3 0xA2
  // ê 0xC3 0xAA
  // î 0xC3 0xAE
  // ô 0xC3 0xB4
  // û 0xC3 0xBB
  //
  // trema
  // ë 0xC3 0xAB
  // ï 0xC3 0xAF
  // ü 0xC3 0xBC
  //
  // É 0xC3 0x89
  //
  // À 0xC3 0x80
  // È 0xC3 0x88
  // Ù 0xC3 0x99
  // Ç 0xC3 0x87
  // Â 0xC3 0x82
  // Ê 0xC3 0x8A
  // Î 0xC3 0x8E
  // Ô 0xC3 0x94
  // Û 0xC3 0x9B
  // Ë 0xC3 0x8B
  // Ï 0xC3 0x8F
  // Ü 0xC3 0x9C

  if (word.size() != 2 || static_cast<uint8_t>(word[0]) != 0xc3) {
    return false;
  }

  auto c = static_cast<uint8_t>(word[1]);
  if (c == 0xa9 || c == 0xa0 || c == 0xa8 || c == 0xb9 || c == 0xa7 ||
      c == 0xa2 || c == 0xaa || c == 0xae || c == 0xb4 || c == 0xbb ||
      c == 0xab || c == 0xaf || c == 0xbc || c == 0x89 || c == 0x80 ||
      c == 0x88 || c == 0x99 || c == 0x87 || c == 0x82 || c == 0x8a ||
      c == 0x8e || c == 0x94 || c == 0x9b || c == 0x8b || c == 0x8f ||
      c == 0x9c) {
    return true;
  }
  return false;
}

static bool IsSpecial(const std::string &w) {
  bool ans = IsGermanUmlaut(w) || IsSpanishDiacritic(w) || IsFrenchDiacritic(w);

  // for french d’impossible
  // ’ 0xE2 0x80 0x99
  bool ans2 = false;
  if (w.size() == 3) {
    auto c0 = static_cast<uint8_t>(w[0]);
    auto c1 = static_cast<uint8_t>(w[1]);
    auto c2 = static_cast<uint8_t>(w[2]);
    if (c0 == 0xe2 && c1 == 0x80 && c2 == 0x99) {
      ans2 = true;
    }
  }

  return ans || ans2;
}

static std::vector<std::string> MergeCharactersIntoWords(
    const std::vector<std::string> &words) {
  std::vector<std::string> ans;

  int32_t n = static_cast<int32_t>(words.size());
  int32_t i = 0;
  int32_t prev = -1;

  while (i < n) {
    const auto &w = words[i];
    if (w.size() >= 3 || (w.size() == 2 && !IsSpecial(w)) ||
        (w.size() == 1 && (IsPunct(w[0]) || std::isspace(w[0])))) {
      if (prev != -1) {
        std::string t;
        for (; prev < i; ++prev) {
          t.append(words[prev]);
        }
        prev = -1;
        ans.push_back(std::move(t));
      }

      if (!std::isspace(w[0])) {
        ans.push_back(w);
      }
      ++i;
      continue;
    }

    // e.g., öffnen
    if (w.size() == 1 || (w.size() == 2 && IsSpecial(w))) {
      if (prev == -1) {
        prev = i;
      }
      ++i;
      continue;
    }

    SHERPA_ONNX_LOGE("Ignore %s", w.c_str());
    ++i;
  }

  if (prev != -1) {
    std::string t;
    for (; prev < i; ++prev) {
      t.append(words[prev]);
    }
    ans.push_back(std::move(t));
  }

  return ans;
}

std::vector<std::string> SplitUtf8(const std::string &text) {
  const uint8_t *begin = reinterpret_cast<const uint8_t *>(text.c_str());
  const uint8_t *end = begin + text.size();

  // Note that English words are split into single characters.
  // We need to invoke MergeCharactersIntoWords() to merge them
  std::vector<std::string> ans;

  auto start = begin;
  while (start < end) {
    uint8_t c = *start;
    uint8_t i = 0x80;
    int32_t num_bytes = 0;

    // see
    // https://en.wikipedia.org/wiki/UTF-8
    for (; c & i; i >>= 1) {
      ++num_bytes;
    }

    if (num_bytes == 0) {
      // this is an ascii
      ans.emplace_back(reinterpret_cast<const char *>(start), 1);
      ++start;
    } else if (2 <= num_bytes && num_bytes <= 4) {
      ans.emplace_back(reinterpret_cast<const char *>(start), num_bytes);
      start += num_bytes;
    } else {
      SHERPA_ONNX_LOGE("Invalid byte at position: %d",
                       static_cast<int32_t>(start - begin));
      // skip this byte
      ++start;
    }
  }

  return MergeCharactersIntoWords(ans);
}

std::string ToLowerCase(const std::string &s) {
  return ToString(ToLowerCase(ToWideString(s)));
}

void ToLowerCase(std::string *in_out) {
  std::transform(in_out->begin(), in_out->end(), in_out->begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

std::wstring ToLowerCase(const std::wstring &s) {
  std::wstring ans(s.size(), 0);
  std::transform(s.begin(), s.end(), ans.begin(), [](wchar_t c) -> wchar_t {
    switch (c) {
      // French
      case L'À':
        return L'à';
      case L'Â':
        return L'â';
      case L'Æ':
        return L'æ';
      case L'Ç':
        return L'ç';
      case L'È':
        return L'è';
      case L'É':
        return L'é';
      case L'Ë':
        return L'ë';
      case L'Î':
        return L'î';
      case L'Ï':
        return L'ï';
      case L'Ô':
        return L'ô';
      case L'Ù':
        return L'ù';
      case L'Û':
        return L'û';
      case L'Ü':
        return L'ü';

      // others
      case L'Á':
        return L'á';
      case L'Í':
        return L'í';
      case L'Ó':
        return L'ó';
      case L'Ú':
        return L'ú';
      case L'Ñ':
        return L'ñ';
      case L'Ì':
        return L'ì';
      case L'Ò':
        return L'ò';
      case L'Ä':
        return L'ä';
      case L'Ö':
        return L'ö';
        // TODO(fangjun): Add more

      default:
        return std::towlower(c);
    }
  });
  return ans;
}

static inline bool InRange(uint8_t x, uint8_t low, uint8_t high) {
  return low <= x && x <= high;
}

/*
Please see
https://stackoverflow.com/questions/6555015/check-for-invalid-utf8


Table 3-7. Well-Formed UTF-8 Byte Sequences

Code Points        First Byte Second Byte Third Byte Fourth Byte
U+0000..U+007F     00..7F
U+0080..U+07FF     C2..DF     80..BF
U+0800..U+0FFF     E0         A0..BF      80..BF
U+1000..U+CFFF     E1..EC     80..BF      80..BF
U+D000..U+D7FF     ED         80..9F      80..BF
U+E000..U+FFFF     EE..EF     80..BF      80..BF
U+10000..U+3FFFF   F0         90..BF      80..BF     80..BF
U+40000..U+FFFFF   F1..F3     80..BF      80..BF     80..BF
U+100000..U+10FFFF F4         80..8F      80..BF     80..BF
 */
std::string RemoveInvalidUtf8Sequences(const std::string &text,
                                       bool show_debug_msg /*= false*/) {
  int32_t n = static_cast<int32_t>(text.size());

  std::string ans;
  ans.reserve(n);

  int32_t i = 0;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(text.data());
  while (i < n) {
    if (p[i] <= 0x7f) {
      ans.append(text, i, 1);
      i += 1;
      continue;
    }

    if (InRange(p[i], 0xc2, 0xdf) && i + 1 < n &&
        InRange(p[i + 1], 0x80, 0xbf)) {
      ans.append(text, i, 2);
      i += 2;
      continue;
    }

    if (p[i] == 0xe0 && i + 2 < n && InRange(p[i + 1], 0xa0, 0xbf) &&
        InRange(p[i + 2], 0x80, 0xbf)) {
      ans.append(text, i, 3);
      i += 3;
      continue;
    }

    if (InRange(p[i], 0xe1, 0xec) && i + 2 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf)) {
      ans.append(text, i, 3);
      i += 3;
      continue;
    }

    if (p[i] == 0xed && i + 2 < n && InRange(p[i + 1], 0x80, 0x9f) &&
        InRange(p[i + 2], 0x80, 0xbf)) {
      ans.append(text, i, 3);
      i += 3;
      continue;
    }

    if (InRange(p[i], 0xee, 0xef) && i + 2 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf)) {
      ans.append(text, i, 3);
      i += 3;
      continue;
    }

    if (p[i] == 0xf0 && i + 3 < n && InRange(p[i + 1], 0x90, 0xbf) &&
        InRange(p[i + 2], 0x80, 0xbf) && InRange(p[i + 3], 0x80, 0xbf)) {
      ans.append(text, i, 4);
      i += 4;
      continue;
    }

    if (InRange(p[i], 0xf1, 0xf3) && i + 3 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf) &&
        InRange(p[i + 3], 0x80, 0xbf)) {
      ans.append(text, i, 4);
      i += 4;
      continue;
    }

    if (p[i] == 0xf4 && i + 3 < n && InRange(p[i + 1], 0x80, 0x8f) &&
        InRange(p[i + 2], 0x80, 0xbf) && InRange(p[i + 3], 0x80, 0xbf)) {
      ans.append(text, i, 4);
      i += 4;
      continue;
    }

    if (show_debug_msg) {
      SHERPA_ONNX_LOGE("Ignore invalid utf8 sequence at pos: %d, value: %02x",
                       i, p[i]);
    }

    i += 1;
  }

  return ans;
}

bool IsUtf8(const std::string &text) {
  int32_t n = static_cast<int32_t>(text.size());
  int32_t i = 0;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(text.data());
  while (i < n) {
    if (p[i] <= 0x7f) {
      i += 1;
      continue;
    }

    if (InRange(p[i], 0xc2, 0xdf) && i + 1 < n &&
        InRange(p[i + 1], 0x80, 0xbf)) {
      i += 2;
      continue;
    }

    if (p[i] == 0xe0 && i + 2 < n && InRange(p[i + 1], 0xa0, 0xbf) &&
        InRange(p[i + 2], 0x80, 0xbf)) {
      i += 3;
      continue;
    }

    if (InRange(p[i], 0xe1, 0xec) && i + 2 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf)) {
      i += 3;
      continue;
    }

    if (p[i] == 0xed && i + 2 < n && InRange(p[i + 1], 0x80, 0x9f) &&
        InRange(p[i + 2], 0x80, 0xbf)) {
      i += 3;
      continue;
    }

    if (InRange(p[i], 0xee, 0xef) && i + 2 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf)) {
      i += 3;
      continue;
    }

    if (p[i] == 0xf0 && i + 3 < n && InRange(p[i + 1], 0x90, 0xbf) &&
        InRange(p[i + 2], 0x80, 0xbf) && InRange(p[i + 3], 0x80, 0xbf)) {
      i += 4;
      continue;
    }

    if (InRange(p[i], 0xf1, 0xf3) && i + 3 < n &&
        InRange(p[i + 1], 0x80, 0xbf) && InRange(p[i + 2], 0x80, 0xbf) &&
        InRange(p[i + 3], 0x80, 0xbf)) {
      i += 4;
      continue;
    }

    if (p[i] == 0xf4 && i + 3 < n && InRange(p[i + 1], 0x80, 0x8f) &&
        InRange(p[i + 2], 0x80, 0xbf) && InRange(p[i + 3], 0x80, 0xbf)) {
      i += 4;
      continue;
    }

    return false;
  }

  return true;
}

bool IsGB2312(const std::string &text) {
  int32_t n = static_cast<int32_t>(text.size());
  int32_t i = 0;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(text.data());
  while (i < n) {
    if (p[i] <= 0x7f) {
      i += 1;
      continue;
    }

    if (InRange(p[i], 0xa1, 0xf7) && i + 1 < n &&
        InRange(p[i + 1], 0xa1, 0xfe)) {
      i += 2;
      continue;
    }

    return false;
  }

  return true;
}

#if defined(_WIN32)
std::string Gb2312ToUtf8(const std::string &text) {
  // https://learn.microsoft.com/en-us/windows/win32/api/stringapiset/nf-stringapiset-multibytetowidechar
  // 936 is from
  // https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers
  // GB2312 -> 936
  int32_t num_wchars =
      MultiByteToWideChar(936, 0, text.c_str(), text.size(), nullptr, 0);
  SHERPA_ONNX_LOGE("num of wchars: %d", num_wchars);
  if (num_wchars == 0) {
    return {};
  }

  std::wstring wstr;
  wstr.resize(num_wchars);
  MultiByteToWideChar(936, 0, text.c_str(), text.size(), wstr.data(),
                      num_wchars);
  // https://learn.microsoft.com/en-us/windows/win32/api/stringapiset/nf-stringapiset-widechartomultibyte
  int32_t num_chars = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr,
                                          0, nullptr, nullptr);
  if (num_chars == 0) {
    return {};
  }

  std::string ans(num_chars, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, ans.data(), num_chars,
                      nullptr, nullptr);

  return ans;
}
#endif

std::wstring ToWideString(const std::string &s) {
  // see
  // https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(s);
}

std::string ToString(const std::wstring &s) {
  // see
  // https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.to_bytes(s);
}

bool EndsWith(const std::string &haystack, const std::string &needle) {
  if (needle.size() > haystack.size()) {
    return false;
  }

  return std::equal(needle.rbegin(), needle.rend(), haystack.rbegin());
}

}  // namespace sherpa_mnn
