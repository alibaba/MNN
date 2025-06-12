// sherpa-mnn/csrc/text-utils.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#define SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#include <errno.h>
#include <stdlib.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) \
  _strtoi64(cur_cstr, end_cstr, 10);
#else
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.h

namespace sherpa_mnn {

/// Converts a string into an integer via strtoll and returns false if there was
/// any kind of problem (i.e. the string was not an integer or contained extra
/// non-whitespace junk, or the integer was too large to fit into the type it is
/// being converted into).  Only sets *out if everything was OK and it returns
/// true.
template <class Int>
bool ConvertStringToInteger(const std::string &str, Int *out) {
  // copied from kaldi/src/util/text-util.h
  static_assert(std::is_integral<Int>::value, "");
  const char *this_str = str.c_str();
  char *end = nullptr;
  errno = 0;
  int i = SHERPA_ONNX_STRTOLL(this_str, &end);
  if (end != this_str) {
    while (isspace(*end)) ++end;
  }
  if (end == this_str || *end != '\0' || errno != 0) return false;
  Int iInt = static_cast<Int>(i);
  if (static_cast<int>(iInt) != i ||
      (i < 0 && !std::numeric_limits<Int>::is_signed)) {
    return false;
  }
  *out = iInt;
  return true;
}

/// Split a string using any of the single character delimiters.
/// If omit_empty_strings == true, the output will contain any
/// nonempty strings after splitting on any of the
/// characters in the delimiter.  If omit_empty_strings == false,
/// the output will contain n+1 strings if there are n characters
/// in the set "delim" within the input string.  In this case
/// the empty string is split to a single empty string.
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);

/**
  \brief Split a string (e.g. 1:2:3) into a vector of integers.

  \param [in]  delim  String containing a list of characters, any of which
                      is allowed as a delimiter.
  \param [in] omit_empty_strings If true, empty strings between delimiters are
                      allowed and will not produce an output integer; if false,
                      instances of characters in 'delim' that are consecutive or
                      at the start or end of the string would be an error.
                      You'll normally want this to be true if 'delim' consists
                      of spaces, and false otherwise.
  \param [out] out   The output list of integers.
*/
template <class I>
bool SplitStringToIntegers(const std::string &full, const char *delim,
                           bool omit_empty_strings,  // typically false [but
                                                     // should probably be true
                                                     // if "delim" is spaces].
                           std::vector<I> *out) {
  static_assert(std::is_integral<I>::value, "");
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    const char *this_str = split[i].c_str();
    char *end = NULL;
    int j = 0;
    j = SHERPA_ONNX_STRTOLL(this_str, &end);
    if (end == this_str || *end != '\0') {
      out->clear();
      return false;
    } else {
      I jI = static_cast<I>(j);
      if (static_cast<int>(jI) != j) {
        // output type cannot fit this integer.
        out->clear();
        return false;
      }
      (*out)[i] = jI;
    }
  }
  return true;
}

// This is defined for F = float and double.
template <class F>
bool SplitStringToFloats(const std::string &full, const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out);

// This is defined for F = float and double.
template <typename T>
bool ConvertStringToReal(const std::string &str, T *out);

std::vector<std::string> SplitUtf8(const std::string &text);

std::string ToLowerCase(const std::string &s);
void ToLowerCase(std::string *in_out);

std::wstring ToLowerCase(const std::wstring &s);

std::string RemoveInvalidUtf8Sequences(const std::string &text,
                                       bool show_debug_msg = false);

// Return true if text contains valid utf8 sequence.
// Return false otherwise
bool IsUtf8(const std::string &text);

// Return true if text contains valid gb2312 encoded sequence
// Return false otherwise
bool IsGB2312(const std::string &text);

#if defined(_WIN32)
std::string Gb2312ToUtf8(const std::string &text);
#endif

std::wstring ToWideString(const std::string &s);

std::string ToString(const std::wstring &s);

bool EndsWith(const std::string &haystack, const std::string &needle);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_TEXT_UTILS_H_
