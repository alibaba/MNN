// sherpa-mnn/csrc/base64-decode.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/base64-decode.h"

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

static int32_t Ord(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c - 'A';
  } else if (c >= 'a' && c <= 'z') {
    return c - 'a' + ('Z' - 'A') + 1;
  } else if (c >= '0' && c <= '9') {
    return c - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
  } else if (c == '+') {
    return 62;
  } else if (c == '/') {
    return 63;
  }

  SHERPA_ONNX_LOGE("Unknown character %d, %c\n", c, c);

  exit(-1);
}

// see
// https://github.com/ReneNyffenegger/cpp-base64/blob/master/base64.cpp#L243
std::string Base64Decode(const std::string &s) {
  if (s.empty()) {
    SHERPA_ONNX_LOGE("Empty string!");
    exit(-1);
  }

  int32_t n = static_cast<int32_t>(s.size()) / 4 * 3;

  std::string ans;
  ans.reserve(n);

  int32_t i = 0;
  while (i < static_cast<int32_t>(s.size())) {
    if (s[i] == '=') {
      return " ";
    }

    int32_t first = (Ord(s[i]) << 2) + ((Ord(s[i + 1]) & 0x30) >> 4);
    ans.push_back(static_cast<char>(first));

    if (i + 2 < static_cast<int32_t>(s.size()) && s[i + 2] != '=') {
      int32_t second =
          ((Ord(s[i + 1]) & 0x0f) << 4) + ((Ord(s[i + 2]) & 0x3c) >> 2);
      ans.push_back(static_cast<char>(second));

      if (i + 3 < static_cast<int32_t>(s.size()) && s[i + 3] != '=') {
        int32_t third = ((Ord(s[i + 2]) & 0x03) << 6) + Ord(s[i + 3]);
        ans.push_back(static_cast<char>(third));
      }
    }
    i += 4;
  }

  return ans;
}

}  // namespace sherpa_mnn
