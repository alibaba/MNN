//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef ABSL_STRINGS_STR_CAT_H_
#define ABSL_STRINGS_STR_CAT_H_

#include <sstream>
#include <string>

#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/string_view.h"

namespace absl {

inline std::string StrCat(int v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

inline std::string StrCat(absl::string_view str) {
  return std::string(str.data(), str.size());
}

template <typename... T>
inline std::string StrCat(absl::string_view first, const T &...rest) {
  return StrCat(first) + StrCat(rest...);
}

template <typename... T>
inline std::string StrCat(int first, const T &...rest) {
  return StrCat(first) + StrCat(rest...);
}

inline void StrAppend(std::string *base, absl::string_view str) {
  base->append(str.data(), str.size());
}

}  // namespace absl
#endif  // ABSL_STRINGS_STR_CAT_H_
