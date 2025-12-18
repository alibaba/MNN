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
#ifndef ABSL_STRINGS_STR_FORMAT_H
#define ABSL_STRINGS_STR_FORMAT_H

#include <stdio.h>

#include <string>

#include "third_party/absl/strings/string_view.h"

namespace absl {

template <typename... Args>
std::string StrFormat(const char *format, Args const &... args) {
  const int len = ::snprintf(nullptr, 0, format, args...);
  std::string s;
  s.resize(len);
  ::snprintf(&s[0], s.size() + 1, format, args...);
  return s;
}

}  // namespace absl
#endif  // ABSL_MEMORY_MEMORY_H_
