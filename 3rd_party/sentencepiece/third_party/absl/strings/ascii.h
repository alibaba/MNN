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
#ifndef ABSL_STRINGS_ASCII_H_
#define ABSL_STRINGS_ASCII_H_

#include <ctype.h>

#include <string>

#include "third_party/absl/strings/string_view.h"

namespace absl {

inline std::string AsciiStrToUpper(absl::string_view value) {
  std::string upper_value = std::string(value);
  std::transform(upper_value.begin(), upper_value.end(), upper_value.begin(),
                 ::toupper);
  return upper_value;
}

inline std::string AsciiStrToLower(absl::string_view value) {
  std::string lower_value = std::string(value);
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                 ::tolower);
  return lower_value;
}
}  // namespace absl
#endif  // ABSL_STRINGS_ASCII_H_
