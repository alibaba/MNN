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
// -----------------------------------------------------------------------------
// File: string_view.h
// -----------------------------------------------------------------------------
//
// This file contains the definition of the `absl::string_view` class. A
// `string_view` points to a contiguous span of characters, often part or all of
// another `std::string`, double-quoted std::string literal, character array, or
// even another `string_view`.
//
// This `absl::string_view` abstraction is designed to be a drop-in
// replacement for the C++17 `std::string_view` abstraction.
#ifndef ABSL_STRINGS_STRING_VIEW_H_
#define ABSL_STRINGS_STRING_VIEW_H_

#include <algorithm>
#include <string_view>

namespace absl {
using std::string_view;

// ClippedSubstr()
//
// Like `s.substr(pos, n)`, but clips `pos` to an upper bound of `s.size()`.
// Provided because std::string_view::substr throws if `pos > size()`
inline string_view ClippedSubstr(string_view s, size_t pos,
                                 size_t n = string_view::npos) {
  pos = std::min(pos, static_cast<size_t>(s.size()));
  return s.substr(pos, n);
}

// NullSafeStringView()
//
// Creates an `absl::string_view` from a pointer `p` even if it's null-valued.
// This function should be used where an `absl::string_view` can be created from
// a possibly-null pointer.
inline string_view NullSafeStringView(const char* p) {
  return p ? string_view(p) : string_view();
}

}  // namespace absl

#endif  // ABSL_STRINGS_STRING_VIEW_H_
