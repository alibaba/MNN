// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef ABSL_FLAGS_FLAG_H_
#define ABSL_FLAGS_FLAG_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace absl {
namespace internal {
struct FlagFunc;

void RegisterFlag(const std::string &name, std::shared_ptr<FlagFunc> func);

}  // namespace internal

template <typename T>
class Flag {
 public:
  Flag(const char *name, const char *type, const char *help,
       const T &defautl_value);
  virtual ~Flag();
  const T &value() const;
  void set_value(const T &value);
  void set_value_as_str(const std::string &value_as_str);

 private:
  T value_;
  std::shared_ptr<internal::FlagFunc> func_;
};

template <typename T>
const T &GetFlag(const Flag<T> &flag) {
  return flag.value();
}

template <typename T, typename V>
void SetFlag(Flag<T> *flag, const V &v) {
  const T value(v);
  flag->set_value(value);
}

#define HAS_ABSL_CLEANUP_FLAGS

void CleanupFlags();

}  // namespace absl

#define ABSL_FLAG(Type, name, defautl_value, help) \
  absl::Flag<Type> FLAGS_##name(#name, #Type, help, defautl_value);

#define ABSL_DECLARE_FLAG(Type, name) extern absl::Flag<Type> FLAGS_##name;

#endif  // ABSL_FLAGS_FLAG_H_
