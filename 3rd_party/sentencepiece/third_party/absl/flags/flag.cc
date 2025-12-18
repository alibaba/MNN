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

#include "third_party/absl/flags/flag.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "config.h"
#include "src/common.h"
#include "src/util.h"

ABSL_FLAG(bool, help, false, "show help");
ABSL_FLAG(bool, version, false, "show version");
ABSL_FLAG(int, minloglevel, 0,
          "Messages logged at a lower level than this don't actually get "
          "logged anywhere");

namespace absl {
namespace internal {
namespace {
template <typename T>
std::string to_str(const T &value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <>
std::string to_str<bool>(const bool &value) {
  return value ? "true" : "false";
}

template <>
std::string to_str<std::string>(const std::string &value) {
  return std::string("\"") + value + std::string("\"");
}
}  // namespace

struct FlagFunc {
  const char *name;
  const char *help;
  const char *type;
  std::string default_value;
  std::function<void(const std::string &)> set_value;
};

namespace {

using FlagMap = std::map<std::string, std::shared_ptr<FlagFunc>>;
using FlagList = std::vector<std::shared_ptr<FlagFunc>>;

FlagMap *GetFlagMap() {
  static auto *flag_map = new FlagMap;
  return flag_map;
}

FlagList *GetFlagList() {
  static auto *flag_list = new FlagList;
  return flag_list;
}

bool CommandLineGetFlag(int argc, char **argv, std::string *key,
                        std::string *value, int *used_args) {
  key->clear();
  value->clear();

  *used_args = 1;
  const char *start = argv[0];
  if (start[0] != '-') return false;

  ++start;
  if (start[0] == '-') ++start;
  const std::string arg = start;
  const size_t n = arg.find("=");
  if (n != std::string::npos) {
    *key = arg.substr(0, n);
    *value = arg.substr(n + 1, arg.size() - n);
    return true;
  }

  key->assign(arg);
  value->clear();

  if (argc == 1) return true;

  start = argv[1];
  if (start[0] == '-') return true;

  *used_args = 2;
  value->assign(start);
  return true;
}

std::string PrintHelp(const char *programname) {
  std::ostringstream os;
  os << PACKAGE_STRING << "\n\n";
  os << "Usage: " << programname << " [options] files\n\n";

  for (auto func : *GetFlagList()) {
    os << "   --" << func->name << " (" << func->help << ")";
    os << "  type: " << func->type << " default: " << func->default_value
       << '\n';
  }

  os << "\n\n";

  return os.str();
}
}  // namespace

void RegisterFlag(const std::string &name, std::shared_ptr<FlagFunc> func) {
  GetFlagList()->emplace_back(func);
  GetFlagMap()->emplace(name, func);
}
}  // namespace internal

template <typename T>
Flag<T>::Flag(const char *name, const char *type, const char *help,
              const T &default_value)
    : value_(default_value), func_(new internal::FlagFunc) {
  func_->name = name;
  func_->help = help;
  func_->type = type;
  func_->default_value = internal::to_str<T>(default_value);
  func_->set_value = [this](const std::string &value) {
    this->set_value_as_str(value);
  };
  RegisterFlag(name, func_);
}

template <typename T>
Flag<T>::~Flag() {}

template <typename T>
const T &Flag<T>::value() const {
  return value_;
}

template <typename T>
void Flag<T>::set_value(const T &value) {
  value_ = value;
}

template <typename T>
void Flag<T>::set_value_as_str(const std::string &value_as_str) {
  sentencepiece::string_util::lexical_cast<T>(value_as_str, &value_);
}

template <>
void Flag<bool>::set_value_as_str(const std::string &value_as_str) {
  if (value_as_str.empty())
    value_ = true;
  else
    sentencepiece::string_util::lexical_cast<bool>(value_as_str, &value_);
}

template class Flag<std::string>;
template class Flag<int32_t>;
template class Flag<uint32_t>;
template class Flag<double>;
template class Flag<float>;
template class Flag<bool>;
template class Flag<int64_t>;
template class Flag<uint64_t>;

std::vector<char *> ParseCommandLine(int argc, char *argv[]) {
  if (argc == 0) return {};

  int used_argc = 0;
  std::string key, value;
  std::vector<char *> output_args;
  output_args.reserve(argc);
  output_args.push_back(argv[0]);

  auto set_flag = [](const std::string &name, const std::string &value) {
    const auto *flag_map = internal::GetFlagMap();
    auto it = flag_map->find(name);
    if (it == flag_map->end()) return false;
    it->second->set_value(value);
    return true;
  };

  for (int i = 1; i < argc; i += used_argc) {
    if (!internal::CommandLineGetFlag(argc - i, argv + i, &key, &value,
                                      &used_argc)) {
      output_args.push_back(argv[i]);
      continue;
    }

    if (!set_flag(key, value)) {
      std::cerr << "Unknown/Invalid flag " << key << "\n\n"
                << internal::PrintHelp(argv[0]);
      sentencepiece::error::Exit(1);
    }
  }

  if (absl::GetFlag(FLAGS_help)) {
    std::cout << internal::PrintHelp(argv[0]);
    sentencepiece::error::Exit(1);
  } else if (absl::GetFlag(FLAGS_version)) {
    std::cout << PACKAGE_STRING << " " << VERSION << std::endl;
    sentencepiece::error::Exit(0);
  }

  return output_args;
}

void CleanupFlags() {
  static bool is_shutdown = false;
  if (!is_shutdown) {
    delete internal::GetFlagList();
    delete internal::GetFlagMap();
    is_shutdown = true;
  }
}

}  // namespace absl
