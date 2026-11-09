//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <algorithm>
#include "VectorBoolBenchmarks.h"
#include "Utils.h"

void *
cpp_vector_bool_create_repeating(size_t count, bool value)
{
  auto v = new std::vector<bool>(count, value);
  return v;
}

void
cpp_vector_bool_destroy(void *ptr)
{
  delete static_cast<std::vector<bool> *>(ptr);
}

void
cpp_vector_bool_push_back(const bool *start, size_t count, bool reserve)
{
  auto v = std::vector<bool>();

  if (reserve) {
    v.reserve(v.size() + count);
  }

  for (auto p = start; p < start + count; ++p) {
    v.push_back(*p);
  }

  black_hole(&v);
}

void
cpp_vector_bool_pop_back(void *ptr, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (size_t i = 0; i < count; ++i) {
    v.pop_back();
  }
}

void
cpp_vector_bool_set_indices_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    v[*p] = true;
  }
}

void
cpp_vector_bool_set_indices_at(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    v.at(*p) = true;
  }
}

void
cpp_vector_bool_reset_indices_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    v[*p] = false;
  }
}

void
cpp_vector_bool_reset_indices_at(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    v.at(*p) = false;
  }
}

void
cpp_vector_bool_lookups_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    black_hole(v[*p]);
  }
}

void
cpp_vector_bool_lookups_at(void *ptr, const intptr_t *start, size_t count)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto p = start; p < start + count; ++p) {
    black_hole(v.at(*p));
  }
}

void
cpp_vector_bool_iterate(void *ptr)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  for (auto it = v.cbegin(); it != v.cend(); ++it) {
    black_hole(*it);
  }
}

size_t
cpp_vector_bool_find_true_bits(void *ptr)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  size_t count = 0;
  for (auto it = std::find(v.cbegin(), v.cend(), true);
       it != v.cend();
       it = std::find(it, v.cend(), true)) {
    ++count;
    ++it;
  }
  return count;
}

size_t
cpp_vector_bool_count_true_bits(void *ptr)
{
  auto &v = *static_cast<std::vector<bool> *>(ptr);

  return std::count(v.cbegin(), v.cend(), true);
}
