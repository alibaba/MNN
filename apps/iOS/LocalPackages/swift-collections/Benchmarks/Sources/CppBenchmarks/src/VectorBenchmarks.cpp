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
#include "VectorBenchmarks.h"
#include "Utils.h"

void *
cpp_vector_create(const intptr_t *start, size_t count)
{
  auto vector = new std::vector<intptr_t>(start, start + count);
  return vector;
}

void
cpp_vector_destroy(void *ptr)
{
  delete static_cast<std::vector<intptr_t> *>(ptr);
}

void
cpp_vector_from_int_range(intptr_t count)
{
  auto vector = std::vector<intptr_t>();
  for (intptr_t i = 0; i < count; i++) {
    vector.push_back(identity(i));
  }
  black_hole(&vector);
}

void
cpp_vector_from_int_buffer(const intptr_t *start, size_t count)
{
  auto vector = std::vector<intptr_t>(start, start + count);
  black_hole(&vector);
}

void
cpp_vector_append_integers(const intptr_t *start, size_t count, bool reserve)
{
  auto vector = std::vector<intptr_t>();
  if (reserve) vector.reserve(count);
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    vector.push_back(*identity(p));
  }
  black_hole(&vector);
}

void
cpp_vector_prepend_integers(const intptr_t *start, size_t count, bool reserve)
{
  auto vector = std::vector<intptr_t>();
  if (reserve) vector.reserve(count);
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    vector.insert(vector.cbegin(), *identity(p));
  }
  black_hole(&vector);
}

void
cpp_vector_random_insertions(const intptr_t *start, size_t count, bool reserve)
{
  auto vector = std::vector<intptr_t>();
  if (reserve) vector.reserve(count);
  for (intptr_t i = 0; i < count; i++) {
    vector.insert(vector.cbegin() + start[i], identity(i));
  }
  black_hole(&vector);
}

void
cpp_vector_iterate(void *ptr)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  for (auto it = vector->cbegin(); it != vector->cend(); ++it) {
    black_hole(*it);
  }
}

void
cpp_vector_lookups_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*vector)[*it]);
  }
}

void
cpp_vector_lookups_at(void *ptr, const intptr_t *start, size_t count)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*vector).at(*it));
  }
}

void
cpp_vector_pop_back(void *ptr)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  auto size = vector->size();
  for (int i = 0; i < size; ++i) {
    identity(vector)->pop_back();
  }
}

void
cpp_vector_pop_front(void *ptr)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  auto size = vector->size();
  for (int i = 0; i < size; ++i) {
    identity(vector)->erase(vector->cbegin());
  }
}

void
cpp_vector_random_removals(void *ptr, const intptr_t *start, size_t count)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    identity(vector)->erase(vector->cbegin() + *it);
  }
}

void
cpp_vector_sort(void *ptr)
{
  auto vector = static_cast<std::vector<intptr_t> *>(ptr);
  std::sort(vector->begin(), vector->end());
}
