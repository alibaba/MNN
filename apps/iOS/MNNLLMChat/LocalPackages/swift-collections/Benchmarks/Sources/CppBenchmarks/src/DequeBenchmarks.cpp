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

#include <deque>
#include <algorithm>
#include "DequeBenchmarks.h"
#include "Utils.h"

void *
cpp_deque_create(const intptr_t *start, size_t count)
{
  auto deque = new std::deque<intptr_t>(start, start + count);
  return deque;
}

void
cpp_deque_destroy(void *ptr)
{
  delete static_cast<std::deque<intptr_t> *>(ptr);
}

void
cpp_deque_from_int_range(intptr_t count)
{
  auto deque = std::deque<intptr_t>();
  for (intptr_t i = 0; i < count; i++) {
    deque.push_back(identity(i));
  }
  black_hole(&deque);
}

void
cpp_deque_from_int_buffer(const intptr_t *start, size_t count)
{
  auto deque = std::deque<intptr_t>(start, start + count);
  black_hole(&deque);
}

void
cpp_deque_append_integers(const intptr_t *start, size_t count)
{
  auto deque = std::deque<intptr_t>();
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    deque.push_back(*identity(p));
  }
  black_hole(&deque);
}

void
cpp_deque_prepend_integers(const intptr_t *start, size_t count)
{
  auto deque = std::deque<intptr_t>();
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    deque.push_front(*identity(p));
  }
  black_hole(&deque);
}

void
cpp_deque_random_insertions(const intptr_t *start, size_t count)
{
  auto deque = std::deque<intptr_t>();
  for (intptr_t i = 0; i < count; i++) {
    deque.insert(deque.cbegin() + start[i], identity(i));
  }
  black_hole(&deque);
}

void
cpp_deque_iterate(void *ptr)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  for (auto it = deque->cbegin(); it != deque->cend(); ++it) {
    black_hole(*it);
  }
}

void
cpp_deque_lookups_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*deque)[*it]);
  }
}

void
cpp_deque_lookups_at(void *ptr, const intptr_t *start, size_t count)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*deque).at(*it));
  }
}

void
cpp_deque_pop_back(void *ptr)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  auto size = deque->size();
  for (int i = 0; i < size; ++i) {
    identity(deque)->pop_back();
  }
}

void
cpp_deque_pop_front(void *ptr)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  auto size = deque->size();
  for (int i = 0; i < size; ++i) {
    identity(deque)->pop_front();
  }
}

void
cpp_deque_random_removals(void *ptr, const intptr_t *start, size_t count)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    identity(deque)->erase(deque->cbegin() + *it);
  }
}

void
cpp_deque_sort(void *ptr)
{
  auto deque = static_cast<std::deque<intptr_t> *>(ptr);
  std::sort(deque->begin(), deque->end());
}
