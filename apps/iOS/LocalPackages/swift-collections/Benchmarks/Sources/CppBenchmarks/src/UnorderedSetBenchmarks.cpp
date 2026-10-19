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

#include <unordered_set>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include "UnorderedSetBenchmarks.h"
#include "CustomHash.h"
#include "Utils.h"

typedef std::unordered_set<intptr_t, custom_intptr_hash> custom_set;

void *
cpp_unordered_set_create(const intptr_t *start, size_t count)
{
  auto set = new custom_set(start, start + count);
  return set;
}

void
cpp_unordered_set_destroy(void *ptr)
{
  delete static_cast<custom_set *>(ptr);
}

void
cpp_unordered_set_from_int_range(intptr_t count)
{
  auto set = custom_set();
  for (intptr_t i = 0; i < count; i++) {
    set.insert(identity(i));
  }
  black_hole(&set);
}

void
cpp_unordered_set_from_int_buffer(const intptr_t *start, size_t count)
{
  auto set = custom_set(start, start + count);
  black_hole(&set);
}

void
cpp_unordered_set_insert_integers(const intptr_t *start, size_t count, bool reserve)
{
  auto set = custom_set();
  if (reserve) set.reserve(count);
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    set.insert(*identity(p));
  }
  black_hole(&set);
}

void
cpp_unordered_set_iterate(void *ptr)
{
  auto set = static_cast<custom_set *>(ptr);
  for (auto it = set->cbegin(); it != set->cend(); ++it) {
    black_hole(*it);
  }
}

void
cpp_unordered_set_lookups(void *ptr, const intptr_t *start, size_t count, bool expectMatch)
{
  auto set = static_cast<custom_set *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    auto found = set->find(*it) != set->end();
    if (found != expectMatch) { abort(); }
  }
}

void
cpp_unordered_set_removals(void *ptr, const intptr_t *start, size_t count)
{
  auto set = static_cast<custom_set *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    identity(set)->erase(*it);
  }
}
