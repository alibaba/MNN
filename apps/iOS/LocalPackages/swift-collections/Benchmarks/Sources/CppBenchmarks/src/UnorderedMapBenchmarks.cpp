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

#include <unordered_map>
#include <cstdlib>
#include <algorithm>
#include "UnorderedMapBenchmarks.h"
#include "CustomHash.h"
#include "Utils.h"

typedef std::unordered_map<intptr_t, intptr_t, custom_intptr_hash> custom_map;

void *
cpp_unordered_map_create(const intptr_t *start, size_t count)
{
  auto map = new custom_map();
  for (size_t i = 0; i < count; ++i) {
    map->insert({start[i], 2 * start[i]});
  }
  return map;
}

void
cpp_unordered_map_destroy(void *ptr)
{
  delete static_cast<custom_map *>(ptr);
}

void
cpp_unordered_map_from_int_range(intptr_t count)
{
  auto map = custom_map();
  for (intptr_t i = 0; i < count; i++) {
    map.insert({identity(i), 2 * i});
  }
  black_hole(&map);
}

void
cpp_unordered_map_insert_integers(const intptr_t *start, size_t count, bool reserve)
{
  auto map = custom_map();
  if (reserve) map.reserve(count);
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    auto v = *identity(p);
    map.insert({ v, 2 * v });
  }
  black_hole(&map);
}

void
cpp_unordered_map_iterate(void *ptr)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = map->cbegin(); it != map->cend(); ++it) {
    black_hole(it->first);
    black_hole(it->second);
  }
}

void
cpp_unordered_map_lookups(void *ptr, const intptr_t *start, size_t count, bool expectMatch)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    auto found = map->find(*it) != map->end();
    if (found != expectMatch) { abort(); }
  }
}

void
cpp_unordered_map_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*map)[*it]);
  }
}

void
cpp_unordered_map_removals(void *ptr, const intptr_t *start, size_t count)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    identity(map)->erase(*it);
  }
}
