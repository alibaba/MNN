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

#include "MapBenchmarks.h"
#include <map>
#include <cstdlib>
#include "Utils.h"

typedef std::map<intptr_t, intptr_t> custom_map;

void *
cpp_map_create(const intptr_t *start, size_t count)
{
  auto map = new custom_map();
  for (size_t i = 0; i < count; ++i) {
    map->insert({start[i], 2 * start[i]});
  }
  return map;
}

void
cpp_map_destroy(void *ptr)
{
  delete static_cast<custom_map *>(ptr);
}

void
cpp_map_insert_integers(const intptr_t *start, size_t count)
{
  auto map = custom_map();
  auto end = start + count;
  for (auto p = start; p != end; ++p) {
    auto v = *identity(p);
    map.insert({ v, 2 * v });
  }
  black_hole(&map);
}

__attribute__((noinline))
auto find(custom_map* map, intptr_t value)
{
  return map->find(value);
}

void
cpp_map_lookups(void *ptr, const intptr_t *start, size_t count)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    auto isCorrect = find(map, *it)->second == *it * 2;
    if (!isCorrect) { abort(); }
  }
}

void
cpp_map_subscript(void *ptr, const intptr_t *start, size_t count)
{
  auto map = static_cast<custom_map *>(ptr);
  for (auto it = start; it < start + count; ++it) {
    black_hole((*map)[*it]);
  }
}
