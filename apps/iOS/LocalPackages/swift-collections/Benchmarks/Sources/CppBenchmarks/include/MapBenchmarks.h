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

#ifndef CPPBENCHMARKS_MAP_BENCHMARKS_H
#define CPPBENCHMARKS_MAP_BENCHMARKS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a std::map, populating it with data from the supplied buffer.
/// Returns an opaque pointer to the created instance.
extern void *cpp_map_create(const intptr_t *start, size_t count);

/// Destroys an ordered map previously returned by `cpp_map_create`.
extern void cpp_map_destroy(void *ptr);

extern void cpp_map_insert_integers(const intptr_t *start, size_t count);

extern void cpp_map_lookups(void *ptr, const intptr_t *start, size_t count);
extern void cpp_map_subscript(void *ptr, const intptr_t *start, size_t count);

#ifdef __cplusplus
}
#endif


#endif /* CPPBENCHMARKS_MAP_BENCHMARKS_H */
