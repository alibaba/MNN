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

#ifndef CPPBENCHMARKS_DEQUE_BENCHMARKS_H
#define CPPBENCHMARKS_DEQUE_BENCHMARKS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a std::deque, populating it with data from the supplied buffer.
/// Returns an opaque pointer to the created instance.
extern void *cpp_deque_create(const intptr_t *start, size_t count);

/// Destroys a deque previously returned by `cpp_deque_create`.
extern void cpp_deque_destroy(void *ptr);

extern void cpp_deque_from_int_range(intptr_t count);
extern void cpp_deque_from_int_buffer(const intptr_t *start, size_t count);

extern void cpp_deque_append_integers(const intptr_t *start, size_t count);
extern void cpp_deque_prepend_integers(const intptr_t *start, size_t count);
extern void cpp_deque_random_insertions(const intptr_t *start, size_t count);

extern void cpp_deque_iterate(void *ptr);
extern void cpp_deque_lookups_subscript(void *ptr, const intptr_t *start, size_t count);
extern void cpp_deque_lookups_at(void *ptr, const intptr_t *start, size_t count);
extern void cpp_deque_pop_back(void *ptr);
extern void cpp_deque_pop_front(void *ptr);
extern void cpp_deque_random_removals(void *ptr, const intptr_t *start, size_t count);
extern void cpp_deque_sort(void *ptr);

#ifdef __cplusplus
}
#endif


#endif /* CPPBENCHMARKS_DEQUE_BENCHMARKS_H */
