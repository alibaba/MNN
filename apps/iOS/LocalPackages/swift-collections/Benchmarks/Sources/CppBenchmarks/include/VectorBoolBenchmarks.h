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

#ifndef CPPBENCHMARKS_VECTOR_BOOL_BENCHMARKS_H
#define CPPBENCHMARKS_VECTOR_BOOL_BENCHMARKS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a `std::vector<bool>` of the specified size, filling it with the
/// given value. Returns an opaque pointer to the created instance.
extern void *cpp_vector_bool_create_repeating(size_t count, bool value);

/// Destroy a vector previously returned by `cpp_vector_bool_create`.
extern void cpp_vector_bool_destroy(void *ptr);

extern void cpp_vector_bool_push_back(const bool *start, size_t count, bool reserve);

extern void cpp_vector_bool_pop_back(void *ptr, size_t count);

/// Set bits indexed by a buffer of integers to true, using the unchecked subscript.
extern void cpp_vector_bool_set_indices_subscript(void *ptr, const intptr_t *start, size_t count);

/// Set bits indexed by a buffer of integers to true, using the checked `at` method.
extern void cpp_vector_bool_set_indices_at(void *ptr, const intptr_t *start, size_t count);

/// Set bits indexed by a buffer of integers to false, using the unchecked subscript.
extern void cpp_vector_bool_reset_indices_subscript(void *ptr, const intptr_t *start, size_t count);

/// Set bits indexed by a buffer of integers to false, using the checked `at` method.
extern void cpp_vector_bool_reset_indices_at(void *ptr, const intptr_t *start, size_t count);

/// Retrieve all bits indexed by a buffer of integers, using the unchecked subscript.
extern void cpp_vector_bool_lookups_subscript(void *ptr, const intptr_t *start, size_t count);

/// Retrieve all bits indexed by a buffer of integers, using the checked `at` method.
extern void cpp_vector_bool_lookups_at(void *ptr, const intptr_t *start, size_t count);

/// Iterate through all the bits in a `vector<bool>`.
extern void cpp_vector_bool_iterate(void *ptr);

/// Use `std::find` to visit every true bit in a `vector<bool>`, returning
/// the number of true bits found.
extern size_t cpp_vector_bool_find_true_bits(void *ptr);

/// Use `std::count` to return a count of every true bit in a `vector<bool>`.
extern size_t cpp_vector_bool_count_true_bits(void *ptr);

#ifdef __cplusplus
}
#endif


#endif /* CPPBENCHMARKS_VECTOR_BOOL_BENCHMARKS_H */
