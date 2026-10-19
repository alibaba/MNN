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

#ifndef CPPBENCHMARKS_PRIORITY_QUEUE_BENCHMARKS_H
#define CPPBENCHMARKS_PRIORITY_QUEUE_BENCHMARKS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a `std::priority_queue<intptr_t>`, populating it with data from the
/// supplied buffer. Returns an opaque pointer to the created instance.
extern void *cpp_priority_queue_create(const intptr_t *start, size_t count);

/// Destroys a priority queue previously returned by `cpp_priority_queue_create`.
extern void cpp_priority_queue_destroy(void *ptr);

/// Push a value to a priority queue.
extern void cpp_priority_queue_push(void *ptr, intptr_t value);

/// Loop through the supplied buffer, pushing each value to the queue.
extern void cpp_priority_queue_push_loop(void *ptr, const intptr_t *start, size_t count);

/// Remove and return the top value off of a priority queue.
extern intptr_t cpp_priority_queue_pop(void *ptr);

/// Remove and discard all values in a priority queue one by one in a loop.
extern void cpp_priority_queue_pop_all(void *ptr);

#ifdef __cplusplus
}
#endif


#endif /* CPPBENCHMARKS_PRIORITY_QUEUE_BENCHMARKS_H */
