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

#ifndef CPPBENCHMARKS_HASHING_H
#define CPPBENCHMARKS_HASHING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t (*cpp_hash_fn)(intptr_t);
extern void cpp_set_hash_fn(cpp_hash_fn fn);

// Benchmarks
void cpp_hash(const intptr_t *start, size_t count);
void cpp_custom_hash(const intptr_t *start, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* CPPBENCHMARKS_HASHING_H */
