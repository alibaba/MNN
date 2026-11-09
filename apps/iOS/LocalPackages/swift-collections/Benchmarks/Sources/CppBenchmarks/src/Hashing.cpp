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

#import "Hashing.h"
#import "CustomHash.h"
#import "Utils.h"

cpp_hash_fn custom_hash_fn;

void
cpp_set_hash_fn(cpp_hash_fn fn)
{
  custom_hash_fn = fn;
}

void
cpp_hash(const intptr_t *start, size_t count)
{
  for (auto p = start; p < start + count; ++p) {
    black_hole(std::hash<intptr_t>{}(*p));
  }
}

void
cpp_custom_hash(const intptr_t *start, size_t count)
{
  for (auto p = start; p < start + count; ++p) {
    black_hole(custom_intptr_hash{}(*p));
  }
}

