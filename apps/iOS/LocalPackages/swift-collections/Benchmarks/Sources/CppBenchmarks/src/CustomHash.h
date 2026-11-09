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

#ifndef CUSTOMHASH_H
#define CUSTOMHASH_H

#include <stdint.h>
#include <functional>

#include "Hashing.h"

extern cpp_hash_fn custom_hash_fn;

struct custom_intptr_hash: public std::function<std::size_t(intptr_t)>
{
  std::size_t
  operator()(intptr_t value) const
  {
    return static_cast<std::size_t>(custom_hash_fn(value));
  }
};

#endif /* CUSTOMHASH_H */
