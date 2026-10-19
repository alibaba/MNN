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

#include "Utils.h"

void
black_hole(intptr_t value)
{
  // Do nothing.
}

void
black_hole(void *value)
{
  // Do nothing.
}

intptr_t
identity(intptr_t value)
{
  return value;
}

void *
_identity(void *value)
{
  return value;
}

const void *
_identity(const void *value)
{
  return value;
}
