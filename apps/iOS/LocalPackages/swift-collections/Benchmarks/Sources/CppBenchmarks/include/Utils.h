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

#ifndef BLACK_HOLE_H
#define BLACK_HOLE_H

#include <stdint.h>

#ifdef __cplusplus

// FIXME: Is putting this in a separate compilation unit enough to make
// sure the function call is always emitted?

extern void black_hole(intptr_t value);
extern void black_hole(void *value);

extern intptr_t identity(intptr_t value);
extern void *_identity(void *value);
extern const void *_identity(const void *value);

template <typename T>
static inline T *identity(T *value)
{
  return static_cast<T *>(_identity(value));
}

template <typename T>
static inline const T *identity(const T *value)
{
  return static_cast<const T *>(_identity(value));
}

#endif /* __cplusplus */
#endif /* BLACK_HOLE_H */
