//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

/// A Collection type that is guaranteed to contain elements in monotonically
/// increasing order. (Duplicates are still allowed unless the collection
/// also conforms to `_UniqueCollection`.)
///
/// Types conforming to this protocol must also conform to `Collection`,
/// with an `Element` type that conforms to `Comparable`.
/// (However, this protocol does not specify these as explicit requirements,
/// to allow simple conformance tests such as `someValue is _SortedCollection`
/// to be possible.)
///
/// For any two valid indices `i` and `j` for a conforming collection `c`
/// (both below the end index), it must hold true that if `i < j`, then
/// `c[i] <= c[j]`.
public protocol _SortedCollection {}

extension Slice: _SortedCollection where Base: _SortedCollection {}
