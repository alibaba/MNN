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

/// A Collection type that is guaranteed not to contain any duplicate elements.
///
/// Types conforming to this protocol must also conform to `Collection`,
/// with an `Element` type that conforms to `Equatable`.
/// (However, this protocol does not specify these as explicit requirements,
/// to allow simple conformance tests such as `someValue is _SortedCollection`
/// to be possible.)
///
/// For any two valid indices `i` and `j` in a conforming collection `c`
/// (both below the end index), it must hold true that if `i != j` then
/// `c[i] != c[j]`.
///
/// Types that conform to this protocol should also implement the following
/// underscored requirements in a way that they never return nil values:
///
/// - `Sequence._customContainsEquatableElement`
/// - `Collection._customIndexOfEquatableElement`
/// - `Collection._customLastIndexOfEquatableElement`
///
/// The idea with these is that presumably a collection that can guarantee
/// element uniqueness has a way to quickly find existing elements.
public protocol _UniqueCollection {}

extension Set: _UniqueCollection {}
extension Dictionary.Keys: _UniqueCollection {}
extension Slice: _UniqueCollection where Base: _UniqueCollection {}
