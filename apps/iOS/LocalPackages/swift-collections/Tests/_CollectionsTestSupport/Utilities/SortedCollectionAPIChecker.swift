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

#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import InternalCollectionsUtilities
#endif

/// This protocol simply lists Collection/Sequence extensions that ought to be
/// customized for sorted collections.
///
/// Conforming to this protocol admittedly doesn't do much, as the default
/// implementations already exist for most of these requirements
/// (but they aren't doing the right thing).
public protocol SortedCollectionAPIChecker: Collection, _SortedCollection
where Element: Comparable {
  // This one actually does not come with a default implementation.
  func sorted() -> Self

  // These are also defined on `Sequence`, but the default implementation does
  // a linear search, which isn't what we want.
  func min() -> Element?
  func max() -> Element?
}
