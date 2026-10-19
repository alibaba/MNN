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

// `OrderedSet` does not directly conform to `SetAlgebra` because its definition
// of equality conflicts with `SetAlgebra` requirements. However, it still
// implements most `SetAlgebra` requirements (except `insert`, which is replaced
// by `append`).
//
// `OrderedSet` also provides an `unordered` view that explicitly conforms to
// `SetAlgebra`. That view implements `Equatable` by ignoring element order,
// so it can satisfy `SetAlgebra` requirements.

extension OrderedSet {
  /// Creates an empty set.
  ///
  /// This initializer is equivalent to initializing with an empty array
  /// literal.
  ///
  /// - Complexity: O(1)
  @inlinable
  public init() {
    __storage = nil
    _elements = []
  }
}

extension OrderedSet {
  /// Returns a Boolean value that indicates whether the given element exists
  /// in the set.
  ///
  /// - Parameter element: An element to look for in the set.
  ///
  /// - Returns: `true` if `member` exists in the set; otherwise, `false`.
  ///
  /// - Complexity: This operation is expected to perform O(1) comparisons on
  ///    average, provided that `Element` implements high-quality hashing.
  @inlinable
  public func contains(_ element: Element) -> Bool {
    _find_inlined(element).index != nil
  }
}

extension OrderedSet {
  /// Removes the given element from the set.
  ///
  /// - Parameter member: The element of the set to remove.
  ///
  /// - Returns: The element equal to `member` if `member` is contained in the
  ///    set; otherwise, `nil`. In some cases, the returned element may be
  ///    distinguishable from `member` by identity comparison or some other
  ///    means.
  ///
  /// - Complexity: O(`count`). Removing an element from the middle of the
  ///    underlying ordered set needs to rearrange the remaining elements to
  ///    close the resulting gap.
  ///
  ///    Removing the last element only takes (amortized) O(1)
  ///    hashing/comparisons operations, if `Element` implements high quality
  ///    hashing.
  @inlinable
  @discardableResult
  public mutating func remove(_ member: Element) -> Element? {
    let (idx, bucket) = _find(member)
    guard let index = idx else { return nil }
    return _removeExistingMember(at: index, in: bucket)
  }
}

