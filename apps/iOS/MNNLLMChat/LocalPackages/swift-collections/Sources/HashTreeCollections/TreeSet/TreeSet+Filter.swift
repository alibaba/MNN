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

extension TreeSet {
  /// Returns a new persistent set containing all the members of this persistent
  /// set that satisfy the given predicate.
  ///
  /// - Parameter isIncluded: A closure that takes a value as its
  ///   argument and returns a Boolean value indicating whether the value
  ///   should be included in the returned set.
  ///
  /// - Returns: A set of the values that `isIncluded` allows.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Self {
    let result = try _root.filter(.top) { try isIncluded($0.key) }
    guard let result = result else { return self }
    return TreeSet(_new: result.finalize(.top))
  }

  /// Removes all the elements that satisfy the given predicate.
  ///
  /// Use this method to remove every element in the set that meets
  /// particular criteria.
  /// This example removes all the odd values from a
  /// set of numbers:
  ///
  ///     var numbers: TreeSet = [5, 6, 7, 8, 9, 10, 11]
  ///     numbers.removeAll(where: { $0 % 2 != 0 })
  ///     // numbers == [6, 8, 10]
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   set as its argument and returns a Boolean value indicating
  ///   whether the element should be removed from the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(
    where shouldBeRemoved: (Element) throws -> Bool
  ) rethrows {
    // FIXME: Implement in-place reductions
    self = try filter { try !shouldBeRemoved($0) }
  }
}
