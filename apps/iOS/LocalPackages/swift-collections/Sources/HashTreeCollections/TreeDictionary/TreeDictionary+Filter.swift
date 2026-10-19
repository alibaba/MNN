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

extension TreeDictionary {
  /// Returns a new persistent dictionary containing the key-value pairs of this
  /// dictionary that satisfy the given predicate.
  ///
  /// - Parameter isIncluded: A closure that takes a key-value pair as its
  ///   argument and returns a Boolean value indicating whether it should be
  ///   included in the returned dictionary.
  ///
  /// - Returns: A dictionary of the key-value pairs that `isIncluded` allows.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Self {
    let result = try _root.filter(.top, isIncluded)
    guard let result = result else { return self }
    let r = TreeDictionary(_new: result.finalize(.top))
    r._invariantCheck()
    return r
  }

  /// Removes all the elements that satisfy the given predicate.
  ///
  /// Use this method to remove every element in the dictionary that meets
  /// particular criteria.
  /// This example removes all the odd valued items from a
  /// dictionary mapping strings to numbers:
  ///
  ///     var numbers: TreeDictionary = ["a": 5, "b": 6, "c": 7, "d": 8]
  ///     numbers.removeAll(where: { $0.value % 2 != 0 })
  ///     // numbers == ["b": 6, "d": 8]
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   dictionary as its argument and returns a Boolean value indicating
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
