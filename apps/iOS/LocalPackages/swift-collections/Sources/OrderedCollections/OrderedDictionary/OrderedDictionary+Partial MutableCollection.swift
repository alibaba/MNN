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

extension OrderedDictionary {
  /// Exchanges the key-value pairs at the specified indices of the dictionary.
  ///
  /// Both parameters must be valid indices below `endIndex`. Passing the same
  /// index as both `i` and `j` has no effect.
  ///
  /// - Parameters:
  ///   - i: The index of the first value to swap.
  ///   - j: The index of the second value to swap.
  ///
  /// - Complexity: O(1) when the dictionary's storage isn't shared with another
  ///    value; O(`count`) otherwise.
  @inlinable
  public mutating func swapAt(_ i: Int, _ j: Int) {
    _keys.swapAt(i, j)
    _values.swapAt(i, j)
  }

  /// Reorders the elements of the dictionary such that all the elements that
  /// match the given predicate are after all the elements that don't match.
  ///
  /// After partitioning a collection, there is a pivot index `p` where
  /// no element before `p` satisfies the `belongsInSecondPartition`
  /// predicate and every element at or after `p` satisfies
  /// `belongsInSecondPartition`.
  ///
  /// - Parameter belongsInSecondPartition: A predicate used to partition
  ///   the collection. All elements satisfying this predicate are ordered
  ///   after all elements not satisfying it.
  /// - Returns: The index of the first element in the reordered collection
  ///   that matches `belongsInSecondPartition`. If no elements in the
  ///   collection match `belongsInSecondPartition`, the returned index is
  ///   equal to the collection's `endIndex`.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func partition(
    by belongsInSecondPartition: (Element) throws -> Bool
  ) rethrows -> Int {
    let pivot = try _values.withUnsafeMutableBufferPointer { values in
      try _keys._partition(values: values, by: belongsInSecondPartition)
    }
    _checkInvariants()
    return pivot
  }
}

extension OrderedDictionary {
  /// Sorts the collection in place, using the given predicate as the
  /// comparison between elements.
  ///
  /// When you want to sort a collection of elements that don't conform to
  /// the `Comparable` protocol, pass a closure to this method that returns
  /// `true` when the first element should be ordered before the second.
  ///
  /// Alternatively, use this method to sort a collection of elements that do
  /// conform to `Comparable` when you want the sort to be descending instead
  /// of ascending. Pass the greater-than operator (`>`) operator as the
  /// predicate.
  ///
  /// `areInIncreasingOrder` must be a *strict weak ordering* over the
  /// elements. That is, for any elements `a`, `b`, and `c`, the following
  /// conditions must hold:
  ///
  /// - `areInIncreasingOrder(a, a)` is always `false`. (Irreflexivity)
  /// - If `areInIncreasingOrder(a, b)` and `areInIncreasingOrder(b, c)` are
  ///   both `true`, then `areInIncreasingOrder(a, c)` is also `true`.
  ///   (Transitive comparability)
  /// - Two elements are *incomparable* if neither is ordered before the other
  ///   according to the predicate. If `a` and `b` are incomparable, and `b`
  ///   and `c` are incomparable, then `a` and `c` are also incomparable.
  ///   (Transitive incomparability)
  ///
  /// The sorting algorithm is guaranteed to be stable. A stable sort
  /// preserves the relative order of elements for which
  /// `areInIncreasingOrder` does not establish an order.
  ///
  /// - Parameter areInIncreasingOrder: A predicate that returns `true` if its
  ///   first argument should be ordered before its second argument;
  ///   otherwise, `false`. If `areInIncreasingOrder` throws an error during
  ///   the sort, the elements may be in a different order, but none will be
  ///   lost.
  ///
  /// - Complexity: O(*n* log *n*), where *n* is the length of the collection.
  @inlinable
  public mutating func sort(
    by areInIncreasingOrder: (Element, Element) throws -> Bool
  ) rethrows {
    // FIXME: Implement in-place sorting.
    let temp = try self.sorted(by: areInIncreasingOrder)
    precondition(temp.count == self.count)
    temp.withUnsafeBufferPointer { source in
      _keys = OrderedSet(uncheckedUniqueElements: source.lazy.map { $0.key })
      _values = ContiguousArray(source.lazy.map { $0.value })
    }
    _checkInvariants()
  }
}

extension OrderedDictionary where Key: Comparable {
  /// Sorts the dictionary in place.
  ///
  /// You can sort an ordered dictionary of keys that conform to the
  /// `Comparable` protocol by calling this method. The key-value pairs are
  /// sorted in ascending order. (`Value` doesn't need to conform to
  /// `Comparable` because the keys are guaranteed to be unique.)
  ///
  /// The sorting algorithm is guaranteed to be stable. A stable sort
  /// preserves the relative order of elements that compare as equal.
  ///
  /// - Complexity: O(*n* log *n*), where *n* is the length of the collection.
  @inlinable
  public mutating func sort() {
    sort { $0.key < $1.key }
  }
}

extension OrderedDictionary {
  /// Shuffles the collection in place.
  ///
  /// Use the `shuffle()` method to randomly reorder the elements of an ordered
  /// dictionary.
  ///
  /// This method is equivalent to calling ``shuffle(using:)``, passing in the
  /// system's default random generator.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  @inlinable
  public mutating func shuffle() {
    var generator = SystemRandomNumberGenerator()
    shuffle(using: &generator)
  }

  /// Shuffles the collection in place, using the given generator as a source
  /// for randomness.
  ///
  /// You use this method to randomize the elements of a collection when you
  /// are using a custom random number generator. For example, you can use the
  /// `shuffle(using:)` method to randomly reorder the elements of an array.
  ///
  /// - Parameter generator: The random number generator to use when shuffling
  ///   the collection.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  ///
  /// - Note: The algorithm used to shuffle a collection may change in a future
  ///   version of Swift. If you're passing a generator that results in the
  ///   same shuffled order each time you run your program, that sequence may
  ///   change when your program is compiled using a different version of
  ///   Swift.
  @inlinable
  public mutating func shuffle(
    using generator: inout some RandomNumberGenerator
  ) {
    guard count > 1 else { return }
    var keys = self._keys.elements
    var values = self._values
    self = [:]
    var amount = keys.count
    var current = 0
    while amount > 1 {
      let random = Int.random(in: 0 ..< amount, using: &generator)
      amount -= 1
      keys.swapAt(current, current + random)
      values.swapAt(current, current + random)
      current += 1
    }
    self = OrderedDictionary(uncheckedUniqueKeys: keys, values: values)
  }
}

extension OrderedDictionary {
  /// Reverses the elements of the ordered dictionary in place.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func reverse() {
    _keys.reverse()
    _values.reverse()
  }
}

