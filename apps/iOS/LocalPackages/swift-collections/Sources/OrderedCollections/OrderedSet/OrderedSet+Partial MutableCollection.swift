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

// The parts of MutableCollection that OrderedSet is able to implement.

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension OrderedSet {
  /// Exchanges the values at the specified indices of the set.
  ///
  /// Both parameters must be valid indices below `endIndex`. Passing the same
  /// index as both `i` and `j` has no effect.
  ///
  /// - Parameters:
  ///   - i: The index of the first value to swap.
  ///   - j: The index of the second value to swap.
  ///
  /// - Complexity: O(1) when the set's storage isn't shared with another
  ///    value; O(`count`) otherwise.
  @inlinable
  public mutating func swapAt(_ i: Int, _ j: Int) {
    guard i != j else { return }
    _elements.swapAt(i, j)
    guard _table != nil else { return }
    _ensureUnique()
    _table!.update { hashTable in
      hashTable.swapBucketValues(for: _elements[i], withCurrentValue: j,
                                 and: _elements[j], withCurrentValue: i)
    }
    _checkInvariants()
  }

  /// Reorders the elements of the set such that all the elements that match the
  /// given predicate are after all the elements that don't match.
  ///
  /// After partitioning a collection, there is a pivot index `p` where
  /// no element before `p` satisfies the `belongsInSecondPartition`
  /// predicate and every element at or after `p` satisfies
  /// `belongsInSecondPartition`.
  ///
  /// In the following example, an ordered set of numbers is partitioned by a
  /// predicate that matches elements greater than 30.
  ///
  ///     var numbers: OrderedSet = [30, 40, 20, 30, 30, 60, 10]
  ///     let p = numbers.partition(by: { $0 > 30 })
  ///     // p == 5
  ///     // numbers == [30, 10, 20, 30, 30, 60, 40]
  ///
  /// The `numbers` set is now arranged in two partitions. The first partition,
  /// `numbers[..<p]`, is made up of the elements that are not greater than 30.
  /// The second partition, `numbers[p...]`, is made up of the elements that
  /// *are* greater than 30.
  ///
  ///     let first = numbers[..<p]
  ///     // first == [30, 10, 20, 30, 30]
  ///     let second = numbers[p...]
  ///     // second == [60, 40]
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
    try _partition(by: belongsInSecondPartition, callback: { a, b in })
  }
}

extension OrderedSet {
  @inlinable
  public mutating func _partition(
    by belongsInSecondPartition: (Element) throws -> Bool,
    callback: (Int, Int) -> Void
  ) rethrows -> Int {
    guard _table != nil else {
      return try _elements.partition(by: belongsInSecondPartition)
    }
    _ensureUnique()
    let result: Int = try _table!.update { hashTable in
      let maybeOffset: Int? = try _elements.withContiguousMutableStorageIfAvailable { buffer in
        let pivot = try buffer._partition(
          with: hashTable,
          by: belongsInSecondPartition,
          callback: callback)
        return pivot - buffer.startIndex
      }
      if let offset = maybeOffset {
        return _elements.index(startIndex, offsetBy: offset)
      }
      return try _elements._partition(
        with: hashTable,
        by: belongsInSecondPartition,
        callback: callback)
    }
    _checkInvariants()
    return result
  }
}

extension MutableCollection where Self: RandomAccessCollection, Element: Hashable {
  @inlinable
  internal mutating func _partition(
    with hashTable: _UnsafeHashTable,
    by belongsInSecondPartition: (Element) throws -> Bool,
    callback: (Int, Int) -> Void
  ) rethrows -> Index {
    var low = startIndex
    var high = endIndex

    while true {
      // Invariants at this point:
      // - low <= high
      // - all elements in `startIndex ..< low` belong in the first partition
      // - all elements in `high ..< endIndex` belong in the second partition

      // Find next element from `lo` that may not be in the right place.
      while true {
        if low == high { return low }
        if try belongsInSecondPartition(self[low]) { break }
        formIndex(after: &low)
      }

      // Find next element down from `hi` that we can swap `lo` with.
      while true {
        formIndex(before: &high)
        if low == high { return low }
        if try !belongsInSecondPartition(self[high]) { break }
      }

      // Swap the two elements as well as their associated hash table buckets.
      swapAt(low, high)
      let offsetLow = _offset(of: low)
      let offsetHigh = _offset(of: high)
      hashTable.swapBucketValues(for: self[low], withCurrentValue: offsetHigh,
                                 and: self[high], withCurrentValue: offsetLow)
      callback(offsetLow, offsetHigh)

      formIndex(after: &low)
    }
  }
}

extension _UnsafeHashTable {
  @inlinable
  @inline(__always)
  func swapBucketValues<Element: Hashable>(
    for left: Element, withCurrentValue leftValue: Int,
    and right: Element, withCurrentValue rightValue: Int
  ) {
    let left = idealBucket(for: left)
    let right = idealBucket(for: right)
    swapBucketValues(for: left, withCurrentValue: leftValue,
                     and: right, withCurrentValue: rightValue)
  }

  @usableFromInline
  @_effects(releasenone)
  func swapBucketValues(
    for left: Bucket, withCurrentValue leftValue: Int,
    and right: Bucket, withCurrentValue rightValue: Int
  ) {
    var it = bucketIterator(startingAt: left)
    it.advance(until: leftValue)
    assert(it.isOccupied)
    it.currentValue = rightValue

    it = bucketIterator(startingAt: right)
    it.advance(until: rightValue)
    assert(it.isOccupied)
    // Note: this second update may mistake the bucket for `right` with the
    // bucket for `left` whose value we just updated. The second update will
    // restore the original hash table contents in this case. This is okay!
    // When this happens, the lookup chains for both elements include each
    // other, so leaving the hash table unchanged still leaves us with a
    // working hash table.
    it.currentValue = leftValue
  }
}

extension OrderedSet {
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
    defer {
      // Note: This assumes that `sort(by:)` won't leave duplicate/missing
      // elements in the table when the closure throws. This matches the
      // stdlib's behavior in Swift 5.3, and it seems like a reasonable
      // long-term assumption.
      _regenerateExistingHashTable()
      _checkInvariants()
    }
    try _elements.sort(by: areInIncreasingOrder)
  }
}

extension OrderedSet where Element: Comparable {
  /// Sorts the set in place.
  ///
  /// You can sort an ordered set of elements that conform to the
  /// `Comparable` protocol by calling this method. Elements are sorted in
  /// ascending order.
  ///
  /// Here's an example of sorting a list of students' names. Strings in Swift
  /// conform to the `Comparable` protocol, so the names are sorted in
  /// ascending order according to the less-than operator (`<`).
  ///
  ///     var students: OrderedSet = ["Kofi", "Abena", "Peter", "Kweku", "Akosua"]
  ///     students.sort()
  ///     print(students)
  ///     // Prints "["Abena", "Akosua", "Kofi", "Kweku", "Peter"]"
  ///
  /// To sort the elements of your collection in descending order, pass the
  /// greater-than operator (`>`) to the `sort(by:)` method.
  ///
  ///     students.sort(by: >)
  ///     print(students)
  ///     // Prints "["Peter", "Kweku", "Kofi", "Akosua", "Abena"]"
  ///
  /// The sorting algorithm is guaranteed to be stable. A stable sort
  /// preserves the relative order of elements that compare as equal.
  ///
  /// - Complexity: O(*n* log *n*), where *n* is the length of the collection.
  @inlinable
  public mutating func sort() {
    defer {
      // Note: This assumes that `sort(by:)` won't leave duplicate/missing
      // elements in the table when the closure throws. This matches the
      // stdlib's behavior in Swift 5.3, and it seems like a reasonable
      // long-term assumption.
      _regenerateExistingHashTable()
      _checkInvariants()
    }
    _elements.sort()
  }
}

extension OrderedSet {
  /// Shuffles the collection in place.
  ///
  /// Use the `shuffle()` method to randomly reorder the elements of an ordered
  /// set.
  ///
  ///     var names: OrderedSet
  ///       = ["Alejandro", "Camila", "Diego", "Luciana", "Luis", "Sofía"]
  ///     names.shuffle()
  ///     // names == ["Luis", "Camila", "Luciana", "Sofía", "Alejandro", "Diego"]
  ///
  /// This method is equivalent to calling `shuffle(using:)`, passing in the
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
  ///     var names: OrderedSet
  ///       = ["Alejandro", "Camila", "Diego", "Luciana", "Luis", "Sofía"]
  ///     names.shuffle(using: &myGenerator)
  ///     // names == ["Sofía", "Alejandro", "Camila", "Luis", "Diego", "Luciana"]
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
    _elements.shuffle(using: &generator)
    _regenerateExistingHashTable()
    _checkInvariants()
  }
}

extension OrderedSet {
  /// Reverses the elements of the ordered set in place.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func reverse() {
    _elements.reverse()
    // FIXME: Update hash table contents in place.
    _regenerateHashTable()
    _checkInvariants()
  }
}

extension OrderedSet {

  /// Moves all elements satisfying `belongsInSecondPartition` into a suffix
  /// of the collection, returning the start position of the resulting suffix.
  /// On return, the items before this pivot index remain in the order they
  /// originally appeared in the collection.
  ///
  /// - Complexity: O(*n*) where n is the length of the collection.
  @inlinable
  internal mutating func _halfStablePartition<Value>(
    values: UnsafeMutableBufferPointer<Value>,
    by belongsInSecondPartition: ((key: Element, value: Value)) throws -> Bool
  ) rethrows -> Int {
    precondition(self.count == values.count)
    var i = 0
    try _elements.withUnsafeMutableBufferPointer { keys in
      while i < keys.count, try !belongsInSecondPartition((keys[i], values[i])) {
        i += 1
      }
    }
    guard i < self.count else { return self.count }

    self._ensureUnique()
    let table = _table
    self._table = nil
    defer { self._table = table }

    return try _elements.withUnsafeMutableBufferPointer { keys in
      for j in i + 1 ..< keys.count {
        guard try !belongsInSecondPartition((keys[j], values[j])) else {
          continue
        }
        keys.swapAt(i, j)
        values.swapAt(i, j)
        table?.update { hashTable in
          hashTable.swapBucketValues(for: keys[i], withCurrentValue: j,
                                     and: keys[j], withCurrentValue: i)
        }
        i += 1
      }
      return i
    }
  }

  @inlinable
  internal mutating func _partition<Value>(
    values: UnsafeMutableBufferPointer<Value>,
    by belongsInSecondPartition: ((key: Element, value: Value)) throws -> Bool
  ) rethrows -> Int {
    self._ensureUnique()
    let table = self._table
    self._table = nil
    defer { self._table = table }
    return try _elements.withUnsafeMutableBufferPointer { keys in
      assert(keys.count == values.count)
      var low = keys.startIndex
      var high = keys.endIndex

      while true {
        // Invariants at this point:
        // - low <= high
        // - all elements in `startIndex ..< low` belong in the first partition
        // - all elements in `high ..< endIndex` belong in the second partition

        // Find next element from `lo` that may not be in the right place.
        while true {
          if low == high { return low }
          if try belongsInSecondPartition((keys[low], values[low])) { break }
          low += 1
        }

        // Find next element down from `hi` that we can swap `lo` with.
        while true {
          high -= 1
          if low == high { return low }
          if try !belongsInSecondPartition((keys[high], values[high])) { break }
        }

        // Swap the two elements as well as their associated hash table buckets.
        keys.swapAt(low, high)
        values.swapAt(low, high)
        table?.update { hashTable in
          hashTable.swapBucketValues(for: keys[low], withCurrentValue: high,
                                     and: keys[high], withCurrentValue: low)
        }
        low += 1
      }
    }
  }
}
