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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension BitSet {
  public struct Counted {
    @usableFromInline
    internal var _bits: BitSet
    @usableFromInline
    internal var _count: Int

    internal init(_bits: BitSet, count: Int) {
      self._bits = _bits
      self._count = count
      _checkInvariants()
    }
  }
}

extension BitSet.Counted: Sendable {}

extension BitSet.Counted {
#if COLLECTIONS_INTERNAL_CHECKS
  @inline(never)
  @_effects(releasenone)
  public func _checkInvariants() {
    _bits._checkInvariants()
    precondition(_count == _bits.count)
  }
#else
  @inline(__always) @inlinable
  public func _checkInvariants() {}
#endif // COLLECTIONS_INTERNAL_CHECKS
}

extension BitSet.Counted: _UniqueCollection {}

extension BitSet.Counted {
  public init() {
    self.init(BitSet())
  }

  @inlinable
  public init(words: some Sequence<UInt>) {
    self.init(BitSet(words: words))
  }

  @inlinable
  public init(bitPattern x: some BinaryInteger) {
    self.init(words: x.words)
  }

  public init(_ array: BitArray) {
    self.init(BitSet(array))
  }

  @inlinable
  public init(_ elements: __owned some Sequence<Int>) {
    self.init(BitSet(elements))
  }

  public init(_ range: Range<Int>) {
    self.init(BitSet(range))
  }
}

extension BitSet.Counted {
  public init(_ bits: BitSet) {
    self.init(_bits: bits, count: bits.count)
  }

  public var uncounted: BitSet {
    get { _bits }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      defer {
        _count = _bits.count
      }
      yield &_bits
    }
  }
}

extension BitSet {
  public init(_ bits: BitSet.Counted) {
    self = bits._bits
  }

  public var counted: BitSet.Counted {
    get {
      BitSet.Counted(_bits: self, count: self.count)
    }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      var value = BitSet.Counted(self)
      self = []
      defer { self = value._bits }
      yield &value
    }
  }
}

extension BitSet.Counted: Sequence {
  public typealias Iterator = BitSet.Iterator

  public var underestimatedCount: Int {
    _count
  }

  public func makeIterator() -> BitSet.Iterator {
    _bits.makeIterator()
  }

  public func _customContainsEquatableElement(
    _ element: Int
  ) -> Bool? {
    _bits._customContainsEquatableElement(element)
  }
}

extension BitSet.Counted: BidirectionalCollection {
  public typealias Index = BitSet.Index

  public var isEmpty: Bool { _count == 0 }

  public var count: Int { _count }

  public var startIndex: Index { _bits.startIndex }
  public var endIndex: Index { _bits.endIndex }

  public subscript(position: Index) -> Int {
    _bits[position]
  }

  public func index(after index: Index) -> Index {
    _bits.index(after: index)
  }

  public func index(before index: Index) -> Index {
    _bits.index(before: index)
  }

  public func distance(from start: Index, to end: Index) -> Int {
    _bits.distance(from: start, to: end)
  }

  public func index(_ index: Index, offsetBy distance: Int) -> Index {
    _bits.index(index, offsetBy: distance)
  }

  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    _bits.index(i, offsetBy: distance, limitedBy: limit)
  }

  public func _customIndexOfEquatableElement(_ element: Int) -> Index?? {
    _bits._customIndexOfEquatableElement(element)
  }

  public func _customLastIndexOfEquatableElement(_ element: Int) -> Index?? {
    _bits._customLastIndexOfEquatableElement(element)
  }
}


extension BitSet.Counted: Codable {
  public func encode(to encoder: Encoder) throws {
    try _bits.encode(to: encoder)
  }

  public init(from decoder: Decoder) throws {
    self.init(try BitSet(from: decoder))
  }
}

extension BitSet.Counted: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _bits.description
  }
}

extension BitSet.Counted: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension BitSet.Counted: CustomReflectable {
  public var customMirror: Mirror {
    _bits.customMirror
  }
}

extension BitSet.Counted: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    guard left._count == right._count else { return false }
    return left._bits == right._bits
  }
}

extension BitSet.Counted: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(_bits)
  }
}

extension BitSet.Counted: ExpressibleByArrayLiteral {
  @inlinable
  public init(arrayLiteral elements: Int...) {
    let bits = BitSet(elements)
    self.init(bits)
  }
}

extension BitSet.Counted { // Extras
  public init(reservingCapacity maximumValue: Int) {
    self.init(_bits: BitSet(reservingCapacity: maximumValue), count: 0)
  }

  public mutating func reserveCapacity(_ maximumValue: Int) {
    _bits.reserveCapacity(maximumValue)
  }

  public subscript(member member: Int) -> Bool {
    get { contains(member) }
    set {
      if newValue {
        insert(member)
      } else {
        remove(member)
      }
    }
  }

  public subscript(members bounds: Range<Int>) -> Slice<BitSet> {
    _bits[members: bounds]
  }

  public subscript(members bounds: some RangeExpression<Int>) -> Slice<BitSet> {
    _bits[members: bounds]
  }

  public mutating func remove(at index: Index) -> Int {
    defer { self._count &-= 1 }
    return _bits.remove(at: index)
  }

  public func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Self {
    BitSet.Counted(try _bits.filter(isIncluded))
  }
}

extension BitSet.Counted: _SortedCollection {
  /// Returns the current set (already sorted).
  ///
  /// - Complexity: O(1)
  public func sorted() -> BitSet.Counted { self }

  /// Returns the minimum element in this set.
  ///
  /// Bit sets are sorted, so the minimum element is always at the first
  /// position in the set.
  ///
  /// - Returns: The bit set's minimum element. If the sequence has no
  ///   elements, returns `nil`.
  ///
  /// - Complexity: O(1)
  @warn_unqualified_access
  public func min() -> Element? {
    first
  }

  /// Returns the maximum element in this set.
  ///
  /// Bit sets are sorted, so the maximum element is always at the last
  /// position in the set.
  ///
  /// - Returns: The bit set's maximum element. If the sequence has no
  ///   elements, returns `nil`.
  ///
  /// - Complexity: O(1)
  @warn_unqualified_access
  public func max() -> Element? {
    last
  }
}

extension BitSet.Counted {
  public static func random(upTo limit: Int) -> BitSet.Counted {
    BitSet.Counted(BitSet.random(upTo: limit))
  }

  public static func random(
    upTo limit: Int,
    using rng: inout some RandomNumberGenerator
  ) -> BitSet.Counted {
    BitSet.Counted(BitSet.random(upTo: limit, using: &rng))
  }
}

extension BitSet.Counted: SetAlgebra {
  public func contains(_ member: Int) -> Bool {
    _bits.contains(member)
  }

  @discardableResult
  public mutating func insert(
    _ newMember: Int
  ) -> (inserted: Bool, memberAfterInsert: Int) {
    let r = _bits.insert(newMember)
    if r.inserted { _count += 1 }
    return r
  }

  @discardableResult
  public mutating func update(with newMember: Int) -> Int? {
    _bits.update(with: newMember)
  }

  @discardableResult
  public mutating func remove(_ member: Int) -> Int? {
    let old = _bits.remove(member)
    if old != nil { _count -= 1 }
    return old
  }
}

extension BitSet.Counted {
  /// Returns a new set with the elements of both this and the given set.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: BitSet.Counted) -> BitSet.Counted {
    _bits.union(other).counted
  }

  /// Returns a new set with the elements of both this and the given set.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: BitSet) -> BitSet.Counted {
    _bits.union(other).counted
  }

  /// Returns a new set with the elements of both this set and the given
  /// range of integers.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.union(3 ..< 7) // [1, 2, 3, 4, 5, 6]
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: Range<Int>) -> BitSet.Counted {
    _bits.union(other).counted
  }

  /// Returns a new set with the elements of both this set and the given
  /// sequence of integers.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public func union(
    _ other: __owned some Sequence<Int>
  ) -> Self {
    _bits.union(other).counted
  }
}

extension BitSet.Counted {
  /// Returns a new bit set with the elements that are common to both this set
  /// and the given set.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [6, 4, 2, 0]
  ///     let c = a.intersection(b)
  ///     // c is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public func intersection(_ other: BitSet.Counted) -> BitSet.Counted {
    _bits.intersection(other).counted
  }

  /// Returns a new bit set with the elements that are common to both this set
  /// and the given set.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet = [6, 4, 2, 0]
  ///     let c = a.intersection(b)
  ///     // c is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public func intersection(_ other: BitSet) -> BitSet.Counted {
    _bits.intersection(other).counted
  }

  /// Returns a new bit set with the elements that are common to both this set
  /// and the given range of integers.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let c = a.intersection(-10 ..< 3)
  ///     // c is now [3, 4]
  ///
  /// - Parameter other: A range of integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  public func intersection(_ other: Range<Int>) -> BitSet.Counted {
    _bits.intersection(other).counted
  }

  /// Returns a new bit set with the elements that are common to both this set
  /// and the given sequence.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b = [6, 4, 2, 0]
  ///     let c = a.intersection(b)
  ///     // c is now [2, 4]
  ///
  /// - Parameter other: A sequence of integer values.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func intersection(
    _ other: __owned some Sequence<Int>
  ) -> Self {
    _bits.intersection(other).counted
  }
}

extension BitSet.Counted {
  /// Returns a new bit set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [6, 4, 2, 0]
  ///     set.symmetricDifference(other) // [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public func symmetricDifference(_ other: BitSet.Counted) -> BitSet.Counted {
    _bits.symmetricDifference(other).counted
  }

  /// Returns a new bit set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [6, 4, 2, 0]
  ///     set.symmetricDifference(other) // [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public func symmetricDifference(_ other: BitSet) -> BitSet.Counted {
    _bits.symmetricDifference(other).counted
  }

  /// Returns a new bit set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.formSymmetricDifference(3 ..< 7) // [1, 2, 5, 6]
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func symmetricDifference(_ other: Range<Int>) -> BitSet.Counted {
    _bits.symmetricDifference(other).counted
  }

  /// Returns a new bit set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.formSymmetricDifference(other) // [0, 1, 3, 6]
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public func symmetricDifference(
    _ other: __owned some Sequence<Int>
  ) -> Self {
    _bits.symmetricDifference(other).counted
  }
}

extension BitSet.Counted {
  /// Returns a new set containing the elements of this set that do not occur
  /// in the given other set.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func subtracting(_ other: BitSet.Counted) -> BitSet.Counted {
    _bits.subtracting(other).counted
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given other set.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func subtracting(_ other: BitSet) -> BitSet.Counted {
    _bits.subtracting(other).counted
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given range of integers.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.subtracting(-10 ..< 3) // [3, 4]
  ///
  /// - Parameter other: A range of arbitrary integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in self.
  public func subtracting(_ other: Range<Int>) -> BitSet.Counted {
    _bits.subtracting(other).counted
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given sequence of integers.
  ///
  ///     let set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, -2, -4]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func subtracting(
    _ other: __owned some Sequence<Int>
  ) -> Self {
    _bits.subtracting(other).counted
  }
}

extension BitSet.Counted {
  /// Adds the elements of the given set to this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.formUnion(other)
  ///     // `set` is now `[0, 1, 2, 3, 4, 6]`
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func formUnion(_ other: BitSet.Counted) {
    _bits.formUnion(other._bits)
    _count = _bits.count
    _checkInvariants()
  }

  /// Adds the elements of the given set to this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.formUnion(other)
  ///     // `set` is now `[0, 1, 2, 3, 4, 6]`
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func formUnion(_ other: BitSet) {
    _bits.formUnion(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Adds the elements of the given range of integers to this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.formUnion(3 ..< 7)
  ///     // `set` is now `[1, 2, 3, 4, 5, 6]`
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func formUnion(_ other: Range<Int>) {
    _bits.formUnion(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Adds the elements of the given sequence to this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.formUnion(other)
  ///     // `set` is now `[0, 1, 2, 3, 4, 6]`
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public mutating func formUnion(
    _ other: __owned some Sequence<Int>
  ) {
    _bits.formUnion(other)
    _count = _bits.count
    _checkInvariants()
  }
}

extension BitSet.Counted {
  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formIntersection(_ other: BitSet.Counted) {
    _bits.formIntersection(other._bits)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formIntersection(_ other: BitSet) {
    _bits.formIntersection(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of this set that aren't also in the given range.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.formIntersection(-10 ..< 3)
  ///     // set is now [3, 4]
  ///
  /// - Parameter other: A range of integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public mutating func formIntersection(_ other: Range<Int>) {
    _bits.formIntersection(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of this set that aren't also in the given sequence.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: Set<Int> = [6, 4, 2, 0]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A sequence of integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///     and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public mutating func formIntersection(
    _ other: __owned some Sequence<Int>
  ) {
    _bits.formIntersection(other)
    _count = _bits.count
    _checkInvariants()
  }
}

extension BitSet.Counted {
  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formSymmetricDifference(_ other: BitSet.Counted) {
    _bits.formSymmetricDifference(other._bits)
    _count = _bits.count
    _checkInvariants()
  }

  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formSymmetricDifference(_ other: BitSet) {
    _bits.formSymmetricDifference(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Replace this set with the elements contained in this set or the given
  /// range of integers, but not both.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.formSymmetricDifference(3 ..< 7)
  ///     // set is now [1, 2, 5, 6]
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func formSymmetricDifference(_ other: Range<Int>) {
    _bits.formSymmetricDifference(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Replace this set with the elements contained in this set or the given
  /// sequence, but not both.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public mutating func formSymmetricDifference(
    _ other: __owned some Sequence<Int>
  ) {
    _bits.formSymmetricDifference(other)
    _count = _bits.count
    _checkInvariants()
  }
}

extension BitSet.Counted {
  /// Removes the elements of the given bit set from this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.subtract(other)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func subtract(_ other: BitSet.Counted) {
    _bits.subtract(other._bits)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of the given bit set from this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.subtract(other)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func subtract(_ other: BitSet) {
    _bits.subtract(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of the given range of integers from this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     set.subtract(-10 ..< 3)
  ///     // set is now [3, 4]
  ///
  /// - Parameter other: A range of arbitrary integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in self.
  public mutating func subtract(_ other: Range<Int>) {
    _bits.subtract(other)
    _count = _bits.count
    _checkInvariants()
  }

  /// Removes the elements of the given sequence of integers from this set.
  ///
  ///     var set: BitSet.Counted = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, -2, -4]
  ///     set.subtract(other)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public mutating func subtract(
    _ other: __owned some Sequence<Int>
  ) {
    _bits.subtract(other)
    _count = _bits.count
    _checkInvariants()
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value indicating whether two bit sets are equal. Two
  /// bit sets are considered equal if they contain the same elements.
  ///
  /// - Complexity: O(*max*), where *max* is value of the largest member of
  ///     either set.
  public func isEqualSet(to other: Self) -> Bool {
    guard self.count == other.count else { return false }
    return self._bits.isEqualSet(to: other._bits)
  }

  /// Returns a Boolean value indicating whether a bit set is equal to a counted
  /// bit set, i.e., whether they contain the same values.
  ///
  /// - Complexity: O(*max*), where *max* is value of the largest member of
  ///     either set.
  public func isEqualSet(to other: BitSet) -> Bool {
    self._bits.isEqualSet(to: other)
  }

  /// Returns a Boolean value indicating whether a bit set is equal to a range
  /// of integers, i.e., whether they contain the same values.
  ///
  /// - Complexity: O(min(*max*, `other.upperBound`), where *max* is the largest
  ///    member of `self`.
  public func isEqualSet(to other: Range<Int>) -> Bool {
    guard self.count == other.count else { return false }
    return _bits.isEqualSet(to: other)
  }

  /// Returns a Boolean value indicating whether this bit set contains the same
  /// elements as the given `other` sequence.
  ///
  /// Duplicate items in `other` do not prevent it from comparing equal to
  /// `self`.
  ///
  ///     let bits: BitSet = [0, 1, 5, 6]
  ///     let other = [5, 5, 0, 1, 1, 6, 5, 0, 1, 6, 6, 5]
  ///
  ///     bits.isEqualSet(to: other) // true
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in `other`.
  @inlinable
  public func isEqualSet(to other: some Sequence<Int>) -> Bool {
    guard self.count >= other.underestimatedCount else { return false }
    return _bits.isEqualSet(to: other)
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     a.isSubset(of: a) // true
  ///     b.isSubset(of: a) // true
  ///     c.isSubset(of: a) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: Self) -> Bool {
    if self.count > other.count { return false }
    return self._bits.isSubset(of: other._bits)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: BitSet) -> Bool {
    self._bits.isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given range of integers.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let b: BitSet.Counted = [0, 1, 2]
  ///     let c: BitSet.Counted = [2, 3, 4]
  ///     b.isSubset(of: -10 ..< 4) // true
  ///     c.isSubset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: Range<Int>) -> Bool {
    _bits.isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the values in a given sequence of integers.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let a = [1, 2, 3, 4, -10]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     b.isSubset(of: a) // true
  ///     c.isSubset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isSubset(of other: some Sequence<Int>) -> Bool {
    _bits.isSubset(of: other)
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     a.isSuperset(of: a) // true
  ///     a.isSuperset(of: b) // true
  ///     a.isSuperset(of: c) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isSuperset(of other: Self) -> Bool {
    other.isSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isSuperset(of other: BitSet) -> Bool {
    other.isSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// a given range of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a: BitSet = [0, 1, 2, 3, 4, 10]
  ///     a.isSuperset(of: 0 ..< 4) // true
  ///     a.isSuperset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(`range.count`)
  public func isSuperset(of other: Range<Int>) -> Bool {
    _bits.isSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the values in a given sequence of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a = [1, 2, 3]
  ///     let b: BitSet = [0, 1, 2, 3, 4]
  ///     let c: BitSet = [0, 1, 2]
  ///     b.isSuperset(of: a) // true
  ///     c.isSuperset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers, some of whose members
  ///    may appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: The same as the complexity of iterating over all elements
  ///    in `other`.
  @inlinable
  public func isSuperset(of other: some Sequence<Int>) -> Bool {
    _bits.isSuperset(of: other)
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     a.isStrictSubset(of: a) // false
  ///     b.isStrictSubset(of: a) // true
  ///     c.isStrictSubset(of: a) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: Self) -> Bool {
    guard self.count < other.count else { return false }
    return _bits.isStrictSubset(of: other._bits)
  }

  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: BitSet) -> Bool {
    _bits.isStrictSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let b: BitSet.Counted = [0, 1, 2]
  ///     let c: BitSet.Counted = [2, 3, 4]
  ///     b.isStrictSubset(of: -10 ..< 4) // true
  ///     c.isStrictSubset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: Range<Int>) -> Bool {
    guard self.count < other.count else { return false }
    return _bits.isStrictSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the values in a given sequence of integers.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let a = [1, 2, 3, 4, -10]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     b.isStrictSubset(of: a) // true
  ///     c.isStrictSubset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isStrictSubset(of other: some Sequence<Int>) -> Bool {
    _bits.isStrictSubset(of: other)
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value that indicates whether this set is a strict
  /// superset of another set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [1, 2, 4]
  ///     let c: BitSet.Counted = [0, 1]
  ///     a.isStrictSuperset(of: a) // false
  ///     a.isStrictSuperset(of: b) // true
  ///     a.isStrictSuperset(of: c) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isStrictSuperset(of other: Self) -> Bool {
    other.isStrictSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether this set is a strict
  /// superset of another set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*.
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isStrictSuperset(of other: BitSet) -> Bool {
    other.isStrictSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// a given range of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a: BitSet.Counted = [0, 1, 2, 3, 4, 10]
  ///     a.isSuperset(of: 0 ..< 4) // true
  ///     a.isSuperset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(`range.count`)
  public func isStrictSuperset(of other: Range<Int>) -> Bool {
    _bits.isStrictSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the values in a given sequence of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a = [1, 2, 3]
  ///     let b: BitSet.Counted = [0, 1, 2, 3, 4]
  ///     let c: BitSet.Counted = [0, 1, 2]
  ///     b.isSuperset(of: a) // true
  ///     c.isSuperset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers, some of whose members
  ///    may appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `other`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isStrictSuperset(of other: some Sequence<Int>) -> Bool {
    _bits.isStrictSuperset(of: other)
  }
}

extension BitSet.Counted {
  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func isDisjoint(with other: Self) -> Bool {
    _bits.isDisjoint(with: other._bits)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: BitSet.Counted = [1, 2, 3, 4]
  ///     let b: BitSet = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func isDisjoint(with other: BitSet) -> Bool {
    _bits.isDisjoint(with: other)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given range of integers.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     a.isDisjoint(with: -10 ..< 0) // true
  ///
  /// - Parameter other: A range of arbitrary integers.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isDisjoint(with other: Range<Int>) -> Bool {
    _bits.isDisjoint(with: other)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given sequence of integers.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [5, 6, -10, 42]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isDisjoint(with other: some Sequence<Int>) -> Bool {
    _bits.isDisjoint(with: other)
  }
}
