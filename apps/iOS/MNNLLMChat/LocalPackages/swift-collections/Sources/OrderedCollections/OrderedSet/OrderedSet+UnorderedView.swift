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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension OrderedSet {
  /// An unordered view into an ordered set, providing `SetAlgebra`
  /// conformance.
  @frozen
  public struct UnorderedView {
    @usableFromInline
    internal var _base: OrderedSet

    @inlinable
    @inline(__always)
    internal init(_base: OrderedSet) {
      self._base = _base
    }
  }

  /// Create a new ordered set with the same members as the supplied
  /// unordered view.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public init(_ view: UnorderedView) {
    self = view._base
  }

  /// Access a view of the members of this set as an unordered
  /// `SetAlgebra` value.
  ///
  /// This is useful when you need to pass an ordered set to a
  /// function that is generic over `SetAlgebra`.
  ///
  /// The unordered view has a definition of equality that ignores the
  /// order of members, so that it can satisfy `SetAlgebra`
  /// requirements. New elements inserted to the unordered view get
  /// appended to the end of the set.
  ///
  /// - Complexity: O(1) for both the getter and the setter.
  @inlinable
  public var unordered: UnorderedView {
    @inline(__always)
    get {
      UnorderedView(_base: self)
    }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      var view = UnorderedView(_base: self)
      self = OrderedSet()
      defer { self = view._base }
      yield &view
    }
  }
}

extension OrderedSet.UnorderedView: Sendable where Element: Sendable {}

extension OrderedSet.UnorderedView: CustomStringConvertible {
  /// A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: _base)
  }
}

extension OrderedSet.UnorderedView: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension OrderedSet.UnorderedView: CustomReflectable {
  /// The custom mirror for this instance.
  public var customMirror: Mirror {
    Mirror(self, unlabeledChildren: _base._elements, displayStyle: .collection)
  }
}

extension OrderedSet.UnorderedView: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  /// Two unordered sets are considered equal if they contain the same
  /// elements, but not necessarily in the same order.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    if left._base.__storage != nil,
       left._base.__storage === right._base.__storage
    {
      return true
    }
    guard left._base.count == right._base.count else { return false }

    for item in left._base {
      if !right._base.contains(item) { return false }
    }
    return true
  }
}

extension OrderedSet.UnorderedView: Hashable {
  /// Hashes the essential components of this value by feeding them into the
  /// given hasher.
  ///
  /// Complexity: O(`count`)
  @inlinable
  public func hash(into hasher: inout Hasher) {
    // Generate a seed from a snapshot of the hasher.  This makes members' hash
    // values depend on the state of the hasher, which improves hashing
    // quality. (E.g., it makes it possible to resolve collisions by passing in
    // a different hasher.)
    let copy = hasher
    let seed = copy.finalize()

    var hash = 0
    for member in _base {
      hash ^= member._rawHashValue(seed: seed)
    }
    hasher.combine(hash)
  }
}

extension OrderedSet.UnorderedView: ExpressibleByArrayLiteral {
  /// Creates a new unordered set from the contents of an array literal.
  @inlinable
  @inline(__always)
  public init(arrayLiteral elements: Element...) {
    _base = OrderedSet(elements)
  }
}

extension OrderedSet.UnorderedView: SetAlgebra {
  public typealias Element = OrderedSet.Element
}

extension OrderedSet.UnorderedView {
  /// Creates an empty set.
  ///
  /// This initializer is equivalent to initializing with an empty array
  /// literal.
  @inlinable
  @inline(__always)
  public init() {
    _base = OrderedSet()
  }

  /// Creates a new set from a finite sequence of items.
  ///
  /// - Parameter elements: The elements to use as members of the new set.
  ///
  /// - Complexity: This operation is expected to perform O(*n*)
  ///    comparisons on average (where *n* is the number of elements
  ///    in the sequence), provided that `Element` implements
  ///    high-quality hashing.
  @inlinable
  @inline(__always)
  public init(_ elements: some Sequence<Element>) {
    _base = OrderedSet(elements)
  }

  // Specializations

  /// Creates a new set from a an existing set. This is functionally the same as
  /// copying the value of `elements` into a new variable.
  ///
  /// - Parameter elements: The elements to use as members of the new set.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public init(_ elements: Self) {
    self = elements
  }

  /// Creates a new set from an existing `Set` value.
  ///
  /// - Parameter elements: The elements to use as members of the new set.
  ///
  /// - Complexity: This operation is expected to perform O(*n*)
  ///    comparisons on average (where *n* is the number of elements
  ///    in the set), provided that `Element` implements high-quality
  ///    hashing.
  @inlinable
  @inline(__always)
  public init(_ elements: Set<Element>) {
    self._base = OrderedSet(elements)
  }

  /// Creates a new set from the keys of a dictionary value.
  ///
  /// - Parameter elements: The elements to use as members of the new set.
  ///
  /// - Complexity: This operation is expected to perform O(*n*)
  ///    comparisons on average (where *n* is the number of elements
  ///    in the set), provided that `Element` implements high-quality
  ///    hashing.
  @inlinable
  @inline(__always)
  public init<Value>(_ elements: Dictionary<Element, Value>.Keys) {
    self._base = OrderedSet(elements)
  }
}

extension OrderedSet.UnorderedView {
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
  @inline(__always)
  public func contains(_ element: Element) -> Bool {
    _base.contains(element)
  }
}

extension OrderedSet.UnorderedView {
  /// Inserts the given element in the set if it is not already present.
  ///
  /// If an element equal to `newMember` is already contained in the set, this
  /// method has no effect.
  ///
  /// If `newMember` was not already a member, it gets appended to the end of
  /// the underlying ordered set value.
  ///
  /// - Parameter newMember: An element to insert into the set.
  ///
  /// - Returns: `(true, newMember)` if `newMember` was not contained in the
  ///    set. If an element equal to `newMember` was already contained in the
  ///    set, the method returns `(false, oldMember)`, where `oldMember` is the
  ///    element that was equal to `newMember`. In some cases, `oldMember` may
  ///    be distinguishable from `newMember` by identity comparison or some
  ///    other means.
  ///
  /// - Complexity: This operation is expected to perform O(1)
  ///    hashing/comparison operations on average (over many insertions to the
  ///    same set), provided that `Element` implements high-quality hashing.
  @inlinable
  public mutating func insert(
    _ newMember: __owned Element
  ) -> (inserted: Bool, memberAfterInsert: Element) {
    let (inserted, index) = _base.append(newMember)
    return (inserted, _base[index])
  }

  /// Inserts the given element into the set unconditionally.
  ///
  /// If an element equal to `newMember` is already contained in the set,
  /// `newMember` replaces the existing element.
  ///
  /// If `newMember` was not already a member, it gets appended to the end of
  /// the underlying ordered set value.
  ///
  /// - Parameter newMember: An element to insert into the set.
  ///
  /// - Returns: The original member equal to `newMember` if the set already
  ///    contained such a member; otherwise, `nil`. In some cases, the returned
  ///    element may be distinguishable from `newMember` by identity comparison
  ///    or some other means.
  ///
  /// - Complexity: This operation is expected to perform O(1)
  ///    hashing/comparison operations on average (over many insertions to the
  ///    same set), provided that `Element` implements high-quality hashing.
  @inlinable
  public mutating func update(with newMember: __owned Element) -> Element? {
    let (inserted, index) = _base.append(newMember)
    if inserted { return nil }
    let old = _base._elements[index]
    _base._elements[index] = newMember
    return old
  }
}

extension OrderedSet.UnorderedView {
  /// Removes the given element from the set.
  ///
  /// - Parameter member: The element of the set to remove.
  ///
  /// - Returns: The element equal to `member` if `member` is contained in the
  ///    set; otherwise, `nil`. In some cases, the returned element may be
  ///    distinguishable from `newMember` by identity comparison or some other
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
  @inline(__always)
  @discardableResult
  public mutating func remove(_ member: Element) -> Element? {
    _base.remove(member)
  }
}

extension OrderedSet.UnorderedView {
  /// Adds the elements of the given set to this set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the set, in the order they appear in `other`.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [0, 2, 4, 6]
  ///     set.formUnion(other)
  ///     // `set` is now `[1, 2, 3, 4, 0, 6]`
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func formUnion(_ other: __owned Self) {
    _base.formUnion(other._base)
  }

  /// Returns a new set with the elements of both this and the given set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the result, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [0, 2, 4, 6]
  ///     a.union(b) // [1, 2, 3, 4, 0, 6]
  ///
  /// - Parameter other: The set of elements to add.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func union(_ other: __owned Self) -> Self {
    _base.union(other._base).unordered
  }

  // Generalizations

  /// Adds the elements of the given sequence to this set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the set, in the order they appear in `other`.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.formUnion([0, 2, 4, 6])
  ///     // `set` is now `[1, 2, 3, 4, 0, 6]`
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public mutating func formUnion(_ other: __owned some Sequence<Element>) {
    _base.formUnion(other)
  }

  /// Returns a new set with the elements of both this and the given set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the result, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     a.union([0, 2, 4, 6]) // [1, 2, 3, 4, 0, 6]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func union(
    _ other: __owned some Sequence<Element>
  ) -> Self {
    _base.union(other).unordered
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a new set with the elements that are common to both this set and
  /// the provided other one, in the order they appear in `self`.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.intersection(other) // [2, 4]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public __consuming func intersection(_ other: Self) -> Self {
    _base.intersection(other._base).unordered
  }

  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A set of elements.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public mutating func formIntersection(_ other: Self) {
    _base.formIntersection(other._base)
  }

  // Generalizations

  /// Returns a new set with the elements that are common to both this set and
  /// the provided sequence, in the order they appear in `self`.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.intersection([6, 4, 2, 0] as Array) // [2, 4]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(*n*) on average where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func intersection(
    _ other: some Sequence<Element>
  ) -> Self {
    _base.intersection(other).unordered
  }

  /// Removes the elements of this set that aren't also in the given sequence.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.formIntersection([6, 4, 2, 0] as Array)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(*n*) on average where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public mutating func formIntersection(
    _ other: some Sequence<Element>
  ) {
    _base.formIntersection(other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  /// The result contains elements from `self` followed by elements in `other`,
  /// in the same order they appeared in the original sets.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.symmetricDifference(other) // [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public __consuming func symmetricDifference(_ other: __owned Self) -> Self {
    _base.symmetricDifference(other._base).unordered
  }

  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  /// On return, `self` contains elements originally from `self` followed by
  /// elements in `other`, in the same order they appeared in the input values.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public mutating func formSymmetricDifference(_ other: __owned Self) {
    _base.formSymmetricDifference(other._base)
  }

  // Generalizations

  /// Returns a new set with the elements that are either in this set or in the
  /// given sequence, but not in both.
  ///
  /// The result contains elements from `self` followed by elements in `other`,
  /// in the same order they first appeared in the input values.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.symmetricDifference([6, 4, 2, 0] as Array) // [1, 3, 6, 0]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average where *n* is
  ///    the number of elements in `other`, if `Element` implements high-quality
  ///    hashing.
  @inlinable
  public __consuming func symmetricDifference(
    _ other: __owned some Sequence<Element>
  ) -> Self {
    _base.symmetricDifference(other).unordered
  }

  /// Replace this set with the elements contained in this set or the given
  /// sequence, but not both.
  ///
  /// On return, `self` contains elements originally from `self` followed by
  /// elements in `other`, in the same order they first appeared in the input
  /// values.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.formSymmetricDifference([6, 4, 2, 0] as Array)
  ///     // set is now [1, 3, 6, 0]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average where *n* is
  ///    the number of elements in `other`, if `Element` implements high-quality
  ///    hashing.
  @inlinable
  public mutating func formSymmetricDifference(
    _ other: __owned some Sequence<Element>
  ) {
    _base.formSymmetricDifference(other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a new set containing the elements of this set that do not occur
  /// in the given set.
  ///
  /// The result contains elements in the same order they appear in `self`.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public __consuming func subtracting(_ other: Self) -> Self {
    _base.subtracting(other._base).unordered
  }

  /// Removes the elements of the given set from this set.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let other: OrderedSet<Int>.UnorderedView = [6, 4, 2, 0]
  ///     set.subtract(other)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public mutating func subtract(_ other: Self) {
    _base.subtract(other._base)
  }

  // Generalizations

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given sequence.
  ///
  /// The result contains elements in the same order they appear in `self`.
  ///
  ///     let set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.subtracting([6, 4, 2, 0] as Array) // [1, 3]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public __consuming func subtracting(_ other: some Sequence<Element>) -> Self {
    _base.subtracting(other).unordered
  }

  /// Removes the elements of the given sequence from this set.
  ///
  ///     var set: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     set.subtract([6, 4, 2, 0] as Array)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public mutating func subtract(_ other: some Sequence<Element>) {
    _base.subtract(other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value indicating whether two set values contain the
  /// same elements, but not necessarily in the same order.
  ///
  /// - Note: This member implements different behavior than the `==(_:_:)`
  ///    operator -- the latter implements an ordered comparison, matching
  ///    the stricter concept of equality expected of an ordered collection
  ///    type.
  ///
  /// - Complexity: O(`min(left.count, right.count)`), as long as`Element`
  ///    properly implements hashing.
  public func isEqualSet(to other: Self) -> Bool {
    self == other
  }

  /// Returns a Boolean value indicating whether an ordered set contains the
  /// same values as a given sequence, but not necessarily in the same
  /// order.
  ///
  /// Duplicate items in `other` do not prevent it from comparing equal to
  /// `self`.
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in
  ///    `other`, as long as`Element` properly implements hashing.
  public func isEqualSet(to other: some Sequence<Element>) -> Bool {
    self._base.isEqualSet(to: other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSubset(of other: Self) -> Bool {
    _base.isSubset(of: other._base)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: Set<Int> = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSubset(of other: Set<Element>) -> Bool {
    _base.isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the elements in the given sequence.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: Array = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: A finite sequence.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isSubset(of other: some Sequence<Element>) -> Bool {
    _base.isSubset(of: other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: Self) -> Bool {
    _base.isSuperset(of: other._base)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Set<Int> = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: Set<Element>) -> Bool {
    _base.isSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given sequence.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Array = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(*n*) on average, where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: some Sequence<Element>) -> Bool {
    _base.isSuperset(of: other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSubset(of other: Self) -> Bool {
    _base.isStrictSubset(of: other._base)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: Set<Int> = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSubset(of other: Set<Element>) -> Bool {
    _base.isStrictSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given sequence.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: Array = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isStrictSubset(of other: some Sequence<Element>) -> Bool {
    _base.isStrictSubset(of: other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSuperset(of other: Self) -> Bool {
    _base.isStrictSuperset(of: other._base)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Set = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSuperset(of other: Set<Element>) -> Bool {
    _base.isStrictSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given sequence.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Array = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isStrictSuperset(of other: some Sequence<Element>) -> Bool {
    _base.isStrictSuperset(of: other)
  }
}

extension OrderedSet.UnorderedView {
  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: OrderedSet<Int>.UnorderedView = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(min(`self.count`, `other.count`)) on
  ///    average, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(with other: Self) -> Bool {
    _base.isDisjoint(with: other._base)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Set = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(min(`self.count`, `other.count`)) on
  ///    average, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(with other: Set<Element>) -> Bool {
    _base.isDisjoint(with: other)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given sequence.
  ///
  ///     let a: OrderedSet<Int>.UnorderedView = [1, 2, 3, 4]
  ///     let b: Array = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(*n*) on average, where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(with other: some Sequence<Element>) -> Bool {
    _base.isDisjoint(with: other)
  }
}
