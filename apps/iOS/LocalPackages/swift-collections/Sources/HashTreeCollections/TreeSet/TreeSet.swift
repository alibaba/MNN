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

/// An unordered collection of unique elements, optimized for mutating shared
/// copies and comparing different snapshots of the same collection.
///
/// `TreeSet` has the same functionality as a standard `Set`, and
/// it largely implements the same APIs: both are hashed collection
/// types that conform to `SetAlgebra`, and both are unordered -- neither type
/// provides any guarantees about the ordering of their members.
///
/// However, `TreeSet` is optimizing specifically for use cases that
/// need to mutate shared copies or to compare a set value to one of its older
/// snapshots. To use a term from functional programming,
/// `TreeSet` implements a _persistent data structure_.
///
/// The standard `Set` stores its members in a single, flat hash table, and it
/// implements value semantics with all-or-nothing copy-on-write behavior: every
/// time a shared copy of a set is mutated, the mutation needs to make a full
/// copy of the set's storage. `TreeSet` takes a different approach: it
/// organizes its members into a tree structure, the nodes of which can be
/// freely shared across collection values. When mutating a shared copy of a set
/// value, `TreeSet` is able to simply link the unchanged parts of the
/// tree directly into the result, saving both time and memory.
///
/// This structural sharing also makes it more efficient to compare mutated set
/// values to earlier versions of themselves. When comparing or combining sets,
/// parts that are shared across both inputs can typically be handled in
/// constant time, leading to a dramatic performance boost when the two inputs
/// are still largely unchanged:
///
///     var set = TreeSet(0 ..< 10_000)
///     let copy = set
///     set.insert(20_000) // Expected to be an O(log(n)) operation
///     let diff = set.subtracting(copy) // Also O(log(n))!
///     // `diff` now holds the single item 20_000.
///
/// The tree structure also eliminates the need to reserve capacity in advance:
/// `TreeSet` creates, destroys and resizes individual nodes as needed,
/// always consuming just enough memory to store its contents. As of Swift 5.9,
/// the standard collection types never shrink their storage, so temporary
/// storage spikes can linger as unused but still allocated memory long after
/// the collection has shrunk back to its usual size.
///
/// Of course, switching to a tree structure comes with some trade offs. In
/// particular, inserting new items, removing existing ones, and iterating over
/// a `TreeSet` is expected to be a constant factor slower than a standard
/// `Set` -- allocating/deallocating nodes isn't free, and navigating the tree
/// structure requires more pointer dereferences than accessing a flat hash
/// table. However the algorithmic improvements above usually more than make up
/// for this, as long as the use case can make use of them.
@frozen // Not really -- this package is not at all ABI stable
public struct TreeSet<Element: Hashable> {
  @usableFromInline
  internal typealias _Node = _HashNode<Element, Void>

  @usableFromInline
  internal typealias _UnsafeHandle = _Node.UnsafeHandle

  @usableFromInline
  internal var _root: _Node

  @usableFromInline
  internal var _version: UInt

  @inlinable
  internal init(_root: _Node, version: UInt) {
    self._root = _root
    self._version = version
  }

  @inlinable
  internal init(_new: _Node) {
    self.init(_root: _new, version: _new.initialVersionNumber)
  }
}
