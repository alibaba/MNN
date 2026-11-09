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

extension OrderedSet {
  /// Returns the collection difference between the parameter and the
  /// receiver, using an algorithm specialized to exploit fast membership
  /// testing and the member uniqueness guarantees of `OrderedSet`.
  ///
  /// - Complexity: O(`self.count + other.count`)
  @available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *)
  public func difference(
    from other: Self
  ) -> CollectionDifference<Element> {
    /* While admitting that variables with names like "a", "b", "x", and "y" are not especially readable, their use (and meaning) is standard in the diffing literature and familiarity with that literature will help if you're reading this code anyway. */
    let a = other // The collection we're diffing "from". An element present in this collection but not the other is a "remove"
    var x = 0 // The working index into `a`
    let b = self // The collection we're diffing "to". An element present in this collection but not the other is an "insert"
    var y = 0 // The working index into `b`

    var changes = Array<CollectionDifference<Element>.Change>()
    func remove() {
      let ax = a[x]
      changes.append(.remove(offset: x, element: ax, associatedWith: b.lastIndex(of: ax)))
      x += 1
    }
    func insert() {
      let by = b[y]
      changes.append(.insert(offset: y, element: by, associatedWith: a.lastIndex(of: by)))
      y += 1
    }

    while x < a.count || y < b.count {
      if y == b.count {
        // No more elements to process in `b`, `a[x]` must have been removed
        remove()
      } else if x == a.count {
        // No more elements to process in `a`, `b[y]` must have been inserted
        insert()
      } else if let axinb = b.lastIndex(of: a[x]) {
        if axinb < y {
          // Element has already been processed as an insertion in `b`, generate associated remove for move
          remove()
        }
        else if let byina = a.lastIndex(of: b[y]) {
          if byina < x {
            // Element has already been processed as a remove in `a`, generate associated insert for move
            insert()
          } else if x == byina {
            assert(y == axinb)
            // `a[x]` == `b[y]`
            x += 1; y += 1
          } else if byina - x < axinb - y {
            // `a[x]` exists further away from the current position in `b` than `b[y] does in `a`
            insert()
          } else {
            // `b[y]` exists further away from the current position in `a` than `a[x]` does in `b`
            remove()
          }
        } else {
          // `b[y]` does not exist in `a`, the element must have been inserted
          insert()
        }
      } else {
        // `a[x]` does not exist in `b`, the element must have been removed
        remove()
      }
    }
    return CollectionDifference(changes)!
  }

  /* Replicating RangeReplaceableCollection.applying here would involve
   * duplicating a ton of IPI from the stdlib. Dropping down to Array and back
   * to OrderedSet is still O(n)-ish which still beats the naïve application
   * algorithm of "remove, then insert"
   */
  /// Applies the given difference to this collection.
  ///
  /// - Parameter difference: The difference to be applied.
  ///
  /// - Returns: An instance representing the state of the receiver with the
  ///   difference applied, or `nil` if the difference is incompatible with
  ///   the receiver's state.
  ///
  /// - Complexity: O(*n* + *c*), where *n* is `self.count` and *c*
  ///   is the number of changes contained by the parameter.
  @available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *)
  public func applying(_ difference: CollectionDifference<Element>) -> Self? {
    guard let array = self.elements.applying(difference) else { return nil }
    let result = OrderedSet(array)
    return result.count == array.count ? result : nil
  }
}
