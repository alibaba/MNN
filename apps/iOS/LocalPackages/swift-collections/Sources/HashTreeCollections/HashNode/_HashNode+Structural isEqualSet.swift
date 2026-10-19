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

// TODO: `Equatable` needs more test coverage, apart from hash-collision smoke test
extension _HashNode {
  @inlinable
  internal func isEqualSet<Value2>(
    to other: _HashNode<Key, Value2>,
    by areEquivalent: (Value, Value2) -> Bool
  ) -> Bool {
    if self.raw.storage === other.raw.storage { return true }

    guard self.count == other.count else { return false }

    if self.isCollisionNode {
      guard other.isCollisionNode else { return false }
      return self.read { lhs in
        other.read { rhs in
          guard lhs.collisionHash == rhs.collisionHash else { return false }
          let l = lhs.reverseItems
          let r = rhs.reverseItems
          assert(l.count == r.count) // Already checked above
          for i in l.indices {
            let found = r.contains {
              l[i].key == $0.key && areEquivalent(l[i].value, $0.value)
            }
            guard found else { return false }
          }
          return true
        }
      }
    }
    guard !other.isCollisionNode else { return false }

    return self.read { l in
      other.read { r in
        guard l.itemMap == r.itemMap else { return false }
        guard l.childMap == r.childMap else { return false }

        guard l.reverseItems.elementsEqual(
          r.reverseItems,
          by: { $0.key == $1.key && areEquivalent($0.value, $1.value) })
        else { return false }

        let lc = l.children
        let rc = r.children
        return lc.elementsEqual(
          rc,
          by: { $0.isEqualSet(to: $1, by: areEquivalent) })
      }
    }
  }
}
