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

extension _HashNode {
  @inlinable
  internal func mapValues<T>(
    _ transform: (Element) throws -> T
  ) rethrows -> _HashNode<Key, T> {
    let c = self.count
    return try read { source in
      var result: _HashNode<Key, T>
      if isCollisionNode {
        result = _HashNode<Key, T>.allocateCollision(
          count: c, source.collisionHash,
          initializingWith: { _ in }
        ).node
      } else {
        result = _HashNode<Key, T>.allocate(
          itemMap: source.itemMap,
          childMap: source.childMap,
          count: c,
          initializingWith: { _, _ in }
        ).node
      }
      try result.update { target in
        let sourceItems = source.reverseItems
        let targetItems = target.reverseItems
        assert(sourceItems.count == targetItems.count)

        let sourceChildren = source.children
        let targetChildren = target.children
        assert(sourceChildren.count == targetChildren.count)

        var i = 0
        var j = 0

        var success = false

        defer {
          if !success {
            targetItems.prefix(i).deinitialize()
            targetChildren.prefix(j).deinitialize()
            target.clear()
          }
        }

        while i < targetItems.count {
          let key = sourceItems[i].key
          let value = try transform(sourceItems[i])
          targetItems.initializeElement(at: i, to: (key, value))
          i += 1
        }
        while j < targetChildren.count {
          let child = try sourceChildren[j].mapValues(transform)
          targetChildren.initializeElement(at: j, to: child)
          j += 1
        }
        success = true
      }
      result._invariantCheck()
      return result
    }
  }

  @inlinable
  internal func mapValuesToVoid(
    copy: Bool = false, extraBytes: Int = 0
  ) -> _HashNode<Key, Void> {
    if Value.self == Void.self {
      let node = unsafeBitCast(self, to: _HashNode<Key, Void>.self)
      guard copy || !node.hasFreeSpace(extraBytes) else { return node }
      return node.copy(withFreeSpace: extraBytes)
    }
    let node = mapValues { _ in () }
    guard !node.hasFreeSpace(extraBytes) else { return node }
    return node.copy(withFreeSpace: extraBytes)
  }
}
