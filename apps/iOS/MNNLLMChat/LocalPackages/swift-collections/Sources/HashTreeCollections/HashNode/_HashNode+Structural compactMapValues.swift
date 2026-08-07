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

extension _HashNode {
  @inlinable
  internal func compactMapValues<T>(
    _ level: _HashLevel,
    _ transform: (Value) throws -> T?
  ) rethrows -> _HashNode<Key, T>.Builder {
    return try self.read {
      var result: _HashNode<Key, T>.Builder = .empty(level)

      if isCollisionNode {
        let items = $0.reverseItems
        for i in items.indices {
          if let v = try transform(items[i].value) {
            result.addNewCollision(level, (items[i].key, v), $0.collisionHash)
          }
        }
        return result
      }

      for (bucket, slot) in $0.itemMap {
        let p = $0.itemPtr(at: slot)
        if let v = try transform(p.pointee.value) {
          result.addNewItem(level, (p.pointee.key, v), at: bucket)
        }
      }

      for (bucket, slot) in $0.childMap {
        let branch = try $0[child: slot]
          .compactMapValues(level.descend(), transform)
        result.addNewChildBranch(level, branch, at: bucket)
      }
      return result
    }
  }
}
