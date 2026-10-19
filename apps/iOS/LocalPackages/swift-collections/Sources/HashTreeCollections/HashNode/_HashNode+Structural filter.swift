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
  internal func filter(
    _ level: _HashLevel,
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Builder? {
    guard !isCollisionNode else {
      return try _filter_slow(level, isIncluded)
    }
    return try self.read {
      var result: Builder = .empty(level)
      var removing = false // true if we need to remove something

      for (bucket, slot) in $0.itemMap {
        let p = $0.itemPtr(at: slot)
        let include = try isIncluded(p.pointee)
        switch (include, removing) {
        case (true, true):
          result.addNewItem(level, p.pointee, at: bucket)
        case (false, false):
          removing = true
          result.copyItems(level, from: $0, upTo: bucket)
        default:
          break
        }
      }

      for (bucket, slot) in $0.childMap {
        let branch = try $0[child: slot].filter(level.descend(), isIncluded)
        if let branch = branch {
          assert(branch.count < self.count)
          if !removing {
            removing = true
            result.copyItemsAndChildren(level, from: $0, upTo: bucket)
          }
          result.addNewChildBranch(level, branch, at: bucket)
        } else if removing {
          result.addNewChildNode(level, $0[child: slot], at: bucket)
        }
      }

      guard removing else { return nil }
      return result
    }
  }

  @inlinable @inline(never)
  internal func _filter_slow(
    _ level: _HashLevel,
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Builder? {
    try self.read {
      var result: Builder = .empty(level)
      var removing = false

      for slot: _HashSlot in stride(from: .zero, to: $0.itemsEndSlot, by: 1) {
        let p = $0.itemPtr(at: slot)
        let include = try isIncluded(p.pointee)
        if include, removing {
          result.addNewCollision(level, p.pointee, $0.collisionHash)
        }
        else if !include, !removing {
          removing = true
          result.copyCollisions(from: $0, upTo: slot)
        }
      }
      guard removing else { return nil }
      assert(result.count < self.count)
      return result
    }
  }
}
