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
  @usableFromInline
  internal func dump(
    iterationOrder: Bool = false,
    limit: Int = Int.max,
    firstPrefix: String = "",
    restPrefix: String = "",
    depth: Int = 0
  ) {
    read {
      $0.dump(
        iterationOrder: iterationOrder,
        limit: limit,
        extra: "count: \(count), ",
        firstPrefix: firstPrefix,
        restPrefix: restPrefix,
        depth: depth)
    }
  }
}

extension _HashNode.Storage {
  @usableFromInline
  final internal func dump(iterationOrder: Bool = false) {
    UnsafeHandle.read(self) { $0.dump(iterationOrder: iterationOrder) }
  }
}

extension _HashNode {
  internal static func _itemString(for item: Element) -> String {
    let hash = _Hash(item.key).description
    return "hash: \(hash), key: \(item.key), value: \(item.value)"
  }
}
extension _HashNode.UnsafeHandle {
  internal func _itemString(at slot: _HashSlot) -> String {
    let item = self[item: slot]
    return _HashNode._itemString(for: item)
  }

  @usableFromInline
  internal func dump(
    iterationOrder: Bool = false,
    limit: Int = .max,
    extra: String = "",
    firstPrefix: String = "",
    restPrefix: String = "",
    depth: Int = 0
  ) {
    var firstPrefix = firstPrefix
    var restPrefix = restPrefix
    if iterationOrder && depth == 0 {
      firstPrefix += "@"
      restPrefix += "@"
    }
    if iterationOrder {
      firstPrefix += "  "
    }
    print("""
      \(firstPrefix)\(isCollisionNode ? "CollisionNode" : "Node")(\
      at: \(_addressString(for: _header)), \
      \(isCollisionNode ? "hash: \(collisionHash), " : "")\
      \(extra)\
      byteCapacity: \(byteCapacity), \
      freeBytes: \(bytesFree))
      """)
    guard limit > 0 else { return }
    if iterationOrder {
      for slot in stride(from: .zero, to: itemsEndSlot, by: 1) {
        print("  \(restPrefix)[\(slot)]  \(_itemString(at: slot))")
      }
      for slot in stride(from: .zero, to: childrenEndSlot, by: 1) {
        self[child: slot].dump(
          iterationOrder: true,
          limit: limit - 1,
          firstPrefix: "  \(restPrefix).\(slot)",
          restPrefix: "  \(restPrefix).\(slot)",
          depth: depth + 1)
      }
    }
    else if isCollisionNode {
      for slot in stride(from: .zero, to: itemsEndSlot, by: 1) {
        print("\(restPrefix)[\(slot)] \(_itemString(at: slot))")
      }
    } else {
      var itemSlot: _HashSlot = .zero
      var childSlot: _HashSlot = .zero
      for b in 0 ..< UInt(_Bitmap.capacity) {
        let bucket = _Bucket(b)
        let bucketStr = "#\(String(b, radix: _Bitmap.capacity, uppercase: true))"
        if itemMap.contains(bucket) {
          print("\(restPrefix)  \(bucketStr) \(_itemString(at: itemSlot))")
          itemSlot = itemSlot.next()
        } else if childMap.contains(bucket) {
          self[child: childSlot].dump(
            iterationOrder: false,
            limit: limit - 1,
            firstPrefix: "\(restPrefix)  \(bucketStr) ",
            restPrefix: "\(restPrefix)     ",
            depth: depth + 1)
          childSlot = childSlot.next()
        }
      }
    }
  }
}
