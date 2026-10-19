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

extension _HashNodeHeader {
#if COLLECTIONS_INTERNAL_CHECKS
  @usableFromInline @inline(never)
  internal func _invariantCheck() {
    precondition(bytesFree <= byteCapacity)
    if isCollisionNode {
      precondition(itemMap == childMap)
      precondition(!itemMap.isEmpty)
    } else {
      precondition(itemMap.intersection(childMap).isEmpty)
    }
  }
#else
  @inlinable @inline(__always)
  internal func _invariantCheck() {}
#endif
}

extension _HashNode {
#if COLLECTIONS_INTERNAL_CHECKS
  @usableFromInline @inline(never)
  internal func _invariantCheck() {
    raw.storage.header._invariantCheck()
    read {
      let itemBytes = $0.itemCount * MemoryLayout<Element>.stride

      if $0.isCollisionNode {
        let hashBytes = MemoryLayout<_Hash>.stride
        assert($0.itemCount >= 1)
        assert($0.childCount == 0)
        assert(itemBytes + $0.bytesFree + hashBytes == $0.byteCapacity)

        assert($0.collisionHash == _Hash($0[item: .zero].key))
      } else {
        let childBytes = $0.childCount * MemoryLayout<_HashNode>.stride
        assert(itemBytes + $0.bytesFree + childBytes == $0.byteCapacity)
      }

      let actualCount = $0.children.reduce($0.itemCount, { $0 + $1.count })
      assert(actualCount == self.count)
    }
  }
#else
  @inlinable @inline(__always)
  internal func _invariantCheck() {}
#endif

  @inlinable @inline(__always)
  public func _fullInvariantCheck() {
    self._fullInvariantCheck(.top, .emptyPath)
  }

#if COLLECTIONS_INTERNAL_CHECKS
  @inlinable @inline(never)
  internal func _fullInvariantCheck(_ level: _HashLevel, _ path: _Hash) {
    _invariantCheck()
    read {
      precondition(level.isAtRoot || !hasSingletonItem)
      precondition(!isAtrophied)
      if $0.isCollisionNode {
        precondition(count == $0.itemCount)
        precondition(count > 0)
        let hash = $0.collisionHash
        precondition(
          hash.isEqual(to: path, upTo: level),
          "Misplaced collision node: \(path) isn't a prefix of \(hash)")
        for item in $0.reverseItems {
          precondition(_Hash(item.key) == hash)
        }
      }
      var itemSlot: _HashSlot = .zero
      var childSlot: _HashSlot = .zero
      for b in 0 ..< UInt(_Bitmap.capacity) {
        let bucket = _Bucket(b)
        let path = path.appending(bucket, at: level)
        if $0.itemMap.contains(bucket) {
          let key = $0[item: itemSlot].key
          let hash = _Hash(key)
          precondition(
            hash.isEqual(to: path, upTo: level.descend()),
            "Misplaced key '\(key)': \(path) isn't a prefix of \(hash)")
          itemSlot = itemSlot.next()
        }
        if $0.hasChildren && $0.childMap.contains(bucket) {
          $0[child: childSlot]._fullInvariantCheck(level.descend(), path)
          childSlot = childSlot.next()
        }
      }
    }
  }
#else
  @inlinable @inline(__always)
  internal func _fullInvariantCheck(_ level: _HashLevel, _ path: _Hash) {}
#endif
}

