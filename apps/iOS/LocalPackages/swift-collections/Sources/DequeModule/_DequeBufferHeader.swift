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

@usableFromInline
internal struct _DequeBufferHeader {
  @usableFromInline
  var capacity: Int

  @usableFromInline
  var count: Int

  @usableFromInline
  var startSlot: _DequeSlot

  @usableFromInline
  init(capacity: Int, count: Int, startSlot: _DequeSlot) {
    self.capacity = capacity
    self.count = count
    self.startSlot = startSlot
    _checkInvariants()
  }

  #if COLLECTIONS_INTERNAL_CHECKS
  @usableFromInline @inline(never) @_effects(releasenone)
  internal func _checkInvariants() {
    precondition(capacity >= 0)
    precondition(count >= 0 && count <= capacity)
    precondition(startSlot.position >= 0 && startSlot.position <= capacity)
  }
  #else
  @inlinable @inline(__always)
  internal func _checkInvariants() {}
  #endif // COLLECTIONS_INTERNAL_CHECKS
}

extension _DequeBufferHeader: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    "(capacity: \(capacity), count: \(count), startSlot: \(startSlot))"
  }
}
