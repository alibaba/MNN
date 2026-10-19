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

@_fixed_layout
@usableFromInline
internal final class _DequeBuffer<Element>: ManagedBuffer<_DequeBufferHeader, Element> {
  @inlinable
  deinit {
    self.withUnsafeMutablePointers { header, elements in
      header.pointee._checkInvariants()

      let capacity = header.pointee.capacity
      let count = header.pointee.count
      let startSlot = header.pointee.startSlot

      if startSlot.position + count <= capacity {
        (elements + startSlot.position).deinitialize(count: count)
      } else {
        let firstRegion = capacity - startSlot.position
        (elements + startSlot.position).deinitialize(count: firstRegion)
        elements.deinitialize(count: count - firstRegion)
      }
    }
  }
}

extension _DequeBuffer: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    withUnsafeMutablePointerToHeader { "_DequeStorage<\(Element.self)>\($0.pointee)" }
  }
}

/// The type-punned empty singleton storage instance.
@usableFromInline
nonisolated(unsafe) internal let _emptyDequeStorage
  = _DequeBuffer<Void>.create(
    minimumCapacity: 0,
    makingHeaderWith: { _ in
      _DequeBufferHeader(capacity: 0, count: 0, startSlot: .init(at: 0))
    })
