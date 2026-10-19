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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

@usableFromInline
internal typealias _Word = _UnsafeBitSet._Word

extension Array where Element == _Word {
  internal func _encodeAsUInt64(
    to container: inout UnkeyedEncodingContainer
  ) throws {
    if _Word.capacity == 64 {
      for word in self {
        try container.encode(UInt64(truncatingIfNeeded: word.value))
      }
      return
    }
    assert(_Word.capacity == 32, "Unsupported platform")
    var first = true
    var w: UInt64 = 0
    for word in self {
      if first {
        w = UInt64(truncatingIfNeeded: word.value)
        first = false
      } else {
        w |= UInt64(truncatingIfNeeded: word.value) &<< 32
        try container.encode(w)
        first = true
      }
    }
    if !first {
      try container.encode(w)
    }
  }

  internal init(
    _fromUInt64 container: inout UnkeyedDecodingContainer,
    reservingCount count: Int? = nil
  ) throws {
    self = []
    if Element.capacity == 64 {
      if let c = count {
        self.reserveCapacity(c)
      }
      while !container.isAtEnd {
        let v = try container.decode(UInt64.self)
        self.append(Element(UInt(truncatingIfNeeded: v)))
      }
      return
    }
    assert(Element.capacity == 32, "Unsupported platform")
    if let c = count {
      self.reserveCapacity(2 * c)
    }
    while !container.isAtEnd {
      let v = try container.decode(UInt64.self)
      self.append(Element(UInt(truncatingIfNeeded: v)))
      self.append(Element(UInt(truncatingIfNeeded: v &>> 32)))
    }
  }
}
