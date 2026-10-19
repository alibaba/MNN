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

/// A fixed-size array of just enough size to hold an ancestor path in a
/// `TreeDictionary`.
@usableFromInline
@frozen
internal struct _HashStack<Element> {
#if _pointerBitWidth(_64)
  @inlinable
  @inline(__always)
  internal static var capacity: Int { 13 }

  // xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xxxx
  @usableFromInline
  internal var _contents: (
    Element, Element, Element, Element,
    Element, Element, Element, Element,
    Element, Element, Element, Element,
    Element
  )
#elseif _pointerBitWidth(_32)
  @inlinable
  @inline(__always)
  internal static var capacity: Int { 7 }

  // xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx xx
  @usableFromInline
  internal var _contents: (
    Element, Element, Element, Element,
    Element, Element, Element
  )
#else
#error("Unexpected pointer bit width")
#endif

  @usableFromInline
  internal var _count: UInt8

  @inlinable
  internal init(filledWith value: Element) {
    assert(Self.capacity == _HashLevel.limit)
#if _pointerBitWidth(_64)
    _contents = (
      value, value, value, value,
      value, value, value, value,
      value, value, value, value,
      value
    )
#elseif _pointerBitWidth(_32)
    _contents = (
      value, value, value, value,
      value, value, value
    )
#else
#error("Unexpected pointer bit width")
#endif
    self._count = 0
  }

  @inlinable
  @inline(__always)
  internal var capacity: Int { Self.capacity }

  @inlinable
  @inline(__always)
  internal var count: Int { Int(truncatingIfNeeded: _count) }

  @inlinable
  @inline(__always)
  internal var isEmpty: Bool { _count == 0 }

  @inlinable
  subscript(level: UInt8) -> Element {
    mutating get {
      assert(level < _count)
      return withUnsafeBytes(of: &_contents) { buffer in
        // Homogeneous tuples are layout compatible with their element type
        let start = buffer.baseAddress!.assumingMemoryBound(to: Element.self)
        return start[Int(truncatingIfNeeded: level)]
      }
    }
    set {
      assert(level < capacity)
      withUnsafeMutableBytes(of: &_contents) { buffer in
        // Homogeneous tuples are layout compatible with their element type
        let start = buffer.baseAddress!.assumingMemoryBound(to: Element.self)
        start[Int(truncatingIfNeeded: level)] = newValue
      }
    }
  }

  @inlinable
  mutating func push(_ item: Element) {
    assert(_count < capacity)
    self[_count] = item
    _count &+= 1
  }

  @inlinable
  mutating func pop() -> Element {
    assert(_count > 0)
    defer { _count &-= 1 }
    return self[_count &- 1]
  }

  @inlinable
  mutating func peek() -> Element {
    assert(count > 0)
    return self[_count &- 1]
  }
}
