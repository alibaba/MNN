//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Rope {
  @inlinable
  public mutating func append(_ other: __owned Self) {
    self = Rope.join(self, other)
  }
  
  @inlinable
  public mutating func prepend(_ other: __owned Self) {
    self = Rope.join(other, self)
  }
  
  @inlinable
  internal mutating func _append(_ other: __owned _Node) {
    append(Self(root: other))
  }
  
  @inlinable
  internal mutating func _prepend(_ other: __owned _Node) {
    prepend(Self(root: other))
  }
  
  /// Concatenate `left` and `right` by linking up the two trees.
  @inlinable
  public static func join(_ left: __owned Self, _ right: __owned Self) -> Self {
    guard !right.isEmpty else { return left }
    guard !left.isEmpty else { return right }
    
    var left = left.root
    var right = right.root
    
    if left.height >= right.height {
      let r = left._graftBack(&right)
      guard let remainder = r.remainder else { return Self(root: left) }
      assert(left.height == remainder.height)
      let root = _Node.createInner(children: left, remainder)
      return Self(root: root)
      
    }
    let r = right._graftFront(&left)
    guard let remainder = r.remainder else { return Self(root: right) }
    assert(right.height == remainder.height)
    let root = _Node.createInner(children: remainder, right)
    return Self(root: root)
  }
}

extension Rope._Node {
  @inlinable
  internal mutating func _graftFront(
    _ scion: inout Self
  ) -> (remainder: Self?, delta: Summary) {
    assert(self.height >= scion.height)

    self.ensureUnique()
    scion.ensureUnique()

    guard self.height > scion.height else {
      assert(self.height == scion.height)
      let d = scion.summary
      if self.rebalance(prevNeighbor: &scion) {
        return (nil, d)
      }
      assert(!scion.isEmpty)
      return (scion, d.subtracting(scion.summary))
    }
    
    var (remainder, delta) = self.updateInner { h in
      h.mutableChildren[0]._graftFront(&scion)
    }
    self.summary.add(delta)
    guard let remainder = remainder else { return (nil, delta) }
    assert(self.height == remainder.height + 1)
    assert(!remainder.isUndersized)
    guard self.isFull else {
      delta.add(remainder.summary)
      self._insertNode(remainder, at: 0)
      return (nil, delta)
    }
    var splinter = self.split(keeping: self.childCount / 2)
    
    swap(&self, &splinter)
    delta.subtract(splinter.summary)
    splinter._insertNode(remainder, at: 0)
    return (splinter, delta)
  }

  @inlinable
  internal mutating func _graftBack(
    _ scion: inout Self
  ) -> (remainder: Self?, delta: Summary) {
    assert(self.height >= scion.height)

    self.ensureUnique()
    scion.ensureUnique()

    guard self.height > scion.height else {
      assert(self.height == scion.height)
      let origSum = self.summary
      let emptied = self.rebalance(nextNeighbor: &scion)
      return (emptied ? nil : scion, self.summary.subtracting(origSum))
    }
    
    var (remainder, delta) = self.updateInner { h in
      h.mutableChildren[h.childCount - 1]._graftBack(&scion)
    }
    self.summary.add(delta)
    guard let remainder = remainder else { return (nil, delta) }
    assert(self.height == remainder.height + 1)
    assert(!remainder.isUndersized)
    guard self.isFull else {
      delta.add(remainder.summary)
      self._appendNode(remainder)
      return (nil, delta)
    }
    var splinter = self.split(keeping: self.childCount / 2)
    delta.subtract(splinter.summary)
    splinter._appendNode(remainder)
    return (splinter, delta)
  }
}
