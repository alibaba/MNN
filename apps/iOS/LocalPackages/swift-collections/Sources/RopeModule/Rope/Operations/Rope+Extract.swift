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
  public func extract(_ offsetRange: Range<Int>, in metric: some RopeMetric<Element>) -> Self {
    extract(from: offsetRange.lowerBound, to: offsetRange.upperBound, in: metric)
  }

  @inlinable
  public func extract(from start: Int, to end: Int, in metric: some RopeMetric<Element>) -> Self {
    if _root == nil {
      precondition(start == 0 && end == 0, "Invalid range")
      return Self()
    }
    var builder = Builder()
    root.extract(from: start, to: end, in: metric, into: &builder)
    return builder.finalize()
  }
}

extension Rope._Node {
  @inlinable
  internal func extract(
    from start: Int,
    to end: Int,
    in metric: some RopeMetric<Element>,
    into builder: inout Rope.Builder
  ) {
    let size = metric.size(of: summary)
    precondition(start >= 0 && start <= end && end <= size, "Range out of bounds")

    guard start != end else { return }

    if self.isLeaf {
      self.readLeaf {
        let l = $0.findSlot(at: start, in: metric, preferEnd: false)
        let u = $0.findSlot(from: l, offsetBy: end - start, in: metric, preferEnd: true)
        let c = $0.children
        if l.slot == u.slot {
          var item = c[l.slot]
          let i = metric.index(at: l.remaining, in: item.value)
          var item2 = item.split(at: i)
          let j = metric.index(
            at: u.remaining - metric._nonnegativeSize(of: item.summary),
            in: item2.value)
          _ = item2.split(at: j)
          builder._insertBeforeTip(item2)
          return
        }
        assert(l.slot < u.slot)
        var left = c[l.slot]
        left = left.split(at: metric.index(at: l.remaining, in: left.value))
        builder._insertBeforeTip(left)
        for i in l.slot + 1 ..< u.slot {
          builder._insertBeforeTip(c[i])
        }
        var right = c[u.slot]
        _ = right.split(at: metric.index(at: u.remaining, in: right.value))
        builder._insertBeforeTip(right)
      }
      return
    }

    self.readInner {
      let l = $0.findSlot(at: start, in: metric, preferEnd: false)
      let u = $0.findSlot(from: l, offsetBy: end - start, in: metric, preferEnd: true)
      let c = $0.children
      if l.slot == u.slot {
        c[l.slot].extract(
          from: l.remaining, to: u.remaining, in: metric, into: &builder)
        return
      }
      assert(l.slot < u.slot)
      let lsize = metric._nonnegativeSize(of: c[l.slot].summary)
      c[l.slot].extract(from: l.remaining, to: lsize, in: metric, into: &builder)
      for i in l.slot + 1 ..< u.slot {
        builder._insertBeforeTip(c[i])
      }
      c[u.slot].extract(from: 0, to: u.remaining, in: metric, into: &builder)
    }
  }
}
