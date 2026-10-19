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

public struct IndexRangeCollection<Bound: Strideable>
where Bound.Stride == Int
{
  var _bounds: Range<Bound>
  
  public init(bounds: Range<Bound>) {
    self._bounds = bounds
  }
}

extension IndexRangeCollection: RandomAccessCollection {
  public typealias Element = Range<Bound>
  public typealias Iterator = IndexingIterator<Self>
  public typealias SubSequence = Slice<Self>
  public typealias Indices = DefaultIndices<Self>

  public struct Index: Comparable {
    var _start: Int
    var _end: Int
    
    internal init(_start: Int, end: Int) {
      assert(_start >= 0 && _start <= end)
      self._start = _start
      self._end = end
    }

    internal init(_offset: Int) {
      assert(_offset >= 0)
      let end = ((8 * _offset + 1)._squareRoot() - 1) / 2
      let base = end * (end + 1) / 2
      self._start = _offset - base
      self._end = end
    }

    internal var _offset: Int {
      return _end * (_end + 1) / 2 + _start
    }
    
    public static func ==(left: Self, right: Self) -> Bool {
      left._start == right._start && left._end == right._end
    }

    public static func <(left: Self, right: Self) -> Bool {
      (left._end, left._start) < (right._end, right._start)
    }
  }

  public var count: Int { (_bounds.count + 1) * (_bounds.count + 2) / 2 }

  public var isEmpty: Bool { false }
  public var startIndex: Index { Index(_start: 0, end: 0) }
  public var endIndex: Index { Index(_start: 0, end: _bounds.count + 1) }
  
  public func index(after i: Index) -> Index {
    guard i._start < i._end else {
      return Index(_start: 0, end: i._end + 1)
    }
    return Index(_start: i._start + 1, end: i._end)
  }

  public func index(before i: Index) -> Index {
    guard i._start > 0 else {
      return Index(_start: i._end - 1, end: i._end - 1)
    }
    return Index(_start: i._start - 1, end: i._end)
  }

  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    Index(_offset: i._offset + distance)
  }
  
  public subscript(position: Index) -> Range<Bound> {
    precondition(position._end <= _bounds.count)
    return Range(
      uncheckedBounds: (
        lower: _bounds.lowerBound.advanced(by: position._start),
        upper: _bounds.lowerBound.advanced(by: position._end)))
  }
}
