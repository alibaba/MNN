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

extension RandomAccessCollection {
  #if SWIFT_PACKAGE
  @inline(__always)
  internal func _index(at offset: Int) -> Index {
    index(startIndex, offsetBy: offset)
  }

  @inline(__always)
  internal func _offset(of index: Index) -> Int {
    distance(from: startIndex, to: index)
  }

  @inline(__always)
  internal subscript(_offset offset: Int) -> Element {
    self[_index(at: offset)]
  }
  #endif

  @inline(__always)
  internal func _indexRange(at offsets: Range<Int>) -> Range<Index> {
    _index(at: offsets.lowerBound) ..< _index(at: offsets.upperBound)
  }

  @inline(__always)
  internal func _indexRange<R: RangeExpression>(at offsets: R) -> Range<Index>
  where R.Bound == Int {
    return _indexRange(at: offsets.relative(to: 0 ..< self.count))
  }

  @inline(__always)
  internal func _offsetRange(of range: Range<Index>) -> Range<Int> {
    _offset(of: range.lowerBound) ..< _offset(of: range.upperBound)
  }

  @inline(__always)
  internal func _offsetRange<R: RangeExpression>(of range: R) -> Range<Int>
  where R.Bound == Index {
    return _offsetRange(of: range.relative(to: self))
  }

  @inline(__always)
  internal subscript(_offsets range: Range<Int>) -> SubSequence {
    self[_indexRange(at: range)]
  }

  @inline(__always)
  internal subscript<R: RangeExpression>(_offsets range: R) -> SubSequence
  where R.Bound == Int {
    self[_offsets: range.relative(to: 0 ..< count)]
  }
}

