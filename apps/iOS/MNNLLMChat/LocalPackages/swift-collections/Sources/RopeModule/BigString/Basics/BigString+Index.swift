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

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public  struct Index: Sendable {
    typealias _Rope = BigString._Rope

    // ┌───────────────────────────────┬───┬───────────┬────────────────────┐
    // │ b63:b11                       │b10│ b9:b8     │ b7:b0              │
    // ├───────────────────────────────┼───┼───────────┼────────────────────┤
    // │ UTF-8 global offset           │ T │ alignment │ UTF-8 chunk offset │
    // └───────────────────────────────┴───┴───────────┴────────────────────┘
    // b10 (T): UTF-16 trailing surrogate indicator
    // b9: isCharacterAligned
    // b8: isScalarAligned
    //
    // 100: UTF-16 trailing surrogate
    // 001: Index known to be scalar aligned
    // 011: Index known to be Character aligned
    var _rawBits: UInt64

    /// A (possibly invalid) rope index.
    var _rope: _Rope.Index?

    internal init(_raw: UInt64, _rope: _Rope.Index?) {
      self._rawBits = _raw
      self._rope = _rope
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index {
  @inline(__always)
  internal static func _bitsForUTF8Offset(_ utf8Offset: Int) -> UInt64 {
    let v = UInt64(truncatingIfNeeded: UInt(bitPattern: utf8Offset))
    assert(v &>> 53 == 0)
    return v &<< 11
  }

  @inline(__always)
  internal static var _flagsMask: UInt64 { 0x700 }

  @inline(__always)
  internal static var _utf16TrailingSurrogateBits: UInt64 { 0x400 }

  @inline(__always)
  internal static var _characterAlignmentBit: UInt64 { 0x200 }

  @inline(__always)
  internal static var _scalarAlignmentBit: UInt64 { 0x100 }

  public var utf8Offset: Int {
    Int(truncatingIfNeeded: _rawBits &>> 11)
  }

  @inline(__always)
  internal var _orderingValue: UInt64 {
    _rawBits &>> 10
  }

  /// The offset within the addressed chunk. Only valid if `_rope` is not nil.
  internal var _utf8ChunkOffset: Int {
    assert(_rope != nil)
    return Int(truncatingIfNeeded: _rawBits & 0xFF)
  }

  /// The base offset of the addressed chunk. Only valid if `_rope` is not nil.
  internal var _utf8BaseOffset: Int {
    utf8Offset - _utf8ChunkOffset
  }

  @inline(__always)
  internal var _flags: UInt64 {
    get {
      _rawBits & Self._flagsMask
    }
    set {
      assert(newValue & ~Self._flagsMask == 0)
      _rawBits &= ~Self._flagsMask
      _rawBits |= newValue
    }
  }
}

extension String.Index {
  @available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
  func _copyingAlignmentBits(from i: BigString.Index) -> String.Index {
    var bits = _abi_rawBits & ~3
    bits |= (i._flags &>> 8) & 3
    return String.Index(_rawBits: bits)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index {
  internal var _chunkIndex: String.Index {
    assert(_rope != nil)
    return String.Index(
      _utf8Offset: _utf8ChunkOffset, utf16TrailingSurrogate: _isUTF16TrailingSurrogate
    )._copyingAlignmentBits(from: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index {
  internal mutating func _clearUTF16TrailingSurrogate() {
    _flags = 0
  }

  public var _isUTF16TrailingSurrogate: Bool {
    _orderingValue & 1 != 0
  }

  internal func _knownScalarAligned() -> Self {
    var copy = self
    copy._flags = Self._scalarAlignmentBit
    return copy
  }

  internal func _knownCharacterAligned() -> Self {
    var copy = self
    copy._flags = Self._characterAlignmentBit | Self._scalarAlignmentBit
    return copy
  }

  public var _isKnownScalarAligned: Bool {
    _rawBits & Self._scalarAlignmentBit != 0
  }

  public var _isKnownCharacterAligned: Bool {
    _rawBits & Self._characterAlignmentBit != 0
  }

  public init(_utf8Offset: Int) {
    _rawBits = Self._bitsForUTF8Offset(_utf8Offset)
    _rope = nil
  }

  public init(_utf8Offset: Int, utf16TrailingSurrogate: Bool) {
    _rawBits = Self._bitsForUTF8Offset(_utf8Offset)
    if utf16TrailingSurrogate {
      _rawBits |= Self._utf16TrailingSurrogateBits
    }
    _rope = nil
  }

  internal init(
    _utf8Offset: Int, utf16TrailingSurrogate: Bool = false, _rope: _Rope.Index, chunkOffset: Int
  ) {
    _rawBits = Self._bitsForUTF8Offset(_utf8Offset)
    if utf16TrailingSurrogate {
      _rawBits |= Self._utf16TrailingSurrogateBits
    }
    assert(chunkOffset >= 0 && chunkOffset <= 0xFF)
    _rawBits |= UInt64(truncatingIfNeeded: chunkOffset) & 0xFF
    self._rope = _rope
  }

  internal init(baseUTF8Offset: Int, _rope: _Rope.Index, chunk: String.Index) {
    let chunkUTF8Offset = chunk._utf8Offset
    self.init(
      _utf8Offset: baseUTF8Offset + chunkUTF8Offset,
      utf16TrailingSurrogate: chunk._isUTF16TrailingSurrogate,
      _rope: _rope,
      chunkOffset: chunkUTF8Offset)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    left._orderingValue == right._orderingValue
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index: Comparable {
  public static func <(left: Self, right: Self) -> Bool {
    left._orderingValue < right._orderingValue
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(_orderingValue)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Index: CustomStringConvertible {
  public var description: String {
    let utf16Offset = _isUTF16TrailingSurrogate ? "+1" : ""
    return "\(utf8Offset)[utf8]\(utf16Offset)"
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  func resolve(_ i: Index, preferEnd: Bool) -> Index {
    if var ri = i._rope, _rope.isValid(ri) {
      if preferEnd {
        guard i.utf8Offset > 0, i._utf8ChunkOffset == 0, !i._isUTF16TrailingSurrogate else {
          return i
        }
        _rope.formIndex(before: &ri)
        let length = _rope[ri].utf8Count
        let ci = String.Index(_utf8Offset: length)
        var j = Index(baseUTF8Offset: i.utf8Offset - length, _rope: ri, chunk: ci)
        j._flags = i._flags
        return j
      }
      guard i.utf8Offset < _utf8Count, i._utf8ChunkOffset == _rope[ri].utf8Count else {
        return i
      }
      _rope.formIndex(after: &ri)
      let ci = String.Index(_utf8Offset: 0)
      var j = Index(baseUTF8Offset: i.utf8Offset, _rope: ri, chunk: ci)
      j._flags = i._flags
      return j
    }

    // Indices addressing trailing surrogates must never be resolved at the end of chunk,
    // because the +1 doesn't make sense on any endIndex.
    let trailingSurrogate = i._isUTF16TrailingSurrogate

    let (ri, chunkOffset) = _rope.find(
      at: i.utf8Offset,
      in: _UTF8Metric(),
      preferEnd: preferEnd && !trailingSurrogate)

    let ci = String.Index(
      _utf8Offset: chunkOffset,
      utf16TrailingSurrogate: trailingSurrogate)
    return Index(baseUTF8Offset: i.utf8Offset - ci._utf8Offset, _rope: ri, chunk: ci)
  }
}
