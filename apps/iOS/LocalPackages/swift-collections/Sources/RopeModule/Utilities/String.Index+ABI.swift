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

// Bits of String.Index that are ABI but aren't exposed by public API.
extension String.Index {
  @inline(__always)
  var _abi_rawBits: UInt64 {
    unsafeBitCast(self, to: UInt64.self)
  }
  
  @inline(__always)
  var _abi_encodedOffset: Int {
    Int(truncatingIfNeeded: _abi_rawBits &>> 16)
  }

  @inline(__always)
  var _abi_transcodedOffset: Int {
    Int(truncatingIfNeeded: (_abi_rawBits &>> 14) & 0x3)
  }

  @inline(__always)
  static var _abi_scalarAlignmentBit: UInt64 { 0x1 }

  @inline(__always)
  static var _abi_characterAlignmentBit: UInt64 { 0x2 }

  @inline(__always)
  static var _abi_utf8Bit: UInt64 { 0x4 }

  @inline(__always)
  static var _abi_utf16Bit: UInt64 { 0x8 }

  
  @inline(__always)
  var _encodingBits: UInt64 {
    _abi_rawBits & (Self._abi_utf8Bit | Self._abi_utf16Bit)
  }
  
  @inline(__always)
  var _canBeUTF8: Bool {
    // The only way an index cannot be UTF-8 is it has only the UTF-16 flag set.
    _encodingBits != Self._abi_utf16Bit
  }
  
  var _isKnownScalarAligned: Bool {
    0 != _abi_rawBits & Self._abi_scalarAlignmentBit
  }
  
  var _isKnownCharacterAligned: Bool {
    0 != _abi_rawBits & Self._abi_characterAlignmentBit
  }
  
  var _knownCharacterAligned: String.Index {
    let r = _abi_rawBits | Self._abi_characterAlignmentBit | Self._abi_scalarAlignmentBit
    return unsafeBitCast(r, to: String.Index.self)
  }

  var _knownScalarAligned: String.Index {
    let r = _abi_rawBits | Self._abi_scalarAlignmentBit
    return unsafeBitCast(r, to: String.Index.self)
  }
}

extension String.Index {
  @inline(__always)
  var _utf8Offset: Int {
    assert(_canBeUTF8)
    return _abi_encodedOffset
  }

  @inline(__always)
  var _isUTF16TrailingSurrogate: Bool {
    assert(_canBeUTF8)
    let r = _abi_transcodedOffset
    assert(r <= 1)
    return r > 0
  }

  @inline(__always)
  init(_rawBits: UInt64) {
    self = unsafeBitCast(_rawBits, to: String.Index.self)
  }

  @inline(__always)
  init(_utf8Offset: Int) {
    self.init(_rawBits: (UInt64(_utf8Offset) &<< 16) | Self._abi_utf8Bit)
  }
  
  init(_utf8Offset: Int, utf16TrailingSurrogate: Bool) {
    let transcodedOffset: UInt64 = (utf16TrailingSurrogate ? 1 &<< 14 : 0)
    self.init(_rawBits: (UInt64(_utf8Offset) &<< 16) | transcodedOffset | Self._abi_utf8Bit)
  }
}

extension String.Index {
  @inline(__always)
  var _chunkData: UInt16 {
    UInt16(_abi_rawBits &>> 14)
  }

  @inline(__always)
  init(_chunkData: UInt16) {
    self.init(_rawBits: UInt64(_chunkData) &<< 14)
  }
}
