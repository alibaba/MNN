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
extension BigString: Hashable {
  public func hash(into hasher: inout Hasher) {
    hashCharacters(into: &hasher)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  internal func hashCharacters(into hasher: inout Hasher) {
    // FIXME: Implement properly normalized comparisons & hashing.
    // This is somewhat tricky as we shouldn't just normalize individual pieces of the string
    // split up on random Character boundaries -- Unicode does not promise that
    // norm(a + c) == norm(a) + norm(b) in this case.
    // To do this properly, we'll probably need to expose new stdlib entry points. :-/
    var it = self.makeIterator()
    while let character = it.next() {
      let s = String(character)
      s._withNFCCodeUnits { hasher.combine($0) }
    }
    hasher.combine(0xFF as UInt8)
  }
  
  /// Feed the UTF-8 encoding of `self` into hasher, with a terminating byte.
  internal func hashUTF8(into hasher: inout Hasher) {
    for chunk in self._rope {
      var string = chunk.string
      string.withUTF8 {
        hasher.combine(bytes: .init($0))
      }
    }
    hasher.combine(0xFF as UInt8)
  }

  /// Feed the UTF-8 encoding of `self[start..<end]` into hasher, with a terminating byte.
  internal func hashUTF8(into hasher: inout Hasher, from start: Index, to end: Index) {
    assert(start <= end)
    _foreachChunk(from: start, to: end) { str in
      var str = str
      str.withUTF8 {
        let buffer = UnsafeRawBufferPointer($0)
        hasher.combine(bytes: buffer)
      }
    }
    hasher.combine(0xFF as UInt8)
  }
}
