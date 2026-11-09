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
  internal struct _Chunk {
    typealias Slice = (string: Substring, characters: Int, prefix: Int, suffix: Int)
    
    var string: String
    var counts: Counts
    
    init() {
      self.string = ""
      self.counts = Counts()
    }
    
    init(_ string: String, _ counts: Counts) {
      self.string = string
      self.string.makeContiguousUTF8()
      self.counts = counts
      invariantCheck()
    }
    
    init(_ string: Substring, _ counts: Counts) {
      self.string = String(string)
      self.counts = counts
      invariantCheck()
    }
    
    init(_ slice: Slice) {
      self.string = String(slice.string)
      self.string.makeContiguousUTF8()
      self.counts = Counts((self.string[...], slice.characters, slice.prefix, slice.suffix))
      invariantCheck()
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  @inline(__always)
  static var maxUTF8Count: Int { 255 }
  
  @inline(__always)
  static var minUTF8Count: Int { maxUTF8Count / 2 - maxSlicingError }
  
  @inline(__always)
  static var maxSlicingError: Int { 3 }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  @inline(__always)
  mutating func take() -> Self {
    let r = self
    self = Self()
    return r
  }

  @inline(__always)
  mutating func modify<R>(
    _ body: (inout Self) -> R
  ) -> R {
    body(&self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  @inline(__always)
  var characterCount: Int { counts.characters }
  
  @inline(__always)
  var unicodeScalarCount: Int { Int(counts.unicodeScalars) }
  
  @inline(__always)
  var utf16Count: Int { Int(counts.utf16) }
  
  @inline(__always)
  var utf8Count: Int { Int(counts.utf8) }
  
  @inline(__always)
  var prefixCount: Int { counts.prefix }
  
  @inline(__always)
  var suffixCount: Int { counts.suffix }
  
  var firstScalar: UnicodeScalar { string.unicodeScalars.first! }
  var lastScalar: UnicodeScalar { string.unicodeScalars.last! }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  var availableSpace: Int { Swift.max(0, Self.maxUTF8Count - utf8Count) }
  
  mutating func clear() {
    string = ""
    counts = Counts()
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  func hasSpaceToMerge(_ other: some StringProtocol) -> Bool {
    utf8Count + other.utf8.count <= Self.maxUTF8Count
  }
  
  func hasSpaceToMerge(_ other: Self) -> Bool {
    utf8Count + other.utf8Count <= Self.maxUTF8Count
  }
}
