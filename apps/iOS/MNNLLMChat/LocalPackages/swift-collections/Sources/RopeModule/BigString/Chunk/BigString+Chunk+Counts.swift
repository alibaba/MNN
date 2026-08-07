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
extension BigString._Chunk {
  struct Counts: Equatable {
    /// The number of UTF-8 code units within this chunk.
    var utf8: UInt8
    /// The number of UTF-16 code units within this chunk.
    var utf16: UInt8
    /// The number of Unicode scalars within this chunk.
    var unicodeScalars: UInt8
    /// The number of Unicode scalars within this chunk that start a Character.
    var _characters: UInt8
    /// The number of UTF-8 code units at the start of this chunk that continue a Character
    /// whose start scalar is in a previous chunk.
    var _prefix: UInt8
    /// The number of UTF-8 code units at the end of this chunk that form the start a Character
    /// whose end scalar is in a subsequent chunk.
    var _suffix: UInt8
    
    init() {
      self.utf8 = 0
      self.utf16 = 0
      self.unicodeScalars = 0
      self._characters = 0
      self._prefix =  0
      self._suffix = 0
    }
    
    init(
      utf8: UInt8,
      utf16: UInt8,
      unicodeScalars: UInt8,
      characters: UInt8,
      prefix: UInt8,
      suffix: UInt8
    ) {
      assert(characters >= 0 && characters <= unicodeScalars && unicodeScalars <= utf16)
      self.utf8 = utf8
      self.utf16 = utf16
      self.unicodeScalars = unicodeScalars
      self._characters = characters
      self._prefix = prefix
      self._suffix = suffix
    }
    
    init(
      utf8: Int,
      utf16: Int,
      unicodeScalars: Int,
      characters: Int,
      prefix: Int,
      suffix: Int
    ) {
      assert(characters >= 0 && characters <= unicodeScalars && unicodeScalars <= utf16)
      self.utf8 = UInt8(utf8)
      self.utf16 = UInt8(utf16)
      self.unicodeScalars = UInt8(unicodeScalars)
      self._characters = UInt8(characters)
      self._prefix = UInt8(prefix)
      self._suffix = UInt8(suffix)
    }
    
    init(
      anomalousUTF8 utf8: Int,
      utf16: Int,
      unicodeScalars: Int
    ) {
      self.utf8 = UInt8(utf8)
      self.utf16 = UInt8(utf16)
      self.unicodeScalars = UInt8(unicodeScalars)
      self._characters = 0
      self._prefix = self.utf8
      self._suffix = self.utf8
    }
    
    init(_ slice: Slice) {
      let c = slice.string.utf8.count
      precondition(c <= BigString._Chunk.maxUTF8Count)
      self.init(
        utf8: slice.string.utf8.count,
        utf16: slice.string.utf16.count,
        unicodeScalars: slice.string.unicodeScalars.count,
        characters: slice.characters,
        prefix: slice.prefix,
        suffix: slice.suffix)
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk.Counts {
  var characters: Int {
    get { Int(_characters) }
    set { _characters = UInt8(newValue) }
  }
  
  var prefix: Int {
    get { Int(_prefix) }
    set { _prefix = UInt8(newValue) }
  }
  
  var suffix: Int {
    get { Int(_suffix) }
    set { _suffix = UInt8(newValue) }
  }
  
  var hasBreaks: Bool {
    _prefix < utf8
  }
  
  func hasSpaceToMerge(_ other: Self) -> Bool {
    Int(utf8) + Int(other.utf8) <= BigString._Chunk.maxUTF8Count
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk.Counts {
  mutating func append(_ other: Self) {
    assert(hasSpaceToMerge(other))
    
    switch (self.hasBreaks, other.hasBreaks) {
    case (true, true):
      self._suffix = other._suffix
    case (true, false):
      self._suffix += other._suffix
    case (false, true):
      self._prefix += other._prefix
      self._suffix = other._suffix
    case (false, false):
      self._prefix += other._prefix
      self._suffix += other._suffix
    }
    self.utf8 += other.utf8
    self.utf16 += other.utf16
    self.unicodeScalars += other.unicodeScalars
    self._characters += other._characters
  }
}
