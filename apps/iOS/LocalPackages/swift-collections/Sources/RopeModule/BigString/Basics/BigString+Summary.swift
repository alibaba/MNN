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
  struct Summary {
    // FIXME: We only need 48 * 3 = 192 bits to represent a nonnegative value; pack these better
    // (Unfortunately we also need to represent negative values right now.)
    private(set) var characters: Int
    private(set) var unicodeScalars: Int
    private(set) var utf16: Int
    private(set) var utf8: Int

    init() {
      characters = 0
      unicodeScalars = 0
      utf16 = 0
      utf8 = 0
    }

    init(_ chunk: BigString._Chunk) {
      self.utf8 = chunk.utf8Count
      self.utf16 = Int(chunk.counts.utf16)
      self.unicodeScalars = Int(chunk.counts.unicodeScalars)
      self.characters = Int(chunk.counts.characters)
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Summary: CustomStringConvertible {
  var description: String {
    "❨\(utf8)⋅\(utf16)⋅\(unicodeScalars)⋅\(characters)❩"
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.Summary: RopeSummary {
  @inline(__always)
  static var maxNodeSize: Int {
    #if DEBUG
    return 10
    #else
    return 15
    #endif
  }

  @inline(__always)
  static var nodeSizeBitWidth: Int { 4 }

  @inline(__always)
  static var zero: Self { Self() }

  @inline(__always)
  var isZero: Bool { utf8 == 0 }

  mutating func add(_ other: BigString.Summary) {
    characters += other.characters
    unicodeScalars += other.unicodeScalars
    utf16 += other.utf16
    utf8 += other.utf8
  }

  mutating func subtract(_ other: BigString.Summary) {
    characters -= other.characters
    unicodeScalars -= other.unicodeScalars
    utf16 -= other.utf16
    utf8 -= other.utf8
  }
}
