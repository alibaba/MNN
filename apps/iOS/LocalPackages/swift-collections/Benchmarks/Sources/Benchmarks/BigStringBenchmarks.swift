//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2024 - 2025 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import CollectionsBenchmark
import _RopeModule
import Foundation

let someLatinSymbols: [UnicodeScalar] = [
  0x20 ..< 0x7f,
  0xa1 ..< 0xad,
  0xae ..< 0x2af,
  0x300 ..< 0x370,
  0x1e00 ..< 0x1eff,
].flatMap {
  $0.map { UnicodeScalar($0)! }
}

extension UnicodeScalar {
  static func randomLatin(
    using rng: inout some RandomNumberGenerator
  ) -> Self {
    someLatinSymbols.randomElement(using: &rng)!
  }
}

extension String.UnicodeScalarView {
  static func randomLatin(
    runeCount: Int, using rng: inout some RandomNumberGenerator
  ) -> Self {
    var result = String.UnicodeScalarView()
    for _ in 0 ..< runeCount {
      result.append(UnicodeScalar.randomLatin(using: &rng))
    }
    return result
  }
}

extension String {
  static func randomLatin(
    runeCount: Int, using rng: inout some RandomNumberGenerator
  ) -> Self {
    let text = String.UnicodeScalarView.randomLatin(
      runeCount: runeCount, using: &rng)
    return String(text)
  }
}

struct NativeStringInput {
  let value: String

  init(runeCount: Int, using rng: inout some RandomNumberGenerator) {
    self.value = String.randomLatin(runeCount: runeCount, using: &rng)
  }
}

struct BridgedStringInput {
  let value: String

  init(runeCount: Int, using rng: inout some RandomNumberGenerator) {
    let string = String.randomLatin(runeCount: runeCount, using: &rng)
    let utf16 = Array(string.utf16)
    let cocoa = utf16.withUnsafeBufferPointer {
      NSString(characters: $0.baseAddress!, length: $0.count)
    }
    self.value = cocoa as String
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
struct BigStringInput {
  let flat: String
  let big: BigString

  init(runeCount: Int, using rng: inout some RandomNumberGenerator) {
    self.flat = String.randomLatin(runeCount: runeCount, using: &rng)
    self.big = BigString(self.flat)
  }
}


extension Benchmark {
  public mutating func addBigStringBenchmarks() {
    guard #available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *) else {
      return
    }

    self.registerInputGenerator(for: NativeStringInput.self) { c in
      var rng = SystemRandomNumberGenerator()
      return NativeStringInput(runeCount: c, using: &rng)
    }

    self.registerInputGenerator(for: BridgedStringInput.self) { c in
      var rng = SystemRandomNumberGenerator()
      return BridgedStringInput(runeCount: c, using: &rng)
    }

    self.registerInputGenerator(for: BigStringInput.self) { c in
      var rng = SystemRandomNumberGenerator()
      return BigStringInput(runeCount: c, using: &rng)
    }

    // MARK: BigString initialization

    self.addSimple(
      title: "BigString init from native string",
      input: NativeStringInput.self
    ) { input in
      blackHole(BigString(input.value))
    }

    self.addSimple(
      title: "BigString init from bridged string",
      input: BridgedStringInput.self
    ) { input in
      blackHole(BigString(input.value))
    }

    // MARK: BigString iteration

    self.addSimple(
      title: "BigString character iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for char in input.big {
        blackHole(char)
      }
    }

    self.addSimple(
      title: "BigString scalar iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for scalar in input.big.unicodeScalars {
        blackHole(scalar)
      }
    }

    self.addSimple(
      title: "BigString UTF-8 iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for u8 in input.big.utf8 {
        blackHole(u8)
      }
    }

    self.addSimple(
      title: "BigString UTF-16 iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for u16 in input.big.utf16 {
        blackHole(u16)
      }
    }

    // MARK: String iteration

    self.addSimple(
      title: "String character iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for char in input.flat {
        blackHole(char)
      }
    }

    self.addSimple(
      title: "String scalar iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for scalar in input.flat.unicodeScalars {
        blackHole(scalar)
      }
    }

    self.addSimple(
      title: "String UTF-8 iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for u8 in input.flat.utf8 {
        blackHole(u8)
      }
    }

    self.addSimple(
      title: "String UTF-16 iteration",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      for u16 in input.flat.utf16 {
        blackHole(u16)
      }
    }

    // MARK: BigString index(offsetBy:)

    self.addSimple(
      title: "BigString.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let start = str.startIndex
      for i in 0 ... str.count {
        let index = str.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "BigString.unicodeScalars.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let start = str.startIndex
      for i in 0 ... str.unicodeScalars.count {
        let index = str.unicodeScalars.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "BigString.utf8.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let start = str.startIndex
      for i in 0 ... str.utf8.count {
        let index = str.utf8.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "BigString.utf16.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let start = str.startIndex
      for i in 0 ... str.utf16.count {
        let index = str.utf16.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    // MARK: String index(offsetBy:)

    self.addSimple(
      title: "String.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let start = str.startIndex
      for i in 0 ... str.count {
        let index = str.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "String.unicodeScalars.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let start = str.startIndex
      for i in 0 ... str.unicodeScalars.count {
        let index = str.unicodeScalars.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "String.utf8.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let start = str.startIndex
      for i in 0 ... str.utf8.count {
        let index = str.utf8.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    self.addSimple(
      title: "String.utf16.index(offsetBy:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let start = str.startIndex
      for i in 0 ... str.utf16.count {
        let index = str.utf16.index(start, offsetBy: i)
        blackHole(index)
      }
    }

    // MARK: BigString distance(from:to:)

    self.add(
      title: "BigString.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let indices = Array(str.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "BigString.unicodeScalars.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let indices = Array(str.unicodeScalars.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.unicodeScalars.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "BigString.utf8.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let indices = Array(str.utf8.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.utf8.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "BigString.utf16.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.big
      let indices = Array(str.utf16.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.utf16.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    // MARK: String distance(from:to:)

    self.add(
      title: "String.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let indices = Array(str.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "String.unicodeScalars.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let indices = Array(str.unicodeScalars.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.unicodeScalars.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "String.utf8.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let indices = Array(str.utf8.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.utf8.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }

    self.add(
      title: "String.utf16.distance(from:to:)",
      input: BigStringInput.self,
      maxSize: nil
    ) { input in
      let str = input.flat
      let indices = Array(str.utf16.indices) + [str.endIndex]
      return { timer in
        for i in 0 ..< indices.count {
          let j = (i + indices.count) / 2
          let expected = j - i
          let actual = str.utf16.distance(from: indices[i], to: indices[j])
          precondition(actual == expected)
        }
      }
    }
  }
}
