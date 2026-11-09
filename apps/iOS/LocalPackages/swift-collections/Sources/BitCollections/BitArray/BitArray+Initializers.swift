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

extension BitArray {
  /// Initialize a bit array from a bit set.
  ///
  /// The result contains exactly as many bits as the maximum item in
  /// the source set, plus one. If the set is empty, the result will
  /// be empty, too.
  ///
  ///     BitArray([] as BitSet)        // (empty)
  ///     BitArray([0] as BitSet)       // 1
  ///     BitArray([1] as BitSet)       // 10
  ///     BitArray([1, 2, 4] as BitSet) // 1011
  ///     BitArray([7] as BitSet)       // 1000000
  ///
  /// - Complexity: O(1)
  public init(_ set: BitSet) {
    guard let l = set.last else { self.init(); return }
    self.init(_storage: set._storage, count: l + 1)
  }

  /// Initialize a bit array from the binary representation of an integer.
  /// The result will contain exactly `value.bitWidth` bits.
  ///
  ///     BitArray(bitPattern: 3 as UInt8)  // 00000011
  ///     BitArray(bitPattern: 42 as Int8)  // 00101010
  ///     BitArray(bitPattern: -1 as Int8)  // 11111111
  ///     BitArray(bitPattern: 3 as Int16)  // 0000000000000111
  ///     BitArray(bitPattern: 42 as Int16) // 0000000000101010
  ///     BitArray(bitPattern: -1 as Int16) // 1111111111111111
  ///     BitArray(bitPattern: 3 as Int)    // 0000000000...0000000111
  ///     BitArray(bitPattern: 42 as Int)   // 0000000000...0000101010
  ///     BitArray(bitPattern: -1 as Int)   // 1111111111...1111111111
  ///
  /// - Complexity: O(value.bitWidth)
  public init(bitPattern value: some BinaryInteger) {
    var words = value.words.map { _Word($0) }
    let count = value.bitWidth
    if words.isEmpty {
      precondition(count == 0, "Inconsistent bitWidth")
    } else {
      let (w, b) = _UnsafeHandle._BitPosition(count).endSplit
      precondition(words.count == w + 1, "Inconsistent bitWidth")
      words[w].formIntersection(_Word(upTo: b))
    }
    self.init(_storage: words, count: count)
  }

  /// Creates a new, empty bit array with preallocated space for at least the
  /// specified number of elements.
  public init(minimumCapacity: Int) {
    self.init()
    reserveCapacity(minimumCapacity)
  }
}

extension BinaryInteger {
  @inlinable
  internal static func _convert(
    _ source: BitArray
  ) -> (value: Self, isNegative: Bool) {
    var value: Self = .zero
    let isNegative = source._foreachTwosComplementWordDownward(
      isSigned: Self.isSigned
    ) { _, word in
      value <<= UInt.bitWidth
      value |= Self(truncatingIfNeeded: word)
      return true
    }
    return (value, isNegative)
  }

  /// Creates a new instance by truncating or extending the bits in the given
  /// bit array, as needed. The bit at position 0 in `source` will correspond
  /// to the least-significant bit in the result.
  ///
  /// If `Self` is an unsigned integer type, then the result will contain as
  /// many bits from `source` it can accommodate, truncating off any extras.
  ///
  ///     UInt8(truncatingIfNeeded: "" as BitArray) // 0
  ///     UInt8(truncatingIfNeeded: "0" as BitArray) // 0
  ///     UInt8(truncatingIfNeeded: "1" as BitArray) // 1
  ///     UInt8(truncatingIfNeeded: "11" as BitArray) // 3
  ///     UInt8(truncatingIfNeeded: "11111111" as BitArray) // 255
  ///     UInt8(truncatingIfNeeded: "1100000001" as BitArray) // 1
  ///     UInt8(truncatingIfNeeded: "1100000101" as BitArray) // 5
  ///
  /// If `Self` is a signed integer type, then the contents of the bit array
  /// are interpreted to be a two's complement representation of a signed
  /// integer value, with the last bit in the array representing the sign of
  /// the result.
  ///
  ///     Int8(truncatingIfNeeded: "" as BitArray) // 0
  ///     Int8(truncatingIfNeeded: "0" as BitArray) // 0
  ///     Int8(truncatingIfNeeded: "1" as BitArray) // -1
  ///     Int8(truncatingIfNeeded: "01" as BitArray) // 1
  ///     Int8(truncatingIfNeeded: "101" as BitArray) // -3
  ///     Int8(truncatingIfNeeded: "0101" as BitArray) // 5
  ///
  ///     Int8(truncatingIfNeeded: "00000001" as BitArray) // 1
  ///     Int8(truncatingIfNeeded: "00000101" as BitArray) // 5
  ///     Int8(truncatingIfNeeded: "01111111" as BitArray) // 127
  ///     Int8(truncatingIfNeeded: "10000000" as BitArray) // -128
  ///     Int8(truncatingIfNeeded: "11111111" as BitArray) // -1
  ///
  ///     Int8(truncatingIfNeeded: "000011111111" as BitArray) // -1
  ///     Int8(truncatingIfNeeded: "111100000000" as BitArray) // 0
  ///     Int8(truncatingIfNeeded: "111100000001" as BitArray) // 1
  @inlinable
  public init(truncatingIfNeeded source: BitArray) {
    self = Self._convert(source).value
  }

  /// Creates a new instance from the bits in the given bit array, if the
  /// corresponding integer value can be represented exactly.
  /// If the value is not representable exactly, then the result is `nil`.
  ///
  /// If `Self` is an unsigned integer type, then the contents of the bit array
  /// are interpreted to be the binary representation of a nonnegative
  /// integer value. The bit array is allowed to contain bits in unrepresentable
  /// positions, as long as they are all cleared.
  ///
  ///     UInt8(exactly: "" as BitArray) // 0
  ///     UInt8(exactly: "0" as BitArray) // 0
  ///     UInt8(exactly: "1" as BitArray) // 1
  ///     UInt8(exactly: "10" as BitArray) // 2
  ///     UInt8(exactly: "00000000" as BitArray) // 0
  ///     UInt8(exactly: "11111111" as BitArray) // 255
  ///     UInt8(exactly: "0000000000000" as BitArray) // 0
  ///     UInt8(exactly: "0000011111111" as BitArray) // 255
  ///     UInt8(exactly: "0000100000000" as BitArray) // nil
  ///     UInt8(exactly: "1111111111111" as BitArray) // nil
  ///
  /// If `Self` is a signed integer type, then the contents of the bit array
  /// are interpreted to be a two's complement representation of a signed
  /// integer value, with the last bit in the array representing the sign of
  /// the result.
  ///
  ///     Int8(exactly: "" as BitArray) // 0
  ///     Int8(exactly: "0" as BitArray) // 0
  ///     Int8(exactly: "1" as BitArray) // -1
  ///     Int8(exactly: "01" as BitArray) // 1
  ///     Int8(exactly: "101" as BitArray) // -3
  ///     Int8(exactly: "0101" as BitArray) // 5
  ///
  ///     Int8(exactly: "00000001" as BitArray) // 1
  ///     Int8(exactly: "00000101" as BitArray) // 5
  ///     Int8(exactly: "01111111" as BitArray) // 127
  ///     Int8(exactly: "10000000" as BitArray) // -128
  ///     Int8(exactly: "11111111" as BitArray) // -1
  ///
  ///     Int8(exactly: "00000000000" as BitArray) // 0
  ///     Int8(exactly: "00001111111" as BitArray) // 127
  ///     Int8(exactly: "00010000000" as BitArray) // nil
  ///     Int8(exactly: "11101111111" as BitArray) // nil
  ///     Int8(exactly: "11110000000" as BitArray) // -128
  ///     Int8(exactly: "11111111111" as BitArray) // -1
  @inlinable
  public init?(exactly source: BitArray) {
    let (value, isNegative) = Self._convert(source)
    guard isNegative == (value < 0) else { return nil }
    let words = value.words
    var equal = true
    _ = source._foreachTwosComplementWordDownward(isSigned: Self.isSigned) { i, word in
      assert(equal)
      let w = (
        i < words.count ? words[i]
        : isNegative ? UInt.max
        : UInt.zero)
      equal = (w == word)
      return equal
    }
    guard equal else { return nil }
    self = value
  }

  /// Creates a new instance from the bits in the given bit array, if the
  /// corresponding integer value can be represented exactly.
  /// If the value is not representable exactly, then a runtime error will
  /// occur.
  ///
  /// If `Self` is an unsigned integer type, then the contents of the bit array
  /// are interpreted to be the binary representation of a nonnegative
  /// integer value. The bit array is allowed to contain bits in unrepresentable
  /// positions, as long as they are all cleared.
  ///
  ///     UInt8("" as BitArray) // 0
  ///     UInt8("0" as BitArray) // 0
  ///     UInt8("1" as BitArray) // 1
  ///     UInt8("10" as BitArray) // 2
  ///     UInt8("00000000" as BitArray) // 0
  ///     UInt8("11111111" as BitArray) // 255
  ///     UInt8("0000000000000" as BitArray) // 0
  ///     UInt8("0000011111111" as BitArray) // 255
  ///     UInt8("0000100000000" as BitArray) // ERROR
  ///     UInt8("1111111111111" as BitArray) // ERROR
  ///
  /// If `Self` is a signed integer type, then the contents of the bit array
  /// are interpreted to be a two's complement representation of a signed
  /// integer value, with the last bit in the array representing the sign of
  /// the result.
  ///
  ///     Int8("" as BitArray) // 0
  ///     Int8("0" as BitArray) // 0
  ///     Int8("1" as BitArray) // -1
  ///     Int8("01" as BitArray) // 1
  ///     Int8("101" as BitArray) // -3
  ///     Int8("0101" as BitArray) // 5
  ///
  ///     Int8("00000001" as BitArray) // 1
  ///     Int8("00000101" as BitArray) // 5
  ///     Int8("01111111" as BitArray) // 127
  ///     Int8("10000000" as BitArray) // -128
  ///     Int8("11111111" as BitArray) // -1
  ///
  ///     Int8("00000000000" as BitArray) // 0
  ///     Int8("00001111111" as BitArray) // 127
  ///     Int8("00010000000" as BitArray) // ERROR
  ///     Int8("11101111111" as BitArray) // ERROR
  ///     Int8("11110000000" as BitArray) // -128
  ///     Int8("11111111111" as BitArray) // -1
  @inlinable
  public init(_ source: BitArray) {
    guard let value = Self(exactly: source) else {
      fatalError("""
        BitArray value cannot be converted to \(Self.self) because it is \
        outside the representable range
        """)
    }
    self = value
  }
}

extension BitArray {
  @usableFromInline
  internal func _foreachTwosComplementWordDownward(
    isSigned: Bool,
    body: (Int, UInt) -> Bool
  ) -> Bool {
    self._read {
      guard $0._words.count > 0 else { return false }

      var isNegative = false
      let end = $0.end.endSplit
      assert(end.bit > 0)
      let last = $0._words[end.word]
      if isSigned, last.contains(end.bit - 1) {
        // Sign extend last word
        isNegative = true
        if !body(end.word, last.union(_Word(upTo: end.bit).complement()).value) {
          return isNegative
        }
      } else if !body(end.word, last.value) {
        return isNegative
      }
      for i in stride(from: end.word - 1, through: 0, by: -1) {
        if !body(i, $0._words[i].value) { return isNegative }
      }
      return isNegative
    }
  }
}
