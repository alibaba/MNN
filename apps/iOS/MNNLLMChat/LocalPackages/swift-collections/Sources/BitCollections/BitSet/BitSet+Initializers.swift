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

extension BitSet {
  /// Initializes a new, empty bit set.
  ///
  /// This is equivalent to initializing with an empty array literal.
  /// For example:
  ///
  ///     let set1 = BitSet()
  ///     print(set1.isEmpty) // true
  ///
  ///     let set2: BitSet = []
  ///     print(set2.isEmpty) // true
  ///
  /// - Complexity: O(1)
  public init() {
    self.init(_rawStorage: [])
  }

  @usableFromInline
  init(_words: [_Word]) {
    self._storage = _words
    _shrink()
    _checkInvariants()
  }
  
  /// Initialize a new bit set from the raw bits of the supplied sequence of
  /// words. (The term "words" is used here to mean a sequence of `UInt`
  /// values, as in the `words` property of `BinaryInteger`.)
  ///
  /// The resulting bit set will contain precisely those integers that
  /// correspond to `true` bits within `words`. Bits are counted from least
  /// to most significant within each word.
  ///
  ///     let bits = BitSet(words: [5, 2])
  ///     // bits is [0, 2, UInt.bitWidth + 1]
  ///
  /// - Complexity: O(`words.count`)
  @inlinable
  public init(words: some Sequence<UInt>) {
    self.init(_words: words.map { _Word($0) })
  }
  
  /// Initialize a new bit set from the raw bits of the supplied integer value.
  ///
  /// The resulting bit set will contain precisely those integers that
  /// correspond to `true` bits within `x`. Bits are counted from least
  /// to most significant.
  ///
  ///     let bits = BitSet(bitPattern: 42)
  ///     // bits is [1, 3, 5]
  ///
  /// - Complexity: O(`x.bitWidth`)
  @inlinable
  public init(bitPattern x: some BinaryInteger) {
    self.init(words: x.words)
  }

  /// Initialize a new bit set from the storage bits of the given bit array.
  /// The resulting bit set will contain exactly those integers that address
  /// `true` elements in the array.
  ///
  /// Note that this conversion is lossy -- it discards the precise length of
  /// the input array.
  ///
  /// - Complexity: O(`array.count`)
  public init(_ array: BitArray) {
    self.init(_words: array._storage)
  }

  /// Create a new bit set containing the elements of a sequence.
  ///
  /// - Parameters:
  ///   - elements: The sequence of elements to turn into a bit set.
  ///
  /// - Complexity: O(*n*), where *n* is the number of elements in the sequence.
  @inlinable
  public init(
    _ elements: __owned some Sequence<Int>
  ) {
    if let elements = _specialize(elements, for: BitSet.self) {
      self = elements
      return
    }
    if let elements = _specialize(elements, for: BitSet.Counted.self) {
      self = elements._bits
      return
    }
    self.init()
    for value in elements {
      self.insert(value)
    }
  }

  /// Create a new bit set containing all the nonnegative elements of a
  /// sequence of integers.
  ///
  /// Items in `elements` that are negative are silently ignored.
  ///
  /// - Parameters:
  ///   - elements: The sequence of elements to turn into a bit set.
  ///
  /// - Complexity: O(*n*), where *n* is the number of elements in the sequence.
  @inlinable
  internal init(
    _validMembersOf elements: __owned some Sequence<Int>
  ) {
    if let elements = _specialize(elements, for: BitSet.self) {
      self = elements
      return
    }
    if let elements = _specialize(elements, for: BitSet.Counted.self) {
      self = elements._bits
      return
    }
    if let elements = _specialize(elements, for: Range<Int>.self) {
      let r = elements
      self.init(_range: r._clampedToUInt())
      return
    }
    self.init()
    for value in elements {
      guard let value = UInt(exactly: value) else { continue }
      self._insert(value)
    }
  }
}

extension BitSet {
  /// Create a new bit set containing the elements of a range of integers.
  ///
  /// - Parameters:
  ///   - range: The range to turn into a bit set. The range must not contain
  ///      negative values.
  ///
  /// - Complexity: O(`range.upperBound`)
  public init(_ range: Range<Int>) {
    guard let range = range._toUInt() else {
      preconditionFailure("BitSet can only hold nonnegative integers")
    }
    self.init(_range: range)
  }

  @usableFromInline
  internal init(_range range: Range<UInt>) {
    _storage = []
    let lower = _UnsafeHandle.Index(range.lowerBound)
    let upper = _UnsafeHandle.Index(range.upperBound)
    if lower.word > 0 {
      _storage.append(contentsOf: repeatElement(.empty, count: lower.word))
    }
    if lower.word == upper.word {
      _storage.append(_Word(from: lower.bit, to: upper.bit))
    } else {
      _storage.append(_Word(upTo: lower.bit).complement())
      let filledWords = upper.word &- lower.word
      if filledWords > 0 {
        _storage.append(
          contentsOf: repeatElement(.allBits, count: filledWords &- 1))
      }
      _storage.append(_Word(upTo: upper.bit))
    }
    _shrink()
    _checkInvariants()
  }
}

extension BitSet {
  internal init(
    _combining handles: (_UnsafeHandle, _UnsafeHandle),
    includingTail: Bool,
    using function: (_Word, _Word) -> _Word
  ) {
    let w1 = handles.0._words
    let w2 = handles.1._words
    let capacity = (
      includingTail
      ? Swift.max(w1.count, w2.count)
      : Swift.min(w1.count, w2.count))
    _storage = Array(unsafeUninitializedCapacity: capacity) { buffer, count in
      let sharedCount = Swift.min(w1.count, w2.count)
      for w in 0 ..< sharedCount {
        buffer.initializeElement(at: w, to: function(w1[w], w2[w]))
      }
      if includingTail {
        if w1.count < w2.count {
          for w in w1.count ..< w2.count {
            buffer.initializeElement(at: w, to: function(_Word.empty, w2[w]))
          }
        } else {
          for w in w2.count ..< w1.count {
            buffer.initializeElement(at: w, to: function(w1[w], _Word.empty))
          }
        }
      }
      // Adjust the word count based on results.
      count = capacity
      while count > 0, buffer[count - 1].isEmpty {
        count -= 1
      }
    }
    _checkInvariants()
  }
}
