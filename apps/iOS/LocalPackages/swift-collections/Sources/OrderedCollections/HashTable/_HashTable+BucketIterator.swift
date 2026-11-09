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

extension _HashTable {
  /// An iterator construct for visiting a chain of buckets within the hash
  /// table. This is a convenient tool for implementing linear probing.
  ///
  /// Beyond merely providing bucket values, bucket iterators can also tell
  /// you their current opposition within the hash table, and (for mutable hash
  /// tables) they allow you update the value of the currently visited bucket.
  /// (This is useful when implementing simple insertions, for example.)
  ///
  /// The bucket iterator caches some bucket contents, so if you are looping
  /// over an iterator you must be careful to only modify hash table contents
  /// through the iterator itself.
  ///
  /// - Warning: Like `UnsafeHandle`, `BucketIterator` does not have
  ///     ownership of its underlying storage buffer. You must not escape
  ///     iterator values outside the closure call that produced the original
  ///     hash table.
  @usableFromInline
  internal struct BucketIterator {
    @usableFromInline
    internal typealias Bucket = _HashTable.Bucket

    /// The hash table we are iterating over.
    internal let _hashTable: _UnsafeHashTable

    /// The current position within the hash table.
    @usableFromInline
    internal var _currentBucket: Bucket

    /// The raw bucket value corresponding to `_currentBucket`.
    internal var _currentRawValue: UInt64

    /// Remaining bits not yet processed from the last word read.
    internal var _nextBits: UInt64

    /// Count of usable bits in `_nextBits`. (They start at bit 0.)
    internal var _remainingBitCount: Int

    internal var _wrappedAround = false

    /// Create a new iterator starting at the specified bucket.
    @_effects(releasenone)
    @usableFromInline
    internal init(hashTable: _UnsafeHashTable, startingAt bucket: Bucket) {
      assert(hashTable.scale >= _HashTable.minimumScale)
      assert(bucket.offset >= 0 && bucket.offset < hashTable.bucketCount)
      self._hashTable = hashTable
      self._currentBucket = bucket
      (self._currentRawValue, self._nextBits, self._remainingBitCount)
        = hashTable._startIterator(bucket: bucket)
    }
  }
}

extension _HashTable.UnsafeHandle {
  @usableFromInline
  internal typealias BucketIterator = _HashTable.BucketIterator

  @_effects(releasenone)
  @inlinable
  @inline(__always)
  internal func idealBucket(forHashValue hashValue: Int) -> Bucket {
    return Bucket(offset: hashValue & (bucketCount - 1))
  }

  @inlinable
  @inline(__always)
  internal func idealBucket<Element: Hashable>(for element: Element) -> Bucket {
    let hashValue = element._rawHashValue(seed: seed)
    return idealBucket(forHashValue: hashValue)
  }

  /// Return a bucket iterator for the chain starting at the bucket corresponding
  /// to the specified value.
  @inlinable
  @inline(__always)
  internal func bucketIterator<Element: Hashable>(for element: Element) -> BucketIterator {
    let bucket = idealBucket(for: element)
    return bucketIterator(startingAt: bucket)
  }

  /// Return a bucket iterator for the chain starting at the specified bucket.
  @inlinable
  @inline(__always)
  internal func bucketIterator(startingAt bucket: Bucket) -> BucketIterator {
    BucketIterator(hashTable: self, startingAt: bucket)
  }

  @usableFromInline
  @_effects(releasenone)
  internal func startFind(
    _ startBucket: Bucket
  ) -> (iterator: BucketIterator, currentValue: Int?) {
    let iterator = bucketIterator(startingAt: startBucket)
    return (iterator, iterator.currentValue)
  }

  @_effects(readonly)
  @usableFromInline
  internal func _startIterator(
    bucket: Bucket
  ) -> (currentBits: UInt64, nextBits: UInt64, remainingBitCount: Int) {
    // The `scale == 5` case is special because the last word is only half filled there,
    // which is why the code below needs to special case it.
    // (For all scales > 5, the last bucket ends exactly on a word boundary.)

    var (word, bit) = self.position(of: bucket)
    if bit + scale <= 64 {
      // We're in luck, the current bucket is stored entirely within one word.
      let w = self[word: word]
      let currentRawValue = (w &>> bit) & bucketMask
      let c = (scale == 5 && word == wordCount - 1 ? 32 : 64)
      let remainingBitCount = c - (bit + scale)
      let nextBits = (remainingBitCount == 0 ? 0 : w &>> (bit + scale))
      assert(remainingBitCount >= 0)
      assert(bit < c)
      return (currentRawValue, nextBits, remainingBitCount)
    } else {
      // We need to read two words.
      assert(scale != 5 || word < wordCount - 1)
      assert(bit > 0)
      let w1 = self[word: word]
      word = self.word(after: word)
      let w2 = self[word: word]
      let currentRawValue = ((w1 &>> bit) | (w2 &<< (64 - bit))) & bucketMask
      let overhang = scale - (64 - bit)
      let nextBits = w2 &>> overhang
      let c = (scale == 5 && word == wordCount - 1 ? 32 : 64)
      let remainingBitCount = c - overhang
      return (currentRawValue, nextBits, remainingBitCount)
    }
  }
}

extension _HashTable.BucketIterator {
  /// The scale of the hash table. A table of scale *n* holds 2^*n* buckets,
  /// each of which contain an *n*-bit value.
  @inline(__always)
  internal var _scale: Int { _hashTable.scale }

  /// The current position within the hash table.
  @inlinable
  @inline(__always)
  internal var currentBucket: Bucket { _currentBucket }

  @usableFromInline
  internal var isOccupied: Bool {
    @_effects(readonly)
    @inline(__always)
    get {
      _currentRawValue != 0
    }
  }

  /// The value of the bucket at the current position in the hash table.
  /// Setting this property overwrites the bucket value.
  ///
  /// A nil value indicates an empty bucket.
  @usableFromInline
  internal var currentValue: Int? {
    @inline(__always)
    @_effects(readonly)
    get { _hashTable._value(forBucketContents: _currentRawValue) }
    @_effects(releasenone)
    set {
      _hashTable.assertMutable()
      let v = _hashTable._bucketContents(for: newValue)
      let pattern = v ^ _currentRawValue

      assert(_currentBucket.offset < _hashTable.bucketCount)
      let (word, bit) = _hashTable.position(of: _currentBucket)
      _hashTable[word: word] ^= pattern &<< bit
      let extractedBits = 64 - bit
      if extractedBits < _scale {
        let word2 = _hashTable.word(after: word)
        _hashTable[word: word2] ^= pattern &>> extractedBits
      }
      _currentRawValue = v
    }
  }

  /// Advance this iterator to the next bucket within the hash table.
  /// The buckets form a cycle, so the last bucket is logically followed
  /// by the first. Therefore, the iterator never runs out of buckets --
  /// you must devise some way to guarantee to stop iterating.
  ///
  /// In the typical case, you stop iterating buckets when you find the
  /// element you're looking for, or when you run across an empty bucket
  /// (terminating the chain with a negative lookup result).
  ///
  /// To catch mistakes (and corrupt tables), `advance` traps the second
  /// time it needs to wrap around to the beginning of the table.
  @usableFromInline
  @_effects(releasenone)
  internal mutating func advance() {
    // Advance to next bucket, checking for wraparound condition.
    _currentBucket.offset &+= 1
    if _currentBucket.offset == _hashTable.bucketCount {
      guard !_wrappedAround else {
        // Prevent wasting battery in an infinite loop if a hash table
        // somehow becomes corrupt.
        fatalError("Hash table has no unoccupied buckets")
      }
      _wrappedAround = true
      _currentBucket.offset = 0
    }

    // If we have loaded enough bits, eat them and return.
    if _remainingBitCount >= _scale {
      _currentRawValue = _nextBits & _hashTable.bucketMask
      _nextBits &>>= _scale
      _remainingBitCount -= _scale
      return
    }

    // Load the next batch of bits.
    var word = _hashTable.position(of: _currentBucket).word
    if _remainingBitCount != 0 {
      word = _hashTable.word(after: word)
    }
    let c = (_hashTable.scale == 5 && word == _hashTable.wordCount - 1 ? 32 : 64)
    let w = _hashTable[word: word]
    _currentRawValue = (_nextBits | (w &<< _remainingBitCount)) & _hashTable.bucketMask
    _nextBits = w &>> (_scale - _remainingBitCount)
    _remainingBitCount = c - (_scale - _remainingBitCount)
  }

  @usableFromInline
  @_effects(releasenone)
  internal mutating func findNext() -> Int? {
    advance()
    return currentValue
  }

  /// Advance this iterator until it points to an occupied bucket with the
  /// specified value, or an unoccupied bucket -- whichever comes first.
  @inlinable
  @_effects(releasenone)
  internal mutating func advance(until expected: Int) {
    while isOccupied && currentValue != expected {
      advance()
    }
  }

  /// Advance this iterator until it points to an unoccupied bucket.
  /// Useful when inserting an element that we know isn't already in the table.
  @inlinable
  @_effects(releasenone)
  internal mutating func advanceToNextUnoccupiedBucket() {
    while isOccupied {
      advance()
    }
  }
}
