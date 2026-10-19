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

@usableFromInline
internal typealias _UnsafeHashTable = _HashTable.UnsafeHandle

extension _HashTable {
  /// A non-owning handle to hash table storage, implementing higher-level
  /// table operations.
  ///
  /// - Warning: `_UnsafeHashTable` values do not have ownership of their
  ///    underlying storage buffer. You must not escape these handles outside
  ///    the closure call that produced them.
  @usableFromInline
  @frozen
  internal struct UnsafeHandle {
    @usableFromInline
    internal typealias Bucket = _HashTable.Bucket

    /// A pointer to the table header.
    @usableFromInline
    internal var _header: UnsafeMutablePointer<Header>

    /// A pointer to bucket storage.
    @usableFromInline
    internal var _buckets: UnsafeMutablePointer<UInt64>

    #if DEBUG
    /// True when this handle does not support table mutations.
    /// (This is only checked in debug builds.)
    @usableFromInline
    internal let _readonly: Bool
    #endif

    /// Initialize a new hash table handle for storage at the supplied locations.
    @inlinable
    @inline(__always)
    internal init(
      header: UnsafeMutablePointer<Header>,
      buckets: UnsafeMutablePointer<UInt64>,
      readonly: Bool
    ) {
      self._header = header
      self._buckets = buckets
      #if DEBUG
      self._readonly = readonly
      #endif
    }

    /// Check that this handle supports mutating operations.
    /// Every member that mutates table data must start by calling this function.
    /// This helps preventing COW violations.
    ///
    /// Note that this is a noop in release builds.
    @inlinable
    @inline(__always)
    func assertMutable() {
      #if DEBUG
      assert(!_readonly, "Attempt to mutate a hash table through a read-only handle")
      #endif
    }
  }
}

extension _HashTable.UnsafeHandle {
  /// The scale of the hash table. A table of scale *n* holds 2^*n* buckets,
  /// each of which contain an *n*-bit value.
  @inlinable
  @inline(__always)
  internal var scale: Int { _header.pointee.scale }

  /// The scale corresponding to the last call to `reserveCapacity`.
  /// We store this to make sure we don't shrink the table below its reserved size.
  @inlinable
  @inline(__always)
  internal var reservedScale: Int { _header.pointee.reservedScale }

  /// The hasher seed to use within this hash table.
  @inlinable
  @inline(__always)
  internal var seed: Int { _header.pointee.seed }

  /// A bias value that needs to be added to buckets to convert them into offsets
  /// into element storage. (This allows O(1) insertions at the front when the
  /// underlying storage supports it.)
  @inlinable
  @inline(__always)
  internal var bias: Int {
    get { _header.pointee.bias }
    nonmutating set { _header.pointee.bias = newValue }
  }

  /// The number of buckets within this hash table. This is always a power of two.
  @inlinable
  @inline(__always)
  internal var bucketCount: Int { 1 &<< scale }

  @inlinable
  @inline(__always)
  internal var bucketMask: UInt64 { UInt64(truncatingIfNeeded: bucketCount) - 1 }

  /// The number of bits used to store all the buckets in this hash table.
  /// Each bucket holds a value that is `scale` bits wide.
  @inlinable
  @inline(__always)
  internal var bitCount: Int { scale &<< scale }

  /// The number of 64-bit words that are available in the storage buffer,
  /// rounded up to the nearest whole number if necessary.
  @inlinable
  @inline(__always)
  internal var wordCount: Int { (bitCount + UInt64.bitWidth - 1) / UInt64.bitWidth }

  /// The maximum number of items that can fit into this table.
  @inlinable
  @inline(__always)
  internal var capacity: Int { _HashTable.maximumCapacity(forScale: scale) }

  /// Return the bucket logically following `bucket` in this hash table.
  /// The buckets form a cycle, so the last bucket is logically followed by the first.
  @inlinable
  @inline(__always)
  func bucket(after bucket: Bucket) -> Bucket {
    var offset = bucket.offset + 1
    if offset == bucketCount {
      offset = 0
    }
    return Bucket(offset: offset)
  }

  /// Return the bucket logically preceding `bucket` in this hash table.
  /// The buckets form a cycle, so the first bucket is logically preceded by the last.
  @inlinable
  @inline(__always)
  func bucket(before bucket: Bucket) -> Bucket {
    let offset = (bucket.offset == 0 ? bucketCount : bucket.offset) - 1
    return Bucket(offset: offset)
  }

  /// Return the index of the word logically following `word` in this hash table.
  /// The buckets form a cycle, so the last word is logically followed by the first.
  ///
  /// Note that the last word may be only partially filled if `scale` is less than 6.
  @inlinable
  @inline(__always)
  func word(after word: Int) -> Int {
    var result = word + 1
    if result == wordCount {
      result = 0
    }
    return result
  }

  /// Return the index of the word logically preceding `word` in this hash table.
  /// The buckets form a cycle, so the first word is logically preceded by the first.
  ///
  /// Note that the last word may be only partially filled if `scale` is less than 6.
  @inlinable
  @inline(__always)
  func word(before word: Int) -> Int {
    if word == 0 {
      return wordCount - 1
    }
    return word - 1
  }

  /// Return the index of the 64-bit storage word that holds the first bit
  /// corresponding to `bucket`, along with its bit position within the word.
  @inlinable
  internal func position(of bucket: Bucket) -> (word: Int, bit: Int) {
    let start = bucket.offset &* scale
    return (start &>> 6, start & 0x3F)
  }
}

extension _HashTable.UnsafeHandle {
  /// Decode and return the logical value corresponding to the specified bucket value.
  ///
  /// The nil value is represented by an all-zero bit pattern.
  /// Other values are stored as the complement of the lowest `scale` bits
  /// after taking `bias` into account.
  /// The range of representable values is `0 ..< bucketCount - 1`.
  /// (Note that the value `bucketCount - 1` is missing from this range, as its
  /// encoding is used for `nil`. This isn't an issue, because the maximum load
  /// factor guarantees that the hash table will never be completely full.)
  @inlinable
  func _value(forBucketContents bucketContents: UInt64) -> Int? {
    let mask = bucketMask
    assert(bucketContents <= mask)
    guard bucketContents != 0 else { return nil }
    let v = (bucketContents ^ mask) &+ UInt64(truncatingIfNeeded: bias)
    return Int(truncatingIfNeeded: v >= mask ? v - mask : v)
  }

  /// Encodes the specified logical value into a `scale`-bit bit pattern suitable
  /// for storing into a bucket.
  ///
  /// The nil value is represented by an all-zero bit pattern.
  /// Other values are stored as the complement of their lowest `scale` bits.
  /// The range of representable values is `0 ..< bucketCount - 1`.
  /// (Note that the value `bucketCount - 1` is missing from this range, as it
  /// its encoding is used for `nil`. This isn't an issue, because the maximum
  /// load factor guarantees that the hash table will never be completely full.)
  @inlinable
  func _bucketContents(for value: Int?) -> UInt64 {
    guard var value = value else { return 0 }
    let mask = Int(truncatingIfNeeded: bucketMask)
    assert(value >= 0 && value < mask)
    value &-= bias
    if value < 0 { value += mask }
    assert(value >= 0 && value < mask)
    return UInt64(truncatingIfNeeded: value ^ mask)
  }

  @inlinable
  subscript(word word: Int) -> UInt64 {
    @inline(__always) get {
      assert(word >= 0 && word < bucketCount)
      return _buckets[word]
    }
    @inline(__always) nonmutating set {
      assert(word >= 0 && word < bucketCount)
      assertMutable()
      _buckets[word] = newValue
    }
  }

  @inlinable
  subscript(raw bucket: Bucket) -> UInt64 {
    get {
      assert(bucket.offset < bucketCount)
      let (word, bit) = position(of: bucket)
      var value = self[word: word] &>> bit
      let extractedBits = 64 - bit
      if extractedBits < scale {
        let word2 = self.word(after: word)
        value &= (1 &<< extractedBits) - 1
        value |= self[word: word2] &<< extractedBits
      }
      return value & bucketMask
    }
    nonmutating set {
      assertMutable()
      assert(bucket.offset < bucketCount)
      let mask = bucketMask
      assert(newValue <= mask)
      let (word, bit) = position(of: bucket)
      self[word: word] &= ~(mask &<< bit)
      self[word: word] |= newValue &<< bit
      let extractedBits = 64 - bit
      if extractedBits < scale {
        let word2 = self.word(after: word)
        self[word: word2] &= ~((1 &<< (scale - extractedBits)) - 1)
        self[word: word2] |= newValue &>> extractedBits
      }
    }
  }

  @inlinable
  @inline(__always)
  func isOccupied(_ bucket: Bucket) -> Bool {
    self[raw: bucket] != 0
  }

  /// Return or update the current value stored in the specified bucket.
  /// A nil value indicates that the bucket is empty.
  @inlinable
  internal subscript(bucket: Bucket) -> Int? {
    get {
      let contents = self[raw: bucket]
      return _value(forBucketContents: contents)
    }
    nonmutating set {
      assertMutable()
      let v = _bucketContents(for: newValue)
      self[raw: bucket] = v
    }
  }
}

extension _UnsafeHashTable {
  @inlinable
  internal func _find<Element: Hashable>(
    _ item: Element,
    in elements: ContiguousArray<Element>
  ) -> (index: Int?, bucket: Bucket) {
    elements.withUnsafeBufferPointer { buffer in
      _find(item, in: buffer)
    }
  }

  @inlinable
  internal func _find<Element: Hashable>(
    _ item: Element,
    in elements: UnsafeBufferPointer<Element>
  ) -> (index: Int?, bucket: Bucket) {
    let start = idealBucket(for: item)
    var (iterator, value) = startFind(start)
    while let index = value {
      if elements[_offset: index] == item {
        return (index, iterator.currentBucket)
      }
      value = iterator.findNext()
    }
    return (nil, iterator.currentBucket)
  }
}

extension _UnsafeHashTable {
  @usableFromInline
  internal func firstOccupiedBucketInChain(with bucket: Bucket) -> Bucket {
    var bucket = bucket
    repeat {
      bucket = self.bucket(before: bucket)
    } while isOccupied(bucket)
    return self.bucket(after: bucket)
  }

  @inlinable
  internal func delete(
    bucket: Bucket,
    hashValueGenerator: (Int, Int) -> Int // (offset, seed) -> hashValue
  ) {
    assertMutable()
    var it = bucketIterator(startingAt: bucket)
    assert(it.isOccupied)
    it.advance()
    guard it.isOccupied else {
      // Fast path: Don't get the start bucket when there's nothing to do.
      self[bucket] = nil
      return
    }
    // If we've put a hole in the middle of a collision chain, some element after
    // the hole may belong where the new hole is.

    // Find the first bucket in the collision chain that contains the entry we've just deleted.
    let start = firstOccupiedBucketInChain(with: bucket)
    var hole = bucket

    while it.isOccupied {
      let hash = hashValueGenerator(it.currentValue!, seed)
      let candidate = idealBucket(forHashValue: hash)

      // Does this element belong between start and hole?  We need two
      // separate tests depending on whether [start, hole] wraps around the
      // end of the storage.
      let c0 = candidate.offset >= start.offset
      let c1 = candidate.offset <= hole.offset
      if start.offset <= hole.offset ? (c0 && c1) : (c0 || c1) {
        // Fill the hole. Here we are mutating table contents behind the back of
        // the iterator; this is okay since we know we are never going to revisit
        // `hole` with it.
        self[hole] = it.currentValue
        hole = it.currentBucket
      }
      it.advance()
    }
    self[hole] = nil
  }
}

extension _UnsafeHashTable {
  @inlinable
  internal func adjustContents<Base: RandomAccessCollection>(
    preparingForInsertionOfElementAtOffset offset: Int,
    in elements: Base
  ) where Base.Element: Hashable {
    assertMutable()
    let index = elements._index(at: offset)
    if offset < elements.count / 2 {
      self.bias += 1
      if offset <= capacity / 3 {
        var i = 1
        for item in elements[..<index] {
          var it = bucketIterator(for: item)
          it.advance(until: i)
          it.currentValue! -= 1
          i += 1
        }
      } else {
        var it = bucketIterator(startingAt: Bucket(offset: 0))
        repeat {
          if let value = it.currentValue, value <= offset {
            it.currentValue = value - 1
          }
          it.advance()
        } while it.currentBucket.offset != 0
      }
    } else {
      if elements.count - offset - 1 <= capacity / 3 {
        var i = offset
        for item in elements[index...] {
          var it = bucketIterator(for: item)
          it.advance(until: i)
          it.currentValue! += 1
          i += 1
        }
      } else {
        var it = bucketIterator(startingAt: Bucket(offset: 0))
        repeat {
          if let value = it.currentValue, value >= offset {
            it.currentValue = value + 1
          }
          it.advance()
        } while it.currentBucket.offset != 0
      }
    }
  }
}

extension _UnsafeHashTable {
  @inlinable
  @inline(__always)
  internal func adjustContents<Base: RandomAccessCollection>(
    preparingForRemovalOf index: Base.Index,
    in elements: Base
  ) where Base.Element: Hashable {
    let next = elements.index(after: index)
    adjustContents(preparingForRemovalOf: index ..< next, in: elements)
  }

  @inlinable
  internal func adjustContents<Base: RandomAccessCollection>(
    preparingForRemovalOf bounds: Range<Base.Index>,
    in elements: Base
  ) where Base.Element: Hashable {
    assertMutable()
    let startOffset = elements._offset(of: bounds.lowerBound)
    let endOffset = elements._offset(of: bounds.upperBound)
    let c = endOffset - startOffset
    guard c > 0 else { return }
    let remainingCount = elements.count - c

    if startOffset >= remainingCount / 2 {
      let tailCount = elements.count - endOffset
      if tailCount < capacity / 3 {
        var i = endOffset
        for item in elements[bounds.upperBound...] {
          var it = self.bucketIterator(for: item)
          it.advance(until: i)
          it.currentValue = i - c
          i += 1
        }
      } else {
        var it = bucketIterator(startingAt: Bucket(offset: 0))
        repeat {
          if let value = it.currentValue {
            if value >= endOffset {
              it.currentValue = value - c
            } else {
              assert(value < startOffset)
            }
          }
          it.advance()
        } while it.currentBucket.offset != 0
      }
    } else {
      if startOffset < capacity / 3 {
        var i = 0
        for item in elements[..<bounds.lowerBound] {
          var it = self.bucketIterator(for: item)
          it.advance(until: i)
          it.currentValue = i + c
          i += 1
        }
      } else {
        var it = bucketIterator(startingAt: Bucket(offset: 0))
        repeat {
          if let value = it.currentValue {
            if value < startOffset {
              it.currentValue = value + c
            } else {
              assert(value >= endOffset)
            }
          }
          it.advance()
        } while it.currentBucket.offset != 0
      }
      self.bias -= c
    }
  }
}

extension _UnsafeHashTable {
  @usableFromInline
  internal func clear() {
    assertMutable()
    _buckets.update(repeating: 0, count: wordCount)
  }
}

extension _UnsafeHashTable {
  /// Fill an empty hash table by populating it with data from `elements`.
  ///
  /// - Parameter elements: A random-access collection for which this table is being generated.
  @inlinable
  internal func fill<C: RandomAccessCollection>(
    uncheckedUniqueElements elements: C
  ) where C.Element: Hashable {
    assertMutable()
    assert(elements.count <= capacity)
    // fast path that doesn't allocate per element if _read accessor can't
    // be inlined because this function doesn't get specialized e.g.
    // if `Element` isn't known at compile time.
    let fastPath: Void? = elements.withContiguousStorageIfAvailable { elements in
      // Iterate over elements and insert their offset into the hash table.
      var offset = 0
      for index in elements.indices {
        // Find the insertion position. We know that we're inserting a new item,
        // so there is no need to compare it with any of the existing ones.
        var it = bucketIterator(for: elements[index])
        it.advanceToNextUnoccupiedBucket()
        it.currentValue = offset
        offset += 1
      }
    }
    if fastPath != nil {
      return
    }
    // Iterate over elements and insert their offset into the hash table.
    var offset = 0
    for index in elements.indices {
      // Find the insertion position. We know that we're inserting a new item,
      // so there is no need to compare it with any of the existing ones.
      var it = bucketIterator(for: elements[index])
      it.advanceToNextUnoccupiedBucket()
      it.currentValue = offset
      offset += 1
    }
  }

  /// Fill an empty hash table by populating it with data from `elements`.
  ///
  /// - Parameter elements: A random-access collection for which this table is being generated.
  /// - Parameter stoppingOnFirstDuplicateValue: If true, check for duplicate values and stop inserting items when one is found.
  /// - Returns: `(success, index)` where `success` is a boolean value indicating that every value in `elements` was successfully inserted. A false success indicates that duplicate elements have been found; in this case `index` points to the first duplicate value; otherwise `index` is set to `elements.endIndex`.
  @inlinable
  internal func fill<C: RandomAccessCollection>(
    untilFirstDuplicateIn elements: C
  ) -> (success: Bool, end: C.Index)
  where C.Element: Hashable {
    assertMutable()
    assert(elements.count <= capacity)
    // Iterate over elements and insert their offset into the hash table.
    
    // fast path that doesn't allocate per element if _read accessor can't
    // be inlined because this function doesn't get specialized e.g.
    // if `Element` isn't known at compile time.
    let fastPath: (success: Bool, end: Int)? = elements.withContiguousStorageIfAvailable { elements in
      var offset = 0
      for index in elements.indices {
        // Find the insertion position. We know that we're inserting a new item,
        // so there is no need to compare it with any of the existing ones.
        var it = bucketIterator(for: elements[index])
        while let offset = it.currentValue {
          guard elements[_offset: offset] != elements[index] else {
            return (false, index)
          }
          it.advance()
        }
        it.currentValue = offset
        offset += 1
      }
      return (true, elements.endIndex)
    }
    if let fastPath {
      return (fastPath.success, elements.index(elements.startIndex, offsetBy: fastPath.end))
    }
    var offset = 0
    for index in elements.indices {
      // Find the insertion position. We know that we're inserting a new item,
      // so there is no need to compare it with any of the existing ones.
      var it = bucketIterator(for: elements[index])
      while let offset = it.currentValue {
        guard elements[_offset: offset] != elements[index] else {
          return (false, index)
        }
        it.advance()
      }
      it.currentValue = offset
      offset += 1
    }
    return (true, elements.endIndex)
  }
}
