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

import XCTest

#if DEBUG // These unit tests use internal decls

#if COLLECTIONS_SINGLE_MODULE
@_spi(Testing) @testable import Collections
#else
import _CollectionsTestSupport
@_spi(Testing) @testable import OrderedCollections
#endif

class HashTableTests: CollectionTestCase {
  typealias Bucket = _HashTable.Bucket

  func test_capacity() {
    withEvery("capacity", in: 0 ..< 1000) { capacity in
      let scale = _HashTable.scale(forCapacity: capacity)
      let maximumCapacity = _HashTable.maximumCapacity(forScale: scale)
      let minimumCapacity = _HashTable.minimumCapacity(forScale: scale)

      if scale == 0 {
        expectEqual(minimumCapacity, 0)
        expectEqual(maximumCapacity, _HashTable.maximumUnhashedCount)
        expectGreaterThanOrEqual(maximumCapacity, capacity)
      } else {
        let bucketCount = 1 &<< scale
        expectGreaterThanOrEqual(scale, _HashTable.minimumScale)
        expectLessThan(scale, _HashTable.maximumScale)
        expectGreaterThanOrEqual(maximumCapacity, capacity)
        expectGreaterThan(bucketCount, maximumCapacity)
        expectLessThan(minimumCapacity, maximumCapacity)
        expectLessThanOrEqual(minimumCapacity, capacity)
        expectGreaterThan(minimumCapacity, 0)
      }
    }
  }

  func test_Storage_create() {
    let s5 = _HashTable(scale: 5)
    expectEqual(s5.header.scale, 5)
    expectEqual(s5.header.reservedScale, 0)
    expectTrue(s5.description.starts(with: "_HashTable(scale: 5, reservedScale: 0, bias: 0, seed: "))
    expectTrue(s5.header.description.starts(with: "_HashTable.Header(scale: 5, reservedScale: 0, bias: 0, seed: "))
    s5.read { hashTable in
      for i in 0 ..< 32 {
        let bucket = Bucket(offset: i)
        expectNil(hashTable[bucket], "\(bucket.offset)")
      }
    }

    let s6 = _HashTable(scale: 6)
    expectEqual(s6.header.scale, 6)
    expectEqual(s6.header.reservedScale, 0)
    expectTrue(s6.description.starts(with: "_HashTable(scale: 6, reservedScale: 0, bias: 0, seed: "))
    expectTrue(s6.header.description.starts(with: "_HashTable.Header(scale: 6, reservedScale: 0, bias: 0, seed: "))
    s6.read { hashTable in
      for i in 0 ..< 64 {
        let bucket = Bucket(offset: i)
        expectNil(hashTable[bucket], "\(bucket.offset)")
      }
    }

    expectNotEqual(s6.header.seed, s5.header.seed) // This is somewhat shaky b/c we're losing address bits

    expectEqual(s5.header.capacity, 24) // 0.75 * 2^5
    expectEqual(s6.header.capacity, 48) // 0.75 * 2^6
  }

  func test_Storage_read() {
    let s5 = _HashTable(scale: 5)
    s5.read { hashTable in
      expectEqual(hashTable.scale, 5)
      expectEqual(hashTable.reservedScale, 0)
      expectEqual(hashTable.seed, s5.header.seed)
      expectTrue(hashTable.description.starts(with: "_HashTable.UnsafeHandle(scale: 5, reservedScale: 0, bias: 0, seed: "))
    }
    let s6 = _HashTable(scale: 6)
    s6.read { hashTable in
      expectEqual(hashTable.scale, 6)
      expectEqual(hashTable.reservedScale, 0)
      expectEqual(hashTable.seed, s6.header.seed)
      expectTrue(hashTable.description.starts(with: "_HashTable.UnsafeHandle(scale: 6, reservedScale: 0, bias: 0, seed: "))
    }
  }

  func test_counts() {
    let s5 = _HashTable(scale: 5)
    s5.read { hashTable in
      expectEqual(hashTable.bucketCount, 32)
      expectEqual(hashTable.bitCount, 5 * 32)
      expectEqual(hashTable.wordCount, 3) // 2.5 rounded up
      expectEqual(hashTable.capacity, 24) // 0.75 * 32
    }
    let s6 = _HashTable(scale: 6)
    s6.read { hashTable in
      expectEqual(hashTable.bucketCount, 64)
      expectEqual(hashTable.bitCount, 6 * 64)
      expectEqual(hashTable.wordCount, 6)
      expectEqual(hashTable.capacity, 48) // 0.75 * 64
    }
    let s12 = _HashTable(scale: 12)
    s12.read { hashTable in
      expectEqual(hashTable.bucketCount, 4096)
      expectEqual(hashTable.bitCount, 12 * 4096)
      expectEqual(hashTable.wordCount, 768)
      expectEqual(hashTable.capacity, 3072) // 0.75 * 4096
    }
  }

  func test_bucket_after() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        withEvery("offset", in: 0 ..< hashTable.bucketCount) { offset in
          let input = Bucket(offset: offset)
          let expected = Bucket(offset: offset == hashTable.bucketCount - 1 ? 0 : offset + 1)
          let actual = hashTable.bucket(after: input)
          expectEqual(actual, expected)
        }
      }
    }
  }
  func test_bucket_before() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        withEvery("offset", in: 0 ..< hashTable.bucketCount) { offset in
          let input = Bucket(offset: offset)
          let expected = Bucket(offset: offset == 0 ? hashTable.bucketCount - 1 : offset - 1)
          let actual = hashTable.bucket(before: input)
          expectEqual(actual, expected)
        }
      }
    }
  }
  func test_word_after() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        withEvery("input", in: 0 ..< hashTable.wordCount) { input in
          let expected = (input == hashTable.wordCount - 1 ? 0 : input + 1)
          let actual = hashTable.word(after: input)
          expectEqual(actual, expected)
        }
      }
    }
  }
  func test_word_before() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        withEvery("input", in: 0 ..< hashTable.wordCount) { input in
          let expected = (input == 0 ? hashTable.wordCount - 1 : input - 1)
          let actual = hashTable.word(before: input)
          expectEqual(actual, expected)
        }
      }
    }
  }
  func test_coordinates_for_bucket() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        withEvery("offset", in: 0 ..< hashTable.bucketCount) { offset in
          let (word, bit) = hashTable.position(of: Bucket(offset: offset))
          expectGreaterThanOrEqual(bit, 0)
          expectGreaterThanOrEqual(word, 0)
          expectLessThan(bit, 64)
          expectLessThan(word, hashTable.wordCount)
          let pos = 64 * word + bit
          expectEqual(pos % scale, 0)
          expectEqual(pos / scale, offset)
        }
      }
    }
  }

  func test_bucketContents_forValue() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      storage.read { hashTable in
        expectEqual(hashTable._bucketContents(for: nil), 0)
        expectEqual(hashTable._value(forBucketContents: 0), nil)
        withEvery("value", in: 0 ..< hashTable.bucketCount - 1) { value in
          let contents = hashTable._bucketContents(for: value)
          expectEqual(hashTable._value(forBucketContents: contents), value)
        }
      }
    }
  }

  func test_subscript_by_bucket() {
    withEvery("scale", in: [5, 6, 12]) { scale in
      let storage = _HashTable(scale: scale)
      let bucketCount = 1 << scale
      let contents: [Int?] = Array(0 ..< bucketCount - 1) + [nil]
      storage.update { hashTable in
        for offset in 0 ..< bucketCount {
          let bucket = Bucket(offset: offset)
          hashTable[bucket] = contents[offset]
        }
      }
      storage.read { hashTable in
        withEvery("offset", in: 0 ..< bucketCount) { offset in
          let bucket = Bucket(offset: offset)
          expectEqual(hashTable[bucket], contents[offset])
        }
      }
    }
  }

  /// Create hash table storage of scale `scale`, filled with equal-sized lookup chains
  /// of length `chainLength`, separated by holes of length `holeLength`, with the
  /// first chain starting at `startOffset`. The chains contain bucket values that
  /// sequentially count up from 0.
  func sampleTable(
    scale: Int,
    chainLength: Int,
    holeLength: Int,
    startingAt startOffset: Int
  ) -> (table: _HashTable, contents: [Int?]) {
    let bucketCount = 1 << scale

    var contents: [Int?] = []
    contents.reserveCapacity(bucketCount + chainLength + holeLength)
    contents.append(contentsOf: repeatElement(nil, count: startOffset))
    var i = 0
    while contents.count < bucketCount {
      contents.append(contentsOf: (i ..< i + chainLength).map { $0 })
      i += chainLength
      contents.append(contentsOf: repeatElement(nil, count: holeLength))
    }
    contents.removeLast(contents.count - bucketCount)

    let storage = _HashTable(scale: scale)
    storage.update { hashTable in
      for offset in 0 ..< bucketCount {
        let bucket = Bucket(offset: offset)
        hashTable[bucket] = contents[offset]
      }
    }
    return (storage, contents)
  }

  /// Call `body` with a hash table of scale `scale`, filled with equal-sized lookup chains
  /// of length `chainLength`, separated by holes of length `holeLength`, with the
  /// first chain starting at `startOffset`. The chains contain bucket values that
  /// sequentially count up from 0.
  func withSampleHashTable<R>(
    scale: Int,
    chainLength: Int,
    holeLength: Int,
    startingAt startOffset: Int,
    body: (_UnsafeHashTable, [Int?]) throws -> R
  ) rethrows -> R {
    let (storage, contents) = sampleTable(
      scale: scale,
      chainLength: chainLength,
      holeLength: holeLength,
      startingAt: startOffset)
    return try storage.update { hashTable in
      try body(hashTable, contents)
    }
  }

  /// Create hash table storage of scale `scale`, filled with a single huge lookup chain
  /// of buckets counting sequentially up starting at 0, terminating with the penultimate
  /// bucket. The last bucket remains unoccupied.
  ///
  /// This is useful for testing all possible bucket values.
  func sampleTable(scale: Int) -> (table: _HashTable, contents: [Int?]) {
    sampleTable(scale: scale, chainLength: (1 << scale) - 1, holeLength: 1, startingAt: 0)
  }

  /// Call `body` with a hash table of scale `scale`, filled with a single huge lookup chain
  /// of buckets counting sequentially up starting at 0, terminating with the penultimate
  /// bucket. The last bucket remains unoccupied.
  ///
  /// This is useful for testing all possible bucket values.
  func withSampleHashTable<R>(
    scale: Int,
    body: (_UnsafeHashTable, [Int?]) throws -> R
  ) rethrows -> R {
    let (table, contents) = sampleTable(scale: scale)
    return try table.update { hashTable in
      try body(hashTable, contents)
    }
  }

  func test_bucketIterator_read() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withSampleHashTable(scale: scale) { hashTable, contents in
        // Start an iterator at the first bucket and cycle through
        // all buckets twice, checking the values we see at every step.
        let bucket = Bucket(offset: 0)
        var it = hashTable.bucketIterator(startingAt: bucket)
        expectEqual(it.currentBucket, bucket)
        expectEqual(it.isOccupied, contents[bucket.offset] != nil)
        expectEqual(it.currentValue, contents[bucket.offset])
        withEvery("iteration", in: 0 ..< 2) { iteration in
          var c = hashTable.bucketCount
          if iteration == 1 { c /= 2 } // Don't let advance() trap
          withEvery("offset", in: 0 ..< c) { offset in
            let expected = contents[offset]
            expectEqual(it.currentBucket, Bucket(offset: offset))
            expectEqual(it.isOccupied, expected != nil)
            expectEqual(it.currentValue, expected)
            it.advance()
          }
        }
      }
    }
  }

  func test_bucketIterator_read_with_holes() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withEvery("chainLength", in: [1, 2, 3, 4]) { chainLength in
        withEvery("holeLength", in: [1, 2, 3]) { holeLength in
          withEvery("startOffset", in: 0 ... chainLength) { startOffset in
            withSampleHashTable(
              scale: scale,
              chainLength: chainLength,
              holeLength: holeLength,
              startingAt: startOffset
            ) { hashTable, contents in
              // Start an iterator at the first bucket and cycle through
              // all buckets twice, checking the values we see at every step.
              let bucket = Bucket(offset: 0)
              var it = hashTable.bucketIterator(startingAt: bucket)
              expectEqual(it.currentBucket, bucket)
              expectEqual(it.isOccupied, contents[bucket.offset] != nil)
              expectEqual(it.currentValue, contents[bucket.offset])
              withEvery("iteration", in: 0 ..< 2) { iteration in
                var c = hashTable.bucketCount
                if iteration == 1 { c /= 2 } // Don't let advance() trap
                withEvery("offset", in: 0 ..< c) { offset in
                  let expected = contents[offset]
                  expectEqual(it.currentBucket, Bucket(offset: offset))
                  expectEqual(it.isOccupied, expected != nil)
                  expectEqual(it.currentValue, expected)
                  it.advance()
                }
              }
            }
          }
        }
      }
    }
  }

  func test_bucketIterator_start_in_middle() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withSampleHashTable(scale: scale) { hashTable, contents in
        var expected = hashTable.bucketIterator(startingAt: Bucket(offset: 0))
        // Start a new iterator at every bucket and check that its initial
        // state is match exactly what we'd get by starting at the first
        // bucket and advancing step by step to the same position.
        withEvery("start", in: 0 ..< hashTable.bucketCount) { start in
          let bucket = Bucket(offset: start)
          let actual = hashTable.bucketIterator(startingAt: bucket)

          expectEqual(expected.currentBucket, bucket)
          expectEqual(expected.currentValue, contents[start])

          expectEqual(actual.currentBucket, bucket)
          expectEqual(actual.currentValue, contents[start])

          expectEqual(actual._currentBucket, expected._currentBucket)
          expectEqual(actual._currentRawValue, expected._currentRawValue)
          expectEqual(actual._nextBits, expected._nextBits)
          expectEqual(actual._remainingBitCount, expected._remainingBitCount)

          expected.advance()
        }
      }
    }
  }

  func test_bucketIterator_start_in_middle_with_holes() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withEvery("chainLength", in: [1, 2, 3, 4]) { chainLength in
        withEvery("holeLength", in: [1, 2, 3]) { holeLength in
          withEvery("startOffset", in: 0 ... chainLength) { startOffset in
            withSampleHashTable(
              scale: scale,
              chainLength: chainLength,
              holeLength: holeLength,
              startingAt: startOffset
            ) { hashTable, contents in
              var expected = hashTable.bucketIterator(startingAt: Bucket(offset: 0))
              // Start a new iterator at every bucket and check that its initial
              // state is match exactly what we'd get by starting at the first
              // bucket and advancing step by step to the same position.
              withEvery("start", in: 0 ..< hashTable.bucketCount) { start in
                let bucket = Bucket(offset: start)
                let actual = hashTable.bucketIterator(startingAt: bucket)

                expectEqual(expected.currentBucket, bucket)
                expectEqual(expected.currentValue, contents[start])

                expectEqual(actual.currentBucket, bucket)
                expectEqual(actual.currentValue, contents[start])

                expectEqual(actual._currentBucket, expected._currentBucket)
                expectEqual(actual._currentRawValue, expected._currentRawValue)
                expectEqual(actual._nextBits, expected._nextBits)
                expectEqual(actual._remainingBitCount, expected._remainingBitCount)

                expected.advance()
              }
            }
          }
        }
      }
    }
  }

  func test_bucketIterator_advance_until() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withEvery("chainLength", in: [1, 2, 3, 4]) { chainLength in
        withEvery("holeLength", in: [1, 2, 3]) { holeLength in
          withEvery("startOffset", in: 0 ... chainLength) { startOffset in
            withSampleHashTable(
              scale: scale,
              chainLength: chainLength,
              holeLength: holeLength,
              startingAt: startOffset
            ) { hashTable, contents in
              var offset = 0
              var it = hashTable.bucketIterator(startingAt: Bucket(offset: offset))
              let step = 2

              for _ in 0 ..< hashTable.bucketCount {
                if let value = contents[offset] {
                  // `advance(until:)` should not move if we're already on the right value.
                  it.advance(until: value)
                  expectEqual(it.currentBucket.offset, offset)

                  // `advance(until:)` should stop when it find the value we're looking for,
                  // or at the next hole, whichever comes first.
                  it.advance(until: value + step)
                  expectTrue(it.currentValue == nil || it.currentValue == value + step)
                  var o = offset
                  while true {
                    o += 1
                    if o == contents.count { o = 0 }
                    if o == it.currentBucket.offset { break }
                    expectNotNil(contents[o])
                  }
                  offset = o
                } else {
                  // `advance(until:)` should not move if we're on a hole.
                  it.advance(until: 0)
                  expectEqual(it.currentBucket.offset, offset)

                  // Move to the next bucket.
                  offset += 1
                  if offset == contents.count { offset = 0 }
                  it.advance()
                }
              }
            }
          }
        }
      }
    }
  }

  func test_bucketIterator_advance_toNextUnoccupiedBucket() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withEvery("chainLength", in: [1, 2, 3, 4]) { chainLength in
        withEvery("holeLength", in: [1, 2, 3]) { holeLength in
          withEvery("startOffset", in: 0 ... chainLength) { startOffset in
            withSampleHashTable(
              scale: scale,
              chainLength: chainLength,
              holeLength: holeLength,
              startingAt: startOffset
            ) { hashTable, contents in
              var offset = 0
              var it = hashTable.bucketIterator(startingAt: Bucket(offset: offset))

              let holeCount = contents.reduce(into: 0) { count, item in
                if item == nil { count += 1 }
              }

              for _ in 0 ..< holeCount * 3 / 2 {
                if contents[offset] != nil {
                  it.advanceToNextUnoccupiedBucket()

                  expectNil(it.currentValue)

                  while contents[offset] != nil {
                    offset += 1
                    if offset == contents.count { offset = 0 }
                  }
                  expectEqual(it.currentBucket.offset, offset)
                } else {
                  // `advanceToNextUnoccupiedBucket()` should not move if we're already on a hole.
                  it.advanceToNextUnoccupiedBucket()
                  expectEqual(it.currentBucket.offset, offset)

                  // Move to the next bucket.
                  offset += 1
                  if offset == contents.count { offset = 0 }
                  it.advance()
                }
              }
            }
          }
        }
      }
    }
  }

  func test_bucketIterator_write() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      withSampleHashTable(scale: scale) { hashTable, contents in
        let storage = _HashTable(scale: scale)
        let bucketCount = 1 << scale
        let contents: [Int?] = Array(0 ..< bucketCount - 1) + [nil]
        storage.update { hashTable in
          // Start an iterator at the first bucket and cycle through
          // all buckets, updating values at every step.
          let bucket = Bucket(offset: 0)
          var it = hashTable.bucketIterator(startingAt: bucket)
          withEvery("writeOffset", in: 0 ..< contents.count) { writeOffset in
            it.currentValue = contents[writeOffset]
            it.advance()
          }
          // Cycle through one more time, verifying that contents match.
          it = hashTable.bucketIterator(startingAt: bucket)
          withEvery("readOffset", in: 0 ..< contents.count) { readOffset in
            expectEqual(it.currentValue, contents[readOffset])
            it.advance()
          }
        }
      }
    }
  }

  func test_Storage_copy() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      let (s1, contents) = sampleTable(scale: scale)
      let s2 = s1.copy()
      s1.read { original in
        s2.read { copy in
          expectEqual(copy.scale, original.scale)
          expectEqual(copy.reservedScale, original.reservedScale)
          expectEqual(copy.seed, original.seed)
          var it = copy.bucketIterator(startingAt: Bucket(offset: 0))
          withEvery("offset", in: 0 ..< contents.count) { offset in
            expectEqual(it.currentValue, contents[offset])
            it.advance()
          }
        }
      }
    }
  }

  func test_bias() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      let (storage, contents) = sampleTable(scale: scale)
      withEvery("bias", in: [0, 1, -1, 2, -2, 30, -30]) { bias in
        storage.update { hashTable in hashTable.bias = bias }
        storage.read { hashTable in
          var it = hashTable.bucketIterator(startingAt: Bucket(offset: 0))
          repeat {
            if let v = contents[it.currentBucket.offset] {
              let m = hashTable.bucketCount - 1
              var expected = (v + bias) % m
              if expected < 0 { expected += m }
              expectEqual(it.currentValue, expected)
            } else {
              expectNil(it.currentValue)
            }
            it.advance()
          } while it.currentBucket.offset != 0
        }
      }
    }
  }

  func test_rawSubscript() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      let (storage, contents) = sampleTable(scale: scale)
      storage.read { hashTable in
        var bucket = Bucket(offset: 0)
        repeat {
          let expected = hashTable._bucketContents(for: contents[bucket.offset])
          expectEqual(hashTable[raw: bucket], expected)
          bucket = hashTable.bucket(after: bucket)
        } while bucket.offset != 0
      }
    }
  }

  func test_isOccupied() {
    withEvery("scale", in: [5, 6, 7, 8, 12]) { scale in
      let (storage, contents) = sampleTable(scale: scale)
      storage.read { hashTable in
        var bucket = Bucket(offset: 0)
        repeat {
          expectEqual(hashTable.isOccupied(bucket), contents[bucket.offset] != nil)
          bucket = hashTable.bucket(after: bucket)
        } while bucket.offset != 0
      }
    }
  }

}

#endif
