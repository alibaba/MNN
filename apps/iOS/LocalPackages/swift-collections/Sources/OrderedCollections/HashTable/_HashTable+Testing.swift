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

extension _HashTable.Bucket: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String { "Bucket(@\(offset))"}
}

extension _UnsafeHashTable {
  internal func debugOccupiedCount() -> Int {
    var count = 0
    var it = bucketIterator(startingAt: Bucket(offset: 0))
    repeat {
      if it.isOccupied {
        count += 1
      }
      it.advance()
    } while it.currentBucket.offset != 0
    return count
  }

  internal func debugLoadFactor() -> Double {
    return Double(debugOccupiedCount()) / Double(bucketCount)
  }

  internal func debugContents() -> [Int?] {
    var result: [Int?] = []
    result.reserveCapacity(bucketCount)
    var it = bucketIterator(startingAt: Bucket(offset: 0))
    repeat {
      result.append(it.currentValue)
      it.advance()
    } while it.currentBucket.offset != 0
    return result
  }
}

extension _UnsafeHashTable.BucketIterator: CustomStringConvertible {
  @usableFromInline
  var description: String {
    func pad(_ s: String, to length: Int, by padding: Character = " ") -> String {
      let c = s.count
      guard c < length else { return s }
      return String(repeating: padding, count: length - c) + s
    }
    let offset = pad(String(_currentBucket.offset), to: 4)
    let value: String
    if let v = currentValue {
      value = pad(String(v), to: 4)
    } else {
      value = " nil"
    }
    let remainingBits = pad(String(_nextBits, radix: 2), to: _remainingBitCount, by: "0")
    return "BucketIterator(scale: \(_scale), bucket: \(offset), value: \(value), bits: \(remainingBits) (\(_remainingBitCount) bits))"
  }
}
