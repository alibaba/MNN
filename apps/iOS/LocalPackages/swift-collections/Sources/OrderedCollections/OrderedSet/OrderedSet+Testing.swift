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

extension OrderedSet._UnstableInternals {
  @_spi(Testing) public var capacity: Int { base._capacity }
  @_spi(Testing) public var minimumCapacity: Int { base._minimumCapacity }
  @_spi(Testing) public var scale: Int { base._scale }
  @_spi(Testing) public var reservedScale: Int { base._reservedScale }
  @_spi(Testing) public var bias: Int { base._bias }

  public static var isConsistencyCheckingEnabled: Bool {
    _isCollectionsInternalCheckingEnabled
  }
}

extension OrderedSet {
  @_spi(Testing)
  @_alwaysEmitIntoClient
  public static var _minimumScale: Int {
    _HashTable.minimumScale
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public static func _minimumCapacity(forScale scale: Int) -> Int {
    _HashTable.minimumCapacity(forScale: scale)
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public static func _maximumCapacity(forScale scale: Int) -> Int {
    _HashTable.maximumCapacity(forScale: scale)
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public static func _scale(forCapacity capacity: Int) -> Int {
    _HashTable.scale(forCapacity: capacity)
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public static func _biasRange(scale: Int) -> Range<Int> {
    guard scale != 0 else { return Range(uncheckedBounds: (0, 1)) }
    return Range(uncheckedBounds: (0, (1 &<< scale) - 1))
  }
}

extension OrderedSet._UnstableInternals {
  @_spi(Testing)
  @_alwaysEmitIntoClient
  public var hasHashTable: Bool { base._table != nil }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public var hashTableIdentity: ObjectIdentifier? {
    guard let storage = base.__storage else { return nil }
    return ObjectIdentifier(storage)
  }

  @_spi(Testing)
  public var hashTableContents: [Int?] {
    guard let table = base._table else { return [] }
    return table.read { hashTable in
      hashTable.debugContents()
    }
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  mutating public func _regenerateHashTable(bias: Int) {
    base._ensureUnique()
    let new = base._table!.copy()
    base._table!.read { source in
      new.update { target in
        target.bias = bias
        var it = source.bucketIterator(startingAt: _Bucket(offset: 0))
        repeat {
          target[it.currentBucket] = it.currentValue
          it.advance()
        } while it.currentBucket.offset != 0
      }
    }
    base._table = new
    base._checkInvariants()
  }

  @_spi(Testing)
  @_alwaysEmitIntoClient
  public mutating func reserveCapacity(
    _ minimumCapacity: Int,
    persistent: Bool
  ) {
    base._reserveCapacity(minimumCapacity, persistent: persistent)
    base._checkInvariants()
  }
}

extension OrderedSet {
  @_spi(Testing)
  public init(
    _scale scale: Int,
    bias: Int,
    contents: some Sequence<Element>
  ) {
    let contents = ContiguousArray(contents)
    precondition(scale >= _HashTable.scale(forCapacity: contents.count))
    precondition(scale <= _HashTable.maximumScale)
    precondition(bias >= 0 && Self._biasRange(scale: scale).contains(bias))
    precondition(scale >= _HashTable.minimumScale || bias == 0)
    let table = _HashTable(scale: Swift.max(scale, _HashTable.minimumScale))
    table.header.bias = bias
    let (success, index) = table.update { hashTable in
      hashTable.fill(untilFirstDuplicateIn: contents)
    }
    precondition(success, "Duplicate element at index \(index)")
    self.init(
      _uniqueElements: contents,
      scale < _HashTable.minimumScale ? nil : table)
    precondition(self._scale == scale)
    precondition(self._bias == bias)
    _checkInvariants()
  }
}
