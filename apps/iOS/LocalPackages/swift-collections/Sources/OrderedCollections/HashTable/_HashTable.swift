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

@usableFromInline
@frozen
internal struct _HashTable {
  @usableFromInline
  internal var _storage: Storage

  @inlinable
  @inline(__always)
  internal init(_ storage: Storage) {
    _storage = storage
  }
}

extension _HashTable {
  /// A class holding hash table storage for a `OrderedSet` collection.
  /// Values in the hash table are offsets into separate element storage, so
  /// this class doesn't need to be generic over `OrderedSet`'s `Element` type.
  @usableFromInline
  internal final class Storage
  : ManagedBuffer<Header, UInt64>
  {}
}

extension _HashTable {
  /// Allocate a new empty hash table buffer of the specified scale.
  @usableFromInline
  @_effects(releasenone)
  internal init(scale: Int, reservedScale: Int = 0) {
    assert(scale >= Self.minimumScale && scale <= Self.maximumScale)
    let wordCount = Self.wordCount(forScale: scale)
    let storage = Storage.create(
      minimumCapacity: wordCount,
      makingHeaderWith: { object in
        #if COLLECTIONS_DETERMINISTIC_HASHING
        let seed = scale << 6
        #else
        let seed = Int(bitPattern: Unmanaged.passUnretained(object).toOpaque())
        #endif
        return Header(scale: scale, reservedScale: reservedScale, seed: seed)
      })
    storage.withUnsafeMutablePointerToElements { elements in
      elements.initialize(repeating: 0, count: wordCount)
    }
    self.init(unsafeDowncast(storage, to: Storage.self))
  }

  /// Populate a new hash table with data from `elements`.
  ///
  /// - Parameter scale: The desired hash table scale or nil to use the minimum scale that satisfies invariants.
  /// - Parameter reservedScale: The reserved scale to remember in the returned storage.
  /// - Parameter duplicates: The strategy to use to handle duplicate items.
  /// - Returns: `(storage, index)` where `storage` is a storage instance. The contents of `storage` reflects all elements in `contents[contents.startIndex ..< index]`. `index` is usually `contents.endIndex`, except when the function was asked to reject duplicates, in which case `index` addresses the first duplicate element in `contents` (if any).
  @inlinable
  @inline(never)
  @_effects(releasenone)
  static func create<C: RandomAccessCollection>(
    uncheckedUniqueElements elements: C,
    scale: Int? = nil,
    reservedScale: Int = 0
  ) -> _HashTable?
  where C.Element: Hashable {
    let minScale = Self.scale(forCapacity: elements.count)
    let scale = Swift.max(Swift.max(scale ?? 0, minScale),
                          reservedScale)
    if scale < Self.minimumScale { return nil }
    let hashTable = Self(scale: scale, reservedScale: reservedScale)
    hashTable.update { handle in
      handle.fill(uncheckedUniqueElements: elements)
    }
    return hashTable
  }

  /// Populate a new hash table with data from `elements`.
  ///
  /// - Parameter scale: The desired hash table scale or nil to use the minimum scale that satisfies invariants.
  /// - Parameter reservedScale: The reserved scale to remember in the returned storage.
  /// - Parameter duplicates: The strategy to use to handle duplicate items.
  /// - Returns: `(storage, index)` where `storage` is a storage instance. The contents of `storage` reflects all elements in `contents[contents.startIndex ..< index]`. `index` is usually `contents.endIndex`, except when the function was asked to reject duplicates, in which case `index` addresses the first duplicate element in `contents` (if any).
  @inlinable
  @inline(never)
  @_effects(releasenone)
  static func create<C: RandomAccessCollection>(
    untilFirstDuplicateIn elements: C,
    scale: Int? = nil,
    reservedScale: Int = 0
  ) -> (hashTable: _HashTable?, end: C.Index)
  where C.Element: Hashable {
    let minScale = Self.scale(forCapacity: elements.count)
    let scale = Swift.max(Swift.max(scale ?? 0, minScale),
                          reservedScale)

    if scale < Self.minimumScale {
      // Don't hash anything.
      if elements.count < 2 { return (nil, elements.endIndex) }
      // fast path that doesn't allocate per element if _read accessor can't
      // be inlined because this function doesn't get specialized e.g.
      // if `Element` isn't known at compile time.
      let firstDuplicateIndexFastPath: Int? = elements.withContiguousStorageIfAvailable { elements in
        var temp: ContiguousArray<C.Element> = []
        temp.reserveCapacity(elements.count)
        for i in elements.indices {
          let item = elements[i]
          guard !temp._contains(item) else { return i }
          temp.append(item)
        }
        return elements.endIndex
      }
      if let firstDuplicateIndexFastPath {
        return (nil, elements.index(elements.startIndex, offsetBy: firstDuplicateIndexFastPath))
      }
      var temp: ContiguousArray<C.Element> = []
      temp.reserveCapacity(elements.count)
      for i in elements.indices {
        let item = elements[i]
        guard !temp._contains(item) else { return (nil, i) }
        temp.append(item)
      }
      return (nil, elements.endIndex)
    }
    let hashTable = Self(scale: scale, reservedScale: reservedScale)
    let (_, index) = hashTable.update { handle in
      handle.fill(untilFirstDuplicateIn: elements)
    }
    return (hashTable, index)
  }

  /// Create and return a new copy of this instance. The result has the same
  /// scale and seed, and contains the exact same bucket data as the original instance.
  @usableFromInline
  @_effects(releasenone)
  internal func copy() -> _HashTable {
    self.read { handle in
      let wordCount = handle.wordCount
      let new = Storage.create(
        minimumCapacity: wordCount,
        makingHeaderWith: { _ in handle._header.pointee })
      new.withUnsafeMutablePointerToElements { elements in
        elements.initialize(from: handle._buckets, count: wordCount)
      }
      return Self(unsafeDowncast(new, to: Storage.self))
    }
  }
}

/// copy of the standard library implementation of `Sequence.contains(_:)`
/// to allow partial specialization if `Element` is not known at compile time.
/// also some modification to remove some more generic overhead.
extension ContiguousArray where Element: Equatable {
  @inlinable
  func _contains(_ element: Element) -> Bool {
    for index in 0..<count {
      if element == self[index] {
        return true
      }
    }
    return false
  }
}



extension _HashTable {
  /// Call `body` with a hash table handle suitable for read-only use.
  ///
  /// - Warning: The handle supplied to `body` is only valid for the duration of
  ///    the closure call. The closure must not escape it outside the call.
  @inlinable
  @inline(__always)
  internal func read<R>(_ body: (_UnsafeHashTable) throws -> R) rethrows -> R {
    try _storage.withUnsafeMutablePointers { header, elements in
      let handle = _UnsafeHashTable(header: header, buckets: elements, readonly: true)
      return try body(handle)
    }
  }

  /// Call `body` with a hash table handle suitable for mutating use.
  ///
  /// - Warning: The handle supplied to `body` is only valid for the duration of
  ///    the closure call. The closure must not escape it outside the call.
  @inlinable
  @inline(__always)
  internal func update<R>(_ body: (_UnsafeHashTable) throws -> R) rethrows -> R {
    try _storage.withUnsafeMutablePointers { header, elements in
      let handle = _UnsafeHashTable(header: header, buckets: elements, readonly: false)
      return try body(handle)
    }
  }
}

extension _HashTable {
  @inlinable
  internal var header: Header {
    get { _storage.header }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    nonmutating _modify { yield &_storage.header }
  }

  @inlinable
  internal var capacity: Int {
    _storage.header.capacity
  }

  @inlinable
  internal var minimumCapacity: Int {
    if scale == reservedScale { return 0 }
    return Self.minimumCapacity(forScale: scale)
  }

  @inlinable
  internal var scale: Int {
    _storage.header.scale
  }

  @inlinable
  internal var reservedScale: Int {
    _storage.header.reservedScale
  }

  @inlinable
  internal var bias: Int {
    _storage.header.bias
  }
}
