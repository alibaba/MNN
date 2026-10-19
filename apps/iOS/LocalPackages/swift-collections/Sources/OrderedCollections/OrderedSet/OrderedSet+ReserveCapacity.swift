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

extension OrderedSet {
  /// Creates an empty set with preallocated space for at least the
  /// specified number of elements.
  ///
  /// Use this initializer to avoid intermediate reallocations of a
  /// set's storage buffer when you know in advance how many elements
  /// you'll insert into the set after creation.
  ///
  /// If you have a good idea of the expected working size of the set, calling
  /// this initializer with `persistent` set to true can sometimes improve
  /// performance by eliminating churn due to repeated rehashings when the set
  /// temporarily shrinks below its regular size.
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the newly
  ///   created set should be able to store without reallocating its storage.
  ///
  /// - Parameter persistent: If set to true, prevent removals from shrinking
  ///   storage below the specified capacity. By default, removals are allowed
  ///   to shrink storage below any previously reserved capacity.
  ///
  /// - Complexity: O(`minimumCapacity`)
  @inlinable
  public init(minimumCapacity: Int, persistent: Bool = false) {
    self.init()
    self._reserveCapacity(minimumCapacity, persistent: persistent)
  }
}

extension OrderedSet {
  /// Reserves enough space to store the specified number of elements.
  ///
  /// This method ensures that the set has unique mutable storage, with space
  /// allocated for at least the requested number of elements.
  ///
  /// If you are adding a known number of elements to a set, call this method
  /// once before the first insertion to avoid multiple reallocations.
  ///
  /// Do not call this method in a loop -- it does not use an exponential
  /// allocation strategy, so doing that can result in quadratic instead of
  /// linear performance.
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the set
  ///   should be able to store without reallocating its storage.
  ///
  /// - Complexity: O(`max(count, minimumCapacity)`)
  @inlinable
  public mutating func reserveCapacity(_ minimumCapacity: Int) {
    self._reserveCapacity(minimumCapacity, persistent: false)
  }
}

extension OrderedSet {
  /// Reserves enough space to store the specified number of elements.
  ///
  /// This method ensures that the set has unique mutable storage, with space
  /// allocated for at least the requested number of elements.
  ///
  /// If you are adding a known number of elements to a set, call this method
  /// once before the first insertion to avoid multiple reallocations.
  ///
  /// Do not call this method in a loop -- it does not use an exponential
  /// allocation strategy, so doing that can result in quadratic instead of
  /// linear performance.
  ///
  /// If you have a good idea of the expected working size of the set, calling
  /// this method with `persistent` set to true can sometimes improve
  /// performance by eliminating churn due to repeated rehashings when the set
  /// temporarily shrinks below its regular size. You can cancel any capacity
  /// you've previously reserved by persistently reserving a capacity of zero.
  /// (This also shrinks the hash table to the ideal size for its current number
  /// elements.)
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the set
  ///   should be able to store without reallocating its storage.
  ///
  /// - Parameter persistent: If set to true, prevent removals from shrinking
  ///   storage below the specified capacity. By default, removals are allowed
  ///   to shrink storage below any previously reserved capacity.
  ///
  /// - Complexity: O(`max(count, minimumCapacity)`)
  @inlinable
  internal mutating func _reserveCapacity(
    _ minimumCapacity: Int,
    persistent: Bool
  ) {
    precondition(minimumCapacity >= 0, "Minimum capacity cannot be negative")
    defer { _checkInvariants() }

    _elements.reserveCapacity(minimumCapacity)

    let currentScale = _scale
    let newScale = _HashTable.scale(forCapacity: minimumCapacity)

    let reservedScale = persistent ? newScale : _reservedScale

    if currentScale < newScale {
      // Grow the table.
      _regenerateHashTable(scale: newScale, reservedScale: reservedScale)
      return
    }

    let requiredScale = _HashTable.scale(forCapacity: self.count)
    let minScale = Swift.max(Swift.max(newScale, reservedScale), requiredScale)
    if minScale < currentScale {
      // Shrink the table.
      _regenerateHashTable(scale: minScale, reservedScale: reservedScale)
      return
    }

    // When we have the right size table, ensure it's unique and it has the
    // right persisted reservation.
    _ensureUnique()
    if _reservedScale != reservedScale {
      // Remember reserved scale.
      __storage!.header.reservedScale = reservedScale
    }
  }
}
