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

extension Heap {
  /// True if consistency checking is enabled in the implementation of this
  /// type, false otherwise.
  ///
  /// Documented performance promises are null and void when this property
  /// returns true -- for example, operations that are documented to take
  /// O(1) time might take O(*n*) time, or worse.
  public static var _isConsistencyCheckingEnabled: Bool {
    _isCollectionsInternalCheckingEnabled
  }

  #if COLLECTIONS_INTERNAL_CHECKS
  /// Visits each item in the heap in depth-first order, verifying that the
  /// contents satisfy the min-max heap property.
  @inlinable
  @inline(never)
  internal func _checkInvariants() {
    guard count > 1 else { return }
    _checkInvariants(node: .root, min: nil, max: nil)
  }

  @inlinable
  internal func _checkInvariants(node: _HeapNode, min: Element?, max: Element?) {
    let value = _storage[node.offset]
    if let min = min {
      precondition(value >= min,
                   "Element \(value) at \(node) is less than min \(min)")
    }
    if let max = max {
      precondition(value <= max,
                   "Element \(value) at \(node) is greater than max \(max)")
    }
    let left = node.leftChild()
    let right = node.rightChild()
    if node.isMinLevel {
      if left.offset < count {
        _checkInvariants(node: left, min: value, max: max)
      }
      if right.offset < count {
        _checkInvariants(node: right, min: value, max: max)
      }
    } else {
      if left.offset < count {
        _checkInvariants(node: left, min: min, max: value)
      }
      if right.offset < count {
        _checkInvariants(node: right, min: min, max: value)
      }
    }
  }
  #else
  @inlinable
  @inline(__always)
  public func _checkInvariants() {}
  #endif  // COLLECTIONS_INTERNAL_CHECKS
}
