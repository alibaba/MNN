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
  @inline(never)
  @_effects(releasenone)
  public func _checkInvariants() {
    //let actualCount = _storage.reduce(into: 0) { $0 += $1.count }
    //precondition(_count == actualCount, "Invalid count")

    precondition(_storage.isEmpty || !_storage.last!.isEmpty,
                 "Extraneous tail slot")
  }
#else
  @inline(__always) @inlinable
  public func _checkInvariants() {}
#endif // COLLECTIONS_INTERNAL_CHECKS
}
