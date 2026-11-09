//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension TreeDictionary {
  /// True if consistency checking is enabled in the implementation of this
  /// type, false otherwise.
  ///
  /// Documented performance promises are null and void when this property
  /// returns true -- for example, operations that are documented to take
  /// O(1) time might take O(*n*) time, or worse.
  public static var _isConsistencyCheckingEnabled: Bool {
    _isCollectionsInternalCheckingEnabled
  }

  @inlinable
  public func _invariantCheck() {
    _root._fullInvariantCheck()
  }

  public func _dump(iterationOrder: Bool = false) {
    _root.dump(iterationOrder: iterationOrder)
  }

  public static var _maxDepth: Int {
    _HashLevel.limit
  }

  public var _statistics: _HashTreeStatistics {
    var stats = _HashTreeStatistics()
    _root.gatherStatistics(.top, &stats)
    return stats
  }
}
