//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2019 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension TreeDictionary: Hashable where Value: Hashable {
  /// Hashes the essential components of this value by feeding them into the
  /// given hasher.
  ///
  /// Complexity: O(`count`)
  @inlinable
  public func hash(into hasher: inout Hasher) {
    var commutativeHash = 0
    for (key, value) in self {
      var elementHasher = hasher
      elementHasher.combine(key)
      elementHasher.combine(value)
      commutativeHash ^= elementHasher.finalize()
    }
    hasher.combine(commutativeHash)
  }
}
