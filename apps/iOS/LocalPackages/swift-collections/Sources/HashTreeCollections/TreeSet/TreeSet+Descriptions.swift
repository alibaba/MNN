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

extension TreeSet: CustomStringConvertible {
  /// A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: self)
  }
}

extension TreeSet: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}
