//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension BitArray: ExpressibleByStringLiteral {
  /// Creates an instance initialized with the given elements.
  @inlinable
  public init(stringLiteral value: String) {
    guard let bits = Self(value) else {
      fatalError("Invalid bit array literal")
    }
    self = bits
  }
}
