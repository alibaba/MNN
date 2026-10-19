//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

public struct StringConvertibleValue {
  public var value: Int
  public init(_ value: Int) { self.value = value }
}

extension StringConvertibleValue: ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Int) {
    self.init(value)
  }
}

extension StringConvertibleValue: CustomStringConvertible {
  public var description: String {
    "description(\(value))"
  }
}

extension StringConvertibleValue: CustomDebugStringConvertible {
  public var debugDescription: String {
    "debugDescription(\(value))"
  }
}
