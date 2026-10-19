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

// Loosely adapted from https://github.com/apple/swift/tree/main/stdlib/private/StdlibUnittest

public class ResettableValue<Value> {
  public init(_ value: Value) {
    self.defaultValue = value
    self.value = value
  }

  public func reset() {
    value = defaultValue
  }

  public let defaultValue: Value
  public var value: Value
}

extension ResettableValue where Value: Strideable {
  public func increment(by delta: Value.Stride = 1) {
    value = value.advanced(by: delta)
  }
}

extension ResettableValue: CustomStringConvertible {
  public var description: String { "\(value)" }
}
