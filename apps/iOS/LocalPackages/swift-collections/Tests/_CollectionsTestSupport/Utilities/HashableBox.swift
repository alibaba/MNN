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

public final class HashableBox<T: Hashable>: Hashable {
  public init(_ value: T) { self.value = value }
  public var value: T

  public static func ==(left: HashableBox, right: HashableBox) -> Bool {
    left.value == right.value
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(value)
  }
}
