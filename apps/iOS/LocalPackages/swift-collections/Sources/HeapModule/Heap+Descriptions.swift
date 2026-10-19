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

extension Heap: CustomStringConvertible {
  /// A textual representation of this instance.
  public var description: String {
    "<\(count) item\(count == 1 ? "" : "s") @\(_idString)>"
  }

  internal var _idString: String {
    // "<32 items @0x2787abcf>"
    _storage.withUnsafeBytes {
      guard let p = $0.baseAddress else {
        return "nil"
      }
      return String(UInt(bitPattern: p), radix: 16)
    }
  }
}

extension Heap: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

