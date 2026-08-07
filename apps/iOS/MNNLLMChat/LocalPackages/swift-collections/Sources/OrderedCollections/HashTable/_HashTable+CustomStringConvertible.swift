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

// These are primarily for debugging.

extension _HashTable.Header: CustomStringConvertible {
  @usableFromInline
  internal var _description: String {
    "(scale: \(scale), reservedScale: \(reservedScale), bias: \(bias), seed: \(String(seed, radix: 16)))"
  }

  @usableFromInline
  internal var description: String {
    "_HashTable.Header\(_description)"
  }
}

extension _HashTable.UnsafeHandle: CustomStringConvertible {
  internal func _description(type: String) -> String {
    var d = """
      \(type)\(_header.pointee._description)
        load factor: \(debugLoadFactor())
      """
    if bucketCount < 128 {
      d += "\n  "
      d += debugContents()
        .lazy
        .map { $0 == nil ? "_" : "\($0!)" }
        .joined(separator: " ")
    }
    return d
  }

  @usableFromInline
  internal var description: String {
    _description(type: "_HashTable.UnsafeHandle")
  }
}

extension _HashTable: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    read { $0._description(type: "_HashTable") }
  }
}

extension _HashTable.Storage: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    _HashTable(self).read { $0._description(type: "_HashTable.Storage") }
  }
}

