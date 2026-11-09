//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Collection {
  func mapEmpty(_ replacement: () -> Self) -> Self {
    isEmpty ? replacement() : self
  }
}

extension MutableCollection {
  mutating func withEach(_ body: (inout Element) throws -> Void) rethrows {
    var i = startIndex
    while i < endIndex {
      try body(&self[i])
      formIndex(after: &i)
    }
  }
}
