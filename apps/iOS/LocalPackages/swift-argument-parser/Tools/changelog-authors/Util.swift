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

// MARK: Helpers

extension Sequence {
  func uniqued<T: Hashable>(by transform: (Element) throws -> T) rethrows -> [Element] {
    var seen: Set<T> = []
    var result: [Element] = []
    
    for element in self {
      if try seen.insert(transform(element)).inserted {
        result.append(element)
      }
    }
    return result
  }
}
