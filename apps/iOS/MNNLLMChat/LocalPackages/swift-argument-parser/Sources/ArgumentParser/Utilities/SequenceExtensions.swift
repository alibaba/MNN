//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Sequence where Element: Hashable {
  /// Returns an array with only the unique elements of this sequence, in the
  /// order of the first occurrence of each unique element.
  func uniquing() -> [Element] {
    var seen = Set<Element>()
    return self.filter { seen.insert($0).0 }
  }

  /// Returns an array, collapsing runs of consecutive equal elements into
  /// the first element of each run.
  ///
  ///     [1, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1].uniquingAdjacentElements()
  ///     // [1, 2, 3, 2, 1]
  func uniquingAdjacentElements() -> [Element] {
    var iterator = makeIterator()
    guard let first = iterator.next()
      else { return [] }
    
    var result = [first]
    while let element = iterator.next() {
      if result.last != element {
        result.append(element)
      }
    }
    return result
  }
}
