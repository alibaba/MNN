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

extension OrderedDictionary: ExpressibleByDictionaryLiteral {
  /// Creates a new ordered dictionary from the contents of a dictionary
  /// literal.
  ///
  /// Do not call this initializer directly. It is used by the compiler when you
  /// use a dictionary literal. Instead, create a new ordered dictionary using a
  /// dictionary literal as its value by enclosing a comma-separated list of
  /// key-value pairs in square brackets. You can use a dictionary literal
  /// anywhere an ordered dictionary is expected by the type context.
  ///
  /// - Parameter elements: A variadic list of key-value pairs for the new
  ///    ordered dictionary.
  ///
  /// - Complexity: O(`elements.count`) if `Key` implements
  ///    high-quality hashing.
  @inlinable
  public init(dictionaryLiteral elements: (Key, Value)...) {
    self.init(uniqueKeysWithValues: elements)
  }
}
