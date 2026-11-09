//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2019 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension TreeDictionary: ExpressibleByDictionaryLiteral {
  /// Creates a new dictionary from the contents of a dictionary literal.
  ///
  /// The literal must not contain duplicate elements.
  ///
  /// Do not call this initializer directly. It is used by the compiler when
  /// you use a dictionary literal. Instead, create a new dictionary using a
  /// dictionary literal as its value by enclosing a comma-separated list of
  /// values in square brackets. You can use a dictionary literal anywhere a
  /// persistent dictionary is expected by the type context.
  ///
  /// Like the standard `Dictionary`, persistent dictionaries do not preserve
  /// the order of elements inside the array literal.
  ///
  /// - Parameter elements: A variadic list of elements of the new set.
  ///
  /// - Complexity: O(`elements.count`) if `Element` properly implements
  ///    hashing.
  @inlinable
  @inline(__always)
  public init(dictionaryLiteral elements: (Key, Value)...) {
    self.init(uniqueKeysWithValues: elements)
  }
}
