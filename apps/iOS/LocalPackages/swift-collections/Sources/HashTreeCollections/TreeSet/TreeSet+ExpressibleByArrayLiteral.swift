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

extension TreeSet: ExpressibleByArrayLiteral {
  /// Creates a new set from the contents of an array literal.
  ///
  /// Duplicate elements in the literal are allowed, but the resulting
  /// persistent set will only contain the first occurrence of each.
  ///
  /// Do not call this initializer directly. It is used by the compiler when
  /// you use an array literal. Instead, create a new persistent set using an
  /// array literal as its value by enclosing a comma-separated list of values
  /// in square brackets. You can use an array literal anywhere a persistent set
  /// is expected by the type context.
  ///
  /// Like the standard `Set`, persistent sets do not preserve the order of
  /// elements inside the array literal.
  ///
  /// - Parameter elements: A variadic list of elements of the new set.
  ///
  /// - Complexity: O(`elements.count`) if `Element` properly implements
  ///    hashing.
  public init(arrayLiteral elements: Element...) {
    self.init(elements)
  }
}
