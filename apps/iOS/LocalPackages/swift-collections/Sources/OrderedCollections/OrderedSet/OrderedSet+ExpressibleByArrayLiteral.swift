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

extension OrderedSet: ExpressibleByArrayLiteral {
  /// Creates a new ordered set from the contents of an array literal.
  ///
  /// Duplicate elements in the literal are allowed, but the resulting ordered
  /// set will only contain the first occurrence of each.
  ///
  /// Do not call this initializer directly. It is used by the compiler when
  /// you use an array literal. Instead, create a new ordered set using an array
  /// literal as its value by enclosing a comma-separated list of values in
  /// square brackets. You can use an array literal anywhere an ordered set is
  /// expected by the type context.
  ///
  /// - Parameter elements: A variadic list of elements of the new ordered set.
  ///
  /// - Complexity: O(`elements.count`) if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public init(arrayLiteral elements: Element...) {
    self.init(elements)
  }
}
