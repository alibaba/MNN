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

/// A test protocol for validating that set-like types implement users'
/// baseline expectations.
///
/// To ensure maximum utility, this protocol refines neither `Collection` nor
/// `SetAlgebra` although it does share some of the same requirements.
public protocol SetAPIChecker {
  associatedtype Element
  associatedtype Index

  var isEmpty: Bool { get }
  var count: Int { get }

  init()

  mutating func remove(at index: Index) -> Element

  func filter(_ isIncluded: (Element) throws -> Bool) rethrows -> Self

  func isSubset<S: Sequence>(of other: S) -> Bool
  where S.Element == Element

  func isSuperset<S: Sequence>(of other: S) -> Bool
  where S.Element == Element

  func isStrictSubset<S: Sequence>(of other: S) -> Bool
  where S.Element == Element

  func isStrictSuperset<S: Sequence>(of other: S) -> Bool
  where S.Element == Element

  func isDisjoint<S: Sequence>(with other: S) -> Bool
  where S.Element == Element


  func intersection<S: Sequence>(_ other: S) -> Self
  where S.Element == Element

  func union<S: Sequence>(_ other: __owned S) -> Self
  where S.Element == Element

  __consuming func subtracting<S: Sequence>(_ other: S) -> Self
  where S.Element == Element

  func symmetricDifference<S: Sequence>(_ other: __owned S) -> Self
  where S.Element == Element

  mutating func formIntersection<S: Sequence>(_ other: S)
  where S.Element == Element

  mutating func formUnion<S: Sequence>(_ other: __owned S)
  where S.Element == Element

  mutating func subtract<S: Sequence>(_ other: S)
  where S.Element == Element

  mutating func formSymmetricDifference<S: Sequence>(_ other: __owned S)
  where S.Element == Element
}

extension Set: SetAPIChecker {}

public protocol SetAPIExtras: SetAPIChecker {
  // Non-standard extensions

  mutating func update(_ member: Element, at index: Index) -> Element

  func isEqualSet(to other: Self) -> Bool
  func isEqualSet<S: Sequence>(to other: S) -> Bool
  where S.Element == Element
}
