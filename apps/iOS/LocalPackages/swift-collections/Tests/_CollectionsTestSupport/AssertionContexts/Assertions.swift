//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

import XCTest

public func expectFailure(
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  let message = message()
  XCTFail(
    TestContext.currentTrace(message),
    file: file, line: line)
  if trapping {
    fatalError(message, file: file, line: line)
  }
}

public func _expectFailure(
  _ diagnostic: String,
  _ message: () -> String,
  trapping: Bool,
  file: StaticString,
  line: UInt
) {
  let message = message()
  XCTFail(
    TestContext.currentTrace(
      """
      \(diagnostic)
      \(message)
      """),
    file: file, line: line)
  if trapping {
    fatalError(message, file: file, line: line)
  }
}

public func expectTrue(
  _ value: Bool,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if value { return }
  _expectFailure(
    "'\(value)' is not true",
    message, trapping: trapping, file: file, line: line)
}

public func expectFalse(
  _ value: Bool,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if !value { return }
  _expectFailure(
    "'\(value)' is not false",
    message, trapping: trapping, file: file, line: line)
}

public func expectNil<T>(
  _ value: Optional<T>,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if value == nil { return }
  _expectFailure(
    "'\(value!)' is not nil",
    message, trapping: trapping, file: file, line: line)
}

public func expectNotNil<T>(
  _ value: Optional<T>,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if value != nil { return }
  _expectFailure(
    "value is nil",
    message, trapping: trapping, file: file, line: line)
}

public func expectNotNil<T>(
  _ value: Optional<T>,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line,
  _ handler: (T) throws -> Void = { _ in }
) rethrows {
  if let value = value {
    try handler(value)
    return
  }
  _expectFailure(
    "value is nil",
    message, trapping: trapping, file: file, line: line)
}

public func expectIdentical<T: AnyObject>(
  _ left: T?, _ right: T?,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left === right { return }
  let l = left.map { "\($0)" } ?? "nil"
  let r = right.map { "\($0)" } ?? "nil"
  _expectFailure(
    "'\(l)' is not identical to '\(r)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectNotIdentical<T: AnyObject>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left !== right { return }
  _expectFailure(
    "'\(left)' is identical to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectEquivalent<A, B>(
  _ left: A, _ right: B,
  by areEquivalent: (A, B) -> Bool,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if areEquivalent(left, right) { return }
  _expectFailure(
    "'\(left)' is not equivalent to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectEquivalent<A, B>(
  _ left: A?, _ right: B?,
  by areEquivalent: (A, B) -> Bool,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if let left = left, let right = right, areEquivalent(left, right) { return }
  let l = left.map { "\($0)" } ?? "nil"
  let r = right.map { "\($0)" } ?? "nil"
  _expectFailure(
    "'\(l)' is not equivalent to '\(r)'",
    message, trapping: trapping, file: file, line: line)
}


public func expectEqual<T: Equatable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left == right { return }
  _expectFailure(
    "'\(left)' is not equal to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectEqual<Key: Equatable, Value: Equatable>(
  _ left: (key: Key, value: Value), _ right: (key: Key, value: Value),
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left == right { return }
  _expectFailure(
    "'\(left)' is not equal to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}


public func expectEqual<T: Equatable>(
  _ left: T?, _ right: T?,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left == right { return }
  let l = left.map { "\($0)" } ?? "nil"
  let r = right.map { "\($0)" } ?? "nil"
  _expectFailure(
    "'\(l)' is not equal to '\(r)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectNotEqual<T: Equatable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left != right { return }
  _expectFailure(
    "'\(left)' is equal to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectLessThan<T: Comparable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left < right { return }
  _expectFailure(
    "'\(left)' is not less than '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectLessThanOrEqual<T: Comparable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left <= right { return }
  _expectFailure(
    "'\(left)' is not less than or equal to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectGreaterThan<T: Comparable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left > right { return }
  _expectFailure(
    "'\(left)' is not greater than '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectGreaterThanOrEqual<T: Comparable>(
  _ left: T, _ right: T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  if left >= right { return }
  _expectFailure(
    "'\(left)' is not less than or equal to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

/// Check if `left` and `right` contain equal elements in the same order.
/// Note: `left` and `right` must be restartable sequences.
public func expectEqualElements<S1: Sequence, S2: Sequence>(
  _ left: S1,
  _ right: S2,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) where S1.Element == S2.Element, S1.Element: Equatable {
  let left = Array(left)
  let right = Array(right)
  if left.elementsEqual(right) { return }
  _expectFailure(
    "'\(left)' does not have equal elements to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectEquivalentElements<S1: Sequence, S2: Sequence>(
  _ left: S1,
  _ right: S2,
  by areEquivalent: (S1.Element, S2.Element) -> Bool,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  let left = Array(left)
  let right = Array(right)
  if left.elementsEqual(right, by: areEquivalent) { return }
  _expectFailure(
    "'\(left)' does not have equivalent elements to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

/// Check if `left` and `right` contain equal elements in the same order.
/// Note: `left` and `right` must be restartable sequences.
public func expectEqualElements<
  S1: Sequence, S2: Sequence,
  A: Equatable, B: Equatable
>(
  _ left: S1,
  _ right: S2,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) where S1.Element == (key: A, value: B), S2.Element == (key: A, value: B) {
  let left = Array(left)
  let right = Array(right)
  if
    left.elementsEqual(
      right,
      by: { $0.key == $1.key && $0.value == $1.value })
  { return }
  _expectFailure(
    "'\(left)' does not have equal elements to '\(right)'",
    message, trapping: trapping, file: file, line: line)
}

public func expectMonotonicallyIncreasing<S: Sequence>(
  _ items: S,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) where S.Element: Comparable {
  let items = Array(items)
  var it = items.makeIterator()
  guard var prev = it.next() else { return }
  while let next = it.next() {
    guard prev <= next else {
      _expectFailure(
        "'\(items)' is not monotonically increasing",
        message, trapping: trapping, file: file, line: line)
      return
    }
    prev = next
  }
}

public func expectStrictlyMonotonicallyIncreasing<S: Sequence>(
  _ items: S,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) where S.Element: Comparable {
  let items = Array(items)
  var it = items.makeIterator()
  guard var prev = it.next() else { return }
  while let next = it.next() {
    guard prev < next else {
      _expectFailure(
        "'\(items)' is not strictly monotonically increasing",
        message, trapping: trapping, file: file, line: line)
      return
    }
    prev = next
  }
}

public func expectThrows<T>(
  _ expression: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line,
  _ errorHandler: (Error) -> Void = { _ in }
) {
  do {
    let result = try expression()
    expectFailure("Expression did not throw"
                    + (T.self == Void.self ? "" : " (returned '\(result)' instead)"),
                  trapping: trapping,
                  file: file, line: line)
  } catch {
    errorHandler(error)
  }
}

public func expectType<T>(_ actual: T, _ expectedType: Any.Type) {
  expectTrue(type(of: actual) == expectedType)
}
