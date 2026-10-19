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

import XCTest

public final class TestContext {
  internal var _nextStateId = 0
  internal var _nextIndexId = 0

  /// Stack of labels with associated source positions.
  /// Useful for tracking failed cases in combinatorial tests.
  internal var _trace: [Entry] = []

  // FIXME: This ought to be a thread-local variable.
  internal static var _current: TestContext?

  public init() {}
}

extension TestContext {
  public static var current: TestContext {
    guard let current = _current else {
      fatalError("There is no current test context")
    }
    return current
  }

  public static func pushNew() -> TestContext {
    let context = TestContext()
    push(context)
    return context
  }

  public static func push(_ context: TestContext) {
    precondition(_current == nil, "Can't nest test contexts")
    _current = context
  }

  public static func pop(_ context: TestContext) {
    precondition(_current === context, "Can't pop mismatching context")
    _current = nil
  }
}

extension TestContext {
  /// An entry in the stack trace. Associates a user-specified label with its associated source position.
  public struct Entry: Hashable, CustomStringConvertible {
    let label: String
    let file: StaticString
    let line: UInt

    public init(
      label: String,
      file: StaticString = #file,
      line: UInt = #line
    ) {
      self.label = label
      self.file = file
      self.line = line
    }

    public var description: String {
      "\(label) (\(file):\(line))"
    }

    public static func ==(left: Self, right: Self) -> Bool {
      left.label == right.label
        && left.file.utf8Start == right.file.utf8Start
        && left.file.utf8CodeUnitCount == right.file.utf8CodeUnitCount
        && left.line == right.line
    }

    public func hash(into hasher: inout Hasher) {
      hasher.combine(label)
      hasher.combine(file.utf8Start)
      hasher.combine(file.utf8CodeUnitCount)
      hasher.combine(line)
    }
  }
}

extension TestContext: Equatable {
  public static func ==(left: TestContext, right: TestContext) -> Bool {
    return left === right
  }
}

extension TestContext {
  internal func nextStateId() -> Int {
    defer { _nextStateId += 1 }
    return _nextStateId
  }

  internal func nextIndexId() -> Int {
    defer { _nextIndexId += 1 }
    return _nextIndexId
  }
}

extension TestContext {
  /// Add the specified trace to the test context trace stack.
  /// This call must be paired with a `pop` call with the same value,
  /// with no intervening unpopped pushes.
  @discardableResult
  public func push(_ entry: Entry) -> Entry {
    _trace.append(entry)
    return entry
  }

  /// Add the specified label to the test context trace stack.
  /// This call must be paired with a `pop` call with the same value,
  /// with no intervening unpopped pushes.
  @discardableResult
  public func push(
    _ label: String,
    file: StaticString = #file,
    line: UInt = #line
  ) -> Entry {
    return push(Entry(label: label, file: file, line: line))
  }

  /// Cancel the trace push that returned `trace` and pop the context trace back
  /// to the state it was before the push. All pushes since the one that pushed `trace`
  /// must have been popped at the time this is called.
  public func pop(_ entry: Entry) {
    let old = _trace.removeLast()
    precondition(
      old == entry,
      """
      Push/pop pairing violation: top of stack doesn't match expectation.
      expected: \(entry)
      actual: \(old)

      """)
  }

  /// Execute the supplied closure in a new nested trace entry `entry`.
  /// Assertion failure messages within the closure will include the specified information to aid with debugging.
  public func withTrace<R>(
    _ entry: Entry,
    _ body: () throws -> R
  ) rethrows -> R {
    push(entry)
    defer { pop(entry) }
    return try body()
  }

  /// Execute the supplied closure in a new nested trace entry.
  /// Assertion failure messages within the closure will include the specified information to aid with debugging.
  public func withTrace<R>(
    _ label: String,
    file: StaticString = #file,
    line: UInt = #line,
    _ body: () throws -> R
  ) rethrows -> R {
    let entry = push(label, file: file, line: line)
    defer { pop(entry) }
    return try body()
  }

  /// The current stack of tracing labels with their associated source positions.
  public var currentTrace: [Entry] { _trace }

  /// Return a (multi-line) string describing the current trace stack.
  /// This string can be used to identify a particular test context,
  /// for use in `failIfTraceMatches`.
  public func currentTrace(
    _ message: String = "",
    title: String = "Trace"
  ) -> String {
    guard !_trace.isEmpty else {
      return """
        \(message)
        \(title): (empty)

        """
    }
    var result = """
      \(message)
      \(title):

      """
    for trace in _trace {
      result += "  - \(trace.label)\n"
    }
    return result
  }

  public static func currentTrace(
    _ message: String = "",
    title: String = "Trace"
  ) -> String {
    guard let context = _current else { return message }
    return context.currentTrace(message, title: title)
  }

  /// Set a breakpoint on this function to stop execution only when `failIfTraceMatches`
  /// triggers a test failure. The alternative is to set up a "Test Failure" breakpoint,
  /// but if you have lots of test failures, that one might trigger too many times.
  @inline(never)
  public func debuggerBreak(
    _ message: String,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    XCTFail(message, file: file, line: line)
  }

  /// Call this function to emit a test failure when the current test trace matches
  /// the (typically multi-line) string given. By setting a breakpoint on test failures
  /// (or on the `debuggerBreak` method above), you can then pause execution to
  /// debug the test in that particular context.
  ///
  /// The string you need to pass to this function can be copy and pasted from
  /// the failure message of the `expect` family of assertion methods, or
  /// it can be manually generated by calling `currentTrace()` during a debug
  /// session.
  ///
  /// For example, here we trigger a test failure when `count` is 8 and `offset` is 3:
  ///
  ///     func testFoo() {
  ///       withEvery("count", in: 0 ..< 100) { count in
  ///         withEvery("offset", in: 0 ... count) { offset in
  ///           failIfTraceMatches("""
  ///             Trace:
  ///               - count: 8
  ///               - offset: 3
  ///             """)
  ///             deque.buggyMethod(...)
  ///         }
  ///       }
  ///     }
  ///
  public func failIfTraceMatches(
    _ expectedTrace: String,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    // Filter for lines that match the regex " *- "
    let breakOnTrace: [String] =
      expectedTrace.split(separator: "\n").compactMap { line in
      guard let i = line.firstIndex(where: { $0 != " " }) else { return nil }
      guard line[i...].starts(with: "- ") else { return nil }
      return String(line[i...].dropFirst(2))
    }
    let labels = _trace.map({ $0.label })
    if labels == breakOnTrace {
      debuggerBreak(currentTrace(title: "Hit trace"), file: file, line: line)
    }
  }
}
