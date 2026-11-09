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

/// Run the supplied closure with all values in `items` in a loop,
/// recording the current value in the current test trace stack.
public func withEvery<Element>(
  _ label: String,
  by generator: () -> Element?,
  file: StaticString = #file,
  line: UInt = #line,
  run body: (Element) throws -> Void
) rethrows {
  let context = TestContext.current
  while let item = generator() {
    let entry = context.push("\(label): \(item)", file: file, line: line)
    var done = false
    defer {
      context.pop(entry)
      if !done {
        print(context.currentTrace(title: "Throwing trace"))
      }
    }
    try body(item)
    done = true
  }
}

/// Run the supplied closure with all values in `items` in a loop,
/// recording the current value in the current test trace stack.
public func withEvery<S: Sequence>(
  _ label: String,
  in items: S,
  file: StaticString = #file,
  line: UInt = #line,
  run body: (S.Element) throws -> Void
) rethrows {
  let context = TestContext.current
  for item in items {
    let entry = context.push("\(label): \(item)", file: file, line: line)
    var done = false
    defer {
      context.pop(entry)
      if !done {
        print(context.currentTrace(title: "Throwing trace"))
      }
    }
    try body(item)
    done = true
  }
}

public func withEveryRange<T: Strideable>(
  _ label: String,
  in bounds: Range<T>,
  file: StaticString = #file,
  line: UInt = #line,
  run body: (Range<T>) throws -> Void
) rethrows where T.Stride == Int {
  let context = TestContext.current
  for lowerBound in bounds.lowerBound ... bounds.upperBound {
    for upperBound in lowerBound ... bounds.upperBound {
      let range = lowerBound ..< upperBound
      let entry = context.push("\(label): \(range)", file: file, line: line)
      var done = false
      defer {
        context.pop(entry)
        if !done {
          print(context.currentTrace(title: "Throwing trace"))
        }
      }
      try body(range)
      done = true
    }
  }
}

internal func _samples<C: Collection>(from items: C) -> [C.Element] {
  let c = items.count
  guard c > 7 else { return Array(items) }
  let offsets = [0, 1, c / 2 - 1, c / 2, c / 2 + 1, c - 2, c - 1]
  var offset = 0
  var index = items.startIndex
  var result: [C.Element] = []
  result.reserveCapacity(7)
  for o in offsets {
    items.formIndex(&index, offsetBy: o - offset)
    offset = o
    result.append(items[index])
  }
  return result
}

/// Run the supplied closure with all values in `items` in a loop,
/// recording the current value in the current test trace stack.
public func withSome<C: Collection>(
  _ label: String,
  in items: C,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line,
  run body: (C.Element) throws -> Void
) rethrows {
  let context = TestContext.current

  func check(item: C.Element) throws {
    let entry = context.push("\(label): \(item)", file: file, line: line)
    var done = false
    defer {
      if !done {
        print(context.currentTrace(title: "Throwing trace"))
      }
      context.pop(entry)
    }
    try body(item)
    done = true
  }

  let c = items.count
  
  if let maxSamples = maxSamples, c > maxSamples {
    // FIXME: Use randomSample() from swift-algorithms to eliminate dupes
    for _ in 0 ..< maxSamples {
      let r = Int.random(in: 0 ..< c)
      let i = items.index(items.startIndex, offsetBy: r)
      try check(item: items[i])
    }
  } else {
    for item in items {
      try check(item: item)
    }
  }
}

public func withSomeRanges<T: Strideable>(
  _ label: String,
  in bounds: Range<T>,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line,
  run body: (Range<T>) throws -> Void
) rethrows where T.Stride == Int {
  try withSome(
    label,
    in: IndexRangeCollection(bounds: bounds),
    maxSamples: maxSamples,
    run: body)
}

/// Utility function for testing mutations with value semantics.
///
/// Calls `body` with on given collection value, while optionally keeping
/// hidden a hidden copy of it around. Once `body` returns, checks that the copy
/// remain unchanged.
///
/// - Parameters:
///    - `enabled`: if `false`, then no copies are made -- the values are passed to `body` with no processing.
///    - `value`: The collection value that is being tested.
///    - `checker`: An optional function that is used to check the consistency of the hidden copy.
///    - `body`: A closure performing a mutation on `value`.
public func withHiddenCopies<S: Sequence, R>(
  if enabled: Bool,
  of value: inout S,
  checker: (S) -> Void = { _ in },
  file: StaticString = #file, line: UInt = #line,
  _ body: (inout S) throws -> R
) rethrows -> R where S.Element: Equatable {
  guard enabled else { return try body(&value) }
  let copy = value
  let expected = Array(value)
  let result = try body(&value)
  expectEqualElements(copy, expected, file: file, line: line)
  checker(copy)
  return result
}

/// Utility function for testing mutations with value semantics.
///
/// Calls `body` with on given collection value, while optionally keeping
/// hidden a hidden copy of it around. Once `body` returns, checks that the copy
/// remain unchanged.
///
/// - Parameters:
///    - `enabled`: if `false`, then no copies are made -- the values are passed to `body` with no processing.
///    - `value`: The collection value that is being tested.
///    - `checker`: An optional function that is used to check the consistency of the hidden copy.
///    - `body`: A closure performing a mutation on `value`.
public func withHiddenCopies<
  S: Sequence,
  Key: Equatable,
  Value: Equatable,
  R
>(
  if enabled: Bool,
  of value: inout S,
  checker: (S) -> Void = { _ in },
  file: StaticString = #file, line: UInt = #line,
  _ body: (inout S) throws -> R
) rethrows -> R where S.Element == (key: Key, value: Value) {
  guard enabled else { return try body(&value) }
  let copy = value
  let expected = Array(value)
  let result = try body(&value)
  expectEqualElements(copy, expected, file: file, line: line)
  checker(copy)
  return result
}

/// Run the supplied closure with all subsets `items` in a loop,
/// recording the current subset in the test trace stack.
///
/// The subsets are generated so that they contain elements in the same order
/// as the original collection.
public func withEverySubset<C: Collection>(
  _ label: String,
  of items: C,
  body: ([C.Element]) -> Void
) {
  var set: [C.Element] = []
  _withEverySubset(label, of: items, extending: &set, body: body)
}

func _withEverySubset<C: Collection>(
  _ label: String,
  of items: C,
  extending set: inout [C.Element],
  body: ([C.Element]) -> Void
) {
  guard let item = items.first else {
    let entry = TestContext.current.push("\(label): \(set)")
    defer { TestContext.current.pop(entry) }
    body(set)
    return
  }
  _withEverySubset(label, of: items.dropFirst(), extending: &set, body: body)
  set.append(item)
  _withEverySubset(label, of: items.dropFirst(), extending: &set, body: body)
  set.removeLast()
}

public func withEveryPermutation<C: Collection>(
  _ label: String,
  of items: C,
  body: ([C.Element]) -> Void
) {
  func send(_ items: [C.Element]) {
    let entry = TestContext.current.push("\(label): \(items)")
    body(items)
    TestContext.current.pop(entry)
  }

  // Heap's algorithm.
  var items = Array(items)
  send(items)
  var i = 1
  var c = [Int](repeating: 0, count: items.count)
  while i < items.count {
    if c[i] < i {
      if i.isMultiple(of: 2) {
        items.swapAt(0, i)
      } else {
        items.swapAt(c[i], i)
      }
      send(items)
      c[i] += 1
      i = 1
    } else {
      c[i] = 0
      i += 1
    }
  }
}
