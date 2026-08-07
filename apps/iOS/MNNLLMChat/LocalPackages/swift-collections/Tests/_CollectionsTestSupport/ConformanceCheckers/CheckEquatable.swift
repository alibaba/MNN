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

// Loosely adapted from https://github.com/apple/swift/tree/main/stdlib/private/StdlibUnittest

public func checkEquatable<Instance: Equatable>(
  equivalenceClasses: [[Instance]],
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) {
  let instances = equivalenceClasses.flatMap { $0 }
  let oracle = equivalenceClasses.indices.flatMap { i in repeatElement(i, count: equivalenceClasses[i].count) }
  checkEquatable(instances, oracle: { oracle[$0] == oracle[$1] }, file: file, line: line)
}

public func checkEquatable<C: Collection>(
  _ instances: C,
  oracle: (C.Index, C.Index) -> Bool,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) where C.Element: Equatable {
  let indices = Array(instances.indices)
  checkEquatable(
    Array(instances),
    oracle: { oracle(indices[$0], indices[$1]) },
    file: file, line: line)
}

public func checkEquatable<T : Equatable>(
  expectedEqual: Bool, _ lhs: T, _ rhs: T,
  file: StaticString = #file, line: UInt = #line
) {
  checkEquatable(
    [lhs, rhs],
    oracle: { expectedEqual || $0 == $1 },
    file: file, line: line)
}

public func checkEquatable<Instance: Equatable>(
  _ instances: [Instance],
  oracle: (Int, Int) -> Bool,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) {
  let entry = TestContext.current.push("checkEquatable", file: file, line: line)
  defer { TestContext.current.pop(entry) }
  // For each index (which corresponds to an instance being tested) track the
  // set of equal instances.
  var transitivityScoreboard: [Box<Set<Int>>] = instances.map { _ in Box([]) }

  withSomeRanges(
    "range", in: 0 ..< instances.count - 1, maxSamples: maxSamples
  ) { range in
    let i = range.lowerBound
    let j = range.upperBound

    if i == j {
      expectTrue(oracle(i, i),
                 "bad oracle: broken reflexivity at index \(i)")
    }
    let x = instances[i]
    let y = instances[j]

    let expectedXY = oracle(i, j)
    expectEqual(oracle(j, i), expectedXY,
                "bad oracle: broken symmetry between indices \(i), \(j)")
    
    let actualXY = (x == y)
    let actualYX = (y == x)
    expectEqual(
      actualXY, expectedXY,
        """
        Elements \((expectedXY
                    ? "expected equal, found not equal"
                    : "expected not equal, found equal")):
        lhs (at index \(i)): \(String(reflecting: x))
        rhs (at index \(j)): \(String(reflecting: y))
        """)

    expectEqual(
      actualYX, actualXY,
        """
        Reflexivity violation:
        \(actualXY
          ? "lhs == rhs but !(rhs == lhs)"
          : "!(lhs == rhs) but rhs == lhs")
        lhs (at index \(i)): \(String(reflecting: x))
        rhs (at index \(j)): \(String(reflecting: y))
        """)

    // Not-equal must be the inverse of equal.
    expectNotEqual(
      x != y, actualXY,
        """
        `!=` returns the same result as `==`:
        lhs (at index \(i)): \(String(reflecting: x))
        rhs (at index \(j)): \(String(reflecting: y))
        """)
    
    // Check transitivity of the predicate represented by the oracle.
    // If we are adding the instance `j` into an equivalence set, check that
    // it is equal to every other instance in the set.
    if expectedXY && i < j && transitivityScoreboard[i].value.insert(j).inserted {
      if transitivityScoreboard[i].value.count == 1 {
        transitivityScoreboard[i].value.insert(i)
      }
      for k in transitivityScoreboard[i].value {
        expectTrue(
          oracle(j, k),
            """
            bad oracle: transitivity violation
            x (at index \(i)): \(String(reflecting: x))
            y (at index \(j)): \(String(reflecting: y))
            z (at index \(k)): \(String(reflecting: instances[k]))
            """)
        // No need to check equality between actual values, we will check
        // them with the checks above.
      }
      precondition(transitivityScoreboard[j].value.isEmpty)
      transitivityScoreboard[j] = transitivityScoreboard[i]
    }
  }
}
