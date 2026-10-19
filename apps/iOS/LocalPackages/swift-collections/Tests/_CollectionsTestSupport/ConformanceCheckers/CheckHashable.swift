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

/// Produce an integer hash value for `value` by feeding it to a dedicated
/// `Hasher`. This is always done by calling the `hash(into:)` method.
/// If a non-nil `seed` is given, it is used to perturb the hasher state;
/// this is useful for resolving accidental hash collisions.
private func _hash<H: Hashable>(_ value: H, seed: Int? = nil) -> Int {
  var hasher = Hasher()
  if let seed = seed {
    hasher.combine(seed)
  }
  hasher.combine(value)
  return hasher.finalize()
}

/// Test that the elements of `equivalenceClasses` consist of instances that
/// satisfy the semantic requirements of `Hashable`, with each group defining
/// a distinct equivalence class under `==`.
public func checkHashable<Instance: Hashable>(
  equivalenceClasses: [[Instance]],
  file: StaticString = #file, line: UInt = #line
) {
  let instances = equivalenceClasses.flatMap { $0 }
  // oracle[i] is the index of the equivalence class that contains instances[i].
  let oracle = equivalenceClasses.indices.flatMap { i in repeatElement(i, count: equivalenceClasses[i].count) }
  checkHashable(
    instances,
    equalityOracle: { oracle[$0] == oracle[$1] },
    file: file, line: line)
}

public func checkHashable<T : Hashable>(
  expectedEqual: Bool, _ lhs: T, _ rhs: T,
  file: StaticString = #file, line: UInt = #line
) {
  checkHashable(
    [lhs, rhs], equalityOracle: { expectedEqual || $0 == $1 }, file: file, line: line)
}

/// Test that the elements of `instances` satisfy the semantic requirements of
/// `Hashable`, using `equalityOracle` to generate equality and hashing
/// expectations from pairs of positions in `instances`.
public func checkHashable<Instances: Collection>(
  _ instances: Instances,
  equalityOracle: (Instances.Index, Instances.Index) -> Bool,
  file: StaticString = #file, line: UInt = #line
) where Instances.Element: Hashable {
  checkEquatable(instances, oracle: equalityOracle, file: file, line: line)
  _checkHashable(instances, equalityOracle: equalityOracle, file: file, line: line)
}

/// Same as `checkHashable(_:equalityOracle:file:line:)` but doesn't check
/// `Equatable` conformance. Useful for preventing duplicate testing.
public func _checkHashable<Instances: Collection>(
  _ instances: Instances,
  equalityOracle: (Instances.Index, Instances.Index) -> Bool,
  file: StaticString = #file, line: UInt = #line
) where Instances.Element: Hashable {
  let entry = TestContext.current.push("checkHashable", file: file, line: line)
  defer { TestContext.current.pop(entry) }
  for i in instances.indices {
    let x = instances[i]
    for j in instances.indices {
      let y = instances[j]
      let expected = equalityOracle(i, j)
      if expected {
        expectEqual(
          _hash(x), _hash(y),
          """
          hash(into:) expected to match, found to differ:
          lhs (at index \(i)): \(x)
          rhs (at index \(j)): \(y)
          """)
        expectEqual(
          x.hashValue, y.hashValue,
          """
          hashValue expected to match, found to differ
          lhs (at index \(i)): \(x)
          rhs (at index \(j)): \(y)
          """)
        expectEqual(
          x._rawHashValue(seed: 0), y._rawHashValue(seed: 0),
          """
          _rawHashValue(seed:) expected to match, found to differ
          lhs (at index \(i)): \(x)
          rhs (at index \(j)): \(y)
          """)
      } else {
        // Try a few different seeds; at least one of them should discriminate
        // between the hashes. It is extremely unlikely this check will fail
        // all ten attempts, unless the type's hash encoding is not unique,
        // or unless the hash equality oracle is wrong.
        expectTrue(
          (0 ..< 10).contains { _hash(x, seed: $0) != _hash(y, seed: $0) },
          """
          hash(into:) expected to differ, found to match
          lhs (at index \(i)): \(x)
          rhs (at index \(j)): \(y)
          """)
        expectTrue(
          (0 ..< 10).contains { i in
            x._rawHashValue(seed: i) != y._rawHashValue(seed: i)
          },
          """
          _rawHashValue(seed:) expected to differ, found to match
          lhs (at index \(i)): \(x)
          rhs (at index \(j)): \(y)
          """)
      }
    }
  }
}
