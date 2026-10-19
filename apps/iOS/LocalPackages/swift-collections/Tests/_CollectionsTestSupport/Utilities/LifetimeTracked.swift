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

/// A type that tracks the number of live instances.
///
/// `LifetimeTracked` conforms to `CustomStringConvertible`, and conditionally
/// conforms to `Equatable`, `Hashable` and `Comparable` when its payload does.
///
/// `LifetimeTracked` is useful to check for leaks in algorithms and data
/// structures. The easiest way to produce instances is to use the
/// `withLifetimeTracking` function:
///
///      class FooTests: XCTestCase {
///        func testFoo() {
///          withLifetimeTracking([1, 2, 3]) { instances in
///            _ = instances.sorted(by: >)
///          }
///        }
///      }
public class LifetimeTracked<Payload> {
  public let tracker: LifetimeTracker
  internal var serialNumber: Int = 0
  public let payload: Payload

  public init(_ payload: Payload, for tracker: LifetimeTracker) {
    tracker.instances += 1
    tracker._nextSerialNumber += 1
    self.tracker = tracker
    self.serialNumber = tracker._nextSerialNumber
    self.payload = payload
  }

  deinit {
    precondition(serialNumber != 0, "Double deinit")
    tracker.instances -= 1
    serialNumber = -serialNumber
  }
}

extension LifetimeTracked: CustomStringConvertible {
  public var description: String {
    return "\(payload)"
  }
}

extension LifetimeTracked: Equatable where Payload: Equatable {
  public static func == (left: LifetimeTracked, right: LifetimeTracked) -> Bool {
    return left.payload == right.payload
  }
}

extension LifetimeTracked: Hashable where Payload: Hashable {
  public func _rawHashValue(seed: Int) -> Int {
    payload._rawHashValue(seed: seed)
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(payload)
  }

  public var hashValue: Int {
    payload.hashValue
  }
}

extension LifetimeTracked: Comparable where Payload: Comparable {
  public static func < (left: LifetimeTracked, right: LifetimeTracked) -> Bool {
    return left.payload < right.payload
  }
}

extension LifetimeTracked: Encodable where Payload: Encodable {
  public func encode(to encoder: Encoder) throws {
    try payload.encode(to: encoder)
  }
}
