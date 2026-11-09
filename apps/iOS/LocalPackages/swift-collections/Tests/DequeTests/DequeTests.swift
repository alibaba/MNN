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
#if COLLECTIONS_SINGLE_MODULE
@_spi(Testing) import Collections
#else
import _CollectionsTestSupport
@_spi(Testing) import DequeModule
#endif

final class DequeTests: CollectionTestCase {
  func test_testingSPIs() {
    let deque = Deque(_capacity: 5, startSlot: 2, contents: [10, 20, 30, 40])
    expectEqual(deque.count, 4)
    expectEqual(deque._capacity, 5)
    expectEqual(deque._startSlot, 2)
    expectEqualElements(deque, [10, 20, 30, 40])
  }

  func test_CollectionConformance() {
    checkBidirectionalCollection(Deque<Int>(), expectedContents: [])
    checkBidirectionalCollection(Deque([1]), expectedContents: [1])
    checkBidirectionalCollection(Deque([1, 2, 3]), expectedContents: [1, 2, 3])
    checkBidirectionalCollection(Deque(0 ..< 10), expectedContents: 0 ..< 10)

    // Exhaustive tests for all deque layouts of various capacities
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withLifetimeTracking { tracker in
        let (deque, contents) = tracker.deque(with: layout)
        checkBidirectionalCollection(deque, expectedContents: contents)
      }
    }
  }

  func test_ExpressibleByArrayLiteral() {
    let deque: Deque<Int> = [1, 2, 3, 4]
    expectEqual(Array(deque), [1, 2, 3, 4])
  }

  func test_description() {
    expectEqual("\([] as Deque<Int>)", "[]")
    expectEqual("\([1, 2, 3] as Deque<Int>)", "[1, 2, 3]")
    expectEqual("\([1, 2, nil, 3] as Deque<Int?>)", "[Optional(1), Optional(2), nil, Optional(3)]")

    let deque: Deque<StringConvertibleValue> = [1, 2, 3]
    expectEqual("\(deque)", "[debugDescription(1), debugDescription(2), debugDescription(3)]")
  }

  func test_debugDescription() {
    expectEqual(String(reflecting: [] as Deque<Int>), "[]")
    expectEqual(String(reflecting: [1, 2, 3] as Deque<Int>), "[1, 2, 3]")
    expectEqual(
      String(reflecting: [1, 2, nil, 3] as Deque<Int?>),
      "[Optional(1), Optional(2), nil, Optional(3)]")

    let deque: Deque<StringConvertibleValue> = [1, 2, 3]
    expectEqual(String(reflecting: deque), "[debugDescription(1), debugDescription(2), debugDescription(3)]")
  }

  func test_customMirror() {
    let deque: Deque<Int> = [1, 2, 3]
    let mirror = Mirror(reflecting: deque)
    expectEqual(mirror.displayStyle, .collection)
    expectNil(mirror.superclassMirror)
    expectTrue(mirror.children.compactMap { $0.label }.isEmpty) // No label
    expectEqualElements(mirror.children.map { $0.value as? Int }, deque.map { $0 })
  }

  func test_Equatable_Hashable() {
    let c1 = [1, 2, 3, 4]
    let c2 = [1, 2]
    let equivalenceClasses: [[Deque<Int>]] = [
      [
        Deque(_capacity: 4, startSlot: 0, contents: c1),
        Deque(_capacity: 6, startSlot: 0, contents: c1),
        Deque(_capacity: 4, startSlot: 2, contents: c1),
      ],
      [
        Deque(_capacity: 2, startSlot: 0, contents: c2),
        Deque(_capacity: 6, startSlot: 0, contents: c2),
        Deque(_capacity: 2, startSlot: 1, contents: c2),
      ],
      [
        Deque(),
        Deque(_capacity: 0, startSlot: 0, contents: []),
        Deque(_capacity: 6, startSlot: 0, contents: []),
        Deque(_capacity: 2, startSlot: 1, contents: []),
      ],
    ]
    checkHashable(equivalenceClasses: equivalenceClasses)
  }

  func test_Encodable() throws {
    let d1: Deque<Int> = []
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalEncoder.encode(d1), v1)

    let d2: Deque<Int> = [0, 1, 2, 3]
    let v2: MinimalEncoder.Value = .array([.int(0), .int(1), .int(2), .int(3)])
    expectEqual(try MinimalEncoder.encode(d2), v2)

    try withEveryDeque("deque", ofCapacities: [0, 1, 3]) { layout in
      try withLifetimeTracking { tracker in
        let (deque, contents) = tracker.deque(with: layout)
        let encoded = try MinimalEncoder.encode(deque)
        expectEqual(encoded, .array(contents.map { .int($0.payload) }))
      }
    }
  }

  func test_Decodable() throws {
    let d1: Deque<Int> = []
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalDecoder.decode(v1, as: Deque<Int>.self), d1)

    let d2: Deque<Int> = [0, 1, 2, 3]
    let v2: MinimalEncoder.Value = .array([.int(0), .int(1), .int(2), .int(3)])
    expectEqual(try MinimalDecoder.decode(v2, as: Deque<Int>.self), d2)

    try withEveryDeque("deque", ofCapacities: [0, 1, 3]) { layout in
      let contents = Array(0 ..< layout.count)
      let deque = Deque(layout: layout, contents: contents)
      let input: MinimalDecoder.Value = .array(contents.map { .int($0) })
      let decoded = try MinimalDecoder.decode(input, as: Deque<Int>.self)
      expectEqual(decoded, deque)
    }
  }

  func test_copyToContiguousArray() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withLifetimeTracking { tracker in
        let (deque, contents) = tracker.deque(with: layout)
        let actual = deque._copyToContiguousArray()
        expectEqualElements(actual, contents)
      }
    }
  }

  func test_partial_copyContents() {
    // `Deque` supports `_copyContents` invocations with buffer sizes below
    // `underestimatedCount`. This isn't a requirement, so the `Collection`
    // checker doesn't cover this case.
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("prefix", in: 0 ... layout.count) { prefix in
        withLifetimeTracking { tracker in
          let (deque, contents) = tracker.deque(with: layout)

          var it: Deque<LifetimeTracked<Int>>.Iterator?
          let head = Array<LifetimeTracked<Int>>(
            unsafeUninitializedCapacity: prefix
          ) { buffer, count in
            (it, count) = deque._copyContents(initializing: buffer)
          }
          let tail = Array(IteratorSequence(it!))
          expectEqualElements(head, contents.prefix(upTo: prefix))
          expectEqualElements(tail, contents.suffix(from: prefix))
        }
      }
    }
  }

  func test_unsafeUninitializedInitializer_nothrow() {
    withEvery("capacity", in: 0 ..< 100) { cap in
      withEvery("count", in: [0, cap / 3, cap / 2, 2 * cap / 3, cap] as Set) { count in
        withLifetimeTracking { tracker in
          let contents = tracker.instances(for: 0 ..< count)
          let d1 = Deque<LifetimeTracked<Int>>(
            unsafeUninitializedCapacity: cap,
            initializingWith: { target, c in
              expectNotNil(target.baseAddress)
              expectEqual(target.count, cap)
              expectEqual(c, 0)
              contents.withUnsafeBufferPointer { source in
                precondition(source.count <= target.count)
                target.baseAddress!.initialize(
                  from: source.baseAddress!,
                  count: source.count)
              }
              c = count
            })
          expectEqualElements(d1, contents)
        }
      }
    }
  }

  struct TestError: Error, Equatable {
    let value: Int
    init(_ value: Int) { self.value = value }
  }

  func test_unsafeUninitializedInitializer_throw() {
    func workaroundSR14134(cap: Int, count: Int, tracker: LifetimeTracker) {
      // This function works around https://bugs.swift.org/browse/SR-14134
      let contents = tracker.instances(for: 0 ..< count)
      expectThrows(
        try Deque<LifetimeTracked<Int>>(
          unsafeUninitializedCapacity: cap,
          initializingWith: { target, c in
            expectNotNil(target.baseAddress)
            expectEqual(target.count, cap)
            expectEqual(c, 0)
            contents.withUnsafeBufferPointer { source in
              precondition(source.count <= target.count)
              target.baseAddress!.initialize(
                from: source.baseAddress!,
                count: source.count)
            }
            c = count
            throw TestError(count)
          })
      ) { error in
        expectEqual(error as? TestError, TestError(count))
      }
    }

    withEvery("capacity", in: 0 ..< 100) { cap in
      withEvery("count", in: [0, cap / 3, cap / 2, 2 * cap / 3, cap] as Set<Int>) { count in
        withLifetimeTracking { tracker in
          workaroundSR14134(cap: cap, count: count, tracker: tracker)
        }
      }
    }
  }

  func test_popFirst() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            let expected = contents[...].popFirst()
            let actual = deque.popFirst()
            expectEqual(actual, expected)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_prependOne() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          let extra = tracker.instance(for: layout.count)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.insert(extra, at: 0)
            deque.prepend(extra)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_prependManyFromMinimalSequence() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("prependCount", in: 0 ..< 10) { prependCount in
        withEvery("underestimatedCount", in: [UnderestimatedCountBehavior.precise, .half, .value(min(1, prependCount))]) { underestimatedCount in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              let extras = tracker.instances(for: layout.count ..< layout.count + prependCount)
              let sequence = MinimalSequence(elements: extras, underestimatedCount: underestimatedCount)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                contents.insert(contentsOf: extras, at: 0)
                deque.prepend(contentsOf: sequence)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_prependManyFromMinimalCollection() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("prependCount", in: 0 ..< 10) { prependCount in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extra = tracker.instances(for: layout.count ..< layout.count + prependCount)
            let minimal = MinimalCollection(extra)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.insert(contentsOf: extra, at: 0)
              deque.prepend(contentsOf: minimal)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_prependManyFromContiguousArray_asCollection() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("prependCount", in: 0 ..< 10) { prependCount in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extraRange = layout.count ..< layout.count + prependCount
            let extra = ContiguousArray(tracker.instances(for: extraRange))
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.insert(contentsOf: extra, at: 0)
              deque.prepend(contentsOf: extra)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_prependManyFromContiguousArray_asSequence() {
    // This calls the Sequence-based `Deque.prepend` overload, even if
    // `elements` happens to be of a Collection type.
    func prependSequence<S: Sequence>(
      contentsOf elements: S,
      to deque: inout Deque<S.Element>
    ) {
      deque.prepend(contentsOf: elements)
    }

    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("prependCount", in: 0 ..< 10) { prependCount in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extraRange = layout.count ..< layout.count + prependCount
            let extra = ContiguousArray(tracker.instances(for: extraRange))
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.insert(contentsOf: extra, at: 0)
              prependSequence(contentsOf: extra, to: &deque)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_prependManyFromBridgedArray() {
    // https://github.com/apple/swift-collections/issues/27
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("appendCount", in: 0 ..< 10) { appendCount in
        withEvery("isShared", in: [false, true]) { isShared in
          var contents: [NSObject] = (0 ..< layout.count).map { _ in NSObject() }
          var deque = Deque(layout: layout, contents: contents)
          let extra: [NSObject] = (0 ..< appendCount)
            .map { _ in NSObject() }
            .withUnsafeBufferPointer { buffer in
              NSArray(objects: buffer.baseAddress, count: buffer.count) as! [NSObject]
            }
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.insert(contentsOf: extra, at: 0)
            deque.prepend(contentsOf: extra)
            expectEquivalentElements(deque, contents, by: ===)
          }
        }
      }
    }
  }

}

