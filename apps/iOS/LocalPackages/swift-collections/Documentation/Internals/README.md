# Package Internals

## Benchmarks

The package includes an extensive library of benchmarks in the [Benchmarks](./Benchmarks) directory, driven by a command-line executable target, called `benchmark`. These benchmarks, the executable target, and its command-line interface are not considered part of the public interface of the package. As such, new releases may break them without any special ceremony. We do expect the benchmarks will stabilize with time.

For more information on our benchmarking tool, please see its dedicated package, [**Swift Collections Benchmark**][swift-collections-benchmark].

[swift-collections-benchmark]: https://github.com/apple/swift-collections-benchmark

## Test Support Library

The package comes with a rich test support library in the [Sources/_CollectionsTestSupport](../../Sources/_CollectionsTestSupport) directory. These were loosely adapted from the contents of the `StdlibUnittest*` modules in the [Swift compiler repository](https://github.com/apple/swift/tree/main/stdlib/private), with some custom additions.

These components would likely be of interest to the wider Swift community, but they aren't yet stable enough (or documented enough) to publish them. Accordingly, these testing helpers are currently considered implementation details of this package, and are subject to change at whim.

The test support library currently provides the following functionality:

- [`AssertionContexts`](../../Sources/_CollectionsTestSupport/AssertionContexts): Custom test assertions with support for keeping track of nested context information, including stopping execution when the current context matches a particular value. (Useful for debugging combinatorial tests.)

  <details>
   <summary><strong>Click here for a short demonstration</strong></summary>

    ```swift
    import XCTest
    import _CollectionsTestSupport

    final class DequeTests: CollectionTestCase {
      func test_demo() {
        let values = [0, 10, 20, 30, 42, 50, 60]
        for i in values.indices {
          context.withTrace("i: \(i)") {
            expectEqual(values[i], 10 * i)
          }
        }
      }
    }
    ```

    ```
    DemoTests.swift:21: error: -[DemoTests.DemoTests test_demo] : XCTAssertEqual failed: ("42") is not equal to ("40") - 
    Trace:
      - i: 4
    ```
    
    To debug issues, copy the trace message into a `context.failIfTraceMatches(_:)` invocation and set a breakpoint on test failures.
    
    ```swift
        let values = [0, 10, 20, 30, 42, 50, 60]
        for i in values.indices {
          context.withTrace("i: \(i)") {
            // This will report a test failure before executing the `i == 4` case,
            // letting us investigate what's going on.
            context.failIfTraceMatches("""
              Trace:
                - i: 4
              """)
            expectEqual(values[i], 10 * i)
          }
        }
    ```

    </details>

- [`Combinatorics`](../../Sources/_CollectionsTestSupport/AssertionContexts/Combinatorics.swift): Basic support for exhaustive combinatorial testing. This allows us to easily verify that a collection operation works correctly on all possible instances up to a certain size, including behavioral variations such as unique/shared storage. This file also contains a basic set of functions for executing the same test code for all subsets or all permutations of a given collection, with each iteration registered in the current test context, for easy reproduction of failed cases.

  <details>
   <summary><strong>Click here for an example</strong></summary>

    ```swift
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
    ```
    
  </details>
  
- [`ConformanceCheckers`](../../Sources/_CollectionsTestSupport/ConformanceCheckers): A set of generic, semi-automated protocol conformance tests for some Standard Library protocols. These can be used to easily validate the custom protocol conformances provided by this package. These checks aren't (can't be) complete -- but when used correctly, they are able to detect most accidental mistakes. 

    We currently have conformance checkers for the following protocols:
    
    - [`Sequence`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckSequence.swift)
    - [`Collection`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckCollection.swift)
    - [`BidirectionalCollection`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckBidirectionalCollection.swift)
    - [`Equatable`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckEquatable.swift)
    - [`Hashable`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckHashable.swift)
    - [`Comparable`](../../Sources/_CollectionsTestSupport/ConformanceCheckers/CheckComparable.swift)

- [`MinimalTypes`](../../Sources/_CollectionsTestSupport/MinimalTypes): Minimally conforming implementations for standard protocols. These types conform to various standard protocols by implementing the requirements in as narrow-minded way as possible -- sometimes going to extreme lengths to, say, implement collection index invalidation logic in the most unhelpful way possible.

    - [`MinimalSequence`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalSequence.swift)
    - [`MinimalCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalCollection.swift)
    - [`MinimalBidirectionalCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalBidirectionalCollection.swift)
    - [`MinimalRandomAccessCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalRandomAccessCollection.swift)
    - [`MinimalMutableRandomAccessCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalMutableRandomAccessCollection.swift)
    - [`MinimalRangeReplaceableRandomAccessCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalRangeReplaceableRandomAccessCollection.swift)
    - [`MinimalMutableRangeReplaceableRandomAccessCollection`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalMutableRangeReplaceableRandomAccessCollection.swift)
    - [`MinimalIterator`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalIterator.swift)
    - [`MinimalIndex`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalIndex.swift)
    - [`MinimalEncoder`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalEncoder.swift)
    - [`MinimalDecoder`](../../Sources/_CollectionsTestSupport/MinimalTypes/MinimalDecoder.swift)

- [`Utilities`](../../Sources/_CollectionsTestSupport/Utilities): Utility types. Wrapper types for boxed values, a simple deterministic random number generator, and a lifetime tracker for catching simple memory management issues such as memory leaks. (The [Address Sanitizer][asan] can be used to catch more serious problems.)

[asan]: https://developer.apple.com/documentation/xcode/diagnosing_memory_thread_and_crash_issues_early?language=objc


