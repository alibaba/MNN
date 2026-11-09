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


/// A container type implementing a double-ended priority queue.
/// `Heap` is a container of `Comparable` elements that provides immediate
/// access to its minimal and maximal members, and supports removing these items
/// or inserting arbitrary new items in (amortized) logarithmic complexity.
///
///     var queue: Heap<Int> = [3, 4, 1, 2]
///     queue.insert(0)
///     print(queue.min)      // 0
///     print(queue.popMax()) // 4
///     print(queue.max)      // 3
///
/// `Heap` implements the min-max heap data structure, based on
/// [Atkinson et al. 1986].
///
/// [Atkinson et al. 1986]: https://doi.org/10.1145/6617.6621
///
/// > M.D. Atkinson, J.-R. Sack, N. Santoro, T. Strothotte.
/// "Min-Max Heaps and Generalized Priority Queues."
/// *Communications of the ACM*, vol. 29, no. 10, Oct. 1986., pp. 996-1000,
/// doi:[10.1145/6617.6621](https://doi.org/10.1145/6617.6621)
///
/// To efficiently implement these operations, a min-max heap arranges its items
/// into a complete binary tree, maintaining a specific invariant across levels,
/// called the "min-max heap property": each node at an even level in the tree
/// must be less than or equal to all its descendants, while each node at an odd
/// level in the tree must be greater than or equal to all of its descendants.
/// To achieve a compact representation, this tree is stored in breadth-first
/// order inside a single contiguous array value.
///
/// Unlike most container types, `Heap` doesn't provide a direct way to iterate
/// over the elements it contains -- it isn't a `Sequence` (nor a `Collection`).
/// This is because the order of items in a heap is unspecified and unstable:
/// it may vary between heaps that contain the same set of items, and it may
/// sometimes change in between versions of this library. In particular, the
/// items are (almost) never expected to be in sorted order.
///
/// For cases where you do need to access the contents of a heap directly and
/// you don't care about their (lack of) order, you can do so by invoking the
/// `unordered` view. This read-only view gives you direct access to the
/// underlying array value:
///
///     for item in queue.unordered {
///       ...
///     }
///
/// The name `unordered` highlights the lack of ordering guarantees on the
/// contents, and it helps avoid relying on any particular order.
@frozen
public struct Heap<Element: Comparable> {
  @usableFromInline
  internal var _storage: ContiguousArray<Element>

  /// Creates an empty heap.
  @inlinable
  public init() {
    _storage = []
  }
}

extension Heap: Sendable where Element: Sendable {}

extension Heap {
  /// A Boolean value indicating whether or not the heap is empty.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public var isEmpty: Bool {
    _storage.isEmpty
  }

  /// The number of elements in the heap.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public var count: Int {
    _storage.count
  }

  /// A read-only view into the underlying array.
  ///
  /// Note: The elements aren't _arbitrarily_ ordered (it is, after all, a
  /// heap). However, no guarantees are given as to the ordering of the elements
  /// or that this won't change in future versions of the library.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var unordered: [Element] {
    Array(_storage)
  }

  /// Creates an empty heap with preallocated space for at least the
  /// specified number of elements.
  ///
  /// Use this initializer to avoid intermediate reallocations of a heap's
  /// storage when you know in advance how many elements you'll insert into it
  /// after creation.
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the newly
  ///   created heap should be able to store without reallocating its storage.
  ///
  /// - Complexity: O(1) allocations
  @inlinable
  public init(minimumCapacity: Int) {
    self.init()
    self.reserveCapacity(minimumCapacity)
  }

  /// Reserves enough space to store the specified number of elements.
  ///
  /// If you are adding a known number of elements to a heap, use this method
  /// to avoid multiple reallocations. This method ensures that the heap has
  /// unique, mutable, contiguous storage, with space allocated for at least
  /// the requested number of elements.
  ///
  /// For performance reasons, the size of the newly allocated storage might be
  /// greater than the requested capacity.
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the
  ///   resulting heap should be able to store without reallocating its storage.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func reserveCapacity(_ minimumCapacity: Int) {
    _storage.reserveCapacity(minimumCapacity)
  }

  /// Inserts the given element into the heap.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  public mutating func insert(_ element: Element) {
    _storage.append(element)

    _update { handle in
      handle.bubbleUp(_HeapNode(offset: handle.count - 1))
    }
    _checkInvariants()
  }

  /// Returns the element with the lowest priority, if available.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var min: Element? {
    _storage.first
  }

  /// Returns the element with the highest priority, if available.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var max: Element? {
    _storage.withUnsafeBufferPointer { buffer in
      guard buffer.count > 2 else {
        // If count is 0, `last` will return `nil`
        // If count is 1, the last (and only) item is the max
        // If count is 2, the last item is the max (as it's the only item in the
        // first max level)
        return buffer.last
      }
      // We have at least 3 items -- return the larger of the two in the first
      // max level
      return Swift.max(buffer[1], buffer[2])
    }
  }

  /// Removes and returns the element with the lowest priority, if available.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  public mutating func popMin() -> Element? {
    guard _storage.count > 0 else { return nil }

    var removed = _storage.removeLast()

    if _storage.count > 0 {
      _update { handle in
        let minNode = _HeapNode.root
        handle.swapAt(minNode, with: &removed)
        handle.trickleDownMin(minNode)
      }
    }

    _checkInvariants()
    return removed
  }

  /// Removes and returns the element with the highest priority, if available.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  public mutating func popMax() -> Element? {
    guard _storage.count > 2 else { return _storage.popLast() }

    var removed = _storage.removeLast()

    _update { handle in
      if handle.count == 2 {
        if handle[.leftMax] > removed {
          handle.swapAt(.leftMax, with: &removed)
        }
      } else {
        let maxNode = handle.maxValue(.leftMax, .rightMax)
        handle.swapAt(maxNode, with: &removed)
        handle.trickleDownMax(maxNode)
      }
    }

    _checkInvariants()
    return removed
  }

  /// Removes and returns the element with the lowest priority.
  ///
  /// The heap *must not* be empty.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  @discardableResult
  public mutating func removeMin() -> Element {
    return popMin()!
  }

  /// Removes and returns the element with the highest priority.
  ///
  /// The heap *must not* be empty.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  @discardableResult
  public mutating func removeMax() -> Element {
    return popMax()!
  }

  /// Replaces the minimum value in the heap with the given replacement,
  /// then updates heap contents to reflect the change.
  ///
  /// The heap must not be empty.
  ///
  /// - Parameter replacement: The value that is to replace the current
  ///   minimum value.
  /// - Returns: The original minimum value before the replacement.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  @discardableResult
  public mutating func replaceMin(with replacement: Element) -> Element {
    precondition(!isEmpty, "No element to replace")

    var removed = replacement
    _update { handle in
      let minNode = _HeapNode.root
      handle.swapAt(minNode, with: &removed)
      handle.trickleDownMin(minNode)
    }
    _checkInvariants()
    return removed
  }
    
  /// Removes all the elements that satisfy the given predicate.
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   heap as its argument and returns a Boolean value indicating
  ///   whether the element should be removed from the heap.
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in the heap.
  @inlinable
  public mutating func removeAll(
    where shouldBeRemoved: (Element) throws -> Bool
  ) rethrows {
    defer {
      if _storage.count > 1 {
        _update { handle in
          handle.heapify()
        }
      }
      _checkInvariants()
    }
    try _storage.removeAll(where: shouldBeRemoved)
  }

  /// Replaces the maximum value in the heap with the given replacement,
  /// then updates heap contents to reflect the change.
  ///
  /// The heap must not be empty.
  ///
  /// - Parameter replacement: The value that is to replace the current maximum
  ///   value.
  /// - Returns: The original maximum value before the replacement.
  ///
  /// - Complexity: O(log(`count`)) element comparisons
  @inlinable
  @discardableResult
  public mutating func replaceMax(with replacement: Element) -> Element {
    precondition(!isEmpty, "No element to replace")

    var removed = replacement
    _update { handle in
      switch handle.count {
      case 1:
        handle.swapAt(.root, with: &removed)
      case 2:
        handle.swapAt(.leftMax, with: &removed)
        handle.bubbleUp(.leftMax)
      default:
        let maxNode = handle.maxValue(.leftMax, .rightMax)
        handle.swapAt(maxNode, with: &removed)
        handle.bubbleUp(maxNode)  // This must happen first
        handle.trickleDownMax(maxNode)  // Either new element or dethroned min
      }
    }
    _checkInvariants()
    return removed
  }
}

// MARK: -

extension Heap {
  /// Initializes a heap from a sequence.
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in `elements`.
  @inlinable
  public init(_ elements: some Sequence<Element>) {
    _storage = ContiguousArray(elements)
    guard _storage.count > 1 else { return }

    _update { handle in
      handle.heapify()
    }
    _checkInvariants()
  }

  /// Inserts the elements in the given sequence into the heap.
  ///
  /// - Parameter newElements: The new elements to insert into the heap.
  ///
  /// - Complexity: O(`count` + *k*), where *k* is the length of `newElements`.
  @inlinable
  public mutating func insert(
    contentsOf newElements: some Sequence<Element>
  ) {
    let origCount = self.count
    if origCount == 0 {
      self = Self(newElements)
      return
    }
    defer { _checkInvariants() }
    _storage.append(contentsOf: newElements)
    let newCount = self.count

    guard newCount > origCount, newCount > 1 else {
      // If we didn't append, or the result is too small to violate heapness,
      // then we have nothing else to dp.
      return
    }

    // Otherwise we can either insert items one by one, or we can run Floyd's
    // algorithm to re-heapify our entire storage from scratch.
    //
    // If n is the original count, and k is the number of items we need to
    // append, then Floyd's costs O(n + k) comparisons/swaps, while
    // the naive loop costs k * log(n + k) -- so we expect that Floyd will
    // be cheaper whenever k is "large enough" relative to n.
    //
    // Floyd's algorithm has a worst-case upper complexity bound of 2 * (n + k),
    // so one simple heuristic is to use it whenever k * log(n + k) exceeds
    // that.
    //
    // FIXME: Write a benchmark to verify this heuristic.
    let heuristicLimit = 2 * newCount / newCount._binaryLogarithm()
    let useFloyd = (newCount - origCount) < heuristicLimit
    _update { handle in
      if useFloyd {
        handle.heapify()
      } else {
        for offset in origCount ..< handle.count {
          handle.bubbleUp(_HeapNode(offset: offset))
        }
      }
    }
  }
}
