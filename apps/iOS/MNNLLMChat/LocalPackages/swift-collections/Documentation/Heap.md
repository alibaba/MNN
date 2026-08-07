# Heap

A partially-ordered tree of elements with performant insertion and removal operations.

## Declaration

```swift
public struct Heap<Element: Comparable>
```

## Overview

Array-backed [binary heaps](https://en.wikipedia.org/wiki/Heap_(data_structure)) provide performant lookups (`O(1)`) of the smallest or largest element (depending on whether it's a min-heap or a max-heap, respectively) as well as insertion and removal (`O(log n)`). Heaps are commonly used as the backing storage for a priority queue.

A variant on this, the [min-max heap](https://en.wikipedia.org/wiki/Min-max_heap), allows for performant lookups and removal of both the smallest **and** largest elements by interleaving min and max levels in the backing array. `Heap` is an implementation of a min-max heap.

### Initialization

There are a couple of options for initializing a `Heap`. To create an empty `Heap`, call `init()`:

```swift
var heap = Heap<Int>()
```

You can also create a `Heap` from an existing sequence in linear time:

```swift
var heap = Heap((1...).prefix(20))
```

Finally, a `Heap` can be created from an array literal:

```swift
var heap: Heap<Double> = [0.1, 0.6, 1.0, 0.15, 0.42]
```

### Insertion

#### Of a single element

To insert an element into a `Heap`, call `insert(_:)`:

```swift
var heap = Heap<Int>()
heap.insert(6)
heap.insert(2)
```

This works by adding the new element into the end of the backing array and then bubbling it up to where it belongs in the heap.

#### Of a sequence of elements

You can also insert a sequence of elements into a `Heap`:

```swift
var heap = Heap(0 ..< 10)
heap.insert(contentsOf: (20 ... 100).shuffled())
heap.insert(contentsOf: [-5, -6, -8, -12, -3])
```

### Lookup

As mentioned earlier, the smallest and largest elements can be queried in constant time:

```swift
var heap = Heap(1 ... 20)
let min = heap.min  // 1
let max = heap.max  // 20
```

In a min-max heap, the smallest element is stored at index 0 in the backing array; the largest element is stored at either index 1 or index 2, the first max level in the heap (so to look up the largest, we compare the two and return the larger one).

We also expose a read-only view into the backing array, should somebody need that.

```swift
let heap = Heap((1...100).shuffled())
for val in heap.unordered {
   ...
}
```

> Note: The elements aren't _arbitrarily_ ordered (it is, after all, a heap). However, no guarantees are given as to the ordering of the elements or that this won't change in future versions of the library.

### Removal

Removal has logarithmic complexity, and removing both the smallest and largest elements is supported:

```swift
var heap = Heap((1...20).shuffled())
var heap2 = heap

while let min = heap.popMin() {
    print("Next smallest element:", min)
}

while let max = heap2.popMax() {
    print("Next largest element:", max)
}
```

To remove the smallest element, we remove and return the element at index 0. The element at the end of the backing array is put in its place at index 0, and then we trickle it down to where it belongs in the heap. To remove the largest element, we do the same except the index is whatever the index of the largest element is (see above) instead of 0.

We also have non-optional flavors that assume the heap isn't empty, `removeMin()` and `removeMax()`.

### Iteration

`Heap` itself doesn't conform to `Sequence` because of the potential confusion around which direction it should iterate (largest-to-smallest? smallest-to-largest?).

### Performance

| Operation | Complexity |
|-----------|------------|
| Insert | O(log n) |
| Get smallest element | O(1) |
| Get largest element | O(1) |
| Remove smallest element | O(log n) |
| Remove largest element | O(log n) |

In all of the above, `n` is the number of elements in the heap.

![Heap performance graph](Images/Heap%20Performance.png)

The above graph was generated in release mode on a MacBook Pro (16-inch, 2019) with a 2.3 GHz 8-Core Intel Core i9 using the benchmarks defined in the `swift-collections-benchmark` target.

## Implementation Details

The implementation is based on the min-max heap data structure as introduced by [Atkinson et al. 1986].

[Atkinson et al. 1986]: https://doi.org/10.1145/6617.6621

Min-max heaps are complete binary trees represented implicitly as an array of their elements. Each node at an even level in the tree is less than or equal to all its descendants, while each node at an odd level in the tree is greater or equal to all of its descendants.

```
// Min-max heap:
level 0 (min):         ┌────── A ──────┐
level 1 (max):     ┌── J ──┐       ┌── G ──┐
level 2 (min):   ┌ D ┐   ┌ B       F       C
level 3 (max):   I   E   H 

// Array representation:
["A", "J", "G", "D", "B", "F", "C", "I", "E", "H"]
```

By the min-max property above, the root node is an on even level, so its value ("A" in this example) must be the minimum of the entire heap. Its two children are on an odd level, so they hold the maximum value for their respective subtrees; it follows that one of them holds the maximum value for the whole tree -- in this case, "J". Accessing the minimum and maximum values in the heap can therefore be done in O(1) comparisons.

Mutations of the heap (insertions, removals) must ensure that items remain arranged in a way that maintain the min-max property. Inserting a single new element or removing the current minimum/maximum can be done by rearranging items on a single path in the tree; accordingly, these operations execute O(log(`count`)) comparisons/swaps.

---

M.D. Atkinson, J.-R. Sack, N. Santoro, T. Strothotte.
"Min-Max Heaps and Generalized Priority Queues."
*Communications of the ACM*, vol. 29, no. 10, Oct. 1986., pp. 996-1000,
doi:[10.1145/6617.6621](https://doi.org/10.1145/6617.6621)
