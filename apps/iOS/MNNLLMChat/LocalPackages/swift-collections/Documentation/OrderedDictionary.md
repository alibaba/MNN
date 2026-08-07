# `OrderedDictionary`

An ordered collection of key-value pairs.

## Declaration

```swift
import OrderedCollections

@frozen struct OrderedDictionary<Key: Hashable, Value>
```

## Overview

Like the standard `Dictionary`, ordered dictionaries use a hash table to
ensure that no two entries have the same keys, and to efficiently look up
values corresponding to specific keys. However, like an `Array` (and
unlike `Dictionary`), ordered dictionaries maintain their elements in a
particular user-specified order, and they support efficient random-access
traversal of their entries.

`OrderedDictionary` is a useful alternative to `Dictionary` when the order
of elements is important, or when you need to be able to efficiently access
elements at various positions within the collection.

You can create an ordered dictionary with any key type that conforms to the
`Hashable` protocol.

```swift
let responses: OrderedDictionary = [
  200: "OK",
  403: "Access forbidden",
  404: "File not found",
  500: "Internal server error",
]
```

### Equality of Ordered Dictionaries

Two ordered dictionaries are considered equal if they contain the same
elements, and *in the same order*. This matches the concept of equality of
an `Array`, and it is different from the unordered `Dictionary`.

```swift
let a: OrderedDictionary = [1: "one", 2: "two"]
let b: OrderedDictionary = [2: "two", 1: "one"]
a == b // false
b.swapAt(0, 1) // `b` now has value [1: "one", 2: "two"]
a == b // true
```

(`OrderedDictionary` only conforms to `Equatable` when its `Value` is
equatable.)

### Dictionary Operations

`OrderedDictionary` provides many of the same operations as `Dictionary`.

For example, you can look up and add/remove values using the familiar
key-based subscript, returning an optional value:

```swift
var dictionary: OrderedDictionary<String, Int> = [:]
dictionary["one"] = 1
dictionary["two"] = 2
dictionary["three"] // nil
// dictionary is now ["one": 1, "two": 2]
```

If a new entry is added using the subscript setter, it gets appended to the
end of the dictionary. (So that by default, the dictionary contains its
elements in the order they were originally inserted.)

`OrderedDictionary` also implements the variant of this subscript that takes
a default value. Like with `Dictionary`, this is useful when you want to
perform in-place mutations on values:

```swift
let text = "short string"
var counts: OrderedDictionary<Character, Int> = [:]
for character in text {
  counts[character, default: 0] += 1
}
// counts is ["s": 2, "h": 1, "o": 1,
//            "r": 2, "t": 2, " ": 1,
//            "i": 1, "n": 1, "g": 1]
```

If the `Value` type implements reference semantics, or when you need to
perform a series of individual mutations on the values, the closure-based
`updateValue(forKey:default:with:)` method provides an easier-to-use
alternative to the defaulted key-based subscript.

```swift
let text = "short string"
var counts: OrderedDictionary<Character, Int> = [:]
for character in text {
  counts.updateValue(forKey: character, default: 0) { value in
    value += 1
  }
}
// Same result as before
```

(This isn't currently available on the regular `Dictionary`.)

The `Dictionary` type's original `updateValue(_:forKey:)` method is also
available, and so is `index(forKey:)`, grouping/uniquing initializers
(`init(uniqueKeysWithValues:)`, `init(_:uniquingKeysWith:)`,
`init(grouping:by:)`), methods for merging one dictionary with another
(`merge`, `merging`), filtering dictionary entries (`filter(_:)`),
transforming values (`mapValues(_:)`), and a combination of these two
(`compactMapValues(_:)`).

### Sequence and Collection Operations

Ordered dictionaries use integer indices representing offsets from the
beginning of the collection. However, to avoid ambiguity between key-based
and indexing subscripts, `OrderedDictionary` doesn't directly conform to
`Collection`. Instead, it only conforms to `Sequence`, and provides a
random-access collection view over its key-value pairs:

```swift
responses[0] // `nil` (key-based subscript)
responses.elements[0] // `(200, "OK")` (index-based subscript)
```

Because ordered dictionaries need to maintain unique keys, neither
`OrderedDictionary` nor its `elements` view can conform to the full
`MutableCollection` or `RangeReplaceableCollection` protocols. However, they
are able to partially implement requirements: they support mutations
that merely change the order of elements, or just remove a subset of
existing members:

```swift
// Permutation operations from MutableCollection:
func swapAt(_ i: Index, _ j: Index)
func partition(by predicate: (Element) throws -> Bool) rethrows -> Index
func sort() where Element: Comparable
func sort(by predicate: (Element, Element) throws -> Bool) rethrows
func shuffle()
func shuffle<T: RandomNumberGenerator>(using generator: inout T)

// Removal operations from RangeReplaceableCollection:
func removeAll(keepingCapacity: Bool = false)
func remove(at index: Index) -> Element
func removeSubrange(_ bounds: Range<Int>)
func removeLast() -> Element
func removeLast(_ n: Int)
func removeFirst() -> Element
func removeFirst(_ n: Int)
func removeAll(where shouldBeRemoved: (Element) throws -> Bool) rethrows
```

`OrderedDictionary` also implements `reserveCapacity(_)` from
`RangeReplaceableCollection`, to allow for efficient insertion of a known
number of elements. (However, unlike `Array` and `Dictionary`,
`OrderedDictionary` does not provide a `capacity` property.)

### Keys and Values Views

Like the standard `Dictionary`, `OrderedDictionary` provides `keys` and
`values` properties that provide lightweight views into the corresponding
parts of the dictionary.

The `keys` collection is of type `OrderedSet<Key>`, containing all the keys
in the original dictionary.

```swift
let d: OrderedDictionary = [2: "two", 1: "one", 0: "zero"]
d.keys // [2, 1, 0] as OrderedSet<Int>
```

The `keys` property is read-only, so you cannot mutate the dictionary
through it. However, it returns an ordinary ordered set value, which can be
copied out and then mutated if desired. (Such mutations won't affect the
original dictionary value.)

The `values` collection is a mutable random-access collection of the values
in the dictionary:

```swift
d.values // "two", "one", "zero"
d.values[2] = "nada"
// `d` is now [2: "two", 1: "one", 0: "nada"]
d.values.sort()
// `d` is now [2: "nada", 1: "one", 0: "two"]
```

Both views store their contents in regular `Array` values, accessible
through their `elements` property.

## Performance

Like the standard `Dictionary` type, the performance of hashing operations
in `OrderedDictionary` is highly sensitive to the quality of hashing
implemented by the `Key` type. Failing to correctly implement hashing can
easily lead to unacceptable performance, with the severity of the effect
increasing with the size of the hash table.

In particular, if a certain set of keys all produce the same hash value,
then hash table lookups regress to searching an element in an unsorted
array, i.e., a linear operation. To ensure hashed collection types exhibit
their target performance, it is important to ensure that such collisions
cannot be induced merely by adding a particular list of keys to the
dictionary.

The easiest way to achieve this is to make sure `Key` implements hashing
following `Hashable`'s documented best practices. The conformance must
implement the `hash(into:)` requirement, and every bit of information that
is compared in `==` needs to be combined into the supplied `Hasher` value.
When used correctly, `Hasher` produces high-quality, randomly seeded hash
values that prevent repeatable hash collisions.

When `Key` correctly conforms to `Hashable`, key-based lookups in an ordered
dictionary is expected to take O(1) equality checks on average. Hash
collisions can still occur organically, so the worst-case lookup performance
is technically still O(*n*) (where *n* is the size of the dictionary);
however, long lookup chains are unlikely to occur in practice.

## Implementation Details

An ordered dictionary consists of an ordered set of keys, alongside a
regular `Array` value that contains their associated values.
