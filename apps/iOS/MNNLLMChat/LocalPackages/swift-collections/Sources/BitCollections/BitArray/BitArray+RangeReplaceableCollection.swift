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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension BitArray: RangeReplaceableCollection {}

extension BitArray {
  /// Prepares the bit array to store the specified number of bits.
  ///
  /// If you are adding a known number of elements to an array, use this
  /// method to avoid multiple reallocations.
  ///
  /// - Parameter n: The requested number of bits to store.
  public mutating func reserveCapacity(_ n: Int) {
    let wordCount = _Word.wordCount(forBitCount: UInt(n))
    _storage.reserveCapacity(wordCount)
  }
  
  /// Creates a new, empty bit array.
  ///
  /// - Complexity: O(1)
  public init() {
    self.init(_storage: [], count: 0)
  }

  /// Creates a new bit array containing the specified number of a single,
  /// repeated Boolean value.
  ///
  /// - Parameters:
  ///   - repeatedValue: The Boolean value to repeat.
  ///   - count: The number of times to repeat the value passed in the
  ///     `repeating` parameter. `count` must be zero or greater.
  public init(repeating repeatedValue: Bool, count: Int) {
    let wordCount = _Word.wordCount(forBitCount: UInt(count))
    var storage: [_Word] = .init(
      repeating: repeatedValue ? .allBits : .empty, count: wordCount)
    if repeatedValue, _BitPosition(count).bit != 0 {
      // Clear upper bits of last word.
      storage[wordCount - 1] = _Word(upTo: _BitPosition(count).bit)
    }
    self.init(_storage: storage, count: count)
  }
}

extension BitArray {
  /// Creates a new bit array containing the Boolean values in a sequence.
  ///
  /// - Parameter elements: The sequence of elements for the new collection.
  ///   `elements` must be finite.
  /// - Complexity: O(*count*) where *count* is the number of values in
  ///   `elements`.
  @inlinable
  public init(_ elements: some Sequence<Bool>) {
    defer { _checkInvariants() }
    if let elements = _specialize(elements, for: BitArray.self) {
      self.init(elements)
      return
    }
    if let elements = _specialize(elements, for: BitArray.SubSequence.self) {
      self.init(elements)
      return
    }
    self.init()
    self.reserveCapacity(elements.underestimatedCount)
    self.append(contentsOf: elements)
  }
  
  // Specializations

  /// Creates a new bit array containing the Boolean values in a sequence.
  ///
  /// - Parameter elements: The sequence of elements for the new collection.
  ///   `elements` must be finite.
  /// - Complexity: O(*count*) where *count* is the number of values in
  ///   `elements`.
  @inlinable
  public init(_ elements: BitArray) {
    self = elements
  }

  /// Creates a new bit array containing the Boolean values in a sequence.
  ///
  /// - Parameter elements: The sequence of elements for the new collection.
  ///   `elements` must be finite.
  /// - Complexity: O(*count*) where *count* is the number of values in
  ///   `elements`.
  public init(_ elements: BitArray.SubSequence) {
    let wordCount = _Word.wordCount(forBitCount: UInt(elements.count))
    let storage = Array(repeating: _Word.empty, count: wordCount)
    self.init(_storage: storage, count: elements.count)
    self._copy(from: elements, to: 0)
    _checkInvariants()
  }
}

extension BitArray {
  internal mutating func _prepareForReplaceSubrange(
    _ range: Range<Int>, replacementCount c: Int
  ) {
    precondition(range.lowerBound >= 0 && range.upperBound <= self.count)

    let origCount = self.count
    if range.count < c {
      _extend(by: c - range.count)
    }
  
    _copy(from: range.upperBound ..< origCount, to: range.lowerBound + c)

    if c < range.count {
      _removeLast(range.count - c)
    }
  }

  /// Replaces the specified subrange of bits with the values in the given
  /// collection.
  ///
  /// This method has the effect of removing the specified range of elements
  /// from the collection and inserting the new elements at the same location.
  /// The number of new elements need not match the number of elements being
  /// removed.
  ///
  /// If you pass a zero-length range as the `range` parameter, this method
  /// inserts the elements of `newElements` at `range.startIndex`. Calling
  /// the `insert(contentsOf:at:)` method instead is preferred.
  ///
  /// Likewise, if you pass a zero-length collection as the `newElements`
  /// parameter, this method removes the elements in the given subrange
  /// without replacement. Calling the `removeSubrange(_:)` method instead is
  /// preferred.
  ///
  /// - Parameters:
  ///   - range: The subrange of the collection to replace. The bounds of
  ///     the range must be valid indices of the collection.
  ///   - newElements: The new elements to add to the collection.
  ///
  /// - Complexity: O(*n* + *m*), where *n* is length of this collection and
  ///   *m* is the length of `newElements`. If the call to this method simply
  ///   appends the contents of `newElements` to the collection, this method is
  ///   equivalent to `append(contentsOf:)`.
  public mutating func replaceSubrange(
    _ range: Range<Int>,
    with newElements: __owned some Collection<Bool>
  ) {
    let c = newElements.count
    _prepareForReplaceSubrange(range, replacementCount: c)
    if let newElements = _specialize(newElements, for: BitArray.self) {
      _copy(from: newElements, to: range.lowerBound)
    } else if let newElements = _specialize(
      newElements, for: BitArray.SubSequence.self
    ) {
      _copy(from: newElements, to: range.lowerBound)
    } else {
      _copy(from: newElements, to: range.lowerBound ..< range.lowerBound + c)
    }
    _checkInvariants()
  }
  
  /// Replaces the specified subrange of bits with the values in the given
  /// collection.
  ///
  /// This method has the effect of removing the specified range of elements
  /// from the collection and inserting the new elements at the same location.
  /// The number of new elements need not match the number of elements being
  /// removed.
  ///
  /// If you pass a zero-length range as the `range` parameter, this method
  /// inserts the elements of `newElements` at `range.startIndex`. Calling
  /// the `insert(contentsOf:at:)` method instead is preferred.
  ///
  /// Likewise, if you pass a zero-length collection as the `newElements`
  /// parameter, this method removes the elements in the given subrange
  /// without replacement. Calling the `removeSubrange(_:)` method instead is
  /// preferred.
  ///
  /// - Parameters:
  ///   - range: The subrange of the collection to replace. The bounds of
  ///     the range must be valid indices of the collection.
  ///   - newElements: The new elements to add to the collection.
  ///
  /// - Complexity: O(*n* + *m*), where *n* is length of this collection and
  ///   *m* is the length of `newElements`. If the call to this method simply
  ///   appends the contents of `newElements` to the collection, this method is
  ///   equivalent to `append(contentsOf:)`.
  public mutating func replaceSubrange(
    _ range: Range<Int>,
    with newElements: __owned BitArray
  ) {
    replaceSubrange(range, with: newElements[...])
  }

  /// Replaces the specified subrange of bits with the values in the given
  /// collection.
  ///
  /// This method has the effect of removing the specified range of elements
  /// from the collection and inserting the new elements at the same location.
  /// The number of new elements need not match the number of elements being
  /// removed.
  ///
  /// If you pass a zero-length range as the `range` parameter, this method
  /// inserts the elements of `newElements` at `range.startIndex`. Calling
  /// the `insert(contentsOf:at:)` method instead is preferred.
  ///
  /// Likewise, if you pass a zero-length collection as the `newElements`
  /// parameter, this method removes the elements in the given subrange
  /// without replacement. Calling the `removeSubrange(_:)` method instead is
  /// preferred.
  ///
  /// - Parameters:
  ///   - range: The subrange of the collection to replace. The bounds of
  ///     the range must be valid indices of the collection.
  ///   - newElements: The new elements to add to the collection.
  ///
  /// - Complexity: O(*n* + *m*), where *n* is length of this collection and
  ///   *m* is the length of `newElements`. If the call to this method simply
  ///   appends the contents of `newElements` to the collection, this method is
  ///   equivalent to `append(contentsOf:)`.
  public mutating func replaceSubrange(
    _ range: Range<Int>,
    with newElements: __owned BitArray.SubSequence
  ) {
    _prepareForReplaceSubrange(range, replacementCount: newElements.count)
    _copy(from: newElements, to: range.lowerBound)
    _checkInvariants()
  }
}

extension BitArray {
  /// Adds a new element to the end of the bit array.
  ///
  /// - Parameter newElement: The element to append to the bit array.
  ///
  /// - Complexity: Amortized O(1), averaged over many calls to `append(_:)`
  ///    on the same bit array.
  public mutating func append(_ newElement: Bool) {
    let (word, bit) = _BitPosition(_count).split
    if bit == 0 {
      _storage.append(_Word.empty)
    }
    _count += 1
    if newElement {
      _update { handle in
        handle._mutableWords[word].value |= 1 &<< bit
      }
    }
    _checkInvariants()
  }
  
  /// Adds the elements of a sequence or collection to the end of this
  /// bit array.
  ///
  /// The collection being appended to allocates any additional necessary
  /// storage to hold the new elements.
  ///
  /// - Parameter newElements: The elements to append to the bit array.
  ///
  /// - Complexity: O(*m*), where *m* is the length of `newElements`, if
  ///    `self` is the only copy of this bit array. Otherwise O(`count` + *m*).
  public mutating func append(
    contentsOf newElements: __owned some Sequence<Bool>
  ) {
    if let newElements = _specialize(newElements, for: BitArray.self) {
      self.append(contentsOf: newElements)
      return
    }
    if let newElements = _specialize(
      newElements, for: BitArray.SubSequence.self
    ) {
      self.append(contentsOf: newElements)
      return
    }
    var it = newElements.makeIterator()
    var pos = _BitPosition(_count)
    if pos.bit > 0 {
      let (bits, count) = it._nextChunk(
        maximumCount: UInt(_Word.capacity) - pos.bit)
      guard count > 0 else { return }
      _count += count
      _update { $0._copy(bits: bits, count: count, to: pos) }
      pos.value += count
    }
    while true {
      let (bits, count) = it._nextChunk()
      guard count > 0 else { break }
      assert(pos.bit == 0)
      _storage.append(.empty)
      _count += count
      _update { $0._copy(bits: bits, count: count, to: pos) }
      pos.value += count
    }
    _checkInvariants()
  }
  
  /// Adds the elements of a sequence or collection to the end of this
  /// bit array.
  ///
  /// The collection being appended to allocates any additional necessary
  /// storage to hold the new elements.
  ///
  /// - Parameter newElements: The elements to append to the bit array.
  ///
  /// - Complexity: O(*m*), where *m* is the length of `newElements`, if
  ///    `self` is the only copy of this bit array. Otherwise O(`count` + *m*).
  public mutating func append(contentsOf newElements: BitArray) {
    _extend(by: newElements.count)
    _copy(from: newElements, to: count - newElements.count)
    _checkInvariants()
  }

  /// Adds the elements of a sequence or collection to the end of this
  /// bit array.
  ///
  /// The collection being appended to allocates any additional necessary
  /// storage to hold the new elements.
  ///
  /// - Parameter newElements: The elements to append to the bit array.
  ///
  /// - Complexity: O(*m*), where *m* is the length of `newElements`, if
  ///    `self` is the only copy of this bit array. Otherwise O(`count` + *m*).
  public mutating func append(contentsOf newElements: BitArray.SubSequence) {
    _extend(by: newElements.count)
    _copy(from: newElements, to: count - newElements.count)
    _checkInvariants()
  }
}

extension BitArray {
  /// Inserts a new element into the bit array at the specified position.
  ///
  /// The new element is inserted before the element currently at the
  /// specified index. If you pass the bit array's `endIndex` as
  /// the `index` parameter, then the new element is appended to the
  /// collection.
  ///
  ///     var bits = [false, true, true, false, true]
  ///     bits.insert(true, at: 3)
  ///     bits.insert(false, at: numbers.endIndex)
  ///
  ///     print(bits)
  ///     // Prints "[false, true, true, true, false, true, false]"
  ///
  /// - Parameter newElement: The new element to insert into the bit array.
  /// - Parameter i: The position at which to insert the new element.
  ///   `index` must be a valid index into the bit array.
  ///
  /// - Complexity: O(`count` - i), if `self` is the only copy of this bit
  ///    array. Otherwise O(`count`).
  public mutating func insert(_ newElement: Bool, at i: Int) {
    if _BitPosition(_count).bit == 0 {
      _storage.append(_Word.empty)
    }
    let c = count
    _count += 1
    _update { handle in
      handle.copy(from: i ..< c, to: i + 1)
      handle[i] = newElement
    }
    _checkInvariants()
  }
  
  /// Inserts the elements of a collection into the bit array at the specified
  /// position.
  ///
  /// The new elements are inserted before the element currently at the
  /// specified index. If you pass the collection's `endIndex` property as the
  /// `index` parameter, the new elements are appended to the collection.
  ///
  /// - Parameter newElements: The new elements to insert into the bit array.
  /// - Parameter i: The position at which to insert the new elements. `index`
  ///   must be a valid index of the collection.
  ///
  /// - Complexity: O(`self.count` + `newElements.count`).
  ///   If `i == endIndex`, this method is equivalent to `append(contentsOf:)`.
  public mutating func insert(
    contentsOf newElements: __owned some Collection<Bool>,
    at i: Int
  ) {
    precondition(i >= 0 && i <= count)
    let c = newElements.count
    guard c > 0 else { return }
    _extend(by: c)
    _copy(from: i ..< count - c, to: i + c)
    
    if let newElements = _specialize(newElements, for: BitArray.self) {
      _copy(from: newElements, to: i)
    } else if let newElements = _specialize(
      newElements, for: BitArray.SubSequence.self
    ) {
      _copy(from: newElements, to: i)
    } else {
      _copy(from: newElements, to: i ..< i + c)
    }

    _checkInvariants()
  }

  /// Inserts the elements of a collection into the bit array at the specified
  /// position.
  ///
  /// The new elements are inserted before the element currently at the
  /// specified index. If you pass the collection's `endIndex` property as the
  /// `index` parameter, the new elements are appended to the collection.
  ///
  /// - Parameter newElements: The new elements to insert into the bit array.
  /// - Parameter i: The position at which to insert the new elements. `index`
  ///   must be a valid index of the collection.
  ///
  /// - Complexity: O(`self.count` + `newElements.count`).
  ///   If `i == endIndex`, this method is equivalent to `append(contentsOf:)`.
  public mutating func insert(
    contentsOf newElements: __owned BitArray,
    at i: Int
  ) {
    insert(contentsOf: newElements[...], at: i)
  }

  /// Inserts the elements of a collection into the bit array at the specified
  /// position.
  ///
  /// The new elements are inserted before the element currently at the
  /// specified index. If you pass the collection's `endIndex` property as the
  /// `index` parameter, the new elements are appended to the collection.
  ///
  /// - Parameter newElements: The new elements to insert into the bit array.
  /// - Parameter i: The position at which to insert the new elements. `index`
  ///   must be a valid index of the collection.
  ///
  /// - Complexity: O(`self.count` + `newElements.count`).
  ///   If `i == endIndex`, this method is equivalent to `append(contentsOf:)`.
  public mutating func insert(
    contentsOf newElements: __owned BitArray.SubSequence,
    at i: Int
  ) {
    let c = newElements.count
    guard c > 0 else { return }
    _extend(by: c)
    _copy(from: i ..< count - c, to: i + c)
    _copy(from: newElements, to: i)
    _checkInvariants()
  }
}

extension BitArray {
  /// Removes and returns the element at the specified position.
  ///
  /// All the elements following the specified position are moved to close the
  /// gap.
  ///
  /// - Parameter i: The position of the element to remove. `index` must be
  ///   a valid index of the collection that is not equal to the collection's
  ///   end index.
  /// - Returns: The removed element.
  ///
  /// - Complexity: O(`count`)
  @discardableResult
  public mutating func remove(at i: Int) -> Bool {
    let result = self[i]
    _copy(from: i + 1 ..< count, to: i)
    _removeLast()
    _checkInvariants()
    return result
  }

  /// Removes the specified subrange of elements from the collection.
  ///
  /// - Parameter range: The subrange of the collection to remove. The bounds
  ///   of the range must be valid indices of the collection.
  ///
  /// - Complexity: O(`count`)
  public mutating func removeSubrange(_ range: Range<Int>) {
    precondition(
      range.lowerBound >= 0 && range.upperBound <= count,
    "Bounds out of range")
    _copy(
      from: Range(uncheckedBounds: (range.upperBound, count)),
      to: range.lowerBound)
    _removeLast(range.count)
    _checkInvariants()
  }
  
  public mutating func _customRemoveLast() -> Bool? {
    precondition(_count > 0)
    let result = self[count - 1]
    _removeLast()
    _checkInvariants()
    return result
  }

  public mutating func _customRemoveLast(_ n: Int) -> Bool {
    precondition(n >= 0 && n <= count)
    _removeLast(n)
    _checkInvariants()
    return true
  }

  /// Removes and returns the first element of the bit array.
  ///
  /// The bit array must not be empty.
  ///
  /// - Returns: The removed element.
  ///
  /// - Complexity: O(`count`)
  @discardableResult
  public mutating func removeFirst() -> Bool {
    precondition(_count > 0)
    let result = self[0]
    _copy(from: 1 ..< count, to: 0)
    _removeLast()
    _checkInvariants()
    return result
  }
  
  /// Removes the specified number of elements from the beginning of the
  /// bit array.
  ///
  /// - Parameter k: The number of elements to remove from the bit array.
  ///   `k` must be greater than or equal to zero and must not exceed the
  ///   number of elements in the bit array.
  ///
  /// - Complexity: O(`count`)
  public mutating func removeFirst(_ k: Int) {
    precondition(k >= 0 && k <= _count)
    _copy(from: k ..< count, to: 0)
    _removeLast(k)
    _checkInvariants()
  }

  /// Removes all elements from the bit array.
  ///
  /// - Parameter keepCapacity: Pass `true` to request that the collection
  ///   avoid releasing its storage. Retaining the collection's storage can
  ///   be a useful optimization when you're planning to grow the collection
  ///   again. The default value is `false`.
  ///
  /// - Complexity: O(`count`)
  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    _storage.removeAll(keepingCapacity: keepCapacity)
    _count = 0
    _checkInvariants()
  }
}
