//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension Rope {
  @inlinable
  public init(_ items: some Sequence<Element>) {
    if let items = items as? Self {
      self = items
      return
    }
    var builder = Builder()
    for item in items {
      builder.insertBeforeTip(item)
    }
    self = builder.finalize()
  }
}

extension Rope {
  @frozen // Not really! This module isn't ABI stable.
  public struct Builder {
    //       ║                    ║
    //       ║ ║                ║ ║
    //       ║ ║ ║            ║ ║ ║
    //       ║ ║ ║ ║        ║ ║ ║ ║
    //     ──╨─╨─╨─╨──╨──╨──╨─╨─╨─╨──
    // →prefixTrees→  ↑  ↑  ←suffixTrees←
    //           prefix  suffix
    
    @usableFromInline internal var _prefixTrees: [Rope] = []
    @usableFromInline internal var _prefixLeaf: Rope._Node?

    @usableFromInline internal var _prefix: Rope._Item?
    
    @usableFromInline internal var _suffix: Rope._Item?
    @usableFromInline internal var _suffixTrees: [Rope] = []

    @inlinable
    public init() {}

    @inlinable
    public var isPrefixEmpty: Bool {
      if _prefix != nil { return false }
      if let leaf = self._prefixLeaf, !leaf.isEmpty { return false }
      return _prefixTrees.isEmpty
    }
    
    @inlinable
    public var isSuffixEmpty: Bool {
      if _suffix != nil { return false }
      return _suffixTrees.isEmpty
    }

    @inlinable
    public var prefixSummary: Summary {
      var sum = Summary.zero
      for sapling in _prefixTrees {
        sum.add(sapling.summary)
      }
      if let leaf = _prefixLeaf { sum.add(leaf.summary) }
      if let item = _prefix { sum.add(item.summary) }
      return sum
    }

    @inlinable
    public var suffixSummary: Summary {
      var sum = Summary.zero
      if let item = _suffix { sum.add(item.summary) }
      for rope in _suffixTrees {
        sum.add(rope.summary)
      }
      return sum
    }

    @inlinable
    var _lastPrefixItem: Rope._Item {
      get {
        assert(!isPrefixEmpty)
        if let item = self._prefix { return item }
        if let leaf = self._prefixLeaf { return leaf.lastItem }
        return _prefixTrees.last!.root.lastItem
      }
      _modify {
        assert(!isPrefixEmpty)
        if _prefix != nil {
          yield &_prefix!
        } else if _prefixLeaf?.isEmpty == false {
          yield &_prefixLeaf!.lastItem
        } else {
          yield &_prefixTrees[_prefixTrees.count - 1].root.lastItem
        }
      }
    }
    
    @inlinable
    var _firstSuffixItem: Rope._Item {
      get {
        assert(!isSuffixEmpty)
        if let _suffix { return _suffix }
        return _suffixTrees[_suffixTrees.count - 1].root.firstItem
      }
      _modify {
        assert(!isSuffixEmpty)
        if _suffix != nil {
          yield &_suffix!
        } else {
          yield &_suffixTrees[_suffixTrees.count - 1].root.firstItem
        }
      }
    }

    @inlinable
    public func forEachElementInPrefix(
      from position: Int,
      in metric: some RopeMetric<Element>,
      _ body: (Element, Element.Index?) -> Bool
    ) -> Bool {
      var position = position
      var i = 0
      while i < _prefixTrees.count {
        let size = metric.size(of: _prefixTrees[i].summary)
        if position < size { break }
        position -= size
        i += 1
      }
      if i < _prefixTrees.count {
        guard _prefixTrees[i].forEachWhile(from: position, in: metric, body) else { return false }
        i += 1
        while i < _prefixTrees.count {
          guard _prefixTrees[i].forEachWhile({ body($0, nil) }) else { return false }
          i += 1
        }
        if let leaf = self._prefixLeaf {
          guard leaf.forEachWhile({ body($0, nil) }) else { return false }
        }
        if let item = self._prefix {
          guard body(item.value, nil) else { return false }
        }
        return true
      }

      if let leaf = self._prefixLeaf {
        let size = metric.size(of: leaf.summary)
        if position < size {
          guard leaf.forEachWhile(from: position, in: metric, body) else { return false }
          if let item = self._prefix {
            guard body(item.value, nil) else { return false }
          }
          return true
        }
        position -= size
      }
      if let item = self._prefix {
        let i = metric.index(at: position, in: item.value)
        guard body(item.value, i) else { return false}
      }
      return true
    }
    
    @inlinable
    public mutating func mutatingForEachSuffix<R>(
      _ body: (inout Element) -> R?
    ) -> R? {
      if self._suffix != nil,
         let r = body(&self._suffix!.value) {
        return r
      }
      for i in stride(from: _suffixTrees.count - 1, through: 0, by: -1) {
        if let r = self._suffixTrees[i].mutatingForEach(body) {
          return r
        }
      }
      return nil
    }
    
    @inlinable
    public mutating func insertBeforeTip(_ item: __owned Element) {
      _insertBeforeTip(Rope._Item(item))
    }
    
    @inlinable
    mutating func _insertBeforeTip(_ item: __owned Rope._Item) {
      guard !item.isEmpty else { return }
      guard var prefix = self._prefix._take() else {
        self._prefix = item
        return
      }
      var item = item
      if (prefix.isUndersized || item.isUndersized), prefix.rebalance(nextNeighbor: &item) {
        self._prefix = prefix
        return
      }
      _appendNow(prefix)
      self._prefix = item
    }
    
    @inlinable
    mutating func _appendNow(_ item: __owned Rope._Item) {
      assert(self._prefix == nil)
      assert(!item.isUndersized)
      var leaf = self._prefixLeaf._take() ?? .createLeaf()
      leaf._appendItem(item)
      if leaf.isFull {
        self._appendNow(leaf)
      } else {
        self._prefixLeaf = leaf
      }
      _invariantCheck()
    }
    
    @inlinable
    public mutating func insertBeforeTip(_ rope: __owned Rope) {
      guard rope._root != nil else { return }
      _insertBeforeTip(rope.root)
    }

    @inlinable
    public mutating func insertBeforeTip(_ items: __owned some Sequence<Element>) {
      if let items = _specialize(items, for: Rope.self) {
        self.insertBeforeTip(items)
      } else {
        for item in items {
          self.insertBeforeTip(item)
        }
      }
    }

    @inlinable
    mutating func _insertBeforeTip(_ node: __owned Rope._Node) {
      defer { _invariantCheck() }
      var node = node
      if node.height == 0 {
        if node.childCount == 1 {
          _insertBeforeTip(node.firstItem)
          return
        }
        if let item = self._prefix._take() {
          if let spawn = node.prepend(item) {
            _insertBeforeTip(node)
            _insertBeforeTip(spawn)
            return
          }
        }
        if var leaf = self._prefixLeaf._take() {
          if leaf.rebalance(nextNeighbor: &node), !leaf.isFull {
            self._prefixLeaf = leaf
            return
          }
          self._appendNow(leaf)
        }
        
        if node.isFull {
          self._appendNow(node)
        } else {
          self._prefixLeaf = node
        }
        return
      }
      
      if var prefix = self._prefix._take() {
        if !prefix.isUndersized || !node.firstItem.rebalance(prevNeighbor: &prefix) {
          self._appendNow(prefix)
        }
      }
      if let leaf = self._prefixLeaf._take() {
        _appendNow(leaf)
      }
      _appendNow(node)
    }

    @inlinable
    mutating func _appendNow(_ rope: __owned Rope._Node) {
      assert(self._prefix == nil && self._prefixLeaf == nil)
      var new = rope
      while !_prefixTrees.isEmpty {
        // Join previous saplings together until they grow at least as deep as the new one.
        var previous = _prefixTrees.removeLast()
        while previous._height < new.height {
          if _prefixTrees.isEmpty {
            previous._append(new)
            _prefixTrees.append(previous)
            return
          }
          previous.prepend(_prefixTrees.removeLast())
        }
        
        if previous._height == new.height {
          if previous.root.rebalance(nextNeighbor: &new) {
            new = previous.root
          } else {
            new = .createInner(children: previous.root, new)
          }
          continue
        }
        
        if new.isFull, !previous.root.isFull, previous._height == new.height + 1 {
          // Graft node under the last sapling, as a new child branch.
          previous.root._appendNode(new)
          new = previous.root
          continue
        }
        
        // The new seedling can be appended to the line and we're done.
        _prefixTrees.append(previous)
        break
      }
      _prefixTrees.append(Rope(root: new))
    }

    @inlinable
    public mutating func insertAfterTip(_ item: __owned Element) {
      _insertAfterTip(_Item(item))
    }

    @inlinable
    mutating func _insertAfterTip(_ item: __owned _Item) {
      guard !item.isEmpty else { return }
      if var suffixItem = self._suffix._take() {
        var item = item
        if !(suffixItem.isUndersized && item.rebalance(nextNeighbor: &suffixItem)) {
          if _suffixTrees.isEmpty {
            _suffixTrees.append(Rope(root: .createLeaf(suffixItem)))
          } else {
            _suffixTrees[_suffixTrees.count - 1].prepend(suffixItem.value)
          }
        }
      }
      self._suffix = item
    }
    
    @inlinable
    public mutating func insertAfterTip(_ rope: __owned Rope) {
      assert(_suffix == nil)
      assert(_suffixTrees.isEmpty || rope._height <= _suffixTrees.last!._height)
      _suffixTrees.append(rope)
    }
    
    @inlinable
    mutating func _insertAfterTip(_ rope: __owned Rope._Node) {
      insertAfterTip(Rope(root: rope))
    }

    @inlinable
    mutating func _insertBeforeTip(slots: Range<Int>, in node: __owned Rope._Node) {
      assert(slots.lowerBound >= 0 && slots.upperBound <= node.childCount)
      let c = slots.count
      guard c > 0 else { return }
      if c == 1 {
        if node.isLeaf {
          let item = node.readLeaf { $0.children[slots.lowerBound] }
          _insertBeforeTip(item)
        } else {
          let child = node.readInner { $0.children[slots.lowerBound] }
          _insertBeforeTip(child)
        }
        return
      }
      let copy = node.copy(slots: slots)
      _insertBeforeTip(copy)
    }

    @inlinable
    mutating func _insertAfterTip(slots: Range<Int>, in node: __owned Rope._Node) {
      assert(slots.lowerBound >= 0 && slots.upperBound <= node.childCount)
      let c = slots.count
      guard c > 0 else { return }
      if c == 1 {
        if node.isLeaf {
          let item = node.readLeaf { $0.children[slots.lowerBound] }
          _insertAfterTip(item)
        } else {
          let child = node.readInner { $0.children[slots.lowerBound] }
          _insertAfterTip(child)
        }
        return
      }
      let copy = node.copy(slots: slots)
      _insertAfterTip(copy)
    }

    @inlinable
    public mutating func finalize() -> Rope {
      // Integrate prefix & suffix chunks.
      if let suffixItem = self._suffix._take() {
        _insertBeforeTip(suffixItem)
      }
      if var prefix = self._prefix._take() {
        if !prefix.isUndersized {
          _appendNow(prefix)
        } else if !self.isPrefixEmpty {
          if !self._lastPrefixItem.rebalance(nextNeighbor: &prefix) {
            _appendNow(prefix)
          }
        } else if !self.isSuffixEmpty {
          if !self._firstSuffixItem.rebalance(prevNeighbor: &prefix) {
            _appendNow(prefix)
          }
        } else {
          // We only have the seed; allow undersized chunk
          return Rope(root: .createLeaf(prefix))
        }
      }
      assert(self._prefix == nil && self._suffix == nil)
      while let tree = _suffixTrees.popLast() {
        insertBeforeTip(tree)
      }
      // Merge all saplings, the seedling and the seed into a single rope.
      if let item = self._prefix._take() {
        _appendNow(item)
      }
      var rope = Rope(root: _prefixLeaf._take())
      while let tree = _prefixTrees.popLast() {
        rope.prepend(tree)
      }
      assert(_prefixLeaf == nil && _prefixTrees.isEmpty && _suffixTrees.isEmpty)
      rope._invariantCheck()
      return rope
    }
    
    @inlinable
    public func _invariantCheck() {
#if COLLECTIONS_INTERNAL_CHECKS
      var h = UInt8.max
      for sapling in _prefixTrees {
        precondition(sapling._height <= h)
        sapling._invariantCheck()
        h = sapling._height
      }
      if let leaf = self._prefixLeaf {
        precondition(leaf.height == 0)
        precondition(!leaf.isFull)
        leaf.invariantCheck(depth: 0, height: 0)
      }
      h = 0
      for tree in _suffixTrees.reversed() {
        precondition(tree._height >= h)
        tree._invariantCheck()
        h = tree._height
      }
#endif
      
    }
    
    public func _dump(heightLimit: Int = Int.max) {
      for i in self._prefixTrees.indices {
        print("Sapling \(i):")
        self._prefixTrees[i]._dump(heightLimit: heightLimit, firstPrefix: "  ", restPrefix: "  ")
      }
      if let leaf = self._prefixLeaf {
        print("Seedling:")
        leaf.dump(heightLimit: heightLimit, firstPrefix: "  ", restPrefix: "  ")
      }
      if let item = self._prefix {
        print("Seed:")
        print("  \(item)")
      }
      print("---")
      var i = 0
      if let item = self._suffix {
        print("Suffix \(i):")
        i += 1
        print("  \(item)")
      }
      for tree in self._suffixTrees.reversed() {
        print("Suffix \(i)")
        i += 1
        tree._dump(heightLimit: heightLimit, firstPrefix: "  ", restPrefix: "  ")
      }
    }
  }
}
