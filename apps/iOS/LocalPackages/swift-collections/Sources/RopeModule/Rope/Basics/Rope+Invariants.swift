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

extension Rope {
  @inlinable @inline(__always)
  public func _invariantCheck() {
#if COLLECTIONS_INTERNAL_CHECKS
    _root?.invariantCheck(depth: 0, height: root.height, recursive: true)
#endif
  }
}

extension Rope._Node {
#if COLLECTIONS_INTERNAL_CHECKS
  @usableFromInline
  internal func invariantCheck(depth: UInt8, height: UInt8, recursive: Bool = true) {
    precondition(height == self.height, "Mismatching rope height")
    if isLeaf {
      precondition(self.childCount <= Summary.maxNodeSize, "Oversized leaf")
      precondition(height == 0, "Leaf with height > 0")
      precondition(depth == 0 || self.childCount >= Summary.minNodeSize, "Undersized leaf")
      
      let sum: Summary = readLeaf {
        $0.children.reduce(into: .zero) { $0.add($1.summary) }
      }
      precondition(self.summary == sum, "Mismatching summary")
      
      guard recursive else { return }
      readLeaf { leaf in
        for child in leaf.children {
          child.value.invariantCheck()
        }
      }
      return
    }
    
    precondition(self.childCount <= Summary.maxNodeSize, "Oversized node")
    if depth == 0 {
      precondition(self.childCount > 1, "Undersize root node")
    } else {
      precondition(self.childCount >= Summary.minNodeSize, "Undersized internal node")
    }
    
    let sum: Summary = readInner {
      $0.children.reduce(into: .zero) { $0.add($1.summary) }
    }
    precondition(self.summary == sum, "Mismatching summary")
    
    guard recursive else { return }
    readInner { h in
      for child in h.children {
        child.invariantCheck(depth: depth + 1, height: height - 1, recursive: true)
      }
    }
  }
#else
  @inlinable @inline(__always)
  internal func invariantCheck(depth: UInt8, height: UInt8, recursive: Bool = true) {
    // Do nothing.
  }
#endif
}
