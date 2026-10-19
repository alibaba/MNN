//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension BitSet {
  public static func random(upTo limit: Int) -> BitSet {
    var rng = SystemRandomNumberGenerator()
    return random(upTo: limit, using: &rng)
  }

  public static func random(
    upTo limit: Int,
    using rng: inout some RandomNumberGenerator
  ) -> BitSet {
    precondition(limit >= 0, "Invalid limit value")
    guard limit > 0 else { return BitSet() }
    let (w, b) = _UnsafeHandle.Index(limit).endSplit
    var words = (0 ... w).map { _ in _Word(rng.next() as UInt) }
    words[w].formIntersection(_Word(upTo: b))
    return BitSet(_words: words)
  }
}
