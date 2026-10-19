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

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public func _invariantCheck() {
#if COLLECTIONS_INTERNAL_CHECKS
    _rope._invariantCheck()
    let allowUndersize = _rope.isSingleton
    
    var state = _CharacterRecognizer()
    for chunk in _rope {
      precondition(allowUndersize || !chunk.isUndersized, "Undersized chunk")
      let (characters, prefix, suffix) = state.edgeCounts(consuming: chunk.string)
      precondition(
        chunk.prefixCount == prefix,
        "Inconsistent position of first grapheme break in chunk")
      precondition(
        chunk.suffixCount == suffix,
        "Inconsistent position of last grapheme break in chunk")
      precondition(
        chunk.characterCount == characters,
        "Inconsistent character count in chunk")
    }
#endif
  }
}
