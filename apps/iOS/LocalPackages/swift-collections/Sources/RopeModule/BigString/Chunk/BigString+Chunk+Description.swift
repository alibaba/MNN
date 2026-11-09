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
extension BigString._Chunk: CustomStringConvertible {
  var description: String {
    let counts = """
          ❨\(utf8Count)⋅\(utf16Count)⋅\(unicodeScalarCount)⋅\(characterCount)❩
          """._rpad(to: 17)
    let d = _succinctContents(maxLength: 10)
    return "Chunk(\(_identity)\(counts) \(d))"
  }

  var _identity: String {
#if _pointerBitWidth(_64)
    // Let's use the second word of the string representation as the identity; it contains
    // the String's storage reference (if any).
    let b = unsafeBitCast(self.string, to: (UInt64, UInt64).self)
    return "@" + String(b.1, radix: 16)._rpad(to: 17)
#elseif _pointerBitWidth(_32)
    return ""
#else
#error("Unexpected pointer bit width")
#endif
  }

  func _succinctContents(maxLength c: Int) -> String {
    /// 4+"short"-1
    /// 0+"longer...string"-1
    let pc = String(prefixCount)._lpad(to: 3)
    let sc = String(suffixCount)

    let s = String(wholeCharacters)
    if s.isEmpty {
      return "\(pc)+...-\(sc)"
    }
    var result = "\(pc)+\""
    var state = _CharacterRecognizer(consuming: result)

    let i = result._appendQuotedProtectingLeft(s, with: &state, maxLength: c)
    let j = s.index(s.endIndex, offsetBy: -c, limitedBy: string.startIndex) ?? string.startIndex

    if i < j {
      result._appendProtectingRight("...", with: &state)
      result._appendQuotedProtectingLeft(String(s[j...]), with: &state)
    } else if i < s.endIndex {
      let k = s._index(roundingDown: i)
      if i == k {
        result._appendQuotedProtectingLeft(String(s[i...]), with: &state)
      } else if s.index(after: k) < s.endIndex {
        result._appendProtectingRight("...", with: &state)
        result._appendQuotedProtectingLeft(String(s[s.index(after: k)...]), with: &state)
      } else {
        let suffix = String(s[i...].unicodeScalars.suffix(3))
        result._appendProtectingRight("...", with: &state)
        result._appendQuotedProtectingLeft(suffix, with: &state)
      }
    }
    result._appendProtectingRight("\"-\(sc)", with: &state)
    return result
  }
}
