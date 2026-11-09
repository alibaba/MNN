//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension StringProtocol where SubSequence == Substring {
  func wrapped(to columns: Int, wrappingIndent: Int = 0) -> String {
    let columns = columns - wrappingIndent
    guard columns > 0 else {
      // Skip wrapping logic if the number of columns is less than 1 in release
      // builds and assert in debug builds.
      assertionFailure("`columns - wrappingIndent` should be always be greater than 0.")
      return ""
    }

    var result: [Substring] = []
    
    var currentIndex = startIndex
    
    while true {
      let nextChunk = self[currentIndex...].prefix(columns)
      if let lastLineBreak = nextChunk.lastIndex(of: "\n") {
        result.append(contentsOf: self[currentIndex..<lastLineBreak].split(separator: "\n", omittingEmptySubsequences: false))
        currentIndex = index(after: lastLineBreak)
      } else if nextChunk.endIndex == self.endIndex {
        result.append(self[currentIndex...])
        break
      } else if let lastSpace = nextChunk.lastIndex(of: " ") {
        result.append(self[currentIndex..<lastSpace])
        currentIndex = index(after: lastSpace)
      } else if let nextSpace = self[currentIndex...].firstIndex(of: " ") {
        result.append(self[currentIndex..<nextSpace])
        currentIndex = index(after: nextSpace)
      } else {
        result.append(self[currentIndex...])
        break
      }
    }
    
    return result
      .map { $0.isEmpty ? $0 : String(repeating: " ", count: wrappingIndent) + $0 }
      .joined(separator: "\n")
  }
  
  /// Returns this string prefixed using a camel-case style.
  ///
  /// Example:
  ///
  ///     "hello".addingIntercappedPrefix("my")
  ///     // myHello
  func addingIntercappedPrefix(_ prefix: String) -> String {
    guard let firstChar = first else { return prefix }
    return "\(prefix)\(firstChar.uppercased())\(self.dropFirst())"
  }
  
  /// Returns this string prefixed using kebab-, snake-, or camel-case style
  /// depending on what can be detected from the string.
  ///
  /// Examples:
  ///
  ///     "hello".addingPrefixWithAutodetectedStyle("my")
  ///     // my-hello
  ///     "hello_there".addingPrefixWithAutodetectedStyle("my")
  ///     // my_hello_there
  ///     "hello-there".addingPrefixWithAutodetectedStyle("my")
  ///     // my-hello-there
  ///     "helloThere".addingPrefixWithAutodetectedStyle("my")
  ///     // myHelloThere
  func addingPrefixWithAutodetectedStyle(_ prefix: String) -> String {
    if contains("-") {
      return "\(prefix)-\(self)"
    } else if contains("_") {
      return "\(prefix)_\(self)"
    } else if first?.isLowercase == true && contains(where: { $0.isUppercase }) {
      return addingIntercappedPrefix(prefix)
    } else {
      return "\(prefix)-\(self)"
    }
  }
  
  /// Returns a new string with the camel-case-based words of this string
  /// split by the specified separator.
  ///
  /// Examples:
  ///
  ///     "myProperty".convertedToSnakeCase()
  ///     // my_property
  ///     "myURLProperty".convertedToSnakeCase()
  ///     // my_url_property
  ///     "myURLProperty".convertedToSnakeCase(separator: "-")
  ///     // my-url-property
  func convertedToSnakeCase(separator: Character = "_") -> String {
    guard !isEmpty else { return "" }
    var result = ""
    // Whether we should append a separator when we see a uppercase character.
    var separateOnUppercase = true
    for index in indices {
      let nextIndex = self.index(after: index)
      let character = self[index]
      if character.isUppercase {
        if separateOnUppercase && !result.isEmpty {
          // Append the separator.
          result += "\(separator)"
        }
        // If the next character is uppercase and the next-next character is lowercase, like "L" in "URLSession", we should separate words.
        separateOnUppercase = nextIndex < endIndex && self[nextIndex].isUppercase && self.index(after: nextIndex) < endIndex && self[self.index(after: nextIndex)].isLowercase
      } else {
        // If the character is `separator`, we do not want to append another separator when we see the next uppercase character.
        separateOnUppercase = character != separator
      }
      // Append the lowercased character.
      result += character.lowercased()
    }
    return result
  }
  
  /// Returns the edit distance between this string and the provided target string.
  ///
  /// Uses the Levenshtein distance algorithm internally.
  ///
  /// See: https://en.wikipedia.org/wiki/Levenshtein_distance
  ///
  /// Examples:
  ///
  ///     "kitten".editDistance(to: "sitting")
  ///     // 3
  ///     "bar".editDistance(to: "baz")
  ///     // 1
  func editDistance(to target: String) -> Int {
    let rows = self.count
    let columns = target.count
    
    if rows <= 0 || columns <= 0 {
      return Swift.max(rows, columns)
    }
    
    // Trim common prefix and suffix
    var selfStartTrim = self.startIndex
    var targetStartTrim = target.startIndex
    while selfStartTrim < self.endIndex &&
          targetStartTrim < target.endIndex &&
            self[selfStartTrim] == target[targetStartTrim] {
      self.formIndex(after: &selfStartTrim)
      target.formIndex(after: &targetStartTrim)
    }

    var selfEndTrim = self.endIndex
    var targetEndTrim = target.endIndex

    while selfEndTrim > selfStartTrim &&
          targetEndTrim > targetStartTrim {
      let selfIdx = self.index(before: selfEndTrim)
      let targetIdx = target.index(before: targetEndTrim)

      guard self[selfIdx] == target[targetIdx] else {
        break
      }

      selfEndTrim = selfIdx
      targetEndTrim = targetIdx
    }

    // Equal strings
    guard !(selfStartTrim == self.endIndex &&
          targetStartTrim == target.endIndex) else {
      return 0
    }
    
    // After trimming common prefix and suffix, self is empty.
    guard selfStartTrim < selfEndTrim else {
      return target.distance(from: targetStartTrim,
                             to: targetEndTrim)
    }

    // After trimming common prefix and suffix, target is empty.
    guard targetStartTrim < targetEndTrim else {
      return distance(from: selfStartTrim,
                      to: selfEndTrim)
    }

    let newSelf = self[selfStartTrim..<selfEndTrim]
    let newTarget = target[targetStartTrim..<targetEndTrim]

    let m = newSelf.count
    let n = newTarget.count

    // Initialize the levenshtein matrix with only two rows
    // current and previous.
    var previousRow = [Int](repeating: 0, count: n + 1)
    var currentRow = [Int](0...n)

    var sourceIdx = newSelf.startIndex
    for i in 1...m {
      swap(&previousRow, &currentRow)
      currentRow[0] = i

      var targetIdx = newTarget.startIndex
      for j in 1...n {
        // If characteres are equal for the levenshtein algorithm the
        // minimum will always be the substitution cost, so we can fast
        // path here in order to avoid min calls.
        if newSelf[sourceIdx] == newTarget[targetIdx] {
          currentRow[j] = previousRow[j - 1]
        } else {
          let deletion = previousRow[j]
          let insertion = currentRow[j - 1]
          let substitution = previousRow[j - 1]
          currentRow[j] = Swift.min(deletion, Swift.min(insertion, substitution)) + 1
        }
        // j += 1
        newTarget.formIndex(after: &targetIdx)
      }
      // i += 1
      newSelf.formIndex(after: &sourceIdx)
    }
    return currentRow[n]
  }
  
  func indentingEachLine(by n: Int) -> String {
    let lines = self.split(separator: "\n", omittingEmptySubsequences: false)
    let spacer = String(repeating: " ", count: n)
    return lines.map {
      $0.isEmpty ? $0 : spacer + $0
    }.joined(separator: "\n")
  }
  
  func hangingIndentingEachLine(by n: Int) -> String {
    let lines = self.split(
      separator: "\n",
      maxSplits: 1,
      omittingEmptySubsequences: false)
    guard lines.count == 2 else { return lines.joined(separator: "") }
    return "\(lines[0])\n\(lines[1].indentingEachLine(by: n))"
  }
  
  var nonEmpty: Self? {
    isEmpty ? nil : self
  }
}
