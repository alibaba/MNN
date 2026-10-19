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

/// Specifies where a given input came from.
///
/// When reading from the command line, a value might originate from a single
/// index, multiple indices, or from part of an index. For this command:
///
///     struct Example: ParsableCommand {
///         @Flag(name: .short) var verbose = false
///         @Flag(name: .short) var expert = false
///
///         @Option var count: Int
///     }
///
/// ...with this usage:
///
///     $ example -ve --count 5
///
/// The parsed value for the `count` property will come from indices `1` and
/// `2`, while the value for `verbose` will come from index `1`, sub-index `0`.
struct InputOrigin: Equatable, ExpressibleByArrayLiteral {
  enum Element: Comparable, Hashable {
    /// The input value came from a property's default value, not from a
    /// command line argument.
    case defaultValue
    
    /// The input value came from the specified index in the argument string.
    case argumentIndex(SplitArguments.Index)
    
    var baseIndex: Int? {
      switch self {
      case .defaultValue:
        return nil
      case .argumentIndex(let i):
        return i.inputIndex.rawValue
      }
    }
    
    var subIndex: Int? {
      switch self {
      case .defaultValue:
        return nil
      case .argumentIndex(let i):
        switch i.subIndex {
        case .complete: return nil
        case .sub(let n): return n
        }
      }
    }
  }
  
  private var _elements: Set<Element> = []
  var elements: [Element] {
    Array(_elements).sorted()
  }
  
  init() {
  }
  
  init(elements: [Element]) {
    _elements = Set(elements)
  }
  
  init(element: Element) {
    _elements = Set([element])
  }
  
  init(arrayLiteral elements: Element...) {
    self.init(elements: elements)
  }

  init(argumentIndex: SplitArguments.Index) {
    self.init(element: .argumentIndex(argumentIndex))
  }
  
  mutating func insert(_ other: Element) {
    guard !_elements.contains(other) else { return }
    _elements.insert(other)
  }
  
  func inserting(_ other: Element) -> Self {
    guard !_elements.contains(other) else { return self }
    var result = self
    result.insert(other)
    return result
  }
  
  mutating func formUnion(_ other: InputOrigin) {
    _elements.formUnion(other._elements)
  }

  func forEach(_ closure: (Element) -> Void) {
    _elements.forEach(closure)
  }
}

extension InputOrigin {
  var isDefaultValue: Bool {
    return _elements.count == 1 && _elements.first == .defaultValue
  }
}

extension InputOrigin.Element {
  static func < (lhs: Self, rhs: Self) -> Bool {
    switch (lhs, rhs) {
    case (.argumentIndex(let l), .argumentIndex(let r)):
      return l < r
    case (.argumentIndex, .defaultValue):
      return true
    case (.defaultValue, _):
      return false
    }
  }
}
