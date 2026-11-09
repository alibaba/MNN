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

/// The resulting values after parsing the command-line arguments.
///
/// This is a flat key-value list of values.
struct ParsedValues {
  struct Element {
    var key: InputKey
    var value: Any?
    /// Where in the input that this came from.
    var inputOrigin: InputOrigin
    fileprivate var shouldClearArrayIfParsed = true
  }
  
  /// These are the parsed key-value pairs.
  var elements: [InputKey: Element] = [:]
  
  /// This is the *original* array of arguments that this was parsed from.
  ///
  /// This is used for error output generation.
  var originalInput: [String]
  
  /// Any arguments that are captured into an `.allUnrecognized` argument.
  var capturedUnrecognizedArguments = SplitArguments(originalInput: [])
}

extension ParsedValues {
  mutating func set(_ new: Any?, forKey key: InputKey, inputOrigin: InputOrigin) {
    set(Element(key: key, value: new, inputOrigin: inputOrigin))
  }
  
  mutating func set(_ element: Element) {
    if let e = elements[element.key] {
      // Merge the source values. We need to keep track
      // of any previous source indexes we have used for
      // this key.
      var element = element
      element.inputOrigin.formUnion(e.inputOrigin)
      elements[element.key] = element
    } else {
      elements[element.key] = element
    }
  }
  
  func element(forKey key: InputKey) -> Element? {
    elements[key]
  }
  
  mutating func update<A>(forKey key: InputKey, inputOrigin: InputOrigin, initial: A, closure: (inout A) -> Void) {
    var e = element(forKey: key) ?? Element(key: key, value: initial, inputOrigin: InputOrigin())
    var v = (e.value as? A ) ?? initial
    closure(&v)
    e.value = v
    e.inputOrigin.formUnion(inputOrigin)
    set(e)
  }
  
  mutating func update<A>(forKey key: InputKey, inputOrigin: InputOrigin, initial: [A], closure: (inout [A]) -> Void) {
    var e = element(forKey: key) ?? Element(key: key, value: initial, inputOrigin: InputOrigin())
    var v = (e.value as? [A] ) ?? initial
    // The first time a value is parsed from command line, empty array of any default values.
    if e.shouldClearArrayIfParsed {
      v.removeAll()
      e.shouldClearArrayIfParsed = false
    }
    closure(&v)
    e.value = v
    e.inputOrigin.formUnion(inputOrigin)
    set(e)
  }
}
