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

/// A single `-f`, `--foo`, or `--foo=bar`.
///
/// When parsing, we might see `"--foo"` or `"--foo=bar"`.
enum ParsedArgument: Equatable, CustomStringConvertible {
  /// `--foo` or `-f`
  case name(Name)
  /// `--foo=bar`
  case nameWithValue(Name, String)
  
  init<S: StringProtocol>(_ str: S) where S.SubSequence == Substring {
    let indexOfEqualSign = str.firstIndex(of: "=") ?? str.endIndex
    let (baseName, value) = (str[..<indexOfEqualSign], str[indexOfEqualSign...].dropFirst())
    let name = Name(baseName)
    self = value.isEmpty
      ? .name(name)
      : .nameWithValue(name, String(value))
  }
  
  /// An array of short arguments and their indices in the original base
  /// name, if this argument could be a combined pack of short arguments.
  ///
  /// For `subarguments` to be non-empty:
  ///
  /// 1) This must have a single-dash prefix (not `--foo`)
  /// 2) This must not have an attached value (not `-foo=bar`)
  var subarguments: [(Int, ParsedArgument)] {
    switch self {
    case .nameWithValue: return []
    case .name(let name):
      switch name {
      case .longWithSingleDash(let base):
        return base.enumerated().map {
          ($0, .name(.short($1)))
        }
      case .long, .short:
        return []
      }
    }
  }
  
  var name: Name {
    switch self {
    case let .name(n): return n
    case let .nameWithValue(n, _): return n
    }
  }
  
  var value: String? {
    switch self {
    case .name: return nil
    case let .nameWithValue(_, v): return v
    }
  }

  var description: String {
    switch self {
    case .name(let name):
      return name.synopsisString
    case .nameWithValue(let name, let value):
      return "\(name.synopsisString)=\(value)"
    }
  }
}

/// A collection of parsed command-line arguments.
///
/// This is a flat list of *values* and *options*. E.g. the
/// arguments `["--foo", "bar"]` would be parsed into
/// `[.option(.name(.long("foo"))), .value("bar")]`.
struct SplitArguments {
  struct Element: Equatable {
    enum Value: Equatable {
      case option(ParsedArgument)
      case value(String)
      /// The `--` marker
      case terminator
      
      var valueString: String? {
        switch self {
        case .value(let str):
          return str
        case .option, .terminator:
          return nil
        }
      }
    }
    
    var value: Value
    var index: Index

    static func option(_ arg: ParsedArgument, index: Index) -> Element {
      Element(value: .option(arg), index: index)
    }
    
    static func value(_ str: String, index: Index) -> Element {
      Element(value: .value(str), index: index)
    }
    
    static func terminator(index: Index) -> Element {
      Element(value: .terminator, index: index)
    }
  }
  
  /// The position of the original input string for an element.
  ///
  /// For example, if `originalInput` is `["--foo", "-vh"]`, there are index
  /// positions 0 (`--foo`) and 1 (`-vh`).
  struct InputIndex: RawRepresentable, Hashable, Comparable {
    var rawValue: Int
    
    static func <(lhs: InputIndex, rhs: InputIndex) -> Bool {
      lhs.rawValue < rhs.rawValue
    }
  }
  
  /// The position within an option for an element.
  ///
  /// Single-dash prefixed options can be treated as a whole option or as a
  /// group of individual short options. For example, the input `-vh` is split
  /// into three elements, with distinct sub-indexes:
  ///
  /// - `-vh`: `.complete`
  /// - `-v`: `.sub(0)`
  /// - `-h`: `.sub(1)`
  enum SubIndex: Hashable, Comparable {
    case complete
    case sub(Int)
    
    static func <(lhs: SubIndex, rhs: SubIndex) -> Bool {
      switch (lhs, rhs) {
      case (.complete, .sub):
        return true
      case (.sub(let l), .sub(let r)) where l < r:
        return true
      default:
        return false
      }
    }
  }
  
  /// An index into the original input and the sub-index of an element.
  struct Index: Hashable, Comparable {
    static func < (lhs: SplitArguments.Index, rhs: SplitArguments.Index) -> Bool {
      if lhs.inputIndex < rhs.inputIndex {
        return true
      } else if lhs.inputIndex == rhs.inputIndex {
        return lhs.subIndex < rhs.subIndex
      } else {
        return false
      }
    }
    
    var inputIndex: InputIndex
    var subIndex: SubIndex = .complete
    
    var completeIndex: Index {
      return Index(inputIndex: inputIndex)
    }
  }
  
  /// The parsed arguments.
  var _elements: [Element] = []
  var firstUnused: Int = 0

  /// The original array of arguments that was used to generate this instance.
  var originalInput: [String]

  /// The unused arguments represented by this instance.
  var elements: ArraySlice<Element> {
    _elements[firstUnused...]
  }
  
  var count: Int {
    elements.count
  }
}

extension SplitArguments: Equatable {}

extension SplitArguments.Element: CustomDebugStringConvertible {
  var debugDescription: String {
    switch value {
    case .option(.name(let name)):
      return name.synopsisString
    case .option(.nameWithValue(let name, let value)):
      return name.synopsisString + "; value '\(value)'"
    case .value(let value):
      return "value '\(value)'"
    case .terminator:
      return "terminator"
    }
  }
}

extension SplitArguments.Index: CustomStringConvertible {
  var description: String {
    switch subIndex {
    case .complete: return "\(inputIndex.rawValue)"
    case .sub(let sub): return "\(inputIndex.rawValue).\(sub)"
    }
  }
}

extension SplitArguments: CustomStringConvertible {
  var description: String {
    guard !isEmpty else { return "<empty>" }
    return elements
      .map { element -> String in
        switch element.value {
        case .option(.name(let name)):
          return "[\(element.index)] \(name.synopsisString)"
        case .option(.nameWithValue(let name, let value)):
          return "[\(element.index)] \(name.synopsisString)='\(value)'"
        case .value(let value):
          return "[\(element.index)] '\(value)'"
        case .terminator:
          return "[\(element.index)] --"
        }
    }
    .joined(separator: " ")
  }
}

extension SplitArguments.Element {
  var isValue: Bool {
    switch value {
    case .value: return true
    case .option, .terminator: return false
    }
  }
  
  var isTerminator: Bool {
    switch value {
    case .terminator: return true
    case .option, .value: return false
    }
  }
}

extension SplitArguments {
  /// `true` if the arguments are empty.
  var isEmpty: Bool {
    elements.isEmpty
  }

  /// `false` if the arguments are empty, or if the only remaining argument is
  /// the `--` terminator.
  var containsNonTerminatorArguments: Bool {
    if elements.isEmpty { return false }
    if elements.count > 1 { return true }
    
    if elements.first?.isTerminator == true { return false }
    else { return true }
  }

  /// Returns the original input string at the given origin, or `nil` if
  /// `origin` is a sub-index.
  func originalInput(at origin: InputOrigin.Element) -> String? {
    guard case let .argumentIndex(index) = origin else {
      return nil
    }
    return originalInput[index.inputIndex.rawValue]
  }
  
  /// Returns the position in `elements` of the given input origin.
  func position(of origin: InputOrigin.Element) -> Int? {
    guard case let .argumentIndex(index) = origin else { return nil }
    return elements.firstIndex(where: { $0.index == index })
  }
  
  /// Returns the position in `elements` of the first element after the given
  /// input origin.
  func position(after origin: InputOrigin.Element) -> Int? {
    guard case let .argumentIndex(index) = origin else { return nil }
    return elements.firstIndex(where: { $0.index > index })
  }
  
  mutating func popNext() -> (InputOrigin.Element, Element)? {
    guard let element = elements.first else { return nil }
    removeFirst()
    return (.argumentIndex(element.index), element)
  }
  
  func peekNext() -> (InputOrigin.Element, Element)? {
    guard let element = elements.first else { return nil }
    return (.argumentIndex(element.index), element)
  }
  
  func extractJoinedElement(at origin: InputOrigin.Element) -> (InputOrigin.Element, String)? {
    guard case let .argumentIndex(index) = origin else { return nil }
    
    // Joined arguments only apply when parsing the first sub-element of a
    // larger input argument.
    guard index.subIndex == .sub(0) else { return nil }

    // Rebuild the origin position for the full argument string, e.g. `-Ddebug`
    // instead of just the `-D` portion.
    let completeOrigin = InputOrigin.Element.argumentIndex(index.completeIndex)
    
    // Get the value from the original string, following the dash and short
    // option name. For example, for `-Ddebug`, drop the `-D`, leaving `debug`
    // as the value.
    let value = String(originalInput(at: completeOrigin)!.dropFirst(2))
    
    return (completeOrigin, value)
  }
  
  /// Pops the element immediately after the given index, if it is a `.value`.
  ///
  /// This is used to get the next value in `-fb name` where `name` is the
  /// value for `-f`, or `--foo name` where `name` is the value for `--foo`.
  /// If `--foo` expects a value, an input of `--foo --bar name` will return
  /// `nil`, since the option `--bar` comes before the value `name`.
  mutating func popNextElementIfValue(after origin: InputOrigin.Element) -> (InputOrigin.Element, String)? {
    // Look for the index of the input that comes from immediately after
    // `origin` in the input string. We look at the input index so that
    // packed short options can be followed, in order, by their values.
    // e.g. "-fn f-value n-value"
    guard let start = position(after: origin),
      let elementIndex = elements[start...].firstIndex(where: { $0.index.subIndex == .complete })
      else { return nil }
    
    // Only succeed if the element is a value (not prefixed with a dash)
    guard case .value(let value) = elements[elementIndex].value
      else { return nil }

    defer { remove(at: elementIndex) }
    let matchedArgumentIndex = elements[elementIndex].index
    return (.argumentIndex(matchedArgumentIndex), value)
  }
  
  /// Pops the next `.value` after the given index.
  ///
  /// This is used to get the next value in `-f -b name` where `name` is the value of `-f`.
  mutating func popNextValue(after origin: InputOrigin.Element) -> (InputOrigin.Element, String)? {
    guard let start = position(after: origin) else { return nil }
    guard let resultIndex = elements[start...].firstIndex(where: { $0.isValue }) else { return nil }
    
    defer { remove(at: resultIndex) }
    return (.argumentIndex(elements[resultIndex].index), elements[resultIndex].value.valueString!)
  }
  
  /// Pops the element after the given index as a value.
  ///
  /// This will re-interpret `.option` and `.terminator` as values, i.e.
  /// read from the `originalInput`.
  ///
  /// For an input such as `--a --b foo`, if passed the origin of `--a`,
  /// this will first pop the value `--b`, then the value `foo`.
  mutating func popNextElementAsValue(after origin: InputOrigin.Element) -> (InputOrigin.Element, String)? {
    guard let start = position(after: origin) else { return nil }
    // Elements are sorted by their `InputIndex`. Find the first `InputIndex`
    // after `origin`:
    guard let nextIndex = elements[start...].first(where: { $0.index.subIndex == .complete })?.index else { return nil }
    // Remove all elements with this `InputIndex`:
    remove(at: nextIndex)
    // Return the original input
    return (.argumentIndex(nextIndex), originalInput[nextIndex.inputIndex.rawValue])
  }
  
  /// Pops the next element if it is a value.
  ///
  /// If the current elements are `--b foo`, this will return `nil`. If the
  /// elements are `foo --b`, this will return the value `foo`.
  mutating func popNextElementIfValue() -> (InputOrigin.Element, String)? {
    guard let element = elements.first, element.isValue else { return nil }
    removeFirst()
    return (.argumentIndex(element.index), element.value.valueString!)
  }
  
  /// Finds and "pops" the next element that is a value.
  ///
  /// If the current elements are `--a --b foo`, this will remove and return
  /// `foo`.
  mutating func popNextValue() -> (Index, String)? {
    guard let idx = elements.firstIndex(where: { $0.isValue })
      else { return nil }
    let e = elements[idx]
    remove(at: idx)
    return (e.index, e.value.valueString!)
  }
  
  /// Finds and returns the next element that is a value.
  func peekNextValue() -> (Index, String)? {
    guard let idx = elements.firstIndex(where: { $0.isValue })
      else { return nil }
    let e = elements[idx]
    return (e.index, e.value.valueString!)
  }
  
  /// Removes the first element in `elements`.
  mutating func removeFirst() {
    firstUnused += 1
  }
  
  /// Removes the element at the given position.
  mutating func remove(at position: Int) {
    guard position >= firstUnused else {
      return
    }
    
    // This leaves duplicates of still to-be-used arguments in the unused
    // portion of the _elements array.
    for i in (firstUnused..<position).reversed() {
      _elements[i + 1] = _elements[i]
    }
    firstUnused += 1
  }
  
  /// Removes the elements in the given subrange.
  mutating func remove(subrange: Range<Int>) {
    var lo = subrange.startIndex
    var hi = subrange.endIndex
    
    // This leaves duplicates of still to-be-used arguments in the unused
    // portion of the _elements array.
    while lo > firstUnused {
      hi -= 1
      lo -= 1
      _elements[hi] = _elements[lo]
    }
    firstUnused += subrange.count
  }
  
  /// Removes the element(s) at the given `Index`.
  ///
  /// - Note: This may remove multiple elements.
  ///
  /// For combined _short_ arguments such as `-ab`, these will gets parsed into
  /// 3 elements: The _long with short dash_ `ab`, and 2 _short_ `a` and `b`. All of these
  /// will have the same `inputIndex` but different `subIndex`. When either of the short ones
  /// is removed, that will remove the _long with short dash_ as well. Likewise, if the
  /// _long with short dash_ is removed, that will remove both of the _short_ elements.
  mutating func remove(at position: Index) {
    guard !isEmpty else { return }
    
    // Find the first element at the given input index. Since `elements` is
    // always sorted by input index, we can leave this method if we see a
    // higher value than `position`.
    var start = elements.startIndex
    while start < elements.endIndex {
      if elements[start].index.inputIndex == position.inputIndex { break }
      if elements[start].index.inputIndex > position.inputIndex { return }
      start += 1
    }
    guard start < elements.endIndex else { return }
    
    if case .complete = position.subIndex {
      // When removing a `.complete` position, we need to remove both the
      // complete element and any sub-elements with the same input index.
      
      // Remove up to the first element where the input index doesn't match.
      let end = elements[start...].firstIndex(where: { $0.index.inputIndex != position.inputIndex })
        ?? elements.endIndex

      remove(subrange: start..<end)
    } else {
      // When removing a `.sub` (i.e. non-`.complete`) position, we need to
      // also remove the `.complete` position, if it exists. Since `.complete`
      // positions always come before sub-positions, if one exists it  will be
      // the position found as `start`.
      if elements[start].index.subIndex == .complete {
        remove(at: start)
        start += 1
      }
      
      if let sub = elements[start...].firstIndex(where: { $0.index == position }) {
        remove(at: sub)
      }
    }
  }
  
  mutating func removeAll(in origin: InputOrigin) {
    origin.forEach {
      remove(at: $0)
    }
  }
  
  /// Removes the element(s) at the given position.
  ///
  /// - Note: This may remove multiple elements.
  mutating func remove(at origin: InputOrigin.Element) {
    guard case .argumentIndex(let i) = origin else { return }
    remove(at: i)
  }
  
  func coalescedExtraElements() -> [(InputOrigin, String)] {
    let completeIndexes: [InputIndex] = elements
      .compactMap {
        guard case .complete = $0.index.subIndex else { return nil }
        return $0.index.inputIndex
    }
    
    // Now return all non-terminator elements that are either:
    // 1) `.complete`
    // 2) `.sub` but not in `completeIndexes`
    
    let extraElements = elements.filter {
      if $0.isTerminator { return false }
      
      switch $0.index.subIndex {
      case .complete:
        return true
      case .sub:
        return !completeIndexes.contains($0.index.inputIndex)
      }
    }
    return extraElements.map { element -> (InputOrigin, String) in
      let input: String
      switch element.index.subIndex {
      case .complete:
        input = originalInput[element.index.inputIndex.rawValue]
      case .sub:
        if case .option(let option) = element.value {
          input = String(describing: option)
        } else {
          // Odd case. Fall back to entire input at that index:
          input = originalInput[element.index.inputIndex.rawValue]
        }
      }
      return (.init(argumentIndex: element.index), input)
    }
  }
}

func parseIndividualArg(_ arg: String, at position: Int) throws -> [SplitArguments.Element] {
  let index = SplitArguments.Index(inputIndex: .init(rawValue: position))
  if let nonDashIdx = arg.firstIndex(where: { $0 != "-" }) {
    let dashCount = arg.distance(from: arg.startIndex, to: nonDashIdx)
    let remainder = arg[nonDashIdx..<arg.endIndex]
    switch dashCount {
    case 0:
      return [.value(arg, index: index)]
    case 1:
      // Long option:
      let parsed = try ParsedArgument(longArgWithSingleDashRemainder: remainder)
      
      // Short options:
      let parts = parsed.subarguments
      switch parts.count {
      case 0:
        // This is a '-name=value' style argument
        return [.option(parsed, index: index)]
      case 1:
        // This is a single short '-n' style argument
        return [.option(.name(.short(remainder.first!)), index: index)]
      default:
        var result: [SplitArguments.Element] = [.option(parsed, index: index)]
        for (sub, a) in parts {
          var i = index
          i.subIndex = .sub(sub)
          result.append(.option(a, index: i))
        }
        return result
      }
    case 2:
      return [.option(ParsedArgument(arg), index: index)]
    default:
      throw ParserError.invalidOption(arg)
    }
  } else {
    // All dashes
    let dashCount = arg.count
    switch dashCount {
    case 0, 1:
      // Empty string or single dash
      return [.value(arg, index: index)]
    case 2:
      // We found the 1st "--". All the remaining are positional.
      return [.terminator(index: index)]
    default:
      throw ParserError.invalidOption(arg)
    }
  }
}

extension SplitArguments {
  /// Parses the given input into an array of `Element`.
  ///
  /// - Parameter arguments: The input from the command line.
  init(arguments: [String]) throws {
    self.init(originalInput: arguments)
    
    var position = 0
    var args = arguments[...]
    argLoop: while let arg = args.popFirst() {
      defer {
        position += 1
      }
      
      let parsedElements = try parseIndividualArg(arg, at: position)
      _elements.append(contentsOf: parsedElements)
      if parsedElements.first!.isTerminator {
        break
      }
    }
    
    for arg in args {
      let i = Index(inputIndex: InputIndex(rawValue: position))
      _elements.append(.value(arg, index: i))
      position += 1
    }
  }
}

private extension ParsedArgument {
  init(longArgRemainder remainder: Substring) throws {
    try self.init(longArgRemainder: remainder, makeName: { Name.long(String($0)) })
  }
  
  init(longArgWithSingleDashRemainder remainder: Substring) throws {
    try self.init(longArgRemainder: remainder, makeName: {
      /// If an argument has a single dash and single character,
      /// followed by a value, treat it as a short name.
      ///     `-c=1`      ->  `Name.short("c")`
      /// Otherwise, treat it as a long name with single dash.
      ///     `-count=1`  ->  `Name.longWithSingleDash("count")`
      $0.count == 1 ? Name.short($0.first!) : Name.longWithSingleDash(String($0))
    })
  }
  
  init(longArgRemainder remainder: Substring, makeName: (Substring) -> Name) throws {
    if let equalIdx = remainder.firstIndex(of: "=") {
      let name = remainder[remainder.startIndex..<equalIdx]
      guard !name.isEmpty else {
        throw ParserError.invalidOption(makeName(remainder).synopsisString)
      }
      let after = remainder.index(after: equalIdx)
      let value = String(remainder[after..<remainder.endIndex])
      self = .nameWithValue(makeName(name), value)
    } else {
      self = .name(makeName(remainder))
    }
  }
  
  static func shortOptions(shortArgRemainder: Substring) throws -> [ParsedArgument] {
    var result: [ParsedArgument] = []
    var remainder = shortArgRemainder
    while let char = remainder.popFirst() {
      guard char.isLetter || char.isNumber else {
        throw ParserError.nonAlphanumericShortOption(char)
      }
      result.append(.name(.short(char)))
    }
    return result
  }
}
