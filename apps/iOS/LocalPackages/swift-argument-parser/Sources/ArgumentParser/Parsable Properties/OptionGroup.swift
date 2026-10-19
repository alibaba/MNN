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

/// A wrapper that transparently includes a parsable type.
///
/// Use an option group to include a group of options, flags, or arguments
/// declared in a parsable type.
///
/// ```swift
/// struct GlobalOptions: ParsableArguments {
///     @Flag(name: .shortAndLong)
///     var verbose: Bool
///
///     @Argument var values: [Int]
/// }
///
/// struct Options: ParsableArguments {
///     @Option var name: String
///     @OptionGroup var globals: GlobalOptions
/// }
/// ```
///
/// The flag and positional arguments declared as part of `GlobalOptions` are
/// included when parsing `Options`.
@propertyWrapper
public struct OptionGroup<Value: ParsableArguments>: Decodable, ParsedWrapper {
  internal var _parsedValue: Parsed<Value>
  internal var _visibility: ArgumentVisibility

  // FIXME: Adding this property works around the crasher described in
  // https://github.com/apple/swift-argument-parser/issues/338
  internal var _dummy: Bool = false
  
  /// The title to use in the help screen for this option group.
  public var title: String = ""

  internal init(_parsedValue: Parsed<Value>) {
    self._parsedValue = _parsedValue
    self._visibility = .default
  }
  
  public init(from _decoder: Decoder) throws {
    if let d = _decoder as? SingleValueDecoder,
      let value = try? d.previousValue(Value.self)
    {
      self.init(_parsedValue: .value(value))
    } else {
      try self.init(_decoder: _decoder)
      if let d = _decoder as? SingleValueDecoder {
        d.saveValue(wrappedValue)
      }
    }
    
    do {
      try wrappedValue.validate()
    } catch {
      throw ParserError.userValidationError(error)
    }
  }

  /// Creates a property that represents another parsable type, using the
  /// specified title and visibility.
  ///
  /// - Parameters:
  ///   - title: A title for grouping this option group's members in your
  ///     command's help screen. If `title` is empty, the members will be
  ///     displayed alongside the other arguments, flags, and options declared
  ///     by your command.
  ///   - visibility: The visibility to use for the entire option group.
  public init(
    title: String = "",
    visibility: ArgumentVisibility = .default
  ) {
    self.init(_parsedValue: .init { parentKey in
      var args = ArgumentSet(Value.self, visibility: .private, parent: parentKey)
      if !title.isEmpty {
        args.content.withEach {
          $0.help.parentTitle = title
        }
      }
      return args
    })
    self._visibility = visibility
    self.title = title
  }

  /// The value presented by this property wrapper.
  public var wrappedValue: Value {
    get {
      switch _parsedValue {
      case .value(let v):
        return v
      case .definition:
        fatalError(directlyInitializedError)
      }
    }
    set {
      _parsedValue = .value(newValue)
    }
  }
}

extension OptionGroup: Sendable where Value: Sendable {}

extension OptionGroup: CustomStringConvertible {
  public var description: String {
    switch _parsedValue {
    case .value(let v):
      return String(describing: v)
    case .definition:
      return "OptionGroup(*definition*)"
    }
  }
}

// Experimental use with caution
extension OptionGroup {
  @available(*, deprecated, renamed: "init(visibility:)")
  public init(_hiddenFromHelp: Bool) {
    self.init(visibility: .hidden)
  }
  
  /// Creates a property that represents another parsable type.
  @available(*, deprecated, renamed: "init(visibility:)")
  @_disfavoredOverload
  public init() {
    self.init(visibility: .default)
  }
}

// MARK: Deprecated

extension OptionGroup {
  @_disfavoredOverload
  @available(*, deprecated, renamed: "init(title:visibility:)")
  public init(
    visibility _visibility: ArgumentVisibility = .default
  ) {
    self.init(title: "", visibility: _visibility)
  }
}
