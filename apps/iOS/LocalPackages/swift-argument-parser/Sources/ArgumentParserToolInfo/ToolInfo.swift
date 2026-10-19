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

fileprivate extension Collection {
  /// - returns: A non-empty collection or `nil`.
  var nonEmpty: Self? { isEmpty ? nil : self }
}

/// Header used to validate serialization version of an encoded ToolInfo struct.
public struct ToolInfoHeader: Decodable {
  /// A sentinel value indicating the version of the ToolInfo struct used to
  /// generate the serialized form.
  public var serializationVersion: Int

  public init(serializationVersion: Int) {
    self.serializationVersion = serializationVersion
  }
}

/// Top-level structure containing serialization version and information for all
/// commands in a tool.
public struct ToolInfoV0: Codable, Hashable {
  /// A sentinel value indicating the version of the ToolInfo struct used to
  /// generate the serialized form.
  public var serializationVersion = 0
  /// Root command of the tool.
  public var command: CommandInfoV0

  public init(command: CommandInfoV0) {
    self.command = command
  }
}

/// All information about a particular command, including arguments and
/// subcommands.
public struct CommandInfoV0: Codable, Hashable {
  /// Super commands and tools.
  public var superCommands: [String]?

  /// Name used to invoke the command.
  public var commandName: String
  /// Short description of the command's functionality.
  public var abstract: String?
  /// Extended description of the command's functionality.
  public var discussion: String?

  /// Optional name of the subcommand invoked when the command is invoked with
  /// no arguments.
  public var defaultSubcommand: String?
  /// List of nested commands.
  public var subcommands: [CommandInfoV0]?
  /// List of supported arguments.
  public var arguments: [ArgumentInfoV0]?

  public init(
    superCommands: [String],
    commandName: String,
    abstract: String,
    discussion: String,
    defaultSubcommand: String?,
    subcommands: [CommandInfoV0],
    arguments: [ArgumentInfoV0]
  ) {
    self.superCommands = superCommands.nonEmpty

    self.commandName = commandName
    self.abstract = abstract.nonEmpty
    self.discussion = discussion.nonEmpty

    self.defaultSubcommand = defaultSubcommand?.nonEmpty
    self.subcommands = subcommands.nonEmpty
    self.arguments = arguments.nonEmpty
  }
}

/// All information about a particular argument, including display names and
/// options.
public struct ArgumentInfoV0: Codable, Hashable {
  /// Information about an argument's name.
  public struct NameInfoV0: Codable, Hashable {
    /// Kind of prefix of an argument's name.
    public enum KindV0: String, Codable, Hashable {
      /// A multi-character name preceded by two dashes.
      case long
      /// A single character name preceded by a single dash.
      case short
      /// A multi-character name preceded by a single dash.
      case longWithSingleDash
    }

    /// Kind of prefix the NameInfoV0 describes.
    public var kind: KindV0
    /// Single or multi-character name of the argument.
    public var name: String

    public init(kind: NameInfoV0.KindV0, name: String) {
      self.kind = kind
      self.name = name
    }
  }

  /// Kind of argument.
  public enum KindV0: String, Codable, Hashable {
    /// Argument specified as a bare value on the command line.
    case positional
    /// Argument specified as a value prefixed by a `--flag` on the command line.
    case option
    /// Argument specified only as a `--flag` on the command line.
    case flag
  }

  /// Kind of argument the ArgumentInfo describes.
  public var kind: KindV0

  /// Argument should appear in help displays.
  public var shouldDisplay: Bool
  /// Custom name of argument's section.
  public var sectionTitle: String?
  
  /// Argument can be omitted.
  public var isOptional: Bool
  /// Argument can be specified multiple times.
  public var isRepeating: Bool

  /// All names of the argument.
  public var names: [NameInfoV0]?
  /// The best name to use when referring to the argument in help displays.
  public var preferredName: NameInfoV0?

  /// Name of argument's value.
  public var valueName: String?
  /// Default value of the argument is none is specified on the command line.
  public var defaultValue: String?
  // NOTE: this property will not be renamed to 'allValueStrings' to avoid
  // breaking compatibility with the current serialized format.
  /// List of all valid values.
  public var allValues: [String]?
  /// List of all valid values.
  public var allValueStrings: [String]? {
    get { self.allValues }
    set { self.allValues = newValue }
  }

  /// Short description of the argument's functionality.
  public var abstract: String?
  /// Extended description of the argument's functionality.
  public var discussion: String?

  public init(
    kind: KindV0,
    shouldDisplay: Bool,
    sectionTitle: String?,
    isOptional: Bool,
    isRepeating: Bool,
    names: [NameInfoV0]?,
    preferredName: NameInfoV0?,
    valueName: String?,
    defaultValue: String?,
    allValues: [String]?,
    abstract: String?,
    discussion: String?
  ) {
    self.kind = kind

    self.shouldDisplay = shouldDisplay
    self.sectionTitle = sectionTitle
    
    self.isOptional = isOptional
    self.isRepeating = isRepeating

    self.names = names?.nonEmpty
    self.preferredName = preferredName

    self.valueName = valueName?.nonEmpty
    self.defaultValue = defaultValue?.nonEmpty
    self.allValueStrings = allValues?.nonEmpty

    self.abstract = abstract?.nonEmpty
    self.discussion = discussion?.nonEmpty
  }
}

