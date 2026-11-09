//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------*- openbsd -*-===//
//
// This source file contains descriptions of mandoc syntax tree nodes derived
// from their original descriptions in the mandoc source found here:
// https://github.com/openbsd/src/blob/master/share/man/man7/mdoc.7
//
// $Id: LICENSE,v 1.22 2021/09/19 11:02:09 schwarze Exp $
//
// With the exceptions noted below, all non-trivial files contained
// in the mandoc toolkit are protected by the Copyright of the following
// developers:
//
// Copyright (c) 2008-2012, 2014 Kristaps Dzonsons <kristaps@bsd.lv>
// Copyright (c) 2010-2021 Ingo Schwarze <schwarze@openbsd.org>
// Copyright (c) 1999, 2004, 2017 Marc Espie <espie@openbsd.org>
// Copyright (c) 2009, 2010, 2011, 2012 Joerg Sonnenberger <joerg@netbsd.org>
// Copyright (c) 2013 Franco Fichtner <franco@lastsummer.de>
// Copyright (c) 2014 Baptiste Daroussin <bapt@freebsd.org>
// Copyright (c) 2016 Ed Maste <emaste@freebsd.org>
// Copyright (c) 2017 Michael Stapelberg <stapelberg@debian.org>
// Copyright (c) 2017 Anthony Bentley <bentley@openbsd.org>
// Copyright (c) 1998, 2004, 2010, 2015 Todd C. Miller <Todd.Miller@courtesan.com>
// Copyright (c) 2008, 2017 Otto Moerbeek <otto@drijf.net>
// Copyright (c) 2004 Ted Unangst <tedu@openbsd.org>
// Copyright (c) 1994 Christos Zoulas <christos@netbsd.org>
// Copyright (c) 2003, 2007, 2008, 2014 Jason McIntyre <jmc@openbsd.org>
//
// See the individual files for information about who contributed
// to which file during which years.
//
//
// The mandoc distribution as a whole is distributed by its developers
// under the following license:
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//
// The following files included from outside sources are protected by
// other people's Copyright and are distributed under various 2-clause
// and 3-clause BSD licenses; see these individual files for details.
//
// soelim.c, soelim.1:
// Copyright (c) 2014 Baptiste Daroussin <bapt@freebsd.org>
//
// compat_err.c, compat_fts.c, compat_fts.h,
// compat_getsubopt.c, compat_strcasestr.c, compat_strsep.c,
// man.1:
// Copyright (c) 1989,1990,1993,1994 The Regents of the University of California
//
// compat_stringlist.c, compat_stringlist.h:
// Copyright (c) 1994 Christos Zoulas <christos@netbsd.org>
//
// See https://mandoc.bsd.lv/LICENSE for license information
//
//===----------------------------------------------------------------------===//

fileprivate extension Array {
  mutating func append(optional newElement: Element?) {
    if let newElement = newElement {
      append(newElement)
    }
  }
}

/// `MDocMacroProtocol` defines the properties required to serialize a
/// strongly-typed mdoc macro to the raw format.
public protocol MDocMacroProtocol: MDocASTNode {
  /// The underlying `mdoc` macro string; used during serialization.
  static var kind: String { get }
  /// The arguments passed to the underlying `mdoc` macro; used during
  /// serialization.
  var arguments: [MDocASTNode] { get set }
}

extension MDocMacroProtocol {
  /// Append unchecked arguments to a `MDocMacroProtocol`.
  public func withUnsafeChildren(nodes: [MDocASTNode]) -> Self {
    var copy = self
    copy.arguments.append(contentsOf: nodes)
    return copy
  }
}

extension MDocMacroProtocol {
  public func _serialized(context: MDocSerializationContext) -> String {
    var result = ""

    // Prepend a dot if we aren't already in a macroLine context
    if !context.macroLine {
      result += "."
    }
    result += Self.kind

    if !arguments.isEmpty {
      var context = context
      context.macroLine = true

      result += " "
      result += arguments
        .map { $0._serialized(context: context) }
        .joined(separator: " ")
    }
    return result
  }
}

/// `MDocMacro` is a namespace for types conforming to ``MDocMacroProtocol``.
public enum MDocMacro {
  /// Comment placed inline in the manual page.
  ///
  /// Comment are not displayed by tools consuming serialized manual pages.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Comment("WIP: History section...")
  ///   ```
  public struct Comment: MDocMacroProtocol {
    public static let kind = #"\""#
    public var arguments: [MDocASTNode]
    /// Creates a new `Comment` macro.
    ///
    /// - Parameters:
    ///   - comment: A string to insert as an inline comment.
    public init(_ comment: String) {
      self.arguments = [comment]
    }
  }

  // MARK: - Document preamble and NAME section macros

  /// Document date displayed in the manual page footer.
  ///
  /// This must be the first macro in any `mdoc` document.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   DocumentDate(day: 9, month: "September", year: 2014)
  ///   ```
  public struct DocumentDate: MDocMacroProtocol {
    public static let kind = "Dd"
    public var arguments: [MDocASTNode]
    /// Creates a new `DocumentDate` macro.
    ///
    /// - Parameters:
    ///   - day: An integer number day of the month the manual was written.
    ///   - month: The full English month name the manual was written.
    ///   - year: The four digit year the manual was written.
    public init(day: Int, month: String, year: Int) {
      arguments = [month, "\(day),", year]
    }
  }

  /// Document title displayed in the manual page header.
  ///
  /// This must be the second macro in any `mdoc` document.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   DocumentTitle(title: "swift", section: 1)
  ///   DocumentTitle(title: "swift", section: 1, arch: "arm64e")
  ///   ```
  public struct DocumentTitle: MDocMacroProtocol {
    public static let kind = "Dt"
    public var arguments: [MDocASTNode]
    /// Creates a new `DocumentTitle` macro.
    ///
    /// - Parameters:
    ///   - title: The document's title or name. By convention the title should
    ///   be all caps.
    ///   - section: The manual section. The section should match the manual
    ///   page's file extension. Must be one of the following values:
    ///     1. General Commands
    ///     2. System Calls
    ///     3. Library Functions
    ///     4. Device Drivers
    ///     5. File Formats
    ///     6. Games
    ///     7. Miscellaneous Information
    ///     8. System Manager's Manual
    ///     9. Kernel Developer's Manual
    ///   - arch: The machine architecture the manual page applies to, for
    ///     example: `alpha`, `i386`, `x86_64` or `arm64e`.
    public init(title: String, section: Int, arch: String? = nil) {
      precondition((1...9).contains(section))
      self.arguments = [title, section]
      self.arguments.append(optional: arch)
    }
  }

  /// Operating system and version displayed in the manual page footer.
  ///
  /// This must be the third macro in any `mdoc` document.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   OperatingSystem()
  ///   OperatingSystem(name: "macOS")
  ///   OperatingSystem(name: "macOS", version: "10.13")
  ///   ```
  public struct OperatingSystem: MDocMacroProtocol {
    public static let kind = "Os"
    public var arguments: [MDocASTNode]
    /// Creates a new `OperatingSystem` macro.
    ///
    /// - Note: The `version` parameter must not be specified without the `name`
    ///   parameter.
    ///
    /// - Parameters:
    ///   - name: The operating system the manual page contents is valid for.
    ///     Omitting `name` is recommended and will result in the user's
    ///     operating system name being used.
    ///   - version: The version the of the operating system specified by `name`
    ///     the manual page contents is valid for. Omitting `version` is
    ///     recommended.
    public init(name: String? = nil, version: String? = nil) {
      precondition(!(name == nil && version != nil))
      self.arguments = []
      self.arguments.append(optional: name)
      self.arguments.append(optional: version)
    }
  }

  /// The name of the manual page.
  ///
  /// The first use of ``DocumentName`` is typically in the "NAME" section. The
  /// name provided to the created ``DocumentName`` will be remembered and
  /// subsequent uses of the ``DocumentName`` can omit the name argument.
  ///
  /// - Note: Manual pages in sections 1, 6, and 8 may use the name of command
  ///   or feature documented in the manual page as the name.
  ///
  /// In sections 2, 3, and 9 use the ``FunctionName`` macro instead of the
  /// ``DocumentName`` macro to indicate the name of the document.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   SectionHeader(title: "SYNOPSIS")
  ///   DocumentName(name: "swift")
  ///   OptionalCommandLineComponent(arguments: CommandArgument(arguments: ["h"]))
  ///   ```
  public struct DocumentName: MDocMacroProtocol {
    public static let kind = "Nm"
    public var arguments: [MDocASTNode]
    /// Creates a new `DocumentName` macro.
    ///
    /// - Parameters:
    ///   - name: The name of the manual page.
    public init(name: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: name)
    }
  }

  /// Single line description of the manual page.
  ///
  /// This must be the last macro in the "NAME" section `mdoc` document and
  /// should not appear in any other section.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   DocumentDescription(description: "Safe, fast, and expressive general-purpose programming language")
  ///   ```
  public struct DocumentDescription: MDocMacroProtocol {
    public static let kind = "Nd"
    public var arguments: [MDocASTNode]
    /// Creates a new `DocumentDescription` macro.
    ///
    /// - Parameters:
    ///   - description: The description of the manual page.
    public init(description: String) {
      self.arguments = [description]
    }
  }

  // MARK: - Sections and cross references

  /// Start a new manual section.
  ///
  /// See [Manual Structure](http://mandoc.bsd.lv/man/mdoc.7.html#MANUAL_STRUCTURE)
  /// for a list of standard sections. Custom sections should be avoided though
  /// can be used.
  ///
  /// - Note: Section names should be unique so they can be referenced using a
  ///   ``SectionReference``.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   SectionHeader(title: "NAME")
  ///   ```
  public struct SectionHeader: MDocMacroProtocol {
    public static let kind = "Sh"
    public var arguments: [MDocASTNode]
    /// Creates a new `SectionHeader` macro.
    ///
    /// - Parameters:
    ///   - title: The title of the section.
    public init(title: String) {
      self.arguments = [title]
    }
  }

  /// Start a new manual subsection.
  ///
  /// There is no standard naming convention of subsections.
  ///
  /// - Note: Subsection names should be unique so they can be referenced using
  ///   a ``SectionReference``.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   SubsectionHeader(title: "DETAILS")
  ///   ```
  public struct SubsectionHeader: MDocMacroProtocol {
    public static let kind = "Ss"
    public var arguments: [MDocASTNode]
    /// Creates a new `SubsectionHeader` macro.
    ///
    /// - Parameters:
    ///   - title: The title of the subsection.
    public init(title: String) {
      self.arguments = [title]
    }
  }

  /// Reference a section or subsection in the same manual page.
  ///
  /// The section or subsection title must exactly match the title passed to
  /// ``SectionReference``.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   SectionReference(title: "NAME")
  ///   ```
  public struct SectionReference: MDocMacroProtocol {
    public static let kind = "Sx"
    public var arguments: [MDocASTNode]
    /// Creates a new `SectionReference` macro.
    ///
    /// - Parameters:
    ///   - title: The title of the section or subsection to reference.
    public init(title: String) {
      self.arguments = [title]
    }
  }

  /// Reference another manual page.  
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   CrossManualReference(title: "swift", section: 1)
  ///   ```
  public struct CrossManualReference: MDocMacroProtocol {
    public static let kind = "Xr"
    public var arguments: [MDocASTNode]
    /// Creates a new `CrossManualReference` macro.
    ///
    /// - Parameters:
    ///   - title: The title of the section or subsection to reference.
    public init(title: String, section: Int) {
      precondition((1...9).contains(section))
      self.arguments = [title, section]
    }
  }

  /// Whitespace break between paragaphs.
  ///
  /// Breaks should not be inserted immeediately before or after
  /// ``SectionHeader``, ``SubsectionHeader``, and ``BeginList`` macros.
  public struct ParagraphBreak: MDocMacroProtocol {
    public static let kind = "Pp"
    public var arguments: [MDocASTNode]
    /// Creates a new `ParagraphBreak` macro.
    public init() {
      self.arguments = []
    }
  }

  // MARK: - Displays and lists

  // Display block: -type [-offset width] [-compact].
  // TODO: "Ed"

  // Indented display (one line).
  // TODO: "D1"

  // Indented literal display (one line).
  // TODO: "Dl"

  // In-line literal display: ‘text’.
  // TODO: "Ql"

  // FIXME: Documentation
  /// Open a list scope.
  ///
  /// Closed by an ``EndList`` macro.
  ///
  /// Lists are made of ``ListItem``s which are displayed in a variety of styles
  /// depending on the ``ListStyle`` used to create the list scope.
  /// List scopes can be nested in other list scopes, however nesting `.column`
  /// and `ListStyle.enum` lists is not recommended as they may display inconsistently
  /// between tools.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginList(style: .tag, width: 6)
  ///   ListItem(title: "Hello, Swift!")
  ///   "Welcome to the Swift programming language."
  ///   ListItem(title: "Goodbye!")
  ///   EndList()
  ///   ```
  public struct BeginList: MDocMacroProtocol {
    /// Enumeration of styles supported by the ``BeginList`` macro.
    public enum ListStyle: String {
      /// A bulleted list.
      /// 
      /// Item titles should not be provided, instead item bodies are displayed
      /// indented from a preceding bullet point using the specified width.
      case bullet
      // TODO: case column
      // /// A columnated list.
      // case column
      /// A dashed list.
      ///
      /// Identical to `.bullet` except dashes precede each item.
      case dash
      /// An unindented list without newlines following important item titles
      /// without macro parsing.
      ///
      /// Identical to `.inset` except item titles are displayed with importance
      /// and are not parsed for macros. `.diag` is typically used in the
      /// "DIAGNOSTICS" section with errors as the item titles.
      case diag
      /// An enumerated list.
      ///
      /// Identical to `.bullet` except increasing numbers starting at 1 precede
      /// each item.
      case `enum`
      /// An indented list without joined item titles and bodies.
      ///
      /// Identical to `.tag` except item bodies always on the line after the
      /// item title.
      case hang
      /// Alias for `.dash`.
      case hyphen
      /// An unindented list without newlines following item titles.
      ///
      /// Identical to `.ohang` except item titles are not followed by newlines.
      case inset
      /// An unindented list without item titles.
      ///
      /// Identical to `.ohang` except item titles should not be provided and
      /// are not displayed.
      case item
      /// An unindented list.
      ///
      /// Item titles are displayed on a single line, with unindented item
      /// bodies on the succeeding lines.
      case ohang
      /// An indented list.
      ///
      /// Item titles are displayed on a single line with item bodies indented
      /// using the specified width on succeeding lines. If the item title is
      /// shorter than the indentation width, item bodies are displayed on the
      /// same as the title.
      case tag
    }
    public static let kind = "Bl"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginList` macro.
    ///
    /// - Parameters:
    ///   - style: Display style.
    ///   - width: Number of characters to indent item bodies from titles.
    ///   - offset: Number of characters to indent both the item titles and bodies.
    ///   - compact: Disable vertical spacing between list items.
    public init(style: ListStyle, width: Int? = nil, offset: Int? = nil, compact: Bool = false) {
      self.arguments = ["-\(style)"]
      switch style {
      case .bullet, .dash, .`enum`, .hang, .hyphen, .tag:
        if let width = width {
          self.arguments.append(contentsOf: ["-width", "\(width)n"])
        }
      case /*.column, */.diag, .inset, .item, .ohang:
        assert(width == nil, "`width` should be nil for style: \(style)")
      }
      if let offset = offset {
        self.arguments.append(contentsOf: ["-offset", "\(offset)n"])
      }
      if compact {
        self.arguments.append(contentsOf: ["-compact"])
      }
    }
  }

  /// A list item.
  ///
  /// ``ListItem`` begins a list item scope continuing until another
  /// ``ListItem`` is encountered or the enclosing list scope is closed by
  /// ``EndList``. ``ListItem``s may include a title if the the enclosing list
  /// scope was constructed with one of the following styles:
  ///   - `.bullet`
  ///   - `.dash`
  ///   - `.enum`
  ///   - `.hang`
  ///   - `.hyphen`
  ///   - `.tag`
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginList(style: .tag, width: 6)
  ///   ListItem(title: "Hello, Swift!")
  ///   "Welcome to the Swift programming language."
  ///   ListItem(title: "Goodbye!")
  ///   EndList()
  ///   ```
  public struct ListItem: MDocMacroProtocol {
    public static let kind = "It"
    public var arguments: [MDocASTNode]
    /// Creates a new `ListItem` macro.
    ///
    /// - Parameters:
    ///   - title: List item title, only valid depending on the ``ListStyle``.
    public init(title: MDocASTNode? = nil) {
      arguments = []
      arguments.append(optional: title)
    }
  }

  // Table cell separator in Bl -column lists.
  // TODO: "Ta"

  /// Close a list scope opened by a ``BeginList`` macro.
  public struct EndList: MDocMacroProtocol {
    public static let kind = "El"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndList` macro.
    public init() {
      self.arguments = []
    }
  }

  // Bibliographic block (references).
  // TODO: "Re"

  // MARK: Spacing control

  /// Text without a trailing space.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   WithoutTrailingSpace(text: "swift")
  ///   ```
  public struct WithoutTrailingSpace: MDocMacroProtocol {
    public static let kind = "Pf"
    public var arguments: [MDocASTNode]
    /// Creates a new `WithoutTrailingSpace` macro.
    ///
    /// - Parameters:
    ///   - text: The text to display without a trailing space.
    public init(text: String) {
      self.arguments = [text]
    }
  }

  /// Text without a leading space.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   WithoutLeadingSpace(text: "swift")
  ///   ```
  public struct WithoutLeadingSpace: MDocMacroProtocol {
    public static let kind = "Ns"
    public var arguments: [MDocASTNode]
    /// Creates a new `WithoutLeadingSpace` macro.
    ///
    /// - Parameters:
    ///   - text: The text to display without a trailing space.
    public init(text: String) {
      self.arguments = [text]
    }
  }

  /// An apostrophe without leading and trailing spaces.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Apostrophe()
  ///   ```
  public struct Apostrophe: MDocMacroProtocol {
    public static let kind = "Ap"
    public var arguments: [MDocASTNode]
    /// Creates a new `Apostrophe` macro.
    public init() {
      self.arguments = []
    }
  }

  // TODO: HorizontalSpacing
  // /// Switch horizontal spacing mode: [on | off].
  // public struct HorizontalSpacing: MDocMacroProtocol {
  //   public static let kind = "Sm"
  //   public var arguments: [MDocASTNode]
  //   public init() {
  //     self.arguments = []
  //   }
  // }

  // Keep block: -words.
  // TODO: "Ek"

  // MARK: - Semantic markup for command-line utilities

  /// Command-line flags and options.
  ///
  /// Displays a hyphen (`-`) before each argument. ``CommandOption`` is
  /// typically used in the "SYNOPSIS" and "DESCRIPTION" sections when listing
  /// and describing options in a manual page.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   CommandOption(arguments: ["-version"])
  ///     .withUnsafeChildren(CommandArgument(arguments: "version"))
  ///   ```
  public struct CommandOption: MDocMacroProtocol {
    public static let kind = "Fl"
    public var arguments: [MDocASTNode]
    /// Creates a new `CommandOption` macro.
    ///
    /// - Parameters:
    ///   - arguments: Command-line flags and options.
    public init(options: [MDocASTNode]) {
      self.arguments = options
    }
  }

  /// Command-line modifiers.
  ///
  /// ``CommandModifier`` is typically used to denote strings exactly passed as
  /// arguments, if and only if, ``CommandOption`` is not appropriate.
  /// ``CommandModifier`` can also be used to specify configuration options and
  /// keys.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   CommandModifier(modifiers: ["Configuration File"])
  ///     .withUnsafeChildren(nodes: [FilePath(path: "$HOME/.swiftpm")])
  ///   ```
  public struct CommandModifier: MDocMacroProtocol {
    public static let kind = "Cm"
    public var arguments: [MDocASTNode]
    /// Creates a new `CommandModifier` macro.
    ///
    /// - Parameters:
    ///   - modifiers: Command-line modifiers.
    public init(modifiers: [MDocASTNode]) {
      self.arguments = modifiers
    }
  }

  /// Command-line placeholders.
  ///
  /// ``CommandArgument`` displays emphasized placeholders for command-line
  /// flags, options and arguments. Flag and option names must use
  /// ``CommandOption`` or `CommandModifier` macros. If no arguments are
  /// provided to ``CommandArgument``, the string `"file ..."` is used.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   CommandArgument()
  ///   CommandArgument(arguments: [arg1, ",", arg2, "."])
  ///   CommandOption(arguments: ["-version"])
  ///     .withUnsafeChildren(CommandArgument(arguments: "version"))
  ///   ```
  public struct CommandArgument: MDocMacroProtocol {
    public static let kind = "Ar"
    public var arguments: [MDocASTNode]
    /// Creates a new `CommandArgument` macro.
    ///
    /// - Parameters:
    ///   - arguments: Command-line argument placeholders.
    public init(arguments: [MDocASTNode]) {
      self.arguments = arguments
    }
  }

  /// Single-line optional command-line components.
  ///
  /// Displays the arguments in `[squareBrackets]`.
  /// ``OptionalCommandLineComponent`` is typically used in the "SYNOPSIS"
  /// section.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   SectionHeader(title: "SYNOPSIS")
  ///   DocumentName(name: "swift")
  ///   OptionalCommandLineComponent(arguments: CommandArgument(arguments: ["h"]))
  ///   ```
  public struct OptionalCommandLineComponent: MDocMacroProtocol {
    public static let kind = "Op"
    public var arguments: [MDocASTNode]
    /// Creates a new `OptionalCommandLineComponent` macro.
    ///
    /// - Parameters:
    ///   - arguments: Command-line components to enclose.
    public init(arguments: [MDocASTNode]) {
      self.arguments = arguments
    }
  }

  /// Begin a multi-line optional command-line comment scope.
  ///
  /// Displays the scope contents in `[squareBrackets]`.
  /// ``BeginOptionalCommandLineComponent`` is typically used in the "SYNOPSIS"
  /// section.
  ///
  /// Closed by an ``EndOptionalCommandLineComponent`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginOptionalCommandLineComponent()
  ///   "Hello, Swift!"
  ///   EndOptionalCommandLineComponent()
  ///   ```
  public struct BeginOptionalCommandLineComponent: MDocMacroProtocol {
    public static let kind = "Oo"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginOptionalCommandLineComponent` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a ```BeginOptionalCommandLineComponent``` block.
  public struct EndOptionalCommandLineComponent: MDocMacroProtocol {
    public static let kind = "Oc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndOptionalCommandLineComponent` macro.
    public init() {
      self.arguments = []
    }
  }

  /// An interactive command.
  /// 
  /// ``InteractiveCommand`` is similar to ``CommandModifier`` but should be used
  /// to describe commands instead of arguments. For example,
  /// ``InteractiveCommand`` can be used to describe the commands to editors
  /// like `emacs` and `vim` or shells like `bash` or `fish`.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   InteractiveCommand(name: "print")
  ///   InteractiveCommand(name: "save")
  ///   InteractiveCommand(name: "quit")
  ///   ```
  public struct InteractiveCommand: MDocMacroProtocol {
    public static let kind = "Ic"
    public var arguments: [MDocASTNode]
    /// Creates a new `InteractiveCommand` macro.
    ///
    /// - Parameters:
    ///   - name: Name of the interactive command.
    public init(name: String) {
      self.arguments = [name]
    }
  }

  /// An environment variable.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   EnvironmentVariable(variable: "DISPLAY")
  ///   EnvironmentVariable(variable: "PATH")
  ///   ```
  public struct EnvironmentVariable: MDocMacroProtocol {
    public static let kind = "Ev"
    public var arguments: [MDocASTNode]
    /// Creates a new `EnvironmentVariable` macro.
    ///
    /// - Parameters:
    ///   - name: Name of the environment variable.
    public init(name: String) {
      self.arguments = [name]
    }
  }

  /// A file path.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   FilePath()
  ///   FilePath(path: "/usr/bin/swift")
  ///   FilePath(path: "/usr/share/man/man1/swift.1")
  ///   ```
  public struct FilePath: MDocMacroProtocol {
    public static let kind = "Pa"
    public var arguments: [MDocASTNode]
    /// Creates a new `FilePath` macro.
    ///
    /// - Parameters:
    ///   - path: An optional absolute or relative path or a file or directory.
    ///     Tilde (`~`) will be used, if no path is used.
    public init(path: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: path)
    }
  }

  // MARK: - Semantic markup for function libraries

  // Function library (one argument).
  // TODO: "Lb"

  // Include file (one argument).
  // TODO: "In"

  // Other preprocessor directive (>0 arguments).
  // TODO: "Fd"

  // Function type (>0 arguments).
  // TODO: "Ft"

  // Function block: funcname.
  // TODO: "Fc"

  // Function name: funcname [argument ...].
  // TODO: "Fn"

  // Function argument (>0 arguments).
  // TODO: "Fa"

  // Variable type (>0 arguments).
  // TODO: "Vt"

  // Variable name (>0 arguments).
  // TODO: "Va"

  /// Defined variable or preprocessor constant (>0 arguments).
  // TODO: "Dv"

  /// Error constant (>0 arguments).
  // TODO: "Er"

  /// Environmental variable (>0 arguments).
  // TODO: "Ev"

  // MARK: - Various semantic markup

  /// An author's name.
  /// 
  /// ``Author`` can be used to designate any author. Specifying an author of
  /// the manual page itself should only occur in the "AUTHORS" section.
  ///
  /// ``Author`` also controls the display mode of authors. In the split mode,
  /// a new-line will be inserted before each author, otherwise authors will
  /// appear inline with other macros and text.  Outside of the "AUTHORS"
  /// section, the default display mode is unsplit. The display mode is reset at
  /// the start of the "AUTHORS" section. In the "AUTHORS" section, the first
  /// use of ``Author`` will use the unsplit mode and subsequent uses with use
  /// the split mode. This behavior can be overridden by inserting an author
  /// display mode macro before the normal author macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Author(split: false)
  ///   Author(name: "Rauhul Varma")
  ///   ```
  public struct Author: MDocMacroProtocol {
    public static let kind = "An"
    public var arguments: [MDocASTNode]
    /// Creates a new `Author` macro.
    ///
    /// - Parameters:
    ///   - name: The author name to display.
    public init(name: String) {
      self.arguments = [name]
    }
    /// Creates a new `Author` macro.
    ///
    /// - Parameters:
    ///   - split: The split display mode to use for subsequent uses of
    ///     ``Author``.
    public init(split: Bool) {
      self.arguments = [split ? "-split" : "-nosplit"]
    }
  }

  /// A website hyperlink.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Hyperlink(url: "http://swift.org")
  ///   Hyperlink(url: "http://swift.org", displayText: "Programming in Swift")
  ///   ```
  public struct Hyperlink: MDocMacroProtocol {
    public static let kind = "Lk"
    public var arguments: [MDocASTNode]
    /// Creates a new `Hyperlink` macro.
    ///
    /// - Parameters:
    ///   - url: The website address to link.
    ///   - displayText: Optional title text accompanying the url.
    public init(url: String, displayText: String? = nil) {
      self.arguments = [url]
      self.arguments.append(optional: displayText)
    }
  }

  /// An email hyperlink.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   MailTo(email: "swift+evolution-discuss@forums.swift.org")
  ///   ```
  public struct MailTo: MDocMacroProtocol {
    public static let kind = "Mt"
    public var arguments: [MDocASTNode]
    /// Creates a new `MailTo` macro.
    ///
    /// - Parameters:
    ///   - email: The email address to link.
    public init(email: String) {
      self.arguments = [email]
    }
  }

// TODO: KernelConfiguration
//  /// Kernel configuration declaration (>0 arguments).
//  public struct KernelConfiguration: MDocMacroProtocol {
//    public static let kind = "Cd"
//    public var arguments: [MDocASTNode]
//    public init() {
//      self.arguments = []
//    }
//  }

// TODO: MemoryAddress
//  /// Memory address (>0 arguments).
//  public struct MemoryAddress: MDocMacroProtocol {
//    public static let kind = "Ad"
//    public var arguments: [MDocASTNode]
//    public init() {
//      self.arguments = []
//    }
//  }

// TODO: MathematicalSymbol
//  /// Mathematical symbol (>0 arguments).
//  public struct MathematicalSymbol: MDocMacroProtocol {
//    public static let kind = "Ms"
//    public var arguments: [MDocASTNode]
//    public init() {
//      self.arguments = []
//    }
//  }

  // MARK: - Physical markup

  /// Emphasize single-line text.
  ///
  /// ``Emphasis`` should only be used when no other semantic macros are
  /// appropriate. ``Emphasis`` is used to express "emphasis"; for example:
  /// ``Emphasis`` can be used to highlight technical terms and placeholders,
  /// except when they appear in syntactic elements. ``Emphasis`` should not be
  /// conflated with "importance" which should be expressed using ``Boldface``.
  ///
  /// - Note: Emphasizes text is usually italicized. If the output program does
  ///   not support italicizing text, it is underlined instead.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Emphasis(arguments: ["Hello", ", "Swift!"])
  ///   ```
  public struct Emphasis: MDocMacroProtocol {
    public static let kind = "Em"
    public var arguments: [MDocASTNode]
    /// Creates a new `Emphasis` macro.
    ///
    /// - Parameters:
    ///   - arguments: Text to emphasize.
    public init(arguments: [MDocASTNode]) {
      self.arguments = arguments
    }
  }

  /// Embolden single-line text.
  ///
  /// ``Boldface`` should only be used when no other semantic macros are
  /// appropriate. ``Boldface`` is used to express "importance"; for example:
  /// ``Boldface`` can be used to highlight required arguments and exact text.
  /// ``Boldface`` should not be conflated with "emphasis" which
  /// should be expressed using ``Emphasis``.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Boldface(arguments: ["Hello,", " Swift!"])
  ///   ```
  public struct Boldface: MDocMacroProtocol {
    public static let kind = "Sy"
    public var arguments: [MDocASTNode]
    /// Creates a new `Boldface` macro.
    ///
    /// - Parameters:
    ///   - arguments: Text to embolden.
    public init(arguments: [MDocASTNode]) {
      self.arguments = arguments
    }
  }

  /// Reset the font style, set by a single-line text macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   Boldface(arguments: ["Hello,"])
  ///     .withUnsafeChildren(nodes: [NormalText(), " Swift!"])
  ///   ```
  public struct NormalText: MDocMacroProtocol {
    public static let kind = "No"
    public var arguments: [MDocASTNode]
    /// Creates a new `NormalText` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a font scope with a font style.
  ///
  /// Closed by a ``EndFont`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginFont(style: .boldface)
  ///   "Hello, Swift!"
  ///   EndFont()
  ///   ```
  public struct BeginFont: MDocMacroProtocol {
    /// Enumeration of font styles supported by `mdoc`.
    public enum FontStyle {
      /// Italic font style.
      case emphasis
      /// Typewriter font style.
      ///
      /// `literal` should not be used because it is visually identical to
      /// normal text.
      case literal
      /// Bold font style.
      case boldface
    }

    public static let kind = "Bf"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginFont` macro.
    ///
    /// - Parameters:
    ///   - style: The style of font scope the macro opens.
    public init(style: FontStyle) {
      switch style {
      case .emphasis:
        self.arguments = ["-emphasis"]
      case .literal:
        self.arguments = ["-literal"]
      case .boldface:
        self.arguments = ["-symbolic"]
      }
    }
  }

  /// Close a font scope opened by a ``BeginFont`` macro.
  public struct EndFont: MDocMacroProtocol {
    public static let kind = "Ef"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndFont` macro.
    public init() {
      self.arguments = []
    }
  }

  // MARK: - Physical enclosures

  /// Open a scope enclosed by `“typographic”` double-quotes.
  ///
  /// Closed by a ``EndTypographicDoubleQuotes`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginTypographicDoubleQuotes()
  ///   "Hello, Swift!"
  ///   EndTypographicDoubleQuotes()
  ///   ```
  public struct BeginTypographicDoubleQuotes: MDocMacroProtocol {
    public static let kind = "Do"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginTypographicDoubleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginTypographicDoubleQuotes`` macro.
  public struct EndTypographicDoubleQuotes: MDocMacroProtocol {
    public static let kind = "Dc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndTypographicDoubleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `"typewriter"` double-quotes.
  ///
  /// Closed by a ``EndTypewriterDoubleQuotes`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginTypewriterDoubleQuotes()
  ///   "Hello, Swift!"
  ///   EndTypewriterDoubleQuotes()
  ///   ```
  public struct BeginTypewriterDoubleQuotes: MDocMacroProtocol {
    public static let kind = "Qo"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginTypewriterDoubleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginTypewriterDoubleQuotes`` macro.
  public struct EndTypewriterDoubleQuotes: MDocMacroProtocol {
    public static let kind = "Qc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndTypewriterDoubleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `'single'` quotes.
  ///
  /// Closed by a ``EndSingleQuotes`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginSingleQuotes()
  ///   "Hello, Swift!"
  ///   EndSingleQuotes()
  ///   ```
  public struct BeginSingleQuotes: MDocMacroProtocol {
    public static let kind = "So"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginSingleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginSingleQuotes`` macro.
  public struct EndSingleQuotes: MDocMacroProtocol {
    public static let kind = "Sc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndSingleQuotes` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `(parentheses)`.
  ///
  /// Closed by a ``EndParentheses`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginParentheses()
  ///   "Hello, Swift!"
  ///   EndParentheses()
  ///   ```
  public struct BeginParentheses: MDocMacroProtocol {
    public static let kind = "Po"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginParentheses` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginParentheses`` macro.
  public struct EndParentheses: MDocMacroProtocol {
    public static let kind = "Pc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndParentheses` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `[squareBrackets]`.
  ///
  /// Closed by a ``EndSquareBrackets`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginSquareBrackets()
  ///   "Hello, Swift!"
  ///   EndSquareBrackets()
  ///   ```
  public struct BeginSquareBrackets: MDocMacroProtocol {
    public static let kind = "Bo"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginSquareBrackets` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginSquareBrackets`` macro.
  public struct EndSquareBrackets: MDocMacroProtocol {
    public static let kind = "Bc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndSquareBrackets` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `{curlyBraces}`.
  ///
  /// Closed by a ``EndCurlyBraces`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginCurlyBraces()
  ///   "Hello, Swift!"
  ///   EndCurlyBraces()
  ///   ```
  public struct BeginCurlyBraces: MDocMacroProtocol {
    public static let kind = "Bro"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginCurlyBraces` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginCurlyBraces`` macro.
  public struct EndCurlyBraces: MDocMacroProtocol {
    public static let kind = "Brc"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndCurlyBraces` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Open a scope enclosed by `<angleBrackets>`.
  ///
  /// Closed by a ``EndAngleBrackets`` macro.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BeginAngleBrackets()
  ///   "Hello, Swift!"
  ///   EndAngleBrackets()
  ///   ```
  public struct BeginAngleBrackets: MDocMacroProtocol {
    public static let kind = "Ao"
    public var arguments: [MDocASTNode]
    /// Creates a new `BeginAngleBrackets` macro.
    public init() {
      self.arguments = []
    }
  }

  /// Close a scope opened by a ``BeginAngleBrackets`` macro.
  public struct EndAngleBrackets: MDocMacroProtocol {
    public static let kind = "Ac"
    public var arguments: [MDocASTNode]
    /// Creates a new `EndAngleBrackets` macro.
    public init() {
      self.arguments = []
    }
  }

  // TODO: GenericEnclosure
  // /// Enclose another element generically.
  // case genericEnclosure(MDocLowLevelASTNode)

  // MARK: - Text production

  /// Display a standard line about the exit code of specified utilities.
  ///
  /// This macro indicates the specified utilities exit 0 on success and other
  /// values on failure. ``ExitStandard``` should be only included in the
  /// "EXIT STATUS" section.
  ///
  /// ``ExitStandard`` should only be used in sections 1, 6, and 8.
  public struct ExitStandard: MDocMacroProtocol {
    public static let kind = "Ex"
    public var arguments: [MDocASTNode]
    /// Creates a new `ExitStandard` macro.
    ///
    /// - Parameters:
    ///   - utilities: A list of utilities the exit standard applies to. If no
    ///     utilities are specified the document's name set by ``DocumentName``
    ///     is used.
    public init(utilities: [String] = []) {
      self.arguments = ["-std"] + utilities
    }
  }

// TODO: ReturnStandard
//  /// Insert a standard sentence regarding a function call's return value of 0 on success and -1 on error, with the errno libc global variable set on error.
//  ///
//  /// If function is not specified, the document's name set by ``DocumentName`` is used. Multiple function arguments are treated as separate functions.
//  public struct ReturnStandard: MDocMacroProtocol {
//    public static let kind = "Rv"
//    public var arguments: [MDocASTNode]
//    public init() {
//      self.arguments = []
//    }
//  }

// TODO: StandardsReference
//  /// Reference to a standards document (one argument).
//  public struct StandardsReference: MDocMacroProtocol {
//    public static let kind = "St"
//    public var arguments: [MDocASTNode]
//    public init() {
//      self.arguments = []
//    }
//  }

  /// Display a formatted version of AT&T UNIX.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   AttUnix()
  ///   AttUnix(version: "V.1")
  ///   ```
  public struct AttUnix: MDocMacroProtocol {
    public static let kind = "At"
    public var arguments: [MDocASTNode]
    /// Creates a new `AttUnix` macro.
    ///
    /// - Parameters:
    ///   - version: The version of Att Unix to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    ///     `version` should be one of the following values;
    ///     - `v[1-7] | 32v` - A version of AT&T UNIX.
    ///     - `III` - AT&T System III UNIX.
    ///     - `V | V.[1-4]` - A version of AT&T System V UNIX.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted variant and version of BSD.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BSD()
  ///   BSD(name: "Ghost")
  ///   BSD(name: "Ghost", version: "21.04.27")
  ///   ```
  public struct BSD: MDocMacroProtocol {
    public static let kind = "Bx"
    public var arguments: [MDocASTNode]
    /// Creates a new `BSD` macro.
    ///
    /// - Note: The `version` parameter must not be specified without
    ///   the `name` parameter.
    ///
    /// - Parameters:
    ///   - name: The name of the BSD variant to stylize.
    ///   - version: The version `name` to stylize. Omitting `version`
    ///     will result in an unversioned OS being displayed.
    public init(name: String? = nil, version: String? = nil) {
      precondition(!(name == nil && version != nil))
      self.arguments = []
      self.arguments.append(optional: name)
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted version of BSD/OS.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   BSDOS()
  ///   BSDOS(version: "5.1")
  ///   ```
  public struct BSDOS: MDocMacroProtocol {
    public static let kind = "Bsx"
    public var arguments: [MDocASTNode]
    /// Creates a new `BSDOS` macro.
    ///
    /// - Parameters:
    ///   - version: The version of BSD/OS to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted version of NetBSD.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   NetBSD()
  ///   NetBSD(version: "9.2")
  ///   ```
  public struct NetBSD: MDocMacroProtocol {
    public static let kind = "Nx"
    public var arguments: [MDocASTNode]
    /// Creates a new `NetBSD` macro.
    ///
    /// - Parameters:
    ///   - version: The version of NetBSD to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted version of FreeBSD.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   FreeBSD()
  ///   FreeBSD(version: "13.0")
  ///   ```
  public struct FreeBSD: MDocMacroProtocol {
    public static let kind = "Fx"
    public var arguments: [MDocASTNode]
    /// Creates a new `FreeBSD` macro.
    ///
    /// - Parameters:
    ///   - version: The version of FreeBSD to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted version of OpenBSD.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   OpenBSD()
  ///   OpenBSD(version: "6.9")
  ///   ```
  public struct OpenBSD: MDocMacroProtocol {
    public static let kind = "Ox"
    public var arguments: [MDocASTNode]
    /// Creates a new `OpenBSD` macro.
    ///
    /// - Parameters:
    ///   - version: The version of OpenBSD to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }

  /// Display a formatted version of DragonFly.
  ///
  /// __Example Usage__:
  ///   ```swift
  ///   DragonFly()
  ///   DragonFly(version: "6.0")
  ///   ```
  public struct DragonFly: MDocMacroProtocol {
    public static let kind = "Dx"
    public var arguments: [MDocASTNode]
    /// Creates a new `DragonFly` macro.
    ///
    /// - Parameters:
    ///   - version: The version of DragonFly to stylize. Omitting
    ///     `version` will result in an unversioned OS being displayed.
    public init(version: String? = nil) {
      self.arguments = []
      self.arguments.append(optional: version)
    }
  }
}
