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

// Escaping rules
// https://mandoc.bsd.lv/mdoc/intro/escaping.html
extension String {
  func escapedMacroArgument() -> String {
    var escaped = ""
    var containsBlankCharacter = false

    // TODO: maybe drop `where character.isASCII` clause
    for character in self where character.isASCII {
      switch character {
      case " ":
        escaped.append(character)
        containsBlankCharacter = true

      // backslashes:
      // To output a backslash, use the escape sequence `\e`. Never use the escape sequence `\\` in any context.
      case #"\"#:
        escaped += #"\e"#

      // double quotes in macro arguments:
      // If a macro argument needs to contain a double quote character, write it as “\(dq”. No escaping is needed on text input lines.
      case "\"":
        escaped += #"\(dq"#

      default:
        // Custom addition:
        // newlines in macro arguments:
        // If a macro argument contains a newline character, replace it with a blank character.
        if character.isNewline {
          escaped.append(" ")
          containsBlankCharacter = true
        } else {
          escaped.append(character)
        }
      }
    }

    // FIXME:
    // macro names as macro arguments:
    // If the name of another mdoc(7) macro occurs as an argument on an mdoc(7) macro line, the former macro is called, and any remaining arguments are passed to it. To prevent this call and instead render the name of the former macro literally, prepend the name with a zero-width space (‘\&’). See the MACRO SYNTAX section of the mdoc(7) manual for details.

    // blanks in macro arguments
    // If a macro argument needs to contain a blank character, enclose the whole argument in double quotes. For example, this often occurs with Fa macros. See the MACRO SYNTAX in the roff(7) manual for details.
    if escaped.isEmpty || containsBlankCharacter {
      return "\"\(escaped)\""
    }

    return escaped
  }

  func escapedTextLine() -> String {
    var escaped = ""
    var atBeginning = true

    // TODO: maybe drop `where character.isASCII` clause
    for character in self where character.isASCII {
      switch (character, atBeginning) {

      // backslashes:
      // To output a backslash, use the escape sequence `\e`. Never use the escape sequence `\\` in any context.
      case (#"\"#, _):
        escaped += #"\e"#
        atBeginning = false

      // dots and apostrophes at the beginning of text lines:
      // If a text input line needs to begin with a dot (`.`) or apostrophe (`'`), prepend a zero-width space (`\&`) to prevent the line from being mistaken for a macro line. Never use the escape sequence `\.` in any context.
      case (".", true), ("'", true):
        escaped += #"\&"#
        escaped.append(character)
        atBeginning = false

      // blank characters at the beginning of text lines:
      // If a text input line needs to begin with a blank character (` `) and no line break is desired before that line, prepend a zero-width space (`\&`).
      case (" ", true):
        escaped += #"\&"#
        escaped.append(character)

      default:
        escaped.append(character)
        atBeginning = false
      }
    }

    return escaped
  }
}
