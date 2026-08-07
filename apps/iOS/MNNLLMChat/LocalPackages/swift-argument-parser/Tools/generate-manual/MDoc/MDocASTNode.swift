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

/// `MDocASTNode` represents a single abstract syntax tree node in an `mdoc`
/// document. `mdoc` is a semantic markup language for formatting manual pages.
///
/// See: https://mandoc.bsd.lv/man/mdoc.7.html for more information.
public protocol MDocASTNode {
  /// `_serialized` is an implementation detail and should not be used directly.
  /// Please use `serialized` instead.
  func _serialized(context: MDocSerializationContext) -> String
}

extension MDocASTNode {
  /// `serialized` Serializes an MDocASTNode and children into its string
  /// representation for use with other tools.
  public func serialized() -> String {
    _serialized(context: MDocSerializationContext())
  }
}

extension Int: MDocASTNode {
  public func _serialized(context: MDocSerializationContext) -> String {
    "\(self)"
  }
}

extension String: MDocASTNode {
  public func _serialized(context: MDocSerializationContext) -> String {
    context.macroLine
      ? self.escapedMacroArgument()
      : self.escapedTextLine()
  }
}
