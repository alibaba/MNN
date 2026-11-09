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

@resultBuilder
struct MDocBuilder {
  static func buildBlock(_ components: MDocComponent...) -> MDocComponent { Container(children: components) }
  static func buildArray(_ components: [MDocComponent]) -> MDocComponent { Container(children: components) }
  static func buildOptional(_ component: MDocComponent?) -> MDocComponent { component ?? Empty() }
  static func buildEither(first component: MDocComponent) -> MDocComponent { component }
  static func buildEither(second component: MDocComponent) -> MDocComponent { component }
  static func buildExpression(_ expression: MDocComponent) -> MDocComponent { expression }
  static func buildExpression(_ expression: MDocASTNode) -> MDocComponent { MDocASTNodeWrapper(node: expression) }
}
