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

struct MDocASTNodeWrapper: MDocComponent {
  var ast: [MDocASTNode] { [node] }
  var body: MDocComponent { self }
  var node: MDocASTNode
}
