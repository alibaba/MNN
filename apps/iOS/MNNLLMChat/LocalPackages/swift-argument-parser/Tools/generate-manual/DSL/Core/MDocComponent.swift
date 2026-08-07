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

protocol MDocComponent {
  var ast: [MDocASTNode] { get }
  @MDocBuilder
  var body: MDocComponent { get }
}

extension MDocComponent {
  var ast: [MDocASTNode] { body.ast }
}
