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

struct Section: MDocComponent {
  var title: String
  var content: MDocComponent

  init(title: String, @MDocBuilder content: () -> MDocComponent) {
    self.title = title
    self.content = content()
  }

  var body: MDocComponent {
    if !content.ast.isEmpty {
      MDocMacro.SectionHeader(title: title.uppercased())
      content
    }
  }
}
