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

struct ForEach<C>: MDocComponent where C: Collection {
  var items: C
  var builder: (C.Element, C.Index) -> MDocComponent

  init(_ items: C, @MDocBuilder builder: @escaping (C.Element, C.Index) -> MDocComponent) {
    self.items = items
    self.builder = builder
  }

  var body: MDocComponent {
    guard !items.isEmpty else { return Empty() }
    var currentIndex = items.startIndex
    var components = [MDocComponent]()
    while currentIndex < items.endIndex {
      let item = items[currentIndex]
      components.append(builder(item, currentIndex))
      currentIndex = items.index(after: currentIndex)
    }
    return Container(children: components)
  }
}
