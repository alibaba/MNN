//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import CollectionsBenchmark

class Box {}

struct Large: Hashable {
  let v0: Int
  let v1: Int
  let v2: Int
  let v3: Int
  let v4: Int
  let v5: Int
  let v6: Int
  let v7: Int
  let v8: Int
  let v9: Int

  init(_ value: Int) {
    self.v0 = value
    self.v1 = value
    self.v2 = value
    self.v3 = value
    self.v4 = value
    self.v5 = value
    self.v6 = value
    self.v7 = value
    self.v8 = value
    self.v9 = value
  }

  static func ==(left: Self, right: Self) -> Bool {
    left.v0 == right.v0
  }

  func hash(into hasher: inout Hasher) {
    hasher.combine(v0)
  }
}

extension Benchmark {
  public mutating func registerCustomGenerators() {
    self.registerInputGenerator(for: [String].self) { size in
      (0 ..< size).map { String($0) }.shuffled()
    }

    self.registerInputGenerator(for: ([String], [String]).self) { size in
      let items = (0 ..< size).map { String($0) }
      return (items.shuffled(), items.shuffled())
    }

    self.registerInputGenerator(for: [Large].self) { size in
      (0 ..< size).map { Large($0) }.shuffled()
    }

    self.registerInputGenerator(for: ([Large], [Large]).self) { size in
      let items = (0 ..< size).map { Large($0) }
      return (items.shuffled(), items.shuffled())
    }
  }
}
