//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import CollectionsBenchmark

extension Benchmark {
  public mutating func addDictionaryBenchmarks() {
    self.add(
      title: "Dictionary<Int, Int> init(uniqueKeysWithValues:)",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      return { timer in
        blackHole(Dictionary(uniqueKeysWithValues: keysAndValues))
      }
    }

    self.add(
      title: "Dictionary<Int, Int> sequential iteration",
      input: [Int].self
    ) { input in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for item in d {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int>.Keys sequential iteration",
      input: [Int].self
    ) { input in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for item in d.keys {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int>.Values sequential iteration",
      input: [Int].self
    ) { input in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for item in d.values {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> sequential iteration, indices",
      input: [Int].self
    ) { input in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in d.indices {
          blackHole(d[i])
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> indexing subscript",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      let indices = lookups.map { d.index(forKey: $0)! }
      return { timer in
        for i in indices {
          blackHole(d[i])
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, successful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in lookups {
          precondition(d[i] == 2 * i)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, unsuccessful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      let c = input.count
      return { timer in
        for i in lookups {
          precondition(d[i + c] == nil)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, noop setter",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[i + c] = nil
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, set existing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i] = 0
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, _modify",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i]! *= 2
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.addSimple(
      title: "Dictionary<Int, Int> subscript, insert, unique",
      input: [Int].self
    ) { input in
      var d: [Int: Int] = [:]
      for i in input {
        d[i] = 2 * i
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.addSimple(
      title: "Dictionary<Int, Int> subscript, insert, shared",
      input: [Int].self
    ) { input in
      var d: [Int: Int] = [:]
      for i in input {
        let copy = d
        d[i] = 2 * i
        blackHole((copy, d))
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.addSimple(
      title: "Dictionary<Int, Int> subscript, insert, reserving capacity",
      input: [Int].self
    ) { input in
      var d: [Int: Int] = [:]
      d.reserveCapacity(input.count)
      for i in input {
        d[i] = 2 * i
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, remove existing, unique",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i] = nil
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, remove existing, shared",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            let copy = d
            d[i] = nil
            blackHole((copy, d))
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript, remove missing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[i + c] = nil
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> subscript(position:)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      let indices = lookups.map { d.index(forKey: $0)! }
      return { timer in
        for i in indices {
          blackHole(d[i])
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> defaulted subscript, successful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in lookups {
          precondition(d[i, default: -1] != -1)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> defaulted subscript, unsuccessful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        let c = d.count
        for i in lookups {
          precondition(d[i + c, default: -1] == -1)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> defaulted subscript, _modify existing",
      input: [Int].self
    ) { input in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in input {
            d[i, default: -1] *= 2
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> defaulted subscript, _modify missing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[c + i, default: -1] *= 2
          }
        }
        precondition(d.count == 2 * input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> successful index(forKey:)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in lookups {
          precondition(d.index(forKey: i) != nil)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> unsuccessful index(forKey:)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in lookups {
          precondition(d.index(forKey: lookups.count + i) == nil)
        }
      }
    }

    self.add(
      title: "Dictionary<Int, Int> updateValue(_:forKey:), existing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d.updateValue(0, forKey: i)
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> updateValue(_:forKey:), insert",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d.updateValue(0, forKey: input.count + i)
          }
        }
        precondition(d.count == 2 * input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> random removals (existing keys)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            precondition(d.removeValue(forKey: i) != nil)
          }
        }
        precondition(d.count == 0)
        blackHole(d)
      }
    }

    self.add(
      title: "Dictionary<Int, Int> random removals (missing keys)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        let c = input.count
        var d = Dictionary(uniqueKeysWithValues: input.lazy.map { (c + $0, 2 * $0) })
        timer.measure {
          for i in lookups {
            precondition(d.removeValue(forKey: i) == nil)
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }
    
    self.add(
      title: "Dictionary<Int, Int> equality, unique",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = Dictionary(uniqueKeysWithValues: keysAndValues)
      let right = Dictionary(uniqueKeysWithValues: keysAndValues)
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "Dictionary<Int, Int> equality, shared",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = Dictionary(uniqueKeysWithValues: keysAndValues)
      let right = left
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
  }
}
