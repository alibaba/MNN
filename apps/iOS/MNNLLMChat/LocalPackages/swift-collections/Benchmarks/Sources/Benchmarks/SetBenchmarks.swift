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
  public mutating func addSetBenchmarks() {
    self.addSimple(
      title: "Int.hashValue on each value",
      input: Int.self
    ) { size in
      for i in 0 ..< size {
        blackHole(i.hashValue)
      }
    }

    self.addSimple(
      title: "Hasher.combine on a single buffer of integers",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        var hasher = Hasher()
        hasher.combine(bytes: UnsafeRawBufferPointer(buffer))
        blackHole(hasher.finalize())
      }
    }

    self.addSimple(
      title: "Set<Int> init from range",
      input: Int.self
    ) { size in
      blackHole(Set(0 ..< size))
    }

    self.addSimple(
      title: "Set<Int> init from unsafe buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        blackHole(Set(buffer))
      }
    }

    self.add(
      title: "Set<Int> sequential iteration",
      input: Int.self
    ) { size in
      let set = Set(0 ..< size)
      return { timer in
        for i in set {
          blackHole(i)
        }
      }
    }

    self.add(
      title: "Set<Int> sequential iteration, indices",
      input: Int.self
    ) { size in
      let set = Set(0 ..< size)
      return { timer in
        for i in set.indices {
          blackHole(set[i])
        }
      }
    }

    self.add(
      title: "Set<Int> successful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = Set(input)
      return { timer in
        for i in lookups {
          precondition(set.contains(i))
        }
      }
    }

    self.add(
      title: "Set<Int> unsuccessful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = Set(input)
      let lookups = lookups.map { $0 + input.count }
      return { timer in
        for i in lookups {
          precondition(!set.contains(i))
        }
      }
    }

    self.addSimple(
      title: "Set<Int> insert",
      input: [Int].self
    ) { input in
      var set: Set<Int> = []
      for i in input {
        set.insert(i)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "Set<Int> insert, reserving capacity",
      input: [Int].self
    ) { input in
      var set: Set<Int> = []
      set.reserveCapacity(input.count)
      for i in input {
        set.insert(i)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "Set<Int> insert, shared",
      input: [Int].self
    ) { input in
      var set: Set<Int> = []
      for i in input {
        let copy = set
        set.insert(i)
        blackHole(copy)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.add(
      title: "Set<Int> insert one + subtract, shared",
      input: [Int].self
    ) { input in
      let original = Set(input)
      let newMember = input.count
      return { timer in
        var copy = original
        copy.insert(newMember)
        let diff = copy.subtracting(original)
        precondition(diff.count == 1 && diff.first == newMember)
        blackHole(copy)
      }
    }

    self.addSimple(
      title: "Set<Int> model diffing",
      input: Int.self
    ) { input in
      typealias Model = Set<Int>

      var _state: Model = [] // Private
      func updateState(
        with model: Model
      ) -> (insertions: Model, removals: Model) {
        let insertions = model.subtracting(_state)
        let removals = _state.subtracting(model)
        _state = model
        return (insertions, removals)
      }

      var model: Model = []
      for i in 0 ..< input {
        model.insert(i)
        let r = updateState(with: model)
        precondition(r.insertions.count == 1 && r.removals.count == 0)
      }
    }

    self.add(
      title: "Set<Int> remove",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        var set = Set(input)
        for i in removals {
          set.remove(i)
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    self.add(
      title: "Set<Int> remove, shared",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        var set = Set(input)
        for i in removals {
          let copy = set
          set.remove(i)
          blackHole(copy)
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    let overlaps: [(String, (Int) -> Int)] = [
      ("0%",   { c in c }),
      ("25%",  { c in 3 * c / 4 }),
      ("50%",  { c in c / 2 }),
      ("75%",  { c in c / 4 }),
      ("100%", { c in 0 }),
    ]

    // SetAlgebra operations with Self
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> union with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Set(start ..< start + input.count)
          return { timer in
            blackHole(a.union(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> intersection with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Set(start ..< start + input.count)
          return { timer in
            blackHole(a.intersection(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> symmetricDifference with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Set(start ..< start + input.count)
          return { timer in
            blackHole(a.symmetricDifference(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> subtracting Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Set(start ..< start + input.count)
          return { timer in
            blackHole(a.subtracting(identity(b)))
          }
        }
      }
    }

    // SetAlgebra operations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> union with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.union(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> intersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.intersection(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> symmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.symmetricDifference(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> subtracting Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = Set(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.subtracting(identity(b)))
          }
        }
      }
    }

    // SetAlgebra mutations with Self
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formUnion with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Set(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formUnion(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formIntersection with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Set(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formIntersection(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formSymmetricDifference with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Set(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formSymmetricDifference(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> subtract Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Set(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.subtract(identity(b))
            }
            blackHole(a)
          }
        }
      }
    }

    // SetAlgebra mutations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formUnion with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formUnion(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formIntersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formIntersection(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> formSymmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.formSymmetricDifference(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "Set<Int> subtract Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = Set(input)
            timer.measure {
              a.subtract(identity(b))
            }
            blackHole(a)
          }
        }
      }
    }
    
    self.add(
      title: "Set<Int> equality, unique",
      input: Int.self
    ) { size in
      return { timer in
        let left = Set(0 ..< size)
        let right = Set(0 ..< size)
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "Set<Int> equality, shared",
      input: Int.self
    ) { size in
      return { timer in
        let left = Set(0 ..< size)
        let right = left
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
  }
}
