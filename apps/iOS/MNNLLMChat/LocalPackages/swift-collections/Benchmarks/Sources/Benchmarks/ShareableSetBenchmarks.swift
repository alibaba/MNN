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
import HashTreeCollections

extension Benchmark {
  public mutating func addTreeSetBenchmarks() {
    self.addSimple(
      title: "TreeSet<Int> init from range",
      input: Int.self
    ) { size in
      blackHole(TreeSet(0 ..< size))
    }

    self.addSimple(
      title: "TreeSet<Int> init from unsafe buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        blackHole(TreeSet(buffer))
      }
    }

    self.add(
      title: "TreeSet<Int> sequential iteration",
      input: Int.self
    ) { size in
      let set = TreeSet(0 ..< size)
      return { timer in
        for i in set {
          blackHole(i)
        }
      }
    }

    self.add(
      title: "TreeSet<Int> sequential iteration, indices",
      input: Int.self
    ) { size in
      let set = TreeSet(0 ..< size)
      return { timer in
        for i in set.indices {
          blackHole(set[i])
        }
      }
    }

    self.add(
      title: "TreeSet<Int> successful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = TreeSet(input)
      return { timer in
        for i in lookups {
          precondition(set.contains(i))
        }
      }
    }

    self.add(
      title: "TreeSet<Int> unsuccessful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = TreeSet(input)
      let lookups = lookups.map { $0 + input.count }
      return { timer in
        for i in lookups {
          precondition(!set.contains(i))
        }
      }
    }

    self.addSimple(
      title: "TreeSet<Int> insert",
      input: [Int].self
    ) { input in
      var set: TreeSet<Int> = []
      for i in input {
        set.insert(i)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "TreeSet<Int> insert, shared",
      input: [Int].self
    ) { input in
      var set: TreeSet<Int> = []
      for i in input {
        let copy = set
        set.insert(i)
        blackHole(copy)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "TreeSet<Int> model diffing",
      input: Int.self
    ) { input in
      typealias Model = TreeSet<Int>

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
      title: "TreeSet<Int> remove",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        var set = TreeSet(input)
        for i in removals {
          set.remove(i)
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    self.add(
      title: "TreeSet<Int> remove, shared",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        var set = TreeSet(input)
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
      func makeB(
        _ a: TreeSet<Int>, _ range: Range<Int>, shared: Bool
      ) -> TreeSet<Int> {
        guard shared else {
          return TreeSet(range)
        }
        var b = a
        b.subtract(0 ..< range.lowerBound)
        b.formUnion(range)
        return b
      }

      for (percentage, start) in overlaps {
        for shared in [false, true] {
          let qualifier = "\(percentage) overlap, \(shared ? "shared" : "distinct")"

          self.add(
            title: "TreeSet<Int> union with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            let a = TreeSet(input)
            let b = makeB(a, start ..< start + input.count, shared: shared)
            return { timer in
              blackHole(a.union(identity(b)))
            }
          }

          self.add(
            title: "TreeSet<Int> intersection with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            let a = TreeSet(input)
            let b = makeB(a, start ..< start + input.count, shared: shared)
            return { timer in
              blackHole(a.intersection(identity(b)))
            }
          }

          self.add(
            title: "TreeSet<Int> symmetricDifference with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            let a = TreeSet(input)
            let b = makeB(a, start ..< start + input.count, shared: shared)
            return { timer in
              blackHole(a.symmetricDifference(identity(b)))
            }
          }

          self.add(
            title: "TreeSet<Int> subtracting Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            let a = TreeSet(input)
            let b = makeB(a, start ..< start + input.count, shared: shared)
            return { timer in
              blackHole(a.subtracting(identity(b)))
            }
          }
        }
      }
    }

    // SetAlgebra operations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> union with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = TreeSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.union(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> intersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = TreeSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.intersection(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> symmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = TreeSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.symmetricDifference(identity(b)))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> subtracting Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = TreeSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.subtracting(identity(b)))
          }
        }
      }
    }

    // SetAlgebra mutations with Self
    do {
      func makeB(
        _ a: TreeSet<Int>, _ range: Range<Int>, shared: Bool
      ) -> TreeSet<Int> {
        guard shared else {
          return TreeSet(range)
        }
        var b = a
        b.subtract(0 ..< range.lowerBound)
        b.formUnion(range)
        return b
      }

      for (percentage, start) in overlaps {
        for shared in [false, true] {
          let qualifier = "\(percentage) overlap, \(shared ? "shared" : "distinct")"

          self.add(
            title: "TreeSet<Int> formUnion with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            return { timer in
              var a = TreeSet(input)
              let b = makeB(a, start ..< start + input.count, shared: shared)
              timer.measure {
                a.formUnion(identity(b))
              }
              blackHole(a)
            }
          }

          self.add(
            title: "TreeSet<Int> formIntersection with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            return { timer in
              var a = TreeSet(input)
              let b = makeB(a, start ..< start + input.count, shared: shared)
              timer.measure {
                a.formIntersection(identity(b))
              }
              blackHole(a)
            }
          }

          self.add(
            title: "TreeSet<Int> formSymmetricDifference with Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            return { timer in
              var a = TreeSet(input)
              let b = makeB(a, start ..< start + input.count, shared: shared)
              timer.measure {
                a.formSymmetricDifference(identity(b))
              }
              blackHole(a)
            }
          }

          self.add(
            title: "TreeSet<Int> subtract Self (\(qualifier))",
            input: [Int].self
          ) { input in
            let start = start(input.count)
            return { timer in
              var a = TreeSet(input)
              let b = makeB(a, start ..< start + input.count, shared: shared)
              timer.measure {
                a.subtract(identity(b))
              }
              blackHole(a)
            }
          }
        }
      }
    }

    // SetAlgebra mutations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> formUnion with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = TreeSet(input)
            timer.measure {
              a.formUnion(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> formIntersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = TreeSet(input)
            timer.measure {
              a.formIntersection(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> formSymmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = TreeSet(input)
            timer.measure {
              a.formSymmetricDifference(identity(b))
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "TreeSet<Int> subtract Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = TreeSet(input)
            timer.measure {
              a.subtract(identity(b))
            }
            blackHole(a)
          }
        }
      }
    }
    
    self.add(
      title: "TreeSet<Int> equality, unique",
      input: Int.self
    ) { size in
      return { timer in
        let left = TreeSet(0 ..< size)
        let right = TreeSet(0 ..< size)
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "TreeSet<Int> equality, shared",
      input: Int.self
    ) { size in
      return { timer in
        let left = TreeSet(0 ..< size)
        let right = left
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
  }
}
