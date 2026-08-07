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
import BitCollections

extension Benchmark {
  
  public mutating func addBitSetBenchmarks() {
    let fillRatios: [(String, (Int) -> Int)] = [
      ("0%",    { c in 0 }),
      ("0.01%",  { c in c / 10_000 }),
      ("0.1%",  { c in c / 1_000 }),
      ("1%",   { c in c / 100 }),
      ("10%",   { c in c / 10 }),
      ("25%",  { c in c / 4 }),
      ("50%",  { c in c / 2 }),
      ("75%",  { c in 3 * c / 4 }),
      ("100%", { c in c }),
    ]

    for (percentage, count) in fillRatios {
      self.add(
        title: "BitSet iteration (\(percentage) filled)",
        input: Int.self
      ) { input in
        guard input > 0 else { return nil }

        var set = BitSet(reservingCapacity: input)
        for i in (0 ..< input).shuffled().prefix(count(input)) {
          set.insert(i)
        }
        // Make sure the set actually fills its storage capacity.
        set.insert(input - 1)

        return { timer in
          var c = 0
          timer.measure {
            for _ in set {
              c += 1
            }
          }
          precondition(c == set.count)
        }
      }
    }

    self.add(
      title: "BitSet distance(from:to:)",
      input: Int.self
    ) { input in
      let set = BitSet.random(upTo: input)
      let c = set.count
      return { timer in
        let d = set.distance(from: set.startIndex, to: set.endIndex)
        precondition(d == c)
      }
    }

    self.add(
      title: "BitSet index(offsetBy:) +25% steps",
      input: Int.self
    ) { input in
      let set = BitSet.random(upTo: input)
      let c = set.count
      return { timer in
        var i = set.startIndex
        var j = 0
        for step in 1 ... 4 {
          let nextJ = step * c / 4
          i = set.index(i, offsetBy: nextJ - j)
          j = nextJ
        }
        precondition(i == set.endIndex)
      }
    }

    self.add(
      title: "BitSet index(offsetBy:) -25% steps",
      input: Int.self
    ) { input in
      let set = BitSet.random(upTo: input)
      let c = set.count
      return { timer in
        var i = set.endIndex
        var j = c
        for step in stride(from: 3, through: 0, by: -1) {
          let nextJ = step * c / 4
          i = set.index(i, offsetBy: nextJ - j)
          j = nextJ
        }
        precondition(i == set.startIndex)
      }
    }

    self.add(
      title: "BitSet contains (within bounds)",
      input: Int.self
    ) { input in
      let set = BitSet.random(upTo: input)
      return { timer in
        var c = 0
        timer.measure {
          for value in 0 ..< input {
            if set.contains(value) { c += 1 }
          }
        }
        precondition(c == set.count)
      }
    }
    
    self.add(
      title: "BitSet contains (out of bounds)",
      input: Int.self
    ) { input in
      let set = BitSet.random(upTo: input)
      return { timer in
        for value in input ..< input * 2 {
          precondition(!set.contains(value))
        }
      }
    }
    
    self.add(
      title: "BitSet insert",
      input: [Int].self
    ) { input in
      return { timer in
        var set: BitSet = []
        for i in input {
          precondition(set.insert(i).inserted)
        }
      }
    }

    self.add(
      title: "BitSet insert, reserving capacity",
      input: [Int].self
    ) { input in
      return { timer in
        var set: BitSet = []
        set.reserveCapacity(input.count)
        for i in input {
          precondition(set.insert(i).inserted)
        }
      }
    }

    self.add(
      title: "BitSet union with Self",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = BitSet.random(upTo: input)
      return { timer in
        blackHole(a.union(b))
      }
    }

    self.add(
      title: "BitSet union with Array<Int>",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        blackHole(a.union(b))
      }
    }

    self.add(
      title: "BitSet formUnion with Self",
      input: Int.self
    ) { input in
      let b = BitSet.random(upTo: input)
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formUnion(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet formUnion with Array<Int>",
      input: Int.self
    ) { input in
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formUnion(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet intersection with Self",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = BitSet.random(upTo: input)
      return { timer in
        blackHole(a.intersection(b))
      }
    }
    
    self.add(
      title: "BitSet intersection with Array<Int>",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        blackHole(a.intersection(b))
      }
    }

    self.add(
      title: "BitSet formIntersection with Self",
      input: Int.self
    ) { input in
      let b = BitSet.random(upTo: input)
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formIntersection(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet formIntersection with Array<Int>",
      input: Int.self
    ) { input in
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formIntersection(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet symmetricDifference with Self",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = BitSet.random(upTo: input)
      return { timer in
        blackHole(a.symmetricDifference(b))
      }
    }
    
    self.add(
      title: "BitSet symmetricDifference with Array<Int>",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        blackHole(a.symmetricDifference(b))
      }
    }

    self.add(
      title: "BitSet formSymmetricDifference with Self",
      input: Int.self
    ) { input in
      let b = BitSet.random(upTo: input)
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formSymmetricDifference(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet formSymmetricDifference with Array<Int>",
      input: Int.self
    ) { input in
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.formSymmetricDifference(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet subtracting Self",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = BitSet.random(upTo: input)
      return { timer in
        blackHole(a.subtracting(b))
      }
    }

    self.add(
      title: "BitSet subtracting Array<Int>",
      input: Int.self
    ) { input in
      let a = BitSet.random(upTo: input)
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        blackHole(a.subtracting(b))
      }
    }

    self.add(
      title: "BitSet subtract Self",
      input: Int.self
    ) { input in
      let b = BitSet.random(upTo: input)
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.subtract(b)
        }
        blackHole(a)
      }
    }

    self.add(
      title: "BitSet subtract Array<Int>",
      input: Int.self
    ) { input in
      let b = Array((0 ..< input).shuffled()[0 ..< input / 2])
      return { timer in
        var a = BitSet.random(upTo: input)
        timer.measure {
          a.subtract(b)
        }
        blackHole(a)
      }
    }
  }
}
