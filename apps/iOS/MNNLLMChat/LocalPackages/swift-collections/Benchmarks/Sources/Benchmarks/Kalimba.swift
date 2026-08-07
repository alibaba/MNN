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

import DequeModule

extension Sequence {
  func kalimbaOrdered() -> [Element] {
    var kalimba: [Element] = []
    kalimba.reserveCapacity(underestimatedCount)
    var insertAtStart = false
    for element in self {
      if insertAtStart {
        kalimba.insert(element, at: 0)
      } else {
        kalimba.append(element)
      }
      insertAtStart.toggle()
    }
    return kalimba
  }

  func kalimbaOrdered2() -> Deque<Element> {
    var kalimba: Deque<Element> = []
    kalimba.reserveCapacity(underestimatedCount)
    var insertAtStart = false
    for element in self {
      if insertAtStart {
        kalimba.prepend(element)
      } else {
        kalimba.append(element)
      }
      insertAtStart.toggle()
    }
    return kalimba
  }

  func kalimbaOrdered3() -> [Element] {
    var odds: [Element] = []
    var evens: [Element] = []
    odds.reserveCapacity(underestimatedCount)
    evens.reserveCapacity(underestimatedCount / 2)
    var insertAtStart = false
    for element in self {
      if insertAtStart {
        odds.append(element)
      } else {
        evens.append(element)
      }
      insertAtStart.toggle()
    }
    odds.reverse()
    odds.append(contentsOf: evens)
    return odds
  }

}
