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

import ArgumentParser
import Foundation

extension Foundation.Date: ArgumentParser.ExpressibleByArgument {
  // parsed as `yyyy-mm-dd`
  public init?(argument: String) {
    // ensure the input argument is composed of exactly 3 components separated
    // by dashes ('-')
    let components = argument.split(separator: "-")
    let empty = components.filter { $0.isEmpty }
    guard components.count == 3, empty.count == 0 else { return nil }

    // ensure the year component is exactly 4 characters
    let _year = components[0]
    guard _year.count == 4, let year = Int(_year) else { return nil }

    // ensure the month component is exactly 2 characters
    let _month = components[1]
    guard _month.count == 2, let month = Int(_month) else { return nil }

    // ensure the day component is exactly 2 characters
    let _day = components[2]
    guard _day.count == 2, let day = Int(_day) else { return nil }

    // ensure the combination of year, month, day is valid
    let dateComponents = DateComponents(
      calendar: Calendar(identifier: .iso8601),
      timeZone: TimeZone(identifier: "UTC"),
      year: year,
      month: month,
      day: day)
    guard dateComponents.isValidDate else { return nil }
    guard let date = dateComponents.date else { return nil }
    self = date
  }
}
