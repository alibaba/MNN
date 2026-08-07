//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Substring {
  func trimmed() -> Substring {
    guard let i = lastIndex(where: { $0 != " "}) else {
      return ""
    }
    return self[...i]
  }
}

extension String {
  public func trimmingLines() -> String {
    return self
      .split(separator: "\n", omittingEmptySubsequences: false)
      .map { $0.trimmed() }
      .joined(separator: "\n")
  }
}
