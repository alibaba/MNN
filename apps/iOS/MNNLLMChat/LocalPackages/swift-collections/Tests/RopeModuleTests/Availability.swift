//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

var isRunningOnSwiftStdlib5_8: Bool {
  if #available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *) {
    return true
  }
  return false
}
