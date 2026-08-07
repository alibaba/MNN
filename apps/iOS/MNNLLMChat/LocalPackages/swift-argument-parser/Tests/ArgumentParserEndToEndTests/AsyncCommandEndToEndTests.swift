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

import XCTest
import ArgumentParser

final class AsyncCommandEndToEndTests: XCTestCase {
}

actor AsyncStatusCheck {
  struct Status: OptionSet {
    var rawValue: UInt8
    
    static var root: Self { .init(rawValue: 1 << 0) }
    static var sub: Self  { .init(rawValue: 1 << 1) }
  }
  
  @MainActor
  var status: Status = []
  
  @MainActor
  func update(_ status: Status) {
    self.status.insert(status)
  }
}

var statusCheck = AsyncStatusCheck()

// MARK: AsyncParsableCommand.main() testing

struct AsyncCommand: AsyncParsableCommand {
  static var configuration: CommandConfiguration {
    .init(subcommands: [SubCommand.self])
  }
  
  func run() async throws {
    await statusCheck.update(.root)
  }
  
  struct SubCommand: AsyncParsableCommand {
    func run() async throws {
      await statusCheck.update(.sub)
    }
  }
}

extension AsyncCommandEndToEndTests {
  @MainActor
  func testAsyncMain_root() async throws {
    XCTAssertFalse(statusCheck.status.contains(.root))
    await AsyncCommand.main([])
    XCTAssertTrue(statusCheck.status.contains(.root))
  }
  
  @MainActor
  func testAsyncMain_sub() async throws {
    XCTAssertFalse(statusCheck.status.contains(.sub))
    await AsyncCommand.main(["sub-command"])
    XCTAssertTrue(statusCheck.status.contains(.sub))
  }
}
