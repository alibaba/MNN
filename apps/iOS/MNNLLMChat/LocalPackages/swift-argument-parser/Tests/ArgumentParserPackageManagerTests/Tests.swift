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
import ArgumentParserTestHelpers

final class Tests: XCTestCase {
  override func setUp() {
    #if !os(Windows) && !os(WASI)
    unsetenv("COLUMNS")
    #endif
  }
}

extension Tests {
  func testParsing() throws {
    AssertParseCommand(Package.self, Package.Clean.self, ["clean"]) { clean in
      let options = clean.options
      XCTAssertEqual(options.buildPath, "./.build")
      XCTAssertEqual(options.configuration, .debug)
      XCTAssertEqual(options.automaticResolution, true)
      XCTAssertEqual(options.indexStore, true)
      XCTAssertEqual(options.packageManifestCaching, true)
      XCTAssertEqual(options.prefetching, true)
      XCTAssertEqual(options.sandbox, true)
      XCTAssertEqual(options.pubgrubResolver, false)
      XCTAssertEqual(options.staticSwiftStdlib, false)
      XCTAssertEqual(options.packagePath, ".")
      XCTAssertEqual(options.sanitize, false)
      XCTAssertEqual(options.skipUpdate, false)
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.cCompilerFlags, [])
      XCTAssertEqual(options.cxxCompilerFlags, [])
      XCTAssertEqual(options.linkerFlags, [])
      XCTAssertEqual(options.swiftCompilerFlags, [])
    }
  }
  
  func testParsingWithGlobalOption_1() {
    AssertParseCommand(Package.self, Package.GenerateXcodeProject.self, ["generate-xcodeproj", "--watch", "--output", "Foo", "--enable-automatic-resolution"]) { generate in
      XCTAssertEqual(generate.output, "Foo")
      XCTAssertFalse(generate.enableCodeCoverage)
      XCTAssertTrue(generate.watch)
      
      let options = generate.options
      // Default global option
      XCTAssertEqual(options.configuration, .debug)
      // Customized global option
      XCTAssertEqual(options.automaticResolution, true)
    }
  }
  
  func testParsingWithGlobalOption_2() {
    AssertParseCommand(Package.self, Package.GenerateXcodeProject.self, ["generate-xcodeproj", "--watch", "--output", "Foo", "--enable-automatic-resolution", "-Xcc", "-Ddebug"]) { generate in
      XCTAssertEqual(generate.output, "Foo")
      XCTAssertFalse(generate.enableCodeCoverage)
      XCTAssertTrue(generate.watch)
      
      let options = generate.options
      // Default global option
      XCTAssertEqual(options.configuration, .debug)
      // Customized global option
      XCTAssertEqual(options.automaticResolution, true)
      XCTAssertEqual(options.cCompilerFlags, ["-Ddebug"])
    }
  }
  
  func testParsingWithGlobalOption_3() {
    AssertParseCommand(Package.self, Package.GenerateXcodeProject.self, ["generate-xcodeproj", "--watch", "--output=Foo", "--enable-automatic-resolution", "-Xcc=-Ddebug"]) { generate in
      XCTAssertEqual(generate.output, "Foo")
      XCTAssertFalse(generate.enableCodeCoverage)
      XCTAssertTrue(generate.watch)
      
      let options = generate.options
      // Default global option
      XCTAssertEqual(options.configuration, .debug)
      // Customized global option
      XCTAssertEqual(options.automaticResolution, true)
      XCTAssertEqual(options.cCompilerFlags, ["-Ddebug"])
    }
  }
}
