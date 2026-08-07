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
import ArgumentParserTestHelpers
import ArgumentParser

final class JoinedEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Foo: ParsableArguments {
  @Option(name: .customShort("f"))
  var file = ""
  
  @Option(name: .customShort("d", allowingJoined: true))
  var debug = ""
  
  @Flag(name: .customLong("fdi", withSingleDash: true))
  var fdi = false
}

extension JoinedEndToEndTests {
  func testSingleValueParsing() throws {
    AssertParse(Foo.self, []) { foo in
      XCTAssertEqual(foo.file, "")
      XCTAssertEqual(foo.debug, "")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-f", "file", "-d=Debug"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-f", "file", "-d", "Debug"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-f", "file", "-dDebug"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-dDebug", "-f", "file"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-dDebug"]) { foo in
      XCTAssertEqual(foo.file, "")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-fd", "file", "Debug"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, false)
    }

    AssertParse(Foo.self, ["-fd", "file", "Debug", "-fdi"]) { foo in
      XCTAssertEqual(foo.file, "file")
      XCTAssertEqual(foo.debug, "Debug")
      XCTAssertEqual(foo.fdi, true)
    }

    AssertParse(Foo.self, ["-fdi"]) { foo in
      XCTAssertEqual(foo.file, "")
      XCTAssertEqual(foo.debug, "")
      XCTAssertEqual(foo.fdi, true)
    }
  }
  
  func testSingleValueParsing_Fails() throws {
    XCTAssertThrowsError(try Foo.parse(["-f", "-d"]))
    XCTAssertThrowsError(try Foo.parse(["-f", "file", "-d"]))
    XCTAssertThrowsError(try Foo.parse(["-fd", "file"]))
    XCTAssertThrowsError(try Foo.parse(["-fdDebug", "file"]))
    XCTAssertThrowsError(try Foo.parse(["-fFile"]))
  }
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Option(name: .customShort("D", allowingJoined: true))
  var debug: [String] = []
}

extension JoinedEndToEndTests {
  func testArrayValueParsing() throws {
    AssertParse(Bar.self, []) { bar in
      XCTAssertEqual(bar.debug, [])
    }

    AssertParse(Bar.self, ["-Ddebug1"]) { bar in
      XCTAssertEqual(bar.debug, ["debug1"])
    }

    AssertParse(Bar.self, ["-Ddebug1", "-Ddebug2", "-Ddebug3"]) { bar in
      XCTAssertEqual(bar.debug, ["debug1", "debug2", "debug3"])
    }

    AssertParse(Bar.self, ["-D", "debug1", "-Ddebug2", "-D", "debug3"]) { bar in
      XCTAssertEqual(bar.debug, ["debug1", "debug2", "debug3"])
    }
  }
  
  func testArrayValueParsing_Fails() throws {
    XCTAssertThrowsError(try Bar.parse(["-D"]))
    XCTAssertThrowsError(try Bar.parse(["-Ddebug1", "debug2"]))
  }
}

// MARK: -

fileprivate struct Baz: ParsableArguments {
  @Option(name: .customShort("D", allowingJoined: true), parsing: .upToNextOption)
  var debug: [String] = []
  
  @Flag var verbose = false
}

extension JoinedEndToEndTests {
  func testArrayUpToNextParsing() throws {
    AssertParse(Baz.self, []) { baz in
      XCTAssertEqual(baz.debug, [])
    }
    
    AssertParse(Baz.self, ["-Ddebug1", "debug2"]) { baz in
      XCTAssertEqual(baz.debug, ["debug1", "debug2"])
      XCTAssertEqual(baz.verbose, false)
    }
    
    AssertParse(Baz.self, ["-Ddebug1", "debug2", "--verbose"]) { baz in
      XCTAssertEqual(baz.debug, ["debug1", "debug2"])
      XCTAssertEqual(baz.verbose, true)
    }
    
    AssertParse(Baz.self, ["-Ddebug1", "debug2", "-Ddebug3", "debug4"]) { baz in
      XCTAssertEqual(baz.debug, ["debug1", "debug2", "debug3", "debug4"])
    }
  }
  
  func testArrayUpToNextParsing_Fails() throws {
    XCTAssertThrowsError(try Baz.parse(["-D", "--other"]))
    XCTAssertThrowsError(try Baz.parse(["-Ddebug", "--other"]))
    XCTAssertThrowsError(try Baz.parse(["-Ddebug", "--other"]))
    XCTAssertThrowsError(try Baz.parse(["-Ddebug", "debug", "--other"]))
  }
}

// MARK: -

fileprivate struct Qux: ParsableArguments {
  @Option(name: .customShort("D", allowingJoined: true), parsing: .remaining)
  var debug: [String] = []
}

extension JoinedEndToEndTests {
  func testArrayRemainingParsing() throws {
    AssertParse(Qux.self, []) { qux in
      XCTAssertEqual(qux.debug, [])
    }
    
    AssertParse(Qux.self, ["-Ddebug1", "debug2"]) { qux in
      XCTAssertEqual(qux.debug, ["debug1", "debug2"])
    }
    
    AssertParse(Qux.self, ["-Ddebug1", "debug2", "-Ddebug3", "debug4", "--other"]) { qux in
      XCTAssertEqual(qux.debug, ["debug1", "debug2", "-Ddebug3", "debug4", "--other"])
    }
  }
  
  func testArrayRemainingParsing_Fails() throws {
    XCTAssertThrowsError(try Baz.parse(["--other", "-Ddebug", "debug"]))
  }
}
