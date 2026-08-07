//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2022 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import XCTest
import ArgumentParserTestHelpers
@testable import ArgumentParser

// This set of tests assert the help output matches the expected value for all
// valid combinations of @Option.

extension HelpGenerationTests {
  enum AtOptionTransform {
    // Not ExpressibleByArgument
    struct A { }

    struct BareNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A? = nil
    }

    struct OptionalDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A] = [A()]
    }
  }

  func testAtOptionTransform_BareNoDefault() {
    AssertHelp(.default, for: AtOptionTransform.BareNoDefault.self, equals: """
      USAGE: bare-no-default --arg0 <arg0>

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_BareDefault() {
    AssertHelp(.default, for: AtOptionTransform.BareDefault.self, equals: """
      USAGE: bare-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_OptionalNoDefault() {
    AssertHelp(.default, for: AtOptionTransform.OptionalNoDefault.self, equals: """
      USAGE: optional-no-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_OptionalDefaultNil() {
    AssertHelp(.default, for: AtOptionTransform.OptionalDefaultNil.self, equals: """
      USAGE: optional-default-nil [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_OptionalDefault() {
    AssertHelp(.default, for: AtOptionTransform.OptionalDefault.self, equals: """
      USAGE: optional-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_ArrayNoDefault() {
    AssertHelp(.default, for: AtOptionTransform.ArrayNoDefault.self, equals: """
      USAGE: array-no-default --arg0 <arg0> ...

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_ArrayDefaultEmpty() {
    AssertHelp(.default, for: AtOptionTransform.ArrayDefaultEmpty.self, equals: """
      USAGE: array-default-empty [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionTransform_ArrayDefault() {
    AssertHelp(.default, for: AtOptionTransform.ArrayDefault.self, equals: """
      USAGE: array-default [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }
}

extension HelpGenerationTests {
  enum AtOptionEBA {
    // ExpressibleByArgument
    struct A: ExpressibleByArgument {
      static var allValueStrings: [String] { ["A()"] }
      var defaultValueDescription: String { "A()" }
      init() { }
      init?(argument: String) { self.init() }
    }

    struct BareNoDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Option(help: "example")
      var arg0: A? = nil
    }

    @available(*, deprecated, message: "Included for test coverage")
    struct OptionalDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Option(help: "example")
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Option(help: "example")
      var arg0: [A] = [A()]
    }
  }

  func testAtOptionEBA_BareNoDefault() {
    AssertHelp(.default, for: AtOptionEBA.BareNoDefault.self, equals: """
      USAGE: bare-no-default --arg0 <arg0>

      OPTIONS:
        --arg0 <arg0>           example (values: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_BareDefault() {
    AssertHelp(.default, for: AtOptionEBA.BareDefault.self, equals: """
      USAGE: bare-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example (values: A(); default: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_OptionalNoDefault() {
    AssertHelp(.default, for: AtOptionEBA.OptionalNoDefault.self, equals: """
      USAGE: optional-no-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example (values: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_OptionalDefaultNil() {
    AssertHelp(.default, for: AtOptionEBA.OptionalDefaultNil.self, equals: """
      USAGE: optional-default-nil [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example (values: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_ArrayNoDefault() {
    AssertHelp(.default, for: AtOptionEBA.ArrayNoDefault.self, equals: """
      USAGE: array-no-default --arg0 <arg0> ...

      OPTIONS:
        --arg0 <arg0>           example (values: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_ArrayDefaultEmpty() {
    AssertHelp(.default, for: AtOptionEBA.ArrayDefaultEmpty.self, equals: """
      USAGE: array-default-empty [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example (values: A())
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBA_ArrayDefault() {
    AssertHelp(.default, for: AtOptionEBA.ArrayDefault.self, equals: """
      USAGE: array-default [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example (values: A(); default: A())
        -h, --help              Show help information.

      """)
  }
}

extension HelpGenerationTests {
  enum AtOptionEBATransform {
    // ExpressibleByArgument with Transform
    struct A: ExpressibleByArgument {
      static var allValueStrings: [String] { ["A()"] }
      var defaultValueDescription: String { "A()" }
      init() { }
      init?(argument: String) { self.init() }
    }

    struct BareNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A? = nil
    }

    struct OptionalDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Option(help: "example", transform: { _ in A() })
      var arg0: [A] = [A()]
    }
  }

  func testAtOptionEBATransform_BareNoDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.BareNoDefault.self, equals: """
      USAGE: bare-no-default --arg0 <arg0>

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_BareDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.BareDefault.self, equals: """
      USAGE: bare-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_OptionalNoDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.OptionalNoDefault.self, equals: """
      USAGE: optional-no-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_OptionalDefaultNil() {
    AssertHelp(.default, for: AtOptionEBATransform.OptionalDefaultNil.self, equals: """
      USAGE: optional-default-nil [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_OptionalDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.OptionalDefault.self, equals: """
      USAGE: optional-default [--arg0 <arg0>]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_ArrayNoDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.ArrayNoDefault.self, equals: """
      USAGE: array-no-default --arg0 <arg0> ...

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_ArrayDefaultEmpty() {
    AssertHelp(.default, for: AtOptionEBATransform.ArrayDefaultEmpty.self, equals: """
      USAGE: array-default-empty [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }

  func testAtOptionEBATransform_ArrayDefault() {
    AssertHelp(.default, for: AtOptionEBATransform.ArrayDefault.self, equals: """
      USAGE: array-default [--arg0 <arg0> ...]

      OPTIONS:
        --arg0 <arg0>           example
        -h, --help              Show help information.

      """)
  }
}
