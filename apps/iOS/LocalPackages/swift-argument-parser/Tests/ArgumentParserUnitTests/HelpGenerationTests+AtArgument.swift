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
// valid combinations of @Argument.

extension HelpGenerationTests {
  enum AtArgumentTransform {
    // Not ExpressibleByArgument
    struct A { }

    struct BareNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A? = nil
    }

    struct OptionalDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A] = [A()]
    }
  }

  func testAtArgumentTransform_BareNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.BareNoDefault.self,
      equals: """
      USAGE: bare-no-default <arg0>

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_BareDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.BareDefault.self,
      equals: """
      USAGE: bare-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_OptionalNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.OptionalNoDefault.self,
      equals: """
      USAGE: optional-no-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_OptionalDefaultNil() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.OptionalDefaultNil.self,
      equals: """
      USAGE: optional-default-nil [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_OptionalDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.OptionalDefault.self,
      equals: """
      USAGE: optional-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_ArrayNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.ArrayNoDefault.self,
      equals: """
      USAGE: array-no-default <arg0> ...

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_ArrayDefaultEmpty() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.ArrayDefaultEmpty.self,
      equals: """
      USAGE: array-default-empty [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentTransform_ArrayDefault() {
    AssertHelp(
      .default,
      for: AtArgumentTransform.ArrayDefault.self,
      equals: """
      USAGE: array-default [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }
}

extension HelpGenerationTests {
  enum AtArgumentEBA {
    // ExpressibleByArgument
    struct A: ExpressibleByArgument {
      static var allValueStrings: [String] { ["A()"] }
      var defaultValueDescription: String { "A()" }
      init() { }
      init?(argument: String) { self.init() }
    }

    struct BareNoDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Argument(help: "example")
      var arg0: A? = nil
    }

    @available(*, deprecated, message: "Included for test coverage")
    struct OptionalDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Argument(help: "example")
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Argument(help: "example")
      var arg0: [A] = [A()]
    }
  }

  func testAtArgumentEBA_BareNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.BareNoDefault.self,
      equals: """
      USAGE: bare-no-default <arg0>

      ARGUMENTS:
        <arg0>                  example (values: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_BareDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.BareDefault.self,
      equals: """
      USAGE: bare-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example (values: A(); default: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_OptionalNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.OptionalNoDefault.self,
      equals: """
      USAGE: optional-no-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example (values: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_OptionalDefaultNil() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.OptionalDefaultNil.self,
      equals: """
      USAGE: optional-default-nil [<arg0>]

      ARGUMENTS:
        <arg0>                  example (values: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_ArrayNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.ArrayNoDefault.self,
      equals: """
      USAGE: array-no-default <arg0> ...

      ARGUMENTS:
        <arg0>                  example (values: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_ArrayDefaultEmpty() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.ArrayDefaultEmpty.self,
      equals: """
      USAGE: array-default-empty [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example (values: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBA_ArrayDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBA.ArrayDefault.self,
      equals: """
      USAGE: array-default [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example (values: A(); default: A())

      OPTIONS:
        -h, --help              Show help information.

      """)
  }
}

extension HelpGenerationTests {
  enum AtArgumentEBATransform {
    // ExpressibleByArgument with Transform
    struct A: ExpressibleByArgument {
      static var allValueStrings: [String] { ["A()"] }
      var defaultValueDescription: String { "A()" }
      init() { }
      init?(argument: String) { self.init() }
    }

    struct BareNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A
    }

    struct BareDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A = A()
    }

    struct OptionalNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A?
    }

    struct OptionalDefaultNil: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A? = nil
    }

    struct OptionalDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: A? = A()
    }

    struct ArrayNoDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A]
    }

    struct ArrayDefaultEmpty: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A] = []
    }

    struct ArrayDefault: ParsableCommand {
      @Argument(help: "example", transform: { _ in A() })
      var arg0: [A] = [A()]
    }
  }

  func testAtArgumentEBATransform_BareNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.BareNoDefault.self,
      equals: """
      USAGE: bare-no-default <arg0>

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_BareDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.BareDefault.self,
      equals: """
      USAGE: bare-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_OptionalNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.OptionalNoDefault.self,
      equals: """
      USAGE: optional-no-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_OptionalDefaultNil() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.OptionalDefaultNil.self,
      equals: """
      USAGE: optional-default-nil [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_OptionalDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.OptionalDefault.self,
      equals: """
      USAGE: optional-default [<arg0>]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_ArrayNoDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.ArrayNoDefault.self,
      equals: """
      USAGE: array-no-default <arg0> ...

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_ArrayDefaultEmpty() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.ArrayDefaultEmpty.self,
      equals: """
      USAGE: array-default-empty [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }

  func testAtArgumentEBATransform_ArrayDefault() {
    AssertHelp(
      .default,
      for: AtArgumentEBATransform.ArrayDefault.self,
      equals: """
      USAGE: array-default [<arg0> ...]

      ARGUMENTS:
        <arg0>                  example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }
}
