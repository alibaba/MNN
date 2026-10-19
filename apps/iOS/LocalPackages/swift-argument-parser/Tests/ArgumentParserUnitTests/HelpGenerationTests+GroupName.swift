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

// This set of tests assert the help output matches the expected value for
// `@OptionGroup`s with names.

extension HelpGenerationTests {
  fileprivate struct Flags: ParsableArguments {
    @Flag(help: "example")
    var verbose: Bool = false
    
    @Flag(help: "example")
    var oversharing: Bool = false
  }
  
  fileprivate struct Options: ParsableArguments {
    @Option(help: "example")
    var name: String = ""
    
    @Option(help: "example")
    var age: Int
  }
  
  fileprivate struct FlagsAndOptions: ParsableArguments {
    @Flag(help: "example")
    var experimental: Bool = false
    
    @Option(help: "example")
    var prefix: String
  }
  
  fileprivate struct ArgsAndFlags: ParsableArguments {
    @Argument(help: "example")
    var name: String?
    
    @Argument(help: .init("example", visibility: .hidden))
    var title: String
    
    @Flag(help: "example")
    var existingUser: Bool = false
  }
  
  fileprivate struct AllVisible: ParsableCommand {
    @OptionGroup(title: "Flags Group")
    var flags: Flags
    
    @OptionGroup(title: "Options Group")
    var options: Options
    
    @OptionGroup
    var flagsAndOptions: FlagsAndOptions
    
    @OptionGroup
    var argsAndFlags: ArgsAndFlags
  }
  
  fileprivate struct ContainsOptionGroup: ParsableCommand {
    @OptionGroup(title: "Flags Group")
    var flags: Flags
    
    @OptionGroup
    var argsAndFlags: ArgsAndFlags
  }
  
  func testAllVisible() {
    AssertHelp(.default, for: AllVisible.self, equals: """
      USAGE: all-visible [--verbose] [--oversharing] [--name <name>] --age <age> [--experimental] --prefix <prefix> [<name>] [--existing-user]

      ARGUMENTS:
        <name>                  example

      FLAGS GROUP:
        --verbose               example
        --oversharing           example

      OPTIONS GROUP:
        --name <name>           example
        --age <age>             example

      OPTIONS:
        --experimental          example
        --prefix <prefix>       example
        --existing-user         example
        -h, --help              Show help information.

      """)
    
    AssertHelp(.hidden, for: AllVisible.self, equals: """
      USAGE: all-visible [--verbose] [--oversharing] [--name <name>] --age <age> [--experimental] --prefix <prefix> [<name>] <title> [--existing-user]

      ARGUMENTS:
        <name>                  example
        <title>                 example

      FLAGS GROUP:
        --verbose               example
        --oversharing           example

      OPTIONS GROUP:
        --name <name>           example
        --age <age>             example

      OPTIONS:
        --experimental          example
        --prefix <prefix>       example
        --existing-user         example
        -h, --help              Show help information.

      """)
  }
  
  fileprivate struct Combined: ParsableCommand {
    @OptionGroup(title: "Extras")
    var flags: Flags
    
    @OptionGroup(title: "Extras")
    var options: Options
    
    @OptionGroup(title: "Others")
    var flagsAndOptions: FlagsAndOptions
    
    @OptionGroup(title: "Others")
    var argsAndFlags: ArgsAndFlags
  }
  
  func testCombined() {
    AssertHelp(.default, for: Combined.self, equals: """
      USAGE: combined [--verbose] [--oversharing] [--name <name>] --age <age> [--experimental] --prefix <prefix> [<name>] [--existing-user]

      EXTRAS:
        --verbose               example
        --oversharing           example
        --name <name>           example
        --age <age>             example

      OTHERS:
        --experimental          example
        --prefix <prefix>       example
        <name>                  example
        --existing-user         example

      OPTIONS:
        -h, --help              Show help information.

      """)
    
    AssertHelp(.hidden, for: Combined.self, equals: """
      USAGE: combined [--verbose] [--oversharing] [--name <name>] --age <age> [--experimental] --prefix <prefix> [<name>] <title> [--existing-user]

      EXTRAS:
        --verbose               example
        --oversharing           example
        --name <name>           example
        --age <age>             example

      OTHERS:
        --experimental          example
        --prefix <prefix>       example
        <name>                  example
        <title>                 example
        --existing-user         example

      OPTIONS:
        -h, --help              Show help information.

      """)
  }
  
  fileprivate struct HiddenGroups: ParsableCommand {
    @OptionGroup(title: "Flags Group", visibility: .hidden)
    var flags: Flags
    
    @OptionGroup(title: "Options Group", visibility: .hidden)
    var options: Options
    
    @OptionGroup(visibility: .hidden)
    var flagsAndOptions: FlagsAndOptions
    
    @OptionGroup(visibility: .private)
    var argsAndFlags: ArgsAndFlags
  }
  
  func testHiddenGroups() {
    AssertHelp(.default, for: HiddenGroups.self, equals: """
      USAGE: hidden-groups

      OPTIONS:
        -h, --help              Show help information.

      """)
    
    AssertHelp(.hidden, for: HiddenGroups.self, equals: """
      USAGE: hidden-groups [--verbose] [--oversharing] [--name <name>] --age <age> [--experimental] --prefix <prefix>

      FLAGS GROUP:
        --verbose               example
        --oversharing           example

      OPTIONS GROUP:
        --name <name>           example
        --age <age>             example

      OPTIONS:
        --experimental          example
        --prefix <prefix>       example
        -h, --help              Show help information.

      """)
  }
  
  fileprivate struct ParentWithGroups: ParsableCommand {
    static var configuration: CommandConfiguration {
      .init(subcommands: [ChildWithGroups.self])
    }
    
    @OptionGroup(title: "Extras")
    var flags: Flags
        
    @OptionGroup
    var argsAndFlags: ArgsAndFlags
  
    fileprivate struct ChildWithGroups: ParsableCommand {
      @OptionGroup(title: "Child Extras")
      var flags: Flags

      @OptionGroup(title: "Extras")
      var options: Options

      @OptionGroup
      var argsAndFlags: ArgsAndFlags
    }
  }

  func testParentChild() {
    AssertHelp(.default, for: ParentWithGroups.self, equals: """
      USAGE: parent-with-groups [--verbose] [--oversharing] [<name>] [--existing-user] <subcommand>

      ARGUMENTS:
        <name>                  example

      EXTRAS:
        --verbose               example
        --oversharing           example

      OPTIONS:
        --existing-user         example
        -h, --help              Show help information.

      SUBCOMMANDS:
        child-with-groups

        See 'parent-with-groups help <subcommand>' for detailed help.
      """)
    
    AssertHelp(.hidden, for: ParentWithGroups.self, equals: """
      USAGE: parent-with-groups [--verbose] [--oversharing] [<name>] <title> [--existing-user] <subcommand>

      ARGUMENTS:
        <name>                  example
        <title>                 example

      EXTRAS:
        --verbose               example
        --oversharing           example

      OPTIONS:
        --existing-user         example
        -h, --help              Show help information.

      SUBCOMMANDS:
        child-with-groups

        See 'parent-with-groups help <subcommand>' for detailed help.
      """)
    
    AssertHelp(.default, for: ParentWithGroups.ChildWithGroups.self, root: ParentWithGroups.self, equals: """
      USAGE: parent-with-groups child-with-groups [--verbose] [--oversharing] [--name <name>] --age <age> [<name>] [--existing-user]

      ARGUMENTS:
        <name>                  example

      CHILD EXTRAS:
        --verbose               example
        --oversharing           example

      EXTRAS:
        --name <name>           example
        --age <age>             example

      OPTIONS:
        --existing-user         example
        -h, --help              Show help information.

      """)

    AssertHelp(.hidden, for: ParentWithGroups.ChildWithGroups.self, root: ParentWithGroups.self, equals: """
      USAGE: parent-with-groups child-with-groups [--verbose] [--oversharing] [--name <name>] --age <age> [<name>] <title> [--existing-user]

      ARGUMENTS:
        <name>                  example
        <title>                 example

      CHILD EXTRAS:
        --verbose               example
        --oversharing           example

      EXTRAS:
        --name <name>           example
        --age <age>             example

      OPTIONS:
        --existing-user         example
        -h, --help              Show help information.

      """)
  }

  fileprivate struct GroupsWithUnnamedGroups: ParsableCommand {
    @OptionGroup
    var extras: ContainsOptionGroup
  }

  func testUnnamedNestedGroups() {
    AssertHelp(.default, for: GroupsWithUnnamedGroups.self, equals: """
      USAGE: groups-with-unnamed-groups [--verbose] [--oversharing] [<name>] [--existing-user]

      ARGUMENTS:
        <name>                  example

      FLAGS GROUP:
        --verbose               example
        --oversharing           example

      OPTIONS:
        --existing-user         example
        -h, --help              Show help information.
      
      """)
  }

  fileprivate struct GroupsWithNamedGroups: ParsableCommand {
    @OptionGroup(title: "Nested")
    var extras: ContainsOptionGroup
  }

  func testNamedNestedGroups() {
    AssertHelp(.default, for: GroupsWithNamedGroups.self, equals: """
      USAGE: groups-with-named-groups [--verbose] [--oversharing] [<name>] [--existing-user]

      NESTED:
        --verbose               example
        --oversharing           example
        <name>                  example
        --existing-user         example

      OPTIONS:
        -h, --help              Show help information.
      
      """)
  }
}
