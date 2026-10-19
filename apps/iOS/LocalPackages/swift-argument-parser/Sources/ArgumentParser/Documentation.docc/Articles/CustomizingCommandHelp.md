# Customizing Help for Commands

Define your command's abstract, extended discussion, or usage string, and set the flags used to invoke the help display.

## Overview

In addition to configuring the command name and subcommands, as described in <doc:CommandsAndSubcommands>, you can also configure a command's help text by providing an abstract, discussion, or custom usage string.

```swift
struct Repeat: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Repeats your input phrase.",
        usage: """
            repeat <phrase>
            repeat --count <count> <phrase>
            """,
        discussion: """
            Prints to stdout forever, or until you halt the program.
            """)

    @Argument(help: "The phrase to repeat.")
    var phrase: String

    @Option(help: "How many times to repeat.")
    var count: Int? = nil

    mutating func run() throws {
        for _ in 0..<(count ?? 2) {
            print(phrase) 
        }
    }
}
```

The customized components now appear in the generated help screen:

```
% repeat --help
OVERVIEW: Repeats your input phrase.

Prints to stdout forever, or until you halt the program.

USAGE: repeat <phrase>
       repeat --count <count> <phrase>

ARGUMENTS:
  <phrase>                The phrase to repeat.

OPTIONS:
  -h, --help              Show help information.

% repeat hello!
hello!
hello!
hello!
hello!
hello!
hello!
...
```

## Modifying the Help Flag Names

Users can see the help screen for a command by passing either the `-h` or the `--help` flag, by default. If you need to use one of those flags for another purpose, you can provide alternative names when configuring a root command.

```swift
struct Example: ParsableCommand {
    static let configuration = CommandConfiguration(
        helpNames: [.long, .customShort("?")])

    @Option(name: .shortAndLong, help: "The number of history entries to show.")
    var historyDepth: Int

    mutating func run() throws {
        printHistory(depth: historyDepth)
    }
}
```

When running the command, `-h` matches the short name of the `historyDepth` property, and `-?` displays the help screen.

```
% example -h 3
nmap -v -sS -O 10.2.2.2
sshnuke 10.2.2.2 -rootpw="Z1ON0101"
ssh 10.2.2.2 -l root
% example -?
USAGE: example --history-depth <history-depth>

ARGUMENTS:
  <phrase>                The phrase to repeat.

OPTIONS:
  -h, --history-depth     The number of history entries to show.
  -?, --help              Show help information.
```

When not overridden, custom help names are inherited by subcommands. In this example, the parent command defines `--help` and `-?` as its help names:

```swift
struct Parent: ParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [Child.self],
        helpNames: [.long, .customShort("?")])

    struct Child: ParsableCommand {
        @Option(name: .shortAndLong, help: "The host the server will run on.")
        var host: String
    }
}
```

The `child` subcommand inherits the parent's help names, allowing the user to distinguish between the host argument (`-h`) and help (`-?`).

```
% parent child -h 192.0.0.0
...
% parent child -?
USAGE: parent child --host <host>

OPTIONS:
  -h, --host <host>       The host the server will run on.
  -?, --help              Show help information.
```

## Hiding Commands

You may not want to show every one of your command as part of your command-line interface. To render a command invisible (but still usable), pass `shouldDisplay: false` to the ``CommandConfiguration`` initializer.

## Generating Help Text Programmatically

The help screen is automatically shown to users when they call your command with the help flag. You can generate the same text from within your program by calling the `helpMessage()` method.

```swift
let help = Repeat.helpMessage()
// `help` matches the output above

let fortyColumnHelp = Repeat.helpMessage(columns: 40)
// `fortyColumnHelp` is the same help screen, but wrapped to 40 columns
```

When generating help text for a subcommand, call `helpMessage(for:)` on the `ParsableCommand` type that represents the root of the command tree and pass the subcommand type as a parameter to ensure the correct display.
