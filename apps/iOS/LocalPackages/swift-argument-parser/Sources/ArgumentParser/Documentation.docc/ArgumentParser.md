# ``ArgumentParser``

Straightforward, type-safe argument parsing for Swift.

## Overview

By using `ArgumentParser`, you can create a command-line interface tool
by declaring simple Swift types.
Begin by declaring a type that defines
the information that you need to collect from the command line.
Decorate each stored property with one of `ArgumentParser`'s property wrappers,
declare conformance to ``ParsableCommand``,
and implement your command's logic in its `run()` method. 
For `async` renditions of `run`, declare ``AsyncParsableCommand`` conformance instead.

```swift
import ArgumentParser

@main
struct Repeat: ParsableCommand {
    @Argument(help: "The phrase to repeat.")
    var phrase: String

    @Option(help: "The number of times to repeat 'phrase'.")
    var count: Int? = nil

    mutating func run() throws {
        let repeatCount = count ?? 2
        for _ in 0..<repeatCount {
            print(phrase)
        }
    }
}
```

When a user executes your command, 
the `ArgumentParser` library parses the command-line arguments,
instantiates your command type,
and then either calls your `run()` method or exits with a useful message.

![The output of the Repeat command, declared above.](repeat.png)

#### Additional Resources

- [`ArgumentParser` on GitHub](https://github.com/apple/swift-argument-parser/)
- [`ArgumentParser` on the Swift Forums](https://forums.swift.org/c/related-projects/argumentparser/60)

## Topics

### Essentials

- <doc:GettingStarted>
- ``ParsableCommand``
- ``AsyncParsableCommand``
- <doc:CommandsAndSubcommands>
- <doc:CustomizingCommandHelp>

### Arguments, Options, and Flags

- <doc:DeclaringArguments>
- ``Argument``
- ``Option``
- ``Flag``
- ``OptionGroup``
- ``ParsableArguments``

### Property Customization

- <doc:CustomizingHelp>
- ``ArgumentHelp``
- ``ArgumentVisibility``
- ``NameSpecification``

### Custom Types

- ``ExpressibleByArgument``
- ``EnumerableFlag``

### Validation and Errors

- <doc:Validation>
- ``ValidationError``
- ``CleanExit``
- ``ExitCode``

### Shell Completion Scripts

- <doc:InstallingCompletionScripts>
- <doc:CustomizingCompletions>
- ``CompletionKind``

### Advanced Topics

- <doc:ManualParsing>
- <doc:ExperimentalFeatures>
