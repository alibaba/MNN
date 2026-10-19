# Defining Commands and Subcommands

Break complex command-line tools into a tree of subcommands.

## Overview

When command-line programs grow larger, it can be useful to divide them into a group of smaller programs, providing an interface through subcommands. Utilities such as `git` and the Swift package manager are able to provide varied interfaces for each of their sub-functions by implementing subcommands such as `git branch` or `swift package init`.

Generally, these subcommands each have their own configuration options, as well as options that are shared across several or all aspects of the larger program.

You can build a program with commands and subcommands by defining multiple command types and specifying each command's subcommands in its configuration. For example, here's the interface of a `math` utility that performs operations on a series of values given on the command line.

```
% math add 10 15 7
32
% math multiply 10 15 7
1050
% math stats average 3 4 13 15 15
10.0
% math stats average --kind median 3 4 13 15 15
13.0
% math stats
OVERVIEW: Calculate descriptive statistics.

USAGE: math stats <subcommand>

OPTIONS:
  -h, --help              Show help information.

SUBCOMMANDS:
  average, avg            Print the average of the values.
  stdev                   Print the standard deviation of the values.
  quantiles               Print the quantiles of the values (TBD).

  See 'math help stats <subcommand>' for detailed help.
```

Start by defining the root `Math` command. You can provide a static ``ParsableCommand/configuration-35km1`` property for a command that specifies its subcommands and a default subcommand, if any.

```swift
struct Math: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "A utility for performing maths.",
        subcommands: [Add.self, Multiply.self, Statistics.self],
        defaultSubcommand: Add.self)
}
```

`Math` lists its three subcommands by their types; we'll see the definitions of `Add`, `Multiply`, and `Statistics` below. `Add` is also given as a default subcommand — this means that it is selected if a user leaves out a subcommand name:

```
% math 10 15 7
32
```

Next, define a ``ParsableArguments`` type with properties that will be shared across multiple subcommands. Types that conform to `ParsableArguments` can be parsed from command-line arguments, but don't provide any execution through a `run()` method.

In this case, the `Options` type accepts a `--hexadecimal-output` flag and expects a list of integers.

```swift
struct Options: ParsableArguments {
    @Flag(name: [.long, .customShort("x")], help: "Use hexadecimal notation for the result.")
    var hexadecimalOutput = false

    @Argument(help: "A group of integers to operate on.")
    var values: [Int]
}
```

It's time to define our first two subcommands: `Add` and `Multiply`. Both of these subcommands include the arguments defined in the `Options` type by denoting that property with the `@OptionGroup` property wrapper (see ``OptionGroup``). `@OptionGroup` doesn't define any new arguments for a command; instead, it splats in the arguments defined by another `ParsableArguments` type.

```swift
extension Math {
    struct Add: ParsableCommand {
        static let configuration
            = CommandConfiguration(abstract: "Print the sum of the values.")

        @OptionGroup var options: Math.Options

        mutating func run() {
            let result = options.values.reduce(0, +)
            print(format(result: result, usingHex: options.hexadecimalOutput))
        }
    }

    struct Multiply: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the product of the values.",
            aliases: ["mul"])

        @OptionGroup var options: Math.Options

        mutating func run() {
            let result = options.values.reduce(1, *)
            print(format(result: result, usingHex: options.hexadecimalOutput))
        }
    }
}
```

One thing to note is the aliases parameter for `CommandConfiguration`. This is useful for subcommands
to define alternative names that can be used to invoke them. In this case we've defined a shorthand
for multiply named mul, so you could invoke the `Multiply` command for our program by either of the below:

```
% math multiply 10 15 7
1050
% math mul 10 15 7
1050
```

Next, we'll define `Statistics`, the third subcommand of `Math`. The `Statistics` command specifies a custom command name (`stats`) in its configuration, overriding the default derived from the type name (`statistics`). It also declares two additional subcommands, meaning that it acts as a forked branch in the command tree, and not a leaf.

```swift
extension Math {
    struct Statistics: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "stats",
            abstract: "Calculate descriptive statistics.",
            subcommands: [Average.self, StandardDeviation.self])
    }
}
```

Let's finish our subcommands with the `Average` and `StandardDeviation` types. Each of them has slightly different arguments, so they don't use the `Options` type defined above. Each subcommand is ultimately independent and can specify a combination of shared and unique arguments.

```swift
extension Math.Statistics {
    struct Average: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the average of the values.",
            aliases: ["avg"])

        enum Kind: String, ExpressibleByArgument {
            case mean, median, mode
        }

        @Option(help: "The kind of average to provide.")
        var kind: Kind = .mean

        @Argument(help: "A group of floating-point values to operate on.")
        var values: [Double] = []

        func calculateMean() -> Double { ... }
        func calculateMedian() -> Double { ... }
        func calculateMode() -> [Double] { ... }

        mutating func run() {
            switch kind {
            case .mean:
                print(calculateMean())
            case .median:
                print(calculateMedian())
            case .mode:
                let result = calculateMode()
                    .map(String.init(describing:))
                    .joined(separator: " ")
                print(result)
            }
        }
    }

    struct StandardDeviation: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "stdev",
            abstract: "Print the standard deviation of the values.")

        @Argument(help: "A group of floating-point values to operate on.")
        var values: [Double] = []

        mutating func run() {
            if values.isEmpty {
                print(0.0)
            } else {
                let sum = values.reduce(0, +)
                let mean = sum / Double(values.count)
                let squaredErrors = values
                    .map { $0 - mean }
                    .map { $0 * $0 }
                let variance = squaredErrors.reduce(0, +) / Double(values.count)
                let result = variance.squareRoot()
                print(result)
            }
        }
    }
}
```

Last but not least, we add the `@main` attribute to the root of our command tree, to tell the compiler to use that as the program's entry point. Upon execution, this parses the command-line arguments, determines whether a subcommand was selected, and then instantiates and calls the `run()` method on that particular subcommand.

> Note: The Swift compiler uses either the type marked with `@main` or a `main.swift` file as the entry point for an executable program. You can use either one, but not both — rename your `main.swift` file to the name of your command when you add `@main`. In this case, rename it to `Math.swift`.

```swift
@main
struct Math: ParsableCommand {
    // ...
}
```

That's it for this doubly-nested `math` command! This example is also provided as a part of the `swift-argument-parser` repository, so you can see it all together and experiment with it [here](https://github.com/apple/swift-argument-parser/blob/main/Examples/math/Math.swift).
