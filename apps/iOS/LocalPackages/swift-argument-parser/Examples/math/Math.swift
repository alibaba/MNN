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

import ArgumentParser

@main
struct Math: ParsableCommand {
    // Customize your command's help and subcommands by implementing the
    // `configuration` property.
    static let configuration = CommandConfiguration(
        // Optional abstracts and discussions are used for help output.
        abstract: "A utility for performing maths.",

        // Commands can define a version for automatic '--version' support.
        version: "1.0.0",

        // Pass an array to `subcommands` to set up a nested tree of subcommands.
        // With language support for type-level introspection, this could be
        // provided by automatically finding nested `ParsableCommand` types.
        subcommands: [Add.self, Multiply.self, Statistics.self],

        // A default subcommand, when provided, is automatically selected if a
        // subcommand is not given on the command line.
        defaultSubcommand: Add.self)

}

struct Options: ParsableArguments {
    @Flag(name: [.customLong("hex-output"), .customShort("x")],
          help: "Use hexadecimal notation for the result.")
    var hexadecimalOutput = false

    @Argument(
        help: "A group of integers to operate on.")
    var values: [Int] = []
}

extension Math {
    static func format(_ result: Int, usingHex: Bool) -> String {
        usingHex ? String(result, radix: 16)
            : String(result)
    }

    struct Add: ParsableCommand {
        static let configuration =
            CommandConfiguration(abstract: "Print the sum of the values.")

        // The `@OptionGroup` attribute includes the flags, options, and
        // arguments defined by another `ParsableArguments` type.
        @OptionGroup var options: Options

        mutating func run() {
            let result = options.values.reduce(0, +)
            print(format(result, usingHex: options.hexadecimalOutput))
        }
    }

    struct Multiply: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the product of the values.",
            aliases: ["mul"])

        @OptionGroup var options: Options

        mutating func run() {
            let result = options.values.reduce(1, *)
            print(format(result, usingHex: options.hexadecimalOutput))
        }
    }
}

// In practice, these nested types could be broken out into different files.
extension Math {
    struct Statistics: ParsableCommand {
        static let configuration = CommandConfiguration(
            // Command names are automatically generated from the type name
            // by default; you can specify an override here.
            commandName: "stats",
            abstract: "Calculate descriptive statistics.",
            subcommands: [Average.self, StandardDeviation.self, Quantiles.self])
    }
}

extension Math.Statistics {
    struct Average: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the average of the values.",
            version: "1.5.0-alpha",
            aliases: ["avg"])

        enum Kind: String, ExpressibleByArgument, CaseIterable {
            case mean, median, mode
        }

        @Option(help: "The kind of average to provide.")
        var kind: Kind = .mean

        @Argument(help: "A group of floating-point values to operate on.")
        var values: [Double] = []

        func validate() throws {
            if (kind == .median || kind == .mode) && values.isEmpty {
                throw ValidationError("Please provide at least one value to calculate the \(kind).")
            }
        }

        func calculateMean() -> Double {
            guard !values.isEmpty else {
                return 0
            }

            let sum = values.reduce(0, +)
            return sum / Double(values.count)
        }

        func calculateMedian() -> Double {
            guard !values.isEmpty else {
                return 0
            }

            let sorted = values.sorted()
            let mid = sorted.count / 2
            if sorted.count.isMultiple(of: 2) {
                return (sorted[mid - 1] + sorted[mid]) / 2
            } else {
                return sorted[mid]
            }
        }

        func calculateMode() -> [Double] {
            guard !values.isEmpty else {
                return []
            }

            let grouped = Dictionary(grouping: values, by: { $0 })
            let highestFrequency = grouped.lazy.map { $0.value.count }.max()!
            return grouped.filter { _, v in v.count == highestFrequency }
                .map { k, _ in k }
        }

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

    struct Quantiles: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the quantiles of the values (TBD).")

        @Argument(completion: .list(["alphabet", "alligator", "branch", "braggart"]))
        var oneOfFour: String?

        @Argument(completion: .custom { _ in ["alabaster", "breakfast", "crunch", "crash"] })
        var customArg: String?

        @Argument(help: "A group of floating-point values to operate on.")
        var values: [Double] = []

        // These args and the validation method are for testing exit codes:
        @Flag(help: .hidden)
        var testSuccessExitCode = false
        @Flag(help: .hidden)
        var testFailureExitCode = false
        @Flag(help: .hidden)
        var testValidationExitCode = false
        @Option(help: .hidden)
        var testCustomExitCode: Int32?

        // These args are for testing custom completion scripts:
        @Option(completion: .file(extensions: ["txt", "md"]))
        var file: String?
        @Option(completion: .directory)
        var directory: String?
        
        @Option(completion: .shellCommand("head -100 /usr/share/dict/words | tail -50"))
        var shell: String?
        
        @Option(completion: .custom(customCompletion))
        var custom: String?

        func validate() throws {
            if testSuccessExitCode {
                throw ExitCode.success
            }

            if testFailureExitCode {
                throw ExitCode.failure
            }

            if testValidationExitCode {
                throw ExitCode.validationFailure
            }

            if let exitCode = testCustomExitCode {
                throw ExitCode(exitCode)
            }
        }
    }
}

func customCompletion(_ s: [String]) -> [String] {
  return (s.last ?? "").starts(with: "a")
    ? ["aardvark", "aaaaalbert"]
    : ["hello", "helicopter", "heliotrope"]
}
