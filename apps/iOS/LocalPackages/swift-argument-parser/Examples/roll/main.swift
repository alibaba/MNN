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

struct RollOptions: ParsableArguments {
    @Option(help: ArgumentHelp("Rolls the dice <n> times.", valueName: "n"))
    var times = 1

    @Option(help: ArgumentHelp(
        "Rolls an <m>-sided dice.",
        discussion: "Use this option to override the default value of a six-sided die.",
        valueName: "m"))
    var sides = 6

    @Option(help: "A seed to use for repeatable random generation.")
    var seed: Int? = nil

    @Flag(name: .shortAndLong, help: "Show all roll results.")
    var verbose = false
}

// If you prefer writing in a "script" style, you can call `parseOrExit()` to
// parse a single `ParsableArguments` type from command-line arguments.
let options = RollOptions.parseOrExit()

let seed = options.seed ?? .random(in: .min ... .max)
var rng = SplitMix64(seed: UInt64(truncatingIfNeeded: seed))

let rolls = (1...options.times).map { _ in
    Int.random(in: 1...options.sides, using: &rng)
}

if options.verbose {
    for (number, roll) in zip(1..., rolls) {
        print("Roll \(number): \(roll)")
    }
}

print(rolls.reduce(0, +))
