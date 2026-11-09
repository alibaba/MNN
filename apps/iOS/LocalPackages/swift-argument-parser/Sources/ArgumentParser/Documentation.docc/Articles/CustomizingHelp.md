# Customizing Help

Support your users (and yourself) by providing rich help for arguments, options, and flags.

## Overview

You can provide help text when declaring any `@Argument`, `@Option`, or `@Flag` by passing a string literal as the `help` parameter:

```swift
struct Example: ParsableCommand {
    @Flag(help: "Display extra information while processing.")
    var verbose = false

    @Option(help: "The number of extra lines to show.")
    var extraLines = 0

    @Argument(help: "The input file.")
    var inputFile: String?
}
```

Users see these strings in the automatically-generated help screen, which is triggered by the `-h` or `--help` flags, by default:

```
% example --help
USAGE: example [--verbose] [--extra-lines <extra-lines>] <input-file>

ARGUMENTS:
  <input-file>            The input file.

OPTIONS:
  --verbose               Display extra information while processing.
  --extra-lines <extra-lines>
                          The number of extra lines to show. (default: 0)
  -h, --help              Show help information.
```

## Customizing Help for Arguments

For more control over the help text, pass an ``ArgumentHelp`` instance instead of a string literal. The `ArgumentHelp` type can include an abstract (which is what the string literal becomes), a discussion, a value name to use in the usage string, and a visibility level for that argument.

Here's the same command with some extra customization:

```swift
struct Example: ParsableCommand {
    @Flag(help: "Display extra information while processing.")
    var verbose = false

    @Option(help: ArgumentHelp(
        "The number of extra lines to show.",
        valueName: "n"))
    var extraLines = 0

    @Argument(help: ArgumentHelp(
        "The input file.",
        discussion: "If no input file is provided, the tool reads from stdin.",
        valueName: "file"))
    var inputFile: String?
}
```

...and the help screen:

```
USAGE: example [--verbose] [--extra-lines <n>] [<file>]

ARGUMENTS:
  <file>                  The input file.
        If no input file is provided, the tool reads from stdin.

OPTIONS:
  --verbose               Display extra information while processing.
  --extra-lines <n>       The number of extra lines to show. (default: 0)
  -h, --help              Show help information.
```

## Enumerating Possible Values

When an argument or option has a fixed set of possible values, listing these values in the help screen can simplify use of your tool. You can customize the displayed set of values for custom ``ExpressibleByArgument`` types by implementing ``ExpressibleByArgument/allValueStrings``. Despite the name, ``ExpressibleByArgument/allValueStrings`` does _not_ need to be an exhaustive list of possible values.

```swift
enum Fruit: String, ExpressibleByArgument {
    case apple
    case banana
    case coconut
    case dragonFruit = "dragon-fruit"

    static var allValueStrings: [String] {
        ["apple", "banana", "coconut", "dragon-fruit"]
    }
}

struct FruitStore: ParsableCommand {
    @Argument(help: "The fruit to purchase")
    var fruit: Fruit
  
    @Option(help: "The number of fruit to purchase")
    var quantity: Int = 1
}
```

The help screen includes the list of values in the description of the `<fruit>` argument:

```
USAGE: fruit-store <fruit> [--quantity <quantity>]

ARGUMENTS:
  <fruit>                 The fruit to purchase (values: apple, banana,
                          coconut, dragon-fruit)

OPTIONS:
  --quantity <quantity>   The number of fruit to purchase (default: 1)
  -h, --help              Show help information.
```

### Deriving Possible Values

ExpressibleByArgument types that conform to ``CaseIterable`` do not need to manually specify ``ExpressibleByArgument/allValueStrings``. Instead, a list of possible values is derived from the type's cases, as in this updated example:

```swift
enum Fruit: String, CaseIterable, ExpressibleByArgument {
    case apple
    case banana
    case coconut
    case dragonFruit = "dragon-fruit"
}

struct FruitStore: ParsableCommand {
    @Argument(help: "The fruit to purchase")
    var fruit: Fruit
  
    @Option(help: "The number of fruit to purchase")
    var quantity: Int = 1
}
```

The help screen still contains all the possible values.

```
USAGE: fruit-store <fruit> [--quantity <quantity>]

ARGUMENTS:
  <fruit>                 The fruit to purchase (values: apple, banana,
                          coconut, dragon-fruit)

OPTIONS:
  --quantity <quantity>   The number of fruit to purchase (default: 1)
  -h, --help              Show help information.
```

For an ``ExpressibleByArgument`` and ``CaseIterable`` type with many cases, you may still want to implement ``ExpressibleByArgument/allValueStrings`` to avoid an overly long list of values appearing in the help screen. For these types it is recommended to include the most common possible values.

## Controlling Argument Visibility

You can specify the visibility of any argument, option, or flag.

```swift
struct Example: ParsableCommand {
    @Flag(help: ArgumentHelp("Show extra info.", visibility: .hidden))
    var verbose: Bool = false

    @Flag(help: ArgumentHelp("Use the legacy format.", visibility: .private))
    var useLegacyFormat: Bool = false
}
```

The `--verbose` flag is only visible in the extended help screen. The `--use-legacy-format` stays hidden even in the extended help screen, due to its `.private` visibility. 

```
% example --help
USAGE: example

OPTIONS:
  -h, --help              Show help information.

% example --help-hidden
USAGE: example [--verbose]

OPTIONS:
  --verbose               Show extra info.
  -h, --help              Show help information.
```

Alternatively, you can group multiple arguments, options, and flags together as part of a ``ParsableArguments`` type, and set the visibility when including them as an `@OptionGroup` property.

```swift
struct ExperimentalFlags: ParsableArguments {
    @Flag(help: "Use the remote access token. (experimental)")
    var experimentalUseRemoteAccessToken: Bool = false

    @Flag(help: "Use advanced security. (experimental)")
    var experimentalAdvancedSecurity: Bool = false
}

struct Example: ParsableCommand {
    @OptionGroup(visibility: .hidden)
    var flags: ExperimentalFlags
}
```

The members of `ExperimentalFlags` are only shown in the extended help screen:

```
% example --help
USAGE: example

OPTIONS:
  -h, --help              Show help information.

% example --help-hidden
USAGE: example [--experimental-use-remote-access-token] [--experimental-advanced-security]

OPTIONS:
  --experimental-use-remote-access-token
                          Use the remote access token. (experimental)
  --experimental-advanced-security
                          Use advanced security. (experimental)
  -h, --help              Show help information.
```

## Grouping Arguments in the Help Screen

When you provide a title in an `@OptionGroup` declaration, that type's  properties are grouped together under your title in the help screen. For example, this command bundles similar arguments together under a  "Build Options" title:

```swift
struct BuildOptions: ParsableArguments {
    @Option(help: "A setting to pass to the compiler.")
    var compilerSetting: [String] = []

    @Option(help: "A setting to pass to the linker.")
    var linkerSetting: [String] = []
}

struct Example: ParsableCommand {
    @Argument(help: "The input file to process.")
    var inputFile: String

    @Flag(help: "Show extra output.")
    var verbose: Bool = false

    @Option(help: "The path to a configuration file.")
    var configFile: String?

    @OptionGroup(title: "Build Options")
    var buildOptions: BuildOptions
}
```

This grouping is reflected in the command's help screen:

```
% example --help
USAGE: example <input-file> [--verbose] [--config-file <config-file>] [--compiler-setting <compiler-setting> ...] [--linker-setting <linker-setting> ...]

ARGUMENTS:
  <input-file>            The input file to process.

BUILD OPTIONS:
  --compiler-setting <compiler-setting>
                          A setting to pass to the compiler.
  --linker-setting <linker-setting>
                          A setting to pass to the linker.

OPTIONS:
  --verbose               Show extra output.
  --config-file <config-file>
                          The path to a configuration file.
  -h, --help              Show help information.
```
