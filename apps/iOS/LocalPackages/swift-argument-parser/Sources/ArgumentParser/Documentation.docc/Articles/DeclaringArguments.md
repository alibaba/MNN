# Declaring Arguments, Options, and Flags

Use the `@Argument`, `@Option` and `@Flag` property wrappers to declare the command-line interface for your command.

## Overview

When creating commands, you can define three primary kinds of command-line inputs:

- *Arguments* are values given by a user and are read in order from first to last (see ``Argument``). For example, this command is called with three file names as arguments:

  ```
  % example file1.swift file2.swift file3.swift
  ```

- *Options* are named key-value pairs. Keys start with one or two dashes (`-` or `--`), and a user can separate the key and value with an equal sign (`=`) or a space (see ``Option``). This command is called with two options:

  ```
  % example --count=5 --index 2
  ```

- *Flags* are like options, but without a paired value. Instead, their presence indicates a particular value (see ``Flag``). This command is called with two flags:

  ```
  % example --verbose --strip-whitespace
  ```

The three preceding examples could be calls of this `Example` command:

```swift
struct Example: ParsableCommand {
    @Argument var files: [String] = []
    @Option var count: Int?
    @Option var index = 0
    @Flag var verbose = false
    @Flag var stripWhitespace = false
}
```

This example shows how `ArgumentParser` provides defaults that speed up your initial development process:

- Option and flag names are derived from the names of your command's properties.
- What kinds of inputs are valid, and whether arguments are required, is based on your properties' types and default values.

In this example, all of the properties have default values (optional properties default to `nil`).

Users must provide values for all properties with no implicit or specified default. For example, this command would require one integer argument and a string with the key `--user-name`.

```swift
struct Example: ParsableCommand {
    @Option var userName: String
    @Argument var value: Int
}
```

When called without both values, the command exits with an error:

```
% example 5
Error: Missing '--user-name <user-name>'
Usage: example --user-name <user-name> <value>
  See 'example --help' for more information.
% example --user-name kjohnson
Error: Missing '<value>'
Usage: example --user-name <user-name> <value>
  See 'example --help' for more information.
```

When providing a default value for an array property, any user-supplied values replace the entire default.

```swift
struct Lucky: ParsableCommand {
    @Argument var numbers = [7, 14, 21]

    mutating func run() throws {
        print("""
        Your lucky numbers are:
        \(numbers.map(String.init).joined(separator: " "))
        """)
    }
}
```

```
% lucky 
Your lucky numbers are:
7 14 21
% lucky 1 2 3
Your lucky numbers are:
1 2 3
```

## Customizing option and flag names

By default, options and flags derive the name that you use on the command line from the name of the property, such as `--count` and `--index`. Camel-case names are converted to lowercase with hyphen-separated words, like `--strip-whitespace`.

You can override this default by specifying one or more name specifications in the `@Option` or `@Flag` initializers. This command demonstrates the four name specifications:

```swift
struct Example: ParsableCommand {
    @Flag(name: .long)  // Same as the default
    var stripWhitespace = false

    @Flag(name: .short)
    var verbose = false

    @Option(name: .customLong("count"))
    var iterationCount: Int

    @Option(name: [.customShort("I"), .long])
    var inputFile: String
}
```

* Specifying `.long` or `.short` uses the property's name as the source of the command-line name. Long names use the whole name, prefixed by two dashes, while short names are a single character prefixed by a single dash. In this example, the `stripWhitespace` and `verbose` flags are specified in this way:

  ```
  % example --strip-whitespace -v
  ```

* Specifying `.customLong(_:)` or `.customShort(_:)` uses the given string or character as the long or short name for the property.

  ```
  % example --count 10 -I file1.swift
  ```

* Use array literal syntax to specify multiple names. The `inputFile` property can alternatively be given with the default long name:

  ```
  % example --input-file file1.swift
  ```

**Note:** You can also pass `withSingleDash: true` to `.customLong` to create a single-dash flag or option, such as `-verbose`. Use this name specification only when necessary, such as when migrating a legacy command-line interface. Using long names with a single-dash prefix can lead to ambiguity with combined short names: it may not be obvious whether `-file` is a single option or the combination of the four short options `-f`, `-i`, `-l`, and `-e`.


## Parsing custom types

Arguments and options can be parsed from any type that conforms to the ``ExpressibleByArgument`` protocol. Standard library integer and floating-point types, strings, and Booleans all conform to `ExpressibleByArgument`.

You can make your own custom types conform to `ExpressibleByArgument` by implementing ``ExpressibleByArgument/init(argument:)``:

```swift
struct Path: ExpressibleByArgument {
    var pathString: String

    init?(argument: String) {
        self.pathString = argument
    }
}

struct Example: ParsableCommand {
    @Argument var inputFile: Path
}
```

The library provides a default implementation for `RawRepresentable` types, like string-backed enumerations, so you only need to declare conformance.

```swift
enum ReleaseMode: String, ExpressibleByArgument {
    case debug, release
}

struct Example: ParsableCommand {
    @Option var mode: ReleaseMode

    mutating func run() throws {
        print(mode)
    }
}
```

The user can provide the raw values on the command line, which are then converted to your custom type. Only valid values are allowed:

```
% example --mode release
release
% example --mode future
Error: The value 'future' is invalid for '--mode <mode>'
```

To use a non-`ExpressibleByArgument` type for an argument or option, you can instead provide a throwing `transform` function that converts the parsed string to your desired type. This is a good idea for custom types that are more complex than a `RawRepresentable` type, or for types you don't define yourself.

```swift
enum Format {
    case text
    case other(String)

    init(_ string: String) throws {
        if string == "text" {
            self = .text
        } else {
            self = .other(string)
        }
    }
}

struct Example: ParsableCommand {
    @Argument(transform: Format.init)
    var format: Format
}
```

Throw an error from the `transform` function to indicate that the user provided an invalid value for that type. See <doc:Validation> for more about customizing `transform` function errors.

## Using flag inversions, enumerations, and counts

Flags are most frequently used for `Bool` properties. You can generate a `true`/`false` pair of flags by specifying a flag inversion:

```swift
struct Example: ParsableCommand {
    @Flag(inversion: .prefixedNo)
    var index = true

    @Flag(inversion: .prefixedEnableDisable)
    var requiredElement: Bool

    mutating func run() throws {
        print(index, requiredElement)
    }
}
```

When declaring a flag with an inversion, set the default by specifying `true` or `false` as the property's initial value. If you want to require that the user specify one of the two inversions, leave off the default value.

In the `Example` command defined above, a flag is required for the `requiredElement` property. The specified prefixes are prepended to the long names for the flags:

```
% example --enable-required-element
true true
% example --no-index --disable-required-element
false false
% example --index
Error: Missing one of: '--enable-required-element', '--disable-required-element'
```

To create a flag with custom names for a Boolean value, to provide an exclusive choice between more than two names, or for collecting multiple values from a set of defined choices, define an enumeration that conforms to the `EnumerableFlag` protocol.

```swift
enum CacheMethod: String, EnumerableFlag {
    case inMemoryCache
    case persistentCache
}

enum Color: String, EnumerableFlag {
    case pink, purple, silver
}

struct Example: ParsableCommand {
    @Flag var cacheMethod: CacheMethod
    @Flag var colors: [Color] = []

    mutating func run() throws {
        print(cacheMethod)
        print(colors)
    }
}
```

The flag names in this case are drawn from the raw values — for information about customizing the names and help text, see the  ``EnumerableFlag`` documentation.

```
% example --in-memory-cache --pink --silver
.inMemoryCache
[.pink, .silver]
% example
Error: Missing one of: '--in-memory-cache', '--persistent-cache'
```

Finally, when a flag is of type `Int`, the value is parsed as a count of the number of times that the flag is specified.

```swift
struct Example: ParsableCommand {
    @Flag(name: .shortAndLong)
    var verbose: Int

    mutating func run() throws {
        print("Verbosity level: \(verbose)")
    }
}
```

In this example, `verbose` defaults to zero, and counts the number of times that `-v` or `--verbose` is given.

```
% example --verbose
Verbosity level: 1
% example -vvvv
Verbosity level: 4
```


## Specifying default values

You can specify default values for almost all supported argument, option, and flag types using normal property initialization syntax:

```swift
enum CustomFlag: String, EnumerableFlag {
    case foo, bar, baz
}

struct Example: ParsableCommand {
    @Flag
    var booleanFlag = false

    @Flag
    var arrayFlag: [CustomFlag] = [.foo, .baz]

    @Option
    var singleOption = 0

    @Option
    var arrayOption = ["bar", "qux"]

    @Argument
    var singleArgument = "quux"

    @Argument
    var arrayArgument = ["quux", "quuz"]
}
```

This includes all of the variants of the argument types above (including `@Option(transform: ...)`, etc.), with a few notable exceptions:
- `Optional`-typed values (which default to `nil` and for which a default would not make sense, as the value could never be `nil`)
- `Int` flags (which are used for counting the number of times a flag is specified and therefore default to `0`)

If a default is not specified, the user must provide a value for that argument/option/flag or will receive an error that the value is missing.

You must also always specify a default of `false` for a non-optional `Bool` flag, as in the example above. This makes the behavior consistent with both normal Swift properties (which either must be explicitly initialized or optional to initialize a `struct`/`class` containing them) and the other property types.


## Specifying a parsing strategy

When parsing a list of command-line inputs, `ArgumentParser` distinguishes between dash-prefixed keys and un-prefixed values. When looking for the value for a key, only an un-prefixed value will be selected by default.

For example, this command defines a `--verbose` flag, a `--name` option, and an optional `file` argument:

```swift
struct Example: ParsableCommand {
    @Flag var verbose = false
    @Option var name: String
    @Argument var file: String?

    mutating func run() throws {
        print("Verbose: \(verbose), name: \(name), file: \(file ?? "none")")
    }
}
```

When calling this command, the value for `--name` must be given immediately after the key. If the `--verbose` flag is placed in between, parsing fails with an error:

```
% example --verbose --name Tomás
Verbose: true, name: Tomás, file: none
% example --name --verbose Tomás
Error: Missing value for '--name <name>'
Usage: example [--verbose] --name <name> [<file>]
  See 'example --help' for more information.
```

Parsing options as arrays is similar — only adjacent key-value pairs are recognized by default.

### Alternative single-value parsing strategies

You can change this behavior by providing a different parsing strategy in the `@Option` initializer. **Be careful when selecting any of the alternative parsing strategies** — they may lead your command-line tool to have unexpected behavior for users!

The `.unconditional` parsing strategy uses the immediate next input for the value of the option, even if it starts with a dash. If `name` were instead defined as `@Option(parsing: .unconditional) var name: String`, the second attempt would result in `"--verbose"` being read as the value of `name`:

```
% example --name --verbose Tomás
Verbose: false, name: --verbose, file: Tomás
```

The `.scanningForValue` strategy, on the other hand, looks ahead in the list of command-line inputs and uses the first un-prefixed value as the input, even if that requires skipping over other flags or options.  If `name` were defined as `@Option(parsing: .scanningForValue) var name: String`, the parser would look ahead to find `Tomás`, then pick up parsing where it left off to get the `--verbose` flag:

```
% example --name --verbose Tomás
Verbose: true, name: Tomás, file: none
```

### Alternative array parsing strategies

The default strategy for parsing options as arrays is to read each value from a key-value pair. For example, this command expects zero or more input file names:

```swift
struct Example: ParsableCommand {
    @Option var file: [String] = []
    @Flag var verbose = false

    mutating func run() throws {
        print("Verbose: \(verbose), files: \(file)")
    }
}
```

As with single values, each time the user provides the `--file` key, they must also provide a value:

```
% example --verbose --file file1.swift --file file2.swift
Verbose: true, files: ["file1.swift", "file2.swift"]
% example --file --verbose file1.swift --file file2.swift
Error: Missing value for '--file <file>'
Usage: example [--file <file> ...] [--verbose]
  See 'example --help' for more information.
```

The `.unconditionalSingleValue` parsing strategy uses whatever input follows the key as its value, even if that input is dash-prefixed. If `file` were defined as `@Option(parsing: .unconditionalSingleValue) var file: [String]`, then the resulting array could include strings that look like options:

```
% example --file file1.swift --file --verbose
Verbose: false, files: ["file1.swift", "--verbose"]
```

The `.upToNextOption` parsing strategy uses the inputs that follow the option key until reaching a dash-prefixed input. If `file` were defined as `@Option(parsing: .upToNextOption) var file: [String]`, then the user could specify multiple files without repeating `--file`:

```
% example --file file1.swift file2.swift
Verbose: false, files: ["file1.swift", "file2.swift"]
% example --file file1.swift file2.swift --verbose
Verbose: true, files: ["file1.swift", "file2.swift"]
```

Finally, the `.remaining` parsing strategy uses all the inputs that follow the option key, regardless of their prefix. If `file` were defined as `@Option(parsing: .remaining) var file: [String]`, then the user would need to specify `--verbose` before the `--file` key for it to be recognized as a flag:

```
% example --verbose --file file1.swift file2.swift
Verbose: true, files: ["file1.swift", "file2.swift"]
% example --file file1.swift file2.swift --verbose
Verbose: false, files: ["file1.swift", "file2.swift", "--verbose"]
```

### Alternative positional argument parsing strategies

The default strategy for parsing arrays of positional arguments is to ignore  all dash-prefixed command-line inputs. For example, this command accepts a `--verbose` flag and a list of file names as positional arguments:

```swift
struct Example: ParsableCommand {
    @Flag var verbose = false
    @Argument var files: [String] = []

    mutating func run() throws {
        print("Verbose: \(verbose), files: \(files)")
    }
}
```

The `files` argument array uses the default `.remaining` parsing strategy, so it only picks up values that don't have a prefix:

```
% example --verbose file1.swift file2.swift
Verbose: true, files: ["file1.swift", "file2.swift"]
% example --verbose file1.swift file2.swift --other
Error: Unexpected argument '--other'
Usage: example [--verbose] [<files> ...]
  See 'example --help' for more information.
```

Any input after the `--` terminator is automatically treated as positional input, so users can provide dash-prefixed values that way even with the default configuration:

```
% example --verbose -- file1.swift file2.swift --other
Verbose: true, files: ["file1.swift", "file2.swift", "--other"]
```

The `.unconditionalRemaining` parsing strategy uses whatever input is left after parsing known options and flags, even if that input is dash-prefixed, including the terminator itself. If `files` were defined as `@Argument(parsing: .unconditionalRemaining) var files: [String]`, then the resulting array would also include strings that look like options:

```
% example --verbose file1.swift file2.swift --other
Verbose: true, files: ["file1.swift", "file2.swift", "--other"]
% example -- --verbose file1.swift file2.swift --other
Verbose: false, files: ["--", "--verbose", "file1.swift", "file2.swift", "--other"]
```

### Ignoring unknown arguments

Different versions of a CLI tool may have full or partial sets of supported flags and options.

By default, `ArgumentParser` throws an error if unknown arguments are passed as command input.
When appropriate, you can process supported arguments and ignore unknown ones by collecting unknowns in special `@Argument` with the `.allUnrecognized` strategy.

```swift
struct Example: ParsableCommand {
    @Flag var verbose = false
    
    @Argument(parsing: .allUnrecognized)
    var unknowns: [String] = []

    func run() throws {
        print("Verbose: \(verbose)")
    }
}
```

This way any unknown parameters are silently captured in the `unknowns` array.

```
% example --flag --verbose --option abc file1.swift
Verbose: true
```
